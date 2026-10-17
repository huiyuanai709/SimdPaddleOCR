#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// WebAssembly SIMD128 (PackedSimd) depthwise channels-last convolution. Same
// structure as the other ISAs: interior pixel pairs accumulate into a
// contiguous strip ([px][32]), border pixels check each tap. A 32-channel
// block runs as two 16-channel half passes (2 pixels x 16 channels = eight
// accumulators + four weight vectors), which fits the 16 XMM registers browsers
// allocate v128 values to on x64; channel tails run 4 channels at a time. Every
// accumulation is a separate multiply then add in ascending tap order, the
// same per-lane arithmetic as the Vector<float> kernel. The fused epilogue
// reuses ActivatePackedSimd from Nhwc.Conv.PackedSimd.cs.
internal static unsafe partial class Nhwc
{
    /// <summary>PackedSimd driver of <see cref="Depthwise"/>; same loop nest as the other ISA drivers.</summary>
    internal static void DepthwisePackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int channels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
        int taps = kernelH * kernelW;
        if ((channels & 7) != 0 || channels <= 0 || taps <= 0 || strideH <= 0 || strideW <= 0)
            throw new ArgumentException("NHWC depthwise convolution shape is not supported.");
        long outVolume = (long)batch * outputHeight * outputWidth * channels;
        if (input.Length < (long)batch * height * width * channels || output.Length < outVolume ||
            packedWeights.Length < (long)taps * channels || (!bias.IsEmpty && bias.Length < channels) ||
            (!residual.IsEmpty && residual.Length < outVolume))
            throw new ArgumentException("NHWC depthwise convolution buffer too small.");
        if (outVolume == 0) return;
        int rowsTotal = batch * outputHeight;
        int workers = threads > 1 && outVolume * taps >= 1_000_000 ? Math.Min(threads, rowsTotal) : 1;
        // Output x range whose every tap lies inside the input width.
        int xLo = Math.Min(outputWidth, (padLeft + strideW - 1) / strideW);
        int interiorEnd = width - kernelW + padLeft;
        int xHi = interiorEnd < 0 ? xLo : Math.Max(xLo, Math.Min(outputWidth, interiorEnd / strideW + 1));
        fixed (float* inPtr = input, wPtr = packedWeights, bPtr = bias, outPtr = output, rPtr = residual)
        {
            nint inA = (nint)inPtr, wA = (nint)wPtr, bA = (nint)bPtr, outA = (nint)outPtr, rA = (nint)rPtr;
            bool hasBias = !bias.IsEmpty, hasResidual = !residual.IsEmpty;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers);
                int rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                float* inBase = (float*)inA, w = (float*)wA, outBase = (float*)outA;
                float* biasBase = hasBias ? (float*)bA : null, resBase = hasResidual ? (float*)rA : null;
                Vector128<float> valpha = PackedSimd.Splat(alpha), vbeta = PackedSimd.Splat(beta), one = Vector128.Create(1f);
                // Contiguous accumulator strip (32 channels x output width): the
                // strided output row would map onto a handful of L1 sets.
                float[] stripArray = System.Buffers.ArrayPool<float>.Shared.Rent(outputWidth * 32);
                try
                {
                    fixed (float* strip = stripArray)
                    {
                    for (int row = rowBegin; row < rowEnd; row++)
                    {
                        int b = row / outputHeight, y = row - b * outputHeight;
                        float* inBatch = inBase + (long)b * height * width * channels;
                        int iy0 = y * strideH - padTop;
                        int kyBegin = Math.Max(0, -iy0), kyEnd = Math.Min(kernelH, height - iy0);
                        long outRow = ((long)b * outputHeight + y) * outputWidth;
                        float* outRowPtr = outBase + outRow * channels;
                        float* resRowPtr = resBase == null ? null : resBase + outRow * channels;
                        int x = 0;
                        for (; x < xLo; x++)
                            BorderPixelPackedSimd(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                                iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, valpha, vbeta, one);
                        int xPairEnd = x + ((xHi - x) & ~1);
                        if (xPairEnd > x)
                        {
                            int pixels = xPairEnd - x;
                            long rowStride = (long)width * channels, pixelStride = (long)strideW * channels;
                            bool strips = kernelH >= DepthwiseStripMinKernelH;
                            for (int c = 0; c < channels; c += 32)
                            {
                                int block = Math.Min(32, channels - c);
                                InitStrip(strip, biasBase == null ? null : biasBase + c, block, pixels);
                                float* inRow0 = inBatch + ((long)(iy0 + kyBegin) * width - padLeft) * channels + (long)x * pixelStride + c;
                                float* wRow0 = w + (long)c * taps + (long)kyBegin * kernelW * block;
                                if (!strips)
                                {
                                    if (block == 32)
                                    {
                                        AccumulateAllTaps16PackedSimd(inRow0, pixelStride, rowStride, wRow0, channels, kyEnd - kyBegin, kernelW, pixels, strip);
                                        AccumulateAllTaps16PackedSimd(inRow0 + 16, pixelStride, rowStride, wRow0 + 16, channels, kyEnd - kyBegin, kernelW, pixels, strip + 16);
                                    }
                                    else
                                        for (int c4 = 0; c4 < block; c4 += 4)
                                            AccumulateAllTaps4PackedSimd(inRow0 + c4, pixelStride, rowStride, wRow0 + c4, channels, block, kyEnd - kyBegin, kernelW, pixels, strip + c4, block);
                                }
                                else
                                {
                                    for (int ky = kyBegin; ky < kyEnd; ky++)
                                    {
                                        float* inRow = inRow0 + (long)(ky - kyBegin) * rowStride;
                                        float* wRow = wRow0 + (long)(ky - kyBegin) * kernelW * block;
                                        if (block == 32)
                                        {
                                            AccumulateAllTaps16PackedSimd(inRow, pixelStride, 0, wRow, channels, 1, kernelW, pixels, strip);
                                            AccumulateAllTaps16PackedSimd(inRow + 16, pixelStride, 0, wRow + 16, channels, 1, kernelW, pixels, strip + 16);
                                        }
                                        else
                                            for (int c4 = 0; c4 < block; c4 += 4)
                                                AccumulateAllTaps4PackedSimd(inRow + c4, pixelStride, 0, wRow + c4, channels, block, 1, kernelW, pixels, strip + c4, block);
                                    }
                                }
                                FinishStripPackedSimd(strip, block, pixels, outRowPtr + (long)x * channels + c, channels,
                                    resRowPtr == null ? null : resRowPtr + (long)x * channels + c, activation, valpha, vbeta, one);
                            }
                            x = xPairEnd;
                        }
                        for (; x < outputWidth; x++)
                            BorderPixelPackedSimd(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                                iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, valpha, vbeta, one);
                    }
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(stripArray);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // ------------------------------------------------ strip accumulate helpers
    // 2 pixels x 16 channels of a 32-wide packed block ([tap][32] weights,
    // [px][32] strip). kyCount = 1 with rowStride = 0 accumulates one kernel row
    // (the tall-kernel strip pass).

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps16PackedSimd(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels,
        int kyCount, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector128<float> a0 = PackedSimd.LoadVector128(acc0), a1 = PackedSimd.LoadVector128(acc0 + 4);
            Vector128<float> a2 = PackedSimd.LoadVector128(acc0 + 8), a3 = PackedSimd.LoadVector128(acc0 + 12);
            Vector128<float> b0 = PackedSimd.LoadVector128(acc1), b1 = PackedSimd.LoadVector128(acc1 + 4);
            Vector128<float> b2 = PackedSimd.LoadVector128(acc1 + 8), b3 = PackedSimd.LoadVector128(acc1 + 12);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
                {
                    Vector128<float> w0 = PackedSimd.LoadVector128(wk), w1 = PackedSimd.LoadVector128(wk + 4);
                    Vector128<float> w2 = PackedSimd.LoadVector128(wk + 8), w3 = PackedSimd.LoadVector128(wk + 12);
                    a0 = PackedSimd.Add(a0, PackedSimd.Multiply(PackedSimd.LoadVector128(r0), w0));
                    b0 = PackedSimd.Add(b0, PackedSimd.Multiply(PackedSimd.LoadVector128(r1), w0));
                    a1 = PackedSimd.Add(a1, PackedSimd.Multiply(PackedSimd.LoadVector128(r0 + 4), w1));
                    b1 = PackedSimd.Add(b1, PackedSimd.Multiply(PackedSimd.LoadVector128(r1 + 4), w1));
                    a2 = PackedSimd.Add(a2, PackedSimd.Multiply(PackedSimd.LoadVector128(r0 + 8), w2));
                    b2 = PackedSimd.Add(b2, PackedSimd.Multiply(PackedSimd.LoadVector128(r1 + 8), w2));
                    a3 = PackedSimd.Add(a3, PackedSimd.Multiply(PackedSimd.LoadVector128(r0 + 12), w3));
                    b3 = PackedSimd.Add(b3, PackedSimd.Multiply(PackedSimd.LoadVector128(r1 + 12), w3));
                }
            }
            PackedSimd.Store(acc0, a0); PackedSimd.Store(acc0 + 4, a1); PackedSimd.Store(acc0 + 8, a2); PackedSimd.Store(acc0 + 12, a3);
            PackedSimd.Store(acc1, b0); PackedSimd.Store(acc1 + 4, b1); PackedSimd.Store(acc1 + 8, b2); PackedSimd.Store(acc1 + 12, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps4PackedSimd(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector128<float> a0 = PackedSimd.LoadVector128(acc0), b0 = PackedSimd.LoadVector128(acc1);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector128<float> w0 = PackedSimd.LoadVector128(wk);
                    a0 = PackedSimd.Add(a0, PackedSimd.Multiply(PackedSimd.LoadVector128(r0), w0));
                    b0 = PackedSimd.Add(b0, PackedSimd.Multiply(PackedSimd.LoadVector128(r1), w0));
                }
            }
            PackedSimd.Store(acc0, a0);
            PackedSimd.Store(acc1, b0);
        }
    }

    // Epilogue pass: strip -> strided output row with residual/activation.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void FinishStripPackedSimd(float* strip, int block, int pixels, float* outRow, int channels, float* resRow,
        NhwcActivation activation, Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        for (int px = 0; px < pixels; px++)
        {
            float* acc = strip + px * block, dst = outRow + (long)px * channels;
            float* res = resRow == null ? null : resRow + (long)px * channels;
            for (int c = 0; c < block; c += 4)
                StoreDepthwisePackedSimd(dst + c, PackedSimd.LoadVector128(acc + c), res == null ? null : res + c, activation, alpha, beta, one);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreDepthwisePackedSimd(float* output, Vector128<float> acc, float* residual, NhwcActivation activation,
        Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        if (residual != null) acc = PackedSimd.Add(acc, PackedSimd.LoadVector128(residual));
        if (activation != NhwcActivation.None) acc = ActivatePackedSimd(acc, activation, alpha, beta, one);
        PackedSimd.Store(output, acc);
    }

    private static void BorderPixelPackedSimd(float* inBatch, float* w, float* bias, float* output, float* residual,
        int channels, int width, int iy0, int ix0, int kernelW, int taps, int kyBegin, int kyEnd,
        NhwcActivation activation, Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        for (int c = 0; c < channels; c += 4)
        {
            int c0 = c & ~31, block = Math.Min(32, channels - c0);
            float* wBlock = w + (long)c0 * taps + (c - c0);
            Vector128<float> acc = bias == null ? Vector128<float>.Zero : PackedSimd.LoadVector128(bias + c);
            for (int ky = kyBegin; ky < kyEnd; ky++)
            {
                float* inRow = inBatch + (long)(iy0 + ky) * width * channels;
                float* wRow = wBlock + (long)ky * kernelW * block;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = ix0 + kx;
                    if ((uint)ix >= (uint)width) continue;
                    acc = PackedSimd.Add(acc, PackedSimd.Multiply(PackedSimd.LoadVector128(inRow + (long)ix * channels + c),
                        PackedSimd.LoadVector128(wRow + (long)kx * block)));
                }
            }
            StoreDepthwisePackedSimd(output + c, acc, residual == null ? null : residual + c, activation, alpha, beta, one);
        }
    }
}
#endif
