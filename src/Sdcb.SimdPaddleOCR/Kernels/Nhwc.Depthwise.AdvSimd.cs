#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// ARM64 AdvSimd (NEON) depthwise channels-last convolution. Same structure as
// the AVX2 kernel (Nhwc.Depthwise.Avx.cs): interior pixels run a 2-pixel x
// 32-channel register block into a contiguous accumulator strip ([px][32]),
// border pixels check each tap. The 32-channel block is eight 128-bit
// accumulators per pixel (16 per pair + 8 weight registers per tap, well
// inside the 32 NEON registers); channel tails not reaching 32 run the 8-wide
// variants (two 128-bit accumulators per pixel). The fused epilogue reuses
// ActivateAdvSimd from Nhwc.Conv.AdvSimd.cs.
internal static unsafe partial class Nhwc
{
    /// <summary>AdvSimd driver of <see cref="Depthwise"/>; same loop nest as the AVX2 driver.</summary>
    internal static void DepthwiseAdvSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
                Vector128<float> valpha = Vector128.Create(alpha), vbeta = Vector128.Create(beta), one = Vector128.Create(1f);
                // Contiguous accumulator strip (32 channels x output width): the
                // strided output row would map onto a handful of L1 sets.
                float[] stripArray = System.Buffers.ArrayPool<float>.Shared.Rent(outputWidth * 32);
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
                        BorderPixelAdvSimd(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                            resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                            iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, valpha, vbeta, one);
                    int xPairEnd = x + ((xHi - x) & ~1);
                    if (xPairEnd > x)
                    {
                        // Interior pixel pairs, 32 channels at a time, accumulated in
                        // a contiguous strip ([px][32]): init with bias, accumulate
                        // (all taps in registers for short kernels; one kernel row
                        // per pass for tall ones, whose rows W*C*4 bytes apart would
                        // otherwise thrash one L1 set), then a single epilogue pass
                        // writes the strided output row. Keeping bias/epilogue out of
                        // the accumulate loops leaves the JIT 16 accumulators + 8
                        // weight registers without spills.
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
                                    AccumulateAllTaps32AdvSimd(inRow0, pixelStride, rowStride, wRow0, channels, kyEnd - kyBegin, kernelW, pixels, strip);
                                else
                                    for (int c8 = 0; c8 < block; c8 += 8)
                                        AccumulateAllTaps8AdvSimd(inRow0 + c8, pixelStride, rowStride, wRow0 + c8, channels, block, kyEnd - kyBegin, kernelW, pixels, strip + c8, block);
                            }
                            else
                            {
                                for (int ky = kyBegin; ky < kyEnd; ky++)
                                {
                                    float* inRow = inRow0 + (long)(ky - kyBegin) * rowStride;
                                    float* wRow = wRow0 + (long)(ky - kyBegin) * kernelW * block;
                                    if (block == 32)
                                        AccumulateRow32AdvSimd(inRow, pixelStride, wRow, channels, kernelW, pixels, strip);
                                    else
                                        for (int c8 = 0; c8 < block; c8 += 8)
                                            AccumulateRow8AdvSimd(inRow + c8, pixelStride, wRow + c8, channels, block, kernelW, pixels, strip + c8, block);
                                }
                            }
                            FinishStripAdvSimd(strip, block, pixels, outRowPtr + (long)x * channels + c, channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels + c, activation, valpha, vbeta, one);
                        }
                        x = xPairEnd;
                    }
                    for (; x < outputWidth; x++)
                        BorderPixelAdvSimd(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                            resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                            iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, valpha, vbeta, one);
                }
                }
                System.Buffers.ArrayPool<float>.Shared.Return(stripArray);
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // ------------------------------------------------ strip accumulate helpers
    // Same strip layout as the AVX2 kernel: [px][block] with block = 32 (or the
    // channel tail, stride `stripStride` for the 8-wide variants). Accumulate
    // loops carry only accumulators and weights in registers.

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps32AdvSimd(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels,
        int kyCount, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector128<float> a0 = AdvSimd.LoadVector128(acc0), a1 = AdvSimd.LoadVector128(acc0 + 4);
            Vector128<float> a2 = AdvSimd.LoadVector128(acc0 + 8), a3 = AdvSimd.LoadVector128(acc0 + 12);
            Vector128<float> a4 = AdvSimd.LoadVector128(acc0 + 16), a5 = AdvSimd.LoadVector128(acc0 + 20);
            Vector128<float> a6 = AdvSimd.LoadVector128(acc0 + 24), a7 = AdvSimd.LoadVector128(acc0 + 28);
            Vector128<float> b0 = AdvSimd.LoadVector128(acc1), b1 = AdvSimd.LoadVector128(acc1 + 4);
            Vector128<float> b2 = AdvSimd.LoadVector128(acc1 + 8), b3 = AdvSimd.LoadVector128(acc1 + 12);
            Vector128<float> b4 = AdvSimd.LoadVector128(acc1 + 16), b5 = AdvSimd.LoadVector128(acc1 + 20);
            Vector128<float> b6 = AdvSimd.LoadVector128(acc1 + 24), b7 = AdvSimd.LoadVector128(acc1 + 28);
            float* in0 = inRow0 + (long)px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
                {
                    Vector128<float> w0 = AdvSimd.LoadVector128(wk), w1 = AdvSimd.LoadVector128(wk + 4);
                    Vector128<float> w2 = AdvSimd.LoadVector128(wk + 8), w3 = AdvSimd.LoadVector128(wk + 12);
                    Vector128<float> w4 = AdvSimd.LoadVector128(wk + 16), w5 = AdvSimd.LoadVector128(wk + 20);
                    Vector128<float> w6 = AdvSimd.LoadVector128(wk + 24), w7 = AdvSimd.LoadVector128(wk + 28);
                    a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(r0), w0);
                    b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(r1), w0);
                    a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(r0 + 4), w1);
                    b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(r1 + 4), w1);
                    a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(r0 + 8), w2);
                    b2 = AdvSimd.FusedMultiplyAdd(b2, AdvSimd.LoadVector128(r1 + 8), w2);
                    a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(r0 + 12), w3);
                    b3 = AdvSimd.FusedMultiplyAdd(b3, AdvSimd.LoadVector128(r1 + 12), w3);
                    a4 = AdvSimd.FusedMultiplyAdd(a4, AdvSimd.LoadVector128(r0 + 16), w4);
                    b4 = AdvSimd.FusedMultiplyAdd(b4, AdvSimd.LoadVector128(r1 + 16), w4);
                    a5 = AdvSimd.FusedMultiplyAdd(a5, AdvSimd.LoadVector128(r0 + 20), w5);
                    b5 = AdvSimd.FusedMultiplyAdd(b5, AdvSimd.LoadVector128(r1 + 20), w5);
                    a6 = AdvSimd.FusedMultiplyAdd(a6, AdvSimd.LoadVector128(r0 + 24), w6);
                    b6 = AdvSimd.FusedMultiplyAdd(b6, AdvSimd.LoadVector128(r1 + 24), w6);
                    a7 = AdvSimd.FusedMultiplyAdd(a7, AdvSimd.LoadVector128(r0 + 28), w7);
                    b7 = AdvSimd.FusedMultiplyAdd(b7, AdvSimd.LoadVector128(r1 + 28), w7);
                }
            }
            AdvSimd.Store(acc0, a0); AdvSimd.Store(acc0 + 4, a1); AdvSimd.Store(acc0 + 8, a2); AdvSimd.Store(acc0 + 12, a3);
            AdvSimd.Store(acc0 + 16, a4); AdvSimd.Store(acc0 + 20, a5); AdvSimd.Store(acc0 + 24, a6); AdvSimd.Store(acc0 + 28, a7);
            AdvSimd.Store(acc1, b0); AdvSimd.Store(acc1 + 4, b1); AdvSimd.Store(acc1 + 8, b2); AdvSimd.Store(acc1 + 12, b3);
            AdvSimd.Store(acc1 + 16, b4); AdvSimd.Store(acc1 + 20, b5); AdvSimd.Store(acc1 + 24, b6); AdvSimd.Store(acc1 + 28, b7);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps8AdvSimd(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector128<float> a0 = AdvSimd.LoadVector128(acc0), a1 = AdvSimd.LoadVector128(acc0 + 4);
            Vector128<float> b0 = AdvSimd.LoadVector128(acc1), b1 = AdvSimd.LoadVector128(acc1 + 4);
            float* in0 = inRow0 + (long)px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector128<float> w0 = AdvSimd.LoadVector128(wk), w1 = AdvSimd.LoadVector128(wk + 4);
                    a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(r0), w0);
                    b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(r1), w0);
                    a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(r0 + 4), w1);
                    b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(r1 + 4), w1);
                }
            }
            AdvSimd.Store(acc0, a0); AdvSimd.Store(acc0 + 4, a1);
            AdvSimd.Store(acc1, b0); AdvSimd.Store(acc1 + 4, b1);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow32AdvSimd(float* inRow, long pixelStride, float* wRow, int channels, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector128<float> a0 = AdvSimd.LoadVector128(acc0), a1 = AdvSimd.LoadVector128(acc0 + 4);
            Vector128<float> a2 = AdvSimd.LoadVector128(acc0 + 8), a3 = AdvSimd.LoadVector128(acc0 + 12);
            Vector128<float> a4 = AdvSimd.LoadVector128(acc0 + 16), a5 = AdvSimd.LoadVector128(acc0 + 20);
            Vector128<float> a6 = AdvSimd.LoadVector128(acc0 + 24), a7 = AdvSimd.LoadVector128(acc0 + 28);
            Vector128<float> b0 = AdvSimd.LoadVector128(acc1), b1 = AdvSimd.LoadVector128(acc1 + 4);
            Vector128<float> b2 = AdvSimd.LoadVector128(acc1 + 8), b3 = AdvSimd.LoadVector128(acc1 + 12);
            Vector128<float> b4 = AdvSimd.LoadVector128(acc1 + 16), b5 = AdvSimd.LoadVector128(acc1 + 20);
            Vector128<float> b6 = AdvSimd.LoadVector128(acc1 + 24), b7 = AdvSimd.LoadVector128(acc1 + 28);
            float* r0 = inRow + (long)px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
            {
                Vector128<float> w0 = AdvSimd.LoadVector128(wk), w1 = AdvSimd.LoadVector128(wk + 4);
                Vector128<float> w2 = AdvSimd.LoadVector128(wk + 8), w3 = AdvSimd.LoadVector128(wk + 12);
                Vector128<float> w4 = AdvSimd.LoadVector128(wk + 16), w5 = AdvSimd.LoadVector128(wk + 20);
                Vector128<float> w6 = AdvSimd.LoadVector128(wk + 24), w7 = AdvSimd.LoadVector128(wk + 28);
                a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(r0), w0);
                b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(r1), w0);
                a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(r0 + 4), w1);
                b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(r1 + 4), w1);
                a2 = AdvSimd.FusedMultiplyAdd(a2, AdvSimd.LoadVector128(r0 + 8), w2);
                b2 = AdvSimd.FusedMultiplyAdd(b2, AdvSimd.LoadVector128(r1 + 8), w2);
                a3 = AdvSimd.FusedMultiplyAdd(a3, AdvSimd.LoadVector128(r0 + 12), w3);
                b3 = AdvSimd.FusedMultiplyAdd(b3, AdvSimd.LoadVector128(r1 + 12), w3);
                a4 = AdvSimd.FusedMultiplyAdd(a4, AdvSimd.LoadVector128(r0 + 16), w4);
                b4 = AdvSimd.FusedMultiplyAdd(b4, AdvSimd.LoadVector128(r1 + 16), w4);
                a5 = AdvSimd.FusedMultiplyAdd(a5, AdvSimd.LoadVector128(r0 + 20), w5);
                b5 = AdvSimd.FusedMultiplyAdd(b5, AdvSimd.LoadVector128(r1 + 20), w5);
                a6 = AdvSimd.FusedMultiplyAdd(a6, AdvSimd.LoadVector128(r0 + 24), w6);
                b6 = AdvSimd.FusedMultiplyAdd(b6, AdvSimd.LoadVector128(r1 + 24), w6);
                a7 = AdvSimd.FusedMultiplyAdd(a7, AdvSimd.LoadVector128(r0 + 28), w7);
                b7 = AdvSimd.FusedMultiplyAdd(b7, AdvSimd.LoadVector128(r1 + 28), w7);
            }
            AdvSimd.Store(acc0, a0); AdvSimd.Store(acc0 + 4, a1); AdvSimd.Store(acc0 + 8, a2); AdvSimd.Store(acc0 + 12, a3);
            AdvSimd.Store(acc0 + 16, a4); AdvSimd.Store(acc0 + 20, a5); AdvSimd.Store(acc0 + 24, a6); AdvSimd.Store(acc0 + 28, a7);
            AdvSimd.Store(acc1, b0); AdvSimd.Store(acc1 + 4, b1); AdvSimd.Store(acc1 + 8, b2); AdvSimd.Store(acc1 + 12, b3);
            AdvSimd.Store(acc1 + 16, b4); AdvSimd.Store(acc1 + 20, b5); AdvSimd.Store(acc1 + 24, b6); AdvSimd.Store(acc1 + 28, b7);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow8AdvSimd(float* inRow, long pixelStride, float* wRow, int channels, int weightStride, int kernelW, int pixels,
        float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector128<float> a0 = AdvSimd.LoadVector128(acc0), a1 = AdvSimd.LoadVector128(acc0 + 4);
            Vector128<float> b0 = AdvSimd.LoadVector128(acc1), b1 = AdvSimd.LoadVector128(acc1 + 4);
            float* r0 = inRow + (long)px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
            {
                Vector128<float> w0 = AdvSimd.LoadVector128(wk), w1 = AdvSimd.LoadVector128(wk + 4);
                a0 = AdvSimd.FusedMultiplyAdd(a0, AdvSimd.LoadVector128(r0), w0);
                b0 = AdvSimd.FusedMultiplyAdd(b0, AdvSimd.LoadVector128(r1), w0);
                a1 = AdvSimd.FusedMultiplyAdd(a1, AdvSimd.LoadVector128(r0 + 4), w1);
                b1 = AdvSimd.FusedMultiplyAdd(b1, AdvSimd.LoadVector128(r1 + 4), w1);
            }
            AdvSimd.Store(acc0, a0); AdvSimd.Store(acc0 + 4, a1);
            AdvSimd.Store(acc1, b0); AdvSimd.Store(acc1 + 4, b1);
        }
    }

    // Epilogue pass: strip -> strided output row with residual/activation.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void FinishStripAdvSimd(float* strip, int block, int pixels, float* outRow, int channels, float* resRow,
        NhwcActivation activation, Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        for (int px = 0; px < pixels; px++)
        {
            float* acc = strip + px * block, dst = outRow + (long)px * channels;
            float* res = resRow == null ? null : resRow + (long)px * channels;
            for (int c = 0; c < block; c += 4)
                StoreDepthwiseAdvSimd(dst + c, AdvSimd.LoadVector128(acc + c), res == null ? null : res + c, activation, alpha, beta, one);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreDepthwiseAdvSimd(float* output, Vector128<float> acc, float* residual, NhwcActivation activation,
        Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        if (residual != null) acc = AdvSimd.Add(acc, AdvSimd.LoadVector128(residual));
        if (activation != NhwcActivation.None) acc = ActivateAdvSimd(acc, activation, alpha, beta, one);
        AdvSimd.Store(output, acc);
    }

    private static void BorderPixelAdvSimd(float* inBatch, float* w, float* bias, float* output, float* residual,
        int channels, int width, int iy0, int ix0, int kernelW, int taps, int kyBegin, int kyEnd,
        NhwcActivation activation, Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        for (int c = 0; c < channels; c += 4)
        {
            int c0 = c & ~31, block = Math.Min(32, channels - c0);
            float* wBlock = w + (long)c0 * taps + (c - c0);
            Vector128<float> acc = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + c);
            for (int ky = kyBegin; ky < kyEnd; ky++)
            {
                float* inRow = inBatch + (long)(iy0 + ky) * width * channels;
                float* wRow = wBlock + (long)ky * kernelW * block;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = ix0 + kx;
                    if ((uint)ix >= (uint)width) continue;
                    acc = AdvSimd.FusedMultiplyAdd(acc, AdvSimd.LoadVector128(inRow + (long)ix * channels + c),
                        AdvSimd.LoadVector128(wRow + (long)kx * block));
                }
            }
            StoreDepthwiseAdvSimd(output + c, acc, residual == null ? null : residual + c, activation, alpha, beta, one);
        }
    }
}
#endif
