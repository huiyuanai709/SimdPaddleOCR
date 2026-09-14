using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class Nhwc
{
    /// <summary>
    /// Depthwise KxK convolution, any stride/padding. Weights are packed
    /// [tap][c] (<see cref="PackDepthwise"/>). Interior pixels run a 2-pixel x
    /// 32-channel register block; border pixels check each tap.
    /// </summary>
    internal static void Depthwise(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
                Vector256<float> valpha = Vector256.Create(alpha), vbeta = Vector256.Create(beta), one = Vector256.Create(1f);
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
                        BorderPixel(inBatch, w, biasBase, outRowPtr + (long)x * channels,
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
                        // the accumulate loops leaves the JIT 8 accumulators + 4
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
                                    AccumulateAllTaps32(inRow0, pixelStride, rowStride, wRow0, channels, kyEnd - kyBegin, kernelW, pixels, strip);
                                else
                                    for (int c8 = 0; c8 < block; c8 += 8)
                                        AccumulateAllTaps8(inRow0 + c8, pixelStride, rowStride, wRow0 + c8, channels, block, kyEnd - kyBegin, kernelW, pixels, strip + c8, block);
                            }
                            else
                            {
                                for (int ky = kyBegin; ky < kyEnd; ky++)
                                {
                                    float* inRow = inRow0 + (long)(ky - kyBegin) * rowStride;
                                    float* wRow = wRow0 + (long)(ky - kyBegin) * kernelW * block;
                                    if (block == 32)
                                        AccumulateRow32(inRow, pixelStride, wRow, channels, kernelW, pixels, strip);
                                    else
                                        for (int c8 = 0; c8 < block; c8 += 8)
                                            AccumulateRow8(inRow + c8, pixelStride, wRow + c8, channels, block, kernelW, pixels, strip + c8, block);
                                }
                            }
                            FinishStrip(strip, block, pixels, outRowPtr + (long)x * channels + c, channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels + c, activation, valpha, vbeta, one);
                        }
                        x = xPairEnd;
                    }
                    for (; x < outputWidth; x++)
                        BorderPixel(inBatch, w, biasBase, outRowPtr + (long)x * channels,
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
    // The strip is [px][block] with block = 32 (or the channel tail, stride
    // `stripStride` for the 8-wide variants). Accumulate loops carry only
    // accumulators and weights in registers.

    private static void InitStrip(float* strip, float* bias, int block, int pixels)
    {
        if (bias == null)
        {
            new Span<float>(strip, pixels * block).Clear();
            return;
        }
        for (int px = 0; px < pixels; px++)
            Buffer.MemoryCopy(bias, strip + px * block, block * sizeof(float), block * sizeof(float));
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps32(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels,
        int kyCount, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector256<float> a0 = Avx.LoadVector256(acc0), a1 = Avx.LoadVector256(acc0 + 8);
            Vector256<float> a2 = Avx.LoadVector256(acc0 + 16), a3 = Avx.LoadVector256(acc0 + 24);
            Vector256<float> b0 = Avx.LoadVector256(acc1), b1 = Avx.LoadVector256(acc1 + 8);
            Vector256<float> b2 = Avx.LoadVector256(acc1 + 16), b3 = Avx.LoadVector256(acc1 + 24);
            float* in0 = inRow0 + (long)px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
                {
                    Vector256<float> w0 = Avx.LoadVector256(wk), w1 = Avx.LoadVector256(wk + 8);
                    Vector256<float> w2 = Avx.LoadVector256(wk + 16), w3 = Avx.LoadVector256(wk + 24);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(r0), w0, a0);
                    b0 = Fma.MultiplyAdd(Avx.LoadVector256(r1), w0, b0);
                    a1 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 8), w1, a1);
                    b1 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 8), w1, b1);
                    a2 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 16), w2, a2);
                    b2 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 16), w2, b2);
                    a3 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 24), w3, a3);
                    b3 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 24), w3, b3);
                }
            }
            Avx.Store(acc0, a0); Avx.Store(acc0 + 8, a1); Avx.Store(acc0 + 16, a2); Avx.Store(acc0 + 24, a3);
            Avx.Store(acc1, b0); Avx.Store(acc1 + 8, b1); Avx.Store(acc1 + 16, b2); Avx.Store(acc1 + 24, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps8(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector256<float> a0 = Avx.LoadVector256(acc0), b0 = Avx.LoadVector256(acc1);
            float* in0 = inRow0 + (long)px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector256<float> w0 = Avx.LoadVector256(wk);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(r0), w0, a0);
                    b0 = Fma.MultiplyAdd(Avx.LoadVector256(r1), w0, b0);
                }
            }
            Avx.Store(acc0, a0);
            Avx.Store(acc1, b0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow32(float* inRow, long pixelStride, float* wRow, int channels, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector256<float> a0 = Avx.LoadVector256(acc0), a1 = Avx.LoadVector256(acc0 + 8);
            Vector256<float> a2 = Avx.LoadVector256(acc0 + 16), a3 = Avx.LoadVector256(acc0 + 24);
            Vector256<float> b0 = Avx.LoadVector256(acc1), b1 = Avx.LoadVector256(acc1 + 8);
            Vector256<float> b2 = Avx.LoadVector256(acc1 + 16), b3 = Avx.LoadVector256(acc1 + 24);
            float* r0 = inRow + (long)px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
            {
                Vector256<float> w0 = Avx.LoadVector256(wk), w1 = Avx.LoadVector256(wk + 8);
                Vector256<float> w2 = Avx.LoadVector256(wk + 16), w3 = Avx.LoadVector256(wk + 24);
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(r0), w0, a0);
                b0 = Fma.MultiplyAdd(Avx.LoadVector256(r1), w0, b0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 8), w1, a1);
                b1 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 8), w1, b1);
                a2 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 16), w2, a2);
                b2 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 16), w2, b2);
                a3 = Fma.MultiplyAdd(Avx.LoadVector256(r0 + 24), w3, a3);
                b3 = Fma.MultiplyAdd(Avx.LoadVector256(r1 + 24), w3, b3);
            }
            Avx.Store(acc0, a0); Avx.Store(acc0 + 8, a1); Avx.Store(acc0 + 16, a2); Avx.Store(acc0 + 24, a3);
            Avx.Store(acc1, b0); Avx.Store(acc1 + 8, b1); Avx.Store(acc1 + 16, b2); Avx.Store(acc1 + 24, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow8(float* inRow, long pixelStride, float* wRow, int channels, int weightStride, int kernelW, int pixels,
        float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector256<float> a0 = Avx.LoadVector256(acc0), b0 = Avx.LoadVector256(acc1);
            float* r0 = inRow + (long)px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
            {
                Vector256<float> w0 = Avx.LoadVector256(wk);
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(r0), w0, a0);
                b0 = Fma.MultiplyAdd(Avx.LoadVector256(r1), w0, b0);
            }
            Avx.Store(acc0, a0);
            Avx.Store(acc1, b0);
        }
    }

    // Epilogue pass: strip -> strided output row with residual/activation.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void FinishStrip(float* strip, int block, int pixels, float* outRow, int channels, float* resRow,
        NhwcActivation activation, Vector256<float> alpha, Vector256<float> beta, Vector256<float> one)
    {
        for (int px = 0; px < pixels; px++)
        {
            float* acc = strip + px * block, dst = outRow + (long)px * channels;
            float* res = resRow == null ? null : resRow + (long)px * channels;
            for (int c = 0; c < block; c += 8)
                StoreDepthwise(dst + c, Avx.LoadVector256(acc + c), res == null ? null : res + c, activation, alpha, beta, one);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreDepthwise(float* output, Vector256<float> acc, float* residual, NhwcActivation activation,
        Vector256<float> alpha, Vector256<float> beta, Vector256<float> one)
    {
        if (residual != null) acc = Avx.Add(acc, Avx.LoadVector256(residual));
        if (activation != NhwcActivation.None) acc = Activate(acc, activation, alpha, beta, one);
        Avx.Store(output, acc);
    }

    private static void BorderPixel(float* inBatch, float* w, float* bias, float* output, float* residual,
        int channels, int width, int iy0, int ix0, int kernelW, int taps, int kyBegin, int kyEnd,
        NhwcActivation activation, Vector256<float> alpha, Vector256<float> beta, Vector256<float> one)
    {
        for (int c = 0; c < channels; c += 8)
        {
            int c0 = c & ~31, block = Math.Min(32, channels - c0);
            float* wBlock = w + (long)c0 * taps + (c - c0);
            Vector256<float> acc = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + c);
            for (int ky = kyBegin; ky < kyEnd; ky++)
            {
                float* inRow = inBatch + (long)(iy0 + ky) * width * channels;
                float* wRow = wBlock + (long)ky * kernelW * block;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = ix0 + kx;
                    if ((uint)ix >= (uint)width) continue;
                    acc = Fma.MultiplyAdd(Avx.LoadVector256(inRow + (long)ix * channels + c),
                        Avx.LoadVector256(wRow + (long)kx * block), acc);
                }
            }
            StoreDepthwise(output + c, acc, residual == null ? null : residual + c, activation, alpha, beta, one);
        }
    }
}

