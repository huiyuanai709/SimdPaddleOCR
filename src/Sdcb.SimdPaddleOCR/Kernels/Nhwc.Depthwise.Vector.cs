#if NETSTANDARD2_0
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

// Vector<float> (netstandard2.0) depthwise kernel: a mechanical port of the
// AVX2 structure. Count == 8 runs a 2-pixel x 32-channel register block
// (eight Vector<float> accumulators + four weight vectors). Count == 4 keeps
// the same eight-accumulator budget as two 16-channel half-block passes over
// the packed [c/32][tap][32] panel. Border pixels check each tap. Every
// accumulation is a separate multiply then add (Vector<T> has no FMA), matching
// the NCHW Vector kernels lane for lane. Other vector widths fall back to a
// scalar per-channel loop with the same tap order.
internal static unsafe partial class Nhwc
{
    /// <summary>
    /// Depthwise KxK convolution, any stride/padding. Weights are packed
    /// [c/32][tap][32] (<see cref="PackDepthwise"/>).
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
                if (Vector<float>.Count != 8 && Vector<float>.Count != 4)
                {
                    for (int row = rowBegin; row < rowEnd; row++)
                    {
                        int b0 = row / outputHeight, y0 = row - b0 * outputHeight;
                        float* inBatch0 = inBase + (long)b0 * height * width * channels;
                        int iy0v = y0 * strideH - padTop;
                        int kyBegin0 = Math.Max(0, -iy0v), kyEnd0 = Math.Min(kernelH, height - iy0v);
                        long outRow0 = ((long)b0 * outputHeight + y0) * outputWidth;
                        float* outRowPtr0 = outBase + outRow0 * channels;
                        float* resRowPtr0 = resBase == null ? null : resBase + outRow0 * channels;
                        for (int x = 0; x < outputWidth; x++)
                            DepthwisePixelScalar(inBatch0, w, biasBase, outRowPtr0 + (long)x * channels,
                                resRowPtr0 == null ? null : resRowPtr0 + (long)x * channels, channels, width,
                                iy0v, x * strideW - padLeft, kernelW, taps, kyBegin0, kyEnd0, activation, alpha, beta);
                    }
                    return;
                }
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
                            BorderPixelVec(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                                iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, alpha, beta);
                        int xPairEnd = x + ((xHi - x) & ~1);
                        if (xPairEnd > x)
                        {
                            // Interior pixel pairs, 32 channels at a time, accumulated in
                            // a contiguous strip ([px][32]): init with bias, accumulate
                            // (all taps in registers for short kernels; one kernel row
                            // per pass for tall ones), then a single epilogue pass
                            // writes the strided output row.
                            int pixels = xPairEnd - x;
                            long rowStride = (long)width * channels, pixelStride = (long)strideW * channels;
                            bool strips = kernelH >= DepthwiseStripMinKernelH;
                            bool wide8 = Vector<float>.Count == 8;
                            for (int c = 0; c < channels; c += 32)
                            {
                                int block = Math.Min(32, channels - c);
                                InitStrip(strip, biasBase == null ? null : biasBase + c, block, pixels);
                                float* inRow0 = inBatch + ((long)(iy0 + kyBegin) * width - padLeft) * channels + (long)x * pixelStride + c;
                                float* wRow0 = w + (long)c * taps + (long)kyBegin * kernelW * block;
                                if (!strips)
                                {
                                    if (wide8)
                                    {
                                        if (block == 32)
                                            AccumulateAllTaps32Vec(inRow0, pixelStride, rowStride, wRow0, channels, kyEnd - kyBegin, kernelW, pixels, strip);
                                        else
                                            for (int c8 = 0; c8 < block; c8 += 8)
                                                AccumulateAllTaps8Vec(inRow0 + c8, pixelStride, rowStride, wRow0 + c8, channels, block, kyEnd - kyBegin, kernelW, pixels, strip + c8, block);
                                    }
                                    else if (block == 32)
                                    {
                                        AccumulateAllTaps16Vec4(inRow0, pixelStride, rowStride, wRow0, channels, 32, kyEnd - kyBegin, kernelW, pixels, strip, 32);
                                        AccumulateAllTaps16Vec4(inRow0 + 16, pixelStride, rowStride, wRow0 + 16, channels, 32, kyEnd - kyBegin, kernelW, pixels, strip + 16, 32);
                                    }
                                    else
                                    {
                                        for (int c4 = 0; c4 < block; c4 += 4)
                                            AccumulateAllTaps4Vec4(inRow0 + c4, pixelStride, rowStride, wRow0 + c4, channels, block, kyEnd - kyBegin, kernelW, pixels, strip + c4, block);
                                    }
                                }
                                else
                                {
                                    for (int ky = kyBegin; ky < kyEnd; ky++)
                                    {
                                        float* inRow = inRow0 + (long)(ky - kyBegin) * rowStride;
                                        float* wRow = wRow0 + (long)(ky - kyBegin) * kernelW * block;
                                        if (wide8)
                                        {
                                            if (block == 32)
                                                AccumulateRow32Vec(inRow, pixelStride, wRow, channels, kernelW, pixels, strip);
                                            else
                                                for (int c8 = 0; c8 < block; c8 += 8)
                                                    AccumulateRow8Vec(inRow + c8, pixelStride, wRow + c8, channels, block, kernelW, pixels, strip + c8, block);
                                        }
                                        else if (block == 32)
                                        {
                                            AccumulateRow16Vec4(inRow, pixelStride, wRow, channels, 32, kernelW, pixels, strip, 32);
                                            AccumulateRow16Vec4(inRow + 16, pixelStride, wRow + 16, channels, 32, kernelW, pixels, strip + 16, 32);
                                        }
                                        else
                                        {
                                            for (int c4 = 0; c4 < block; c4 += 4)
                                                AccumulateRow4Vec4(inRow + c4, pixelStride, wRow + c4, channels, block, kernelW, pixels, strip + c4, block);
                                        }
                                    }
                                }
                                FinishStripVec(strip, block, pixels, outRowPtr + (long)x * channels + c, channels,
                                    resRowPtr == null ? null : resRowPtr + (long)x * channels + c, activation, alpha, beta);
                            }
                            x = xPairEnd;
                        }
                        for (; x < outputWidth; x++)
                            BorderPixelVec(inBatch, w, biasBase, outRowPtr + (long)x * channels,
                                resRowPtr == null ? null : resRowPtr + (long)x * channels, channels, width,
                                iy0, x * strideW - padLeft, kernelW, taps, kyBegin, kyEnd, activation, alpha, beta);
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
    private static void AccumulateAllTaps32Vec(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels,
        int kyCount, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector<float> a0 = VectorLoad(acc0), a1 = VectorLoad(acc0 + 8);
            Vector<float> a2 = VectorLoad(acc0 + 16), a3 = VectorLoad(acc0 + 24);
            Vector<float> b0 = VectorLoad(acc1), b1 = VectorLoad(acc1 + 8);
            Vector<float> b2 = VectorLoad(acc1 + 16), b3 = VectorLoad(acc1 + 24);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
                {
                    Vector<float> w0 = VectorLoad(wk), w1 = VectorLoad(wk + 8);
                    Vector<float> w2 = VectorLoad(wk + 16), w3 = VectorLoad(wk + 24);
                    a0 += VectorLoad(r0) * w0;
                    b0 += VectorLoad(r1) * w0;
                    a1 += VectorLoad(r0 + 8) * w1;
                    b1 += VectorLoad(r1 + 8) * w1;
                    a2 += VectorLoad(r0 + 16) * w2;
                    b2 += VectorLoad(r1 + 16) * w2;
                    a3 += VectorLoad(r0 + 24) * w3;
                    b3 += VectorLoad(r1 + 24) * w3;
                }
            }
            VectorStore(acc0, a0); VectorStore(acc0 + 8, a1); VectorStore(acc0 + 16, a2); VectorStore(acc0 + 24, a3);
            VectorStore(acc1, b0); VectorStore(acc1 + 8, b1); VectorStore(acc1 + 16, b2); VectorStore(acc1 + 24, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps8Vec(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), b0 = VectorLoad(acc1);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector<float> w0 = VectorLoad(wk);
                    a0 += VectorLoad(r0) * w0;
                    b0 += VectorLoad(r1) * w0;
                }
            }
            VectorStore(acc0, a0);
            VectorStore(acc1, b0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow32Vec(float* inRow, long pixelStride, float* wRow, int channels, int kernelW, int pixels, float* strip)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * 32, acc1 = acc0 + 32;
            Vector<float> a0 = VectorLoad(acc0), a1 = VectorLoad(acc0 + 8);
            Vector<float> a2 = VectorLoad(acc0 + 16), a3 = VectorLoad(acc0 + 24);
            Vector<float> b0 = VectorLoad(acc1), b1 = VectorLoad(acc1 + 8);
            Vector<float> b2 = VectorLoad(acc1 + 16), b3 = VectorLoad(acc1 + 24);
            float* r0 = inRow + px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += 32, r0 += channels, r1 += channels)
            {
                Vector<float> w0 = VectorLoad(wk), w1 = VectorLoad(wk + 8);
                Vector<float> w2 = VectorLoad(wk + 16), w3 = VectorLoad(wk + 24);
                a0 += VectorLoad(r0) * w0;
                b0 += VectorLoad(r1) * w0;
                a1 += VectorLoad(r0 + 8) * w1;
                b1 += VectorLoad(r1 + 8) * w1;
                a2 += VectorLoad(r0 + 16) * w2;
                b2 += VectorLoad(r1 + 16) * w2;
                a3 += VectorLoad(r0 + 24) * w3;
                b3 += VectorLoad(r1 + 24) * w3;
            }
            VectorStore(acc0, a0); VectorStore(acc0 + 8, a1); VectorStore(acc0 + 16, a2); VectorStore(acc0 + 24, a3);
            VectorStore(acc1, b0); VectorStore(acc1 + 8, b1); VectorStore(acc1 + 16, b2); VectorStore(acc1 + 24, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow8Vec(float* inRow, long pixelStride, float* wRow, int channels, int weightStride, int kernelW, int pixels,
        float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), b0 = VectorLoad(acc1);
            float* r0 = inRow + px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
            {
                Vector<float> w0 = VectorLoad(wk);
                a0 += VectorLoad(r0) * w0;
                b0 += VectorLoad(r1) * w0;
            }
            VectorStore(acc0, a0);
            VectorStore(acc1, b0);
        }
    }

    // Count == 4: 2 pixels x 16 channels (eight accumulators), weight/strip
    // stride is the packed block width so a 32-channel panel is two half-passes.

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps16Vec4(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), a1 = VectorLoad(acc0 + 4);
            Vector<float> a2 = VectorLoad(acc0 + 8), a3 = VectorLoad(acc0 + 12);
            Vector<float> b0 = VectorLoad(acc1), b1 = VectorLoad(acc1 + 4);
            Vector<float> b2 = VectorLoad(acc1 + 8), b3 = VectorLoad(acc1 + 12);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector<float> w0 = VectorLoad(wk), w1 = VectorLoad(wk + 4);
                    Vector<float> w2 = VectorLoad(wk + 8), w3 = VectorLoad(wk + 12);
                    a0 += VectorLoad(r0) * w0;
                    b0 += VectorLoad(r1) * w0;
                    a1 += VectorLoad(r0 + 4) * w1;
                    b1 += VectorLoad(r1 + 4) * w1;
                    a2 += VectorLoad(r0 + 8) * w2;
                    b2 += VectorLoad(r1 + 8) * w2;
                    a3 += VectorLoad(r0 + 12) * w3;
                    b3 += VectorLoad(r1 + 12) * w3;
                }
            }
            VectorStore(acc0, a0); VectorStore(acc0 + 4, a1); VectorStore(acc0 + 8, a2); VectorStore(acc0 + 12, a3);
            VectorStore(acc1, b0); VectorStore(acc1 + 4, b1); VectorStore(acc1 + 8, b2); VectorStore(acc1 + 12, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateAllTaps4Vec4(float* inRow0, long pixelStride, long rowStride, float* wRow0, int channels, int weightStride,
        int kyCount, int kernelW, int pixels, float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), b0 = VectorLoad(acc1);
            float* in0 = inRow0 + px * pixelStride, wk = wRow0;
            for (int ky = 0; ky < kyCount; ky++, in0 += rowStride)
            {
                float* r0 = in0, r1 = in0 + pixelStride;
                for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
                {
                    Vector<float> w0 = VectorLoad(wk);
                    a0 += VectorLoad(r0) * w0;
                    b0 += VectorLoad(r1) * w0;
                }
            }
            VectorStore(acc0, a0);
            VectorStore(acc1, b0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow16Vec4(float* inRow, long pixelStride, float* wRow, int channels, int weightStride, int kernelW, int pixels,
        float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), a1 = VectorLoad(acc0 + 4);
            Vector<float> a2 = VectorLoad(acc0 + 8), a3 = VectorLoad(acc0 + 12);
            Vector<float> b0 = VectorLoad(acc1), b1 = VectorLoad(acc1 + 4);
            Vector<float> b2 = VectorLoad(acc1 + 8), b3 = VectorLoad(acc1 + 12);
            float* r0 = inRow + px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
            {
                Vector<float> w0 = VectorLoad(wk), w1 = VectorLoad(wk + 4);
                Vector<float> w2 = VectorLoad(wk + 8), w3 = VectorLoad(wk + 12);
                a0 += VectorLoad(r0) * w0;
                b0 += VectorLoad(r1) * w0;
                a1 += VectorLoad(r0 + 4) * w1;
                b1 += VectorLoad(r1 + 4) * w1;
                a2 += VectorLoad(r0 + 8) * w2;
                b2 += VectorLoad(r1 + 8) * w2;
                a3 += VectorLoad(r0 + 12) * w3;
                b3 += VectorLoad(r1 + 12) * w3;
            }
            VectorStore(acc0, a0); VectorStore(acc0 + 4, a1); VectorStore(acc0 + 8, a2); VectorStore(acc0 + 12, a3);
            VectorStore(acc1, b0); VectorStore(acc1 + 4, b1); VectorStore(acc1 + 8, b2); VectorStore(acc1 + 12, b3);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void AccumulateRow4Vec4(float* inRow, long pixelStride, float* wRow, int channels, int weightStride, int kernelW, int pixels,
        float* strip, int stripStride)
    {
        for (int px = 0; px < pixels; px += 2)
        {
            float* acc0 = strip + px * stripStride, acc1 = acc0 + stripStride;
            Vector<float> a0 = VectorLoad(acc0), b0 = VectorLoad(acc1);
            float* r0 = inRow + px * pixelStride, r1 = r0 + pixelStride, wk = wRow;
            for (int kx = 0; kx < kernelW; kx++, wk += weightStride, r0 += channels, r1 += channels)
            {
                Vector<float> w0 = VectorLoad(wk);
                a0 += VectorLoad(r0) * w0;
                b0 += VectorLoad(r1) * w0;
            }
            VectorStore(acc0, a0);
            VectorStore(acc1, b0);
        }
    }

    // Epilogue pass: strip -> strided output row with residual/activation.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void FinishStripVec(float* strip, int block, int pixels, float* outRow, int channels, float* resRow,
        NhwcActivation activation, float alphaScalar, float betaScalar)
    {
        Vector<float> alpha = new(alphaScalar), beta = new(betaScalar), one = new(1f);
        int vecWidth = Vector<float>.Count;
        for (int px = 0; px < pixels; px++)
        {
            float* acc = strip + px * block, dst = outRow + (long)px * channels;
            float* res = resRow == null ? null : resRow + (long)px * channels;
            for (int c = 0; c < block; c += vecWidth)
                StoreDepthwiseVec(dst + c, VectorLoad(acc + c), res == null ? null : res + c, activation, alpha, beta, one);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreDepthwiseVec(float* output, Vector<float> acc, float* residual, NhwcActivation activation,
        Vector<float> alpha, Vector<float> beta, Vector<float> one)
    {
        if (residual != null) acc += VectorLoad(residual);
        if (activation != NhwcActivation.None) acc = ActivateVec(acc, activation, alpha, beta, one);
        VectorStore(output, acc);
    }

    private static void BorderPixelVec(float* inBatch, float* w, float* bias, float* output, float* residual,
        int channels, int width, int iy0, int ix0, int kernelW, int taps, int kyBegin, int kyEnd,
        NhwcActivation activation, float alphaScalar, float betaScalar)
    {
        Vector<float> alpha = new(alphaScalar), beta = new(betaScalar), one = new(1f);
        int vecWidth = Vector<float>.Count;
        for (int c = 0; c < channels; c += vecWidth)
        {
            int c0 = c & ~31, block = Math.Min(32, channels - c0);
            float* wBlock = w + (long)c0 * taps + (c - c0);
            Vector<float> acc = bias == null ? Vector<float>.Zero : VectorLoad(bias + c);
            for (int ky = kyBegin; ky < kyEnd; ky++)
            {
                float* inRow = inBatch + (long)(iy0 + ky) * width * channels;
                float* wRow = wBlock + (long)ky * kernelW * block;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = ix0 + kx;
                    if ((uint)ix >= (uint)width) continue;
                    acc += VectorLoad(inRow + (long)ix * channels + c) * VectorLoad(wRow + (long)kx * block);
                }
            }
            StoreDepthwiseVec(output + c, acc, residual == null ? null : residual + c, activation, alpha, beta, one);
        }
    }

    // Scalar per-channel fallback for vector widths other than 8; same bias
    // init, ascending tap order and epilogue as the vector path.
    private static void DepthwisePixelScalar(float* inBatch, float* w, float* bias, float* output, float* residual,
        int channels, int width, int iy0, int ix0, int kernelW, int taps, int kyBegin, int kyEnd,
        NhwcActivation activation, float alpha, float beta)
    {
        for (int c = 0; c < channels; c++)
        {
            float acc = bias == null ? 0f : bias[c];
            for (int ky = kyBegin; ky < kyEnd; ky++)
            {
                float* inRow = inBatch + (long)(iy0 + ky) * width * channels;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = ix0 + kx;
                    if ((uint)ix >= (uint)width) continue;
                    acc += inRow[(long)ix * channels + c] * w[DepthwiseWeightOffset(channels, taps, c, ky * kernelW + kx)];
                }
            }
            if (residual != null) acc += residual[c];
            output[c] = ActivateScalarValue(acc, activation, alpha, beta);
        }
    }
}
#endif
