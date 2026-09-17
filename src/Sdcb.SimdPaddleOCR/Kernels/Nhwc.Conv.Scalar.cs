using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// True scalar NHWC tiles: same 6x16 blocking, packed [k][16] layout and
// ascending-k accumulation as the Vector kernels, without Vector<T>.
internal static unsafe partial class Nhwc
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadTile16(float* src, out float a0, out float a1, out float a2, out float a3,
        out float a4, out float a5, out float a6, out float a7, out float a8, out float a9,
        out float a10, out float a11, out float a12, out float a13, out float a14, out float a15)
    {
        a0 = src[0]; a1 = src[1]; a2 = src[2]; a3 = src[3];
        a4 = src[4]; a5 = src[5]; a6 = src[6]; a7 = src[7];
        a8 = src[8]; a9 = src[9]; a10 = src[10]; a11 = src[11];
        a12 = src[12]; a13 = src[13]; a14 = src[14]; a15 = src[15];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreTile16(float* dst, float a0, float a1, float a2, float a3,
        float a4, float a5, float a6, float a7, float a8, float a9,
        float a10, float a11, float a12, float a13, float a14, float a15)
    {
        dst[0] = a0; dst[1] = a1; dst[2] = a2; dst[3] = a3;
        dst[4] = a4; dst[5] = a5; dst[6] = a6; dst[7] = a7;
        dst[8] = a8; dst[9] = a9; dst[10] = a10; dst[11] = a11;
        dst[12] = a12; dst[13] = a13; dst[14] = a14; dst[15] = a15;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void GemmTile16Scalar(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        for (int r = 0; r < rows; r++)
        {
            float* pr = input + (long)r * inStride;
            float* acc = partial + r * OcBlock;
            float a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
            if (k0 == 0)
            {
                if (bias == null)
                    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = a10 = a11 = a12 = a13 = a14 = a15 = 0f;
                else
                    LoadTile16(bias, out a0, out a1, out a2, out a3, out a4, out a5, out a6, out a7,
                        out a8, out a9, out a10, out a11, out a12, out a13, out a14, out a15);
            }
            else
                LoadTile16(acc, out a0, out a1, out a2, out a3, out a4, out a5, out a6, out a7,
                    out a8, out a9, out a10, out a11, out a12, out a13, out a14, out a15);
            float* wk = w;
            for (int k = k0; k < kEnd; k++, wk += OcBlock)
            {
                float v = pr[k];
                a0 += v * wk[0]; a1 += v * wk[1]; a2 += v * wk[2]; a3 += v * wk[3];
                a4 += v * wk[4]; a5 += v * wk[5]; a6 += v * wk[6]; a7 += v * wk[7];
                a8 += v * wk[8]; a9 += v * wk[9]; a10 += v * wk[10]; a11 += v * wk[11];
                a12 += v * wk[12]; a13 += v * wk[13]; a14 += v * wk[14]; a15 += v * wk[15];
            }
            StoreTile16(acc, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTile16Scalar(float* input, int rows, int pixelStride,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        for (int r = 0; r < rows; r++)
        {
            float* pr = input + (long)r * pixelStride;
            float* acc = partial + r * OcBlock;
            float a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
            if (k0 == 0)
            {
                if (bias == null)
                    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = a10 = a11 = a12 = a13 = a14 = a15 = 0f;
                else
                    LoadTile16(bias, out a0, out a1, out a2, out a3, out a4, out a5, out a6, out a7,
                        out a8, out a9, out a10, out a11, out a12, out a13, out a14, out a15);
            }
            else
                LoadTile16(acc, out a0, out a1, out a2, out a3, out a4, out a5, out a6, out a7,
                    out a8, out a9, out a10, out a11, out a12, out a13, out a14, out a15);
            int ci0 = k0 / taps, tap0 = k0 - ci0 * taps;
            float* wk = w;
            for (int k = k0; k < kEnd; k++, wk += OcBlock)
            {
                float v = pr[tapOffsets[tap0] + ci0];
                a0 += v * wk[0]; a1 += v * wk[1]; a2 += v * wk[2]; a3 += v * wk[3];
                a4 += v * wk[4]; a5 += v * wk[5]; a6 += v * wk[6]; a7 += v * wk[7];
                a8 += v * wk[8]; a9 += v * wk[9]; a10 += v * wk[10]; a11 += v * wk[11];
                a12 += v * wk[12]; a13 += v * wk[13]; a14 += v * wk[14]; a15 += v * wk[15];
                if (++tap0 == taps) { tap0 = 0; ci0++; }
            }
            StoreTile16(acc, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoreEpilogueScalar(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < OcBlock; c++)
            {
                float v = partial[r * OcBlock + c];
                if (biasAfter != null) v += biasAfter[c];
                if (residual != null) v += residual[r * resStride + c];
                output[r * outStride + c] = ActivateScalarValue(v, activation, alphaScalar, betaScalar);
            }
    }

    internal static void PointwiseScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int pixels, int inputChannels, int outputChannels, ReadOnlySpan<float> residual,
        NhwcActivation activation, float alpha, float beta, int threads)
    {
        if ((outputChannels & 15) != 0 || outputChannels <= 0 || inputChannels <= 0)
            throw new ArgumentException("NHWC pointwise requires output channels to be a multiple of 16.");
        if (input.Length < (long)pixels * inputChannels || output.Length < (long)pixels * outputChannels ||
            packedWeights.Length < (long)inputChannels * outputChannels ||
            (!bias.IsEmpty && bias.Length < outputChannels) ||
            (!residual.IsEmpty && residual.Length < (long)pixels * outputChannels))
            throw new ArgumentException("NHWC pointwise buffer too small.");
        if (pixels <= 0) return;
        int tiles = (pixels + TileRows - 1) / TileRows;
        int groupTiles = Math.Max(1, PointwiseGroupTiles);
        long work = (long)pixels * inputChannels * outputChannels;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, tiles) : 1;
        int kc = PointwiseKc <= 0 ? inputChannels : Math.Min(inputChannels, PointwiseKc);
        fixed (float* inPtr = input, wPtr = packedWeights, bPtr = bias, outPtr = output, rPtr = residual)
        {
            nint inA = (nint)inPtr, wA = (nint)wPtr, bA = (nint)bPtr, outA = (nint)outPtr, rA = (nint)rPtr;
            bool hasBias = !bias.IsEmpty, hasResidual = !residual.IsEmpty;
            void Worker(int worker)
            {
                int tileBegin = (int)((long)tiles * worker / workers);
                int tileEnd = (int)((long)tiles * (worker + 1) / workers);
                if (tileEnd <= tileBegin) return;
                float[] partialArray = ArrayPool<float>.Shared.Rent(groupTiles * PartialFloats);
                try
                {
                    fixed (float* partial = partialArray)
                    {
                        float* inBase = (float*)inA, wBase = (float*)wA, outBase = (float*)outA;
                        float* biasBase = hasBias ? (float*)bA : null, resBase = hasResidual ? (float*)rA : null;
                        int ocBlocks = outputChannels / OcBlock;
                        long panel = (long)inputChannels * OcBlock;
                        int firstGroup = tileBegin / groupTiles, lastGroup = (tileEnd + groupTiles - 1) / groupTiles;
                        for (int group = firstGroup; group < lastGroup; group++)
                        {
                            int tileFrom = Math.Max(tileBegin, group * groupTiles);
                            int tileTo = Math.Min(tileEnd, (group + 1) * groupTiles);
                            if (tileTo <= tileFrom) continue;
                            for (int ocBlock = 0; ocBlock < ocBlocks; ocBlock++)
                            {
                                float* w = wBase + ocBlock * panel;
                                float* b = biasBase == null ? null : biasBase + ocBlock * OcBlock;
                                int oc = ocBlock * OcBlock;
                                for (int k0 = 0; k0 < inputChannels; k0 += kc)
                                {
                                    int kEnd = Math.Min(inputChannels, k0 + kc);
                                    float* wk = w + (long)k0 * OcBlock;
                                    bool final = kEnd == inputChannels;
                                    for (int tile = tileFrom; tile < tileTo; tile++)
                                    {
                                        int pixel = tile * TileRows;
                                        int rows = Math.Min(TileRows, pixels - pixel);
                                        float* tilePartial = partial + (tile - tileFrom) * PartialFloats;
                                        GemmTile16Scalar(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd,
                                            b, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueScalar(tilePartial, rows,
                                                outBase + (long)pixel * outputChannels + oc, outputChannels,
                                                resBase == null ? null : resBase + (long)pixel * outputChannels + oc, outputChannels,
                                                activation, alpha, beta);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(partialArray);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    internal static void DenseScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
        int taps = kernelH * kernelW;
        if ((outputChannels & 15) != 0 || outputChannels <= 0 || inputChannels <= 0 || taps <= 0 || strideH <= 0 || strideW <= 0)
            throw new ArgumentException("NHWC dense convolution shape is not supported.");
        long inVolume = (long)batch * height * width * inputChannels, outVolume = (long)batch * outputHeight * outputWidth * outputChannels;
        if (input.Length < inVolume || output.Length < outVolume ||
            packedWeights.Length < (long)inputChannels * outputChannels * taps ||
            (!bias.IsEmpty && bias.Length < outputChannels) || (!residual.IsEmpty && residual.Length < outVolume))
            throw new ArgumentException("NHWC dense convolution buffer too small.");
        if (outVolume == 0) return;
        if (kernelH == 1 && kernelW == 1 && strideH == 1 && strideW == 1 && padTop == 0 && padLeft == 0 &&
            outputHeight == height && outputWidth == width)
        {
            PointwiseScalar(input, packedWeights, bias, output, batch * height * width, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
            return;
        }

        int xTiles = (outputWidth + TileRows - 1) / TileRows;
        int rowsTotal = batch * outputHeight;
        long work = outVolume * inputChannels * taps;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        int patchWidth = (TileRows - 1) * strideW + kernelW;
        int patchFloats = kernelH * patchWidth * inputChannels;
        int rowStride = width * inputChannels;

        fixed (float* inPtr = input, wPtr = packedWeights, bPtr = bias, outPtr = output, rPtr = residual)
        {
            nint inA = (nint)inPtr, wA = (nint)wPtr, bA = (nint)bPtr, outA = (nint)outPtr, rA = (nint)rPtr;
            bool hasBias = !bias.IsEmpty, hasResidual = !residual.IsEmpty;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers);
                int rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                if (rowEnd <= rowBegin) return;
                int kTotal = inputChannels * taps;
                int kc = DenseKc <= 0 ? kTotal : Math.Min(kTotal, DenseKc);
                int[] tapArray = ArrayPool<int>.Shared.Rent(taps * 2);
                float[] patchArray = ArrayPool<float>.Shared.Rent(patchFloats);
                float[] partialArray = ArrayPool<float>.Shared.Rent(xTiles * PartialFloats);
                try
                {
                    fixed (int* tapOffsets = tapArray)
                    fixed (float* patch = patchArray, partial = partialArray)
                    {
                        int* patchOffsets = tapOffsets + taps;
                        for (int ky = 0, tap = 0; ky < kernelH; ky++)
                            for (int kx = 0; kx < kernelW; kx++, tap++)
                            {
                                tapOffsets[tap] = (ky * width + kx) * inputChannels;
                                patchOffsets[tap] = (ky * patchWidth + kx) * inputChannels;
                            }
                        float* inBase = (float*)inA, wBase = (float*)wA, outBase = (float*)outA;
                        float* biasBase = hasBias ? (float*)bA : null, resBase = hasResidual ? (float*)rA : null;
                        int ocBlocks = outputChannels / OcBlock;
                        long panel = (long)kTotal * OcBlock;
                        int pixelStride = strideW * inputChannels;
                        for (int row = rowBegin; row < rowEnd; row++)
                        {
                            int b = row / outputHeight, y = row - b * outputHeight;
                            float* inBatch = inBase + (long)b * height * rowStride;
                            int iy0 = y * strideH - padTop;
                            bool rowInside = iy0 >= 0 && iy0 + kernelH <= height;
                            long outRow = ((long)b * outputHeight + y) * outputWidth;
                            int insideBegin = xTiles, insideEnd = xTiles;
                            if (rowInside)
                            {
                                insideBegin = 0;
                                while (insideBegin < xTiles && insideBegin * TileRows * strideW - padLeft < 0) insideBegin++;
                                insideEnd = insideBegin;
                                while (insideEnd < xTiles)
                                {
                                    int x0 = insideEnd * TileRows, rows = Math.Min(TileRows, outputWidth - x0);
                                    if (x0 * strideW - padLeft + (rows - 1) * strideW + kernelW > width) break;
                                    insideEnd++;
                                }
                            }
                            for (int ocBlock = 0; ocBlock < ocBlocks; ocBlock++)
                            {
                                float* w = wBase + ocBlock * panel;
                                float* bb = biasBase == null ? null : biasBase + ocBlock * OcBlock;
                                int oc = ocBlock * OcBlock;
                                for (int k0 = 0; k0 < kTotal; k0 += kc)
                                {
                                    int kEnd = Math.Min(kTotal, k0 + kc);
                                    float* wk = w + (long)k0 * OcBlock;
                                    bool final = kEnd == kTotal;
                                    for (int xTile = insideBegin; xTile < insideEnd; xTile++)
                                    {
                                        int x0 = xTile * TileRows;
                                        int rows = Math.Min(TileRows, outputWidth - x0);
                                        int ix0 = x0 * strideW - padLeft;
                                        float* tilePartial = partial + xTile * PartialFloats;
                                        float* origin = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
                                        ConvTile16Scalar(origin, rows, pixelStride, tapOffsets, taps, wk, k0, kEnd, bb, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueScalar(tilePartial, rows,
                                                outBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                                resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                                activation, alpha, beta);
                                        }
                                    }
                                }
                                for (int xTile = 0; xTile < xTiles; xTile++)
                                {
                                    if (xTile >= insideBegin && xTile < insideEnd) continue;
                                    int x0 = xTile * TileRows;
                                    int rows = Math.Min(TileRows, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                    ConvTile16Scalar(patch, rows, pixelStride, patchOffsets, taps, w, 0, kTotal, bb, partial);
                                    StoreEpilogueScalar(partial, rows,
                                        outBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                        resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                        activation, alpha, beta);
                                }
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(tapArray);
                    ArrayPool<float>.Shared.Return(patchArray);
                    ArrayPool<float>.Shared.Return(partialArray);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    private static void ConvTranspose2x2Stride2Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        NhwcActivation activation, int threads)
    {
        int outputHeight = height * 2, outputWidth = width * 2;
        long outVolume = (long)batch * outputHeight * outputWidth * outputChannels;
        if (input.Length < (long)batch * height * width * inputChannels || output.Length < outVolume ||
            (!bias.IsEmpty && bias.Length < outputChannels))
            throw new ArgumentException("NHWC transposed convolution buffer too small.");
        if (outputChannels == 1)
        {
            if (packedWeights.Length < 4L * inputChannels) throw new ArgumentException("NHWC transposed convolution weights too small.");
            ConvTranspose2x2Stride2SingleOutputScalar(input, packedWeights, bias, output, batch, inputChannels, height, width, activation, threads);
            return;
        }
        if ((outputChannels & 15) != 0 || packedWeights.Length < 4L * inputChannels * outputChannels)
            throw new ArgumentException("NHWC transposed convolution requires output channels to be a multiple of 16.");
        int rowsTotal = batch * height;
        long work = outVolume * inputChannels;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        int xTiles = (width + TileRows - 1) / TileRows;
        fixed (float* inPtr = input, wPtr = packedWeights, bPtr = bias, outPtr = output)
        {
            nint inA = (nint)inPtr, wA = (nint)wPtr, bA = (nint)bPtr, outA = (nint)outPtr;
            bool hasBias = !bias.IsEmpty;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers);
                int rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                float* inBase = (float*)inA, wBase = (float*)wA, outBase = (float*)outA;
                float* biasBase = hasBias ? (float*)bA : null;
                int ocBlocks = outputChannels / OcBlock;
                long panel = (long)inputChannels * OcBlock, tapPanel = panel * ocBlocks;
                float* partial = stackalloc float[PartialFloats];
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / height, y = row - b * height;
                    float* inRow = inBase + ((long)b * height + y) * width * inputChannels;
                    for (int tap = 0; tap < 4; tap++)
                    {
                        int ky = tap >> 1, kx = tap & 1;
                        float* outRow = outBase + (((long)b * outputHeight + 2 * y + ky) * outputWidth + kx) * outputChannels;
                        for (int ocBlock = 0; ocBlock < ocBlocks; ocBlock++)
                        {
                            float* w = wBase + tap * tapPanel + ocBlock * panel;
                            float* bb = biasBase == null ? null : biasBase + ocBlock * OcBlock;
                            for (int xTile = 0; xTile < xTiles; xTile++)
                            {
                                int x0 = xTile * TileRows;
                                int rows = Math.Min(TileRows, width - x0);
                                GemmTile16Scalar(inRow + (long)x0 * inputChannels, rows, inputChannels, w, 0, inputChannels,
                                    null, partial);
                                StoreEpilogueScalar(partial, rows,
                                    outRow + (long)(2 * x0) * outputChannels + ocBlock * OcBlock, 2 * outputChannels,
                                    null, 0, activation, 0f, 0f, bb);
                            }
                        }
                    }
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTranspose2x2Stride2SingleOutputScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height, int width,
        NhwcActivation activation, int threads)
    {
        int outputWidth = width * 2, rowsTotal = batch * height;
        int workers = threads > 1 && (long)rowsTotal * width * inputChannels >= 1_000_000 ? Math.Min(threads, rowsTotal) : 1;
        float biasValue = bias.IsEmpty ? 0f : bias[0];
        fixed (float* inPtr = input, wPtr = packedWeights, outPtr = output)
        {
            nint inA = (nint)inPtr, wA = (nint)wPtr, outA = (nint)outPtr;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers);
                int rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                float* inBase = (float*)inA, w = (float*)wA, outBase = (float*)outA;
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / height, y = row - b * height;
                    float* inRow = inBase + ((long)b * height + y) * width * inputChannels;
                    float* out0 = outBase + ((long)b * height * 2 + 2 * y) * outputWidth;
                    float* out1 = out0 + outputWidth;
                    for (int x = 0; x < width; x++)
                    {
                        float* px = inRow + (long)x * inputChannels;
                        float r0 = biasValue, r1 = biasValue, r2 = biasValue, r3 = biasValue;
                        for (int c = 0; c < inputChannels; c++)
                        {
                            float v = px[c];
                            r0 += v * w[c]; r1 += v * w[inputChannels + c];
                            r2 += v * w[2 * inputChannels + c]; r3 += v * w[3 * inputChannels + c];
                        }
                        out0[2 * x] = activation == NhwcActivation.Relu ? MathF.Max(r0, 0f) : r0;
                        out0[2 * x + 1] = activation == NhwcActivation.Relu ? MathF.Max(r1, 0f) : r1;
                        out1[2 * x] = activation == NhwcActivation.Relu ? MathF.Max(r2, 0f) : r2;
                        out1[2 * x + 1] = activation == NhwcActivation.Relu ? MathF.Max(r3, 0f) : r3;
                    }
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }
}
