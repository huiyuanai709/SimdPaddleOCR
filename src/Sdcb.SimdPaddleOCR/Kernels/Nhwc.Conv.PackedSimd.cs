#if !NETSTANDARD2_0
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// WebAssembly SIMD128 (PackedSimd) channels-last convolution kernels. WASM
// SIMD has no FMA (without relaxed SIMD) and browsers map v128 values onto at
// most 15 allocatable XMM registers on x64, so the tile is 4 pixels x 8 output
// channels (8 accumulators), run twice per 16-channel packed panel. 5- and
// 6-row tiles spill in V8 and measured ~1.4x slower.
// Each k step broadcasts one input float per pixel straight from memory
// (v128.load32_splat) and accumulates with a separate multiply then add, in
// ascending k order: the per-lane arithmetic is identical to the Vector<float>
// kernels. Like the other ISAs the tile kernels always spill their
// accumulators to the partial buffer and never call out; the fused epilogue
// runs separately in StoreEpiloguePackedSimd.
internal static unsafe partial class Nhwc
{
    private const int PackedTileRows = 4;
    private const int PackedPartialFloats = PackedTileRows * OcBlock;

    // ---------------------------------------------------------------- epilogue

    // PseudoMin/PseudoMax (f32x4.pmin/pmax) lower to a single minps/maxps,
    // whereas f32x4.min/max need a NaN/-0 fix-up sequence. For finite inputs
    // the results equal Vector.Min/Max (the zero operand is the "a" side so
    // -0 still maps to +0).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ActivatePackedSimd(Vector128<float> v, NhwcActivation activation,
        Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return PackedSimd.PseudoMax(Vector128<float>.Zero, v);
            case NhwcActivation.HardSwish:
            {
                Vector128<float> gate = PackedSimd.Add(PackedSimd.Multiply(v, alpha), beta);
                gate = PackedSimd.PseudoMax(Vector128<float>.Zero, PackedSimd.PseudoMin(one, gate));
                return PackedSimd.Multiply(v, gate);
            }
            case NhwcActivation.Gelu:
                return SimdKernels.GeluVectorPackedSimd(v);
            default:
                return v;
        }
    }

    /// <summary>
    /// Fused epilogue for one finished tile: reloads the partial rows, adds the
    /// deferred bias (transposed convolution) and residual, applies the
    /// activation and stores to the output.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplCompat.AggressiveOptimization)]
    private static void StoreEpiloguePackedSimd(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        Vector128<float> alpha = PackedSimd.Splat(alphaScalar), beta = PackedSimd.Splat(betaScalar), one = Vector128.Create(1f);
        for (int r = 0; r < rows; r++)
        {
            Vector128<float> a0 = PackedSimd.LoadVector128(partial + r * OcBlock);
            Vector128<float> a1 = PackedSimd.LoadVector128(partial + r * OcBlock + 4);
            Vector128<float> a2 = PackedSimd.LoadVector128(partial + r * OcBlock + 8);
            Vector128<float> a3 = PackedSimd.LoadVector128(partial + r * OcBlock + 12);
            if (biasAfter != null)
            {
                a0 = PackedSimd.Add(a0, PackedSimd.LoadVector128(biasAfter));
                a1 = PackedSimd.Add(a1, PackedSimd.LoadVector128(biasAfter + 4));
                a2 = PackedSimd.Add(a2, PackedSimd.LoadVector128(biasAfter + 8));
                a3 = PackedSimd.Add(a3, PackedSimd.LoadVector128(biasAfter + 12));
            }
            if (residual != null)
            {
                float* res = residual + r * resStride;
                a0 = PackedSimd.Add(a0, PackedSimd.LoadVector128(res));
                a1 = PackedSimd.Add(a1, PackedSimd.LoadVector128(res + 4));
                a2 = PackedSimd.Add(a2, PackedSimd.LoadVector128(res + 8));
                a3 = PackedSimd.Add(a3, PackedSimd.LoadVector128(res + 12));
            }
            if (activation != NhwcActivation.None)
            {
                a0 = ActivatePackedSimd(a0, activation, alpha, beta, one);
                a1 = ActivatePackedSimd(a1, activation, alpha, beta, one);
                a2 = ActivatePackedSimd(a2, activation, alpha, beta, one);
                a3 = ActivatePackedSimd(a3, activation, alpha, beta, one);
            }
            float* dst = output + r * outStride;
            PackedSimd.Store(dst, a0);
            PackedSimd.Store(dst + 4, a1);
            PackedSimd.Store(dst + 8, a2);
            PackedSimd.Store(dst + 12, a3);
        }
    }

    // ------------------------------------------------------------ micro-kernels

    /// <summary>
    /// Pointwise tile: <paramref name="rows"/> (1..4) consecutive pixels with
    /// stride <paramref name="inStride"/> floats, input channels
    /// [<paramref name="k0"/>, <paramref name="kEnd"/>) against the weight
    /// panel <paramref name="w"/> ([k][16], already offset to k0). Accumulators
    /// always land in <paramref name="partial"/>. 8 accumulators + 2 weights +
    /// the splat input + the product temporary = 12 live v128 values; 5 or 6
    /// rows spill in V8's x64 register allocator.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void GemmTilePackedSimd(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        for (int oc0 = 0; oc0 < OcBlock; oc0 += 8)
        {
            Vector128<float> a0, a1, a2, a3, b0, b1, b2, b3;
            float* acc = partial + oc0;
            if (k0 == 0)
            {
                Vector128<float> biasLow = bias == null ? Vector128<float>.Zero : PackedSimd.LoadVector128(bias + oc0);
                Vector128<float> biasHigh = bias == null ? Vector128<float>.Zero : PackedSimd.LoadVector128(bias + oc0 + 4);
                a0 = a1 = a2 = a3 = biasLow;
                b0 = b1 = b2 = b3 = biasHigh;
            }
            else
            {
                a0 = PackedSimd.LoadVector128(acc); b0 = PackedSimd.LoadVector128(acc + 4);
                a1 = PackedSimd.LoadVector128(acc + 16); b1 = PackedSimd.LoadVector128(acc + 20);
                a2 = PackedSimd.LoadVector128(acc + 32); b2 = PackedSimd.LoadVector128(acc + 36);
                a3 = PackedSimd.LoadVector128(acc + 48); b3 = PackedSimd.LoadVector128(acc + 52);
            }
            float* wk = w + oc0;
            for (int k = k0; k < kEnd; k++, wk += 16)
            {
                Vector128<float> w0 = PackedSimd.LoadVector128(wk);
                Vector128<float> w1 = PackedSimd.LoadVector128(wk + 4);
                Vector128<float> v = PackedSimd.LoadScalarAndSplatVector128(p0 + k);
                a0 = PackedSimd.Add(a0, PackedSimd.Multiply(v, w0)); b0 = PackedSimd.Add(b0, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p1 + k);
                a1 = PackedSimd.Add(a1, PackedSimd.Multiply(v, w0)); b1 = PackedSimd.Add(b1, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p2 + k);
                a2 = PackedSimd.Add(a2, PackedSimd.Multiply(v, w0)); b2 = PackedSimd.Add(b2, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p3 + k);
                a3 = PackedSimd.Add(a3, PackedSimd.Multiply(v, w0)); b3 = PackedSimd.Add(b3, PackedSimd.Multiply(v, w1));
            }
            PackedSimd.Store(acc, a0); PackedSimd.Store(acc + 4, b0);
            PackedSimd.Store(acc + 16, a1); PackedSimd.Store(acc + 20, b1);
            PackedSimd.Store(acc + 32, a2); PackedSimd.Store(acc + 36, b2);
            PackedSimd.Store(acc + 48, a3); PackedSimd.Store(acc + 52, b3);
        }
    }

    /// <summary>
    /// Dense KxK tile: up to 4 output pixels whose receptive-field origins are
    /// <paramref name="input"/> plus row multiples of <paramref name="pixelStride"/>.
    /// Weights are [ic][tap][16] (already offset to <paramref name="k0"/>); the
    /// flattened reduction index k = ic * taps + tap runs input channels then
    /// taps, matching the reference summation order. <paramref name="kOffsets"/>
    /// holds the input float offset of every k (relative to the origin), so the
    /// k loop carries no tap/channel bookkeeping. Same 4 x 8 register block as
    /// <see cref="GemmTilePackedSimd"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTilePackedSimd(float* input, int rows, int pixelStride,
        int* kOffsets, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? pixelStride : 0);
        float* p2 = p1 + (rows > 2 ? pixelStride : 0);
        float* p3 = p2 + (rows > 3 ? pixelStride : 0);
        for (int oc0 = 0; oc0 < OcBlock; oc0 += 8)
        {
            Vector128<float> a0, a1, a2, a3, b0, b1, b2, b3;
            float* acc = partial + oc0;
            if (k0 == 0)
            {
                Vector128<float> biasLow = bias == null ? Vector128<float>.Zero : PackedSimd.LoadVector128(bias + oc0);
                Vector128<float> biasHigh = bias == null ? Vector128<float>.Zero : PackedSimd.LoadVector128(bias + oc0 + 4);
                a0 = a1 = a2 = a3 = biasLow;
                b0 = b1 = b2 = b3 = biasHigh;
            }
            else
            {
                a0 = PackedSimd.LoadVector128(acc); b0 = PackedSimd.LoadVector128(acc + 4);
                a1 = PackedSimd.LoadVector128(acc + 16); b1 = PackedSimd.LoadVector128(acc + 20);
                a2 = PackedSimd.LoadVector128(acc + 32); b2 = PackedSimd.LoadVector128(acc + 36);
                a3 = PackedSimd.LoadVector128(acc + 48); b3 = PackedSimd.LoadVector128(acc + 52);
            }
            float* wk = w + oc0;
            for (int k = k0; k < kEnd; k++, wk += 16)
            {
                int offset = kOffsets[k];
                Vector128<float> w0 = PackedSimd.LoadVector128(wk);
                Vector128<float> w1 = PackedSimd.LoadVector128(wk + 4);
                Vector128<float> v = PackedSimd.LoadScalarAndSplatVector128(p0 + offset);
                a0 = PackedSimd.Add(a0, PackedSimd.Multiply(v, w0)); b0 = PackedSimd.Add(b0, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p1 + offset);
                a1 = PackedSimd.Add(a1, PackedSimd.Multiply(v, w0)); b1 = PackedSimd.Add(b1, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p2 + offset);
                a2 = PackedSimd.Add(a2, PackedSimd.Multiply(v, w0)); b2 = PackedSimd.Add(b2, PackedSimd.Multiply(v, w1));
                v = PackedSimd.LoadScalarAndSplatVector128(p3 + offset);
                a3 = PackedSimd.Add(a3, PackedSimd.Multiply(v, w0)); b3 = PackedSimd.Add(b3, PackedSimd.Multiply(v, w1));
            }
            PackedSimd.Store(acc, a0); PackedSimd.Store(acc + 4, b0);
            PackedSimd.Store(acc + 16, a1); PackedSimd.Store(acc + 20, b1);
            PackedSimd.Store(acc + 32, a2); PackedSimd.Store(acc + 36, b2);
            PackedSimd.Store(acc + 48, a3); PackedSimd.Store(acc + 52, b3);
        }
    }

    // --------------------------------------------------------------- pointwise

    /// <summary>PackedSimd driver of <see cref="Pointwise"/>; same loop nest as the other ISA drivers.</summary>
    internal static void PointwisePackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
        int tiles = (pixels + PackedTileRows - 1) / PackedTileRows;
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
                float[] partialArray = ArrayPool<float>.Shared.Rent(groupTiles * PackedPartialFloats);
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
                                        int pixel = tile * PackedTileRows;
                                        int rows = Math.Min(PackedTileRows, pixels - pixel);
                                        float* tilePartial = partial + (tile - tileFrom) * PackedPartialFloats;
                                        GemmTilePackedSimd(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd,
                                            b, tilePartial);
                                        if (final)
                                        {
                                            StoreEpiloguePackedSimd(tilePartial, rows,
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

    // ------------------------------------------------------------------- dense

    /// <summary>PackedSimd driver of <see cref="Dense"/>; same loop nest as the other ISA drivers.</summary>
    internal static void DensePackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            PointwisePackedSimd(input, packedWeights, bias, output, batch * height * width, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
            return;
        }

        int xTiles = (outputWidth + PackedTileRows - 1) / PackedTileRows;
        int rowsTotal = batch * outputHeight;
        long work = outVolume * inputChannels * taps;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        // Border tiles gather their receptive field into a zero-padded patch:
        // kernelH rows of (3*strideW + kernelW) pixels.
        int patchWidth = (PackedTileRows - 1) * strideW + kernelW;
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
                int[] offsetArray = ArrayPool<int>.Shared.Rent(kTotal * 2);
                float[] patchArray = ArrayPool<float>.Shared.Rent(patchFloats);
                float[] partialArray = ArrayPool<float>.Shared.Rent(xTiles * PackedPartialFloats);
                try
                {
                    fixed (int* kOffsets = offsetArray)
                    fixed (float* patch = patchArray, partial = partialArray)
                    {
                        int* patchKOffsets = kOffsets + kTotal;
                        for (int ci = 0, k = 0; ci < inputChannels; ci++)
                            for (int ky = 0; ky < kernelH; ky++)
                                for (int kx = 0; kx < kernelW; kx++, k++)
                                {
                                    kOffsets[k] = (ky * width + kx) * inputChannels + ci;
                                    patchKOffsets[k] = (ky * patchWidth + kx) * inputChannels + ci;
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
                            // Interior tiles: every tap inside the input. Range
                            // [insideBegin, insideEnd) of x-tiles.
                            int insideBegin = xTiles, insideEnd = xTiles;
                            if (rowInside)
                            {
                                insideBegin = 0;
                                while (insideBegin < xTiles && insideBegin * PackedTileRows * strideW - padLeft < 0) insideBegin++;
                                insideEnd = insideBegin;
                                while (insideEnd < xTiles)
                                {
                                    int x0 = insideEnd * PackedTileRows, rows = Math.Min(PackedTileRows, outputWidth - x0);
                                    if (x0 * strideW - padLeft + (rows - 1) * strideW + kernelW > width) break;
                                    insideEnd++;
                                }
                            }
                            for (int ocBlock = 0; ocBlock < ocBlocks; ocBlock++)
                            {
                                float* w = wBase + ocBlock * panel;
                                float* bb = biasBase == null ? null : biasBase + ocBlock * OcBlock;
                                int oc = ocBlock * OcBlock;
                                // Interior: k-blocked so one 16 x kc weight panel stays in L1 across the row's tiles.
                                for (int k0 = 0; k0 < kTotal; k0 += kc)
                                {
                                    int kEnd = Math.Min(kTotal, k0 + kc);
                                    float* wk = w + (long)k0 * OcBlock;
                                    bool final = kEnd == kTotal;
                                    for (int xTile = insideBegin; xTile < insideEnd; xTile++)
                                    {
                                        int x0 = xTile * PackedTileRows;
                                        int rows = Math.Min(PackedTileRows, outputWidth - x0);
                                        int ix0 = x0 * strideW - padLeft;
                                        float* tilePartial = partial + xTile * PackedPartialFloats;
                                        float* origin = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
                                        ConvTilePackedSimd(origin, rows, pixelStride, kOffsets, wk, k0, kEnd, bb, tilePartial);
                                        if (final)
                                        {
                                            StoreEpiloguePackedSimd(tilePartial, rows,
                                                outBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                                resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc, outputChannels,
                                                activation, alpha, beta);
                                        }
                                    }
                                }
                                // Border tiles: gathered zero-padded patch, full reduction in one pass.
                                for (int xTile = 0; xTile < xTiles; xTile++)
                                {
                                    if (xTile >= insideBegin && xTile < insideEnd) continue;
                                    int x0 = xTile * PackedTileRows;
                                    int rows = Math.Min(PackedTileRows, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                    ConvTilePackedSimd(patch, rows, pixelStride, patchKOffsets, w, 0, kTotal, bb, partial);
                                    StoreEpiloguePackedSimd(partial, rows,
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
                    ArrayPool<int>.Shared.Return(offsetArray);
                    ArrayPool<float>.Shared.Return(patchArray);
                    ArrayPool<float>.Shared.Return(partialArray);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // ----------------------------------------------------------- conv transpose

    /// <summary>PackedSimd driver of <see cref="ConvTranspose2x2Stride2"/>; same loop nest as the other ISA drivers.</summary>
    internal static void ConvTranspose2x2Stride2PackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            ConvTranspose2x2Stride2SingleOutputPackedSimd(input, packedWeights, bias, output, batch, inputChannels, height, width, activation, threads);
            return;
        }
        if ((outputChannels & 15) != 0 || packedWeights.Length < 4L * inputChannels * outputChannels)
            throw new ArgumentException("NHWC transposed convolution requires output channels to be a multiple of 16.");
        int rowsTotal = batch * height;
        long work = outVolume * inputChannels;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        int xTiles = (width + PackedTileRows - 1) / PackedTileRows;
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
                float* partial = stackalloc float[PackedPartialFloats];
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
                                int x0 = xTile * PackedTileRows;
                                int rows = Math.Min(PackedTileRows, width - x0);
                                GemmTilePackedSimd(inRow + (long)x0 * inputChannels, rows, inputChannels, w, 0, inputChannels,
                                    null, partial);
                                StoreEpiloguePackedSimd(partial, rows,
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

    // Single output channel (detector probability map). The four taps of one
    // input pixel run in the four lanes of one accumulator: weights are
    // re-packed [ic][tap] so each input channel is one splat + one weight
    // vector. Every lane is bias-initialised and accumulates input channels in
    // ascending order with a separate multiply then add, i.e. exactly the
    // scalar order of the Vector kernel.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTranspose2x2Stride2SingleOutputPackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height, int width,
        NhwcActivation activation, int threads)
    {
        int outputWidth = width * 2, rowsTotal = batch * height;
        int workers = threads > 1 && (long)rowsTotal * width * inputChannels >= 1_000_000 ? Math.Min(threads, rowsTotal) : 1;
        float biasValue = bias.IsEmpty ? 0f : bias[0];
        float[] tapMajor = ArrayPool<float>.Shared.Rent(4 * inputChannels);
        try
        {
            for (int c = 0; c < inputChannels; c++)
                for (int tap = 0; tap < 4; tap++)
                    tapMajor[c * 4 + tap] = packedWeights[tap * inputChannels + c];
            fixed (float* inPtr = input, wPtr = tapMajor, outPtr = output)
            {
                nint inA = (nint)inPtr, wA = (nint)wPtr, outA = (nint)outPtr;
                bool relu = activation == NhwcActivation.Relu;
                void Worker(int worker)
                {
                    int rowBegin = (int)((long)rowsTotal * worker / workers);
                    int rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                    float* inBase = (float*)inA, w = (float*)wA, outBase = (float*)outA;
                    Vector128<float> init = PackedSimd.Splat(biasValue);
                    for (int row = rowBegin; row < rowEnd; row++)
                    {
                        int b = row / height, y = row - b * height;
                        float* inRow = inBase + ((long)b * height + y) * width * inputChannels;
                        float* out0 = outBase + ((long)b * height * 2 + 2 * y) * outputWidth;
                        float* out1 = out0 + outputWidth;
                        for (int x = 0; x < width; x++)
                        {
                            float* px = inRow + (long)x * inputChannels;
                            Vector128<float> acc = init;
                            for (int c = 0; c < inputChannels; c++)
                                acc = PackedSimd.Add(acc, PackedSimd.Multiply(
                                    PackedSimd.LoadScalarAndSplatVector128(px + c), PackedSimd.LoadVector128(w + c * 4)));
                            if (relu) acc = PackedSimd.PseudoMax(Vector128<float>.Zero, acc);
                            PackedSimd.StoreSelectedScalar((long*)(out0 + 2 * x), acc.AsInt64(), 0);
                            PackedSimd.StoreSelectedScalar((long*)(out1 + 2 * x), acc.AsInt64(), 1);
                        }
                    }
                }
                if (workers > 1) Parallel.For(0, workers, Worker);
                else Worker(0);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tapMajor);
        }
    }
}
#endif
