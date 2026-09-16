using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// AVX-512 channels-last convolution kernels. Same register-blocking scheme as
// the AVX2 micro-kernel (a run of pixels x 16 output channels, bias-initialised
// FMA chain over the flattened k = ic * taps + tap index in ascending order),
// widened to one 512-bit accumulator per pixel row so every k step costs a
// single 64 B weight load. The wider tile only changes how many rows are
// processed per pass; each output pixel keeps the exact AVX2 reduction order.
// The tile kernels always spill their accumulators to the partial buffer and
// never call out: any call in the method (or an inlined branchy epilogue such
// as the Gelu polynomial) makes RyuJIT park the accumulators on the stack for
// the whole k loop. The fused epilogue runs separately in StoreEpilogue512.
internal static unsafe partial class Nhwc
{
    private const int TileRows512 = 8;
    /// <summary>Partial-sum floats per AVX-512 tile (8 x 16).</summary>
    private const int PartialFloats512 = TileRows512 * OcBlock;

    /// <summary>
    /// NHWC AVX-512 kernels when the ISA is present. Layout on/off is
    /// <c>PPOCR_NHWC</c> in <see cref="OnnxSharp.LayoutPlanner"/>; process-wide
    /// ISA is <c>DOTNET_EnableAVX512</c> via <see cref="Avx512F.IsSupported"/>.
    /// </summary>
    private static readonly bool UseAvx512 = Avx512F.IsSupported;

    // ---------------------------------------------------------------- epilogue

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Activate512(Vector512<float> v, NhwcActivation activation,
        Vector512<float> alpha, Vector512<float> beta, Vector512<float> one)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return Avx512F.Max(v, Vector512<float>.Zero);
            case NhwcActivation.HardSwish:
            {
                Vector512<float> gate = Avx512F.Add(Avx512F.Multiply(v, alpha), beta);
                gate = Avx512F.Max(Vector512<float>.Zero, Avx512F.Min(one, gate));
                return Avx512F.Multiply(v, gate);
            }
            case NhwcActivation.Gelu:
                return SimdKernels.GeluVector512(v);
            default:
                return v;
        }
    }

    /// <summary>
    /// Fused epilogue for one finished tile: reloads the 8 partial rows, adds the
    /// residual, applies the activation and stores to the output, in the same
    /// order as the AVX2 StoreRow. Runs once per tile, so it stays out of the
    /// register-critical k loops (a plain call, no inlining into the kernels).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoreEpilogue512(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar)
    {
        Vector512<float> alpha = Vector512.Create(alphaScalar), beta = Vector512.Create(betaScalar), one = Vector512.Create(1f);
        for (int r = 0; r < rows; r++)
        {
            Vector512<float> a = Avx512F.LoadVector512(partial + r * OcBlock);
            if (residual != null)
            {
                a = Avx512F.Add(a, Avx512F.LoadVector512(residual + r * resStride));
            }
            if (activation != NhwcActivation.None)
            {
                a = Activate512(a, activation, alpha, beta, one);
            }
            Avx512F.Store(output + r * outStride, a);
        }
    }

    // ------------------------------------------------------------ micro-kernels

    /// <summary>
    /// Pointwise tile: <paramref name="rows"/> (1..8) consecutive pixels with
    /// stride <paramref name="inStride"/> floats, input channels
    /// [<paramref name="k0"/>, <paramref name="kEnd"/>) against the weight
    /// panel <paramref name="w"/> ([k][16], already offset to k0). Accumulators
    /// always land in <paramref name="partial"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void GemmTile512(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        float* p4 = p3 + (rows > 4 ? inStride : 0);
        float* p5 = p4 + (rows > 5 ? inStride : 0);
        float* p6 = p5 + (rows > 6 ? inStride : 0);
        float* p7 = p6 + (rows > 7 ? inStride : 0);
        Vector512<float> a0, a1, a2, a3, a4, a5, a6, a7;
        if (k0 == 0)
        {
            Vector512<float> biasV = bias == null ? Vector512<float>.Zero : Avx512F.LoadVector512(bias);
            a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = biasV;
        }
        else
        {
            a0 = Avx512F.LoadVector512(partial);
            a1 = Avx512F.LoadVector512(partial + 16);
            a2 = Avx512F.LoadVector512(partial + 32);
            a3 = Avx512F.LoadVector512(partial + 48);
            a4 = Avx512F.LoadVector512(partial + 64);
            a5 = Avx512F.LoadVector512(partial + 80);
            a6 = Avx512F.LoadVector512(partial + 96);
            a7 = Avx512F.LoadVector512(partial + 112);
        }
        for (int k = k0; k < kEnd; k++, w += 16)
        {
            Vector512<float> wv = Avx512F.LoadVector512(w);
            a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(p0[k]), wv, a0);
            a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(p1[k]), wv, a1);
            a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(p2[k]), wv, a2);
            a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(p3[k]), wv, a3);
            a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(p4[k]), wv, a4);
            a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(p5[k]), wv, a5);
            a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(p6[k]), wv, a6);
            a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(p7[k]), wv, a7);
        }
            Avx512F.Store(partial, a0);
            Avx512F.Store(partial + 16, a1);
            Avx512F.Store(partial + 32, a2);
            Avx512F.Store(partial + 48, a3);
            Avx512F.Store(partial + 64, a4);
            Avx512F.Store(partial + 80, a5);
            Avx512F.Store(partial + 96, a6);
            Avx512F.Store(partial + 112, a7);
    }

    /// <summary>
    /// Dense KxK tile: up to 8 output pixels whose receptive-field origins are
    /// <paramref name="input"/> plus row multiples of <paramref name="pixelStride"/>;
    /// <paramref name="tapOffsets"/> gives the float offset of every tap relative
    /// to the origin. Weights are [ic][tap][16] (already offset to
    /// <paramref name="k0"/>); the flattened reduction index k = ic * taps + tap
    /// runs input channels then taps, matching the reference summation order.
    /// Accumulators always land in <paramref name="partial"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTile512(float* input, int rows, int pixelStride,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? pixelStride : 0);
        float* p2 = p1 + (rows > 2 ? pixelStride : 0);
        float* p3 = p2 + (rows > 3 ? pixelStride : 0);
        float* p4 = p3 + (rows > 4 ? pixelStride : 0);
        float* p5 = p4 + (rows > 5 ? pixelStride : 0);
        float* p6 = p5 + (rows > 6 ? pixelStride : 0);
        float* p7 = p6 + (rows > 7 ? pixelStride : 0);
        Vector512<float> a0, a1, a2, a3, a4, a5, a6, a7;
        if (k0 == 0)
        {
            Vector512<float> biasV = bias == null ? Vector512<float>.Zero : Avx512F.LoadVector512(bias);
            a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = biasV;
        }
        else
        {
            a0 = Avx512F.LoadVector512(partial);
            a1 = Avx512F.LoadVector512(partial + 16);
            a2 = Avx512F.LoadVector512(partial + 32);
            a3 = Avx512F.LoadVector512(partial + 48);
            a4 = Avx512F.LoadVector512(partial + 64);
            a5 = Avx512F.LoadVector512(partial + 80);
            a6 = Avx512F.LoadVector512(partial + 96);
            a7 = Avx512F.LoadVector512(partial + 112);
        }
        int ci = k0 / taps, tap = k0 - ci * taps;
        for (int k = k0; k < kEnd; k++, w += 16)
        {
            int offset = tapOffsets[tap] + ci;
            Vector512<float> wv = Avx512F.LoadVector512(w);
            a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(p0[offset]), wv, a0);
            a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(p1[offset]), wv, a1);
            a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(p2[offset]), wv, a2);
            a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(p3[offset]), wv, a3);
            a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(p4[offset]), wv, a4);
            a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(p5[offset]), wv, a5);
            a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(p6[offset]), wv, a6);
            a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(p7[offset]), wv, a7);
            if (++tap == taps) { tap = 0; ci++; }
        }
            Avx512F.Store(partial, a0);
            Avx512F.Store(partial + 16, a1);
            Avx512F.Store(partial + 32, a2);
            Avx512F.Store(partial + 48, a3);
            Avx512F.Store(partial + 64, a4);
            Avx512F.Store(partial + 80, a5);
            Avx512F.Store(partial + 96, a6);
            Avx512F.Store(partial + 112, a7);
    }

    // --------------------------------------------------------------- pointwise

    /// <summary>AVX-512 driver of <see cref="Pointwise"/>; same loop nest with the wider tile.</summary>
    private static void Pointwise512(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int pixels, int inputChannels, int outputChannels, ReadOnlySpan<float> residual,
        NhwcActivation activation, float alpha, float beta, int threads)
    {
        int tiles = (pixels + TileRows512 - 1) / TileRows512;
        int groupTiles = Math.Max(1, PointwiseGroupTiles);
        long work = (long)pixels * inputChannels * outputChannels;
        // Shard whole tiles rather than tile groups: a group count that does
        // not divide the worker count hands one worker an extra group, and
        // that worker sets the makespan.  512x6x72->1024 is nine groups over
        // eight workers -- a 2:1 split -- and measured 530 GFLOPS at eight
        // workers against 804 at nine.  Tiles are independent, so balancing
        // them changes no arithmetic; each worker still walks its tiles in
        // group-sized steps so weight-panel reuse is unchanged.
        // AVX-512 path uses TileRows512=8, so the tile count differs from
        // AVX2; the load-imbalance defect is the same class.
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
                float[] partialArray = ArrayPool<float>.Shared.Rent(groupTiles * PartialFloats512);
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
                                        int pixel = tile * TileRows512;
                                        int rows = Math.Min(TileRows512, pixels - pixel);
                                        float* tilePartial = partial + (tile - tileFrom) * PartialFloats512;
                                        GemmTile512(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd,
                                            b, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogue512(tilePartial, rows,
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

    /// <summary>AVX-512 driver of <see cref="Dense"/>; same loop nest with the wider tile.</summary>
    private static void Dense512(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
        int taps = kernelH * kernelW;
        int xTiles = (outputWidth + TileRows512 - 1) / TileRows512;
        int rowsTotal = batch * outputHeight;
        long work = (long)batch * outputHeight * outputWidth * outputChannels * inputChannels * taps;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        // Border tiles gather their receptive field into a zero-padded patch:
        // kernelH rows of ((TileRows512-1)*strideW + kernelW) pixels.
        int patchWidth = (TileRows512 - 1) * strideW + kernelW;
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
                float[] partialArray = ArrayPool<float>.Shared.Rent(xTiles * PartialFloats512);
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
                            // Interior tiles: every tap inside the input. Range
                            // [insideBegin, insideEnd) of x-tiles.
                            int insideBegin = xTiles, insideEnd = xTiles;
                            if (rowInside)
                            {
                                insideBegin = 0;
                                while (insideBegin < xTiles && insideBegin * TileRows512 * strideW - padLeft < 0) insideBegin++;
                                insideEnd = insideBegin;
                                while (insideEnd < xTiles)
                                {
                                    int x0 = insideEnd * TileRows512, rows = Math.Min(TileRows512, outputWidth - x0);
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
                                        int x0 = xTile * TileRows512;
                                        int rows = Math.Min(TileRows512, outputWidth - x0);
                                        int ix0 = x0 * strideW - padLeft;
                                        float* tilePartial = partial + xTile * PartialFloats512;
                                        float* origin = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
                                        ConvTile512(origin, rows, pixelStride, tapOffsets, taps, wk, k0, kEnd, bb, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogue512(tilePartial, rows,
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
                                    int x0 = xTile * TileRows512;
                                    int rows = Math.Min(TileRows512, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                    ConvTile512(patch, rows, pixelStride, patchOffsets, taps, w, 0, kTotal, bb, partial);
                                    StoreEpilogue512(partial, rows,
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
}
