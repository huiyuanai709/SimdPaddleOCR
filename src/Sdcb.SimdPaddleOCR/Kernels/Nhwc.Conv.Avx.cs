using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// AVX2+FMA channels-last convolution kernels. One register-blocked
// micro-kernel (6 pixels x 16 output channels, bias-initialised FMA chain over
// input channels then taps) serves pointwise, dense KxK and 2x2/stride-2
// transposed convolutions; the epilogue fuses residual add and activation.
internal static unsafe partial class Nhwc
{
    // ---------------------------------------------------------------- epilogue

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Activate(Vector256<float> v, NhwcActivation activation,
        Vector256<float> alpha, Vector256<float> beta, Vector256<float> one)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return Avx.Max(v, Vector256<float>.Zero);
            case NhwcActivation.HardSwish:
            {
                Vector256<float> gate = Avx.Add(Avx.Multiply(v, alpha), beta);
                gate = Avx.Max(Vector256<float>.Zero, Avx.Min(one, gate));
                return Avx.Multiply(v, gate);
            }
            case NhwcActivation.Gelu:
                return SimdKernels.GeluVector(v);
            default:
                return v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreRow(float* output, Vector256<float> a, Vector256<float> b, float* residual,
        NhwcActivation activation, Vector256<float> alpha, Vector256<float> beta, Vector256<float> one,
        float* biasAfter = null)
    {
        if (biasAfter != null)
        {
            a = Avx.Add(a, Avx.LoadVector256(biasAfter));
            b = Avx.Add(b, Avx.LoadVector256(biasAfter + 8));
        }
        if (residual != null)
        {
            a = Avx.Add(a, Avx.LoadVector256(residual));
            b = Avx.Add(b, Avx.LoadVector256(residual + 8));
        }
        if (activation != NhwcActivation.None)
        {
            a = Activate(a, activation, alpha, beta, one);
            b = Activate(b, activation, alpha, beta, one);
        }
        Avx.Store(output, a);
        Avx.Store(output + 8, b);
    }

    /// <summary>
    /// Fused epilogue for one finished AVX2 tile. Kept out of <see cref="GemmTile16"/>
    /// and <see cref="ConvTile16"/>: those methods already occupy every YMM, and an
    /// inlined activation (GELU in particular) makes RyuJIT spill the accumulators
    /// on every K step. Same store order as the old in-method <see cref="StoreRow"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoreEpilogue256(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        Vector256<float> alpha = Vector256.Create(alphaScalar), beta = Vector256.Create(betaScalar), one = Vector256.Create(1f);
        for (int r = 0; r < rows; r++)
        {
            StoreRow(output + r * outStride,
                Avx.LoadVector256(partial + r * OcBlock),
                Avx.LoadVector256(partial + r * OcBlock + 8),
                residual == null ? null : residual + r * resStride,
                activation, alpha, beta, one, biasAfter);
        }
    }

    // ------------------------------------------------------------ micro-kernels

    /// <summary>
    /// Pointwise tile: <paramref name="rows"/> (1..6) consecutive pixels with
    /// stride <paramref name="inStride"/> floats, input channels
    /// [<paramref name="k0"/>, <paramref name="kEnd"/>) against the weight
    /// panel <paramref name="w"/> ([k][16], already offset to k0). Accumulators
    /// always land in <paramref name="partial"/>; the caller runs
    /// <see cref="StoreEpilogue256"/> after the last K block. NoInlining keeps
    /// the epilogue from being compiled into this register-blocked loop.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void GemmTile16(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        float* p4 = p3 + (rows > 4 ? inStride : 0);
        float* p5 = p4 + (rows > 5 ? inStride : 0);
        Vector256<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
        if (k0 == 0)
        {
            Vector256<float> biasLow = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias);
            Vector256<float> biasHigh = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + 8);
            a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
            b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
        }
        else
        {
            a0 = Avx.LoadVector256(partial); b0 = Avx.LoadVector256(partial + 8);
            a1 = Avx.LoadVector256(partial + 16); b1 = Avx.LoadVector256(partial + 24);
            a2 = Avx.LoadVector256(partial + 32); b2 = Avx.LoadVector256(partial + 40);
            a3 = Avx.LoadVector256(partial + 48); b3 = Avx.LoadVector256(partial + 56);
            a4 = Avx.LoadVector256(partial + 64); b4 = Avx.LoadVector256(partial + 72);
            a5 = Avx.LoadVector256(partial + 80); b5 = Avx.LoadVector256(partial + 88);
        }
        for (int k = k0; k < kEnd; k++, w += 16)
        {
            Vector256<float> w0 = Avx.LoadVector256(w);
            Vector256<float> w1 = Avx.LoadVector256(w + 8);
            Vector256<float> v0 = Avx.BroadcastScalarToVector256(p0 + k);
            Vector256<float> v1 = Avx.BroadcastScalarToVector256(p1 + k);
            a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
            a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
            Vector256<float> v2 = Avx.BroadcastScalarToVector256(p2 + k);
            Vector256<float> v3 = Avx.BroadcastScalarToVector256(p3 + k);
            a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
            a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
            Vector256<float> v4 = Avx.BroadcastScalarToVector256(p4 + k);
            Vector256<float> v5 = Avx.BroadcastScalarToVector256(p5 + k);
            a4 = Fma.MultiplyAdd(v4, w0, a4); b4 = Fma.MultiplyAdd(v4, w1, b4);
            a5 = Fma.MultiplyAdd(v5, w0, a5); b5 = Fma.MultiplyAdd(v5, w1, b5);
        }
        Avx.Store(partial, a0); Avx.Store(partial + 8, b0);
        Avx.Store(partial + 16, a1); Avx.Store(partial + 24, b1);
        Avx.Store(partial + 32, a2); Avx.Store(partial + 40, b2);
        Avx.Store(partial + 48, a3); Avx.Store(partial + 56, b3);
        Avx.Store(partial + 64, a4); Avx.Store(partial + 72, b4);
        Avx.Store(partial + 80, a5); Avx.Store(partial + 88, b5);
    }

    /// <summary>
    /// Dense KxK tile: six output pixels whose receptive-field origins are
    /// <paramref name="p0"/>..<paramref name="p5"/>; <paramref name="tapOffsets"/>
    /// gives the float offset of every tap relative to the origin. Weights are
    /// [ic][tap][16] (already offset to <paramref name="k0"/>); the flattened
    /// reduction index k = ic * taps + tap runs input channels then taps,
    /// matching the reference summation order. Partial sums for non-final
    /// k-blocks live in <paramref name="partial"/>.
    /// </summary>
    /// <summary>
    /// Dense KxK tile: six output pixels whose receptive-field origins are
    /// <paramref name="p0"/>..<paramref name="p5"/>; <paramref name="tapOffsets"/>
    /// gives the float offset of every tap relative to the origin. Weights are
    /// [ic][tap][16] (already offset to <paramref name="k0"/>); the flattened
    /// reduction index k = ic * taps + tap runs input channels then taps,
    /// matching the reference summation order. Accumulators always land in
    /// <paramref name="partial"/>; the caller runs <see cref="StoreEpilogue256"/>
    /// after the last K block. Same NoInlining rationale as <see cref="GemmTile16"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ConvTile16(float* p0, float* p1, float* p2, float* p3, float* p4, float* p5,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        Vector256<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
        if (k0 == 0)
        {
            Vector256<float> biasLow = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias);
            Vector256<float> biasHigh = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + 8);
            a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
            b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
        }
        else
        {
            a0 = Avx.LoadVector256(partial); b0 = Avx.LoadVector256(partial + 8);
            a1 = Avx.LoadVector256(partial + 16); b1 = Avx.LoadVector256(partial + 24);
            a2 = Avx.LoadVector256(partial + 32); b2 = Avx.LoadVector256(partial + 40);
            a3 = Avx.LoadVector256(partial + 48); b3 = Avx.LoadVector256(partial + 56);
            a4 = Avx.LoadVector256(partial + 64); b4 = Avx.LoadVector256(partial + 72);
            a5 = Avx.LoadVector256(partial + 80); b5 = Avx.LoadVector256(partial + 88);
        }
        int ci = k0 / taps, tap = k0 - ci * taps;
        for (int k = k0; k < kEnd; k++, w += 16)
        {
            int offset = tapOffsets[tap] + ci;
            Vector256<float> w0 = Avx.LoadVector256(w);
            Vector256<float> w1 = Avx.LoadVector256(w + 8);
            Vector256<float> v0 = Avx.BroadcastScalarToVector256(p0 + offset);
            Vector256<float> v1 = Avx.BroadcastScalarToVector256(p1 + offset);
            a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
            a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
            Vector256<float> v2 = Avx.BroadcastScalarToVector256(p2 + offset);
            Vector256<float> v3 = Avx.BroadcastScalarToVector256(p3 + offset);
            a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
            a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
            Vector256<float> v4 = Avx.BroadcastScalarToVector256(p4 + offset);
            Vector256<float> v5 = Avx.BroadcastScalarToVector256(p5 + offset);
            a4 = Fma.MultiplyAdd(v4, w0, a4); b4 = Fma.MultiplyAdd(v4, w1, b4);
            a5 = Fma.MultiplyAdd(v5, w0, a5); b5 = Fma.MultiplyAdd(v5, w1, b5);
            if (++tap == taps) { tap = 0; ci++; }
        }
        Avx.Store(partial, a0); Avx.Store(partial + 8, b0);
        Avx.Store(partial + 16, a1); Avx.Store(partial + 24, b1);
        Avx.Store(partial + 32, a2); Avx.Store(partial + 40, b2);
        Avx.Store(partial + 48, a3); Avx.Store(partial + 56, b3);
        Avx.Store(partial + 64, a4); Avx.Store(partial + 72, b4);
        Avx.Store(partial + 80, a5); Avx.Store(partial + 88, b5);
    }

    /// <summary>
    /// Interior tiles of one output row, same FMA order as <see cref="ConvTile16"/>
    /// called once per tile. The epilogue stays a separate method so the 12
    /// accumulators are not spilled across K.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ConvInteriorTiles16(float* inBatch, int iy0, int rowStride, int tileBegin, int tileEnd,
        int outputWidth, int strideW, int padLeft, int pixelStride, int* tapOffsets, int taps, float* wPanel,
        int k0, int kEnd, float* bias, float* partial, float* outRow, float* resRow, int outputChannels, int oc,
        bool epilogue, NhwcActivation activation, float alpha, float beta)
    {
        for (int xTile = tileBegin; xTile < tileEnd; xTile++)
        {
            int x0 = xTile * TileRows;
            int rows = Math.Min(TileRows, outputWidth - x0);
            int ix0 = x0 * strideW - padLeft;
            int inputChannels = pixelStride / strideW;
            float* p0 = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
            float* p1 = p0 + (rows > 1 ? pixelStride : 0);
            float* p2 = p1 + (rows > 2 ? pixelStride : 0);
            float* p3 = p2 + (rows > 3 ? pixelStride : 0);
            float* p4 = p3 + (rows > 4 ? pixelStride : 0);
            float* p5 = p4 + (rows > 5 ? pixelStride : 0);
            float* part = partial + (long)xTile * PartialFloats;
            float* w = wPanel;
            Vector256<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
            if (k0 == 0)
            {
                Vector256<float> biasLow = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias);
                Vector256<float> biasHigh = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + 8);
                a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
                b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
            }
            else
            {
                a0 = Avx.LoadVector256(part); b0 = Avx.LoadVector256(part + 8);
                a1 = Avx.LoadVector256(part + 16); b1 = Avx.LoadVector256(part + 24);
                a2 = Avx.LoadVector256(part + 32); b2 = Avx.LoadVector256(part + 40);
                a3 = Avx.LoadVector256(part + 48); b3 = Avx.LoadVector256(part + 56);
                a4 = Avx.LoadVector256(part + 64); b4 = Avx.LoadVector256(part + 72);
                a5 = Avx.LoadVector256(part + 80); b5 = Avx.LoadVector256(part + 88);
            }
            int ci = k0 / taps, tap = k0 - ci * taps;
            for (int k = k0; k < kEnd; k++, w += 16)
            {
                int offset = tapOffsets[tap] + ci;
                Vector256<float> w0 = Avx.LoadVector256(w);
                Vector256<float> w1 = Avx.LoadVector256(w + 8);
                Vector256<float> v0 = Avx.BroadcastScalarToVector256(p0 + offset);
                Vector256<float> v1 = Avx.BroadcastScalarToVector256(p1 + offset);
                a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
                a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
                Vector256<float> v2 = Avx.BroadcastScalarToVector256(p2 + offset);
                Vector256<float> v3 = Avx.BroadcastScalarToVector256(p3 + offset);
                a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
                a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
                Vector256<float> v4 = Avx.BroadcastScalarToVector256(p4 + offset);
                Vector256<float> v5 = Avx.BroadcastScalarToVector256(p5 + offset);
                a4 = Fma.MultiplyAdd(v4, w0, a4); b4 = Fma.MultiplyAdd(v4, w1, b4);
                a5 = Fma.MultiplyAdd(v5, w0, a5); b5 = Fma.MultiplyAdd(v5, w1, b5);
                if (++tap == taps) { tap = 0; ci++; }
            }
            Avx.Store(part, a0); Avx.Store(part + 8, b0);
            Avx.Store(part + 16, a1); Avx.Store(part + 24, b1);
            Avx.Store(part + 32, a2); Avx.Store(part + 40, b2);
            Avx.Store(part + 48, a3); Avx.Store(part + 56, b3);
            Avx.Store(part + 64, a4); Avx.Store(part + 72, b4);
            Avx.Store(part + 80, a5); Avx.Store(part + 88, b5);
            if (epilogue)
            {
                float* outTile = outRow + (long)x0 * outputChannels + oc;
                float* resTile = resRow == null ? null : resRow + (long)x0 * outputChannels + oc;
                StoreEpilogue256(part, rows, outTile, outputChannels, resTile, outputChannels, activation, alpha, beta);
            }
        }
    }

    // --------------------------------------------------------------- pointwise

    /// <summary>
    /// 1x1 convolution over <paramref name="pixels"/> NHWC pixels. Weights are
    /// packed [oc/16][ic][16] (<see cref="PackDense"/> with a 1x1 kernel).
    /// </summary>
    private static void PointwiseAvx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int pixels, int inputChannels, int outputChannels, ReadOnlySpan<float> residual,
        NhwcActivation activation, float alpha, float beta, int threads)
    {
        if (outputChannels < 8 || (outputChannels & 7) != 0 || inputChannels <= 0)
            throw new ArgumentException("NHWC pointwise requires output channels to be a multiple of 8.");
        if (input.Length < (long)pixels * inputChannels || output.Length < (long)pixels * outputChannels ||
            packedWeights.Length < (long)inputChannels * outputChannels ||
            (!bias.IsEmpty && bias.Length < outputChannels) ||
            (!residual.IsEmpty && residual.Length < (long)pixels * outputChannels))
            throw new ArgumentException("NHWC pointwise buffer too small.");
        if (pixels <= 0) return;
        int tiles = (pixels + TileRows - 1) / TileRows;
        int groupTiles = Math.Max(1, PointwiseGroupTiles);
        long work = (long)pixels * inputChannels * outputChannels;
        // Shard whole tiles rather than tile groups: a group count that does
        // not divide the worker count hands one worker an extra group, and
        // that worker sets the makespan.  512x6x72->1024 is nine groups over
        // eight workers -- a 2:1 split -- and measured 530 GFLOPS at eight
        // workers against 804 at nine.  Tiles are independent, so balancing
        // them changes no arithmetic; each worker still walks its tiles in
        // group-sized steps so weight-panel reuse is unchanged.
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
                                    for (int tile = tileFrom; tile < tileTo; tile++)
                                    {
                                        int pixel = tile * TileRows;
                                        int rows = Math.Min(TileRows, pixels - pixel);
                                        float* outTile = outBase + (long)pixel * outputChannels + oc;
                                        float* resTile = resBase == null ? null : resBase + (long)pixel * outputChannels + oc;
                                        float* part = partial + (tile - tileFrom) * PartialFloats;
                                        GemmTile16(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd, b, part);
                                        if (kEnd == inputChannels)
                                            StoreEpilogue256(part, rows, outTile, outputChannels, resTile, outputChannels,
                                                activation, alpha, beta);
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
            if ((outputChannels >> 4) > 0)
            {
                if (workers > 1) Parallel.For(0, workers, Worker);
                else Worker(0);
            }
        }
        if ((outputChannels & 8) != 0)
            PointwiseTail8Avx(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
    }

    // ------------------------------------------------------------------- dense

    /// <summary>
    /// Dense KxK convolution (groups = 1), any stride, symmetric-or-not padding
    /// given by the top/left pad (bottom/right follow from the output size).
    /// Weights are packed [oc/16][ic][kh*kw][16] (<see cref="PackDense"/>).
    /// </summary>
    private static void DenseAvx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
        int taps = kernelH * kernelW;
        if (outputChannels < 8 || (outputChannels & 7) != 0 || inputChannels <= 0 || taps <= 0 || strideH <= 0 || strideW <= 0)
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
            PointwiseAvx(input, packedWeights, bias, output, batch * height * width, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
            return;
        }

        int xTiles = (outputWidth + TileRows - 1) / TileRows;
        int rowsTotal = batch * outputHeight;
        long work = outVolume * inputChannels * taps;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        // Border tiles gather their receptive field into a zero-padded patch:
        // kernelH rows of (5*strideW + kernelW) pixels.
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
                            // Interior tiles: every tap inside the input. Range
                            // [insideBegin, insideEnd) of x-tiles.
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
                                // Interior: k-blocked so one 16 x kc weight panel stays in L1 across the row's tiles.
                                for (int k0 = 0; k0 < kTotal; k0 += kc)
                                {
                                    int kEnd = Math.Min(kTotal, k0 + kc);
                                    float* wk = w + (long)k0 * OcBlock;
                                    // One call covers the row's interior tiles. A
                                    // per-tile call dominates when K is a single
                                    // 3×3 (the detector stem is K=27).
                                    float* outRowPtr = outBase + outRow * outputChannels;
                                    float* resRowPtr = resBase == null ? null : resBase + outRow * outputChannels;
                                    ConvInteriorTiles16(inBatch, iy0, rowStride, insideBegin, insideEnd,
                                        outputWidth, strideW, padLeft, pixelStride, tapOffsets, taps, wk, k0, kEnd, bb,
                                        partial, outRowPtr, resRowPtr, outputChannels, oc, kEnd == kTotal,
                                        activation, alpha, beta);
                                }
                                // Border tiles: gathered zero-padded patch, full reduction in one pass.
                                for (int xTile = 0; xTile < xTiles; xTile++)
                                {
                                    if (xTile >= insideBegin && xTile < insideEnd) continue;
                                    int x0 = xTile * TileRows;
                                    int rows = Math.Min(TileRows, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    float* outTile = outBase + (outRow + x0) * outputChannels + oc;
                                    float* resTile = resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc;
                                    GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                    float* p0 = patch;
                                    float* p1 = p0 + (rows > 1 ? pixelStride : 0);
                                    float* p2 = p1 + (rows > 2 ? pixelStride : 0);
                                    float* p3 = p2 + (rows > 3 ? pixelStride : 0);
                                    float* p4 = p3 + (rows > 4 ? pixelStride : 0);
                                    float* p5 = p4 + (rows > 5 ? pixelStride : 0);
                                    ConvTile16(p0, p1, p2, p3, p4, p5, patchOffsets, taps, w, 0, kTotal, bb, partial);
                                    StoreEpilogue256(partial, rows, outTile, outputChannels, resTile, outputChannels,
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
            if ((outputChannels >> 4) > 0)
            {
                if (workers > 1) Parallel.For(0, workers, Worker);
                else Worker(0);
            }
        }
        if ((outputChannels & 8) != 0)
            DenseTail8Avx(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
    }

    // ----------------------------------------------------------- conv transpose

    /// <summary>
    /// 2x2 / stride-2 transposed convolution as four strided pointwise GEMMs
    /// (one per output tap). Weights packed [tap][oc/16][ic][16]
    /// (<see cref="PackConvTranspose2x2"/>); bias and activation fused.
    /// </summary>
    private static void ConvTranspose2x2Stride2Avx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            ConvTranspose2x2Stride2SingleOutputAvx(input, packedWeights, bias, output, batch, inputChannels, height, width, activation, threads);
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
                                GemmTile16(inRow + (long)x0 * inputChannels, rows, inputChannels, w, 0, inputChannels, null, partial);
                                StoreEpilogue256(partial, rows,
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

    // Single output channel (detector probability map): four dot products per
    // input pixel. Weights packed [tap][ic].
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTranspose2x2Stride2SingleOutputAvx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
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
                int vectorEnd = inputChannels & ~7;
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / height, y = row - b * height;
                    float* inRow = inBase + ((long)b * height + y) * width * inputChannels;
                    float* out0 = outBase + ((long)b * height * 2 + 2 * y) * outputWidth;
                    float* out1 = out0 + outputWidth;
                    for (int x = 0; x < width; x++)
                    {
                        float* px = inRow + (long)x * inputChannels;
                        Vector256<float> s0 = Vector256<float>.Zero, s1 = Vector256<float>.Zero,
                            s2 = Vector256<float>.Zero, s3 = Vector256<float>.Zero;
                        int c = 0;
                        for (; c < vectorEnd; c += 8)
                        {
                            Vector256<float> v = Avx.LoadVector256(px + c);
                            s0 = Fma.MultiplyAdd(v, Avx.LoadVector256(w + c), s0);
                            s1 = Fma.MultiplyAdd(v, Avx.LoadVector256(w + inputChannels + c), s1);
                            s2 = Fma.MultiplyAdd(v, Avx.LoadVector256(w + 2 * inputChannels + c), s2);
                            s3 = Fma.MultiplyAdd(v, Avx.LoadVector256(w + 3 * inputChannels + c), s3);
                        }
                        float r0 = HorizontalSum(s0), r1 = HorizontalSum(s1), r2 = HorizontalSum(s2), r3 = HorizontalSum(s3);
                        for (; c < inputChannels; c++)
                        {
                            float v = px[c];
                            r0 += v * w[c]; r1 += v * w[inputChannels + c];
                            r2 += v * w[2 * inputChannels + c]; r3 += v * w[3 * inputChannels + c];
                        }
                        out0[2 * x] = ActivateScalar(r0 + biasValue, activation);
                        out0[2 * x + 1] = ActivateScalar(r1 + biasValue, activation);
                        out1[2 * x] = ActivateScalar(r2 + biasValue, activation);
                        out1[2 * x + 1] = ActivateScalar(r3 + biasValue, activation);
                    }
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // --------------------------------------------------------------- oc tail 8

    private const int TailLanes = 8;
    private const int TailPartial = TileRows * TailLanes;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoreEpilogue8(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        Vector256<float> alpha = Vector256.Create(alphaScalar), beta = Vector256.Create(betaScalar), one = Vector256.Create(1f);
        Vector256<float> bias = biasAfter == null ? Vector256<float>.Zero : Avx.LoadVector256(biasAfter);
        for (int r = 0; r < rows; r++)
        {
            Vector256<float> v = Avx.LoadVector256(partial + r * TailLanes);
            if (biasAfter != null) v = Avx.Add(v, bias);
            if (residual != null) v = Avx.Add(v, Avx.LoadVector256(residual + r * resStride));
            if (activation != NhwcActivation.None) v = Activate(v, activation, alpha, beta, one);
            Avx.Store(output + r * outStride, v);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void GemmTile8(float* input, int rows, int inStride, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        float* p4 = p3 + (rows > 4 ? inStride : 0);
        float* p5 = p4 + (rows > 5 ? inStride : 0);
        Vector256<float> a0, a1, a2, a3, a4, a5;
        if (k0 == 0)
        {
            Vector256<float> b = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias);
            a0 = a1 = a2 = a3 = a4 = a5 = b;
        }
        else
        {
            a0 = Avx.LoadVector256(partial);
            a1 = Avx.LoadVector256(partial + 8);
            a2 = Avx.LoadVector256(partial + 16);
            a3 = Avx.LoadVector256(partial + 24);
            a4 = Avx.LoadVector256(partial + 32);
            a5 = Avx.LoadVector256(partial + 40);
        }
        for (int k = k0; k < kEnd; k++, w += 8)
        {
            Vector256<float> wk = Avx.LoadVector256(w);
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p0 + k), wk, a0);
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p1 + k), wk, a1);
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p2 + k), wk, a2);
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p3 + k), wk, a3);
            a4 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p4 + k), wk, a4);
            a5 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p5 + k), wk, a5);
        }
        Avx.Store(partial, a0); Avx.Store(partial + 8, a1);
        Avx.Store(partial + 16, a2); Avx.Store(partial + 24, a3);
        Avx.Store(partial + 32, a4); Avx.Store(partial + 40, a5);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ConvTile8(float* p0, float* p1, float* p2, float* p3, float* p4, float* p5,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        Vector256<float> a0, a1, a2, a3, a4, a5;
        if (k0 == 0)
        {
            Vector256<float> b = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias);
            a0 = a1 = a2 = a3 = a4 = a5 = b;
        }
        else
        {
            a0 = Avx.LoadVector256(partial);
            a1 = Avx.LoadVector256(partial + 8);
            a2 = Avx.LoadVector256(partial + 16);
            a3 = Avx.LoadVector256(partial + 24);
            a4 = Avx.LoadVector256(partial + 32);
            a5 = Avx.LoadVector256(partial + 40);
        }
        int ci = k0 / taps, tap = k0 - ci * taps;
        for (int k = k0; k < kEnd; k++, w += 8)
        {
            int offset = tapOffsets[tap] + ci;
            Vector256<float> wk = Avx.LoadVector256(w);
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p0 + offset), wk, a0);
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p1 + offset), wk, a1);
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p2 + offset), wk, a2);
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p3 + offset), wk, a3);
            a4 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p4 + offset), wk, a4);
            a5 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(p5 + offset), wk, a5);
            if (++tap == taps) { tap = 0; ci++; }
        }
        Avx.Store(partial, a0); Avx.Store(partial + 8, a1);
        Avx.Store(partial + 16, a2); Avx.Store(partial + 24, a3);
        Avx.Store(partial + 32, a4); Avx.Store(partial + 40, a5);
    }

    private static void PointwiseTail8Avx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int pixels, int inputChannels, int outputChannels, ReadOnlySpan<float> residual,
        NhwcActivation activation, float alpha, float beta, int threads)
    {
        int oc0 = outputChannels & ~15;
        int tiles = (pixels + TileRows - 1) / TileRows;
        int groupTiles = Math.Max(1, PointwiseGroupTiles);
        long work = (long)pixels * inputChannels * TailLanes;
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
                float[] partialArray = ArrayPool<float>.Shared.Rent(groupTiles * TailPartial);
                try
                {
                    fixed (float* partial = partialArray)
                    {
                        float* inBase = (float*)inA, outBase = (float*)outA;
                        float* wBase = (float*)wA + (long)(outputChannels >> 4) * inputChannels * OcBlock;
                        float* biasBase = hasBias ? (float*)bA + oc0 : null;
                        float* resBase = hasResidual ? (float*)rA : null;
                        int firstGroup = tileBegin / groupTiles, lastGroup = (tileEnd + groupTiles - 1) / groupTiles;
                        for (int group = firstGroup; group < lastGroup; group++)
                        {
                            int tileFrom = Math.Max(tileBegin, group * groupTiles);
                            int tileTo = Math.Min(tileEnd, (group + 1) * groupTiles);
                            if (tileTo <= tileFrom) continue;
                            for (int k0 = 0; k0 < inputChannels; k0 += kc)
                            {
                                int kEnd = Math.Min(inputChannels, k0 + kc);
                                float* wk = wBase + (long)k0 * TailLanes;
                                for (int tile = tileFrom; tile < tileTo; tile++)
                                {
                                    int pixel = tile * TileRows;
                                    int rows = Math.Min(TileRows, pixels - pixel);
                                    float* part = partial + (tile - tileFrom) * TailPartial;
                                    GemmTile8(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd, biasBase, part);
                                    if (kEnd == inputChannels)
                                        StoreEpilogue8(part, rows, outBase + (long)pixel * outputChannels + oc0, outputChannels,
                                            resBase == null ? null : resBase + (long)pixel * outputChannels + oc0, outputChannels,
                                            activation, alpha, beta);
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

    private static void DenseTail8Avx(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
        int taps = kernelH * kernelW;
        int oc0 = outputChannels & ~15;
        int kTotal = inputChannels * taps;
        int xTiles = (outputWidth + TileRows - 1) / TileRows;
        int rowsTotal = batch * outputHeight;
        long outVolume = (long)batch * outputHeight * outputWidth * TailLanes;
        int workers = threads > 1 && outVolume * kTotal >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
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
                int kc = DenseKc <= 0 ? kTotal : Math.Min(kTotal, DenseKc);
                int[] tapArray = ArrayPool<int>.Shared.Rent(taps * 2);
                float[] patchArray = ArrayPool<float>.Shared.Rent(patchFloats);
                float[] partialArray = ArrayPool<float>.Shared.Rent(xTiles * TailPartial);
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
                        float* inBase = (float*)inA, outBase = (float*)outA;
                        float* wBase = (float*)wA + (long)(outputChannels >> 4) * kTotal * OcBlock;
                        float* biasBase = hasBias ? (float*)bA + oc0 : null;
                        float* resBase = hasResidual ? (float*)rA : null;
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
                            for (int k0 = 0; k0 < kTotal; k0 += kc)
                            {
                                int kEnd = Math.Min(kTotal, k0 + kc);
                                float* wk = wBase + (long)k0 * TailLanes;
                                for (int xTile = insideBegin; xTile < insideEnd; xTile++)
                                {
                                    int x0 = xTile * TileRows;
                                    int rows = Math.Min(TileRows, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    float* p0 = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
                                    float* p1 = p0 + (rows > 1 ? pixelStride : 0);
                                    float* p2 = p1 + (rows > 2 ? pixelStride : 0);
                                    float* p3 = p2 + (rows > 3 ? pixelStride : 0);
                                    float* p4 = p3 + (rows > 4 ? pixelStride : 0);
                                    float* p5 = p4 + (rows > 5 ? pixelStride : 0);
                                    float* part = partial + (long)xTile * TailPartial;
                                    ConvTile8(p0, p1, p2, p3, p4, p5, tapOffsets, taps, wk, k0, kEnd, biasBase, part);
                                    if (kEnd == kTotal)
                                        StoreEpilogue8(part, rows, outBase + (outRow + x0) * outputChannels + oc0, outputChannels,
                                            resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc0, outputChannels,
                                            activation, alpha, beta);
                                }
                            }
                            for (int xTile = 0; xTile < xTiles; xTile++)
                            {
                                if (xTile >= insideBegin && xTile < insideEnd) continue;
                                int x0 = xTile * TileRows;
                                int rows = Math.Min(TileRows, outputWidth - x0);
                                int ix0 = x0 * strideW - padLeft;
                                GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                float* p0 = patch;
                                float* p1 = p0 + (rows > 1 ? pixelStride : 0);
                                float* p2 = p1 + (rows > 2 ? pixelStride : 0);
                                float* p3 = p2 + (rows > 3 ? pixelStride : 0);
                                float* p4 = p3 + (rows > 4 ? pixelStride : 0);
                                float* p5 = p4 + (rows > 5 ? pixelStride : 0);
                                ConvTile8(p0, p1, p2, p3, p4, p5, patchOffsets, taps, wBase, 0, kTotal, biasBase, partial);
                                StoreEpilogue8(partial, rows, outBase + (outRow + x0) * outputChannels + oc0, outputChannels,
                                    resBase == null ? null : resBase + (outRow + x0) * outputChannels + oc0, outputChannels,
                                    activation, alpha, beta);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum(Vector256<float> v)
    {
        Vector128<float> s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 0x55));
        return s.ToScalar();
    }

    // ----------------------------------------------------------------- packing
}
