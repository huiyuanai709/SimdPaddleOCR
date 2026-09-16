#if NETSTANDARD2_0
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

// Vector<float> (netstandard2.0) channels-last convolution kernels. Same
// register-blocking scheme as the AVX2 micro-kernel (6 pixels x 16 output
// channels, bias-initialised chain over the flattened k = ic * taps + tap
// index in ascending order). Count == 8 is one 16-OC pass (12 accumulators +
// 2 weight vectors). Count == 4 keeps the same 12-accumulator budget as two
// 8-OC half-panel passes over the packed [k][16] weights. Other widths fall
// back to a scalar double loop. There is no FMA in Vector<T>: every
// accumulation is a separate multiply then add, exactly matching the per-lane
// arithmetic of the NCHW Vector kernels. Like the AVX-512 kernels, the tile
// kernels always spill their accumulators to the partial buffer and never
// call out (any call in the method makes RyuJIT park the accumulators on the
// stack for the whole k loop); the fused epilogue runs separately in
// StoreEpilogueVec.
internal static unsafe partial class Nhwc
{
    private const int TileRows = 6;
    private const int OcBlock = 16;
    /// <summary>Partial-sum floats per tile (6 x 16).</summary>
    private const int PartialFloats = TileRows * OcBlock;

    // ---------------------------------------------------------------- epilogue

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ActivateVec(Vector<float> v, NhwcActivation activation,
        Vector<float> alpha, Vector<float> beta, Vector<float> one)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return Vector.Max(v, Vector<float>.Zero);
            case NhwcActivation.HardSwish:
            {
                Vector<float> gate = v * alpha + beta;
                gate = Vector.Max(Vector<float>.Zero, Vector.Min(one, gate));
                return v * gate;
            }
            case NhwcActivation.Gelu:
                return SimdKernels.GeluVectorExact(v);
            default:
                return v;
        }
    }

    /// <summary>
    /// Fused epilogue for one finished tile: reloads the partial rows, adds the
    /// deferred bias (transposed convolution) and residual, applies the
    /// activation and stores to the output, in the same order as the AVX2
    /// StoreRow. Runs once per tile, so it stays out of the register-critical
    /// k loops (a plain call, no inlining into the kernels).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoreEpilogueVec(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        if (Vector<float>.Count == 8)
        {
            Vector<float> alpha = new(alphaScalar), beta = new(betaScalar), one = new(1f);
            for (int r = 0; r < rows; r++)
            {
                Vector<float> a = VectorLoad(partial + r * OcBlock);
                Vector<float> b = VectorLoad(partial + r * OcBlock + 8);
                if (biasAfter != null)
                {
                    a += VectorLoad(biasAfter);
                    b += VectorLoad(biasAfter + 8);
                }
                if (residual != null)
                {
                    a += VectorLoad(residual + r * resStride);
                    b += VectorLoad(residual + r * resStride + 8);
                }
                if (activation != NhwcActivation.None)
                {
                    a = ActivateVec(a, activation, alpha, beta, one);
                    b = ActivateVec(b, activation, alpha, beta, one);
                }
                VectorStore(output + r * outStride, a);
                VectorStore(output + r * outStride + 8, b);
            }
            return;
        }
        if (Vector<float>.Count == 4)
        {
            Vector<float> alpha = new(alphaScalar), beta = new(betaScalar), one = new(1f);
            for (int r = 0; r < rows; r++)
            {
                for (int oc0 = 0; oc0 < OcBlock; oc0 += 8)
                {
                    Vector<float> a = VectorLoad(partial + r * OcBlock + oc0);
                    Vector<float> b = VectorLoad(partial + r * OcBlock + oc0 + 4);
                    if (biasAfter != null)
                    {
                        a += VectorLoad(biasAfter + oc0);
                        b += VectorLoad(biasAfter + oc0 + 4);
                    }
                    if (residual != null)
                    {
                        a += VectorLoad(residual + r * resStride + oc0);
                        b += VectorLoad(residual + r * resStride + oc0 + 4);
                    }
                    if (activation != NhwcActivation.None)
                    {
                        a = ActivateVec(a, activation, alpha, beta, one);
                        b = ActivateVec(b, activation, alpha, beta, one);
                    }
                    VectorStore(output + r * outStride + oc0, a);
                    VectorStore(output + r * outStride + oc0 + 4, b);
                }
            }
            return;
        }
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < OcBlock; c++)
            {
                float v = partial[r * OcBlock + c];
                if (biasAfter != null) v += biasAfter[c];
                if (residual != null) v += residual[r * resStride + c];
                output[r * outStride + c] = ActivateScalarValue(v, activation, alphaScalar, betaScalar);
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ActivateScalarValue(float v, NhwcActivation activation, float alpha, float beta)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return MathF.Max(v, 0f);
            case NhwcActivation.HardSwish:
            {
                float gate = MathF.Max(0f, MathF.Min(1f, v * alpha + beta));
                return v * gate;
            }
            case NhwcActivation.Gelu:
            {
                float e = SimdKernels.Erf(v * 0.70710678118654752f) + 1f;
                return v * e * 0.5f;
            }
            default:
                return v;
        }
    }

    // ------------------------------------------------------------ micro-kernels

    /// <summary>
    /// Pointwise tile: <paramref name="rows"/> (1..6) consecutive pixels with
    /// stride <paramref name="inStride"/> floats, input channels
    /// [<paramref name="k0"/>, <paramref name="kEnd"/>) against the weight
    /// panel <paramref name="w"/> ([k][16], already offset to k0). Accumulators
    /// always land in <paramref name="partial"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void GemmTile16Vec(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        if (Vector<float>.Count == 8)
        {
            float* p0 = input;
            float* p1 = p0 + (rows > 1 ? inStride : 0);
            float* p2 = p1 + (rows > 2 ? inStride : 0);
            float* p3 = p2 + (rows > 3 ? inStride : 0);
            float* p4 = p3 + (rows > 4 ? inStride : 0);
            float* p5 = p4 + (rows > 5 ? inStride : 0);
            Vector<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
            if (k0 == 0)
            {
                Vector<float> biasLow = bias == null ? Vector<float>.Zero : VectorLoad(bias);
                Vector<float> biasHigh = bias == null ? Vector<float>.Zero : VectorLoad(bias + 8);
                a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
                b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
            }
            else
            {
                a0 = VectorLoad(partial); b0 = VectorLoad(partial + 8);
                a1 = VectorLoad(partial + 16); b1 = VectorLoad(partial + 24);
                a2 = VectorLoad(partial + 32); b2 = VectorLoad(partial + 40);
                a3 = VectorLoad(partial + 48); b3 = VectorLoad(partial + 56);
                a4 = VectorLoad(partial + 64); b4 = VectorLoad(partial + 72);
                a5 = VectorLoad(partial + 80); b5 = VectorLoad(partial + 88);
            }
            for (int k = k0; k < kEnd; k++, w += 16)
            {
                Vector<float> w0 = VectorLoad(w);
                Vector<float> w1 = VectorLoad(w + 8);
                Vector<float> v0 = new(p0[k]);
                Vector<float> v1 = new(p1[k]);
                a0 += v0 * w0; b0 += v0 * w1;
                a1 += v1 * w0; b1 += v1 * w1;
                Vector<float> v2 = new(p2[k]);
                Vector<float> v3 = new(p3[k]);
                a2 += v2 * w0; b2 += v2 * w1;
                a3 += v3 * w0; b3 += v3 * w1;
                Vector<float> v4 = new(p4[k]);
                Vector<float> v5 = new(p5[k]);
                a4 += v4 * w0; b4 += v4 * w1;
                a5 += v5 * w0; b5 += v5 * w1;
            }
            VectorStore(partial, a0); VectorStore(partial + 8, b0);
            VectorStore(partial + 16, a1); VectorStore(partial + 24, b1);
            VectorStore(partial + 32, a2); VectorStore(partial + 40, b2);
            VectorStore(partial + 48, a3); VectorStore(partial + 56, b3);
            VectorStore(partial + 64, a4); VectorStore(partial + 72, b4);
            VectorStore(partial + 80, a5); VectorStore(partial + 88, b5);
            return;
        }
        if (Vector<float>.Count == 4)
        {
            float* p0 = input;
            float* p1 = p0 + (rows > 1 ? inStride : 0);
            float* p2 = p1 + (rows > 2 ? inStride : 0);
            float* p3 = p2 + (rows > 3 ? inStride : 0);
            float* p4 = p3 + (rows > 4 ? inStride : 0);
            float* p5 = p4 + (rows > 5 ? inStride : 0);
            for (int oc0 = 0; oc0 < OcBlock; oc0 += 8)
            {
                Vector<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
                if (k0 == 0)
                {
                    Vector<float> biasLow = bias == null ? Vector<float>.Zero : VectorLoad(bias + oc0);
                    Vector<float> biasHigh = bias == null ? Vector<float>.Zero : VectorLoad(bias + oc0 + 4);
                    a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
                    b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
                }
                else
                {
                    a0 = VectorLoad(partial + oc0); b0 = VectorLoad(partial + oc0 + 4);
                    a1 = VectorLoad(partial + 16 + oc0); b1 = VectorLoad(partial + 20 + oc0);
                    a2 = VectorLoad(partial + 32 + oc0); b2 = VectorLoad(partial + 36 + oc0);
                    a3 = VectorLoad(partial + 48 + oc0); b3 = VectorLoad(partial + 52 + oc0);
                    a4 = VectorLoad(partial + 64 + oc0); b4 = VectorLoad(partial + 68 + oc0);
                    a5 = VectorLoad(partial + 80 + oc0); b5 = VectorLoad(partial + 84 + oc0);
                }
                float* wk = w + oc0;
                for (int k = k0; k < kEnd; k++, wk += 16)
                {
                    Vector<float> w0 = VectorLoad(wk);
                    Vector<float> w1 = VectorLoad(wk + 4);
                    Vector<float> v0 = new(p0[k]);
                    Vector<float> v1 = new(p1[k]);
                    a0 += v0 * w0; b0 += v0 * w1;
                    a1 += v1 * w0; b1 += v1 * w1;
                    Vector<float> v2 = new(p2[k]);
                    Vector<float> v3 = new(p3[k]);
                    a2 += v2 * w0; b2 += v2 * w1;
                    a3 += v3 * w0; b3 += v3 * w1;
                    Vector<float> v4 = new(p4[k]);
                    Vector<float> v5 = new(p5[k]);
                    a4 += v4 * w0; b4 += v4 * w1;
                    a5 += v5 * w0; b5 += v5 * w1;
                }
                VectorStore(partial + oc0, a0); VectorStore(partial + oc0 + 4, b0);
                VectorStore(partial + 16 + oc0, a1); VectorStore(partial + 20 + oc0, b1);
                VectorStore(partial + 32 + oc0, a2); VectorStore(partial + 36 + oc0, b2);
                VectorStore(partial + 48 + oc0, a3); VectorStore(partial + 52 + oc0, b3);
                VectorStore(partial + 64 + oc0, a4); VectorStore(partial + 68 + oc0, b4);
                VectorStore(partial + 80 + oc0, a5); VectorStore(partial + 84 + oc0, b5);
            }
            return;
        }
        for (int r = 0; r < rows; r++)
        {
            float* pr = input + (long)r * inStride;
            float* acc = partial + r * OcBlock;
            if (k0 == 0)
                for (int c = 0; c < OcBlock; c++) acc[c] = bias == null ? 0f : bias[c];
            float* wk = w;
            for (int k = k0; k < kEnd; k++, wk += OcBlock)
            {
                float v = pr[k];
                for (int c = 0; c < OcBlock; c++) acc[c] += v * wk[c];
            }
        }
    }

    /// <summary>
    /// Dense KxK tile: up to 6 output pixels whose receptive-field origins are
    /// <paramref name="input"/> plus row multiples of <paramref name="pixelStride"/>;
    /// <paramref name="tapOffsets"/> gives the float offset of every tap relative
    /// to the origin. Weights are [ic][tap][16] (already offset to
    /// <paramref name="k0"/>); the flattened reduction index k = ic * taps + tap
    /// runs input channels then taps, matching the reference summation order.
    /// Accumulators always land in <paramref name="partial"/>.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTile16Vec(float* input, int rows, int pixelStride,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        if (Vector<float>.Count == 8)
        {
            float* p0 = input;
            float* p1 = p0 + (rows > 1 ? pixelStride : 0);
            float* p2 = p1 + (rows > 2 ? pixelStride : 0);
            float* p3 = p2 + (rows > 3 ? pixelStride : 0);
            float* p4 = p3 + (rows > 4 ? pixelStride : 0);
            float* p5 = p4 + (rows > 5 ? pixelStride : 0);
            Vector<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
            if (k0 == 0)
            {
                Vector<float> biasLow = bias == null ? Vector<float>.Zero : VectorLoad(bias);
                Vector<float> biasHigh = bias == null ? Vector<float>.Zero : VectorLoad(bias + 8);
                a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
                b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
            }
            else
            {
                a0 = VectorLoad(partial); b0 = VectorLoad(partial + 8);
                a1 = VectorLoad(partial + 16); b1 = VectorLoad(partial + 24);
                a2 = VectorLoad(partial + 32); b2 = VectorLoad(partial + 40);
                a3 = VectorLoad(partial + 48); b3 = VectorLoad(partial + 56);
                a4 = VectorLoad(partial + 64); b4 = VectorLoad(partial + 72);
                a5 = VectorLoad(partial + 80); b5 = VectorLoad(partial + 88);
            }
            int ci = k0 / taps, tap = k0 - ci * taps;
            for (int k = k0; k < kEnd; k++, w += 16)
            {
                int offset = tapOffsets[tap] + ci;
                Vector<float> w0 = VectorLoad(w);
                Vector<float> w1 = VectorLoad(w + 8);
                Vector<float> v0 = new(p0[offset]);
                Vector<float> v1 = new(p1[offset]);
                a0 += v0 * w0; b0 += v0 * w1;
                a1 += v1 * w0; b1 += v1 * w1;
                Vector<float> v2 = new(p2[offset]);
                Vector<float> v3 = new(p3[offset]);
                a2 += v2 * w0; b2 += v2 * w1;
                a3 += v3 * w0; b3 += v3 * w1;
                Vector<float> v4 = new(p4[offset]);
                Vector<float> v5 = new(p5[offset]);
                a4 += v4 * w0; b4 += v4 * w1;
                a5 += v5 * w0; b5 += v5 * w1;
                if (++tap == taps) { tap = 0; ci++; }
            }
            VectorStore(partial, a0); VectorStore(partial + 8, b0);
            VectorStore(partial + 16, a1); VectorStore(partial + 24, b1);
            VectorStore(partial + 32, a2); VectorStore(partial + 40, b2);
            VectorStore(partial + 48, a3); VectorStore(partial + 56, b3);
            VectorStore(partial + 64, a4); VectorStore(partial + 72, b4);
            VectorStore(partial + 80, a5); VectorStore(partial + 88, b5);
            return;
        }
        if (Vector<float>.Count == 4)
        {
            float* p0 = input;
            float* p1 = p0 + (rows > 1 ? pixelStride : 0);
            float* p2 = p1 + (rows > 2 ? pixelStride : 0);
            float* p3 = p2 + (rows > 3 ? pixelStride : 0);
            float* p4 = p3 + (rows > 4 ? pixelStride : 0);
            float* p5 = p4 + (rows > 5 ? pixelStride : 0);
            for (int oc0 = 0; oc0 < OcBlock; oc0 += 8)
            {
                Vector<float> a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
                if (k0 == 0)
                {
                    Vector<float> biasLow = bias == null ? Vector<float>.Zero : VectorLoad(bias + oc0);
                    Vector<float> biasHigh = bias == null ? Vector<float>.Zero : VectorLoad(bias + oc0 + 4);
                    a0 = a1 = a2 = a3 = a4 = a5 = biasLow;
                    b0 = b1 = b2 = b3 = b4 = b5 = biasHigh;
                }
                else
                {
                    a0 = VectorLoad(partial + oc0); b0 = VectorLoad(partial + oc0 + 4);
                    a1 = VectorLoad(partial + 16 + oc0); b1 = VectorLoad(partial + 20 + oc0);
                    a2 = VectorLoad(partial + 32 + oc0); b2 = VectorLoad(partial + 36 + oc0);
                    a3 = VectorLoad(partial + 48 + oc0); b3 = VectorLoad(partial + 52 + oc0);
                    a4 = VectorLoad(partial + 64 + oc0); b4 = VectorLoad(partial + 68 + oc0);
                    a5 = VectorLoad(partial + 80 + oc0); b5 = VectorLoad(partial + 84 + oc0);
                }
                int ci = k0 / taps, tap = k0 - ci * taps;
                float* wk = w + oc0;
                for (int k = k0; k < kEnd; k++, wk += 16)
                {
                    int offset = tapOffsets[tap] + ci;
                    Vector<float> w0 = VectorLoad(wk);
                    Vector<float> w1 = VectorLoad(wk + 4);
                    Vector<float> v0 = new(p0[offset]);
                    Vector<float> v1 = new(p1[offset]);
                    a0 += v0 * w0; b0 += v0 * w1;
                    a1 += v1 * w0; b1 += v1 * w1;
                    Vector<float> v2 = new(p2[offset]);
                    Vector<float> v3 = new(p3[offset]);
                    a2 += v2 * w0; b2 += v2 * w1;
                    a3 += v3 * w0; b3 += v3 * w1;
                    Vector<float> v4 = new(p4[offset]);
                    Vector<float> v5 = new(p5[offset]);
                    a4 += v4 * w0; b4 += v4 * w1;
                    a5 += v5 * w0; b5 += v5 * w1;
                    if (++tap == taps) { tap = 0; ci++; }
                }
                VectorStore(partial + oc0, a0); VectorStore(partial + oc0 + 4, b0);
                VectorStore(partial + 16 + oc0, a1); VectorStore(partial + 20 + oc0, b1);
                VectorStore(partial + 32 + oc0, a2); VectorStore(partial + 36 + oc0, b2);
                VectorStore(partial + 48 + oc0, a3); VectorStore(partial + 52 + oc0, b3);
                VectorStore(partial + 64 + oc0, a4); VectorStore(partial + 68 + oc0, b4);
                VectorStore(partial + 80 + oc0, a5); VectorStore(partial + 84 + oc0, b5);
            }
            return;
        }
        for (int r = 0; r < rows; r++)
        {
            float* pr = input + (long)r * pixelStride;
            float* acc = partial + r * OcBlock;
            if (k0 == 0)
                for (int c = 0; c < OcBlock; c++) acc[c] = bias == null ? 0f : bias[c];
            int ci0 = k0 / taps, tap0 = k0 - ci0 * taps;
            float* wk = w;
            for (int k = k0; k < kEnd; k++, wk += OcBlock)
            {
                float v = pr[tapOffsets[tap0] + ci0];
                for (int c = 0; c < OcBlock; c++) acc[c] += v * wk[c];
                if (++tap0 == taps) { tap0 = 0; ci0++; }
            }
        }
    }

    // --------------------------------------------------------------- pointwise

    /// <summary>
    /// 1x1 convolution over <paramref name="pixels"/> NHWC pixels. Weights are
    /// packed [oc/16][ic][16] (<see cref="PackDense"/> with a 1x1 kernel).
    /// </summary>
    internal static void Pointwise(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
        // Shard whole tiles rather than tile groups: same makespan fix as the
        // AVX2 Pointwise path (PR #8). Tiles are independent; each worker still
        // walks its tiles in group-sized steps so weight-panel reuse is unchanged.
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
                                        GemmTile16Vec(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd,
                                            b, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueVec(tilePartial, rows,
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

    /// <summary>
    /// Dense KxK convolution (groups = 1), any stride, symmetric-or-not padding
    /// given by the top/left pad (bottom/right follow from the output size).
    /// Weights are packed [oc/16][ic][kh*kw][16] (<see cref="PackDense"/>).
    /// </summary>
    internal static void Dense(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            Pointwise(input, packedWeights, bias, output, batch * height * width, inputChannels, outputChannels,
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
                                    bool final = kEnd == kTotal;
                                    for (int xTile = insideBegin; xTile < insideEnd; xTile++)
                                    {
                                        int x0 = xTile * TileRows;
                                        int rows = Math.Min(TileRows, outputWidth - x0);
                                        int ix0 = x0 * strideW - padLeft;
                                        float* tilePartial = partial + xTile * PartialFloats;
                                        float* origin = inBatch + (long)iy0 * rowStride + (long)ix0 * inputChannels;
                                        ConvTile16Vec(origin, rows, pixelStride, tapOffsets, taps, wk, k0, kEnd, bb, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueVec(tilePartial, rows,
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
                                    int x0 = xTile * TileRows;
                                    int rows = Math.Min(TileRows, outputWidth - x0);
                                    int ix0 = x0 * strideW - padLeft;
                                    GatherPatch(inBatch, patch, height, width, inputChannels, iy0, ix0, kernelH, patchWidth);
                                    ConvTile16Vec(patch, rows, pixelStride, patchOffsets, taps, w, 0, kTotal, bb, partial);
                                    StoreEpilogueVec(partial, rows,
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

    /// <summary>Copies the receptive field at (iy0, ix0) into a zero-padded patch of kernelH x patchWidth pixels.</summary>
    private static void GatherPatch(float* input, float* patch, int height, int width, int channels,
        int iy0, int ix0, int kernelH, int patchWidth)
    {
        int rowFloats = patchWidth * channels;
        for (int ky = 0; ky < kernelH; ky++)
        {
            float* dstRow = patch + ky * rowFloats;
            int iy = iy0 + ky;
            if ((uint)iy >= (uint)height)
            {
                new Span<float>(dstRow, rowFloats).Clear();
                continue;
            }
            int xBegin = Math.Max(0, -ix0), xEnd = Math.Min(patchWidth, width - ix0);
            if (xBegin > 0) new Span<float>(dstRow, xBegin * channels).Clear();
            if (xEnd < patchWidth) new Span<float>(dstRow + Math.Max(xEnd, 0) * channels, (patchWidth - Math.Max(xEnd, 0)) * channels).Clear();
            if (xEnd > xBegin)
            {
                float* src = input + ((long)iy * width + ix0 + xBegin) * channels;
                Buffer.MemoryCopy(src, dstRow + xBegin * channels, (long)(xEnd - xBegin) * channels * sizeof(float),
                    (long)(xEnd - xBegin) * channels * sizeof(float));
            }
        }
    }

    // ----------------------------------------------------------- conv transpose

    /// <summary>
    /// 2x2 / stride-2 transposed convolution as four strided pointwise GEMMs
    /// (one per output tap). Weights packed [tap][oc/16][ic][16]
    /// (<see cref="PackConvTranspose2x2"/>); bias and activation fused.
    /// </summary>
    internal static void ConvTranspose2x2Stride2(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            ConvTranspose2x2Stride2SingleOutput(input, packedWeights, bias, output, batch, inputChannels, height, width, activation, threads);
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
                                GemmTile16Vec(inRow + (long)x0 * inputChannels, rows, inputChannels, w, 0, inputChannels,
                                    null, partial);
                                StoreEpilogueVec(partial, rows,
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
    // input pixel, bias-initialised, input channels ascending, matching the
    // NCHW Vector accumulation order lane for lane. Weights packed [tap][ic].
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ConvTranspose2x2Stride2SingleOutput(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
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
#endif
