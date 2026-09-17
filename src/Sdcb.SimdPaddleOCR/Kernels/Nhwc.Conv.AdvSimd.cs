#if !NETSTANDARD2_0
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// ARM64 AdvSimd (NEON) channels-last convolution kernels. Same register-blocking
// scheme as the AVX2 micro-kernel (6 pixels x 16 output channels = 24 128-bit
// accumulators, bias-initialised FMA chain over the flattened k = ic * taps + tap
// index in ascending order). The pointwise kernel groups four k steps per
// iteration: one 16-byte input load per pixel row feeds four
// FMLA-by-element (FusedMultiplyAddBySelectedScalar) instructions against the
// four weight vectors of each k, so 96 FMAs cost 6 input loads + 16 weight
// loads. The dense kernel's tap walk is not contiguous in k, so it uses a
// scalar broadcast (LD1R) per pixel per k instead. Like the AVX-512 kernels,
// the tile kernels always spill their accumulators to the partial buffer and
// never call out: any call in the method (or an inlined branchy epilogue such
// as the Gelu polynomial) makes RyuJIT park the accumulators on the stack for
// the whole k loop. The fused epilogue runs separately in StoreEpilogueAdvSimd.
internal static unsafe partial class Nhwc
{
    // ---------------------------------------------------------------- epilogue

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ActivateAdvSimd(Vector128<float> v, NhwcActivation activation,
        Vector128<float> alpha, Vector128<float> beta, Vector128<float> one)
    {
        switch (activation)
        {
            case NhwcActivation.Relu:
                return AdvSimd.Max(v, Vector128<float>.Zero);
            case NhwcActivation.HardSwish:
            {
                Vector128<float> gate = AdvSimd.Add(AdvSimd.Multiply(v, alpha), beta);
                gate = AdvSimd.Max(Vector128<float>.Zero, AdvSimd.Min(one, gate));
                return AdvSimd.Multiply(v, gate);
            }
            case NhwcActivation.Gelu:
                return SimdKernels.GeluVectorAdvSimd(v);
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
    // NOTE: NoInlining alone compiles as Tier1, which measured ~6x slower on
    // the branchy erf interval selection than the same inlined code under
    // AggressiveOptimization (FullOpts, like SimdKernels.Erf).
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplCompat.AggressiveOptimization)]
    private static void StoreEpilogueAdvSimd(float* partial, int rows, float* output, int outStride,
        float* residual, int resStride, NhwcActivation activation, float alphaScalar, float betaScalar,
        float* biasAfter = null)
    {
        Vector128<float> alpha = Vector128.Create(alphaScalar), beta = Vector128.Create(betaScalar), one = Vector128.Create(1f);
        for (int r = 0; r < rows; r++)
        {
            Vector128<float> a0 = AdvSimd.LoadVector128(partial + r * OcBlock);
            Vector128<float> a1 = AdvSimd.LoadVector128(partial + r * OcBlock + 4);
            Vector128<float> a2 = AdvSimd.LoadVector128(partial + r * OcBlock + 8);
            Vector128<float> a3 = AdvSimd.LoadVector128(partial + r * OcBlock + 12);
            if (biasAfter != null)
            {
                a0 = AdvSimd.Add(a0, AdvSimd.LoadVector128(biasAfter));
                a1 = AdvSimd.Add(a1, AdvSimd.LoadVector128(biasAfter + 4));
                a2 = AdvSimd.Add(a2, AdvSimd.LoadVector128(biasAfter + 8));
                a3 = AdvSimd.Add(a3, AdvSimd.LoadVector128(biasAfter + 12));
            }
            if (residual != null)
            {
                float* res = residual + r * resStride;
                a0 = AdvSimd.Add(a0, AdvSimd.LoadVector128(res));
                a1 = AdvSimd.Add(a1, AdvSimd.LoadVector128(res + 4));
                a2 = AdvSimd.Add(a2, AdvSimd.LoadVector128(res + 8));
                a3 = AdvSimd.Add(a3, AdvSimd.LoadVector128(res + 12));
            }
            if (activation != NhwcActivation.None)
            {
                a0 = ActivateAdvSimd(a0, activation, alpha, beta, one);
                a1 = ActivateAdvSimd(a1, activation, alpha, beta, one);
                a2 = ActivateAdvSimd(a2, activation, alpha, beta, one);
                a3 = ActivateAdvSimd(a3, activation, alpha, beta, one);
            }
            float* dst = output + r * outStride;
            AdvSimd.Store(dst, a0);
            AdvSimd.Store(dst + 4, a1);
            AdvSimd.Store(dst + 8, a2);
            AdvSimd.Store(dst + 12, a3);
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
    private static void GemmTileAdvSimd(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        float* p4 = p3 + (rows > 4 ? inStride : 0);
        float* p5 = p4 + (rows > 5 ? inStride : 0);
        Vector128<float> a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23;
        Vector128<float> a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53;
        if (k0 == 0)
        {
            Vector128<float> bias0 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias);
            Vector128<float> bias1 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 4);
            Vector128<float> bias2 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 8);
            Vector128<float> bias3 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 12);
            a00 = a10 = a20 = a30 = a40 = a50 = bias0;
            a01 = a11 = a21 = a31 = a41 = a51 = bias1;
            a02 = a12 = a22 = a32 = a42 = a52 = bias2;
            a03 = a13 = a23 = a33 = a43 = a53 = bias3;
        }
        else
        {
            a00 = AdvSimd.LoadVector128(partial); a01 = AdvSimd.LoadVector128(partial + 4);
            a02 = AdvSimd.LoadVector128(partial + 8); a03 = AdvSimd.LoadVector128(partial + 12);
            a10 = AdvSimd.LoadVector128(partial + 16); a11 = AdvSimd.LoadVector128(partial + 20);
            a12 = AdvSimd.LoadVector128(partial + 24); a13 = AdvSimd.LoadVector128(partial + 28);
            a20 = AdvSimd.LoadVector128(partial + 32); a21 = AdvSimd.LoadVector128(partial + 36);
            a22 = AdvSimd.LoadVector128(partial + 40); a23 = AdvSimd.LoadVector128(partial + 44);
            a30 = AdvSimd.LoadVector128(partial + 48); a31 = AdvSimd.LoadVector128(partial + 52);
            a32 = AdvSimd.LoadVector128(partial + 56); a33 = AdvSimd.LoadVector128(partial + 60);
            a40 = AdvSimd.LoadVector128(partial + 64); a41 = AdvSimd.LoadVector128(partial + 68);
            a42 = AdvSimd.LoadVector128(partial + 72); a43 = AdvSimd.LoadVector128(partial + 76);
            a50 = AdvSimd.LoadVector128(partial + 80); a51 = AdvSimd.LoadVector128(partial + 84);
            a52 = AdvSimd.LoadVector128(partial + 88); a53 = AdvSimd.LoadVector128(partial + 92);
        }
        int k = k0;
        for (; k + 4 <= kEnd; k += 4, w += 64)
        {
            Vector128<float> v0 = AdvSimd.LoadVector128(p0 + k);
            Vector128<float> v1 = AdvSimd.LoadVector128(p1 + k);
            Vector128<float> v2 = AdvSimd.LoadVector128(p2 + k);
            Vector128<float> v3 = AdvSimd.LoadVector128(p3 + k);
            Vector128<float> v4 = AdvSimd.LoadVector128(p4 + k);
            Vector128<float> v5 = AdvSimd.LoadVector128(p5 + k);
            Vector128<float> w0 = AdvSimd.LoadVector128(w);
            Vector128<float> w1 = AdvSimd.LoadVector128(w + 4);
            Vector128<float> w2 = AdvSimd.LoadVector128(w + 8);
            Vector128<float> w3 = AdvSimd.LoadVector128(w + 12);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 0);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 0);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 0);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 0);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 0);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 0);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 0);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 0);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 0);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 0);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 0);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 0);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 0);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 0);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 0);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 0);
            a40 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a40, w0, v4, 0);
            a41 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a41, w1, v4, 0);
            a42 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a42, w2, v4, 0);
            a43 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a43, w3, v4, 0);
            a50 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a50, w0, v5, 0);
            a51 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a51, w1, v5, 0);
            a52 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a52, w2, v5, 0);
            a53 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a53, w3, v5, 0);
            w0 = AdvSimd.LoadVector128(w + 16);
            w1 = AdvSimd.LoadVector128(w + 20);
            w2 = AdvSimd.LoadVector128(w + 24);
            w3 = AdvSimd.LoadVector128(w + 28);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 1);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 1);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 1);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 1);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 1);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 1);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 1);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 1);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 1);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 1);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 1);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 1);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 1);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 1);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 1);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 1);
            a40 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a40, w0, v4, 1);
            a41 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a41, w1, v4, 1);
            a42 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a42, w2, v4, 1);
            a43 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a43, w3, v4, 1);
            a50 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a50, w0, v5, 1);
            a51 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a51, w1, v5, 1);
            a52 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a52, w2, v5, 1);
            a53 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a53, w3, v5, 1);
            w0 = AdvSimd.LoadVector128(w + 32);
            w1 = AdvSimd.LoadVector128(w + 36);
            w2 = AdvSimd.LoadVector128(w + 40);
            w3 = AdvSimd.LoadVector128(w + 44);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 2);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 2);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 2);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 2);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 2);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 2);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 2);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 2);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 2);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 2);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 2);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 2);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 2);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 2);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 2);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 2);
            a40 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a40, w0, v4, 2);
            a41 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a41, w1, v4, 2);
            a42 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a42, w2, v4, 2);
            a43 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a43, w3, v4, 2);
            a50 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a50, w0, v5, 2);
            a51 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a51, w1, v5, 2);
            a52 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a52, w2, v5, 2);
            a53 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a53, w3, v5, 2);
            w0 = AdvSimd.LoadVector128(w + 48);
            w1 = AdvSimd.LoadVector128(w + 52);
            w2 = AdvSimd.LoadVector128(w + 56);
            w3 = AdvSimd.LoadVector128(w + 60);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 3);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 3);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 3);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 3);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 3);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 3);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 3);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 3);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 3);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 3);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 3);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 3);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 3);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 3);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 3);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 3);
            a40 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a40, w0, v4, 3);
            a41 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a41, w1, v4, 3);
            a42 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a42, w2, v4, 3);
            a43 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a43, w3, v4, 3);
            a50 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a50, w0, v5, 3);
            a51 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a51, w1, v5, 3);
            a52 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a52, w2, v5, 3);
            a53 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a53, w3, v5, 3);
        }
        for (; k < kEnd; k++, w += 16)
        {
            Vector128<float> w0 = AdvSimd.LoadVector128(w);
            Vector128<float> w1 = AdvSimd.LoadVector128(w + 4);
            Vector128<float> w2 = AdvSimd.LoadVector128(w + 8);
            Vector128<float> w3 = AdvSimd.LoadVector128(w + 12);
            Vector128<float> v0 = Vector128.Create(p0[k]);
            Vector128<float> v1 = Vector128.Create(p1[k]);
            a00 = AdvSimd.FusedMultiplyAdd(a00, w0, v0);
            a01 = AdvSimd.FusedMultiplyAdd(a01, w1, v0);
            a02 = AdvSimd.FusedMultiplyAdd(a02, w2, v0);
            a03 = AdvSimd.FusedMultiplyAdd(a03, w3, v0);
            a10 = AdvSimd.FusedMultiplyAdd(a10, w0, v1);
            a11 = AdvSimd.FusedMultiplyAdd(a11, w1, v1);
            a12 = AdvSimd.FusedMultiplyAdd(a12, w2, v1);
            a13 = AdvSimd.FusedMultiplyAdd(a13, w3, v1);
            Vector128<float> v2 = Vector128.Create(p2[k]);
            Vector128<float> v3 = Vector128.Create(p3[k]);
            a20 = AdvSimd.FusedMultiplyAdd(a20, w0, v2);
            a21 = AdvSimd.FusedMultiplyAdd(a21, w1, v2);
            a22 = AdvSimd.FusedMultiplyAdd(a22, w2, v2);
            a23 = AdvSimd.FusedMultiplyAdd(a23, w3, v2);
            a30 = AdvSimd.FusedMultiplyAdd(a30, w0, v3);
            a31 = AdvSimd.FusedMultiplyAdd(a31, w1, v3);
            a32 = AdvSimd.FusedMultiplyAdd(a32, w2, v3);
            a33 = AdvSimd.FusedMultiplyAdd(a33, w3, v3);
            Vector128<float> v4 = Vector128.Create(p4[k]);
            Vector128<float> v5 = Vector128.Create(p5[k]);
            a40 = AdvSimd.FusedMultiplyAdd(a40, w0, v4);
            a41 = AdvSimd.FusedMultiplyAdd(a41, w1, v4);
            a42 = AdvSimd.FusedMultiplyAdd(a42, w2, v4);
            a43 = AdvSimd.FusedMultiplyAdd(a43, w3, v4);
            a50 = AdvSimd.FusedMultiplyAdd(a50, w0, v5);
            a51 = AdvSimd.FusedMultiplyAdd(a51, w1, v5);
            a52 = AdvSimd.FusedMultiplyAdd(a52, w2, v5);
            a53 = AdvSimd.FusedMultiplyAdd(a53, w3, v5);
        }
        AdvSimd.Store(partial, a00); AdvSimd.Store(partial + 4, a01);
        AdvSimd.Store(partial + 8, a02); AdvSimd.Store(partial + 12, a03);
        AdvSimd.Store(partial + 16, a10); AdvSimd.Store(partial + 20, a11);
        AdvSimd.Store(partial + 24, a12); AdvSimd.Store(partial + 28, a13);
        AdvSimd.Store(partial + 32, a20); AdvSimd.Store(partial + 36, a21);
        AdvSimd.Store(partial + 40, a22); AdvSimd.Store(partial + 44, a23);
        AdvSimd.Store(partial + 48, a30); AdvSimd.Store(partial + 52, a31);
        AdvSimd.Store(partial + 56, a32); AdvSimd.Store(partial + 60, a33);
        AdvSimd.Store(partial + 64, a40); AdvSimd.Store(partial + 68, a41);
        AdvSimd.Store(partial + 72, a42); AdvSimd.Store(partial + 76, a43);
        AdvSimd.Store(partial + 80, a50); AdvSimd.Store(partial + 84, a51);
        AdvSimd.Store(partial + 88, a52); AdvSimd.Store(partial + 92, a53);
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
    private static void ConvTileAdvSimd(float* input, int rows, int pixelStride,
        int* tapOffsets, int taps, float* w, int k0, int kEnd, float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? pixelStride : 0);
        float* p2 = p1 + (rows > 2 ? pixelStride : 0);
        float* p3 = p2 + (rows > 3 ? pixelStride : 0);
        float* p4 = p3 + (rows > 4 ? pixelStride : 0);
        float* p5 = p4 + (rows > 5 ? pixelStride : 0);
        Vector128<float> a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23;
        Vector128<float> a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53;
        if (k0 == 0)
        {
            Vector128<float> bias0 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias);
            Vector128<float> bias1 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 4);
            Vector128<float> bias2 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 8);
            Vector128<float> bias3 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 12);
            a00 = a10 = a20 = a30 = a40 = a50 = bias0;
            a01 = a11 = a21 = a31 = a41 = a51 = bias1;
            a02 = a12 = a22 = a32 = a42 = a52 = bias2;
            a03 = a13 = a23 = a33 = a43 = a53 = bias3;
        }
        else
        {
            a00 = AdvSimd.LoadVector128(partial); a01 = AdvSimd.LoadVector128(partial + 4);
            a02 = AdvSimd.LoadVector128(partial + 8); a03 = AdvSimd.LoadVector128(partial + 12);
            a10 = AdvSimd.LoadVector128(partial + 16); a11 = AdvSimd.LoadVector128(partial + 20);
            a12 = AdvSimd.LoadVector128(partial + 24); a13 = AdvSimd.LoadVector128(partial + 28);
            a20 = AdvSimd.LoadVector128(partial + 32); a21 = AdvSimd.LoadVector128(partial + 36);
            a22 = AdvSimd.LoadVector128(partial + 40); a23 = AdvSimd.LoadVector128(partial + 44);
            a30 = AdvSimd.LoadVector128(partial + 48); a31 = AdvSimd.LoadVector128(partial + 52);
            a32 = AdvSimd.LoadVector128(partial + 56); a33 = AdvSimd.LoadVector128(partial + 60);
            a40 = AdvSimd.LoadVector128(partial + 64); a41 = AdvSimd.LoadVector128(partial + 68);
            a42 = AdvSimd.LoadVector128(partial + 72); a43 = AdvSimd.LoadVector128(partial + 76);
            a50 = AdvSimd.LoadVector128(partial + 80); a51 = AdvSimd.LoadVector128(partial + 84);
            a52 = AdvSimd.LoadVector128(partial + 88); a53 = AdvSimd.LoadVector128(partial + 92);
        }
        int ci = k0 / taps, tap = k0 - ci * taps;
        for (int k = k0; k < kEnd; k++, w += 16)
        {
            int offset = tapOffsets[tap] + ci;
            Vector128<float> w0 = AdvSimd.LoadVector128(w);
            Vector128<float> w1 = AdvSimd.LoadVector128(w + 4);
            Vector128<float> w2 = AdvSimd.LoadVector128(w + 8);
            Vector128<float> w3 = AdvSimd.LoadVector128(w + 12);
            Vector128<float> v0 = Vector128.Create(p0[offset]);
            Vector128<float> v1 = Vector128.Create(p1[offset]);
            a00 = AdvSimd.FusedMultiplyAdd(a00, w0, v0);
            a01 = AdvSimd.FusedMultiplyAdd(a01, w1, v0);
            a02 = AdvSimd.FusedMultiplyAdd(a02, w2, v0);
            a03 = AdvSimd.FusedMultiplyAdd(a03, w3, v0);
            a10 = AdvSimd.FusedMultiplyAdd(a10, w0, v1);
            a11 = AdvSimd.FusedMultiplyAdd(a11, w1, v1);
            a12 = AdvSimd.FusedMultiplyAdd(a12, w2, v1);
            a13 = AdvSimd.FusedMultiplyAdd(a13, w3, v1);
            Vector128<float> v2 = Vector128.Create(p2[offset]);
            Vector128<float> v3 = Vector128.Create(p3[offset]);
            a20 = AdvSimd.FusedMultiplyAdd(a20, w0, v2);
            a21 = AdvSimd.FusedMultiplyAdd(a21, w1, v2);
            a22 = AdvSimd.FusedMultiplyAdd(a22, w2, v2);
            a23 = AdvSimd.FusedMultiplyAdd(a23, w3, v2);
            a30 = AdvSimd.FusedMultiplyAdd(a30, w0, v3);
            a31 = AdvSimd.FusedMultiplyAdd(a31, w1, v3);
            a32 = AdvSimd.FusedMultiplyAdd(a32, w2, v3);
            a33 = AdvSimd.FusedMultiplyAdd(a33, w3, v3);
            Vector128<float> v4 = Vector128.Create(p4[offset]);
            Vector128<float> v5 = Vector128.Create(p5[offset]);
            a40 = AdvSimd.FusedMultiplyAdd(a40, w0, v4);
            a41 = AdvSimd.FusedMultiplyAdd(a41, w1, v4);
            a42 = AdvSimd.FusedMultiplyAdd(a42, w2, v4);
            a43 = AdvSimd.FusedMultiplyAdd(a43, w3, v4);
            a50 = AdvSimd.FusedMultiplyAdd(a50, w0, v5);
            a51 = AdvSimd.FusedMultiplyAdd(a51, w1, v5);
            a52 = AdvSimd.FusedMultiplyAdd(a52, w2, v5);
            a53 = AdvSimd.FusedMultiplyAdd(a53, w3, v5);
            if (++tap == taps) { tap = 0; ci++; }
        }
        AdvSimd.Store(partial, a00); AdvSimd.Store(partial + 4, a01);
        AdvSimd.Store(partial + 8, a02); AdvSimd.Store(partial + 12, a03);
        AdvSimd.Store(partial + 16, a10); AdvSimd.Store(partial + 20, a11);
        AdvSimd.Store(partial + 24, a12); AdvSimd.Store(partial + 28, a13);
        AdvSimd.Store(partial + 32, a20); AdvSimd.Store(partial + 36, a21);
        AdvSimd.Store(partial + 40, a22); AdvSimd.Store(partial + 44, a23);
        AdvSimd.Store(partial + 48, a30); AdvSimd.Store(partial + 52, a31);
        AdvSimd.Store(partial + 56, a32); AdvSimd.Store(partial + 60, a33);
        AdvSimd.Store(partial + 64, a40); AdvSimd.Store(partial + 68, a41);
        AdvSimd.Store(partial + 72, a42); AdvSimd.Store(partial + 76, a43);
        AdvSimd.Store(partial + 80, a50); AdvSimd.Store(partial + 84, a51);
        AdvSimd.Store(partial + 88, a52); AdvSimd.Store(partial + 92, a53);
    }

    // --------------------------------------------------------------- pointwise

    /// <summary>
    /// Pointwise tile: <paramref name="rows"/> (1..4) consecutive pixels with
    /// stride <paramref name="inStride"/> floats, input channels
    /// [<paramref name="k0"/>, <paramref name="kEnd"/>) against the weight
    /// panel <paramref name="w"/> ([k][16], already offset to k0). Accumulators
    /// always land in <paramref name="partial"/>.
    /// 4 rows x 16 OC keeps 16 accumulators + 4 input + 4 weight vectors live
    /// (24 &lt; 32 architectural registers); the 6-row tile used by dense conv
    /// spills two accumulators to the stack in this loop (see JIT disasm:
    /// per-update ldr/str q29 pairs), costing ~30% of pointwise throughput.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void GemmTile4AdvSimd(float* input, int rows, int inStride, float* w, int k0, int kEnd,
        float* bias, float* partial)
    {
        float* p0 = input;
        float* p1 = p0 + (rows > 1 ? inStride : 0);
        float* p2 = p1 + (rows > 2 ? inStride : 0);
        float* p3 = p2 + (rows > 3 ? inStride : 0);
        Vector128<float> a00, a01, a02, a03, a10, a11, a12, a13;
        Vector128<float> a20, a21, a22, a23, a30, a31, a32, a33;
        if (k0 == 0)
        {
            Vector128<float> bias0 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias);
            Vector128<float> bias1 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 4);
            Vector128<float> bias2 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 8);
            Vector128<float> bias3 = bias == null ? Vector128<float>.Zero : AdvSimd.LoadVector128(bias + 12);
            a00 = a10 = a20 = a30 = bias0;
            a01 = a11 = a21 = a31 = bias1;
            a02 = a12 = a22 = a32 = bias2;
            a03 = a13 = a23 = a33 = bias3;
        }
        else
        {
            a00 = AdvSimd.LoadVector128(partial); a01 = AdvSimd.LoadVector128(partial + 4);
            a02 = AdvSimd.LoadVector128(partial + 8); a03 = AdvSimd.LoadVector128(partial + 12);
            a10 = AdvSimd.LoadVector128(partial + 16); a11 = AdvSimd.LoadVector128(partial + 20);
            a12 = AdvSimd.LoadVector128(partial + 24); a13 = AdvSimd.LoadVector128(partial + 28);
            a20 = AdvSimd.LoadVector128(partial + 32); a21 = AdvSimd.LoadVector128(partial + 36);
            a22 = AdvSimd.LoadVector128(partial + 40); a23 = AdvSimd.LoadVector128(partial + 44);
            a30 = AdvSimd.LoadVector128(partial + 48); a31 = AdvSimd.LoadVector128(partial + 52);
            a32 = AdvSimd.LoadVector128(partial + 56); a33 = AdvSimd.LoadVector128(partial + 60);
        }
        int k = k0;
        for (; k + 4 <= kEnd; k += 4, w += 64)
        {
            Vector128<float> v0 = AdvSimd.LoadVector128(p0 + k);
            Vector128<float> v1 = AdvSimd.LoadVector128(p1 + k);
            Vector128<float> v2 = AdvSimd.LoadVector128(p2 + k);
            Vector128<float> v3 = AdvSimd.LoadVector128(p3 + k);
            Vector128<float> w0 = AdvSimd.LoadVector128(w);
            Vector128<float> w1 = AdvSimd.LoadVector128(w + 4);
            Vector128<float> w2 = AdvSimd.LoadVector128(w + 8);
            Vector128<float> w3 = AdvSimd.LoadVector128(w + 12);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 0);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 0);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 0);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 0);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 0);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 0);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 0);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 0);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 0);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 0);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 0);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 0);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 0);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 0);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 0);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 0);
            w0 = AdvSimd.LoadVector128(w + 16);
            w1 = AdvSimd.LoadVector128(w + 20);
            w2 = AdvSimd.LoadVector128(w + 24);
            w3 = AdvSimd.LoadVector128(w + 28);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 1);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 1);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 1);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 1);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 1);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 1);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 1);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 1);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 1);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 1);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 1);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 1);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 1);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 1);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 1);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 1);
            w0 = AdvSimd.LoadVector128(w + 32);
            w1 = AdvSimd.LoadVector128(w + 36);
            w2 = AdvSimd.LoadVector128(w + 40);
            w3 = AdvSimd.LoadVector128(w + 44);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 2);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 2);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 2);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 2);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 2);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 2);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 2);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 2);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 2);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 2);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 2);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 2);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 2);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 2);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 2);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 2);
            w0 = AdvSimd.LoadVector128(w + 48);
            w1 = AdvSimd.LoadVector128(w + 52);
            w2 = AdvSimd.LoadVector128(w + 56);
            w3 = AdvSimd.LoadVector128(w + 60);
            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, v0, 3);
            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, v0, 3);
            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, v0, 3);
            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, v0, 3);
            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, v1, 3);
            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, v1, 3);
            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, v1, 3);
            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, v1, 3);
            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, v2, 3);
            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, v2, 3);
            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, v2, 3);
            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, v2, 3);
            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, v3, 3);
            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, v3, 3);
            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, v3, 3);
            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, v3, 3);
        }
        for (; k < kEnd; k++, w += 16)
        {
            Vector128<float> w0 = AdvSimd.LoadVector128(w);
            Vector128<float> w1 = AdvSimd.LoadVector128(w + 4);
            Vector128<float> w2 = AdvSimd.LoadVector128(w + 8);
            Vector128<float> w3 = AdvSimd.LoadVector128(w + 12);
            Vector128<float> v0 = Vector128.Create(p0[k]);
            Vector128<float> v1 = Vector128.Create(p1[k]);
            Vector128<float> v2 = Vector128.Create(p2[k]);
            Vector128<float> v3 = Vector128.Create(p3[k]);
            a00 = AdvSimd.FusedMultiplyAdd(a00, w0, v0);
            a01 = AdvSimd.FusedMultiplyAdd(a01, w1, v0);
            a02 = AdvSimd.FusedMultiplyAdd(a02, w2, v0);
            a03 = AdvSimd.FusedMultiplyAdd(a03, w3, v0);
            a10 = AdvSimd.FusedMultiplyAdd(a10, w0, v1);
            a11 = AdvSimd.FusedMultiplyAdd(a11, w1, v1);
            a12 = AdvSimd.FusedMultiplyAdd(a12, w2, v1);
            a13 = AdvSimd.FusedMultiplyAdd(a13, w3, v1);
            a20 = AdvSimd.FusedMultiplyAdd(a20, w0, v2);
            a21 = AdvSimd.FusedMultiplyAdd(a21, w1, v2);
            a22 = AdvSimd.FusedMultiplyAdd(a22, w2, v2);
            a23 = AdvSimd.FusedMultiplyAdd(a23, w3, v2);
            a30 = AdvSimd.FusedMultiplyAdd(a30, w0, v3);
            a31 = AdvSimd.FusedMultiplyAdd(a31, w1, v3);
            a32 = AdvSimd.FusedMultiplyAdd(a32, w2, v3);
            a33 = AdvSimd.FusedMultiplyAdd(a33, w3, v3);
        }
        AdvSimd.Store(partial, a00); AdvSimd.Store(partial + 4, a01);
        AdvSimd.Store(partial + 8, a02); AdvSimd.Store(partial + 12, a03);
        AdvSimd.Store(partial + 16, a10); AdvSimd.Store(partial + 20, a11);
        AdvSimd.Store(partial + 24, a12); AdvSimd.Store(partial + 28, a13);
        AdvSimd.Store(partial + 32, a20); AdvSimd.Store(partial + 36, a21);
        AdvSimd.Store(partial + 40, a22); AdvSimd.Store(partial + 44, a23);
        AdvSimd.Store(partial + 48, a30); AdvSimd.Store(partial + 52, a31);
        AdvSimd.Store(partial + 56, a32); AdvSimd.Store(partial + 60, a33);
    }

    /// <summary>AdvSimd driver of <see cref="Pointwise"/>; same loop nest as the AVX2/AVX-512 drivers.</summary>
    internal static void PointwiseAdvSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
        // Pointwise tiles are 4 rows (GemmTile4AdvSimd); dense conv keeps the
        // 6-row TileRows kernel. Shard whole tiles like the AVX2 driver so a
        // leftover group cannot unbalance workers.
        const int TileRows4 = 4, PartialFloats4 = TileRows4 * OcBlock;
        int tiles = (pixels + TileRows4 - 1) / TileRows4;
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
                float[] partialArray = ArrayPool<float>.Shared.Rent(groupTiles * PartialFloats4);
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
                                        int pixel = tile * TileRows4;
                                        int rows = Math.Min(TileRows4, pixels - pixel);
                                        float* tilePartial = partial + (tile - tileFrom) * PartialFloats4;
                                        GemmTile4AdvSimd(inBase + (long)pixel * inputChannels, rows, inputChannels, wk, k0, kEnd,
                                            b, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueAdvSimd(tilePartial, rows,
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

    /// <summary>AdvSimd driver of <see cref="Dense"/>; same loop nest as the AVX2/AVX-512 drivers.</summary>
    internal static void DenseAdvSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            PointwiseAdvSimd(input, packedWeights, bias, output, batch * height * width, inputChannels, outputChannels,
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
                                        ConvTileAdvSimd(origin, rows, pixelStride, tapOffsets, taps, wk, k0, kEnd, bb, tilePartial);
                                        if (final)
                                        {
                                            StoreEpilogueAdvSimd(tilePartial, rows,
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
                                    ConvTileAdvSimd(patch, rows, pixelStride, patchOffsets, taps, w, 0, kTotal, bb, partial);
                                    StoreEpilogueAdvSimd(partial, rows,
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

    // ----------------------------------------------------------- conv transpose

    /// <summary>AdvSimd driver of <see cref="ConvTranspose2x2Stride2"/>; same loop nest as the AVX2 driver.</summary>
    internal static void ConvTranspose2x2Stride2AdvSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
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
            ConvTranspose2x2Stride2SingleOutputAdvSimd(input, packedWeights, bias, output, batch, inputChannels, height, width, activation, threads);
            return;
        }
        if ((outputChannels & 15) != 0 || packedWeights.Length < 4L * inputChannels * outputChannels)
            throw new ArgumentException("NHWC transposed convolution requires output channels to be a multiple of 16.");
        int rowsTotal = batch * height;
        long work = outVolume * inputChannels;
        int workers = threads > 1 && work >= 2_000_000 ? Math.Min(threads, rowsTotal) : 1;
        // 4-row tiles (GemmTile4AdvSimd): the 6-row kernel spills accumulators
        // in its 4k-unrolled loop; 16 acc + 4 input + 4 weight vectors fit.
        const int TileRows4 = TileRows, PartialFloats4 = PartialFloats;
        int xTiles = (width + TileRows4 - 1) / TileRows4;
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
                float* partial = stackalloc float[PartialFloats4];
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
                                int x0 = xTile * TileRows4;
                                int rows = Math.Min(TileRows4, width - x0);
                                GemmTileAdvSimd(inRow + (long)x0 * inputChannels, rows, inputChannels, w, 0, inputChannels,
                                    null, partial);
                                StoreEpilogueAdvSimd(partial, rows,
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
    private static void ConvTranspose2x2Stride2SingleOutputAdvSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
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
                int vectorEnd = inputChannels & ~3;
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / height, y = row - b * height;
                    float* inRow = inBase + ((long)b * height + y) * width * inputChannels;
                    float* out0 = outBase + ((long)b * height * 2 + 2 * y) * outputWidth;
                    float* out1 = out0 + outputWidth;
                    for (int x = 0; x < width; x++)
                    {
                        float* px = inRow + (long)x * inputChannels;
                        Vector128<float> s0 = Vector128<float>.Zero, s1 = Vector128<float>.Zero,
                            s2 = Vector128<float>.Zero, s3 = Vector128<float>.Zero;
                        int c = 0;
                        for (; c < vectorEnd; c += 4)
                        {
                            Vector128<float> v = AdvSimd.LoadVector128(px + c);
                            s0 = AdvSimd.FusedMultiplyAdd(s0, v, AdvSimd.LoadVector128(w + c));
                            s1 = AdvSimd.FusedMultiplyAdd(s1, v, AdvSimd.LoadVector128(w + inputChannels + c));
                            s2 = AdvSimd.FusedMultiplyAdd(s2, v, AdvSimd.LoadVector128(w + 2 * inputChannels + c));
                            s3 = AdvSimd.FusedMultiplyAdd(s3, v, AdvSimd.LoadVector128(w + 3 * inputChannels + c));
                        }
                        float r0 = Vector128.Sum(s0);
                        float r1 = Vector128.Sum(s1);
                        float r2 = Vector128.Sum(s2);
                        float r3 = Vector128.Sum(s3);
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
}
#endif
