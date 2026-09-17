using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>Fused epilogue applied by the NHWC convolution kernels before the store.</summary>
internal enum NhwcActivation
{
    None = 0,
    Relu = 1,
    /// <summary>x * clamp(alpha * x + beta, 0, 1); same formula as <see cref="SimdKernels.HardSwish"/>.</summary>
    HardSwish = 2,
    Sigmoid = 3,
    /// <summary>0.5 * x * (1 + erf(x / sqrt2)), same polynomial as <see cref="SimdKernels.Gelu"/>.</summary>
    Gelu = 4,
}

/// <summary>
/// Channels-last (NHWC) kernels used when <see cref="OnnxSharp.LayoutPlanner"/>
/// runs a segment of the graph channels-last. All entry points take logical
/// NCHW dimensions; the data is stored as [n][h][w][c]. Each entry dispatches
/// with <c>Avx512F.IsSupported</c> / AVX2+FMA / AdvSIMD / <see cref="Vector{T}"/>
/// widths 4/8 / scalar, same as the NCHW kernels. The Vector files compile on
/// both TFMs; ns2 cannot name the Avx / AdvSIMD methods.
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <summary>Output-channel block width of the dense/pointwise micro-kernel.</summary>
    internal const int OutputChannelBlock = 16;
    private const int OcBlock = OutputChannelBlock;
    private const int TileRows = 6;
    /// <summary>Partial-sum floats per tile (6 x 16).</summary>
    private const int PartialFloats = TileRows * OcBlock;

    // Tuning knobs (measured on Zen 3; see test KernelBench). 0 disables blocking.
    /// <summary>Input-channel block for the pointwise GEMM; the L2-streamed full panel measured best on Zen 3.</summary>
    internal static int PointwiseKc = 0;
    /// <summary>Pixel tiles per group sharing one weight panel walk.</summary>
    internal static int PointwiseGroupTiles = 8;
    /// <summary>Flattened (ic x tap) reduction block for dense convolutions: 512 x 16 x 4 B = 32 KiB of weights.</summary>
    internal static int DenseKc = 512;
    /// <summary>Depthwise kernels at least this tall accumulate one kernel row per pass instead of all taps in registers.</summary>
    internal static int DepthwiseStripMinKernelH = 6;

    /// <summary>
    /// 1x1 convolution over <paramref name="pixels"/> NHWC pixels. Weights are
    /// packed [oc/16][ic][16] (<see cref="PackDense"/> with a 1x1 kernel).
    /// </summary>
    internal static void Pointwise(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int pixels, int inputChannels, int outputChannels, ReadOnlySpan<float> residual,
        NhwcActivation activation, float alpha, float beta, int threads)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            Pointwise512(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            PointwiseAvx(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
        else if (AdvSimd.Arm64.IsSupported)
            PointwiseAdvSimd(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            PointwiseVec(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
        else
            PointwiseScalar(input, packedWeights, bias, output, pixels, inputChannels, outputChannels,
                residual, activation, alpha, beta, threads);
    }

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
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            Dense512(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            DenseAvx(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
        else if (AdvSimd.Arm64.IsSupported)
            DenseAdvSimd(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            DenseVec(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
        else
            DenseScalar(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels,
                outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft,
                residual, activation, alpha, beta, threads);
    }

    /// <summary>
    /// Depthwise KxK convolution, any stride/padding. Weights are packed
    /// [c/32][tap][32] (<see cref="PackDepthwise"/>).
    /// </summary>
    internal static void Depthwise(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int channels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta, int threads)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            DepthwiseAvx(input, packedWeights, bias, output, batch, channels, height, width, outputHeight, outputWidth,
                kernelH, kernelW, strideH, strideW, padTop, padLeft, residual, activation, alpha, beta, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            DepthwiseAvx(input, packedWeights, bias, output, batch, channels, height, width, outputHeight, outputWidth,
                kernelH, kernelW, strideH, strideW, padTop, padLeft, residual, activation, alpha, beta, threads);
        else if (AdvSimd.Arm64.IsSupported)
            DepthwiseAdvSimd(input, packedWeights, bias, output, batch, channels, height, width, outputHeight, outputWidth,
                kernelH, kernelW, strideH, strideW, padTop, padLeft, residual, activation, alpha, beta, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            DepthwiseVec(input, packedWeights, bias, output, batch, channels, height, width, outputHeight, outputWidth,
                kernelH, kernelW, strideH, strideW, padTop, padLeft, residual, activation, alpha, beta, threads);
        else
            DepthwiseScalar(input, packedWeights, bias, output, batch, channels, height, width, outputHeight, outputWidth,
                kernelH, kernelW, strideH, strideW, padTop, padLeft, residual, activation, alpha, beta, threads);
    }

    /// <summary>
    /// 2x2 / stride-2 transposed convolution as four strided pointwise GEMMs
    /// (one per output tap). Weights packed [tap][oc/16][ic][16]
    /// (<see cref="PackConvTranspose2x2"/>); bias and activation fused.
    /// </summary>
    internal static void ConvTranspose2x2Stride2(ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias,
        Span<float> output, int batch, int inputChannels, int height, int width, int outputChannels,
        NhwcActivation activation, int threads)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            ConvTranspose2x2Stride2Avx(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels, activation, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            ConvTranspose2x2Stride2Avx(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels, activation, threads);
        else if (AdvSimd.Arm64.IsSupported)
            ConvTranspose2x2Stride2AdvSimd(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels, activation, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            ConvTranspose2x2Stride2Vec(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels, activation, threads);
        else
            ConvTranspose2x2Stride2Scalar(input, packedWeights, bias, output, batch, inputChannels, height, width, outputChannels, activation, threads);
    }

    /// <summary>[n][c][plane] -> [n][plane][c].</summary>
    internal static void NchwToNhwc(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            NchwToNhwcAvx(source, destination, batch, channels, plane, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            NchwToNhwcAvx(source, destination, batch, channels, plane, threads);
        else if (AdvSimd.Arm64.IsSupported)
            NchwToNhwcAdvSimd(source, destination, batch, channels, plane, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            NchwToNhwcVec(source, destination, batch, channels, plane, threads);
        else
            NchwToNhwcScalar(source, destination, batch, channels, plane, threads);
    }

    /// <summary>[n][plane][c] -> [n][c][plane].</summary>
    internal static void NhwcToNchw(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            NhwcToNchwAvx(source, destination, batch, channels, plane, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            NhwcToNchwAvx(source, destination, batch, channels, plane, threads);
        else if (AdvSimd.Arm64.IsSupported)
            NhwcToNchwAdvSimd(source, destination, batch, channels, plane, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            NhwcToNchwVec(source, destination, batch, channels, plane, threads);
        else
            NhwcToNchwScalar(source, destination, batch, channels, plane, threads);
    }

    /// <summary>Max / average pooling (average divides by the number of in-bounds taps, as the NCHW path does).</summary>
    internal static void Pool(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft, bool max,
        int threads = 1)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            PoolAvx(input, output, batch, channels, height, width, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft, max, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            PoolAvx(input, output, batch, channels, height, width, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft, max, threads);
        else if (AdvSimd.Arm64.IsSupported)
            PoolAdvSimd(input, output, batch, channels, height, width, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft, max, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            PoolVec(input, output, batch, channels, height, width, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft, max, threads);
        else
            PoolScalar(input, output, batch, channels, height, width, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, padTop, padLeft, max, threads);
    }

    /// <summary>Nearest-neighbour integer upsampling: each input pixel vector is repeated factorW times, each row factorH times.</summary>
    internal static void ResizeNearest(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int factorH, int factorW, int threads = 1)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            ResizeNearestAvx(input, output, batch, channels, height, width, factorH, factorW, threads);
        else if (Avx2.IsSupported && Fma.IsSupported)
            ResizeNearestAvx(input, output, batch, channels, height, width, factorH, factorW, threads);
        else if (AdvSimd.Arm64.IsSupported)
            ResizeNearestAdvSimd(input, output, batch, channels, height, width, factorH, factorW, threads);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            ResizeNearestVec(input, output, batch, channels, height, width, factorH, factorW, threads);
        else
            ResizeNearestScalar(input, output, batch, channels, height, width, factorH, factorW, threads);
    }

    /// <summary>Spatial mean per (batch, channel): output is [n][c] (== NCHW [n,c,1,1]).</summary>
    internal static void ReduceMeanSpatial(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int plane)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            ReduceMeanSpatialAvx(input, output, batch, channels, plane);
        else if (Avx2.IsSupported && Fma.IsSupported)
            ReduceMeanSpatialAvx(input, output, batch, channels, plane);
        else if (AdvSimd.Arm64.IsSupported)
            ReduceMeanSpatialAdvSimd(input, output, batch, channels, plane);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            ReduceMeanSpatialVec(input, output, batch, channels, plane);
        else
            ReduceMeanSpatialScalar(input, output, batch, channels, plane);
    }

    /// <summary>
    /// Binary op between an NHWC activation [n][plane][c] and a channel vector
    /// ([c] shared, or [n][c] when <paramref name="channelPerBatch"/>).
    /// </summary>
    internal static void BinaryChannel<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> channel, Span<float> output,
        int batch, int channels, int plane, bool channelIsLeft, bool channelPerBatch) where TOp : struct, IBinaryOp
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            BinaryChannelAvx<TOp>(left, channel, output, batch, channels, plane, channelIsLeft, channelPerBatch);
        else if (Avx2.IsSupported && Fma.IsSupported)
            BinaryChannelAvx<TOp>(left, channel, output, batch, channels, plane, channelIsLeft, channelPerBatch);
        else if (AdvSimd.Arm64.IsSupported)
            BinaryChannelAdvSimd<TOp>(left, channel, output, batch, channels, plane, channelIsLeft, channelPerBatch);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            BinaryChannelVec<TOp>(left, channel, output, batch, channels, plane, channelIsLeft, channelPerBatch);
        else
            BinaryChannelScalar<TOp>(left, channel, output, batch, channels, plane, channelIsLeft, channelPerBatch);
    }

    /// <summary>Inference batch-norm: (x - mean) * scale / sqrt(var + eps) + bias, per channel.</summary>
    internal static void BatchNorm(ReadOnlySpan<float> input, Span<float> output, int pixels, int channels,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, float epsilon)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            BatchNormAvx(input, output, pixels, channels, scale, bias, mean, variance, epsilon);
        else if (Avx2.IsSupported && Fma.IsSupported)
            BatchNormAvx(input, output, pixels, channels, scale, bias, mean, variance, epsilon);
        else if (AdvSimd.Arm64.IsSupported)
            BatchNormAdvSimd(input, output, pixels, channels, scale, bias, mean, variance, epsilon);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<float>.Count is 8 or 4)
            BatchNormVec(input, output, pixels, channels, scale, bias, mean, variance, epsilon);
        else
            BatchNormScalar(input, output, pixels, channels, scale, bias, mean, variance, epsilon);
    }

    /// <summary>Packs dense [oc][ic][kh][kw] weights as [oc/16][ic][tap][16].</summary>
    internal static float[] PackDense(ReadOnlySpan<float> weights, int outputChannels, int inputChannels, int taps)
    {
        const int block16 = OutputChannelBlock;
        float[] packed = new float[checked(outputChannels * inputChannels * taps)];
        int blocks = outputChannels / block16;
        for (int block = 0; block < blocks; block++)
            for (int ci = 0; ci < inputChannels; ci++)
                for (int tap = 0; tap < taps; tap++)
                {
                    long dst = (((long)block * inputChannels + ci) * taps + tap) * block16;
                    for (int lane = 0; lane < block16; lane++)
                        packed[dst + lane] = weights[((block * block16 + lane) * inputChannels + ci) * taps + tap];
                }
        return packed;
    }

    /// <summary>
    /// Packs depthwise [c][1][kh][kw] weights as [c/32][tap][32] (the final
    /// block holds the channel tail, a multiple of 8, with that tail as its tap
    /// stride). Taps of one channel block are contiguous, so the kernel's tap
    /// walk never lands on the same L1 set as the activation rows.
    /// </summary>
    internal static float[] PackDepthwise(ReadOnlySpan<float> weights, int channels, int taps)
    {
        float[] packed = new float[checked(channels * taps)];
        for (int c0 = 0; c0 < channels; c0 += 32)
        {
            int block = Math.Min(32, channels - c0);
            int blockBase = c0 * taps;
            for (int lane = 0; lane < block; lane++)
                for (int tap = 0; tap < taps; tap++)
                    packed[blockBase + tap * block + lane] = weights[(c0 + lane) * taps + tap];
        }
        return packed;
    }

    /// <summary>Offset of tap <paramref name="tap"/> for channel <paramref name="c"/> in a <see cref="PackDepthwise"/> buffer.</summary>
    internal static int DepthwiseWeightOffset(int channels, int taps, int c, int tap)
    {
        int c0 = c & ~31, block = Math.Min(32, channels - c0);
        return c0 * taps + tap * block + (c - c0);
    }

    /// <summary>Packs ConvTranspose [ic][oc][2][2] weights as [tap][oc/16][ic][16] (or [tap][ic] when oc == 1).</summary>
    internal static float[] PackConvTranspose2x2(ReadOnlySpan<float> weights, int inputChannels, int outputChannels)
    {
        const int block16 = OutputChannelBlock;
        float[] packed = new float[checked(4 * inputChannels * outputChannels)];
        if (outputChannels == 1)
        {
            for (int tap = 0; tap < 4; tap++)
                for (int ci = 0; ci < inputChannels; ci++)
                    packed[tap * inputChannels + ci] = weights[ci * 4 + tap];
            return packed;
        }
        int blocks = outputChannels / block16;
        for (int tap = 0; tap < 4; tap++)
            for (int block = 0; block < blocks; block++)
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    long dst = (((long)tap * blocks + block) * inputChannels + ci) * block16;
                    for (int lane = 0; lane < block16; lane++)
                        packed[dst + lane] = weights[checked((int)(((long)ci * outputChannels + block * block16 + lane) * 4 + tap))];
                }
        return packed;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ActivateScalar(float v, NhwcActivation activation)
        => activation == NhwcActivation.Relu ? MathF.Max(v, 0f) : v;

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
}
