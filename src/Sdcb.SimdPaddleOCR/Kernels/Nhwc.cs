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
/// NCHW dimensions; the data is stored as [n][h][w][c]. The net10 build has
/// AVX2+FMA and AVX-512 implementations; the netstandard2.0 build uses
/// Vector&lt;float&gt; kernels (Nhwc.*.Vector.cs) with identical semantics.
/// </summary>
internal static partial class Nhwc
{
    /// <summary>Output-channel block width of the dense/pointwise micro-kernel.</summary>
    internal const int OutputChannelBlock = 16;

    // Tuning knobs (measured on Zen 3; see test KernelBench). 0 disables blocking.
    /// <summary>Input-channel block for the pointwise GEMM; the L2-streamed full panel measured best on Zen 3.</summary>
    internal static int PointwiseKc = 0;
    /// <summary>Pixel tiles per group sharing one weight panel walk.</summary>
    internal static int PointwiseGroupTiles = 8;
    /// <summary>Flattened (ic x tap) reduction block for dense convolutions: 512 x 16 x 4 B = 32 KiB of weights.</summary>
    internal static int DenseKc = 512;
    /// <summary>Depthwise kernels at least this tall accumulate one kernel row per pass instead of all taps in registers.</summary>
    internal static int DepthwiseStripMinKernelH = 6;

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

}
