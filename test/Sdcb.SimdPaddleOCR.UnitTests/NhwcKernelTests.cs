using System.Numerics;
using Sdcb.SimdPaddleOCR.Kernels;
using static Sdcb.SimdPaddleOCR.UnitTests.KernelCorrectnessTests;

namespace Sdcb.SimdPaddleOCR.UnitTests;

/// <summary>Channels-last kernels against a scalar NCHW reference. Same gate as LayoutPlanner (skip only when PPOCR_NHWC=0).</summary>
public class NhwcKernelTests
{
    private static bool Supported =>
        Environment.GetEnvironmentVariable("PPOCR_NHWC") != "0";

    private static float[] Rand(int length, int seed, float scale = 1f)
    {
        Random random = new(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        return values;
    }

    private static float[] ToNhwc(float[] nchw, int n, int c, int plane)
    {
        float[] result = new float[nchw.Length];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < plane; p++)
                    result[(b * plane + p) * c + ch] = nchw[(b * c + ch) * plane + p];
        return result;
    }

    private static float[] ToNchw(float[] nhwc, int n, int c, int plane)
    {
        float[] result = new float[nhwc.Length];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < plane; p++)
                    result[(b * c + ch) * plane + p] = nhwc[(b * plane + p) * c + ch];
        return result;
    }

    private static float Activate(float v, NhwcActivation activation, float alpha, float beta) => activation switch
    {
        NhwcActivation.Relu => MathF.Max(v, 0f),
        NhwcActivation.HardSwish => v * Math.Clamp(alpha * v + beta, 0f, 1f),
        _ => v,
    };

    // NCHW reference: bias-initialised, input channels ascending then taps.
    private static float[] ConvRef(float[] input, float[] weights, float[] bias, int n, int cin, int h, int w,
        int cout, int groups, int kh, int kw, int sh, int sw, int pt, int pl, int oh, int ow,
        float[]? residualNchw = null, NhwcActivation activation = NhwcActivation.None, float alpha = 0, float beta = 0)
    {
        int cpg = cin / groups, opg = cout / groups;
        float[] output = new float[n * cout * oh * ow];
        for (int b = 0; b < n; b++)
            for (int co = 0; co < cout; co++)
            {
                int g = co / opg;
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        float sum = bias.Length == 0 ? 0f : bias[co];
                        for (int ci = 0; ci < cpg; ci++)
                            for (int ky = 0; ky < kh; ky++)
                            {
                                int iy = y * sh - pt + ky;
                                if ((uint)iy >= (uint)h) continue;
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int ix = x * sw - pl + kx;
                                    if ((uint)ix >= (uint)w) continue;
                                    sum += input[((b * cin + g * cpg + ci) * h + iy) * w + ix] *
                                        weights[((co * cpg + ci) * kh + ky) * kw + kx];
                                }
                            }
                        int index = ((b * cout + co) * oh + y) * ow + x;
                        if (residualNchw is not null) sum += residualNchw[index];
                        output[index] = Activate(sum, activation, alpha, beta);
                    }
            }
        return output;
    }

    [Theory]
    [InlineData(1, 3, 5, 7)]
    [InlineData(2, 19, 4, 11)]
    [InlineData(1, 64, 6, 37)]
    public void LayoutConversion_RoundTrip(int n, int c, int h, int w)
    {
        if (!Supported) return;
        int plane = h * w;
        float[] nchw = Rand(n * c * plane, 1);
        float[] nhwc = new float[nchw.Length];
        Nhwc.NchwToNhwc(nchw, nhwc, n, c, plane, 3);
        Assert.Equal(ToNhwc(nchw, n, c, plane), nhwc);
        float[] back = new float[nchw.Length];
        Nhwc.NhwcToNchw(nhwc, back, n, c, plane, 3);
        Assert.Equal(nchw, back);
    }

    [Theory]
    [InlineData(1, 40, 7, 32, false, 0, 0)]
    [InlineData(1, 16, 6, 8, false, 1, 0)]       // 8-channel tail, no 16-wide panel
    [InlineData(1, 24, 5, 24, true, 2, 7)]       // 16-wide panel plus 8-channel tail, K blocked
    [InlineData(2, 300, 13, 48, true, 1, 0)]
    [InlineData(1, 520, 6, 16, true, 2, 0)]
    [InlineData(1, 64, 100, 64, false, 0, 64)]
    public void Pointwise_MatchesReference(int n, int cin, int plane, int cout, bool residual, int activationId, int kc)
    {
        if (!Supported) return;
        NhwcActivation activation = (NhwcActivation)activationId;
        int saved = Nhwc.PointwiseKc;
        Nhwc.PointwiseKc = kc == 0 ? saved : kc;
        try
        {
            float[] input = Rand(n * cin * plane, 2), weights = Rand(cout * cin, 3, 0.1f), bias = Rand(cout, 4);
            float[]? res = residual ? Rand(n * cout * plane, 5) : null;
            const float alpha = 1f / 6f, beta = 0.5f;
            float[] expected = ConvRef(input, weights, bias, n, cin, 1, plane, cout, 1, 1, 1, 1, 1, 0, 0, 1, plane, res, activation, alpha, beta);
            float[] packed = Nhwc.PackDense(weights, cout, cin, 1);
            float[] actual = new float[expected.Length];
            float[] nhwcIn = ToNhwc(input, n, cin, plane);
            ReadOnlySpan<float> nhwcRes = res is null ? [] : ToNhwc(res, n, cout, plane);
            Nhwc.Pointwise(nhwcIn, packed, bias, actual, n * plane, cin, cout,
                nhwcRes, activation, alpha, beta, 3);
            AssertClose(expected, ToNchw(actual, n, cout, plane), 1e-4f, 1e-4f);
            float[] scalar = new float[expected.Length];
            Nhwc.PointwiseScalar(nhwcIn, packed, bias, scalar, n * plane, cin, cout,
                nhwcRes, activation, alpha, beta, 3);
            AssertClose(expected, ToNchw(scalar, n, cout, plane), 1e-4f, 1e-4f);
        }
        finally
        {
            Nhwc.PointwiseKc = saved;
        }
    }

    [Theory]
    [InlineData(1, 3, 11, 13, 16, 3, 3, 2, 2, 1, 1)]     // stem 3x3 s2
    [InlineData(1, 20, 9, 14, 32, 3, 3, 1, 1, 1, 1)]     // 3x3 s1
    [InlineData(1, 16, 9, 11, 8, 2, 2, 1, 1, 0, 0)]      // 16→8 2x2, the det layout-convert case
    [InlineData(1, 16, 8, 10, 24, 3, 3, 1, 1, 1, 1)]     // 16-wide panel plus 8-channel tail
    [InlineData(2, 8, 7, 9, 16, 2, 2, 1, 1, 0, 0)]       // 2x2 pad end
    [InlineData(1, 12, 10, 12, 32, 7, 7, 1, 1, 3, 3)]    // 7x7
    [InlineData(1, 12, 8, 13, 16, 1, 7, 1, 1, 0, 3)]     // 1x7
    [InlineData(1, 5, 9, 9, 16, 5, 5, 2, 2, 2, 2)]       // 5x5 s2
    public void Dense_MatchesReference(int n, int cin, int h, int w, int cout, int kh, int kw, int sh, int sw, int pt, int pl)
    {
        if (!Supported) return;
        // ONNX output size with pads (pt, pl, pb, pr); use symmetric or pad-end conventions.
        int pb = kh - 1 - pt, pr = kw - 1 - pl;
        int oh = (h + pt + pb - kh) / sh + 1, ow = (w + pl + pr - kw) / sw + 1;
        float[] input = Rand(n * cin * h * w, 6), weights = Rand(cout * cin * kh * kw, 7, 0.1f), bias = Rand(cout, 8);
        float[] res = Rand(n * cout * oh * ow, 9);
        float[] expected = ConvRef(input, weights, bias, n, cin, h, w, cout, 1, kh, kw, sh, sw, pt, pl, oh, ow, res, NhwcActivation.Relu);
        float[] packed = Nhwc.PackDense(weights, cout, cin, kh * kw);
        float[] actual = new float[expected.Length];
        float[] nhwcIn = ToNhwc(input, n, cin, h * w);
        float[] nhwcRes = ToNhwc(res, n, cout, oh * ow);
        Nhwc.Dense(nhwcIn, packed, bias, actual, n, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl,
            nhwcRes, NhwcActivation.Relu, 0, 0, 3);
        AssertClose(expected, ToNchw(actual, n, cout, oh * ow), 1e-4f, 1e-4f);
        float[] scalar = new float[expected.Length];
        Nhwc.DenseScalar(nhwcIn, packed, bias, scalar, n, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl,
            nhwcRes, NhwcActivation.Relu, 0, 0, 3);
        AssertClose(expected, ToNchw(scalar, n, cout, oh * ow), 1e-4f, 1e-4f);

        // Force several flattened-K blocks that straddle tap boundaries.
        int savedKc = Nhwc.DenseKc;
        Nhwc.DenseKc = 40;
        try
        {
            float[] blocked = new float[expected.Length];
            Nhwc.Dense(nhwcIn, packed, bias, blocked, n, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl,
                nhwcRes, NhwcActivation.Relu, 0, 0, 3);
            AssertClose(expected, ToNchw(blocked, n, cout, oh * ow), 1e-4f, 1e-4f);
            float[] blockedScalar = new float[expected.Length];
            Nhwc.DenseScalar(nhwcIn, packed, bias, blockedScalar, n, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl,
                nhwcRes, NhwcActivation.Relu, 0, 0, 3);
            AssertClose(expected, ToNchw(blockedScalar, n, cout, oh * ow), 1e-4f, 1e-4f);
        }
        finally
        {
            Nhwc.DenseKc = savedKc;
        }
    }

    [Theory]
    [InlineData(1, 8, 9, 11, 3, 3, 1, 1, 1, 1)]
    [InlineData(2, 40, 7, 13, 3, 3, 2, 1, 1, 1)]
    [InlineData(1, 24, 6, 10, 3, 3, 2, 2, 1, 1)]
    [InlineData(1, 32, 12, 14, 9, 9, 1, 1, 4, 4)]
    [InlineData(1, 16, 3, 20, 1, 7, 1, 1, 0, 3)]
    [InlineData(1, 8, 2, 3, 5, 5, 1, 1, 2, 2)]           // plane smaller than kernel
    public void Depthwise_MatchesReference(int n, int c, int h, int w, int kh, int kw, int sh, int sw, int pt, int pl)
    {
        if (!Supported) return;
        int oh = (h + 2 * pt - kh) / sh + 1, ow = (w + 2 * pl - kw) / sw + 1;
        float[] input = Rand(n * c * h * w, 10), weights = Rand(c * kh * kw, 11, 0.2f), bias = Rand(c, 12);
        float[] res = Rand(n * c * oh * ow, 13);
        const float alpha = 1f / 6f, beta = 0.5f;
        float[] expected = ConvRef(input, weights, bias, n, c, h, w, c, c, kh, kw, sh, sw, pt, pl, oh, ow, res, NhwcActivation.HardSwish, alpha, beta);
        float[] packed = Nhwc.PackDepthwise(weights, c, kh * kw);
        float[] actual = new float[expected.Length];
        float[] nhwcIn = ToNhwc(input, n, c, h * w);
        float[] nhwcRes = ToNhwc(res, n, c, oh * ow);
        Nhwc.Depthwise(nhwcIn, packed, bias, actual, n, c, h, w, oh, ow, kh, kw, sh, sw, pt, pl,
            nhwcRes, NhwcActivation.HardSwish, alpha, beta, 3);
        AssertClose(expected, ToNchw(actual, n, c, oh * ow), 1e-4f, 1e-4f);
        float[] scalar = new float[expected.Length];
        Nhwc.DepthwiseScalar(nhwcIn, packed, bias, scalar, n, c, h, w, oh, ow, kh, kw, sh, sw, pt, pl,
            nhwcRes, NhwcActivation.HardSwish, alpha, beta, 3);
        AssertClose(expected, ToNchw(scalar, n, c, oh * ow), 1e-4f, 1e-4f);
    }

    [Theory]
    [InlineData(1, 24, 5, 7, 16)]
    [InlineData(2, 64, 4, 9, 1)]
    [InlineData(1, 12, 3, 13, 32)]
    public void ConvTranspose2x2Stride2_MatchesReference(int n, int cin, int h, int w, int cout)
    {
        if (!Supported) return;
        int oh = h * 2, ow = w * 2;
        float[] input = Rand(n * cin * h * w, 14), weights = Rand(cin * cout * 4, 15, 0.1f), bias = Rand(cout, 16);
        float[] expected = new float[n * cout * oh * ow];
        for (int b = 0; b < n; b++)
            for (int ci = 0; ci < cin; ci++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        float v = input[((b * cin + ci) * h + y) * w + x];
                        for (int co = 0; co < cout; co++)
                            for (int ky = 0; ky < 2; ky++)
                                for (int kx = 0; kx < 2; kx++)
                                    expected[((b * cout + co) * oh + 2 * y + ky) * ow + 2 * x + kx] += v * weights[((ci * cout + co) * 2 + ky) * 2 + kx];
                    }
        for (int i = 0; i < expected.Length; i++)
        {
            int co = i / (oh * ow) % cout;
            expected[i] = MathF.Max(expected[i] + bias[co], 0f);
        }
        float[] packed = Nhwc.PackConvTranspose2x2(weights, cin, cout);
        float[] actual = new float[expected.Length];
        Nhwc.ConvTranspose2x2Stride2(ToNhwc(input, n, cin, h * w), packed, bias, actual, n, cin, h, w, cout, NhwcActivation.Relu, 3);
        AssertClose(expected, ToNchw(actual, n, cout, oh * ow), 1e-4f, 1e-4f);
    }

    [Theory]
    [InlineData(true, 2, 2, 1, 1, 0, 0)]    // 2x2 s1 pad-end (LCNet stem)
    [InlineData(true, 3, 3, 2, 2, 1, 1)]
    [InlineData(false, 3, 2, 3, 2, 0, 0)]   // rec neck average pool
    [InlineData(false, 3, 3, 1, 1, 1, 1)]
    public void Pool_MatchesReference(bool max, int kh, int kw, int sh, int sw, int pt, int pl)
    {
        if (!Supported) return;
        const int n = 2, c = 12, h = 9, w = 11;
        int pb = max && sh == 1 ? kh - 1 - pt : pt, pr = max && sw == 1 ? kw - 1 - pl : pl;
        int oh = (h + pt + pb - kh) / sh + 1, ow = (w + pl + pr - kw) / sw + 1;
        float[] input = Rand(n * c * h * w, 17);
        float[] expected = new float[n * c * oh * ow];
        for (int b = 0; b < n; b++) for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
            {
                float z = max ? float.NegativeInfinity : 0; int count = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    int iy = y * sh - pt + ky; if ((uint)iy >= (uint)h) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int ix = x * sw - pl + kx; if ((uint)ix >= (uint)w) continue;
                        float v = input[((b * c + ch) * h + iy) * w + ix];
                        if (max) z = MathF.Max(z, v); else { z += v; count++; }
                    }
                }
                expected[((b * c + ch) * oh + y) * ow + x] = max ? z : z / count;
            }
        float[] actual = new float[expected.Length];
        Nhwc.Pool(ToNhwc(input, n, c, h * w), actual, n, c, h, w, oh, ow, kh, kw, sh, sw, pt, pl, max);
        AssertClose(expected, ToNchw(actual, n, c, oh * ow));
    }

    [Fact]
    public void ResizeNearest_ReduceMean_BinaryChannel_BatchNorm_MatchReference()
    {
        if (!Supported) return;
        const int n = 2, c = 20, h = 3, w = 5;
        float[] input = Rand(n * c * h * w, 18);
        float[] nhwc = ToNhwc(input, n, c, h * w);

        float[] resized = new float[n * c * h * 2 * w * 4];
        Nhwc.ResizeNearest(nhwc, resized, n, c, h, w, 2, 4);
        float[] resizedNchw = ToNchw(resized, n, c, h * 2 * w * 4);
        for (int b = 0; b < n; b++) for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < h * 2; y++) for (int x = 0; x < w * 4; x++)
                Assert.Equal(input[((b * c + ch) * h + y / 2) * w + x / 4], resizedNchw[((b * c + ch) * h * 2 + y) * w * 4 + x]);

        float[] mean = new float[n * c];
        Nhwc.ReduceMeanSpatial(nhwc, mean, n, c, h * w);
        for (int b = 0; b < n; b++) for (int ch = 0; ch < c; ch++)
        {
            float sum = 0;
            for (int p = 0; p < h * w; p++) sum += input[(b * c + ch) * h * w + p];
            Assert.InRange(mean[b * c + ch], sum / (h * w) - 1e-5f, sum / (h * w) + 1e-5f);
        }

        float[] channel = Rand(n * c, 19), product = new float[nhwc.Length], quotient = new float[nhwc.Length];
        Nhwc.BinaryChannel<MulOp>(nhwc, channel, product, n, c, h * w, channelIsLeft: false, channelPerBatch: true);
        Nhwc.BinaryChannel<DivOp>(nhwc, channel, quotient, n, c, h * w, channelIsLeft: true, channelPerBatch: false);
        float[] productNchw = ToNchw(product, n, c, h * w), quotientNchw = ToNchw(quotient, n, c, h * w);
        for (int b = 0; b < n; b++) for (int ch = 0; ch < c; ch++) for (int p = 0; p < h * w; p++)
        {
            int i = (b * c + ch) * h * w + p;
            Assert.Equal(input[i] * channel[b * c + ch], productNchw[i]);
            Assert.Equal(channel[ch] / input[i], quotientNchw[i]);
        }

        float[] scale = Rand(c, 20), bias = Rand(c, 21), mu = Rand(c, 22), variance = Rand(c, 23, 0.5f);
        for (int i = 0; i < c; i++) variance[i] = MathF.Abs(variance[i]) + 0.1f;
        float[] normalized = new float[nhwc.Length];
        Nhwc.BatchNorm(nhwc, normalized, n * h * w, c, scale, bias, mu, variance, 1e-5f);
        float[] normalizedNchw = ToNchw(normalized, n, c, h * w);
        for (int b = 0; b < n; b++) for (int ch = 0; ch < c; ch++) for (int p = 0; p < h * w; p++)
        {
            int i = (b * c + ch) * h * w + p;
            float k = scale[ch] / MathF.Sqrt(variance[ch] + 1e-5f);
            Assert.InRange(normalizedNchw[i], (input[i] - mu[ch]) * k + bias[ch] - 1e-5f, (input[i] - mu[ch]) * k + bias[ch] + 1e-5f);
        }
    }
}
