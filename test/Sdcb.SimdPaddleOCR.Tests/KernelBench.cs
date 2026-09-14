using System.Diagnostics;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.Tests;

/// <summary>
/// `--kernel-bench [threads] [repeats]`: times the channels-last kernels on
/// the medium detector's hot shapes in isolation (no pipeline noise).
/// </summary>
static class KernelBench
{
    public static int Run(string[] args)
    {
        int threads = args.Length > 1 ? int.Parse(args[1]) : 8;
        int repeats = args.Length > 2 ? int.Parse(args[2]) : 20;
        Console.WriteLine($"kernel bench threads={threads} repeats={repeats} (median ms, GFLOPS)");

        Dense("3x3 256->64 @240x208", 256, 240, 208, 64, 3, 3, 1, 1, 1, 1, threads, repeats);
        Dense("3x3s2 128->64 @480x416", 128, 480, 416, 64, 3, 3, 2, 2, 1, 1, threads, repeats);
        Dense("7x7 32->32 @240x216", 32, 240, 216, 32, 7, 7, 1, 1, 3, 3, threads, repeats);
        Dense("2x2 32->64 @480x432", 32, 480, 432, 64, 2, 2, 1, 1, 0, 0, threads, repeats);
        Dense("3x3s2 3->64 @960x864", 3, 960, 864, 64, 3, 3, 2, 2, 1, 1, threads, repeats);
        Pointwise("1x1 128->256 @240x216", 128, 240 * 216, 256, threads, repeats);
        Pointwise("1x1 1024->512 @6x320", 1024, 6 * 320, 512, threads, repeats);
        Pointwise("1x1 512->1024 @6x320", 512, 6 * 320, 1024, threads, repeats);
        Depthwise("dw9x9 256 @240x208", 256, 240, 208, 9, 9, 1, 1, 4, 4, threads, repeats);
        Depthwise("dw3x3 512 @6x320", 512, 6, 320, 3, 3, 1, 1, 1, 1, threads, repeats);
        Depthwise("dw3x3 128 @240x216", 128, 240, 216, 3, 3, 1, 1, 1, 1, threads, repeats);
        return 0;
    }

    private static float[] Rand(int length, int seed)
    {
        Random random = new(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static void Report(string name, List<double> times, double flops)
    {
        times.Sort();
        double median = times[times.Count / 2];
        Console.WriteLine($"{name,-28} {median,8:F3} ms  {flops / median / 1e6,8:F1} GFLOPS  (min {times[0]:F3})");
    }

    private static void Dense(string name, int cin, int h, int w, int cout, int kh, int kw, int sh, int sw, int pt, int pl,
        int threads, int repeats)
    {
        int pb = kh - 1 - pt, pr = kw - 1 - pl;
        int oh = (h + pt + pb - kh) / sh + 1, ow = (w + pl + pr - kw) / sw + 1;
        float[] input = Rand(cin * h * w, 1), weights = Rand(cout * cin * kh * kw, 2), bias = Rand(cout, 3);
        float[] packed = Nhwc.PackDense(weights, cout, cin, kh * kw);
        float[] output = new float[cout * oh * ow];
        int[] variants = [0, 1024, 512, 256];
        List<double>[] times = variants.Select(_ => new List<double>()).ToArray();
        int saved = Nhwc.DenseKc;
        for (int r = 0; r < repeats + 2; r++)
            for (int v = 0; v < variants.Length; v++)
            {
                Nhwc.DenseKc = variants[v];
                Stopwatch sw2 = Stopwatch.StartNew();
                Nhwc.Dense(input, packed, bias, output, 1, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl, [], NhwcActivation.Relu, 0, 0, threads);
                sw2.Stop();
                if (r >= 2) times[v].Add(sw2.Elapsed.TotalMilliseconds);
            }
        Nhwc.DenseKc = saved;
        for (int v = 0; v < variants.Length; v++)
            Report($"{name} kc={variants[v]}", times[v], 2.0 * cin * cout * kh * kw * oh * ow);
    }

    private static void Pointwise(string name, int cin, int pixels, int cout, int threads, int repeats)
    {
        float[] input = Rand(cin * pixels, 4), weights = Rand(cout * cin, 5), bias = Rand(cout, 6);
        float[] packed = Nhwc.PackDense(weights, cout, cin, 1);
        float[] output = new float[cout * pixels];
        (int Kc, int Group)[] variants = [(0, 8), (256, 8), (256, 16), (128, 16), (512, 4)];
        List<double>[] times = variants.Select(_ => new List<double>()).ToArray();
        (int savedKc, int savedGroup) = (Nhwc.PointwiseKc, Nhwc.PointwiseGroupTiles);
        for (int r = 0; r < repeats + 2; r++)
            for (int v = 0; v < variants.Length; v++)
            {
                (Nhwc.PointwiseKc, Nhwc.PointwiseGroupTiles) = variants[v];
                Stopwatch sw = Stopwatch.StartNew();
                Nhwc.Pointwise(input, packed, bias, output, pixels, cin, cout, [], NhwcActivation.None, 0, 0, threads);
                sw.Stop();
                if (r >= 2) times[v].Add(sw.Elapsed.TotalMilliseconds);
            }
        (Nhwc.PointwiseKc, Nhwc.PointwiseGroupTiles) = (savedKc, savedGroup);
        for (int v = 0; v < variants.Length; v++)
            Report($"{name} kc={variants[v].Kc} g={variants[v].Group}", times[v], 2.0 * cin * cout * pixels);
    }

    private static void Depthwise(string name, int c, int h, int w, int kh, int kw, int sh, int sw, int pt, int pl, int threads, int repeats)
    {
        int oh = (h + 2 * pt - kh) / sh + 1, ow = (w + 2 * pl - kw) / sw + 1;
        float[] input = Rand(c * h * w, 7), weights = Rand(c * kh * kw, 8), bias = Rand(c, 9);
        float[] packed = Nhwc.PackDepthwise(weights, c, kh * kw);
        float[] output = new float[c * oh * ow];
        List<double> times = [], naiveTimes = [];
        int saved = Nhwc.DepthwiseStripMinKernelH;
        for (int r = 0; r < repeats + 2; r++)
        {
            Nhwc.DepthwiseStripMinKernelH = 6;
            Stopwatch sw2 = Stopwatch.StartNew();
            Nhwc.Depthwise(input, packed, bias, output, 1, c, h, w, oh, ow, kh, kw, sh, sw, pt, pl, [], NhwcActivation.None, 0, 0, threads);
            sw2.Stop();
            if (r >= 2) times.Add(sw2.Elapsed.TotalMilliseconds);
            Nhwc.DepthwiseStripMinKernelH = 100;
            sw2.Restart();
            Nhwc.Depthwise(input, packed, bias, output, 1, c, h, w, oh, ow, kh, kw, sh, sw, pt, pl, [], NhwcActivation.None, 0, 0, threads);
            sw2.Stop();
            if (r >= 2) naiveTimes.Add(sw2.Elapsed.TotalMilliseconds);
        }
        Nhwc.DepthwiseStripMinKernelH = saved;
        Report(name, times, 2.0 * c * kh * kw * oh * ow);
        Report(name + " regs", naiveTimes, 2.0 * c * kh * kw * oh * ow);
    }
}
