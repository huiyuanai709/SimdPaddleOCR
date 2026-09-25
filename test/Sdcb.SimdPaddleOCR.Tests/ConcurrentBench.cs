using System.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Sdcb.SimdPaddleOCR;
using ImageSharpImage = SixLabors.ImageSharp.Image;

namespace Sdcb.SimdPaddleOCR.Tests;

/// <summary>
/// Times one shared <see cref="PaddleOcrAll"/> when several pages run at once.
/// <c>--concurrent-bench --input dir --pages 4 --count 16 --workers 4</c>
/// </summary>
static class ConcurrentBench
{
    public static int Run(string[] args)
    {
        string? input = null;
        int pages = 4, count = 16, workers = 4, warmup = 2;
        string model = "tiny";
        for (int i = 1; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {args[i]}");
            switch (args[i])
            {
                case "--input": input = Next(); break;
                case "--pages": pages = int.Parse(Next()); break;
                case "--count": count = int.Parse(Next()); break;
                case "--workers": workers = int.Parse(Next()); break;
                case "--warmup": warmup = int.Parse(Next()); break;
                case "--model": model = Next(); break;
                default: throw new ArgumentException($"unknown argument: {args[i]}");
            }
        }
        if (input is null) throw new ArgumentException("--input is required");
        string[] files = Directory.GetFiles(input, "*.jpg").OrderBy(x => x, StringComparer.OrdinalIgnoreCase).Take(count).ToArray();
        if (files.Length == 0) throw new InvalidOperationException("no jpg files");
        var decoded = new (byte[] Pixels, int W, int H)[files.Length];
        for (int i = 0; i < files.Length; i++)
        {
            using Image<Rgb24> image = ImageSharpImage.Load<Rgb24>(files[i]);
            byte[] pixels = new byte[checked(image.Width * image.Height * 3)];
            for (int y = 0; y < image.Height; y++)
                for (int x = 0; x < image.Width; x++)
                {
                    Rgb24 p = image[x, y];
                    int o = (y * image.Width + x) * 3;
                    pixels[o] = p.B; pixels[o + 1] = p.G; pixels[o + 2] = p.R;
                }
            decoded[i] = (pixels, image.Width, image.Height);
        }

        using PaddleOcrAll ocr = PaddleOcrAll.Load(BenchEngines.Bundle(model), new PaddleOcrOptions { LineWorkerCount = workers });
        string[] sequential = new string[decoded.Length];
        for (int i = 0; i < Math.Min(warmup, decoded.Length); i++)
            ocr.Run(decoded[i].Pixels, decoded[i].W, decoded[i].H, decoded[i].W * 3);

        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < decoded.Length; i++)
        {
            PaddleOcrResult result = ocr.Run(decoded[i].Pixels, decoded[i].W, decoded[i].H, decoded[i].W * 3);
            sequential[i] = result.Text;
        }
        sw.Stop();
        double seqMs = sw.Elapsed.TotalMilliseconds;

        string[] parallelTexts = new string[decoded.Length];
        sw.Restart();
        Parallel.For(0, decoded.Length, new ParallelOptions { MaxDegreeOfParallelism = pages }, i =>
        {
            PaddleOcrResult result = ocr.Run(decoded[i].Pixels, decoded[i].W, decoded[i].H, decoded[i].W * 3);
            parallelTexts[i] = result.Text;
        });
        sw.Stop();
        double parMs = sw.Elapsed.TotalMilliseconds;
        int mismatch = 0;
        for (int i = 0; i < decoded.Length; i++)
            if (sequential[i] != parallelTexts[i]) mismatch++;

        Console.WriteLine($"cpu={Environment.ProcessorCount} workers={ocr.EffectiveLineWorkerCount} pages={pages} n={decoded.Length}");
        Console.WriteLine($"sequential {seqMs / decoded.Length,7:F2} ms/img   wall {seqMs:F0} ms");
        Console.WriteLine($"parallel   {parMs / decoded.Length,7:F2} ms/img   wall {parMs:F0} ms   text mismatches={mismatch}");
        return mismatch == 0 ? 0 : 1;
    }
}
