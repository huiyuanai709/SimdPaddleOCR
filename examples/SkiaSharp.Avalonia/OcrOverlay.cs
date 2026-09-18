using System;
using System.Collections.Generic;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;

namespace SkiaSharp.Avalonia;

// Standalone in-place OCR overlay. Usage:
//   SKBitmap overlay = OcrOverlay.Draw(source, lines);
internal static class OcrOverlay
{
    const string FontFamilyName = "Microsoft YaHei UI";

    public static SKBitmap Draw(SKBitmap source, IEnumerable<PaddleOcrLine>? lines)
    {
        if (source is null)
            throw new ArgumentNullException(nameof(source));

        SKBitmap dest = source.Copy()
            ?? throw new InvalidOperationException("Unable to copy the source bitmap.");
        if (lines is null)
            return dest;

        using SKCanvas canvas = new(dest);
        foreach (PaddleOcrLine line in lines)
            DrawLine(dest, canvas, line);
        canvas.Flush();
        return dest;
    }

    static void DrawLine(SKBitmap bitmap, SKCanvas canvas, PaddleOcrLine line)
    {
        if (!TryParseBox(line.Box, out SKPoint[] pts))
            return;
        if (!TryQuadGeometry(pts, out QuadGeometry geom) || geom.Len < 2 || geom.Ht < 2)
            return;

        List<int[]> pixels = CollectInterior(bitmap, pts);
        Rgb background = ModeBackground(pixels);
        Rgb? ink = PickInkColor(pixels, background);
        FillQuad(canvas, pts, background);

        string text = line.Text ?? "";
        if (text.Length == 0)
            return;

        using SKTypeface typeface = SKTypeface.FromFamilyName(FontFamilyName, SKFontStyle.Bold)
            ?? SKTypeface.CreateDefault();
        float size = FitFontSize(typeface, text, geom.Len * 0.94f, geom.Ht * 0.9f);
        using SKFont font = new(typeface, size);
        using SKPaint paint = new()
        {
            IsAntialias = true,
            Color = ToColor(ink ?? ContrastInk(background))
        };
        font.GetFontMetrics(out SKFontMetrics metrics);
        float baseline = -(metrics.Ascent + metrics.Descent) / 2f;
        canvas.Save();
        canvas.Translate(geom.Cx, geom.Cy);
        canvas.RotateDegrees(geom.AngleDegrees);
        canvas.DrawText(text, 0, baseline, SKTextAlign.Center, font, paint);
        canvas.Restore();
    }

    static bool TryParseBox(PaddleOcrDetectionBox box, out SKPoint[] pts)
    {
        pts =
        [
            new SKPoint(box.X1, box.Y1),
            new SKPoint(box.X2, box.Y2),
            new SKPoint(box.X3, box.Y3),
            new SKPoint(box.X4, box.Y4)
        ];
        for (int i = 0; i < 4; i++)
        {
            if (!IsFinite(pts[i].X) || !IsFinite(pts[i].Y))
                return false;
        }

        return true;
    }

    static bool TryQuadGeometry(SKPoint[] pts, out QuadGeometry geom)
    {
        SKPoint origin = pts[0];
        SKPoint along = pts[1];
        float len = Dist(origin, along);
        float ht = Dist(origin, pts[3]);
        if (ht > len)
        {
            along = pts[3];
            float swap = len;
            len = ht;
            ht = swap;
        }

        geom = new QuadGeometry(
            (pts[0].X + pts[1].X + pts[2].X + pts[3].X) / 4f,
            (pts[0].Y + pts[1].Y + pts[2].Y + pts[3].Y) / 4f,
            len,
            ht,
            (float)(Math.Atan2(along.Y - origin.Y, along.X - origin.X) * 180.0 / Math.PI));
        return true;
    }

    static float Dist(SKPoint a, SKPoint b)
    {
        float dx = b.X - a.X;
        float dy = b.Y - a.Y;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    static float Cross(SKPoint origin, SKPoint a, SKPoint b) =>
        (a.X - origin.X) * (b.Y - origin.Y) - (a.Y - origin.Y) * (b.X - origin.X);

    static bool PointInQuad(float x, float y, SKPoint[] quad)
    {
        int sign = 0;
        SKPoint point = new(x, y);
        for (int i = 0; i < 4; i++)
        {
            float value = Cross(quad[i], quad[(i + 1) % 4], point);
            if (value == 0)
                continue;
            int next = Math.Sign(value);
            if (sign == 0)
                sign = next;
            else if (next != sign)
                return false;
        }

        return true;
    }

    static unsafe List<int[]> CollectInterior(SKBitmap bitmap, SKPoint[] pts)
    {
        float minXf = pts[0].X, minYf = pts[0].Y, maxXf = pts[0].X, maxYf = pts[0].Y;
        for (int i = 1; i < 4; i++)
        {
            minXf = Math.Min(minXf, pts[i].X);
            minYf = Math.Min(minYf, pts[i].Y);
            maxXf = Math.Max(maxXf, pts[i].X);
            maxYf = Math.Max(maxYf, pts[i].Y);
        }

        int minX = Math.Max(0, (int)Math.Floor(minXf - 1));
        int minY = Math.Max(0, (int)Math.Floor(minYf - 1));
        int maxX = Math.Min(bitmap.Width, (int)Math.Ceiling(maxXf + 1));
        int maxY = Math.Min(bitmap.Height, (int)Math.Ceiling(maxYf + 1));
        int width = maxX - minX;
        int height = maxY - minY;
        List<int[]> pixels = [];
        if (width < 1 || height < 1)
            return pixels;

        SKPixmap pixmap = bitmap.PeekPixels()
            ?? throw new InvalidOperationException("Unable to read bitmap pixels.");
        IntPtr addr = pixmap.GetPixels();
        if (addr == IntPtr.Zero)
            return pixels;

        int stride = pixmap.RowBytes;
        byte* scan0 = (byte*)addr;
        for (int y = 0; y < height; y++)
        {
            byte* row = scan0 + (minY + y) * stride;
            for (int x = 0; x < width; x++)
            {
                if (!PointInQuad(minX + x + 0.5f, minY + y + 0.5f, pts))
                    continue;
                byte* px = row + (minX + x) * 4;
                pixels.Add([px[2], px[1], px[0]]);
            }
        }

        return pixels;
    }

    static Rgb ModeBackground(List<int[]> pixels)
    {
        if (pixels.Count == 0)
            return new Rgb(255, 255, 255);

        Dictionary<int, Bucket> buckets = [];
        foreach (int[] pixel in pixels)
        {
            int key = ((pixel[0] >> 4) << 8) | ((pixel[1] >> 4) << 4) | (pixel[2] >> 4);
            if (!buckets.TryGetValue(key, out Bucket? rec))
            {
                rec = new Bucket();
                buckets[key] = rec;
            }

            rec.N++;
            rec.R += pixel[0];
            rec.G += pixel[1];
            rec.B += pixel[2];
        }

        Bucket? best = null;
        Bucket? second = null;
        foreach (Bucket rec in buckets.Values)
        {
            if (best is null || rec.N > best.N)
            {
                second = best;
                best = rec;
            }
            else if (second is null || rec.N > second.N)
            {
                second = rec;
            }
        }

        Rgb bg = ToRgb(best!);
        if (second is not null
            && best!.N < pixels.Count * 0.55
            && Luma(bg) < 80
            && Luma(ToRgb(second)) > 140)
        {
            return ToRgb(second);
        }

        return bg;
    }

    static Rgb? PickInkColor(List<int[]> pixels, Rgb background)
    {
        List<InkSample> cores = [];
        foreach (int[] pixel in pixels)
        {
            double gap = ColorDist(pixel[0], pixel[1], pixel[2], background);
            if (gap >= 16)
                cores.Add(new InkSample(pixel[0], pixel[1], pixel[2], gap));
        }

        if (cores.Count == 0)
            return null;

        double[] distances = new double[cores.Count];
        for (int i = 0; i < cores.Count; i++)
            distances[i] = cores[i].D;
        double cutoff = Median(distances);
        List<InkSample> strong = [];
        foreach (InkSample sample in cores)
        {
            if (sample.D >= cutoff)
                strong.Add(sample);
        }

        List<InkSample> use = strong.Count > 0 ? strong : cores;
        int[] rs = new int[use.Count];
        int[] gs = new int[use.Count];
        int[] bs = new int[use.Count];
        for (int i = 0; i < use.Count; i++)
        {
            rs[i] = use[i].R;
            gs[i] = use[i].G;
            bs[i] = use[i].B;
        }

        return new Rgb(Median(rs), Median(gs), Median(bs));
    }

    static void FillQuad(SKCanvas canvas, SKPoint[] pts, Rgb color)
    {
        using SKPathBuilder builder = new();
        builder.MoveTo(pts[0]);
        builder.LineTo(pts[1]);
        builder.LineTo(pts[2]);
        builder.LineTo(pts[3]);
        builder.Close();
        using SKPath path = builder.Detach();
        using SKPaint paint = new()
        {
            IsAntialias = true,
            Color = ToColor(color),
            Style = SKPaintStyle.Fill
        };
        canvas.DrawPath(path, paint);
    }

    static float FitFontSize(SKTypeface typeface, string text, float maxW, float maxH)
    {
        float lo = 6;
        float hi = Math.Max(8, maxH * 0.82f);
        float best = 6;
        while (hi - lo > 0.5f)
        {
            float mid = (lo + hi) / 2f;
            using SKFont font = new(typeface, mid);
            font.GetFontMetrics(out SKFontMetrics metrics);
            float width = font.MeasureText(text);
            float height = metrics.Descent - metrics.Ascent;
            if (width <= maxW && height <= maxH)
            {
                best = mid;
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }

        return best;
    }

    static Rgb ContrastInk(Rgb color) =>
        Luma(color) > 140 ? new Rgb(17, 17, 17) : new Rgb(245, 245, 247);

    static double Luma(Rgb color) => 0.2126 * color.R + 0.7152 * color.G + 0.0722 * color.B;

    static double ColorDist(int r, int g, int b, Rgb other)
    {
        double dr = r - other.R;
        double dg = g - other.G;
        double db = b - other.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }

    static int Median(int[] values)
    {
        Array.Sort(values);
        return values[values.Length / 2];
    }

    static double Median(double[] values)
    {
        Array.Sort(values);
        return values[values.Length / 2];
    }

    static Rgb ToRgb(Bucket rec) => new(rec.R / rec.N, rec.G / rec.N, rec.B / rec.N);

    static SKColor ToColor(Rgb color) =>
        new((byte)Clamp(color.R), (byte)Clamp(color.G), (byte)Clamp(color.B));

    static int Clamp(double value)
    {
        if (value < 0)
            return 0;
        if (value > 255)
            return 255;
        return (int)Math.Round(value);
    }

    static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    readonly struct QuadGeometry(float cx, float cy, float len, float ht, float angleDegrees)
    {
        public float Cx { get; } = cx;
        public float Cy { get; } = cy;
        public float Len { get; } = len;
        public float Ht { get; } = ht;
        public float AngleDegrees { get; } = angleDegrees;
    }

    readonly struct Rgb(double r, double g, double b)
    {
        public double R { get; } = r;
        public double G { get; } = g;
        public double B { get; } = b;
    }

    sealed class Bucket
    {
        public int N;
        public double R;
        public double G;
        public double B;
    }

    readonly struct InkSample(int r, int g, int b, double d)
    {
        public int R { get; } = r;
        public int G { get; } = g;
        public int B { get; } = b;
        public double D { get; } = d;
    }
}
