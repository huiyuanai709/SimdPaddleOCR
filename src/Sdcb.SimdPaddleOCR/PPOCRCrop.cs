using System.Runtime.CompilerServices;

using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR;

internal static class PPOCRCrop
{
    private const double PerspectiveEpsilon = 1e-12;

    private readonly record struct Point(double X, double Y);

    private readonly record struct PerspectiveTransform(
        double A, double B, double C, double D, double E, double F, double G, double H,
        int UnrotatedWidth, int UnrotatedHeight, bool RotateVertical);

    public static (int Width, int Height, int ByteCount) GetSize(in PaddleOcrDetectionBox box)
    {
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid detection quadrilateral.");
        int width = transform.RotateVertical ? transform.UnrotatedHeight : transform.UnrotatedWidth;
        int height = transform.RotateVertical ? transform.UnrotatedWidth : transform.UnrotatedHeight;
        long bytes = checked((long)width * height * 3);
        return (width, height, checked((int)bytes));
    }

    public static byte[] Extract(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, out int outputWidth, out int outputHeight,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        (int Width, int Height, int ByteCount) size = GetSize(box);
        outputWidth = size.Width;
        outputHeight = size.Height;
        byte[] crop = new byte[size.ByteCount];
        ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, crop, outputWidth, outputHeight,
            format);
        return crop;
    }

    public static void ExtractInto(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> destination, out int outputWidth,
        out int outputHeight, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        (int Width, int Height, int ByteCount) size = GetSize(box);
        if (destination.Length < size.ByteCount)
            throw new ArgumentException("Destination buffer is too small.", nameof(destination));
        outputWidth = size.Width;
        outputHeight = size.Height;
        ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, destination,
            outputWidth, outputHeight, format);
    }

    /// <summary>
    /// Height of the unrotated sampling grid, i.e. how many rows
    /// <see cref="ExtractRangeInto"/> can split the crop into.
    /// </summary>
    public static int UnrotatedHeight(in PaddleOcrDetectionBox box)
        => TryComputePerspective(box, out PerspectiveTransform transform)
            ? transform.UnrotatedHeight
            : throw new InvalidDataException("Invalid detection quadrilateral.");

    /// <summary>
    /// Extracts rows [<paramref name="yBegin"/>, <paramref name="yEnd"/>) of the same
    /// perspective crop. Each sampled row writes a disjoint destination range, so
    /// bands can run on different workers; the per-row arithmetic is untouched.
    /// </summary>
    public static void ExtractRangeInto(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> destination,
        int yBegin, int yEnd, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid detection quadrilateral.");
        int outputWidth = transform.RotateVertical ? transform.UnrotatedHeight : transform.UnrotatedWidth;
        int outputHeight = transform.RotateVertical ? transform.UnrotatedWidth : transform.UnrotatedHeight;
        if (destination.Length < checked(outputWidth * outputHeight * 3))
            throw new ArgumentException("Destination buffer is too small.", nameof(destination));
        ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, destination,
            outputWidth, outputHeight, format, yBegin, yEnd);
    }

    private static unsafe void ExtractCore(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> crop, int outputWidth, int outputHeight,
        ImagePixelFormat format)
        => ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, crop, outputWidth, outputHeight,
            format, 0, int.MaxValue);

    private static unsafe void ExtractCore(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> crop, int outputWidth, int outputHeight,
        ImagePixelFormat format, int yBegin, int yEnd)
    {
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid perspective transform.");
        double a = transform.A, b = transform.B, c = transform.C, d = transform.D,
            e = transform.E, f = transform.F, g = transform.G, h = transform.H;
        int unrotatedWidth = transform.UnrotatedWidth, unrotatedHeight = transform.UnrotatedHeight;
        bool rotateVertical = transform.RotateVertical;
        Warp.PixelAccess access = Warp.PixelAccess.Of(format);

        fixed (byte* sourcePtr = source)
        fixed (byte* cropPtr = crop)
        {
            for (int y = yBegin; y < Math.Min(yEnd, unrotatedHeight); y++)
            {
                double v = (double)y / unrotatedHeight;
                int x = 0;
                Warp.MapRow(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                    unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x,
                    access);
                for (; x < unrotatedWidth; x++)
                    Warp.ProcessPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x, y, v, access);
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Rotate180(Span<byte> pixels, int width, int height)
    {
        int count = checked(width * height);
        fixed (byte* data = pixels)
        {
            for (int left = 0; left < count / 2; left++)
            {
                int right = count - 1 - left;
                byte* l = data + left * 3, r = data + right * 3;
                byte value = l[0]; l[0] = r[0]; r[0] = value;
                value = l[1]; l[1] = r[1]; r[1] = value;
                value = l[2]; l[2] = r[2]; r[2] = value;
            }
        }
    }

    private static bool TryComputePerspective(in PaddleOcrDetectionBox box,
        out PerspectiveTransform transform)
    {
        transform = default;
        Span<Point> points = stackalloc Point[4];
        LoadPoints(box, points);
        double widthValue = Math.Max(Distance(points[0], points[1]), Distance(points[2], points[3]));
        double heightValue = Math.Max(Distance(points[0], points[3]), Distance(points[1], points[2]));
        if (!MathCompat.IsFinite(widthValue) || !MathCompat.IsFinite(heightValue) || widthValue < 1 ||
            heightValue < 1 || widthValue > uint.MaxValue || heightValue > uint.MaxValue)
            return false;
        int unrotatedWidth = Math.Max(1, checked((int)Math.Floor(widthValue)));
        int unrotatedHeight = Math.Max(1, checked((int)Math.Floor(heightValue)));
        bool rotateVertical = unrotatedHeight >= unrotatedWidth * 1.5;

        double dx1 = points[1].X - points[2].X;
        double dx2 = points[3].X - points[2].X;
        double dx3 = points[0].X - points[1].X + points[2].X - points[3].X;
        double dy1 = points[1].Y - points[2].Y;
        double dy2 = points[3].Y - points[2].Y;
        double dy3 = points[0].Y - points[1].Y + points[2].Y - points[3].Y;
        double g = 0, h = 0;
        if (Math.Abs(dx3) > PerspectiveEpsilon || Math.Abs(dy3) > PerspectiveEpsilon)
        {
            double denominator = dx1 * dy2 - dx2 * dy1;
            if (!MathCompat.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
                return false;
            g = (dx3 * dy2 - dx2 * dy3) / denominator;
            h = (dx1 * dy3 - dx3 * dy1) / denominator;
            if (!MathCompat.IsFinite(g) || !MathCompat.IsFinite(h))
                return false;
        }
        double a = points[1].X - points[0].X + g * points[1].X;
        double b = points[3].X - points[0].X + h * points[3].X;
        double c = points[0].X;
        double d = points[1].Y - points[0].Y + g * points[1].Y;
        double e = points[3].Y - points[0].Y + h * points[3].Y;
        double f = points[0].Y;
        transform = new PerspectiveTransform(a, b, c, d, e, f, g, h,
            unrotatedWidth, unrotatedHeight, rotateVertical);
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadPoints(in PaddleOcrDetectionBox box, Span<Point> points)
    {
        points[0] = new(box.X1, box.Y1);
        points[1] = new(box.X2, box.Y2);
        points[2] = new(box.X3, box.Y3);
        points[3] = new(box.X4, box.Y4);
    }

    /// <summary>
    /// Places each CTC run on the detection quad. <paramref name="contentWidth"/>
    /// is the resized crop; time steps span <paramref name="tensorWidth"/>, so
    /// right padding is the trailing fraction and is not part of the line.
    /// A 180° classification flips the crop before recognition, and a tall quad
    /// is stored rotated 90° (crop x = 0 is the bottom edge). Both are undone
    /// here so the quad sits on the source image.
    /// </summary>
    internal static PaddleOcrCharacterBox[] EstimateCharacterBoxes(
        in PaddleOcrDetectionBox box, ReadOnlySpan<PaddleOcrCtcSpan> spans,
        int timeSteps, int contentWidth, int tensorWidth, int appliedRotationDegrees)
    {
        if (spans.Length == 0) return [];
        if (timeSteps <= 0) throw new ArgumentOutOfRangeException(nameof(timeSteps));
        if (contentWidth <= 0) throw new ArgumentOutOfRangeException(nameof(contentWidth));
        if (tensorWidth < contentWidth) throw new ArgumentOutOfRangeException(nameof(tensorWidth));
        if (appliedRotationDegrees is not (0 or 180))
            throw new ArgumentOutOfRangeException(nameof(appliedRotationDegrees));
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid detection quadrilateral.");

        PaddleOcrCharacterBox[] boxes = new PaddleOcrCharacterBox[spans.Length];
        double widthScale = (double)tensorWidth / contentWidth / timeSteps;
        bool flip = appliedRotationDegrees == 180;
        for (int i = 0; i < spans.Length; i++)
        {
            PaddleOcrCtcSpan span = spans[i];
            if ((uint)span.StartColumn >= (uint)timeSteps || span.EndColumn <= span.StartColumn ||
                span.EndColumn > timeSteps)
                throw new ArgumentOutOfRangeException(nameof(spans));
            // The span is only the fired run. The blank gap between two runs is
            // split at its midpoint so the boxes share an edge. Outer edges stay
            // on the trigger, because there is no neighbor to take that blank.
            double left = span.StartColumn;
            double right = span.EndColumn;
            if (i > 0)
            {
                int previousEnd = spans[i - 1].EndColumn;
                if (previousEnd > span.StartColumn)
                    throw new ArgumentOutOfRangeException(nameof(spans));
                left = (previousEnd + (double)span.StartColumn) / 2;
            }
            if (i + 1 < spans.Length)
            {
                int nextStart = spans[i + 1].StartColumn;
                if (span.EndColumn > nextStart)
                    throw new ArgumentOutOfRangeException(nameof(spans));
                right = (span.EndColumn + (double)nextStart) / 2;
            }
            double t0 = left * widthScale;
            double t1 = right * widthScale;
            if (flip)
            {
                double flipped = 1 - t1;
                t1 = 1 - t0;
                t0 = flipped;
            }
            if (t0 < 0) t0 = 0;
            else if (t0 > 1) t0 = 1;
            if (t1 < 0) t1 = 0;
            else if (t1 > 1) t1 = 1;
            if (t1 < t0) t1 = t0;
            MapCropSpan(transform, t0, t1,
                out double x1, out double y1, out double x2, out double y2,
                out double x3, out double y3, out double x4, out double y4);
            boxes[i] = new PaddleOcrCharacterBox(span.Text, span.Score,
                (float)x1, (float)y1, (float)x2, (float)y2,
                (float)x3, (float)y3, (float)x4, (float)y4);
        }
        return boxes;
    }

    private static void MapCropSpan(in PerspectiveTransform transform, double t0, double t1,
        out double x1, out double y1, out double x2, out double y2,
        out double x3, out double y3, out double x4, out double y4)
    {
        if (transform.RotateVertical)
        {
            // Crop x runs from the bottom edge (v = 1) toward the top (v = 0).
            // Crop y runs across the short side (u).
            Map(transform, 0, 1 - t0, out x1, out y1);
            Map(transform, 0, 1 - t1, out x2, out y2);
            Map(transform, 1, 1 - t1, out x3, out y3);
            Map(transform, 1, 1 - t0, out x4, out y4);
            return;
        }
        Map(transform, t0, 0, out x1, out y1);
        Map(transform, t1, 0, out x2, out y2);
        Map(transform, t1, 1, out x3, out y3);
        Map(transform, t0, 1, out x4, out y4);
    }

    private static void Map(in PerspectiveTransform transform, double u, double v, out double x, out double y)
    {
        double denominator = transform.G * u + transform.H * v + 1;
        if (!MathCompat.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
            throw new InvalidDataException("Invalid perspective transform.");
        x = (transform.A * u + transform.B * v + transform.C) / denominator;
        y = (transform.D * u + transform.E * v + transform.F) / denominator;
        if (!MathCompat.IsFinite(x) || !MathCompat.IsFinite(y))
            throw new InvalidDataException("Invalid perspective transform.");
    }

    private static double Distance(Point a, Point b) => Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));

    private static void ValidateSource(ReadOnlySpan<byte> source, int width, int height, int stride,
        ImagePixelFormat format)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int bpp = ImagePixels.BytesPerPixel(format);
        if (stride < checked(width * bpp)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * (long)bpp);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }
}
