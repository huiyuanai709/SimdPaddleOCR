namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Straightens one seal-text polygon the way PaddleX <c>CropByPolys.get_poly_rect_crop</c>
/// does: a near-rectangular ring uses the min-area quad crop, and a curved ring is split
/// into two sidelines, resampled, and stitched with per-segment homographies
/// (<c>CurveTextRectifier.dc_homo</c>, mode <c>homography</c>).
/// </summary>
public static class PaddleOcrSealRectify
{
    /// <summary>
    /// Rectifies <paramref name="polygon"/> into a horizontal BGR strip the recognizer can read.
    /// </summary>
    public static byte[] Rectify(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        ReadOnlySpan<PaddleOcrPoint> polygon, out int width, out int height, int sourceStride = 0,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        Validate(source, sourceWidth, sourceHeight, sourceStride, format);
        if (polygon.Length < 3) throw new InvalidDataException("Seal text polygon needs at least 3 points.");
        if (TryCurved(source, sourceWidth, sourceHeight, sourceStride, polygon, format,
                out byte[]? curved, out width, out height))
            return curved!;
        if (TryMinArea(source, sourceWidth, sourceHeight, sourceStride, polygon, format,
                out byte[]? straight, out width, out height))
            return straight!;
        throw new InvalidDataException("Invalid seal text polygon.");
    }

    private readonly record struct V(double X, double Y);

    private static bool TryCurved(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight, int sourceStride,
        ReadOnlySpan<PaddleOcrPoint> polygon, ImagePixelFormat format, out byte[]? pixels, out int width, out int height)
    {
        pixels = null;
        width = height = 0;
        if (polygon.Length < 4 || NearlyRectangular(polygon)) return false;
        V[] working = ToVectors(polygon);
        V[] merged = MergeShortEdges(working);
        V[] shape = merged.Length >= 4 ? merged : ResampleClosed(working, 48);
        if (shape.Length < 4) return false;
        if (shape.Length > 96) shape = ResampleClosed(shape, 64);
        if (!TrySidelines(shape, out V[] sideA, out V[] sideB)) return false;
        V[] top = ResampleOpen(sideA, 15);
        V[] bottom = ResampleOpen(sideB, 15);
        if (top.Length < 2 || top.Length != bottom.Length) return false;
        if (MeanY(top) > MeanY(bottom)) (top, bottom) = (bottom, top);
        V[] combined = new V[top.Length + bottom.Length];
        top.CopyTo(combined, 0);
        bottom.CopyTo(combined, top.Length);
        bool horizontal = IsHorizontal(combined);
        V[] imagePoints = combined;
        V[] world = BuildWorld(combined, out _);
        if (!horizontal)
        {
            if (!TryVerticalFrame(combined, sourceHeight, out imagePoints, out world)) return false;
        }
        if (imagePoints.Length != world.Length || imagePoints.Length < 4) return false;
        if (!TryStitch(source, sourceWidth, sourceHeight, sourceStride, format, imagePoints, world, horizontal,
                out pixels, out width, out height))
            return false;
        if (!horizontal)
            RotateClockwise(ref pixels!, ref width, ref height);
        return width > 0 && height > 0 && pixels is { Length: > 0 };
    }

    private static bool NearlyRectangular(ReadOnlySpan<PaddleOcrPoint> polygon)
    {
        DbPostprocess.Point[] points = new DbPostprocess.Point[polygon.Length];
        for (int i = 0; i < polygon.Length; i++)
            points[i] = new DbPostprocess.Point(polygon[i].X, polygon[i].Y);
        Span<DbPostprocess.Point> corners = stackalloc DbPostprocess.Point[4];
        if (!DbPostprocess.TryOrderedMinAreaQuad(points, corners)) return true;
        double rect = Math.Abs(Shoelace(corners));
        if (rect <= 1e-3) return true;
        double poly = DbPostprocess.AbsoluteArea(points);
        return poly / rect >= 0.7;
    }

    private static bool TryMinArea(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight, int sourceStride,
        ReadOnlySpan<PaddleOcrPoint> polygon, ImagePixelFormat format, out byte[]? pixels, out int width, out int height)
    {
        pixels = null;
        width = height = 0;
        DbPostprocess.Point[] points = new DbPostprocess.Point[polygon.Length];
        for (int i = 0; i < polygon.Length; i++)
            points[i] = new DbPostprocess.Point(polygon[i].X, polygon[i].Y);
        Span<DbPostprocess.Point> corners = stackalloc DbPostprocess.Point[4];
        if (!DbPostprocess.TryOrderedMinAreaQuad(points, corners)) return false;
        PaddleOcrDetectionBox box = new(corners[0].X, corners[0].Y, corners[1].X, corners[1].Y,
            corners[2].X, corners[2].Y, corners[3].X, corners[3].Y, 1f);
        try
        {
            pixels = PPOCRCrop.Extract(source, sourceWidth, sourceHeight, sourceStride, box, out width, out height, format);
            return width > 0 && height > 0;
        }
        catch (InvalidDataException)
        {
            return false;
        }
    }

    private static bool TrySidelines(V[] points, out V[] sideA, out V[] sideB)
    {
        sideA = sideB = [];
        if (!TryHeadTail(points, 2.0, out int headStart, out int headEnd, out int tailStart, out int tailEnd))
            return false;
        int n = points.Length;
        V[] pad = new V[n * 2];
        for (int i = 0; i < n; i++) pad[i] = pad[i + n] = points[i];
        int end = tailEnd < 1 ? n : tailEnd;
        if ((uint)headEnd >= (uint)pad.Length || end < headEnd || end > pad.Length) return false;
        int secondEnd = headEnd + n;
        if (secondEnd < end || secondEnd > pad.Length) return false;
        sideA = CopyRange(pad, headEnd, end);
        sideB = CopyRange(pad, end, secondEnd);
        return sideA.Length >= 2 && sideB.Length >= 2;
    }

    private static bool TryHeadTail(V[] points, double orientationThr, out int headStart, out int headEnd,
        out int tailStart, out int tailEnd)
    {
        headStart = headEnd = tailStart = tailEnd = 0;
        int n = points.Length;
        if (n < 4) return false;
        if (n == 4)
            return QuadHeadTail(points, orientationThr, out headStart, out headEnd, out tailStart, out tailEnd);

        Span<(double X, double Y)> edge = stackalloc (double X, double Y)[n];
        for (int i = 0; i < n; i++)
        {
            V a = points[i], b = points[(i + 1) % n];
            edge[i] = (b.X - a.X, b.Y - a.Y);
        }
        double[] thetaSum = new double[n];
        double[] adjacent = new double[n];
        for (int i = 0; i < n; i++)
        {
            (double X, double Y) prev = edge[(i - 1 + n) % n];
            (double X, double Y) next = edge[(i + 1) % n];
            (double X, double Y) cur = edge[i];
            thetaSum[i] = Angle(cur.X, cur.Y, prev.X, prev.Y) + Angle(cur.X, cur.Y, next.X, next.Y);
            adjacent[i] = Angle(prev.X, prev.Y, next.X, next.Y);
        }
        double[] score = new double[n];
        double maxDist = 0;
        V center = Centroid(points);
        double[] dist = new double[n];
        for (int i = 0; i < n; i++)
        {
            V a = points[i], b = points[(i + 1) % n];
            dist[i] = Math.Max(Distance(a, center), Distance(b, center));
            if (dist[i] > maxDist) maxDist = dist[i];
        }
        if (maxDist <= 1e-8) maxDist = 1;
        for (int i = 0; i < n; i++)
            score[i] = 0.5 * (thetaSum[i] / Math.PI) + 0.15 * (adjacent[i] / Math.PI) + 0.35 * (dist[i] / maxDist);
        if ((n & 1) == 0)
        {
            score[n / 2 - 1] += 0.1;
            score[n - 1] += 0.1;
        }
        int cols = n - 3;
        double[] matrix = new double[n * cols];
        double[] gaussian = Gaussian(cols);
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < cols; k++)
            {
                int pad = (i + 2 + k) % (n * 2);
                // pad_score is score concatenated with itself; index i+2+k is within [i+2, i+n-2].
                int source = i + 2 + k;
                double padded = source < n ? score[source] : score[source - n];
                matrix[i * cols + k] = score[i] + padded * gaussian[k] * 0.3;
            }
        }
        int best = 0;
        for (int i = 1; i < matrix.Length; i++)
            if (matrix[i] > matrix[best]) best = i;
        headStart = best / cols;
        int tailIncrement = best % cols;
        tailStart = (headStart + tailIncrement + 2) % n;
        headEnd = (headStart + 1) % n;
        tailEnd = (tailStart + 1) % n;
        if (headEnd > tailEnd)
        {
            (headStart, tailStart) = (tailStart, headStart);
            (headEnd, tailEnd) = (tailEnd, headEnd);
        }
        return true;
    }

    private static bool QuadHeadTail(V[] points, double orientationThr, out int headStart, out int headEnd,
        out int tailStart, out int tailEnd)
    {
        double slopeSumA = Slope(points[1], points[0]) + Slope(points[3], points[2]);
        double slopeSumB = Slope(points[2], points[1]) + Slope(points[0], points[3]);
        int h0a, h0b, h1a, h1b, v0a, v0b, v1a, v1b;
        if (slopeSumA < slopeSumB)
        {
            h0a = 0; h0b = 1; h1a = 2; h1b = 3;
            v0a = 3; v0b = 0; v1a = 1; v1b = 2;
        }
        else
        {
            h0a = 3; h0b = 0; h1a = 1; h1b = 2;
            v0a = 0; v0b = 1; v1a = 2; v1b = 3;
        }
        double vertical = Distance(points[v0a], points[v0b]) + Distance(points[v1a], points[v1b]);
        double horizontal = Distance(points[h0a], points[h0b]) + Distance(points[h1a], points[h1b]);
        if (vertical > horizontal * orientationThr)
        {
            headStart = h0a; headEnd = h0b; tailStart = h1a; tailEnd = h1b;
        }
        else
        {
            headStart = v0a; headEnd = v0b; tailStart = v1a; tailEnd = v1b;
        }
        return true;
    }

    private static double[] Gaussian(int cols)
    {
        double[] values = new double[cols];
        double denom = Math.Max(1, cols - 1);
        double max = 0;
        for (int i = 0; i < cols; i++)
        {
            double x = i / denom;
            double z = (x - 0.5) / 0.5;
            values[i] = 1.0 / (Math.Sqrt(2.0 * Math.PI) * 0.5) * Math.Exp(-(z * z) / 2.0);
            if (values[i] > max) max = values[i];
        }
        if (max > 0)
            for (int i = 0; i < cols; i++) values[i] /= max;
        return values;
    }

    // PaddleX sample_points_on_bbox: collapse runs of edges shorter than 0.9× the mean.
    private static V[] MergeShortEdges(V[] line)
    {
        int n = line.Length;
        if (n < 2) return line;
        double[] lengths = new double[n - 1];
        double total = 0;
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = Distance(line[i], line[i + 1]);
            total += lengths[i];
        }
        double mean = total / (lengths.Length + 1e-8);
        List<List<int>> groups = [[0]];
        List<int>? last = null;
        for (int i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] >= 0.9 * mean) continue;
            int pointId = i + 1;
            List<int>? host = null;
            foreach (List<int> group in groups)
            {
                if (group.Contains(i)) { host = group; break; }
            }
            if (host is null)
            {
                host = [pointId];
                groups.Add(host);
            }
            else host.Add(pointId);
            last = host;
        }
        if (last is not null && !ReferenceEquals(last, groups[0]) && Distance(line[0], line[^1]) < 0.9 * mean)
        {
            groups[0].AddRange(last);
            groups.Remove(last);
        }
        V[] result = new V[groups.Count];
        for (int g = 0; g < groups.Count; g++)
        {
            double sx = 0, sy = 0;
            foreach (int index in groups[g])
            {
                if ((uint)index >= (uint)n) continue;
                sx += line[index].X;
                sy += line[index].Y;
            }
            int count = groups[g].Count;
            result[g] = count == 0 ? line[0] : new V(sx / count, sy / count);
        }
        return result;
    }

    // sample_points_on_bbox_bp. The Python loop emits n+1 samples (both endpoints included).
    private static V[] ResampleOpen(V[] line, int n)
    {
        if (line.Length < 2 || n < 1) return line;
        double[] lengths = new double[line.Length - 1];
        double total = 0;
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = Distance(line[i], line[i + 1]);
            total += lengths[i];
        }
        double[] cumulative = new double[line.Length];
        for (int i = 1; i < line.Length; i++) cumulative[i] = cumulative[i - 1] + lengths[i - 1];
        V[] result = new V[n + 1];
        result[0] = line[0];
        double delta = total / (n + 1e-8);
        int edge = 0;
        int written = 1;
        for (int i = 1; i < n; i++)
        {
            double target = i * delta;
            while (edge + 1 < cumulative.Length && target >= cumulative[edge + 1]) edge++;
            if (edge >= lengths.Length) break;
            double ratio = lengths[edge] <= 1e-8 ? 0 : (target - cumulative[edge]) / lengths[edge];
            V a = line[edge], b = line[edge + 1];
            result[written++] = new V(a.X + (b.X - a.X) * ratio, a.Y + (b.Y - a.Y) * ratio);
        }
        result[written++] = line[line.Length - 1];
        if (written == result.Length) return result;
        V[] trimmed = new V[written];
        Array.Copy(result, trimmed, written);
        return trimmed;
    }

    private static V[] ResampleClosed(V[] points, int n)
    {
        if (points.Length < 3) return points;
        V[] open = new V[points.Length + 1];
        points.CopyTo(open, 0);
        open[^1] = points[0];
        V[] sampled = ResampleOpen(open, n);
        if (sampled.Length < 2) return sampled;
        V[] closed = new V[sampled.Length - 1];
        Array.Copy(sampled, closed, closed.Length);
        return closed;
    }

    private static V[] BuildWorld(V[] image, out double height)
    {
        int npts = image.Length;
        int half = npts / 2;
        double[] dx = new double[Math.Max(0, npts - 1)];
        for (int i = 1; i < npts; i++) dx[i - 1] = Distance(image[i], image[i - 1]);
        int pair = dx.Length / 2;
        double[] averaged = new double[pair];
        for (int i = 0; i < pair; i++) averaged[i] = (dx[i] + dx[dx.Length - 1 - i]) / 2;
        double dySum = 0;
        for (int i = 0; i < half; i++)
            dySum += Distance(image[i], image[npts - 1 - i]);
        height = half == 0 ? 1 : Math.Round(dySum / half, MidpointRounding.ToEven);
        if (height < 1) height = 1;
        // Top edge walks the averaged arc length; the bottom edge is that polyline reversed.
        int topCount = Math.Max(1, Math.Min(half, averaged.Length + 1));
        double[] topX = new double[topCount];
        double cursor = 0;
        for (int i = 1; i < topCount; i++)
        {
            cursor += averaged[i - 1];
            topX[i] = cursor;
        }
        double centerX = topX[topCount - 1] / 2;
        double centerY = height / 2;
        V[] world = new V[image.Length];
        for (int i = 0; i < world.Length; i++)
        {
            int topIndex = i < topCount ? i : topCount - 1 - (i - topCount);
            if (topIndex < 0) topIndex = 0;
            if (topIndex >= topCount) topIndex = topCount - 1;
            double y = i < topCount ? 0 : height;
            world[i] = new V(topX[topIndex] - centerX, y - centerY);
        }
        return world;
    }

    private static bool TryVerticalFrame(V[] combined, int sourceHeight, out V[] imagePoints, out V[] world)
    {
        imagePoints = [];
        world = [];
        if (combined.Length < 4) return false;
        V[] shifted = new V[combined.Length];
        for (int i = 0; i < combined.Length; i++)
            shifted[i] = combined[(i + 1) % combined.Length];
        V[] adjusted = new V[shifted.Length];
        for (int i = 0; i < shifted.Length; i++)
            adjusted[i] = new V(shifted[i].Y, sourceHeight - shifted[i].X - 1);
        V[] adjustedWorld = BuildWorld(adjusted, out _);
        imagePoints = shifted;
        world = new V[adjustedWorld.Length];
        for (int i = 0; i < world.Length; i++)
            world[i] = new V(-adjustedWorld[i].Y, adjustedWorld[i].X);
        return true;
    }

    private static bool TryStitch(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight, int sourceStride,
        ImagePixelFormat format, V[] image, V[] world, bool horizontal, out byte[]? pixels, out int width, out int height)
    {
        pixels = null;
        width = height = 0;
        int n = image.Length;
        int segments = n / 2 - 1;
        if (segments < 1) return false;
        List<byte[]> tiles = [];
        List<int> tileWidths = [];
        int tileHeight = 0;
        for (int i = 0; i < segments; i++)
        {
            V tl = image[i], tr = image[i + 1];
            V br = image[n - 2 - i], bl = image[n - 1 - i];
            double worldWidth = horizontal
                ? Math.Abs(world[i + 1].X - world[i].X)
                : Math.Abs(world[i + 1].Y - world[i].Y);
            double worldHeight = horizontal
                ? Math.Abs(world[n - 1 - i].Y - world[i].Y)
                : Math.Abs(world[n - 1 - i].X - world[i].X);
            int dw = (int)worldWidth;
            int dh = (int)worldHeight;
            if (dw < 1 || dh < 1) continue;
            if ((long)dw * dh > 16_000_000) return false;
            byte[] tile = new byte[checked(dw * dh * 3)];
            try
            {
                if (!PPOCRCrop.TryWarpQuad(source, sourceWidth, sourceHeight, sourceStride,
                        (float)tl.X, (float)tl.Y, (float)tr.X, (float)tr.Y,
                        (float)br.X, (float)br.Y, (float)bl.X, (float)bl.Y,
                        tile, dw, dh, format))
                    return false;
            }
            catch (InvalidDataException)
            {
                return false;
            }
            tiles.Add(tile);
            tileWidths.Add(dw);
            if (dh > tileHeight) tileHeight = dh;
        }
        if (tiles.Count == 0 || tileHeight < 1) return false;
        int totalWidth = 0;
        foreach (int tileWidth in tileWidths) totalWidth = checked(totalWidth + tileWidth);
        if (totalWidth < 1) return false;
        byte[] strip = new byte[checked(totalWidth * (long)tileHeight * 3)];
        int column = 0;
        for (int t = 0; t < tiles.Count; t++)
        {
            int dw = tileWidths[t];
            int dh = tiles[t].Length / (dw * 3);
            for (int y = 0; y < dh && y < tileHeight; y++)
            {
                tiles[t].AsSpan(y * dw * 3, dw * 3)
                    .CopyTo(strip.AsSpan((y * totalWidth + column) * 3, dw * 3));
            }
            column += dw;
        }
        pixels = strip;
        width = totalWidth;
        height = tileHeight;
        return true;
    }

    private static void RotateClockwise(ref byte[] pixels, ref int width, ref int height)
    {
        int w = width, h = height;
        byte[] rotated = new byte[pixels.Length];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int src = (y * w + x) * 3;
                int dy = x;
                int dx = h - 1 - y;
                int dst = (dy * h + dx) * 3;
                rotated[dst] = pixels[src];
                rotated[dst + 1] = pixels[src + 1];
                rotated[dst + 2] = pixels[src + 2];
            }
        }
        pixels = rotated;
        width = h;
        height = w;
    }

    private static bool IsHorizontal(V[] points)
    {
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
        double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        foreach (V p in points)
        {
            if (p.X < minX) minX = p.X;
            if (p.Y < minY) minY = p.Y;
            if (p.X > maxX) maxX = p.X;
            if (p.Y > maxY) maxY = p.Y;
        }
        double x = Math.Max(1e-6, maxX - minX);
        return (maxY - minY) / x <= 1.5;
    }

    private static V[] CopyRange(V[] source, int start, int end)
    {
        V[] slice = new V[end - start];
        Array.Copy(source, start, slice, 0, slice.Length);
        return slice;
    }

    private static V[] ToVectors(ReadOnlySpan<PaddleOcrPoint> polygon)
    {
        V[] points = new V[polygon.Length];
        for (int i = 0; i < polygon.Length; i++) points[i] = new V(polygon[i].X, polygon[i].Y);
        return points;
    }

    private static double MeanY(V[] points)
    {
        double sum = 0;
        foreach (V p in points) sum += p.Y;
        return sum / points.Length;
    }

    private static V Centroid(V[] points)
    {
        double x = 0, y = 0;
        foreach (V p in points) { x += p.X; y += p.Y; }
        return new V(x / points.Length, y / points.Length);
    }

    private static double Shoelace(ReadOnlySpan<DbPostprocess.Point> points)
    {
        double area = 0;
        for (int i = 0; i < points.Length; i++)
        {
            DbPostprocess.Point a = points[i], b = points[(i + 1) % points.Length];
            area += (double)a.X * b.Y - (double)b.X * a.Y;
        }
        return area * 0.5;
    }

    private static double Distance(V a, V b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static double Slope(V a, V b) => Math.Abs((a.Y - b.Y) / (a.X - b.X + 1e-8));

    private static double Angle(double ax, double ay, double bx, double by)
    {
        double na = Math.Sqrt(ax * ax + ay * ay) + 1e-8;
        double nb = Math.Sqrt(bx * bx + by * by) + 1e-8;
        double dot = ax / na * (bx / nb) + ay / na * (by / nb);
        if (dot > 1) dot = 1;
        else if (dot < -1) dot = -1;
        return Math.Acos(dot);
    }

    private static void Validate(ReadOnlySpan<byte> source, int width, int height, int stride, ImagePixelFormat format)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int bpp = ImagePixels.BytesPerPixel(format);
        if (stride < checked(width * bpp)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * (long)bpp);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }
}
