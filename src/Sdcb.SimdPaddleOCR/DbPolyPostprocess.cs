using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// DB post-process in PaddleOCR <c>box_type=poly</c> mode: contour, Douglas-Peucker,
/// polygon score, round-join unclip, then map the expanded ring back to the source.
/// Quad detection stays in <see cref="DbPostprocess"/>.
/// </summary>
internal static class DbPolyPostprocess
{
    public static PaddleOcrPolygon[] Run(ReadOnlySpan<float> prediction, int mapWidth, int mapHeight,
        PaddleOcrSealDetectorOptions options, int sourceWidth, int sourceHeight)
    {
        if (prediction.Length != checked(mapWidth * mapHeight))
            throw new ArgumentException("Invalid detector output size.");
        int pixels = checked(mapWidth * mapHeight);
        byte[] bitmap = new byte[pixels];
        Threshold.Binarize(prediction, bitmap, pixels, options.BitmapThreshold);
        if (options.UseDilation)
        {
            byte[] dilated = new byte[pixels];
            Dilate2x2(bitmap, dilated, mapWidth, mapHeight, pixels);
        }

        List<PointF> contours = [];
        List<int> contourCounts = [];
        TraceContours(bitmap, mapWidth, mapHeight, options.MaxCandidates, contours, contourCounts);

        List<PaddleOcrPolygon> found = [];
        int offset = 0;
        for (int c = 0; c < contourCounts.Count; c++)
        {
            int count = contourCounts[c];
            if (TryBuild(contours, offset, count, prediction, mapWidth, mapHeight, options,
                    sourceWidth, sourceHeight, out PaddleOcrPolygon? polygon))
                found.Add(polygon!);
            offset += count;
        }
        SortReadingOrder(found);
        return [.. found];
    }

    private readonly record struct PointF(float X, float Y);

    private static bool TryBuild(List<PointF> contours, int offset, int count, ReadOnlySpan<float> prediction,
        int mapWidth, int mapHeight, PaddleOcrSealDetectorOptions options, int sourceWidth, int sourceHeight,
        out PaddleOcrPolygon? polygon)
    {
        polygon = null;
        if (count < 3) return false;
        Span<DbPostprocess.Point> raw = count <= 4096 ? stackalloc DbPostprocess.Point[count] : new DbPostprocess.Point[count];
        for (int i = 0; i < count; i++)
        {
            PointF p = contours[offset + i];
            raw[i] = new DbPostprocess.Point(p.X, p.Y);
        }
        int chain = CompressChain(raw);
        raw = raw[..chain];
        if (raw.Length < 3) return false;
        DbPostprocess.Point[] simplified = ApproxPoly(raw, 0.002 * Perimeter(raw));
        if (simplified.Length < 4) return false;
        float score = DbPostprocess.MeanScore(prediction, mapWidth, mapHeight, simplified);
        if (!MathCompat.IsFinite(score) || score < options.BoxThreshold) return false;
        DbPostprocess.Point[] expanded = DbPostprocess.ExpandPolygon(simplified, options.UnclipRatio);
        if (expanded.Length < 3) return false;
        if (!DbPostprocess.TryShortestSide(expanded, out float shortest) || shortest < 5) return false;

        PaddleOcrPoint[] mapped = new PaddleOcrPoint[expanded.Length];
        int kept = 0;
        for (int i = 0; i < expanded.Length; i++)
        {
            float x = MathF.Round(expanded[i].X / mapWidth * sourceWidth, MidpointRounding.ToEven);
            float y = MathF.Round(expanded[i].Y / mapHeight * sourceHeight, MidpointRounding.ToEven);
            x = Clamp(x, sourceWidth);
            y = Clamp(y, sourceHeight);
            if (kept > 0 && mapped[kept - 1].X == x && mapped[kept - 1].Y == y) continue;
            mapped[kept++] = new PaddleOcrPoint(x, y);
        }
        if (kept >= 2 && mapped[0].X == mapped[kept - 1].X && mapped[0].Y == mapped[kept - 1].Y) kept--;
        if (kept < 3) return false;
        if (kept != mapped.Length)
        {
            PaddleOcrPoint[] trimmed = new PaddleOcrPoint[kept];
            Array.Copy(mapped, trimmed, kept);
            mapped = trimmed;
        }
        polygon = new PaddleOcrPolygon { Points = mapped, Score = score };
        return true;
    }

    private static void TraceContours(byte[] bitmap, int width, int height, int maxContours,
        List<PointF> contours, List<int> counts)
    {
        byte[] seen = new byte[bitmap.Length];
        int[] queue = new int[bitmap.Length];
        // Clockwise 8-neighborhood starting at east.
        ReadOnlySpan<int> dx = [1, 1, 0, -1, -1, -1, 0, 1];
        ReadOnlySpan<int> dy = [0, 1, 1, 1, 0, -1, -1, -1];
        for (int y = 0; y < height && counts.Count < maxContours; y++)
        {
            int row = y * width;
            for (int x = 0; x < width && counts.Count < maxContours; x++)
            {
                int start = row + x;
                if (bitmap[start] == 0 || seen[start] != 0) continue;
                int before = contours.Count;
                TraceOne(bitmap, width, height, x, y, dx, dy, contours);
                counts.Add(contours.Count - before);
                Flood(bitmap, seen, queue, width, height, x, y);
            }
        }
    }

    private static void TraceOne(byte[] bitmap, int width, int height, int startX, int startY,
        ReadOnlySpan<int> dx, ReadOnlySpan<int> dy, List<PointF> contours)
    {
        int cx = startX, cy = startY;
        // The scan hits the topmost row of the component first, so the pixel
        // above the seed is background. Searching clockwise from that entry
        // walks the outer border once.
        int backDir = 6;
        int firstNx = int.MinValue, firstNy = int.MinValue;
        bool hasFirst = false;
        int guard = width * height + 2;
        contours.Add(new PointF(startX, startY));
        while (guard-- > 0)
        {
            int found = -1, nx = 0, ny = 0;
            for (int k = 1; k <= 8; k++)
            {
                int dir = (backDir + k) & 7;
                int tx = cx + dx[dir], ty = cy + dy[dir];
                if ((uint)tx >= (uint)width || (uint)ty >= (uint)height) continue;
                if (bitmap[ty * width + tx] == 0) continue;
                found = dir;
                nx = tx;
                ny = ty;
                break;
            }
            if (found < 0) return;
            if (!hasFirst)
            {
                hasFirst = true;
                firstNx = nx;
                firstNy = ny;
            }
            else if (cx == startX && cy == startY && nx == firstNx && ny == firstNy)
                return;
            backDir = (found + 4) & 7;
            cx = nx;
            cy = ny;
            if (cx != startX || cy != startY)
                contours.Add(new PointF(cx, cy));
        }
    }

    private static void Flood(byte[] bitmap, byte[] seen, int[] queue, int width, int height, int sx, int sy)
    {
        int seed = sy * width + sx;
        if (seen[seed] != 0) return;
        int head = 0, tail = 0;
        queue[tail++] = seed;
        seen[seed] = 1;
        while (head < tail)
        {
            int index = queue[head++];
            int x = index % width, y = index / width;
            for (int dy = -1; dy <= 1; dy++)
            {
                int ny = y + dy;
                if ((uint)ny >= (uint)height) continue;
                int row = ny * width;
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx;
                    if ((uint)nx >= (uint)width) continue;
                    int next = row + nx;
                    if (bitmap[next] == 0 || seen[next] != 0) continue;
                    seen[next] = 1;
                    queue[tail++] = next;
                }
            }
        }
    }

    // CHAIN_APPROX_SIMPLE: drop pixels that continue the same 8-step.
    private static int CompressChain(Span<DbPostprocess.Point> points)
    {
        int count = points.Length;
        if (count < 3) return count;
        int write = 0;
        for (int i = 0; i < count; i++)
        {
            DbPostprocess.Point prev = points[(i + count - 1) % count];
            DbPostprocess.Point cur = points[i];
            DbPostprocess.Point next = points[(i + 1) % count];
            int dx1 = Math.Sign(cur.X - prev.X), dy1 = Math.Sign(cur.Y - prev.Y);
            int dx2 = Math.Sign(next.X - cur.X), dy2 = Math.Sign(next.Y - cur.Y);
            if (dx1 != dx2 || dy1 != dy2)
                points[write++] = cur;
        }
        return write;
    }

    private static DbPostprocess.Point[] ApproxPoly(Span<DbPostprocess.Point> points, double epsilon)
    {
        int n = points.Length;
        if (n < 3) return points.ToArray();
        // cv2.approxPolyDP closed: the chain is treated as ending where it starts.
        DbPostprocess.Point[] chain = new DbPostprocess.Point[n + 1];
        for (int i = 0; i < n; i++) chain[i] = points[i];
        chain[n] = points[0];
        bool[] keep = new bool[chain.Length];
        keep[0] = true;
        keep[n] = true;
        DouglasPeucker(chain, 0, n, epsilon, keep);
        int kept = 0;
        for (int i = 0; i < n; i++) if (keep[i]) kept++;
        if (kept < 3) return points.ToArray();
        DbPostprocess.Point[] result = new DbPostprocess.Point[kept];
        int w = 0;
        for (int i = 0; i < n; i++) if (keep[i]) result[w++] = chain[i];
        return result;
    }

    private static void DouglasPeucker(DbPostprocess.Point[] chain, int start, int end, double epsilon, bool[] keep)
    {
        if (end <= start + 1) return;
        double max = 0;
        int index = start;
        DbPostprocess.Point a = chain[start], b = chain[end];
        for (int i = start + 1; i < end; i++)
        {
            double distance = DistanceToLine(chain[i], a, b);
            if (distance > max)
            {
                max = distance;
                index = i;
            }
        }
        if (max <= epsilon) return;
        keep[index] = true;
        DouglasPeucker(chain, start, index, epsilon, keep);
        DouglasPeucker(chain, index, end, epsilon, keep);
    }

    private static double DistanceToLine(DbPostprocess.Point p, DbPostprocess.Point a, DbPostprocess.Point b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y;
        double len = Math.Sqrt(dx * dx + dy * dy);
        if (len <= 1e-8) return Math.Sqrt((p.X - a.X) * (p.X - a.X) + (p.Y - a.Y) * (p.Y - a.Y));
        return Math.Abs(dx * (a.Y - p.Y) - (a.X - p.X) * dy) / len;
    }

    private static double Perimeter(ReadOnlySpan<DbPostprocess.Point> points)
    {
        double length = 0;
        for (int i = 0; i < points.Length; i++)
        {
            DbPostprocess.Point a = points[i], b = points[(i + 1) % points.Length];
            double dx = b.X - a.X, dy = b.Y - a.Y;
            length += Math.Sqrt(dx * dx + dy * dy);
        }
        return length;
    }

    private static void SortReadingOrder(List<PaddleOcrPolygon> found)
    {
        found.Sort(static (a, b) =>
        {
            int y = a.Points[0].Y.CompareTo(b.Points[0].Y);
            return y != 0 ? y : a.Points[0].X.CompareTo(b.Points[0].X);
        });
        for (int i = 0; i < found.Count - 1; i++)
        {
            for (int j = i; j >= 0; j--)
            {
                PaddleOcrPoint current = found[j].Points[0], next = found[j + 1].Points[0];
                if (MathF.Abs(next.Y - current.Y) >= 10 || next.X >= current.X)
                    break;
                (found[j], found[j + 1]) = (found[j + 1], found[j]);
            }
        }
    }

    private static void Dilate2x2(byte[] bitmap, byte[] dilated, int width, int height, int pixels)
    {
        Array.Clear(dilated, 0, pixels);
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
        {
            int dst = y * width + x;
            if (bitmap[dst] != 0 || (x + 1 < width && bitmap[dst + 1] != 0) ||
                (y + 1 < height && bitmap[dst + width] != 0) ||
                (x + 1 < width && y + 1 < height && bitmap[dst + width + 1] != 0))
                dilated[dst] = 1;
        }
        Buffer.BlockCopy(dilated, 0, bitmap, 0, pixels);
    }

    private static float Clamp(float value, int limit) => value < 0 ? 0 : value > limit ? limit : value;
}
