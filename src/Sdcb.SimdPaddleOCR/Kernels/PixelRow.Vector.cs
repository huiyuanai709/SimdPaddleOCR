using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class PixelRow
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Gather32Vec(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int width = Vector<int>.Count;
        int last = sourceWidth - 1;
        int x = 0;
        for (; x <= destinationWidth - width; x += width)
        {
            Vector<int> b0 = default, g0 = default, r0 = default;
            Vector<int> b1 = default, g1 = default, r1 = default;
            Vector<int> c0 = default, c1 = default;
            for (int lane = 0; lane < width; lane++)
            {
                int sx = offsets[x + lane], sx1 = sx < last ? sx + 1 : last;
                int pix0 = Unsafe.ReadUnaligned<int>(row + sx * 4);
                int pix1 = Unsafe.ReadUnaligned<int>(row + sx1 * 4);
                c0 = c0.WithElement(lane, coefficients[(x + lane) * 2]);
                c1 = c1.WithElement(lane, coefficients[(x + lane) * 2 + 1]);
                SplitBgra(pix0, rgba, out int pb, out int pg, out int pr);
                SplitBgra(pix1, rgba, out int qb, out int qg, out int qr);
                b0 = b0.WithElement(lane, pb);
                g0 = g0.WithElement(lane, pg);
                r0 = r0.WithElement(lane, pr);
                b1 = b1.WithElement(lane, qb);
                g1 = g1.WithElement(lane, qg);
                r1 = r1.WithElement(lane, qr);
            }
            StoreBgr(destination, x, b0 * c0 + b1 * c1, g0 * c0 + g1 * c1, r0 * c0 + r1 * c1, width);
        }
        Gather32ScalarRange(row, last, x, destinationWidth, offsets, coefficients, destination, rgba);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SplitBgra(int pix, bool rgba, out int b, out int g, out int r)
    {
        int c0 = pix & 255, c1 = (pix >> 8) & 255, c2 = (pix >> 16) & 255;
        if (rgba) { b = c2; g = c1; r = c0; }
        else { b = c0; g = c1; r = c2; }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreBgr(int[] destination, int x, Vector<int> b, Vector<int> g, Vector<int> r, int width)
    {
        for (int lane = 0; lane < width; lane++)
        {
            int d = (x + lane) * 3;
            destination[d] = b.GetElement(lane);
            destination[d + 1] = g.GetElement(lane);
            destination[d + 2] = r.GetElement(lane);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Gather32ScalarRange(byte* row, int last, int start, int end,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int bOff = rgba ? 2 : 0, rOff = rgba ? 0 : 2;
        for (int x = start; x < end; x++)
        {
            int sx = offsets[x], sx1 = sx < last ? sx + 1 : last;
            short c0 = coefficients[x * 2], c1 = coefficients[x * 2 + 1];
            byte* p0 = row + sx * 4, p1 = row + sx1 * 4;
            int d = x * 3;
            destination[d] = p0[bOff] * c0 + p1[bOff] * c1;
            destination[d + 1] = p0[1] * c0 + p1[1] * c1;
            destination[d + 2] = p0[rOff] * c0 + p1[rOff] * c1;
        }
    }
}
