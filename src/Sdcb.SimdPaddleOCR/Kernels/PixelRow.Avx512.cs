#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class PixelRow
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Gather32Avx512(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int last = sourceWidth - 1;
        Vector512<int> lastV = Vector512.Create(last);
        Vector512<int> one = Vector512.Create(1);
        Vector512<int> ff = Vector512.Create(255);
        fixed (int* off = offsets)
        fixed (short* coeff = coefficients)
        {
            int x = 0;
            for (; x <= destinationWidth - 16; x += 16)
            {
                Vector512<int> sx = Vector512.Create(Avx.LoadDquVector256(off + x), Avx.LoadDquVector256(off + x + 8));
                Vector512<int> sx1 = Avx512F.Min(Avx512F.Add(sx, one), lastV);
                Vector512<int> pix0 = Vector512.Create(
                    Avx2.GatherVector256((int*)row, sx.GetLower(), 4),
                    Avx2.GatherVector256((int*)row, sx.GetUpper(), 4));
                Vector512<int> pix1 = Vector512.Create(
                    Avx2.GatherVector256((int*)row, sx1.GetLower(), 4),
                    Avx2.GatherVector256((int*)row, sx1.GetUpper(), 4));
                SplitCoeffAvx512(coeff + x * 2, out Vector512<int> c0, out Vector512<int> c1);
                ChannelsAvx512(pix0, rgba, ff, out Vector512<int> b0, out Vector512<int> g0, out Vector512<int> r0);
                ChannelsAvx512(pix1, rgba, ff, out Vector512<int> b1, out Vector512<int> g1, out Vector512<int> r1);
                StoreBgrAvx512(destination, x,
                    Avx512F.Add(Avx512F.MultiplyLow(b0, c0), Avx512F.MultiplyLow(b1, c1)),
                    Avx512F.Add(Avx512F.MultiplyLow(g0, c0), Avx512F.MultiplyLow(g1, c1)),
                    Avx512F.Add(Avx512F.MultiplyLow(r0, c0), Avx512F.MultiplyLow(r1, c1)));
            }
            Gather32ScalarRange(row, last, x, destinationWidth, offsets, coefficients, destination, rgba);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SplitCoeffAvx512(short* coeff, out Vector512<int> c0, out Vector512<int> c1)
    {
        SplitCoeffAvx(coeff, out Vector256<int> c0Lo, out Vector256<int> c1Lo);
        SplitCoeffAvx(coeff + 16, out Vector256<int> c0Hi, out Vector256<int> c1Hi);
        c0 = Vector512.Create(c0Lo, c0Hi);
        c1 = Vector512.Create(c1Lo, c1Hi);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ChannelsAvx512(Vector512<int> pix, bool rgba, Vector512<int> ff,
        out Vector512<int> b, out Vector512<int> g, out Vector512<int> r)
    {
        Vector512<int> c0 = Avx512F.And(pix, ff);
        Vector512<int> c1 = Avx512F.And(Avx512F.ShiftRightLogical(pix, 8), ff);
        Vector512<int> c2 = Avx512F.And(Avx512F.ShiftRightLogical(pix, 16), ff);
        g = c1;
        if (rgba) { b = c2; r = c0; }
        else { b = c0; r = c2; }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreBgrAvx512(int[] destination, int x, Vector512<int> b, Vector512<int> g, Vector512<int> r)
    {
        for (int lane = 0; lane < 16; lane++)
        {
            int d = (x + lane) * 3;
            destination[d] = b.GetElement(lane);
            destination[d + 1] = g.GetElement(lane);
            destination[d + 2] = r.GetElement(lane);
        }
    }
}
#endif
