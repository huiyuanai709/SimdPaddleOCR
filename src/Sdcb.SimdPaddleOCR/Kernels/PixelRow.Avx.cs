#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class PixelRow
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Gather32Avx(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int last = sourceWidth - 1;
        Vector256<int> lastV = Vector256.Create(last);
        Vector256<int> one = Vector256.Create(1);
        Vector256<int> ff = Vector256.Create(255);
        fixed (int* off = offsets)
        fixed (short* coeff = coefficients)
        {
            int x = 0;
            for (; x <= destinationWidth - 8; x += 8)
            {
                Vector256<int> sx = Avx.LoadDquVector256(off + x);
                Vector256<int> sx1 = Avx2.Min(Avx2.Add(sx, one), lastV);
                Vector256<int> pix0 = Avx2.GatherVector256((int*)row, sx, 4);
                Vector256<int> pix1 = Avx2.GatherVector256((int*)row, sx1, 4);
                SplitCoeffAvx(coeff + x * 2, out Vector256<int> c0, out Vector256<int> c1);
                ChannelsAvx(pix0, rgba, ff, out Vector256<int> b0, out Vector256<int> g0, out Vector256<int> r0);
                ChannelsAvx(pix1, rgba, ff, out Vector256<int> b1, out Vector256<int> g1, out Vector256<int> r1);
                StoreBgrAvx(destination, x,
                    Avx2.Add(Avx2.MultiplyLow(b0, c0), Avx2.MultiplyLow(b1, c1)),
                    Avx2.Add(Avx2.MultiplyLow(g0, c0), Avx2.MultiplyLow(g1, c1)),
                    Avx2.Add(Avx2.MultiplyLow(r0, c0), Avx2.MultiplyLow(r1, c1)));
            }
            Gather32ScalarRange(row, last, x, destinationWidth, offsets, coefficients, destination, rgba);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SplitCoeffAvx(short* coeff, out Vector256<int> c0, out Vector256<int> c1)
    {
        Vector256<short> packed = Avx.LoadDquVector256(coeff);
        Vector256<int> lo = Avx2.ConvertToVector256Int32(packed.GetLower());
        Vector256<int> hi = Avx2.ConvertToVector256Int32(packed.GetUpper());
        Vector256<int> even = Vector256.Create(0, 2, 4, 6, 0, 2, 4, 6);
        Vector256<int> odd = Vector256.Create(1, 3, 5, 7, 1, 3, 5, 7);
        c0 = Avx2.Permute2x128(Avx2.PermuteVar8x32(lo, even), Avx2.PermuteVar8x32(hi, even), 0x20);
        c1 = Avx2.Permute2x128(Avx2.PermuteVar8x32(lo, odd), Avx2.PermuteVar8x32(hi, odd), 0x20);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ChannelsAvx(Vector256<int> pix, bool rgba, Vector256<int> ff,
        out Vector256<int> b, out Vector256<int> g, out Vector256<int> r)
    {
        Vector256<int> c0 = Avx2.And(pix, ff);
        Vector256<int> c1 = Avx2.And(Avx2.ShiftRightLogical(pix, 8), ff);
        Vector256<int> c2 = Avx2.And(Avx2.ShiftRightLogical(pix, 16), ff);
        g = c1;
        if (rgba) { b = c2; r = c0; }
        else { b = c0; r = c2; }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreBgrAvx(int[] destination, int x, Vector256<int> b, Vector256<int> g, Vector256<int> r)
    {
        for (int lane = 0; lane < 8; lane++)
        {
            int d = (x + lane) * 3;
            destination[d] = b.GetElement(lane);
            destination[d + 1] = g.GetElement(lane);
            destination[d + 2] = r.GetElement(lane);
        }
    }
}
#endif
