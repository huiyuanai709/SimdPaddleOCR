#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class PixelRow
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Gather32AdvSimd(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int last = sourceWidth - 1;
        Vector128<int> ff = Vector128.Create(255);
        int x = 0;
        for (; x <= destinationWidth - 4; x += 4)
        {
            int s0 = offsets[x], s1 = offsets[x + 1], s2 = offsets[x + 2], s3 = offsets[x + 3];
            Vector128<int> pix0 = Vector128.Create(
                Unsafe.ReadUnaligned<int>(row + s0 * 4),
                Unsafe.ReadUnaligned<int>(row + s1 * 4),
                Unsafe.ReadUnaligned<int>(row + s2 * 4),
                Unsafe.ReadUnaligned<int>(row + s3 * 4));
            Vector128<int> pix1 = Vector128.Create(
                Unsafe.ReadUnaligned<int>(row + (s0 < last ? s0 + 1 : last) * 4),
                Unsafe.ReadUnaligned<int>(row + (s1 < last ? s1 + 1 : last) * 4),
                Unsafe.ReadUnaligned<int>(row + (s2 < last ? s2 + 1 : last) * 4),
                Unsafe.ReadUnaligned<int>(row + (s3 < last ? s3 + 1 : last) * 4));
            Vector128<int> c0 = Vector128.Create(
                (int)coefficients[x * 2], coefficients[x * 2 + 2],
                coefficients[x * 2 + 4], coefficients[x * 2 + 6]);
            Vector128<int> c1 = Vector128.Create(
                (int)coefficients[x * 2 + 1], coefficients[x * 2 + 3],
                coefficients[x * 2 + 5], coefficients[x * 2 + 7]);
            ChannelsAdv(pix0, rgba, ff, out Vector128<int> b0, out Vector128<int> g0, out Vector128<int> r0);
            ChannelsAdv(pix1, rgba, ff, out Vector128<int> b1, out Vector128<int> g1, out Vector128<int> r1);
            StoreBgrAdv(destination, x,
                AdvSimd.Add(AdvSimd.Multiply(b0, c0), AdvSimd.Multiply(b1, c1)),
                AdvSimd.Add(AdvSimd.Multiply(g0, c0), AdvSimd.Multiply(g1, c1)),
                AdvSimd.Add(AdvSimd.Multiply(r0, c0), AdvSimd.Multiply(r1, c1)));
        }
        Gather32ScalarRange(row, last, x, destinationWidth, offsets, coefficients, destination, rgba);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ChannelsAdv(Vector128<int> pix, bool rgba, Vector128<int> ff,
        out Vector128<int> b, out Vector128<int> g, out Vector128<int> r)
    {
        Vector128<int> ch0 = AdvSimd.And(pix, ff);
        Vector128<int> ch1 = AdvSimd.And(AdvSimd.ShiftRightLogical(pix, 8), ff);
        Vector128<int> ch2 = AdvSimd.And(AdvSimd.ShiftRightLogical(pix, 16), ff);
        g = ch1;
        if (rgba) { b = ch2; r = ch0; }
        else { b = ch0; r = ch2; }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreBgrAdv(int[] destination, int x, Vector128<int> b, Vector128<int> g, Vector128<int> r)
    {
        for (int lane = 0; lane < 4; lane++)
        {
            int d = (x + lane) * 3;
            destination[d] = b.GetElement(lane);
            destination[d + 1] = g.GetElement(lane);
            destination[d + 2] = r.GetElement(lane);
        }
    }
}
#endif
