using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class PixelRow
{
    internal static void Gather32Scalar(ReadOnlySpan<byte> row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        fixed (byte* p = row)
            Gather32Scalar(p, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void Gather32Scalar(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        int bOff = rgba ? 2 : 0, rOff = rgba ? 0 : 2;
        for (int x = 0; x < destinationWidth; x++)
        {
            int sx = offsets[x], sx1 = Math.Min(sx + 1, sourceWidth - 1);
            short c0 = coefficients[x * 2], c1 = coefficients[x * 2 + 1];
            byte* p0 = row + sx * 4, p1 = row + sx1 * 4;
            int d = x * 3;
            destination[d] = p0[bOff] * c0 + p1[bOff] * c1;
            destination[d + 1] = p0[1] * c0 + p1[1] * c1;
            destination[d + 2] = p0[rOff] * c0 + p1[rOff] * c1;
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void GatherRgb24(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination)
    {
        for (int x = 0; x < destinationWidth; x++)
        {
            int sx = offsets[x], sx1 = Math.Min(sx + 1, sourceWidth - 1);
            short c0 = coefficients[x * 2], c1 = coefficients[x * 2 + 1];
            int s0 = sx * 3, s1 = sx1 * 3, d = x * 3;
            destination[d] = row[s0 + 2] * c0 + row[s1 + 2] * c1;
            destination[d + 1] = row[s0 + 1] * c0 + row[s1 + 1] * c1;
            destination[d + 2] = row[s0] * c0 + row[s1] * c1;
        }
    }
}
