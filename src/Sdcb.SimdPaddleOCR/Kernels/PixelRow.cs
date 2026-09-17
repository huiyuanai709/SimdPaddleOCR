using System.Numerics;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>
/// Horizontal bilinear gather that reads 32-bit pixels (BGRA or RGBA) and
/// writes B,G,R accumulators. 24-bit BGR stays on the historical scalar loop.
/// </summary>
internal static unsafe partial class PixelRow
{
    internal static void Gather32(ReadOnlySpan<byte> row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
        fixed (byte* p = row)
            Gather32(p, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
    }

    internal static void Gather32(byte* row, int sourceWidth, int destinationWidth,
        int[] offsets, short[] coefficients, int[] destination, bool rgba)
    {
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
            Gather32Avx512(row, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
        else if (Avx2.IsSupported)
            Gather32Avx(row, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
        else if (AdvSimd.Arm64.IsSupported)
            Gather32AdvSimd(row, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
        else
#endif
        if (Vector.IsHardwareAccelerated && Vector<int>.Count is 4 or 8)
            Gather32Vec(row, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
        else
            Gather32Scalar(row, sourceWidth, destinationWidth, offsets, coefficients, destination, rgba);
    }
}
