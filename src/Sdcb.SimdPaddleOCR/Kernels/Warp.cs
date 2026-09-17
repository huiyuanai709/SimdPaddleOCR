using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Warp
{
    internal readonly struct PixelAccess
    {
        public readonly int BytesPerPixel;
        public readonly bool SwapRedBlue;

        public PixelAccess(int bytesPerPixel, bool swapRedBlue)
        {
            BytesPerPixel = bytesPerPixel;
            SwapRedBlue = swapRedBlue;
        }

        public int Bpp => BytesPerPixel == 0 ? 3 : BytesPerPixel;

        public static PixelAccess Of(ImagePixelFormat format) =>
            new(ImagePixels.BytesPerPixel(format), ImagePixels.SwapRedBlue(format));
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void MapRow(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int y, double v, ref int x, PixelAccess access = default)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && unrotatedWidth >= 8)
            MapRowAvx512(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x,
                access);
        else if (Avx2.IsSupported && unrotatedWidth >= 4)
            MapRowAvx(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x,
                access);
        else
        #endif
        if (Vector.IsHardwareAccelerated && Vector<double>.Count >= 2 &&
            unrotatedWidth >= Vector<double>.Count)
            MapRowVector(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x,
                access);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void SampleCubic(byte* source, int width, int height, int stride,
        double x, double y, byte* destination, int destinationOffset, PixelAccess access = default)
    {
        int xBase = (int)Math.Floor(x), yBase = (int)Math.Floor(y);
        if (xBase >= 1 && xBase < width - 3 && yBase >= 1 && yBase < height - 2)
        {
            #if !NETSTANDARD2_0
            if (Avx512F.IsSupported)
            {
                SampleCubicAvx512(source, stride, x, y, xBase, yBase, destination, destinationOffset,
                    access);
                return;
            }
            else if (Avx.IsSupported && Avx2.IsSupported)
            {
                SampleCubicAvx(source, stride, x, y, xBase, yBase, destination, destinationOffset,
                    access);
                return;
            }
            else
            #endif
            if (Vector.IsHardwareAccelerated && Vector<double>.Count == 4)
            {
                SampleCubicVector(source, stride, x, y, xBase, yBase, destination, destinationOffset,
                    access);
                return;
            }
        }
        SampleCubicScalar(source, width, height, stride, x, y, destination, destinationOffset, access);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SampleMappedPixel(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedHeight, bool rotateVertical,
        double pixelX, double pixelY, int x, int y, PixelAccess access)
    {
        int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x;
        int destinationY = rotateVertical ? x : y;
        int destination = checked((destinationY * outputWidth + destinationX) * 3);
        SampleCubic(sourcePtr, sourceWidth, sourceHeight, sourceStride,
            pixelX, pixelY, cropPtr, destination, access);
    }
}
