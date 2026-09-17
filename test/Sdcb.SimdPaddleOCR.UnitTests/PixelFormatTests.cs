using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class PixelFormatTests
{
    [Fact]
    public void DetPreprocess_FourFormats_MatchBgr24()
    {
        (int width, int height, byte[] bgr) = MakeBgr(37, 29);
        CompareDet(bgr, width, height, ToRgb24(bgr), ImagePixelFormat.Rgb24, width * 3);
        CompareDet(bgr, width, height, To32(bgr, rgba: false), ImagePixelFormat.Bgra32, width * 4);
        CompareDet(bgr, width, height, To32(bgr, rgba: true), ImagePixelFormat.Rgba32, width * 4);
    }

    [Fact]
    public void ClsPreprocess_FourFormats_MatchBgr24()
    {
        (int width, int height, byte[] bgr) = MakeBgr(41, 23);
        CompareCls(bgr, width, height, ToRgb24(bgr), ImagePixelFormat.Rgb24, width * 3);
        CompareCls(bgr, width, height, To32(bgr, rgba: false), ImagePixelFormat.Bgra32, width * 4);
        CompareCls(bgr, width, height, To32(bgr, rgba: true), ImagePixelFormat.Rgba32, width * 4);
    }

    [Fact]
    public void RecPreprocess_FourFormats_MatchBgr24()
    {
        (int width, int height, byte[] bgr) = MakeBgr(53, 17);
        CompareRec(bgr, width, height, ToRgb24(bgr), ImagePixelFormat.Rgb24, width * 3);
        CompareRec(bgr, width, height, To32(bgr, rgba: false), ImagePixelFormat.Bgra32, width * 4);
        CompareRec(bgr, width, height, To32(bgr, rgba: true), ImagePixelFormat.Rgba32, width * 4);
    }

    [Fact]
    public void Crop_FourFormats_MatchBgr24()
    {
        (int width, int height, byte[] bgr) = MakeBgr(48, 36);
        PaddleOcrDetectionBox box = new(4, 5, 43, 6, 42, 31, 5, 30, 1);
        byte[] expected = PPOCRCrop.Extract(bgr, width, height, width * 3, box, out int ow, out int oh);
        AssertCropsEqual(expected, PPOCRCrop.Extract(ToRgb24(bgr), width, height, width * 3, box,
            out int w1, out int h1, ImagePixelFormat.Rgb24), ow, oh, w1, h1);
        AssertCropsEqual(expected, PPOCRCrop.Extract(To32(bgr, rgba: false), width, height, width * 4, box,
            out int w2, out int h2, ImagePixelFormat.Bgra32), ow, oh, w2, h2);
        AssertCropsEqual(expected, PPOCRCrop.Extract(To32(bgr, rgba: true), width, height, width * 4, box,
            out int w3, out int h3, ImagePixelFormat.Rgba32), ow, oh, w3, h3);
    }

    [Fact]
    public void PaddedStride32_FourStages_MatchBgr24()
    {
        (int width, int height, byte[] bgr) = MakeBgr(37, 29);
        const int pad = 16;
        byte[] bgra = To32Padded(bgr, width, height, rgba: false, pad);
        byte[] rgba = To32Padded(bgr, width, height, rgba: true, pad);
        int stride = width * 4 + pad;
        CompareDet(bgr, width, height, bgra, ImagePixelFormat.Bgra32, stride);
        CompareDet(bgr, width, height, rgba, ImagePixelFormat.Rgba32, stride);
        CompareCls(bgr, width, height, bgra, ImagePixelFormat.Bgra32, stride);
        CompareCls(bgr, width, height, rgba, ImagePixelFormat.Rgba32, stride);
        CompareRec(bgr, width, height, bgra, ImagePixelFormat.Bgra32, stride);
        CompareRec(bgr, width, height, rgba, ImagePixelFormat.Rgba32, stride);

        PaddleOcrDetectionBox box = new(4, 5, 32, 6, 31, 24, 5, 23, 1);
        byte[] expected = PPOCRCrop.Extract(bgr, width, height, width * 3, box, out int ow, out int oh);
        AssertCropsEqual(expected, PPOCRCrop.Extract(bgra, width, height, stride, box,
            out int w1, out int h1, ImagePixelFormat.Bgra32), ow, oh, w1, h1);
        AssertCropsEqual(expected, PPOCRCrop.Extract(rgba, width, height, stride, box,
            out int w2, out int h2, ImagePixelFormat.Rgba32), ow, oh, w2, h2);
    }

    [Fact]
    public void Gather32_IsaMatchesScalar()
    {
        int sourceWidth = 37, destinationWidth = 64;
        byte[] row = new byte[sourceWidth * 4];
        int[] offsets = new int[destinationWidth];
        short[] coefficients = new short[destinationWidth * 2];
        var rng = new Random(7);
        for (int i = 0; i < row.Length; i++) row[i] = (byte)rng.Next(256);
        for (int x = 0; x < destinationWidth; x++)
        {
            offsets[x] = rng.Next(0, sourceWidth);
            short c0 = (short)rng.Next(0, 2049);
            coefficients[x * 2] = c0;
            coefficients[x * 2 + 1] = (short)(2048 - c0);
        }
        int[] scalar = new int[destinationWidth * 3];
        int[] dispatched = new int[destinationWidth * 3];
        PixelRow.Gather32Scalar(row, sourceWidth, destinationWidth, offsets, coefficients, scalar, rgba: true);
        PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, dispatched, rgba: true);
        Assert.Equal(scalar, dispatched);
        PixelRow.Gather32Scalar(row, sourceWidth, destinationWidth, offsets, coefficients, scalar, rgba: false);
        PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, dispatched, rgba: false);
        Assert.Equal(scalar, dispatched);
    }

    [Fact]
    public void ResolveStride_ZeroUsesBytesPerPixel()
    {
        Assert.Equal(30, ImagePixels.ResolveStride(10, 0, ImagePixelFormat.Bgr24));
        Assert.Equal(30, ImagePixels.ResolveStride(10, 0, ImagePixelFormat.Rgb24));
        Assert.Equal(40, ImagePixels.ResolveStride(10, 0, ImagePixelFormat.Bgra32));
        Assert.Equal(40, ImagePixels.ResolveStride(10, 0, ImagePixelFormat.Rgba32));
        Assert.Equal(48, ImagePixels.ResolveStride(10, 48, ImagePixelFormat.Bgra32));
        Assert.Throws<ArgumentException>(() => ImagePixels.ResolveStride(10, 20, ImagePixelFormat.Rgba32));
    }

    private static void CompareDet(byte[] bgr, int width, int height, byte[] other,
        ImagePixelFormat format, int stride)
    {
        int rw = 64, rh = 96;
        float[] expected = new float[rw * rh * 3];
        float[] actual = new float[expected.Length];
        PPOCRPreprocess.Det(bgr, width, height, width * 3, rw, rh, expected);
        PPOCRPreprocess.Det(other, width, height, stride, rw, rh, actual, format);
        Assert.Equal(expected, actual);
    }

    private static void CompareCls(byte[] bgr, int width, int height, byte[] other,
        ImagePixelFormat format, int stride)
    {
        float[] expected = new float[3 * 80 * 160];
        float[] actual = new float[expected.Length];
        PPOCRPreprocess.Cls(bgr, width, height, width * 3, expected);
        PPOCRPreprocess.Cls(other, width, height, stride, actual, format);
        Assert.Equal(expected, actual);
    }

    private static void CompareRec(byte[] bgr, int width, int height, byte[] other,
        ImagePixelFormat format, int stride)
    {
        int target = 160;
        float[] expected = new float[3 * 48 * target];
        float[] actual = new float[expected.Length];
        Assert.Equal(
            PPOCRPreprocess.Rec(bgr, width, height, width * 3, target, expected),
            PPOCRPreprocess.Rec(other, width, height, stride, target, actual, format));
        Assert.Equal(expected, actual);
    }

    private static void AssertCropsEqual(byte[] expected, byte[] actual, int ow, int oh, int w, int h)
    {
        Assert.Equal(ow, w);
        Assert.Equal(oh, h);
        Assert.Equal(expected, actual);
    }

    private static (int Width, int Height, byte[] Bgr) MakeBgr(int width, int height)
    {
        byte[] bgr = new byte[width * height * 3];
        var rng = new Random(11);
        rng.NextBytes(bgr);
        return (width, height, bgr);
    }

    private static byte[] ToRgb24(byte[] bgr)
    {
        byte[] rgb = new byte[bgr.Length];
        for (int i = 0; i < bgr.Length; i += 3)
        {
            rgb[i] = bgr[i + 2];
            rgb[i + 1] = bgr[i + 1];
            rgb[i + 2] = bgr[i];
        }
        return rgb;
    }

    private static byte[] To32(byte[] bgr, bool rgba)
    {
        byte[] dest = new byte[bgr.Length / 3 * 4];
        for (int i = 0, o = 0; i < bgr.Length; i += 3, o += 4)
        {
            dest[o] = rgba ? bgr[i + 2] : bgr[i];
            dest[o + 1] = bgr[i + 1];
            dest[o + 2] = rgba ? bgr[i] : bgr[i + 2];
            dest[o + 3] = 255;
        }
        return dest;
    }

    private static byte[] To32Padded(byte[] bgr, int width, int height, bool rgba, int pad)
    {
        int stride = width * 4 + pad;
        byte[] dest = new byte[stride * height];
        var rng = new Random(19);
        rng.NextBytes(dest);
        for (int y = 0; y < height; y++)
        {
            int src = y * width * 3;
            int dst = y * stride;
            for (int x = 0; x < width; x++, src += 3, dst += 4)
            {
                dest[dst] = rgba ? bgr[src + 2] : bgr[src];
                dest[dst + 1] = bgr[src + 1];
                dest[dst + 2] = rgba ? bgr[src] : bgr[src + 2];
                dest[dst + 3] = 255;
            }
        }
        return dest;
    }
}
