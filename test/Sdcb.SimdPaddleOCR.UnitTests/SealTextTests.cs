using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Sdcb.SimdPaddleOCR.ModelProvider;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.SealV4Mobile;
using Sdcb.SimdPaddleOCR.Models.SealV4Server;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class SealTextTests
{
    [Fact]
    public void InputSize_Min736_Scales640SquareTo736()
    {
        (int width, int height) = PaddleOcrSealDetector.ComputeInputSize(640, 640, new PaddleOcrSealDetectorOptions());
        Assert.Equal(736, width);
        Assert.Equal(736, height);
    }

    [Fact]
    public void InputSize_ResizeLong_RoundsUpTo128()
    {
        var options = new PaddleOcrSealDetectorOptions { LimitType = PaddleOcrSideLimit.ResizeLong, LimitSideLength = 736 };
        (int width, int height) = PaddleOcrSealDetector.ComputeInputSize(640, 640, options);
        Assert.Equal(768, width);
        Assert.Equal(768, height);
    }

    [Fact]
    public void InputSize_Max_ShrinksTheLongSide()
    {
        var options = new PaddleOcrSealDetectorOptions { LimitType = PaddleOcrSideLimit.Max, LimitSideLength = 736 };
        (int width, int height) = PaddleOcrSealDetector.ComputeInputSize(1000, 500, options);
        Assert.Equal(736, width);
        Assert.Equal(368, height);
    }

    [Fact]
    public void PolyPostprocess_FilledRectangle_ReturnsARing()
    {
        const int width = 96, height = 96;
        float[] prediction = new float[width * height];
        for (int y = 30; y < 50; y++)
            for (int x = 20; x < 70; x++)
                prediction[y * width + x] = 1f;
        PaddleOcrPolygon[] polygons = DbPolyPostprocess.Run(prediction, width, height,
            new PaddleOcrSealDetectorOptions(), width, height);
        Assert.NotEmpty(polygons);
        Assert.True(polygons[0].Points.Length >= 4);
        Assert.InRange(polygons[0].Score, 0.6f, 1.01f);
    }

    [Fact]
    public void Rectify_AxisAlignedRectangle_KeepsTheCropSize()
    {
        const int width = 120, height = 80;
        byte[] image = new byte[width * height * 3];
        Fill(image, width, 10, 20, 90, 40, 255, 255, 255);
        PaddleOcrPoint[] polygon =
        [
            new(10, 20), new(90, 20), new(90, 40), new(10, 40)
        ];
        byte[] strip = PaddleOcrSealRectify.Rectify(image, width, height, polygon, out int cropWidth, out int cropHeight);
        Assert.InRange(cropWidth, 70, 90);
        Assert.InRange(cropHeight, 16, 28);
        Assert.Equal(cropWidth * cropHeight * 3, strip.Length);
    }

    [Fact]
    public void Rectify_CurvedBand_IsWiderThanTheChord()
    {
        const int width = 400, height = 400;
        byte[] image = new byte[width * height * 3];
        PaddleOcrPoint[] polygon = ArcBand(200, 220, 150, 112, 200, 340, 24);
        double minX = polygon.Min(p => p.X), maxX = polygon.Max(p => p.X);
        byte[] strip = PaddleOcrSealRectify.Rectify(image, width, height, polygon, out int cropWidth, out int cropHeight);
        Assert.True(cropWidth > cropHeight, $"strip {cropWidth}x{cropHeight}");
        Assert.True(cropWidth > maxX - minX, $"strip width {cropWidth} chord {maxX - minX}");
        Assert.Equal(cropWidth * cropHeight * 3, strip.Length);
    }

    [Fact]
    public void EmbeddedSealModels_Open()
    {
        using Stream mobile = SealV4MobileModel.Detection.OpenRead();
        using Stream server = SealV4ServerModel.Detection.OpenRead();
        Assert.True(mobile.Length > 4_000_000);
        Assert.InRange(server.Length, 40_000_000, 100_000_000);
        Assert.Equal(PaddleOcrModelKind.Detection, SealV4MobileModel.Detection.Kind);
        Assert.Equal(PaddleOcrModelKind.Detection, SealV4ServerModel.Detection.Kind);
    }

    [Fact]
    public void Float16Bits_RoundTripCommonValues()
    {
        Assert.Equal(0f, Model.SingleFromFloat16Bits(0));
        Assert.Equal(1f, Model.SingleFromFloat16Bits(0x3C00));
        Assert.Equal(-2f, Model.SingleFromFloat16Bits(0xC000));
        Assert.Equal(0.5f, Model.SingleFromFloat16Bits(0x3800));
    }

    [Fact]
    public void MobileSeal_ReadsCompanyNameAndCenterText()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "assets", "seal.png");
        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        byte[] bgr = new byte[image.Width * image.Height * 3];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    int offset = (y * row.Length + x) * 3;
                    bgr[offset] = row[x].B;
                    bgr[offset + 1] = row[x].G;
                    bgr[offset + 2] = row[x].R;
                }
            }
        });

        var options = new PaddleOcrSealOptions
        {
            Detector = new PaddleOcrSealDetectorOptions { Backend = OcrBackend.Cpu },
            Recognizer = new PaddleOcrRecognizerOptions { Backend = OcrBackend.Cpu },
            DetIntraOpThreads = 2
        };
        using PaddleOcrSealAll seal = PaddleOcrSealAll.Load(
            SealV4MobileModel.Detection,
            ChineseV6TinyModel.Recognition,
            ChineseV6TinyModel.Dictionary,
            options);
        PaddleOcrSealResult result = seal.Run(bgr, image.Width, image.Height);
        Assert.NotEmpty(result.Lines);
        Assert.Contains(result.Lines, line => line.Polygon.Points.Length >= 4);
        Assert.Contains("专用章", result.Text, StringComparison.Ordinal);
        Assert.Contains("天津", result.Text, StringComparison.Ordinal);
    }

    private static PaddleOcrPoint[] ArcBand(float cx, float cy, float outer, float inner,
        float startDeg, float endDeg, int samples)
    {
        PaddleOcrPoint[] points = new PaddleOcrPoint[samples * 2];
        for (int i = 0; i < samples; i++)
        {
            double t = startDeg + (endDeg - startDeg) * i / (samples - 1);
            double rad = t * Math.PI / 180;
            points[i] = new((float)(cx + outer * Math.Cos(rad)), (float)(cy - outer * Math.Sin(rad)));
        }
        for (int i = 0; i < samples; i++)
        {
            double t = endDeg - (endDeg - startDeg) * i / (samples - 1);
            double rad = t * Math.PI / 180;
            points[samples + i] = new((float)(cx + inner * Math.Cos(rad)), (float)(cy - inner * Math.Sin(rad)));
        }
        return points;
    }

    private static void Fill(byte[] image, int width, int x0, int y0, int x1, int y1, byte b, byte g, byte r)
    {
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
            {
                int o = (y * width + x) * 3;
                image[o] = b;
                image[o + 1] = g;
                image[o + 2] = r;
            }
    }
}
