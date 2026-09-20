using System.Text.Json.Nodes;
using LwPpocrCSharp;

namespace Sdcb.SimdPaddleOCR.Tests;

sealed class CEngine : IBenchEngine
{
    private readonly NativeOcr _ocr;

    public CEngine(string cAssetsDir, int workers, string modelType)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("--engine c requires Windows (lw_ppocr_c.dll)");
        if (modelType is not ("tiny" or "small" or "medium"))
            throw new ArgumentException("--engine c supports --model tiny|small|medium");

        CAssetSet assets = CAssets.Resolve(cAssetsDir, modelType);
        CAssets.CopyDll(assets.DllPath);
        Extra["cAssets"] = assets.Directory;
        Extra["cDll"] = assets.DllPath;
        Extra["cDet"] = assets.DetPath;
        Extra["cCls"] = assets.ClsPath;
        Extra["cRec"] = assets.RecPath;
        Extra["cDict"] = assets.DictPath;
        Extra["cSource"] = assets.Source;
        Extra["cRemoteDll"] = CAssets.BaseUrl + CAssets.RemoteDllName;

        _ocr = new NativeOcr(
            assets.DetPath,
            assets.ClsPath,
            assets.RecPath,
            assets.DictPath,
            useDirectionClassification: true,
            (uint)workers);
    }

    public string Name => "c";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=c";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        OcrResponse result = _ocr.RecognizeDecoded(new DecodedBgrImage
        {
            Pixels = bgr,
            Width = width,
            Height = height,
            Stride = stride,
        });
        return new BenchEngineOutput
        {
            Detected = result.detected_count,
            Texts = result.result.Select(x => x.text).ToArray(),
            Rotations = result.result.Select(x => x.rotation).ToArray(),
            Boxes = result.result.Select(x => BenchBoxes.Aabb(x.x1, x.y1, x.x2, x.y2, x.x3, x.y3, x.x4, x.y4))
                .ToArray(),
        };
    }

    public void Dispose() => _ocr.Dispose();
}
