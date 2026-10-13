using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR.Models.SealV4Mobile;

public static class SealV4MobileModel
{
    public static IPaddleOcrModelProvider Detection { get; } = new ProviderImpl(
        "PP-OCRv4_mobile_seal_det",
        PaddleOcrModelKind.Detection,
        "det.onnx",
        "onnx");

    private sealed class ProviderImpl(string name, PaddleOcrModelKind kind, string resource, string format) : IPaddleOcrModelProvider
    {
        public string Name { get; } = name;
        public PaddleOcrModelKind Kind { get; } = kind;
        public string Format { get; } = format;
        public string? LanguageCode { get; } = "zh";
        public string? Version { get; } = "v4-mobile-seal";
        public Stream OpenRead() => typeof(SealV4MobileModel).Assembly.GetManifestResourceStream("Sdcb.SimdPaddleOCR.Models.SealV4Mobile." + resource)
            ?? throw new InvalidOperationException("Embedded model resource was not found.");
        public Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(OpenRead());
        }
    }
}
