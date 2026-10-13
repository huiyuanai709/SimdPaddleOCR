using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR.Models.SealV4Server;

public static class SealV4ServerModel
{
    public static IPaddleOcrModelProvider Detection { get; } = new ProviderImpl(
        "PP-OCRv4_server_seal_det",
        PaddleOcrModelKind.Detection,
        "det.onnx",
        "onnx");

    private sealed class ProviderImpl(string name, PaddleOcrModelKind kind, string resource, string format) : IPaddleOcrModelProvider
    {
        public string Name { get; } = name;
        public PaddleOcrModelKind Kind { get; } = kind;
        public string Format { get; } = format;
        public string? LanguageCode { get; } = "zh";
        public string? Version { get; } = "v4-server-seal-f16";
        public Stream OpenRead() => typeof(SealV4ServerModel).Assembly.GetManifestResourceStream("Sdcb.SimdPaddleOCR.Models.SealV4Server." + resource)
            ?? throw new InvalidOperationException("Embedded model resource was not found.");
        public Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(OpenRead());
        }
    }
}
