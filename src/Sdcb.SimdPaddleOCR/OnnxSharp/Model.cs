using System.Collections.Concurrent;
using System.Security.Cryptography;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>A parsed ONNX model (Sdcb.SimdOnnx). The serialized payload is not retained.</summary>
public sealed class Model : IDisposable
{
    private static readonly ConcurrentDictionary<string, Lazy<OnnxModel>> s_shared = new(StringComparer.Ordinal);

    private Model(OnnxModel onnx) => Onnx = onnx;

    internal OnnxModel Onnx { get; }

    public static Model Load(ReadOnlyMemory<byte> source) => Load(source.Span);

    public static Model Load(byte[] source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        return Load(source.AsSpan());
    }

    /// <summary>Loads an ONNX model from the current stream position. The caller owns the stream.</summary>
    public static Model Load(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        using MemoryStream buffer = new();
        source.CopyTo(buffer);
        return Load(buffer.GetBuffer().AsSpan(0, checked((int)buffer.Length)));
    }

    /// <summary>Loads an ONNX model from the current stream position on a worker thread.</summary>
    public static Task<Model> LoadAsync(Stream source, CancellationToken cancellationToken = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() => Load(source), cancellationToken);
    }

    public static Model LoadOnnx(Stream source) => Load(source);
    public static Model LoadOnnx(ReadOnlyMemory<byte> source) => Load(source);
    public static Model LoadOnnx(byte[] source) => Load(source);
    public static Model LoadOnnx(ReadOnlySpan<byte> source) => Load(source);

    /// <summary>
    /// Identical bytes share one <see cref="OnnxModel"/>. GPU const stores are keyed by that
    /// instance, so several engines loaded from the same package upload weights once.
    /// </summary>
    private static Model Load(ReadOnlySpan<byte> source)
    {
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        byte[] owned = source.ToArray();
        string key;
        using (SHA256 sha = SHA256.Create())
            key = Convert.ToBase64String(sha.ComputeHash(owned));
        Lazy<OnnxModel> created = new(() => Parse(owned), LazyThreadSafetyMode.ExecutionAndPublication);
        Lazy<OnnxModel> shared = s_shared.GetOrAdd(key, created);
        return new Model(shared.Value);
    }

    private static OnnxModel Parse(byte[] source)
    {
        byte[]? widened = Float16Onnx.Widen(source);
        return OnnxModel.Load(widened ?? (ReadOnlySpan<byte>)source);
    }

    /// <summary>Weights are managed; kept for API compatibility.</summary>
    public void Dispose() { }
}
