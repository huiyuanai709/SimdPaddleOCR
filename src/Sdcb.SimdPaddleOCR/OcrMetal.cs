namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Process-wide Metal device probe. Metal exists only on macOS; other
/// platforms get null and should stay on CPU.
/// </summary>
public static class OcrMetal
{
    public readonly record struct MetalDeviceInfo(
        string Name,
        ulong RecommendedMaxWorkingSetBytes,
        ulong BufferByteCap);

    private static int s_sessionFallbacks;

    /// <summary>Sessions that failed to build on Metal and ran on CPU instead.</summary>
    public static int SessionFallbackCount => s_sessionFallbacks;

    /// <summary>Creates the shared device when the process is on macOS and Metal is present.</summary>
    public static MetalDeviceInfo? TryProbe()
    {
#if NETSTANDARD2_0
        return null;
#else
        if (!Sdcb.SimdOnnx.Backends.Metal.MetalBackend.TryGetDefault(out Sdcb.SimdOnnx.Backends.Metal.MetalBackend? backend))
            return null;
        ulong workingSet = backend.Dev.RecommendedMaxWorkingSetSize;
        return new MetalDeviceInfo(backend.DeviceName, workingSet, OcrVulkan.BufferByteCap(workingSet));
#endif
    }

    internal static void NoteSessionFallback(Exception ex)
    {
        Interlocked.Increment(ref s_sessionFallbacks);
        string message = $"[metal] session fallback: {ex.GetType().Name}: {ex.Message}";
        Console.Error.WriteLine(message);
        OcrVulkan.Warn(message);
    }
}
