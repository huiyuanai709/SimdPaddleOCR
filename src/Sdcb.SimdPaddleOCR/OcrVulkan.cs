using System.Globalization;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Process-wide Vulkan device probe, memory budget, and allocation counters.
/// Set <see cref="DeviceSelector"/> before the first OCR session. An empty
/// selector picks the first discrete GPU, then any non-CPU device, then a
/// CPU/software device such as lavapipe.
/// </summary>
public static class OcrVulkan
{
    public const ulong FourGiB = 4UL << 30;

    /// <summary>
    /// Physical-device index (<c>0</c>, <c>1</c>, … in
    /// <c>vkEnumeratePhysicalDevices</c> order) or a case-insensitive name
    /// substring (<c>MX450</c>, <c>llvmpipe</c>). Null or blank keeps the
    /// discrete-first default. Also read from <c>SIMD_OCR_VK_DEVICE</c> when
    /// this property is empty.
    /// </summary>
    public static string? DeviceSelector { get; set; }

    /// <summary>Debug-level allocation trace. MiniOCR forwards this to its logger.</summary>
    public static Action<string>? OnDebug { get; set; }

    /// <summary>Raised when one image falls back to CPU after a device-memory failure.</summary>
    public static Action<string>? OnWarning { get; set; }

    /// <summary>Test hook. When set, every storage buffer is capped at this many bytes.</summary>
    public static ulong? BufferByteCapOverride { get; set; }

    private static long s_liveBytes;
    private static long s_allocs;
    private static long s_frees;
    private static VulkanDeviceInfo[] s_devices = [];

    public static long LiveDeviceBytes => Interlocked.Read(ref s_liveBytes);
    public static long AllocationCount => Interlocked.Read(ref s_allocs);
    public static long FreeCount => Interlocked.Read(ref s_frees);
    public static IReadOnlyList<VulkanDeviceInfo> Devices => s_devices;

    public readonly record struct VulkanSelector(bool IsDefault, int Index, string? Name);

    public readonly record struct VulkanDeviceInfo(
        int Index,
        string Name,
        string Kind,
        ulong DeviceLocalBytes,
        ulong BufferByteCap,
        bool Selected);

    public static VulkanSelector ParseSelector(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new VulkanSelector(true, -1, null);
        string trimmed = text.Trim();
        if (int.TryParse(trimmed, NumberStyles.None, CultureInfo.InvariantCulture, out int index)
            && index >= 0
            && index.ToString(CultureInfo.InvariantCulture) == trimmed)
            return new VulkanSelector(false, index, null);
        return new VulkanSelector(false, -1, trimmed);
    }

    /// <summary>
    /// One storage buffer may use at most a quarter of device-local memory,
    /// clamped to 64–256 MB below 4 GB and 64–512 MB at 4 GB and above.
    /// </summary>
    public static ulong BufferByteCap(ulong deviceLocalBytes)
    {
        if (BufferByteCapOverride is ulong over && over > 0)
            return over;
        if (deviceLocalBytes == 0)
            return 256UL << 20;
        ulong quarter = deviceLocalBytes / 4;
        ulong hi = deviceLocalBytes < FourGiB ? 256UL << 20 : 512UL << 20;
        if (quarter < 64UL << 20)
            return 64UL << 20;
        if (quarter > hi)
            return hi;
        return quarter;
    }

    /// <summary>Cards under 4 GB of device-local memory run one engine.</summary>
    public static int RecommendedEngineCount(ulong deviceLocalBytes, int requested)
    {
        if (requested < 1)
            requested = 1;
        if (deviceLocalBytes > 0 && deviceLocalBytes < FourGiB)
            return 1;
        return requested;
    }

    /// <summary>Creates the shared device if a Vulkan loader and GPU exist.</summary>
    public static VulkanDeviceInfo? TryProbe()
    {
#if NETSTANDARD2_0
        return null;
#else
        Backends.Vulkan.VkDevice? dev = Backends.Vulkan.GpuBackend.TryGetDevice();
        return dev?.Info;
#endif
    }

    internal static string? EffectiveSelector =>
        string.IsNullOrWhiteSpace(DeviceSelector)
            ? Environment.GetEnvironmentVariable("SIMD_OCR_VK_DEVICE")
            : DeviceSelector;

    internal static void SetDevices(VulkanDeviceInfo[] devices) => s_devices = devices;

    internal static void NoteAlloc(ulong bytes)
    {
        long live = Interlocked.Add(ref s_liveBytes, (long)bytes);
        long allocs = Interlocked.Increment(ref s_allocs);
        Debug($"vulkan alloc bytes={bytes} liveBytes={live} allocs={allocs} frees={Interlocked.Read(ref s_frees)}");
    }

    internal static void NoteFree(ulong bytes)
    {
        long live = Interlocked.Add(ref s_liveBytes, -(long)bytes);
        long frees = Interlocked.Increment(ref s_frees);
        Debug($"vulkan free bytes={bytes} liveBytes={live} allocs={Interlocked.Read(ref s_allocs)} frees={frees}");
    }

    internal static void Debug(string message)
    {
        Action<string>? sink = OnDebug;
        if (sink is null)
            return;
        try { sink(message); }
        catch { /* diagnostics must not fail OCR */ }
    }

    internal static void Warn(string message)
    {
        Action<string>? sink = OnWarning;
        if (sink is null)
        {
            Console.Error.WriteLine(message);
            return;
        }
        try { sink(message); }
        catch { Console.Error.WriteLine(message); }
    }
}
