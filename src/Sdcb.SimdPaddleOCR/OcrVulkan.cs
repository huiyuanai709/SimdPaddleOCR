using System.Diagnostics;
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

    /// <summary>
    /// Directory for the on-disk Vulkan pipeline cache. Empty uses
    /// <c>%LOCALAPPDATA%/Sdcb.SimdPaddleOCR/vulkan-pipeline-cache</c>
    /// (or the platform local-app-data folder). Set
    /// <c>SIMD_OCR_VK_PIPELINE_CACHE</c> to <c>0</c> to disable, or to a
    /// directory to override this property.
    /// </summary>
    public static string? PipelineCacheDirectory { get; set; }

    public static bool PipelineCacheDisabled
    {
        get
        {
            string? env = Environment.GetEnvironmentVariable("SIMD_OCR_VK_PIPELINE_CACHE");
            return env is "0" or "off" or "false" or "no";
        }
    }

    public static string ResolvePipelineCacheDirectory()
    {
        string? env = Environment.GetEnvironmentVariable("SIMD_OCR_VK_PIPELINE_CACHE");
        if (!string.IsNullOrWhiteSpace(env)
            && !PipelineCacheDisabled
            && env is not "1" and not "on" and not "true")
            return env;
        if (!string.IsNullOrWhiteSpace(PipelineCacheDirectory))
            return PipelineCacheDirectory!;
        string root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        return Path.Combine(root, "Sdcb.SimdPaddleOCR", "vulkan-pipeline-cache");
    }

    /// <summary>File name keyed by vendor, device, driver version, and pipelineCacheUUID.</summary>
    public static string PipelineCacheFileName(uint vendorId, uint deviceId, uint driverVersion, ReadOnlySpan<byte> uuid)
    {
        char[] hex = new char[uuid.Length * 2];
        for (int i = 0; i < uuid.Length; i++)
        {
            byte b = uuid[i];
            hex[i * 2] = "0123456789abcdef"[b >> 4];
            hex[i * 2 + 1] = "0123456789abcdef"[b & 0xF];
        }
        return $"v{vendorId:x8}-d{deviceId:x8}-drv{driverVersion:x8}-{new string(hex)}.bin";
    }

    private static long s_submitTicks, s_waitTicks, s_preTicks, s_postTicks, s_initTicks;
    private static long s_gpuRuns, s_fallbackRuns;
    private static string s_deviceName = "";
    private static string s_fenceWait = "none";
    private static string s_cachePath = "";
    private static int s_cacheRestored;

    public static string DeviceName => s_deviceName;
    public static string FenceWait => s_fenceWait;
    public static string PipelineCachePath => s_cachePath;
    public static bool PipelineCacheRestored => s_cacheRestored != 0;
    public static long GpuRuns => Interlocked.Read(ref s_gpuRuns);
    public static long CpuFallbackRuns => Interlocked.Read(ref s_fallbackRuns);
    public static double InitMs => TicksToMs(Interlocked.Read(ref s_initTicks));

    public readonly record struct GpuTimingSnapshot(
        double GpuSubmitMs,
        double GpuWaitMs,
        double CpuPreMs,
        double CpuPostMs,
        long GpuRuns,
        long CpuFallbackRuns,
        double InitMs,
        string DeviceName,
        string FenceWait);

    public static GpuTimingSnapshot ReadTimings() => new(
        TicksToMs(Interlocked.Read(ref s_submitTicks)),
        TicksToMs(Interlocked.Read(ref s_waitTicks)),
        TicksToMs(Interlocked.Read(ref s_preTicks)),
        TicksToMs(Interlocked.Read(ref s_postTicks)),
        Interlocked.Read(ref s_gpuRuns),
        Interlocked.Read(ref s_fallbackRuns),
        InitMs,
        s_deviceName,
        s_fenceWait);

    /// <summary>
    /// Tracks one OCR call, including work that hops to the thread pool.
    /// A page that used the GPU and also fell back counts as a fallback.
    /// </summary>
    public sealed class GpuCallScope : IDisposable
    {
        internal sealed class Box
        {
            public int Gpu, Fallback;
            public long Submit, Wait, Pre, Post;
        }

        private static readonly AsyncLocal<Box?> s_current = new();
        private readonly Box _box;
        private readonly Box? _previous;
        private bool _disposed;

        private GpuCallScope(Box box, Box? previous)
        {
            _box = box;
            _previous = previous;
        }

        public static GpuCallScope Begin()
        {
            Box box = new();
            Box? previous = s_current.Value;
            s_current.Value = box;
            return new GpuCallScope(box, previous);
        }

        internal static Box? Current => s_current.Value;
        public bool UsedGpu => Volatile.Read(ref _box.Gpu) != 0;
        public bool UsedFallback => Volatile.Read(ref _box.Fallback) != 0;
        public long SubmitTicks => Interlocked.Read(ref _box.Submit);
        public long WaitTicks => Interlocked.Read(ref _box.Wait);
        public long PreTicks => Interlocked.Read(ref _box.Pre);
        public long PostTicks => Interlocked.Read(ref _box.Post);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            s_current.Value = _previous;
        }
    }

    internal static void AddGpuRun()
    {
        Interlocked.Increment(ref s_gpuRuns);
        GpuCallScope.Box? box = GpuCallScope.Current;
        if (box is not null) Interlocked.Increment(ref box.Gpu);
    }

    internal static void AddFallback()
    {
        Interlocked.Increment(ref s_fallbackRuns);
        GpuCallScope.Box? box = GpuCallScope.Current;
        if (box is not null) Interlocked.Increment(ref box.Fallback);
    }

    internal static void AddSubmitTicks(long ticks) => AddPair(ref s_submitTicks, ticks, static (b, t) => Interlocked.Add(ref b.Submit, t));
    internal static void AddWaitTicks(long ticks) => AddPair(ref s_waitTicks, ticks, static (b, t) => Interlocked.Add(ref b.Wait, t));
    internal static void AddCpuTicks(long pre, long post)
    {
        if (pre != 0) AddPair(ref s_preTicks, pre, static (b, t) => Interlocked.Add(ref b.Pre, t));
        if (post != 0) AddPair(ref s_postTicks, post, static (b, t) => Interlocked.Add(ref b.Post, t));
    }

    internal static void AddInitTicks(long ticks)
    {
        if (ticks > 0) Interlocked.Add(ref s_initTicks, ticks);
    }

    internal static void NoteDevice(string name, string fenceWait)
    {
        if (!string.IsNullOrEmpty(name)) s_deviceName = name;
        if (!string.IsNullOrEmpty(fenceWait)) s_fenceWait = fenceWait;
    }

    internal static void NotePipelineCache(string path, bool restored)
    {
        s_cachePath = path;
        if (restored) s_cacheRestored = 1;
    }

    private static void AddPair(ref long global, long ticks, Action<GpuCallScope.Box, long> intoCall)
    {
        if (ticks == 0) return;
        Interlocked.Add(ref global, ticks);
        GpuCallScope.Box? box = GpuCallScope.Current;
        if (box is not null) intoCall(box, ticks);
    }

    private static double TicksToMs(long ticks) =>
        ticks <= 0 || Stopwatch.Frequency <= 0 ? 0 : ticks * 1000.0 / Stopwatch.Frequency;
}
