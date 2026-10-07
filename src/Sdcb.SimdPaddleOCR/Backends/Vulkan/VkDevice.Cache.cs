// Disk VkPipelineCache (invalidated by vendor/device/driver/pipelineCacheUUID)
// and a blocking fence wait. vkWaitForFences(UINT64_MAX) is the portable
// block, but some Windows drivers spin inside it; an exported Win32 event
// (or a Linux sync file) waits in the kernel instead.
using System.Runtime.InteropServices;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

internal unsafe sealed partial class VkDevice
{
    private IntPtr _pipelineCache;
    private string _cachePath = "";
    private uint _fenceExport;
    private delegate* unmanaged[Cdecl]<IntPtr, Vk.VkFenceGetWin32HandleInfoKHR*, IntPtr*, VkResult> _getFenceWin32;
    private delegate* unmanaged[Cdecl]<IntPtr, Vk.VkFenceGetFdInfoKHR*, int*, VkResult> _getFenceFd;
    public string FenceWaitMode { get; private set; } = "blocking";
    public uint DriverVersion;
    public uint DeviceId;
    private byte[] _cacheUuid = new byte[16];

    internal uint ProbeFenceExport(bool win32, bool syncFd)
    {
        uint want = win32 ? VkConst.ExternalFenceOpaqueWin32 : syncFd ? VkConst.ExternalFenceSyncFd : 0;
        if (want == 0) return 0;
        byte* name = stackalloc byte[]
        {
            (byte)'v', (byte)'k', (byte)'G', (byte)'e', (byte)'t', (byte)'P', (byte)'h', (byte)'y', (byte)'s',
            (byte)'i', (byte)'c', (byte)'a', (byte)'l', (byte)'D', (byte)'e', (byte)'v', (byte)'i', (byte)'c',
            (byte)'e', (byte)'E', (byte)'x', (byte)'t', (byte)'e', (byte)'r', (byte)'n', (byte)'a', (byte)'l',
            (byte)'F', (byte)'e', (byte)'n', (byte)'c', (byte)'e', (byte)'P', (byte)'r', (byte)'o', (byte)'p',
            (byte)'e', (byte)'r', (byte)'t', (byte)'i', (byte)'e', (byte)'s', 0,
        };
        IntPtr fn = Vk.vkGetInstanceProcAddr(Instance, name);
        if (fn == IntPtr.Zero) return 0;
        var query = (delegate* unmanaged[Cdecl]<IntPtr, Vk.VkPhysicalDeviceExternalFenceInfo*, Vk.VkExternalFenceProperties*, void>)fn;
        Vk.VkPhysicalDeviceExternalFenceInfo info = new()
        {
            SType = VkConst.StPhysicalDeviceExternalFenceInfo,
            HandleType = want,
        };
        Vk.VkExternalFenceProperties props = new() { SType = VkConst.StExternalFenceProperties };
        query(PhysDevice, &info, &props);
        if ((props.ExternalFenceFeatures & VkConst.ExternalFenceExportable) == 0)
            return 0;
        return want;
    }

    internal void EnableFenceExport(uint handleType)
    {
        _fenceExport = handleType;
        if (handleType == VkConst.ExternalFenceOpaqueWin32)
        {
            byte* name = stackalloc byte[]
            {
                (byte)'v', (byte)'k', (byte)'G', (byte)'e', (byte)'t', (byte)'F', (byte)'e', (byte)'n', (byte)'c',
                (byte)'e', (byte)'W', (byte)'i', (byte)'n', (byte)'3', (byte)'2', (byte)'H', (byte)'a', (byte)'n',
                (byte)'d', (byte)'l', (byte)'e', (byte)'K', (byte)'H', (byte)'R', 0,
            };
            IntPtr fn = Vk.vkGetDeviceProcAddr(Device, name);
            if (fn == IntPtr.Zero) { _fenceExport = 0; return; }
            _getFenceWin32 = (delegate* unmanaged[Cdecl]<IntPtr, Vk.VkFenceGetWin32HandleInfoKHR*, IntPtr*, VkResult>)fn;
            FenceWaitMode = "win32";
        }
        else if (handleType == VkConst.ExternalFenceSyncFd)
        {
            byte* name = stackalloc byte[]
            {
                (byte)'v', (byte)'k', (byte)'G', (byte)'e', (byte)'t', (byte)'F', (byte)'e', (byte)'n', (byte)'c',
                (byte)'e', (byte)'F', (byte)'d', (byte)'K', (byte)'H', (byte)'R', 0,
            };
            IntPtr fn = Vk.vkGetDeviceProcAddr(Device, name);
            if (fn == IntPtr.Zero) { _fenceExport = 0; return; }
            _getFenceFd = (delegate* unmanaged[Cdecl]<IntPtr, Vk.VkFenceGetFdInfoKHR*, int*, VkResult>)fn;
            FenceWaitMode = "syncfd";
        }
        OcrVulkan.NoteDevice(DeviceName, FenceWaitMode);
    }

    internal void OpenPipelineCache(uint vendorId, uint deviceId, uint driverVersion, ReadOnlySpan<byte> uuid)
    {
        DriverVersion = driverVersion;
        DeviceId = deviceId;
        _cacheUuid = uuid.ToArray();
        if (OcrVulkan.PipelineCacheDisabled)
        {
            CreateEmptyCache();
            return;
        }
        _cachePath = Path.Combine(
            OcrVulkan.ResolvePipelineCacheDirectory(),
            OcrVulkan.PipelineCacheFileName(vendorId, deviceId, driverVersion, uuid));
        byte[]? initial = null;
        try
        {
            if (File.Exists(_cachePath))
            {
                byte[] blob = File.ReadAllBytes(_cachePath);
                if (blob.Length is > 0 and <= 64 * 1024 * 1024)
                    initial = blob;
            }
        }
        catch (Exception ex)
        {
            OcrVulkan.Debug($"vulkan pipeline cache read failed: {ex.Message}");
        }
        if (initial is not null && TryCreateCache(initial))
        {
            OcrVulkan.NotePipelineCache(_cachePath, restored: true);
            return;
        }
        CreateEmptyCache();
        OcrVulkan.NotePipelineCache(_cachePath, restored: false);
    }

    private void CreateEmptyCache()
    {
        if (!TryCreateCache(null))
            _pipelineCache = IntPtr.Zero;
    }

    private bool TryCreateCache(byte[]? initial)
    {
        if (_pipelineCache != IntPtr.Zero)
        {
            Vk.vkDestroyPipelineCache(Device, _pipelineCache, null);
            _pipelineCache = IntPtr.Zero;
        }
        if (initial is { Length: > 0 })
        {
            fixed (byte* p = initial)
            {
                Vk.VkPipelineCacheCreateInfo ci = new()
                {
                    SType = VkConst.StPipelineCacheCreateInfo,
                    InitialDataSize = (nuint)initial.Length,
                    PInitialData = p,
                };
                if (Vk.vkCreatePipelineCache(Device, &ci, null, out IntPtr cache) != VkResult.Success)
                    return false;
                _pipelineCache = cache;
                return true;
            }
        }
        Vk.VkPipelineCacheCreateInfo empty = new() { SType = VkConst.StPipelineCacheCreateInfo };
        if (Vk.vkCreatePipelineCache(Device, &empty, null, out IntPtr created) != VkResult.Success)
            return false;
        _pipelineCache = created;
        return true;
    }

    internal IntPtr PipelineCache => _pipelineCache;

    internal void SavePipelineCache()
    {
        if (_pipelineCache == IntPtr.Zero || _cachePath.Length == 0 || OcrVulkan.PipelineCacheDisabled)
            return;
        lock (Sync)
        {
            nuint size = 0;
            if (Vk.vkGetPipelineCacheData(Device, _pipelineCache, &size, null) != VkResult.Success || size == 0)
                return;
            byte[] blob = new byte[(int)size];
            fixed (byte* p = blob)
            {
                if (Vk.vkGetPipelineCacheData(Device, _pipelineCache, &size, p) != VkResult.Success)
                    return;
            }
            if (size < (nuint)blob.Length)
                blob = blob.AsSpan(0, (int)size).ToArray();
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_cachePath)!);
                string tmp = _cachePath + ".tmp";
                File.WriteAllBytes(tmp, blob);
                File.Move(tmp, _cachePath, overwrite: true);
                OcrVulkan.Debug($"vulkan pipeline cache saved bytes={blob.Length} path={_cachePath}");
            }
            catch (Exception ex)
            {
                OcrVulkan.Debug($"vulkan pipeline cache save failed: {ex.Message}");
            }
        }
    }

    internal void DestroyPipelineCache()
    {
        if (_pipelineCache == IntPtr.Zero) return;
        SavePipelineCache();
        Vk.vkDestroyPipelineCache(Device, _pipelineCache, null);
        _pipelineCache = IntPtr.Zero;
    }

    private bool WaitExported(IntPtr fence)
    {
        if (_fenceExport == VkConst.ExternalFenceOpaqueWin32 && _getFenceWin32 != null)
        {
            Vk.VkFenceGetWin32HandleInfoKHR info = new()
            {
                SType = VkConst.StFenceGetWin32HandleInfoKHR,
                Fence = fence,
                HandleType = VkConst.ExternalFenceOpaqueWin32,
            };
            IntPtr handle;
            if (_getFenceWin32(Device, &info, &handle) != VkResult.Success || handle == IntPtr.Zero)
                return false;
            try
            {
                uint waited = WaitForSingleObject(handle, 0xFFFFFFFFu);
                return waited == 0;
            }
            finally
            {
                CloseHandle(handle);
            }
        }
        if (_fenceExport == VkConst.ExternalFenceSyncFd && _getFenceFd != null)
        {
            Vk.VkFenceGetFdInfoKHR info = new()
            {
                SType = VkConst.StFenceGetFdInfoKHR,
                Fence = fence,
                HandleType = VkConst.ExternalFenceSyncFd,
            };
            int fd;
            if (_getFenceFd(Device, &info, &fd) != VkResult.Success)
                return false;
            // Already signaled: the spec returns -1 and no fd to close.
            if (fd < 0) return true;
            try
            {
                PollFd pfd = new() { Fd = fd, Events = 1 };
                int rc = -1;
                for (int attempt = 0; attempt < 8 && rc < 0; attempt++)
                    rc = Poll(&pfd, 1, -1);
                return rc > 0;
            }
            finally
            {
                _ = CloseFd(fd);
            }
        }
        return false;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PollFd
    {
        public int Fd;
        public short Events;
        public short Revents;
    }

    private static uint WaitForSingleObject(IntPtr handle, uint milliseconds)
    {
        delegate* unmanaged[Stdcall]<IntPtr, uint, uint> fn = Win32Wait;
        return fn == null ? 0xFFFFFFFFu : fn(handle, milliseconds);
    }

    private static bool CloseHandle(IntPtr handle)
    {
        delegate* unmanaged[Stdcall]<IntPtr, int> fn = Win32Close;
        return fn != null && fn(handle) != 0;
    }

    private static int Poll(PollFd* fds, nuint nfds, int timeout)
    {
        delegate* unmanaged[Cdecl]<PollFd*, nuint, int, int> fn = LibcPoll;
        return fn == null ? -1 : fn(fds, nfds, timeout);
    }

    private static int CloseFd(int fd)
    {
        delegate* unmanaged[Cdecl]<int, int> fn = LibcClose;
        return fn == null ? -1 : fn(fd);
    }

    private static delegate* unmanaged[Stdcall]<IntPtr, uint, uint> Win32Wait
    {
        get
        {
            if (s_win32Wait != null || !OperatingSystem.IsWindows()) return s_win32Wait;
            if (NativeLibrary.TryLoad("kernel32.dll", out IntPtr lib)
                && NativeLibrary.TryGetExport(lib, "WaitForSingleObject", out IntPtr wait)
                && NativeLibrary.TryGetExport(lib, "CloseHandle", out IntPtr close))
            {
                s_win32Close = (delegate* unmanaged[Stdcall]<IntPtr, int>)close;
                s_win32Wait = (delegate* unmanaged[Stdcall]<IntPtr, uint, uint>)wait;
            }
            return s_win32Wait;
        }
    }

    private static delegate* unmanaged[Stdcall]<IntPtr, int> Win32Close => s_win32Close;

    private static delegate* unmanaged[Cdecl]<PollFd*, nuint, int, int> LibcPoll
    {
        get
        {
            if (s_libcPoll != null || OperatingSystem.IsWindows()) return s_libcPoll;
            if (NativeLibrary.TryLoad("libc.so.6", out IntPtr lib) || NativeLibrary.TryLoad("libc", out lib))
            {
                if (NativeLibrary.TryGetExport(lib, "poll", out IntPtr poll))
                    s_libcPoll = (delegate* unmanaged[Cdecl]<PollFd*, nuint, int, int>)poll;
                if (NativeLibrary.TryGetExport(lib, "close", out IntPtr close))
                    s_libcClose = (delegate* unmanaged[Cdecl]<int, int>)close;
            }
            return s_libcPoll;
        }
    }

    private static delegate* unmanaged[Cdecl]<int, int> LibcClose => s_libcClose;

    private static delegate* unmanaged[Stdcall]<IntPtr, uint, uint> s_win32Wait;
    private static delegate* unmanaged[Stdcall]<IntPtr, int> s_win32Close;
    private static delegate* unmanaged[Cdecl]<PollFd*, nuint, int, int> s_libcPoll;
    private static delegate* unmanaged[Cdecl]<int, int> s_libcClose;
}
