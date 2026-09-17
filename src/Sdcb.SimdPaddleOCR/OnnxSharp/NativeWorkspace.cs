using System.Runtime.InteropServices;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Activation workspace in unmanaged memory. Managed <c>float[]</c> blocks
/// of 50–100 MB that a grow-only session drops on every larger reshape sit
/// in the LOH as uncollected garbage and the freed regions are decommitted
/// lazily, so the process Working Set only ever ratcheted up. Releasing the
/// old block here returns the pages to the OS immediately.
/// </summary>
internal sealed unsafe class NativeWorkspace : IDisposable
{
    // 64-byte alignment matches the planner's 16-float slot alignment.
    private const int Alignment = 64;
    // Slack past the last planned slot so a vector tail that peeks a few
    // floats beyond a tensor can never touch an unmapped page.
    internal const int GuardFloats = 64;

    private void* _raw;
    private float* _pointer;

    public NativeWorkspace(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        Length = length;
        nuint bytes = checked((nuint)(length + GuardFloats) * sizeof(float));
#if NET6_0_OR_GREATER
        _raw = NativeMemory.AlignedAlloc(bytes, Alignment);
        _pointer = (float*)_raw;
#else
        _raw = (void*)Marshal.AllocHGlobal(checked((IntPtr)((long)bytes + Alignment)));
        _pointer = (float*)(((nuint)_raw + Alignment - 1) & ~(nuint)(Alignment - 1));
#endif
    }

    ~NativeWorkspace() => Free();

    /// <summary>Usable length in floats (excludes the guard).</summary>
    public int Length { get; }

    public float* Pointer => _pointer;

    public void Dispose()
    {
        Free();
        GC.SuppressFinalize(this);
    }

    private void Free()
    {
        void* raw = _raw;
        if (raw is null) return;
        _raw = null;
        _pointer = null;
#if NET6_0_OR_GREATER
        NativeMemory.AlignedFree(raw);
#else
        Marshal.FreeHGlobal((IntPtr)raw);
#endif
    }
}
