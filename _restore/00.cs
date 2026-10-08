using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// Shared (per device × compiled model) half of the GPU graph backend:
/// pipelines, fp16 weight uploads and the graph compiler that turns a
/// <see cref="CompiledModel"/> node list into a <see cref="GpuSchedule"/> —
/// one dispatch per (fused) node, all activations in a single fp16 NHWC arena.
/// A schedule is pure managed metadata (pipeline, arena-relative binding
/// offsets, push constants, grid); it owns no Vulkan object, so there is
/// nothing per-shape to evict, invalidate or leak. Sessions
/// (<see cref="GpuDetGraph"/>) own the arena/IO buffers and re-record their
/// command buffer each run with push descriptors, binding whatever arena
/// handle they currently hold.
/// </summary>
internal sealed class GpuGraphModel
{
    // Binding sentinels: schedules reference session-owned buffers by role;
    // the session substitutes its live handle when recording.
    internal static readonly VkBuffer RoleArena = new(), RoleIn = new(),
        RoleOut = new(), RolePart = new();

    // Keyed by model content, not object identity: each engine parses its own
    // Model, and the weight buffers must be uploaded once per ONNX file.
    private static readonly Dictionary<(ulong Key, VkDevice Device), GpuGraphModel> s_models = new();
    private int _refs;

    /// <summary>Ref-counted shared weights and schedules for one ONNX graph
    /// on one device. Released by <see cref="Release"/> when the last session
    /// goes away.</summary>
    internal static GpuGraphModel Acquire(VkDevice dev, CompiledModel compiled)
    {
        ulong key = compiled.Model.ContentKey;
        lock (s_models)
        {
            var id = (key, dev);
            if (!s_models.TryGetValue(id, out GpuGraphModel? m) || m._refs == 0)
            {
                m = new GpuGraphModel(dev, compiled);
                s_models[id] = m;
                OcrVulkan.Debug($"vulkan graph new key={key:x16}");
            }
            else
            {
                if (m._compiled.Disposed)
                {
                    lock (m)
                        m._compiled = compiled;
                }
                OcrVulkan.Debug($"vulkan graph reuse key={key:x16} refs={m._refs}");
            }
            m._refs++;
            return m;
        }
    }

    /// <summary>
    /// A session whose compiled model is still alive replaces a shared graph
    /// that was built from an engine already disposed.
    /// </summary>
    internal void Use(CompiledModel compiled)
    {
        if (!compiled.Disposed && _compiled.Disposed)
        {
            lock (this)
            {
                if (_compiled.Disposed)
                    _compiled = compiled;
            }
        }
    }

    internal void Release()
    {
        lock (s_models)
        {
            if (--_refs > 0) return;
            s_models.Remove((_key, _dev));
            lock (this)
            {
                foreach (VkBuffer b in _allBufs) b.Free();
                _allBufs.Clear();
                _schedules.Clear();
            }
        }
    }

    private readonly VkDevice _dev;
    private readonly ulong _key;
    private readonly Model _model;
    private CompiledModel _compiled;
    internal VkDevice Device => _dev;
    internal VkPipeline OutPipe => _pOut;

    private readonly VkPipeline _pSoftmax;
    private bool _sg32; // device cannot run sg16 coopmat pipes (NVIDIA/AMD)
    private readonly VkPipeline _pConv1x1, _pConv1x1N64, _pConv1x1N32, _pDot,
        _pDw, _pDw4, _pDw4T, _pDense, _pConvT, _pConvT4, _pElem, _pElem4,
        _pReduce, _pReduce4, _pReduce4b, _pPool, _pPool4,
        _pResize, _pResize4, _pResize4Add, _pConcat, _pConcat4, _pNchw, _pOut, _pIm2col,
        _pAddPs, _pSeF, _pConvD, _pConvDF32, _pCatRes,
        _pAvg4, _pAffine, _pLn, _pAttn, _pDotSk;
    // sg32 only (128x128 / 128x64 / 128x32 tiles): implicit-GEMM kxk conv,
    // SE-prescaled pointwise conv
    private readonly VkPipeline? _pConvK, _pConvKN64, _pConvKN32, _pConvPs, _pConvPsN64, _pConvPsN32;
    // sg32 lite family only: direct-load plain GEMM (M >= 16, K % 16 == 0)
    private readonly VkPipeline? _pCmD, _pCmDN64, _pCmDN32, _pDw4A;
    private readonly bool _lite;
    // No coopmat of the selected cm set's shape (VkDevice.CoopGemm):
    // subgroup-free gemm_nc instead of every cm pipe
    // (Intel's compiler aborts the process on those shaders). Auto sessions
    // reach this only on wave64-minimum devices — GpuBackend.UsesGpu.
    private readonly bool _nocm;
    // gemm_nc built with the unrolled no-act / relu / hardswish, single-output tail
    private readonly VkPipeline? _pGemmNcS;
    // no-coopmat tier on a device that cannot pin a narrow subgroup (Adreno
