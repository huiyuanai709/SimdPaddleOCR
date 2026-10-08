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
    internal const int PartCtrElem = 1 << 18;
    internal const int PartBytes = PartCtrElem * 4 + 64 * 1024;
}
