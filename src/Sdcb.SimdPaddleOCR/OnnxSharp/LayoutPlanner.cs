using System.Buffers.Binary;
using System.Numerics;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Graph-level activation layout pass. Convolutional segments whose operators
/// all have channels-last implementations are executed in NHWC so the wide
/// pointwise/dense/depthwise kernels can stream contiguous channel vectors
/// without per-operator NCHW/NHWC transposes. Explicit
/// <see cref="OperatorId.LayoutConvert"/> nodes are inserted at segment
/// boundaries; every tensor keeps its logical NCHW shape and carries the
/// <see cref="Model.TensorNhwc"/> flag when its storage is channels-last.
/// </summary>
internal static class LayoutPlanner
{
    public const ushort ToNhwc = 1, ToNchw = 2;

    /// <summary>
    /// NHWC kernels exist for AVX2+FMA (net10) and for Vector&lt;float&gt;
    /// (netstandard2.0); other ISAs keep the NCHW graph unchanged.
    /// </summary>
    public static bool IsEnabled { get; } = ComputeEnabled();

    private static bool ComputeEnabled()
    {
        if (Environment.GetEnvironmentVariable("PPOCR_NHWC") == "0") return false;
#if NETSTANDARD2_0
        return Vector.IsHardwareAccelerated;
#else
        return Avx2.IsSupported && Fma.IsSupported;
#endif
    }

    public static void Apply(ref TensorRecord[] tensors, ref NodeRecord[] nodes,
        ref byte[][] tensorData, ref byte[][] nodeParameters, uint[] graphInputs, uint[] graphOutputs)
    {
        if (!IsEnabled) return;
        Planner planner = new(tensors, nodes, tensorData, nodeParameters, graphInputs, graphOutputs);
        planner.Run();
        tensors = planner.Tensors;
        nodes = planner.Nodes;
        tensorData = planner.TensorData;
        nodeParameters = planner.NodeParameters;
        if (Environment.GetEnvironmentVariable("PPOCR_DUMP_GRAPH") is not null)
        {
            TensorRecord[] t = tensors;
            Console.Error.WriteLine($"graph {nodes.Length} nodes, {t.Length} tensors");
            for (int i = 0; i < nodes.Length; i++)
            {
                NodeRecord n = nodes[i];
                string Describe(uint ti) => ti == uint.MaxValue ? "-" :
                    $"t{ti}[{string.Join(",", t[ti].Dimensions.Take((int)t[ti].Rank))}]{((t[ti].Flags & Model.TensorNhwc) != 0 ? "@nhwc" : "")}{((t[ti].Flags & Model.TensorConstant) != 0 ? "@const" : "")}";
                Console.Error.WriteLine($"  {i,4} {n.Operator,-18} {string.Join(" ", n.Inputs.Select(Describe))} -> {string.Join(" ", n.Outputs.Select(Describe))}");
            }
        }
    }

    public static int Direction(ReadOnlySpan<byte> parameters)
        => parameters.Length >= 4 ? BinaryPrimitives.ReadUInt16LittleEndian(parameters.Slice(2)) : 0;

    private sealed class Planner
    {
        private readonly List<TensorRecord> _tensors;
        private readonly List<byte[]> _tensorData;
        private readonly List<byte[]> _parameters;
        private readonly NodeRecord[] _source;
        private readonly List<NodeRecord> _nodes = [];
        private readonly HashSet<uint> _outputs;
        private readonly Dictionary<uint, uint> _toNhwc = [], _toNchw = [];
        private readonly HashSet<uint> _nhwc = [];

        public Planner(TensorRecord[] tensors, NodeRecord[] nodes, byte[][] tensorData,
            byte[][] nodeParameters, uint[] graphInputs, uint[] graphOutputs)
        {
            _tensors = [.. tensors];
            _tensorData = [.. tensorData];
            _parameters = [.. nodeParameters];
            _source = nodes;
            _outputs = [.. graphOutputs];
            _ = graphInputs;
        }

        public TensorRecord[] Tensors => [.. _tensors];
        public NodeRecord[] Nodes => [.. _nodes];
        public byte[][] TensorData => [.. _tensorData];
        public byte[][] NodeParameters => [.. _parameters];

        public void Run()
        {
            foreach (NodeRecord node in _source)
            {
                bool nhwcMode = NhwcCapable(node) && WantsNhwc(node);
                if (!nhwcMode)
                {
                    // Consumers without a channels-last implementation read
                    // the NCHW twin of any NHWC input.
                    _nodes.Add(RewriteInputs(node, wantNhwc: false));
                    continue;
                }
                NodeRecord rewritten = RewriteInputs(node, wantNhwc: true);
                uint output = rewritten.Outputs[0];
                if (IsNeutral(output))
                {
                    _nodes.Add(rewritten);
                    continue;
                }
                if (_outputs.Contains(output))
                {
                    // Graph outputs must stay NCHW: produce into a twin and
                    // convert into the real output tensor.
                    uint twin = AddTensor(output, nhwc: true);
                    _nhwc.Add(twin);
                    _nodes.Add(rewritten with { Outputs = [twin] });
                    AddConvert(twin, output, ToNchw);
                    _toNchw[twin] = output;
                    continue;
                }
                _nhwc.Add(output);
                int index = checked((int)output);
                _tensors[index] = _tensors[index] with { Flags = _tensors[index].Flags | Model.TensorNhwc };
                _nodes.Add(rewritten);
            }
        }

        private NodeRecord RewriteInputs(NodeRecord node, bool wantNhwc)
        {
            uint[]? inputs = null;
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                uint t = node.Inputs[i];
                if (t == uint.MaxValue || IsConstant(t) || Rank(t) != 4 || IsNeutral(t)) continue;
                bool isNhwc = _nhwc.Contains(t);
                if (isNhwc == wantNhwc) continue;
                inputs ??= [.. node.Inputs];
                inputs[i] = wantNhwc ? Convert(t, ToNhwc) : Convert(t, ToNchw);
            }
            return inputs is null ? node : node with { Inputs = inputs };
        }

        private uint Convert(uint source, ushort direction)
        {
            Dictionary<uint, uint> cache = direction == ToNhwc ? _toNhwc : _toNchw;
            if (cache.TryGetValue(source, out uint existing)) return existing;
            uint twin = AddTensor(source, nhwc: direction == ToNhwc);
            AddConvert(source, twin, direction);
            cache[source] = twin;
            if (direction == ToNhwc) _nhwc.Add(twin);
            return twin;
        }

        private uint AddTensor(uint like, bool nhwc)
        {
            TensorRecord template = _tensors[checked((int)like)];
            uint index = checked((uint)_tensors.Count);
            _tensors.Add(new TensorRecord(template.DType, template.Rank, [.. template.Dimensions],
                nhwc ? Model.TensorNhwc : 0u));
            _tensorData.Add([]);
            return index;
        }

        private void AddConvert(uint source, uint destination, ushort direction)
        {
            byte[] parameters = new byte[8];
            BinaryPrimitives.WriteUInt16LittleEndian(parameters.AsSpan(0, 2), 1);
            BinaryPrimitives.WriteUInt16LittleEndian(parameters.AsSpan(2, 2), direction);
            uint parameterIndex = checked((uint)_parameters.Count);
            _parameters.Add(parameters);
            _nodes.Add(new NodeRecord(OperatorId.LayoutConvert, [source], [destination], parameterIndex,
                direction == ToNhwc ? "layout.to_nhwc" : "layout.to_nchw", "LayoutConvert"));
        }

        // A capable node runs channels-last when one of its activation inputs
        // already is, or when it is a dense convolution (segment start).
        // Depthwise convolutions only continue segments: starting one in a
        // narrow (OC % 16 != 0) network would ping-pong layouts.
        private bool WantsNhwc(NodeRecord node)
        {
            bool anyNchwActivation = false;
            foreach (uint t in node.Inputs)
            {
                if (t == uint.MaxValue || IsConstant(t) || Rank(t) != 4 || IsNeutral(t)) continue;
                if (_nhwc.Contains(t)) return true;
                anyNchwActivation = true;
            }
            if (!anyNchwActivation) return false;
            if (node.Operator == OperatorId.ConvTranspose) return true;
            if (node.Operator != OperatorId.Conv) return false;
            ReadOnlySpan<byte> p = _parameters[checked((int)node.ParameterIndex)];
            return BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(4)) == 1;
        }

        private bool NhwcCapable(NodeRecord node)
        {
            if (node.Outputs.Length != 1 || node.Inputs.Length == 0) return false;
            ReadOnlySpan<byte> p = _parameters[checked((int)node.ParameterIndex)];
            switch (node.Operator)
            {
                case OperatorId.Conv:
                {
                    if (node.Inputs.Length < 2 || p.Length < 48 || !IsConstant(node.Inputs[1]) || Rank(node.Inputs[1]) != 4)
                        return false;
                    if (node.Inputs.Length > 2 && (!IsConstant(node.Inputs[2]) || Rank(node.Inputs[2]) != 1)) return false;
                    int[] w = Dims(node.Inputs[1]);
                    uint group = BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(4));
                    int dh = I32(p, 24), dw = I32(p, 28), kh = I32(p, 8), kw = I32(p, 12);
                    if (dh != 1 || dw != 1 || kh != w[2] || kw != w[3] || kh <= 0 || kw <= 0) return false;
                    if (I32(p, 16) <= 0 || I32(p, 20) <= 0) return false;
                    if (group == 1) return w[0] >= 16 && (w[0] & 15) == 0;
                    // Depthwise: one input channel per group, one output per group.
                    return w[1] == 1 && w[0] == group && (group & 7) == 0;
                }
                case OperatorId.ConvTranspose:
                {
                    if (node.Inputs.Length < 2 || p.Length < 48 || !IsConstant(node.Inputs[1]) || Rank(node.Inputs[1]) != 4)
                        return false;
                    if (node.Inputs.Length > 2) return false;
                    int[] w = Dims(node.Inputs[1]);
                    return BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(4)) == 1 &&
                        I32(p, 8) == 2 && I32(p, 12) == 2 && I32(p, 16) == 2 && I32(p, 20) == 2 &&
                        I32(p, 24) == 1 && I32(p, 28) == 1 && I32(p, 32) == 0 && I32(p, 36) == 0 &&
                        I32(p, 40) == 0 && I32(p, 44) == 0 && (w[1] == 1 || (w[1] & 15) == 0);
                }
                case OperatorId.Add or OperatorId.Sub or OperatorId.Mul or OperatorId.Div:
                    return node.Inputs.Length == 2 && BinaryCapable(node.Inputs[0], node.Inputs[1]);
                case OperatorId.Pow:
                    return node.Inputs.Length == 2 && IsScalar(node.Inputs[1]);
                case OperatorId.Relu or OperatorId.Sigmoid or OperatorId.Erf or OperatorId.Sqrt or OperatorId.HardSigmoid:
                    return true;
                case OperatorId.BatchNormalization:
                    return node.Inputs.Length == 5 && Rank(node.Inputs[0]) == 4;
                case OperatorId.ReduceMean:
                {
                    if (Rank(node.Inputs[0]) != 4 || p.Length < 48) return false;
                    int count = BinaryPrimitives.ReadUInt16LittleEndian(p.Slice(2));
                    bool keep = BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(4)) != 0;
                    if (!keep || count != 2) return false;
                    int a0 = I32(p, 12), a1 = I32(p, 16);
                    if (a0 < 0) a0 += 4;
                    if (a1 < 0) a1 += 4;
                    return (a0 == 2 && a1 == 3) || (a0 == 3 && a1 == 2);
                }
                case OperatorId.MaxPool or OperatorId.AveragePool:
                    return Rank(node.Inputs[0]) == 4 && p.Length >= 48 &&
                        BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(4)) == 2 &&
                        I32(p, 8) > 0 && I32(p, 12) > 0 && I32(p, 16) > 0 && I32(p, 20) > 0;
                case OperatorId.Resize:
                {
                    if (Rank(node.Inputs[0]) != 4 || p.Length < 20) return false;
                    float sn = F32(p, 4), sc = F32(p, 8), sh = F32(p, 12), sw = F32(p, 16);
                    return sn == 1f && sc == 1f && sh >= 1f && sw >= 1f && sh == MathF.Floor(sh) && sw == MathF.Floor(sw);
                }
                case OperatorId.Concat:
                {
                    if (p.Length < 8) return false;
                    int axis = I32(p, 4);
                    foreach (uint t in node.Inputs)
                        if (t == uint.MaxValue || Rank(t) != 4 || IsConstant(t)) return false;
                    if (axis < 0) axis += 4;
                    return axis == 1;
                }
                default:
                    return false;
            }
        }

        private bool BinaryCapable(uint a, uint b)
        {
            if (a == uint.MaxValue || b == uint.MaxValue) return false;
            if (IsScalar(a) || IsScalar(b)) return true;
            if (Rank(a) != 4 || Rank(b) != 4) return false;
            int[] da = Dims(a), db = Dims(b);
            if (IsChannelVector(da) && da[1] == db[1]) return true;
            if (IsChannelVector(db) && db[1] == da[1]) return true;
            for (int i = 0; i < 4; i++)
                if (da[i] != db[i]) return false;
            return true;
        }

        // Batch may be symbolic (-1) in exported value_info.
        private static bool IsChannelVector(int[] d) => d[0] != 0 && d[1] > 0 && d[2] == 1 && d[3] == 1;

        private bool IsScalar(uint t)
        {
            int[] d = Dims(t);
            if (d.Length == 0) return true;
            long count = 1;
            foreach (int v in d) { if (v < 0) return false; count *= v; }
            return count == 1;
        }

        /// <summary>NHWC and NCHW storage coincide when C == 1 or H == W == 1.</summary>
        private bool IsNeutral(uint t)
        {
            if (Rank(t) != 4) return false;
            int[] d = Dims(t);
            return d[1] == 1 || (d[2] == 1 && d[3] == 1);
        }

        private bool IsConstant(uint t) => (_tensors[checked((int)t)].Flags & Model.TensorConstant) != 0;
        private int Rank(uint t) => checked((int)_tensors[checked((int)t)].Rank);
        private int[] Dims(uint t)
        {
            TensorRecord r = _tensors[checked((int)t)];
            return r.Dimensions.Take(checked((int)r.Rank)).ToArray();
        }

        private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p.Slice(o));
        private static float F32(ReadOnlySpan<byte> p, int o) => BitConverterCompat.Int32BitsToSingle(I32(p, o));
    }
}
