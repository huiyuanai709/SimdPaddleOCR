using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

// Channels-last execution of the operators LayoutPlanner marks NHWC. Tensor
// shapes stay logical NCHW; only the storage order differs.
public sealed partial class InferenceSession
{
    // [rows][k] x const [k][n] as a channels-last GEMM (rows = pixels): same
    // zero-initialised, k-ascending FMA order as MatMulRows4, but threaded.
    private bool TryMatMulNhwc(NodeRecord node, TensorValue a, TensorValue b, TensorValue o)
    {
        int[] ad = a.Shape, bd = b.Shape;
        if (!b.IsConstant || bd.Length != 2 || ad.Length < 2 || ad[^1] != bd[0]) return false;
        float[]? packed = _model.GetPackedWeights(node, Model.PackMatMulNhwc);
        if (packed is null) return false;
        int k = bd[0], n = bd[1];
        if (a.Length % k != 0) return false;
        int rows = a.Length / k;
        if (o.Length != rows * n) return false;
        Nhwc.Pointwise(a.Data, packed, [], o.Data, rows, k, n, [], NhwcActivation.None, 0f, 0f, _intraOpThreads);
        return true;
    }

    private bool TryExecuteLayerNorm(int index)
    {
        NodeRecord[] nodes = _model.Nodes;
        if (!_compiled.MatchLayerNorm(nodes, index)) return false;
        NodeRecord mean = nodes[index], addEps = nodes[index + 4], mul = nodes[index + 7], addBeta = nodes[index + 8];
        TensorValue x = _tensors[mean.Inputs[0]];
        TensorValue destination = _tensors[addBeta.Outputs[0]];
        TensorValue eps = _tensors[addEps.Inputs[0] == nodes[index + 3].Outputs[0] ? addEps.Inputs[1] : addEps.Inputs[0]];
        TensorValue gamma = _tensors[mul.Inputs[0] == nodes[index + 6].Outputs[0] ? mul.Inputs[1] : mul.Inputs[0]];
        TensorValue beta = _tensors[addBeta.Inputs[0] == mul.Outputs[0] ? addBeta.Inputs[1] : addBeta.Inputs[0]];
        int channels = x.Shape[^1];
        if (channels <= 0 || x.Length != destination.Length || gamma.Length != channels || beta.Length != channels ||
            destination.IsConstant || PartialOverlap(x, destination))
            return false;
        LayerNorm.Rows(x.Data, gamma.Data, beta.Data, eps.Data[0], x.Length / channels, channels, destination.Data);
        return true;
    }

    // Conv -> Add(channel bias) -> Div/Erf/Add/Mul/Mul (GELU): the convolution
    // writes GELU(bias + sum) straight into the final Mul's buffer.
    private bool TryExecuteConvBiasGelu(int index)
    {
        if (index + 6 >= _model.Nodes.Length || !_compiled.IsNhwcNode(index)) return false;
        NodeRecord conv = _model.Nodes[index], add = _model.Nodes[index + 1], last = _model.Nodes[index + 6];
        if (conv.Operator != OperatorId.Conv || add.Operator != OperatorId.Add || last.Operator != OperatorId.Mul ||
            conv.Inputs.Length != 2 || add.Inputs.Length != 2 ||
            (add.Inputs[0] != conv.Outputs[0] && add.Inputs[1] != conv.Outputs[0]))
            return false;
        uint biasIndex = add.Inputs[0] == conv.Outputs[0] ? add.Inputs[1] : add.Inputs[0];
        TensorValue bias = _tensors[biasIndex];
        TensorValue convOutput = _tensors[conv.Outputs[0]];
        TensorValue addOutput = _tensors[add.Outputs[0]];
        TensorValue destination = _tensors[last.Outputs[0]];
        if (!bias.IsConstant || destination.IsConstant || convOutput.Length == 0 ||
            convOutput.Length != destination.Length || destination.Shape.Length != 4 ||
            bias.Length != destination.Shape[1] || _tensors[conv.Inputs[0]].Overlaps(destination))
            return false;
        convOutput.ShareStorageWith(destination);
        addOutput.ShareStorageWith(destination);
        ExecuteConv(conv, index, destination, bias, activation: NhwcActivation.Gelu);
        return true;
    }

    private void LayoutConvert(TensorValue x, ReadOnlySpan<byte> p, TensorValue o)
    {
        int[] shape = x.Shape;
        if (shape.Length != 4 || x.Length != o.Length)
            throw new InvalidDataException("LayoutConvert requires matching rank-4 tensors.");
        int n = shape[0], c = shape[1], plane = checked(shape[2] * shape[3]);
        if (LayoutPlanner.Direction(p) == LayoutPlanner.ToNhwc)
            Nhwc.NchwToNhwc(x.Data, o.Data, n, c, plane, _intraOpThreads);
        else
            Nhwc.NhwcToNchw(x.Data, o.Data, n, c, plane, _intraOpThreads);
    }

    /// <summary>Runs layout-sensitive operators channels-last; returns false for layout-agnostic ones.</summary>
    private bool ExecuteNhwcNode(NodeRecord node, int nodeIndex, ReadOnlySpan<byte> p, TensorValue x, TensorValue o)
    {
        switch (node.Operator)
        {
            case OperatorId.Conv:
                ExecuteConv(node, nodeIndex, o, node.Inputs.Length > 2 ? _tensors[node.Inputs[2]] : null);
                return true;
            case OperatorId.ConvTranspose:
                ConvTransposeNhwc(node, x, o, node.Inputs.Length > 2 ? _tensors[node.Inputs[2]].Data : [], NhwcActivation.None);
                return true;
            case OperatorId.Add: BinaryNhwc<AddOp>(x, _tensors[node.Inputs[1]], o); return true;
            case OperatorId.Sub: BinaryNhwc<SubOp>(x, _tensors[node.Inputs[1]], o); return true;
            case OperatorId.Mul: BinaryNhwc<MulOp>(x, _tensors[node.Inputs[1]], o); return true;
            case OperatorId.Div: BinaryNhwc<DivOp>(x, _tensors[node.Inputs[1]], o); return true;
            case OperatorId.BatchNormalization:
            {
                int[] shape = x.Shape;
                int pixels = checked(shape[0] * shape[2] * shape[3]);
                Nhwc.BatchNorm(x.Data, o.Data, pixels, shape[1], _tensors[node.Inputs[1]].Data, _tensors[node.Inputs[2]].Data,
                    _tensors[node.Inputs[3]].Data, _tensors[node.Inputs[4]].Data, F32(p, 4));
                return true;
            }
            case OperatorId.ReduceMean:
            {
                int[] shape = x.Shape;
                if (o.Shape.Length != 4 || o.Shape[2] != 1 || o.Shape[3] != 1 || o.Shape[1] != shape[1])
                    throw new InvalidDataException("NHWC ReduceMean expects a spatial reduction with keepdims.");
                Nhwc.ReduceMeanSpatial(x.Data, o.Data, shape[0], shape[1], checked(shape[2] * shape[3]));
                return true;
            }
            case OperatorId.MaxPool or OperatorId.AveragePool:
            {
                int[] id = x.Shape, od = o.Shape;
                Nhwc.Pool(x.Data, o.Data, id[0], id[1], id[2], id[3], od[2], od[3],
                    I32(p, 8), I32(p, 12), I32(p, 16), I32(p, 20), I32(p, 24), I32(p, 28), node.Operator == OperatorId.MaxPool,
                    _intraOpThreads);
                return true;
            }
            case OperatorId.Resize:
            {
                int[] id = x.Shape, od = o.Shape;
                int factorH = (int)F32(p, 12), factorW = (int)F32(p, 16);
                if (od[0] != id[0] || od[1] != id[1] || od[2] != id[2] * factorH || od[3] != id[3] * factorW)
                    throw new InvalidDataException("NHWC Resize expects integer nearest upsampling.");
                Nhwc.ResizeNearest(x.Data, o.Data, id[0], id[1], id[2], id[3], factorH, factorW, _intraOpThreads);
                return true;
            }
            case OperatorId.Concat:
                ConcatNhwc(node, o);
                return true;
            default:
                return false;
        }
    }

    private void BinaryNhwc<TOp>(TensorValue a, TensorValue b, TensorValue o) where TOp : struct, IBinaryOp
    {
        if (b.Length == 1 && a.Length == o.Length)
        {
            SimdKernels.ElementwiseScalar<TOp>(a.Data, b.Data[0], o.Data);
            return;
        }
        if (a.Length == 1 && b.Length == o.Length)
        {
            // scalar ∘ tensor: fall back to the generic broadcast (rare).
            Binary<TOp>(a, b, o, _intraOpThreads);
            return;
        }
        if (a.Length == o.Length && b.Length == o.Length)
        {
            SimdKernels.ElementwiseParallel<TOp>(a.Data, b.Data, o.Data, _intraOpThreads);
            return;
        }
        int[] ad = a.Shape, bd = b.Shape, od = o.Shape;
        if (ad.Length == 4 && bd.Length == 4 && od.Length == 4)
        {
            if (IsChannelVector(bd, ad) && SameShape(ad, od))
            {
                Nhwc.BinaryChannel<TOp>(a.Data, b.Data, o.Data, ad[0], ad[1], checked(ad[2] * ad[3]),
                    channelIsLeft: false, channelPerBatch: bd[0] == ad[0] && ad[0] > 1);
                return;
            }
            if (IsChannelVector(ad, bd) && SameShape(bd, od))
            {
                Nhwc.BinaryChannel<TOp>(b.Data, a.Data, o.Data, bd[0], bd[1], checked(bd[2] * bd[3]),
                    channelIsLeft: true, channelPerBatch: ad[0] == bd[0] && bd[0] > 1);
                return;
            }
        }
        throw new InvalidDataException($"Unsupported NHWC broadcast: [{string.Join(",", ad)}] ∘ [{string.Join(",", bd)}] -> [{string.Join(",", od)}].");

        static bool IsChannelVector(int[] v, int[] full)
            => v[1] == full[1] && v[2] == 1 && v[3] == 1 && (v[0] == 1 || v[0] == full[0]);
        static bool SameShape(int[] a, int[] b) => a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    }

    private unsafe void ConcatNhwc(NodeRecord node, TensorValue o)
    {
        int[] od = o.Shape;
        int pixels = checked(od[0] * od[2] * od[3]), totalChannels = od[1];
        int channelOffset = 0;
        int workers = _intraOpThreads > 1 && o.Length >= 1 << 18 ? Math.Min(_intraOpThreads, Math.Max(1, pixels / 256)) : 1;
        fixed (float* outPtr = o.Data)
        {
            nint outA = (nint)outPtr;
            foreach (uint ti in node.Inputs)
            {
                TensorValue t = _tensors[ti];
                int[] td = t.Shape;
                if (td.Length != 4 || td[0] != od[0] || td[2] != od[2] || td[3] != od[3])
                    throw new InvalidDataException("NHWC Concat inputs must agree on batch and spatial size.");
                int channels = td[1], offset = channelOffset;
                fixed (float* srcPtr = t.Data)
                {
                    nint srcA = (nint)srcPtr;
                    void Worker(int worker)
                    {
                        int begin = (int)((long)pixels * worker / workers), end = (int)((long)pixels * (worker + 1) / workers);
                        float* src = (float*)srcA + (long)begin * channels;
                        float* dst = (float*)outA + (long)begin * totalChannels + offset;
                        long bytes = (long)channels * sizeof(float);
                        for (int pixel = begin; pixel < end; pixel++, src += channels, dst += totalChannels)
                            Buffer.MemoryCopy(src, dst, bytes, bytes);
                    }
                    if (workers > 1) Parallel.For(0, workers, Worker);
                    else Worker(0);
                }
                channelOffset += channels;
            }
        }
        if (channelOffset != totalChannels)
            throw new InvalidDataException("NHWC Concat channel count mismatch.");
    }

    private void ConvNhwc(NodeRecord conv, TensorValue x, TensorValue o, TensorValue? bias, ReadOnlySpan<byte> p,
        ReadOnlySpan<float> residual, NhwcActivation activation, float alpha, float beta)
    {
        int[] id = x.Shape, od = o.Shape;
        int group = checked((int)U32(p, 4)), kh = I32(p, 8), kw = I32(p, 12), sh = I32(p, 16), sw = I32(p, 20),
            pt = I32(p, 32), pl = I32(p, 36);
        int n = id[0], cin = id[1], h = id[2], w = id[3], cout = od[1], oh = od[2], ow = od[3];
        ReadOnlySpan<float> biasData = bias is null ? [] : bias.Data;
        if (activation == NhwcActivation.Sigmoid)
            throw new InvalidOperationException("Sigmoid is not a fused NHWC convolution activation.");
        if (group == 1)
        {
            float[] packed = _model.GetPackedWeights(conv, Model.PackNhwcDense)
                ?? throw new InvalidOperationException($"NHWC dense weights unavailable for node '{conv.Name}'.");
            Nhwc.Dense(x.Data, packed, biasData, o.Data, n, cin, h, w, cout, oh, ow, kh, kw, sh, sw, pt, pl,
                residual, activation, alpha, beta, _intraOpThreads);
            return;
        }
        if (group == cin && cout == cin)
        {
            float[] packed = _model.GetPackedWeights(conv, Model.PackNhwcDepthwise)
                ?? throw new InvalidOperationException($"NHWC depthwise weights unavailable for node '{conv.Name}'.");
            Nhwc.Depthwise(x.Data, packed, biasData, o.Data, n, cin, h, w, oh, ow, kh, kw, sh, sw, pt, pl,
                residual, activation, alpha, beta, _intraOpThreads);
            return;
        }
        throw new NotSupportedException($"NHWC convolution with groups={group} at node '{conv.Name}' is not supported.");
    }

    private void ConvTransposeNhwc(NodeRecord node, TensorValue x, TensorValue o, ReadOnlySpan<float> bias, NhwcActivation activation)
    {
        int[] id = x.Shape, od = o.Shape;
        float[] packed = _model.GetPackedWeights(node, Model.PackNhwcConvTranspose)
            ?? throw new InvalidOperationException($"NHWC ConvTranspose weights unavailable for node '{node.Name}'.");
        int cout = od[1];
        if (od[2] != id[2] * 2 || od[3] != id[3] * 2)
            throw new InvalidDataException("NHWC ConvTranspose expects 2x2 / stride 2.");
        // Sigmoid is not fused: the shared vector kernel keeps the probability
        // map bit-identical to the NCHW path.
        bool separateSigmoid = activation == NhwcActivation.Sigmoid;
        Nhwc.ConvTranspose2x2Stride2(x.Data, packed, bias, o.Data, id[0], id[1], id[2], id[3], cout,
            separateSigmoid ? NhwcActivation.None : activation, _intraOpThreads);
        if (separateSigmoid)
            SimdKernels.SigmoidParallel(o.Data, o.Data, _intraOpThreads);
    }
}
