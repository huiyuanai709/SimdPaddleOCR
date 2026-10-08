                    if (hswish) ni += 1;
                    long n = numel[outPhys];
                    uint op2 = node.Operator switch
                    {
                        OperatorId.Relu => 1u, OperatorId.HardSigmoid => 3u,
                        OperatorId.Sigmoid => 4u, OperatorId.Erf => 5u,
                        OperatorId.Sqrt => 6u, _ => 0u,
                    };
                    if (hswish) op2 = 14u;
                    uint aux = 0u;
                    if (node.Operator == OperatorId.HardSigmoid)
                        aux = HsAux(_model.GetParameters(node));
                    Emit(n % 4 == 0 ? _pElem4 : _pElem, $"un{node.Operator} n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, 0, 2), (arena, off[outPhys], 2)],
                        [(uint)n, 1u, op2, 0u, 0u, aux], Div256(n % 4 == 0 ? n / 4 : n));
                    break;
                }

                case OperatorId.ReduceMean when skip == 8:
                {
                    // compiler-fused LayerNorm over the last axis
                    int xT = checked((int)node.Inputs[0]);
                    int[] xs = shapes[xT];
                    int cN = xs[^1];
                    int xP = Phys(xT);
                    // last axis must be physically contiguous: rank-3 row-major,
                    // or an NHWC slab whose channel dim is that axis
                    bool contiguous = xs.Length == 3
                        && (shapes[xP].Length != 4 || shapes[xP][1] == cN);
                    int ConstIn(NodeRecord nd, int other)
                    {
                        int a0 = checked((int)nd.Inputs[0]), a1 = checked((int)nd.Inputs[1]);
                        int c = Phys(a0) == Phys(other) ? a1 : Phys(a1) == Phys(other) ? a0 : -1;
                        return c >= 0 && isConst[c] ? c : -1;
                    }
                    NodeRecord nSub = nodes[ni + 1], nPow = nodes[ni + 2], nRm = nodes[ni + 3],
                        nEps = nodes[ni + 4], nSq = nodes[ni + 5], nDiv = nodes[ni + 6],
                        nG = nodes[ni + 7], nB = nodes[ni + 8];
                    int powC = nPow.Operator == OperatorId.Pow
                        ? ConstIn(nPow, checked((int)nSub.Outputs[0])) : -1;
                    int epsC = nEps.Operator == OperatorId.Add
                        ? ConstIn(nEps, checked((int)nRm.Outputs[0])) : -1;
                    int gC = nG.Operator == OperatorId.Mul
                        ? ConstIn(nG, checked((int)nDiv.Outputs[0])) : -1;
                    int bC = nB.Operator == OperatorId.Add
                        ? ConstIn(nB, checked((int)nG.Outputs[0])) : -1;
                    bool lastAxis = U16(p, 2) == 1
                        && (I32(p, 12) == -1 || I32(p, 12) == xs.Length - 1);
                    if (!contiguous || !lastAxis || cN > 1024 || numel[xT] / cN > 65535
                        || nSub.Operator != OperatorId.Sub
                        || nRm.Operator != OperatorId.ReduceMean || nSq.Operator != OperatorId.Sqrt
                        || nDiv.Operator != OperatorId.Div || powC < 0 || epsC < 0 || gC < 0
                        || bC < 0 || CstF32(powC)[0] != 2f || numel[epsC] != 1
                        || numel[gC] != cN || numel[bC] != cN)
                        throw new NotSupportedException($"layernorm pattern at node {ni}");
                    long rowsLn = numel[xT] / cN;
                    Emit(_pLn, $"layernorm n{ni} {rowsLn}x{cN}",
                        [(arena, SlotOf(node.Inputs[0]), 2), (ConstF32(gC), 0, 4),
                         (ConstF32(bC), 0, 4), (arena, off[outPhys], 2)],
                        [(uint)rowsLn, (uint)cN, BitConverter.SingleToUInt32Bits(CstF32(epsC)[0])],
                        (uint)rowsLn);
                    ni += 8;
                    break;
                }

                case OperatorId.ReduceMean:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int hw = ishp[2] * ishp[3], c = ishp[1];
                    int cv4 = c / 4;
                    if (seEmit.TryGetValue(ni, out (int fc1, int fc2, int hs) sev))
                    {
                        NodeRecord f1 = nodes[sev.fc1], f2 = nodes[sev.fc2];
                        ReadOnlySpan<byte> hpp = _model.GetParameters(nodes[sev.hs]);
                        int rDim = shapes[Phys(checked((int)f1.Outputs[0]))][1];
                        uint cvP = 1; while (cvP < (uint)(c / 4)) cvP <<= 1;
                        int hsOut = Phys(checked((int)nodes[sev.hs].Outputs[0]));
                        int s = PartSplits(hw, c);
                        int pp = (hw + s - 1) / s;
                        VkBuffer pb2 = PartBuf(s * c * nb + 4 * nb);
                        // 64 uints = 256 bytes. ctr[batch] stays packed inside
                        // the slot; the next block starts on the next slot.
                        const int seAlign = 64;
                        int seStride = (nb + seAlign - 1) & ~(seAlign - 1);
                        if (seCtr + seStride > (64 * 1024) / 4)
                            throw new NotSupportedException(
                                $"SE ticket region exhausted at node {ni} (batch {nb})");
                        Emit(_pSeF, $"se_f n{ni} c{c} s{s} r{rDim}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (pb2, 0, 2),
                             (pb2, PartCtrElem + (long)seCtr, 4),
                             (ConstF32(checked((int)f1.Inputs[1])), 0, 4),
                             (ConstF32(checked((int)f1.Inputs[2])), 0, 4),
                             (ConstF32(checked((int)f2.Inputs[1])), 0, 4),
                             (ConstF32(checked((int)f2.Inputs[2])), 0, 4),
                             (arena, off[hsOut], 2)],
                            [(uint)hw, (uint)c, (uint)s, (uint)pp, cvP,
                             (uint)rDim,
                             BitConverter.SingleToUInt32Bits(F32(hpp, 4)),
                             BitConverter.SingleToUInt32Bits(F32(hpp, 8))],
                            (uint)s, (uint)nb);
                        seCtr += seStride;
                        break;
                    }
                    // lite and wide no-coopmat parts take it at any size:
                    // reduce_hw's one-channel workgroups read 2 bytes at a C*2 stride
                    if (c % 4 == 0 && 256 % cv4 == 0 && (hw >= 4096 || _lite || _ncWide))
                    {
                        // two-phase: S pixel partitions → fp32 partials → mean
                        int s = PartSplits(hw, c);
                        int pp = (hw + s - 1) / s;
                        VkBuffer pb = PartBuf(s * c * nb);
                        Emit(_pReduce4, $"reduce4a n{ni} c{c}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (pb, 0, 2)],
                            [(uint)hw, (uint)c, (uint)s, (uint)pp],
                            (uint)s, (uint)nb);
                        Emit(_pReduce4b, $"reduce4b n{ni} c{c}",
                            [(pb, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)s, (uint)c, (uint)hw], (uint)c, (uint)nb);
                    }
