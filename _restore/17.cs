                    if (ishp[1] % 4 == 0 && oshp[1] % 4 == 0)
                        Emit(_pConvT4, $"convT4 n{ni} {ishp[1]}->{oshp[1]}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstConvT(checked((int)node.Inputs[1]), ishp[1], oshp[1]), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)oshp[3], (uint)oshp[2], (uint)oshp[1], (uint)ishp[1],
                             (uint)ishp[3], flags2 | (uint)(act << 4)],
                            Div256(numel[outPhys] / 4));
                    else
                    {
                        // scalar-tail convT writing the graph output: emit fp32
                        // straight into the readback buffer (kills the `out` rec)
                        if (outPhys == Phys(outIdx)) { flags2 |= 128u; convTOut = true; }
                        Emit(_pConvT, $"convT n{ni} {ishp[1]}->{oshp[1]}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2),
                             (outF32, 0, 4)],
                            [(uint)oshp[3], (uint)oshp[2], (uint)oshp[1], (uint)ishp[1],
                             (uint)ishp[3], flags2 | (uint)(act << 4)],
                            Div256(numel[outPhys]));
                    }
                    ni += skip;
                    break;
                }

                case OperatorId.Transpose:
                    // rank-3 [0,2,1] transposes were folded into aliases in
                    // pass 1 (channel-last physical layout makes them free).
                    if (Phys(checked((int)node.Outputs[0]))
                        != Phys(checked((int)node.Inputs[0])))
                        throw new NotSupportedException(
                            $"non-trivial Transpose at node {ni}");
                    break;

                case OperatorId.AveragePool:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] osp2 = shapes[node.Outputs[0]];
                    if (ishp[1] % 4 != 0 || ishp.Length != 4)
                        throw new NotSupportedException(
                            $"avgpool C%4 at node {ni}");
                    Emit(_pAvg4, $"avgpool4 n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, off[outPhys], 2)],
                        [(uint)osp2[3], (uint)osp2[2], (uint)ishp[1],
                         (uint)I32(p, 8), (uint)I32(p, 12),
                         (uint)I32(p, 16), (uint)I32(p, 20),
                         (uint)I32(p, 24), (uint)I32(p, 28),
                         (uint)ishp[3], (uint)ishp[2]],
                        Div256(numel[outPhys] / 4), (uint)nb);
                    break;
                }

                case OperatorId.BatchNormalization:
                {
                    // fold to per-channel affine: o = x*s4 + t4 where
                    // s = scale/sqrt(var+eps), t = bias - mean*s
                    float eps = F32(p, 4);
                    int bt = checked((int)node.Inputs[1]);
                    ReadOnlySpan<float> sc = CstF32(bt),
                        bi = CstF32(checked((int)node.Inputs[2])),
                        mn = CstF32(checked((int)node.Inputs[3])),
                        va = CstF32(checked((int)node.Inputs[4]));
                    int cc = sc.Length;
                    float[] s = new float[cc], t = new float[cc];
                    for (int i = 0; i < cc; i++)
                    {
                        s[i] = sc[i] / MathF.Sqrt(va[i] + eps);
                        t[i] = bi[i] - mn[i] * s[i];
                    }
                    long n = numel[outPhys];
                    if (n % 4 != 0 || cc % 4 != 0)
                        throw new NotSupportedException($"bn vec4 at node {ni}");
                    Emit(_pAffine, $"bn4 n{ni} c{cc}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (VecF16((ni, 0), s), 0, 2), (VecF16((ni, 1), t), 0, 2),
                         (arena, off[outPhys], 2)],
                        [(uint)(n / 4), (uint)(cc / 4)], Div256(n / 4));
                    break;
                }

                case OperatorId.Softmax:
                {
                    // last-axis softmax (cls head [n,2]); one invocation per row.
                    int[] ssp = shapes[node.Inputs[0]];
                    long cols = ssp[^1];
                    long rows = numel[outPhys] / cols;
                    if (cols <= 0 || rows <= 0 || rows * cols != numel[outPhys])
                        throw new NotSupportedException($"softmax at node {ni}");
                    Emit(_pSoftmax, $"softmax n{ni} {rows}x{cols}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, off[outPhys], 2)],
                        [(uint)rows, (uint)cols], Div256(rows));
                    break;
                }

                case OperatorId.MatMul:
                {
                    // [n,M,K] x [K,N]: flat rows over batch feed the coopmat
                    // 1x1-conv GEMM as-is; weights repack to [N,K] fp16.
                    int[] asp = shapes[node.Inputs[0]];
                    int[] bsp = shapes[node.Inputs[1]];
                    int mmK = asp[^1], mmN = bsp[^1];
                    if (bsp.Length != 2 || !isConst[(int)node.Inputs[1]]
                        || mmK % 4 != 0 || mmN % 4 != 0)
                        throw new NotSupportedException($"matmul at node {ni}");
                    uint mmM = (uint)(numel[checked((int)node.Inputs[0])] / mmK);
                    // fold a following bias Add [N] into the GEMM epilogue
                    uint mmFlags = 0; VkBuffer mmBias = arena;
                    if (ni + 1 < emitLimit
                        && nodes[ni + 1].Operator == OperatorId.Add
                        && _compiled.FusedSkip(ni + 1) == 0)
                    {
                        NodeRecord ad = nodes[ni + 1];
                        uint oth = ad.Inputs[0] == node.Outputs[0] ? ad.Inputs[1]
                                 : ad.Inputs[1] == node.Outputs[0] ? ad.Inputs[0]
                                 : uint.MaxValue;
                        if (oth != uint.MaxValue && isConst[(int)oth]
                            && numel[(int)oth] == mmN
