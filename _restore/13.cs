                             (uint)inW, (uint)inH, (uint)K, (uint)Kp,
                             cinIn % 4 != 0 ? 1u : 0u],
                            Div256(M * (long)(Kp / 4)));
                        VkBuffer wbuf = ConstTapMajor(checked((int)node.Inputs[1]),
                            cout, cin, kH, kW, Kp, cinPad: cinIn);
                        if (cout % 4 == 0 && scalarBias == 0 && Kp <= 128
                            && !(_nocm && (long)M * cout > 1L << 16))
                        {
                            VkBuffer wk = ConstKMajor(checked((int)node.Inputs[1]),
                                cout, cin, kH * kW, Kp, cinIn);
                            Emit(_pDot, $"im2c_dot n{ni} {M}x{Kp}x{cout}",
                                [(arena, im2colOff, 2), (wk, 0, 2),
                                 (biasBuf, 0, 2), (arena, 0, 2),
                                 (arena, off[outPhys], 2), (arena, 0, 2),
                                 (arena, 0, 2)],
                                [M, (uint)cout, (uint)Kp, (uint)(Kp / 4),
                                 flags, 0u, 0u],
                                Div256(((M + 3) / 4) * (long)(cout / 4)));
                        }
                        else
                        {
                            var (cp, tm, tn) = CmTile(cout);
                            Emit(NcTail(cp, flags), $"im2colconv n{ni} {M}x{Kp}x{cout}",
                                [(arena, im2colOff, 2), (wbuf, 0, 2),
                                 (biasBuf, 0, 2), (arena, 0, 2), (arena, off[outPhys], 2)],
                                [M, (uint)cout, (uint)Kp, flags],
                                (M + tm - 1) / tm, (uint)((cout + tn - 1) / tn));
                        }
                        }
                    }
                    else
                    {
                        if (nb > 1)
                            throw new NotSupportedException(
                                $"dense conv path lacks batch support (n{ni})");
                        Emit(_pDense, $"conv3x3 n{ni} {cin}->{cout} k{kH}s{sH}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)outW, (uint)outH, (uint)cout, (uint)cin,
                             (uint)kH, (uint)kW, (uint)sH, (uint)sW,
                             (uint)pt, (uint)pl, (uint)inW, (uint)inH, flags],
                            Div256((long)outW * outH * cout));
                    }
                    ni += skip;
                    break;
                }

                case OperatorId.Div when skip == 4:
                {
                    // fused GELU: write to the group sink (last Mul output)
                    long n = numel[Phys(checked((int)nodes[ni + 4].Outputs[0]))];
                    Emit(n % 4 == 0 ? _pElem4 : _pElem, $"gelu n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, 0, 2), (arena, off[outPhys], 2)],
                        [(uint)n, 1u, 2u, 0u, 0u], Div256(n % 4 == 0 ? n / 4 : n));
                    ni += 4;
                    break;
                }

                case OperatorId.Add or OperatorId.Mul or OperatorId.Sub or OperatorId.Div
                    or OperatorId.Pow:
                {
                    long n = numel[outPhys];
                    if (addScale.TryGetValue(ni, out (uint x, uint se, uint oth) asc))
                    {
                        if (addpsSrc.ContainsKey(outPhys)) break; // consumer absorbs
                        // a + x*se — the SE Mul got skipped, fold it here
                        // per-batch se vector: imgV = vec4 count of one image
                        long seN = numel[(int)asc.se];
                        Emit(_pAddPs, $"addps n{ni}",
                            [(arena, SlotOf(asc.oth), 2), (arena, SlotOf(asc.x), 2),
                             (arena, SlotOf(asc.se), 2), (arena, off[outPhys], 2)],
                            [(uint)n, (uint)(seN / 4 / nb),
                             (uint)(seN == n ? 0 : n / 4 / nb)], Div256(n / 4));
                        break;
                    }
                    int[] osp = shapes[checked((int)node.Outputs[0])];
                    int chan = osp.Length >= 2 ? osp[1] : (int)n;
                    (VkBuffer bA, long oA, uint mA) = Operand(checked((int)node.Inputs[0]), chan, n);
                    (VkBuffer bB, long oB, uint mB) = Operand(checked((int)node.Inputs[1]), chan, n);
                    // vector broadcast along the contiguous (last physical) dim:
                    // rank-3 tails like [n,T,C] + [C] need C as the modulus —
                    // the vector operand's own numel, not osp[1].
                    {
                        long neA = numel[checked((int)node.Inputs[0])];
                        long neB = numel[checked((int)node.Inputs[1])];
                        // per-image channel vector [n,C,1,1] (nel == chan*nb):
                        // mode 3 = kernel picks b[(i/perImg)*C + i%C] using aux;
                        // flat tail vector [C] vs [n,T,C]: mode 1, chan = neB.
                        if (mB == 2 && neB > 1 && neB < n)
                        { if (neB == (long)chan * nb) mB = 3; else { mB = 1; chan = (int)neB; } }
                        else if (mA == 2 && neA > 1 && neA < n)
                        { if (neA == (long)chan * nb) mA = 3; else { mA = 1; chan = (int)neA; } }
                    }
                    uint op2 = node.Operator switch
                    {
                        OperatorId.Add => 8u, OperatorId.Mul => 9u, OperatorId.Sub => 10u,
                        OperatorId.Div => 11u, OperatorId.Pow => 13u, _ => 8u,
                    };
                    bool v4 = n % 4 == 0 && (mA is not (1 or 3) || chan % 4 == 0)
                        && (mB is not (1 or 3) || chan % 4 == 0);
                    Emit(v4 ? _pElem4 : _pElem, $"bin{node.Operator} n{ni}",
                        [(bA, oA, 2), (bB, oB, 2), (arena, off[outPhys], 2)],
                        [(uint)n, (uint)chan, op2, mA, mB, (uint)(n / nb)], Div256(v4 ? n / 4 : n));
                    break;
                }

                case OperatorId.Relu or OperatorId.Sigmoid or OperatorId.HardSigmoid
                    or OperatorId.Erf or OperatorId.Sqrt:
                {
                    // compiler-fused HS+Mul (hardswish, x*hs(x)): emit op 14
                    // against the group's sink instead of a bare hardsigmoid.
                    bool hswish = node.Operator == OperatorId.HardSigmoid
                        && skip == 1
                        && nodes[ni + 1].Operator == OperatorId.Mul
                        && Phys(checked((int)nodes[ni + 1].Inputs[0]))
                           == Phys(checked((int)node.Outputs[0]))
                        && Phys(checked((int)nodes[ni + 1].Inputs[1]))
                           == Phys(checked((int)node.Inputs[0]));
