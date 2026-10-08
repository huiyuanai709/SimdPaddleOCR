                            if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                            {
                                wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                dcv = (uint)(ospc[1] / 4);
                                dco = (uint)(co / 4);
                                concatAbsorbed.Add(rp);
                            }
                        }
                        bool aps = addpsSrc.TryGetValue(
                            Phys(checked((int)node.Inputs[0])), out var ads);
                        Emit(_pResize4, $"resize4 n{ni} x{osp[2] / ishp[2]}",
                            [(arena, aps ? SlotOf(ads.x)
                                        : SlotOf(node.Inputs[0]), 2),
                             (arena, wO, 2),
                             (arena, aps ? SlotOf(ads.a) : 0, 2),
                             (arena, aps ? SlotOf(ads.se) : 0, 2)],
                            [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                             (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3]),
                             dcv, dco, aps ? 256u : 0u],
                            Div256(numel[outPhys] / 4));
                    }
                    else
                        Emit(_pResize, $"resize n{ni} x{osp[2] / ishp[2]}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                             (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3])],
                            Div256(numel[outPhys]));
                    break;
                }

                case OperatorId.Concat:
                {
                    int[] osp = shapes[node.Outputs[0]];
                    if (osp.Length != 4 || (I32(p, 4) is int cax && cax != 1 && cax != -3))
                        throw new NotSupportedException($"concat axis/rank at node {ni}");
                    // NHWC channel concat: pixels are flat over the batch too
                    long catPx = (long)osp[2] * osp[3] * nb;
                    if (catRes.TryGetValue(ni, out var spec))
                    {
                        var bb = new (VkBuffer, long, int)[13];
                        var pc2 = new uint[19];
                        pc2[0] = (uint)osp[3]; pc2[1] = (uint)osp[2];
                        pc2[2] = (uint)osp[1];
                        for (int s = 0; s < 4; s++)
                        {
                            long xP = 0, raP = 0, seP = 0;
                            uint fl = 0, inW3 = 0, fhfw = 0, offc = 0;
                            if (s < spec.Length)
                            {
                                var t = spec[s];
                                xP = t.Item1; raP = t.Item2; seP = t.Item3;
                                fl = t.Item4; inW3 = t.Item5;
                                fhfw = t.Item6 | (t.Item7 << 16);
                                offc = t.Item8 | (t.Item9 << 16);
                            }
                            bb[s * 3] = (arena, xP, 2);
                            bb[s * 3 + 1] = (arena, raP, 2);
                            bb[s * 3 + 2] = (arena, seP, 2);
                            pc2[3 + s * 4] = offc; pc2[4 + s * 4] = inW3;
                            pc2[5 + s * 4] = fhfw; pc2[6 + s * 4] = fl;
                        }
                        bb[12] = (arena, off[outPhys], 2);
                        Emit(_pCatRes, $"catres n{ni}", bb, pc2,
                            Div256(numel[outPhys] / 4));
                        break;
                    }
                    int cOff = 0;
                    foreach (uint inp in node.Inputs)
                    {
                        int it = checked((int)inp);
                        int ci = shapes[it][1];
                        if (concatAbsorbed.Contains(Phys(it)))
                        { cOff += ci; continue; }
                        if (ci % 4 == 0 && cOff % 4 == 0 && osp[1] % 4 == 0)
                        {
                            bool aps = addpsSrc.TryGetValue(Phys(it), out var ads);
                            if (aps && nb > 1)   // concat4's se index is batch-free
                                throw new NotSupportedException($"batched addps concat at node {ni}");
                            Emit(_pConcat4, $"concat4 n{ni} c{ci}@{cOff}",
                                [(arena, aps ? SlotOf(ads.x) : SlotOf(inp), 2),
                                 (arena, off[outPhys], 2),
                                 (arena, aps ? SlotOf(ads.a) : 0, 2),
                                 (arena, aps ? SlotOf(ads.se) : 0, 2)],
                                [(uint)catPx, (uint)ci, (uint)osp[1],
                                 (uint)cOff, aps ? 256u : 0u],
                                Div256(catPx * (ci / 4)));
                        }
                        else
                            Emit(_pConcat, $"concat n{ni} c{ci}@{cOff}",
                                [(arena, SlotOf(inp), 2), (arena, off[outPhys], 2)],
                                [(uint)catPx, (uint)ci, (uint)osp[1], (uint)cOff],
                                Div256(catPx * ci));
                        cOff += ci;
                    }
                    break;
                }

                case OperatorId.ConvTranspose:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"convtranspose lacks batch support (n{ni})");
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] oshp = shapes[node.Outputs[0]];
                    if (!(I32(p, 8) == 2 && I32(p, 12) == 2 && I32(p, 16) == 2 && I32(p, 20) == 2
                          && I32(p, 32) == 0 && I32(p, 36) == 0))
                        throw new NotSupportedException($"ConvTranspose shape at node {ni}");
                    int act = 0;
                    VkBuffer biasBuf = arena;
                    uint flags2 = 0;
                    if (skip == 2 && nodes[ni + 1].Operator == OperatorId.Add)
                    {
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1]) : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt);
                        flags2 = 1u | (numel[bt] == 1 ? 8u : 0u);
                        OperatorId actOp = nodes[ni + 2].Operator;
                        act = actOp == OperatorId.Relu ? 1 : actOp == OperatorId.Sigmoid ? 4 : 0;
                    }
