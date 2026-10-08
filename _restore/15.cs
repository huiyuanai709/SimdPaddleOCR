                    else
                        Emit(_pReduce, $"reduceHW n{ni} c{c}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)hw, (uint)c], (uint)c, (uint)nb);
                    break;
                }

                case OperatorId.MaxPool:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    // both kernels hard-code 2x2 / stride 1 / pad-end
                    if (!(I32(p, 8) == 2 && I32(p, 12) == 2 && I32(p, 16) == 1
                          && I32(p, 20) == 1 && I32(p, 24) == 0 && I32(p, 28) == 0
                          && shapes[node.Outputs[0]][2] == ishp[2]
                          && shapes[node.Outputs[0]][3] == ishp[3]))
                        throw new NotSupportedException($"maxpool shape at node {ni}");
                    if (ishp[1] % 4 == 0)
                    {
                        long wO = off[outPhys]; uint dcv = 0, dco = 0;
                        {
                            int rp = outPhys;
                            List<int>? cs = consumers[rp];
                            if (!_noCatAbs && refCount[rp] == 1 && cs != null && cs.Count == 1
                                && nodes[cs[0]].Operator == OperatorId.Concat)
                            {
                                int cn = cs[0];
                                int[] ospc = shapes[nodes[cn].Outputs[0]];
                                int co = 0; bool hit = false;
                                foreach (uint cin_ in nodes[cn].Inputs)
                                {
                                    if (Phys(checked((int)cin_)) == rp) { hit = true; break; }
                                    co += shapes[checked((int)cin_)][1];
                                }
                                if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                                {
                                    wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                    dcv = (uint)(ospc[1] / 4);
                                    dco = (uint)(co / 4);
                                    concatAbsorbed.Add(rp);
                                }
                            }
                        }
                        Emit(_pPool4, $"maxpool4 n{ni}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, wO, 2)],
                            [(uint)ishp[3], (uint)ishp[2], (uint)ishp[1], dcv, dco],
                            Div256(numel[outPhys] / 4 / nb), (uint)nb);
                    }
                    else if (nb > 1)
                        throw new NotSupportedException($"scalar maxpool lacks batch support (n{ni})");
                    else
                        Emit(_pPool, $"maxpool n{ni}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)ishp[3], (uint)ishp[2], (uint)ishp[1]],
                            Div256(numel[outPhys]));
                    break;
                }

                case OperatorId.Resize:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"resize lacks batch support (n{ni})");
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] osp = shapes[node.Outputs[0]];
                    if (osp[1] % 4 == 0)
                    {
                        // sole-consumer fusion: resize -> Add (FPN merge) or
                        // resize -> Concat (write into the channel slice)
                        int rp = outPhys;
                        List<int>? cs = consumers[rp];
                        int cn = refCount[rp] == 1 && cs != null && cs.Count == 1
                            ? cs[0] : -1;
                        if (cn >= 0 && nodes[cn].Operator == OperatorId.Add
                            && _compiled.FusedSkip(cn) == 0 && !inConvGroup[cn]
                            && !addScale.ContainsKey(cn))
                        {
                            NodeRecord ad = nodes[cn];
                            uint oth = ad.Inputs[0] == node.Outputs[0]
                                ? ad.Inputs[1] : ad.Inputs[0];
                            int othP = Phys(checked((int)oth));
                            if (!isConst[othP] && numel[othP] == numel[rp]
                                && Phys(checked((int)ad.Inputs[0]))
                                   != Phys(checked((int)ad.Inputs[1])))
                            {
                                int ao = Phys(checked((int)ad.Outputs[0]));
                                // addps-absorb on either operand
                                long xOff4 = SlotOf(node.Inputs[0]);
                                long aOff4 = SlotOf(oth), raOff4 = 0, seOff4 = 0;
                                uint raFl = 0;
                                if (addpsSrc.TryGetValue(
                                    Phys(checked((int)node.Inputs[0])), out var adx))
                                { xOff4 = SlotOf(adx.x); raOff4 = SlotOf(adx.a);
                                  seOff4 = SlotOf(adx.se); raFl |= 256u; }
                                if (addpsSrc.TryGetValue(
                                    Phys(checked((int)oth)), out var ada))
                                { aOff4 = SlotOf(ada.x); raOff4 = SlotOf(ada.a);
                                  seOff4 = SlotOf(ada.se); raFl |= 512u; }
                                Emit(_pResize4Add, $"resizeadd n{ni}+{cn}",
                                    [(arena, xOff4, 2),
                                     (arena, aOff4, 2),
                                     (arena, off[ao], 2),
                                     (arena, raOff4, 2), (arena, seOff4, 2)],
                                    [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                                     (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3]),
                                     raFl],
                                    Div256(numel[rp] / 4));
                                skipEmit.Add(cn);
                                break;
                            }
                        }
                        long wO = off[outPhys]; uint dcv = 0, dco = 0;
                        if (!_noCatAbs && cn >= 0 && nodes[cn].Operator == OperatorId.Concat)
                        {
                            int[] ospc = shapes[nodes[cn].Outputs[0]];
                            int co = 0; bool hit = false;
                            foreach (uint cin_ in nodes[cn].Inputs)
                            {
                                if (Phys(checked((int)cin_)) == rp) { hit = true; break; }
                                co += shapes[checked((int)cin_)][1];
                            }
