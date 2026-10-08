                            && refCount[Phys(checked((int)node.Outputs[0]))] == 1)
                        {
                            mmBias = ConstF16(checked((int)oth));
                            mmFlags |= 1u;
                            outPhys = Phys(checked((int)ad.Outputs[0]));
                            ni += 1;
                        }
                    }
                    // narrow-M token GEMMs (SVTR mixer, n*T rows): the direct
                    // dot kernel keeps occupancy where a 128x128 coopmat tile
                    // would leave most of the M dimension idle. Its epilogue
                    // also folds swish (Sigmoid+Mul) or a residual Add.
                    bool mmDot = mmM <= 16384
                        && Environment.GetEnvironmentVariable("SIMD_OCR_MMCM") == null;
                    if (mmDot)
                    {
                        int cur = outPhys;
                        long resOff = 0;
                        int j = ni + 1;
                        bool Free(int t) => t != Phys(outIdx) && refCount[t] >= 1;
                        if (j + 1 < emitLimit && (mmFlags & 1u) != 0
                            && nodes[j].Operator == OperatorId.Sigmoid
                            && nodes[j + 1].Operator == OperatorId.Mul
                            && _compiled.FusedSkip(j) == 0 && _compiled.FusedSkip(j + 1) == 0
                            && Phys(checked((int)nodes[j].Inputs[0])) == cur
                            && refCount[cur] == 2 && Free(cur)
                            && refCount[Phys(checked((int)nodes[j].Outputs[0]))] == 1
                            && ((Phys(checked((int)nodes[j + 1].Inputs[0])) == cur
                                 && Phys(checked((int)nodes[j + 1].Inputs[1]))
                                    == Phys(checked((int)nodes[j].Outputs[0])))
                             || (Phys(checked((int)nodes[j + 1].Inputs[1])) == cur
                                 && Phys(checked((int)nodes[j + 1].Inputs[0]))
                                    == Phys(checked((int)nodes[j].Outputs[0])))))
                        {
                            mmFlags |= 5u << 4;
                            outPhys = Phys(checked((int)nodes[j + 1].Outputs[0]));
                            ni += 2;
                        }
                        else if (j < emitLimit && nodes[j].Operator == OperatorId.Add
                            && _compiled.FusedSkip(j) == 0 && refCount[cur] == 1 && Free(cur)
                            && !addScale.ContainsKey(j))
                        {
                            NodeRecord ad = nodes[j];
                            int ia = Phys(checked((int)ad.Inputs[0]));
                            int ib = Phys(checked((int)ad.Inputs[1]));
                            int oth = ia == cur ? checked((int)ad.Inputs[1])
                                    : ib == cur ? checked((int)ad.Inputs[0]) : -1;
                            if (oth >= 0 && !isConst[oth] && numel[oth] == numel[cur]
                                && Phys(oth) != cur)
                            {
                                resOff = SlotOf((uint)oth);
                                mmFlags |= 2u;
                                outPhys = Phys(checked((int)ad.Outputs[0]));
                                ni += 1;
                            }
                        }
                        // fewer than ~64 workgroups of 4x4 tiles: the serial K
                        // loop is the latency floor, so split K over 16 lanes
                        long tiles = ((mmM + 3) / 4) * (long)(mmN / 4);
                        bool splitK = mmK >= 64 && tiles <= 16384 && !s_noSplitK;
                        Emit(splitK ? _pDotSk : _pDot, $"mmdot{(splitK ? "sk" : "")} n{ni} {mmM}x{mmK}x{mmN}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                             (mmBias, 0, 2), (arena, resOff, 2),
                             (arena, off[outPhys], 2), (arena, 0, 2),
                             (arena, 0, 2), (arena, 0, 2)],
                            [mmM, (uint)mmN, (uint)mmK, (uint)(mmK / 4), mmFlags, 0u, 0u],
                            splitK ? (uint)((tiles + 15) / 16) : Div256(tiles));
                        break;
                    }
                    var (cp2, tm2, tn2) = CmTile(mmN, mmM, mmK);
                    Emit(NcTail(cp2, mmFlags), $"matmul n{ni} {mmM}x{mmK}x{mmN}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (ConstGemmW(checked((int)node.Inputs[1]), mmK, mmN), 0, 2),
                         (mmBias, 0, 2), (arena, 0, 2),
                         (arena, off[outPhys], 2),
                         (arena, 0, 2)],
                        [mmM, (uint)mmN, (uint)mmK, mmFlags],
                        (mmM + tm2 - 1) / tm2, (uint)((mmN + tn2 - 1) / tn2));
                    break;
                }

                default:
                    throw new NotSupportedException($"op {node.Operator} at node {ni}");
            }
            }
            catch (Exception ex)
            {
                throw new NotSupportedException(
                    $"BuildPlan failed at node {ni} op={node.Operator} " +
                    $"in=[{string.Join(',', node.Inputs)}] out=[{string.Join(',', node.Outputs)}]: " +
                    $"{ex.GetType().Name}: {ex.Message}", ex);
            }
        }

        // finalize: graph output → fp32 readback.
        // If the last node is a STANDALONE Sigmoid (its own elem dispatch was
        // the last rec), replace it with a sigmoid-ing f32 writeback. If the
        // sigmoid was consumed into a fused group (e.g. convT+bias+sigmoid),
        // the sink slot already holds sigmoid'd values → plain copy.
        int lastNi = recs.Count > 0 ? LastEmitNi : -1;
        NodeRecord last = nodes[nodes.Length - 1];
        bool standaloneSigmoid = lastNi == nodes.Length - 1
            && last.Operator == OperatorId.Sigmoid
            && checked((int)last.Outputs[0]) == outIdx;
        int outSrc;
        uint outAct;
        if (standaloneSigmoid)
        {
            recs.RemoveAt(recs.Count - 1);
            outSrc = Phys(checked((int)last.Inputs[0]));
            outAct = 4u;
        }
        else
        {
            outSrc = Phys(outIdx);
            outAct = 0u;
        }
        if (!convTOut)
            Emit(_pOut, "out", [(arena, off[outSrc], 2), (outF32, 0, 4)],
