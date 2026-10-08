                        act = 2;
                        int preG = Phys(checked((int)nodes[ni + 1].Outputs[0]));
                        if (refCount[preG] > 2)
                        {
                            // conv+bias also feeds a residual outside the
                            // group: sink slot gets gelu(acc) via o; the preG
                            // slot gets raw acc via o2 (flag bit10).
                            dualT = (uint)preG; dualRaw = true;
                        }
                    }
                    else if (skip == 2
                             && nodes[ni + 1].Operator == OperatorId.Add
                             && nodes[ni + 2].Operator == OperatorId.Add)
                    {
                        // conv + channel-bias Add + residual Add
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1])
                            : checked((int)add.Inputs[0]);
                        if (!isConst[bt])
                            throw new NotSupportedException(
                                $"conv+add+add mid-add non-const at node {ni + 1}");
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
                        NodeRecord res = nodes[ni + 2];
                        int mid = Phys(checked((int)add.Outputs[0]));
                        uint oth = Phys(checked((int)res.Inputs[0])) == mid
                            ? res.Inputs[1]
                            : Phys(checked((int)res.Inputs[1])) == mid
                                ? res.Inputs[0] : uint.MaxValue;
                        if (oth == uint.MaxValue || isConst[(int)oth]
                            || numel[(int)oth] != numel[mid])
                            throw new NotSupportedException(
                                $"conv+add+add residual form at node {ni + 2}");
                        resT = oth;
                    }
                    else if (skip != 0)
                        throw new NotSupportedException(
                            $"conv fused skip={skip} at node {ni} (next={nodes[ni + 1].Operator})");

                    // ---- emit-time fusion extensions (compiler didn't group these) ----
                    // pointwise conv + residual Add (+ optional gelu pattern) —
                    // cm/dot kernels implement res binding (flags bit1) and act (bits4-6).
                    if (group == 1 && kH == 1 && kW == 1 && sH == 1 && sW == 1
                        && Environment.GetEnvironmentVariable("SIMD_OCR_NOEXTEND") == null)
                    {
                        int ge = ni + skip;
                        int j = ge + 1;
                        // residual Add may only be absorbed when the conv group
                        // carries no activation: kernels evaluate act(conv+res),
                        // but act(conv)+res is the ONNX semantics (e.g. relu then
                        // add must NOT become relu of the sum).
                        if (j < nodes.Length && nodes[j].Operator == OperatorId.Add
                            && !inConvGroup[j] && _compiled.FusedSkip(j) == 0
                            && !addScale.ContainsKey(j) && act == 0
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NOEXTR") == null)
                        {
                            NodeRecord ad = nodes[j];
                            int prev = Phys(checked((int)nodes[ge].Outputs[0]));
                            int ia = Phys(checked((int)ad.Inputs[0]));
                            int ib = Phys(checked((int)ad.Inputs[1]));
                            uint other = ia == prev ? ad.Inputs[1]
                                       : ib == prev ? ad.Inputs[0] : uint.MaxValue;
                            if (other != uint.MaxValue && !isConst[(int)other]
                                && numel[(int)other] == numel[prev]
                                && refCount[prev] == 1)
                            { resT = other; ge = j; j++; }
                        }
                        if (j < nodes.Length
                            && nodes[j].Operator == OperatorId.Div
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NOEXTG") == null
                            && _compiled.FusedSkip(j) == 4
                            && Phys(checked((int)nodes[j].Inputs[0]))
                                == Phys(checked((int)nodes[ge].Outputs[0])))
                        {
                            int preG = Phys(checked((int)nodes[ge].Outputs[0]));
                            if (refCount[preG] == 1 && act == 0)
                            { act = 2; ge = j + 4; }
                            else if (refCount[preG] == 2)
                            {
                                // out also feeds a residual elsewhere — conv dual-
                                // writes plain out (preG) and gelu(out) (gelu sink)
                                dualT = (uint)Phys(checked((int)nodes[j + 4].Outputs[0]));
                                outPhys = preG;
                                ge = j + 4;
                            }
                        }
                        if (ge != ni + skip)
                        {
                            if (dualT == uint.MaxValue)
                                outPhys = Phys(checked((int)nodes[ge].Outputs[0]));
                            skip = ge - ni;
                        }
                    }
                    bool hasPs = prescale.TryGetValue(ni, out (uint x, uint se) psv);
                    uint flags = hasBias | scalarBias | (uint)(act << 4) | flags16
                        | (resT != uint.MaxValue ? 2u : 0u)
                        | (hasPs ? 4u : 0u)
                        | (dualT != uint.MaxValue ? 8u : 0u)
                        | (dualRaw ? 1024u : 0u);

                    if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && group == 1)
                    {
                        int inT = Phys(checked((int)node.Inputs[0]));
                        int cinIn = (inT == inIdx && cin % 4 != 0) ? 4 : cin;
                        if (cinIn % 4 != 0)
                            throw new NotSupportedException($"conv1x1 cin={cin} not %4 at node {ni}");
                        VkBuffer wbuf = ConstF16(checked((int)node.Inputs[1]),
                            padRows: ((cout + 127) / 128) * 128, rowElems: cin,
                            rowPad: cinIn);
                        uint mImg = (uint)(outH * outW);
                        uint M = mImg * (uint)nb;   // flat rows span the batch
                        bool dotOk = cout % 4 == 0 && scalarBias == 0
                            && (cinIn <= 128 || (hasPs && cinIn <= 256))
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NODOT") == null;
                        // sg32 coopmat (SE prescale included) outruns the dot
                        // kernel once the output leaves split-K range (lite:
                        // a quarter of that already);
                        // addps-absorbed inputs need the dot kernel's fused read
                        if (dotOk && _sg32 && (long)M * cout > 1L << (_lite ? 16 : 18)
