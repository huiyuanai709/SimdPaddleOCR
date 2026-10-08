                    srcP = Phys(checked((int)nodes[prod].Inputs[0]));
                    skipTmp.Add(prod);
                }
                else if (refCount[ip] == 1 && consumers[ip]?.Count == 1
                    && consumers[ip][0] == ci2 && prod >= 0
                    && !addpsSrc.ContainsKey(ip))
                {
                    // sole-consumed producer emits a slice-write into this
                    // concat instead of materializing its own tensor —
                    // catRes cannot read it back; keep the old path.
                    okc = false; break;
                }
                uint fl = 0; long xP = off[srcP], raP = 0, seP = 0;
                if (addpsSrc.TryGetValue(srcP, out var ads))
                {
                    xP = off[Phys(checked((int)ads.x))];
                    raP = off[Phys(checked((int)ads.a))];
                    seP = off[Phys(checked((int)ads.se))];
                    fl = 1;
                }
                spec[ii] = (xP, raP, seP, fl, inW3, fh3, fw3,
                            (uint)(cOff3 / 4), (uint)(ci3 / 4));
                cOff3 += ci3;
            }
            if (!okc) continue;
            catRes[ci2] = spec;
            foreach (int p2 in skipTmp) skipEmit.Add(p2);
        }

        for (int ni = 0; ni < emitLimit; ni++)
        {
            NodeRecord node = nodes[ni];
            if (skipEmit.Contains(ni)) continue;
            if (attnAt.TryGetValue(ni, out var att))
            {
                LastEmitNi = ni;
                Emit(_pAttn, $"attn n{ni} T{att.T} H{att.H} D{att.D}",
                    [(arena, off[Phys(att.qkv)], 2), (arena, off[att.outT], 2)],
                    [(uint)att.T, (uint)att.H, (uint)att.D,
                     BitConverter.SingleToUInt32Bits(att.scale)],
                    (uint)((att.T + 63) / 64), (uint)(nb * att.H));
                continue;
            }
            if (attnSkip.Contains(ni)) continue;
            try
            {
            int skip = _compiled.FusedSkip(ni);
            ReadOnlySpan<byte> p = _model.GetParameters(node);
            int outPhys = Phys(checked((int)nodes[ni + skip].Outputs[0]));
            LastEmitNi = ni;
            long SlotOf(uint t) => off[Phys(checked((int)t))];

            switch (node.Operator)
            {
                case OperatorId.LayoutConvert:
                case OperatorId.Squeeze:
                case OperatorId.Unsqueeze:
                case OperatorId.Reshape:
                    break;

                case OperatorId.Conv:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] oshp = shapes[node.Outputs[0]];
                    int cin = ishp[1], cout = oshp[1];
                    int inH = ishp[2], inW = ishp[3], outH = oshp[2], outW = oshp[3];
                    int kH = I32(p, 8), kW = I32(p, 12), sH = I32(p, 16), sW = I32(p, 20);
                    int pt = I32(p, 32), pl = I32(p, 36);
                    int group = Math.Max(1, checked((int)U32(p, 4)));

                    int act = 0;
                    VkBuffer biasBuf = arena;
                    uint resT = uint.MaxValue, dualT = uint.MaxValue;
                    uint hasBias = 0, scalarBias = 0, flags16 = 0;
                    bool dualRaw = false;
                    if (node.Inputs.Length > 2 && node.Inputs[2] != uint.MaxValue)
                    { biasBuf = ConstF16(checked((int)node.Inputs[2])); hasBias = 1; }
                    if (skip == 1 && nodes[ni + 1].Operator == OperatorId.Relu)
                        act = 1;
                    else if (skip == 1 && nodes[ni + 1].Operator == OperatorId.Add)
                    {
                        // conv (no bias) + channel-bias add
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1]) : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
                    }
                    else if (skip == 2 && nodes[ni + 1].Operator == OperatorId.HardSigmoid
                             && nodes[ni + 2].Operator == OperatorId.Mul)
                    {
                        act = 3;
                        ReadOnlySpan<byte> hp = _model.GetParameters(nodes[ni + 1]);
                        float hsA = F32(hp, 4), hsB = F32(hp, 8);
                        if (hsB != 0.5f)
                            throw new NotSupportedException($"hardsigmoid beta={hsB} at node {ni + 1}");
                        flags16 |= (uint)Half2Bits(hsA) << 16;
                    }
                    else if (skip == 6
                             && nodes[ni + 1].Operator == OperatorId.Add
                             && nodes[ni + 2].Operator == OperatorId.Div
                             && nodes[ni + 3].Operator == OperatorId.Erf
                             && nodes[ni + 4].Operator == OperatorId.Add
                             && nodes[ni + 5].Operator == OperatorId.Mul
                             && nodes[ni + 6].Operator == OperatorId.Mul
                             && Phys(checked((int)nodes[ni + 2].Inputs[0]))
                                == Phys(checked((int)nodes[ni + 1].Outputs[0]))
                             && (Phys(checked((int)nodes[ni + 5].Inputs[0]))
                                 == Phys(checked((int)nodes[ni + 1].Outputs[0]))
                              || Phys(checked((int)nodes[ni + 5].Inputs[1]))
                                 == Phys(checked((int)nodes[ni + 1].Outputs[0]))))
                    {
                        // compiler-fused conv + bias-Add + gelu chain:
                        // out = 0.5*x*(1+erf(x/√2)) where x = conv + bias.
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1])
                            : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
