                continue;
            addpsSrc[outP] = (asc.oth, asc.x, asc.se);
        }

        bool Absorbable(int cn, int srcPhys)
        {
            NodeRecord c = nodes[cn];
            switch (c.Operator)
            {
                case OperatorId.Conv:
                {
                    if (Phys(checked((int)c.Inputs[0])) != srcPhys) return false;
                    ReadOnlySpan<byte> cp = _model.GetParameters(c);
                    int[] ish2 = shapes[checked((int)c.Inputs[0])];
                    int[] osh2 = shapes[checked((int)c.Outputs[0])];
                    int cin2 = ish2[1], cout2 = osh2[1];
                    int kH2 = I32(cp, 8), kW2 = I32(cp, 12),
                        sH2 = I32(cp, 16), sW2 = I32(cp, 20);
                    int grp2 = Math.Max(1, checked((int)U32(cp, 4)));
                    if (grp2 == cin2 && cout2 == cin2)
                        return cin2 % 4 == 0 && (DwTiled(kH2, kW2, sH2, sW2)  // conv_dw4t
                            || DwFlatA(kH2, kW2, sH2, sW2));                   // conv_dw4a
                    if (grp2 == 1 && kH2 == 1 && kW2 == 1 && sH2 == 1 && sW2 == 1)
                    {
                        int cinI2 = (srcPhys == inIdx && cin2 % 4 != 0) ? 4 : cin2;
                        if (cinI2 % 4 != 0 || cout2 % 4 != 0 || cinI2 > 128)
                            return false;
                        int cskip = _compiled.FusedSkip(cn);
                        if (cskip == 1 && nodes[cn + 1].Operator == OperatorId.Add)
                        {
                            NodeRecord ad2 = nodes[cn + 1];
                            int bt = ad2.Inputs[0] == c.Outputs[0]
                                ? checked((int)ad2.Inputs[1])
                                : checked((int)ad2.Inputs[0]);
                            if (numel[bt] == 1) return false;   // scalarBias
                        }
                        return !prescale.ContainsKey(cn)
                            && Environment.GetEnvironmentVariable(
                                "SIMD_OCR_NODOT") == null;
                    }
                    return false;
                }
                case OperatorId.Resize:
                    return Phys(checked((int)c.Inputs[0])) == srcPhys
                        && shapes[checked((int)c.Outputs[0])][1] % 4 == 0;
                case OperatorId.Concat:
                {
                    int[] ospc = shapes[checked((int)c.Outputs[0])];
                    if (ospc[1] % 4 != 0) return false;
                    int cOff2 = 0;
                    foreach (uint inp2 in c.Inputs)
                    {
                        int ci2 = shapes[checked((int)inp2)][1];
                        if (Phys(checked((int)inp2)) == srcPhys)
                            return ci2 % 4 == 0 && cOff2 % 4 == 0;
                        cOff2 += ci2;
                    }
                    return false;
                }
                case OperatorId.Add:
                {
                    // consumer Add is the tail of a resize4add fusion: the
                    // other input must come from a Resize that will itself
                    // hit the fused path (sole-consumer, %4, same numel).
                    if (addScale.ContainsKey(cn)) return false;
                    int i0 = Phys(checked((int)c.Inputs[0]));
                    int i1 = Phys(checked((int)c.Inputs[1]));
                    int othP2 = i0 == srcPhys ? i1 : i1 == srcPhys ? i0 : -1;
                    if (othP2 < 0) return false;
                    int prod = prodNode[othP2];
                    if (prod < 0 || nodes[prod].Operator != OperatorId.Resize)
                        return false;
                    int[] osp3 = shapes[checked((int)nodes[prod].Outputs[0])];
                    if (osp3[1] % 4 != 0) return false;
                    if (refCount[othP2] != 1 || consumers[othP2]?.Count != 1
                        || consumers[othP2][0] != cn) return false;
                    return numel[srcPhys] == numel[othP2];
                }
                default: return false;
            }
        }

        // ---- catresize fusion: a Concat whose inputs are sole-consumed
        // Resize nodes (or plain tensors) collapses to ONE dispatch that
        // resamples each source straight into its output slice. The resizes'
        // own recs are skipped (they were the writers, now the kernel writes).
        var catRes = new Dictionary<int,
            (long x, long ra, long se, uint fl, uint inW, uint fh, uint fw,
             uint off4, uint cq4)[]>();
        // catresize kernel is batch-free: batched graphs keep the plain path
        for (int ci2 = 0; nb == 1 && ci2 < nodes.Length; ci2++)
        {
            NodeRecord c = nodes[ci2];
            if (c.Operator != OperatorId.Concat || c.Inputs.Length > 4)
                continue;
            int[] osp = shapes[checked((int)c.Outputs[0])];
            if (osp[1] % 4 != 0) continue;
            var spec = new (long, long, long, uint, uint, uint, uint, uint, uint)
                [c.Inputs.Length];
            int cOff3 = 0; bool okc = true; var skipTmp = new List<int>();
            for (int ii = 0; ii < c.Inputs.Length; ii++)
            {
                uint inp2 = c.Inputs[ii];
                int ip = Phys(checked((int)inp2));
                int ci3 = shapes[checked((int)inp2)][1];
                if (ci3 % 4 != 0 || cOff3 % 4 != 0) { okc = false; break; }
                uint fh3 = 1, fw3 = 1, inW3 = (uint)osp[3];
                int srcP = ip;
                int prod = prodNode[ip];
                if (prod >= 0 && nodes[prod].Operator == OperatorId.Resize
                    && refCount[ip] == 1 && consumers[ip]?.Count == 1
                    && consumers[ip][0] == ci2)
                {
                    int[] rish = shapes[checked((int)nodes[prod].Inputs[0])];
                    if (rish[1] != ci3) { okc = false; break; }
                    fh3 = (uint)(osp[2] / rish[2]);
                    fw3 = (uint)(osp[3] / rish[3]);
                    if (rish[2] * (int)fh3 != osp[2] || rish[3] * (int)fw3 != osp[3])
                    { okc = false; break; }
                    inW3 = (uint)rish[3];
