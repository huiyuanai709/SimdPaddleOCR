                    Console.Error.WriteLine($"  se? hs=n{hs} fc2=n{fc2} fc2pw={(fc2 >= 0 && IsPwConv(fc2))} rc2={(fc2 >= 0 ? refCount[Phys(checked((int)nodes[fc2].Outputs[0]))] : -1)}");
                if (fc2 < 0 || !IsPwConv(fc2)
                    || refCount[Phys(checked((int)nodes[fc2].Outputs[0]))] != 1) continue;
                int fc1 = prodNode[Phys(checked((int)nodes[fc2].Inputs[0]))];
                int reluNi = -1;
                if (fc1 >= 0 && nodes[fc1].Operator == OperatorId.Relu
                    && refCount[Phys(checked((int)nodes[fc1].Outputs[0]))] == 1)
                { reluNi = fc1; fc1 = prodNode[Phys(checked((int)nodes[fc1].Inputs[0]))]; }
                // fc1 may be a Conv+Relu compiler group (skip=1 covering reluNi)
                bool fc1ok = fc1 >= 0 && reluNi >= 0
                    && nodes[fc1].Operator == OperatorId.Conv && !inConvGroup[fc1]
                    && _compiled.FusedSkip(fc1) == 1 && fc1 + 1 == reluNi
                    && nodes[fc1].Inputs.Length > 2 && nodes[fc1].Inputs[2] != uint.MaxValue;
                if (fc1ok)
                {
                    ReadOnlySpan<byte> cp = _model.GetParameters(nodes[fc1]);
                    fc1ok = U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                        && I32(cp, 16) == 1 && I32(cp, 20) == 1;
                }
                if (dbg)
                    Console.Error.WriteLine($"    fc1=n{fc1} relu=n{reluNi} fc1ok={fc1ok}");
                if (!fc1ok
                    || refCount[Phys(checked((int)nodes[fc1].Outputs[0]))] != 1) continue;
                int rm = prodNode[Phys(checked((int)nodes[fc1].Inputs[0]))];
                if (dbg)
                    Console.Error.WriteLine($"    rm=n{rm} op={(rm >= 0 ? nodes[rm].Operator.ToString() : "?")} rcIn={(rm >= 0 ? refCount[Phys(checked((int)nodes[fc1].Inputs[0]))] : -1)}");
                if (rm < 0 || nodes[rm].Operator != OperatorId.ReduceMean
                    || _compiled.FusedSkip(rm) != 0 || inConvGroup[rm]
                    || refCount[Phys(checked((int)nodes[fc1].Inputs[0]))] != 1) continue;
                int[] xs = shapes[Phys(checked((int)nodes[rm].Inputs[0]))];
                int[] rs = shapes[Phys(checked((int)nodes[fc1].Outputs[0]))];
                int rDim = rs.Length >= 2 ? rs[1] : 0;
                if (dbg)
                    Console.Error.WriteLine($"    xs=[{string.Join('x', xs)}] rDim={rDim}");
                if (xs.Length != 4 || xs[1] % 4 != 0 || xs[1] > 256
                    || rDim <= 0 || rDim > 64) continue;
                seEmit[rm] = (fc1, fc2, hs);
                for (int m = fc1; m <= fc1 + _compiled.FusedSkip(fc1); m++)
                    skipEmit.Add(m);   // conv16 + its fused relu
                skipEmit.Add(fc2); skipEmit.Add(hs);
            }
        }
        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1")
        {
            Console.Error.WriteLine($"prescale convs={prescale.Count} addps={addScale.Count} seEmit={seEmit.Count} skipEmit={skipEmit.Count}");
            foreach (int rm in seEmit.Keys.OrderBy(k => k))
                Console.Error.WriteLine($"  se chain rm=n{rm} fc1=n{seEmit[rm].fc1} fc2=n{seEmit[rm].fc2} hs=n{seEmit[rm].hs}");
            foreach (var kv in addScale)
            {
                int px = prodNode[(int)kv.Value.x], ps = prodNode[(int)kv.Value.se], po = prodNode[(int)kv.Value.oth];
                Console.Error.WriteLine($"  addps n{kv.Key}: x=t{kv.Value.x}[{string.Join('x', shapes[(int)kv.Value.x])}] prod={(px < 0 ? "?" : nodes[px].Operator.ToString() + px)} " +
                    $"se=t{kv.Value.se}[{string.Join('x', shapes[(int)kv.Value.se])}] prod={(ps < 0 ? "?" : nodes[ps].Operator.ToString() + ps)} " +
                    $"oth=t{kv.Value.oth} prod={(po < 0 ? "?" : nodes[po].Operator.ToString() + po)}");
            }
            foreach (var kv in prescale)
                Console.Error.WriteLine($"  prescale conv n{kv.Key}: x=t{kv.Value.x} se=t{kv.Value.se}");
        }

        // ---- pass 2: emit ----
        int LastEmitNi = -1;
        List<GpuRec> recs = new();
        GpuRec Emit(VkPipeline pipe, string tag, (VkBuffer buf, long elemOff, int elemSize)[] binds,
                 uint[] pc, uint gx, uint gy = 1)
        {
            var bb = new GpuBind[binds.Length];
            for (int b = 0; b < binds.Length; b++)
                bb[b] = new GpuBind(binds[b].buf, (ulong)(binds[b].elemOff * binds[b].elemSize));
            GpuRec r = new() { Pipe = pipe, Binds = bb, Pc = Pcu(pc), Gx = gx, Gy = gy, Tag = tag };
            recs.Add(r);
            return r;
        }
        uint Div256(long n) => (uint)((n + 255) / 256);
        // depthwise kernel choice (emit and addps absorb must agree): lite
        // parts run k <= 3 on the flat kernel with the absorb read — its
        // channel-quad-consecutive loads beat the tile's per-quad halo
        // there; the rest goes to the shared-tile kernel, which sg32 also
        // uses for the 9x9 taps
        bool DwFlatA(int kh, int kw, int sh, int sw) =>
            _pDw4A is not null && kh <= 3 && kw <= 3 && sh <= 2 && sw <= 2;
        bool DwTiled(int kh, int kw, int sh, int sw) => !DwFlatA(kh, kw, sh, sw)
            && kh <= (_sg32 ? 9 : 5) && kw <= (_sg32 ? 9 : 5) && sh <= 2 && sw <= 2;
        // spatial-reduce partitions: ~512 px per workgroup (a 16-WG cap left
        // most of the GPU idle on large DET maps), bounded by partial storage
        int PartSplits(int hw, int c) =>
            Math.Max(1, Math.Min(Math.Min((hw + 511) / 512, 256), PartCtrElem / (c * nb)));
        // coopmat tile: cout<=32 -> 512x32, cout<=64 -> 256x64, else 128x128
        // sg32: cout<=32 -> 128x32, cout<=64 -> 128x64, else 128x128
        // m/k > 0: a plain GEMM that may take the direct-load kernel (lite)
        VkPipeline NcTail(VkPipeline p, uint f) =>
            _nocm && (f & 8u) == 0 && ((f >> 4) & 7u) is 0 or 1 or 3 ? _pGemmNcS! : p;
        (VkPipeline pipe, uint tm, uint tn) CmTile(int c, long m = 0, int k = 0) =>
            _nocm ? (_pConv1x1, 64u, 64u)
            : _pCmD is not null && m >= 16 && k > 0 && k % 16 == 0
                ? (c <= 32 ? (_pCmDN32!, 128u, 32u) : c <= 64 ? (_pCmDN64!, 128u, 64u) : (_pCmD, 128u, 128u))
            : _sg32 ? (c <= 32 ? (_pConv1x1N32, 128u, 32u)
                     : c <= 64 ? (_pConv1x1N64, 128u, 64u)
                     : (_pConv1x1, 128u, 128u))
            : c <= 32 ? (_pConv1x1N32, 512u, 32u)
            : c <= 64 ? (_pConv1x1N64, 256u, 64u)
            : (_pConv1x1, 128u, 128u);
        (VkBuffer, long, uint) Operand(int t, int chan, long n)
        {
            long nel = numel[t];
            uint mode = nel == n ? 0u : nel == chan ? 1u : 2u;
            return isConst[t] ? (ConstF16(t), 0, mode) : (arena, off[Phys(t)], mode);
        }

        // ---- addps absorb: the addps rec disappears; its sole consumer reads
        // a + x*se inline. A consumer qualifies only when its emit path is one
        // of the kernels with an absorb flag (dw4t / 1x1-dot / resize4 /
        // resize4add / concat4) AND the gates below exactly match emit — an
        // absorbed tensor read by a non-absorbing kernel would be garbage.
        var addpsSrc = new Dictionary<int, (uint a, uint x, uint se)>();
        if (Environment.GetEnvironmentVariable("SIMD_OCR_NOAPS") == null)
        foreach ((int addNi, (uint x, uint se, uint oth) asc) in addScale)
        {
            int outP = Phys(checked((int)nodes[addNi].Outputs[0]));
            if (refCount[outP] != 1) continue;
            List<int>? cs = consumers[outP];
            if (cs == null || cs.Count != 1 || Absorbable(cs[0], outP) == false)
