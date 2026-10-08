        // record time, sized from the schedule's requirements.
        VkBuffer arena = RoleArena, outF32 = RoleOut, inF32 = RoleIn;
        bool inF32Consumed = false;  // stem conv reads fp32 NCHW directly
        bool convTOut = false;   // last convT wrote fp32 out directly


        // ---- pre-emit fusion detection ----
        // Mul(x, seChannelVec) (the SE gate) gets skipped when EVERY consumer can
        // fold the gating: a pointwise Conv applies it as input-prescale, an Add
        // applies it as scaled-residual-add (addps). Any unhandled consumer keeps
        // the Mul emitted.
        var prescale = new Dictionary<int, (uint x, uint se)>();   // convNi -> tensors
        var addScale = new Dictionary<int, (uint x, uint se, uint oth)>();  // addNi -> tensors
        var skipEmit = new HashSet<int>();
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            NodeRecord nd = nodes[ni];
            if (nd.Operator != OperatorId.Mul || _compiled.FusedSkip(ni) != 0 || inConvGroup[ni])
                continue;
            uint a = nd.Inputs[0], b = nd.Inputs[1];
            if (a == uint.MaxValue || b == uint.MaxValue
                || isConst[(int)a] || isConst[(int)b]) continue;
            // se = input whose numel equals the other's channel count
            // (per-image: nb*C for a batch — se tensors are [n,C,1,1])
            long na = numel[(int)a], nelb = numel[(int)b];
            int[] sa = shapes[(int)a], sb = shapes[(int)b];
            uint xT, seT;
            if (na == (sa.Length >= 2 ? (long)sb[1] * nb : -1) && nelb > na) { xT = b; seT = a; }
            else if (nelb == (sb.Length >= 2 ? (long)sa[1] * nb : -1) && na > nelb) { xT = a; seT = b; }
            else continue;
            if (numel[(int)seT] % 4 != 0) continue;

            var pendPrescale = new List<int>();
            var pendAdd = new List<(int c, uint oth)>();
            bool ok = true;
            var stk = new Stack<uint>();
            var seen = new HashSet<uint>();
            stk.Push(nd.Outputs[0]);
            while (stk.Count > 0 && ok)
            {
                uint t = stk.Pop();
                if (!seen.Add(t)) continue;
                List<int>? cs = consumers[(int)t];
                if (cs == null) { ok = false; break; }
                foreach (int c in cs)
                {
                    NodeRecord cn = nodes[c];
                    if (cn.Operator is OperatorId.LayoutConvert or OperatorId.Squeeze
                        or OperatorId.Unsqueeze or OperatorId.Reshape)
                    { stk.Push(cn.Outputs[0]); continue; }
                    if (cn.Operator == OperatorId.Conv && !inConvGroup[c]
                        && Phys((int)cn.Inputs[0]) == Phys((int)nd.Outputs[0]))
                    {
                        ReadOnlySpan<byte> cp = _model.GetParameters(cn);
                        int ccin = shapes[(int)xT][1];
                        int ccout = shapes[(int)cn.Outputs[0]][1];
                        if (U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                            && I32(cp, 16) == 1 && I32(cp, 20) == 1
                            && ccin % 4 == 0 && ccout % 4 == 0 && ccin <= 256
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NOPRESCALE") == null)
                            pendPrescale.Add(c);
                        else ok = false;
                        continue;
                    }
                    if (cn.Operator == OperatorId.Add && !inConvGroup[c]
                        && _compiled.FusedSkip(c) == 0
                        && Environment.GetEnvironmentVariable("SIMD_OCR_NOADDPS") == null)
                    {
                        uint oth = cn.Inputs[0] == t ? cn.Inputs[1]
                                 : cn.Inputs[1] == t ? cn.Inputs[0] : uint.MaxValue;
                        if (oth != uint.MaxValue && !isConst[(int)oth]
                            && numel[(int)oth] == numel[(int)xT]
                            && numel[(int)xT] % 4 == 0)
                        { pendAdd.Add((c, oth)); continue; }
                        ok = false;
                        continue;
                    }
                    ok = false;
                }
            }
            if (!ok)
            {
                foreach (int c in pendPrescale) prescale.Remove(c);
                continue;
            }
            foreach (int c in pendPrescale) prescale[c] = (xT, seT);
            foreach ((int c, uint oth) in pendAdd) addScale[c] = (xT, seT, oth);
            if (Environment.GetEnvironmentVariable("SIMD_OCR_PS_KEEP") != null)
                continue;   // debug: folds registered but mul still emitted
            skipEmit.Add(ni);
        }

        // SE chain fusion: ReduceMean -> Conv1x1(fc1,+bias) -> Conv1x1(fc2,+bias)
        // -> HardSigmoid emits as a single `se` dispatch writing the gate vector
        // directly to the HardSigmoid output slot.
        var seEmit = new Dictionary<int, (int fc1, int fc2, int hs)>();
        var concatAbsorbed = new HashSet<int>();   // phys tensors written directly into a concat slice
        // Fused-SE ticket index, in uints, within the part-buffer tail that
        // starts at PartCtrElem. Each block's descriptor offset must be a
        // multiple of minStorageBufferOffsetAlignment (16 on lavapipe and
        // NVIDIA, 256 on some devices). A 4-uint stride faults those GPUs.
        int seCtr = 0;
        if (Environment.GetEnvironmentVariable("SIMD_OCR_NOSE") == null)
        {
            bool IsPwConv(int ni)
            {
                if (nodes[ni].Operator != OperatorId.Conv || _compiled.FusedSkip(ni) != 0
                    || inConvGroup[ni]) return false;
                ReadOnlySpan<byte> cp = _model.GetParameters(nodes[ni]);
                return U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                    && I32(cp, 16) == 1 && I32(cp, 20) == 1
                    && nodes[ni].Inputs.Length > 2 && nodes[ni].Inputs[2] != uint.MaxValue;
            }
            bool dbg = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1";
            for (int hs = 0; hs < nodes.Length; hs++)
            {
                if (nodes[hs].Operator != OperatorId.HardSigmoid
                    || _compiled.FusedSkip(hs) != 0 || inConvGroup[hs]) continue;
                int fc2 = prodNode[Phys(checked((int)nodes[hs].Inputs[0]))];
                if (dbg)
