            : checked((int)_model.GraphOutputs[0]);
        int inIdx = checked((int)_model.GraphInputs[0]);
        int emitLimit = Math.Min(nodeLimit, nodes.Length);
        int nb = inputShape[0];   // batch — spatial kernels take it via gy

        var alias = new int[nT];          // metadata-only ops alias producer slot
        var isConst = new bool[nT];
        var numel = new long[nT];
        var used = new bool[nT];
        for (int i = 0; i < nT; i++)
        {
            alias[i] = i;
            numel[i] = 1;
            foreach (int d in shapes[i]) numel[i] *= Math.Max(d, 1);
            isConst[i] = _compiled.GetTensor(i).IsConstant;
        }
        int Phys(int t) { while (alias[t] != t) t = alias[t]; return t; }

        // refcount / producer / consumers for emit-time fusion decisions
        var refCount = new int[nT];
        var prodNode = new int[nT];
        Array.Fill(prodNode, -1);
        var consumers = new List<int>[nT];
        // members of conv-headed fused groups are folded into the conv dispatch;
        // groups headed by other ops emit every member standalone, so a Mul
        // inside e.g. a HardSigmoid+Mul (hardswish) group is still foldable.
        var inConvGroup = new bool[nodes.Length];
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            int skip = _compiled.FusedSkip(ni);
            if (nodes[ni].Operator is OperatorId.Conv or OperatorId.ConvTranspose)
                for (int j = ni + 1; j <= ni + skip && j < nodes.Length; j++)
                    inConvGroup[j] = true;
            foreach (uint inp in nodes[ni].Inputs)
            {
                if (inp == uint.MaxValue) continue;
                refCount[(int)inp]++;
                (consumers[(int)inp] ??= new List<int>()).Add(ni);
            }
            prodNode[checked((int)nodes[ni].Outputs[0])] = ni;
        }

        // ---- fused MHSA (SVTR mixer): the Transpose5D -> Slice/Squeeze ->
        // q*scale, k^T -> MatMul -> Softmax -> MatMul -> Transpose chain
        // collapses into one `attn` dispatch reading the packed qkv tensor.
        // Members get no arena slots; the chain's out-Transpose slot is the
        // kernel's destination (the trailing Reshape aliases onto it).
        var attnAt = new Dictionary<int, (int qkv, int outT, int T, int H, int D, float scale)>();
        var attnSkip = new HashSet<int>();
        if (Environment.GetEnvironmentVariable("SIMD_OCR_NOATTN") == null)
        {
            var mem = new List<int>();
            for (int ni = 0; ni < nodes.Length; ni++)
                if (nodes[ni].Operator == OperatorId.Transpose
                    && MatchAttention(ni, mem) is { } spec)
                {
                    attnAt[ni] = spec;
                    foreach (int m in mem) attnSkip.Add(m);
                }
        }

        (int qkv, int outT, int T, int H, int D, float scale)? MatchAttention(int tr, List<int> mem)
        {
            mem.Clear();
            int Sole(int t) => refCount[t] == 1 && consumers[t] is { Count: 1 } c ? c[0] : -1;
            bool Perm(int ni, params int[] perm)
            {
                ReadOnlySpan<byte> tp = _model.GetParameters(nodes[ni]);
                if (U16(tp, 2) != perm.Length) return false;
                for (int i = 0; i < perm.Length; i++)
                    if (I32(tp, 4 + 4 * i) != perm[i]) return false;
                return true;
            }
            NodeRecord t5 = nodes[tr];
            int in5 = checked((int)t5.Inputs[0]);
            int[] s5 = shapes[in5];
            if (s5.Length != 5 || s5[0] != nb || s5[2] != 3 || s5[4] > 32
                || !Perm(tr, 2, 0, 3, 1, 4)) return null;
            int r5 = prodNode[in5];
            if (r5 < 0 || nodes[r5].Operator != OperatorId.Reshape) return null;
            int qkvT = checked((int)nodes[r5].Inputs[0]);
            int T = s5[1], H = s5[3], D = s5[4];
            if (shapes[qkvT].Length != 3 || numel[qkvT] != numel[in5]) return null;
            int tOut = checked((int)t5.Outputs[0]);
            List<int>? cs = consumers[tOut];
            if (cs is not { Count: 3 }) return null;
            int[] role = [-1, -1, -1];
            mem.Add(tr);
            foreach (int sl in cs)
            {
                if (nodes[sl].Operator != OperatorId.Slice) return null;
                (int[] st, int[] stp) = _compiled.ResolveSliceBounds(shapes[tOut], nodes[sl], out _);
                int[] so = shapes[checked((int)nodes[sl].Outputs[0])];
                if (st.Length != 5 || so.Length != 5 || so[0] != 1 || (uint)st[0] > 2
                    || role[st[0]] != -1) return null;
                for (int a = 0; a < 5; a++)
                    if (stp[a] != 1 || (a > 0 && (st[a] != 0 || so[a] != shapes[tOut][a])))
                        return null;
                int sq = Sole(checked((int)nodes[sl].Outputs[0]));
                if (sq < 0 || nodes[sq].Operator != OperatorId.Squeeze) return null;
                role[st[0]] = checked((int)nodes[sq].Outputs[0]);
                mem.Add(sl); mem.Add(sq);
            }
            // q: Mul(scalar const) [-> Reshape] -> MatMul1.A
            int mul = Sole(role[0]);
            if (mul < 0 || nodes[mul].Operator != OperatorId.Mul) return null;
            int sc = nodes[mul].Inputs[0] == (uint)role[0]
                ? checked((int)nodes[mul].Inputs[1]) : checked((int)nodes[mul].Inputs[0]);
            if (!isConst[sc] || numel[sc] != 1) return null;
            float scale = CstF32(sc)[0];
            mem.Add(mul);
            int qt = checked((int)nodes[mul].Outputs[0]);
            int mm1 = Sole(qt);
            if (mm1 >= 0 && nodes[mm1].Operator == OperatorId.Reshape
                && numel[checked((int)nodes[mm1].Outputs[0])] == numel[qt])
            {
                mem.Add(mm1);
                qt = checked((int)nodes[mm1].Outputs[0]);
                mm1 = Sole(qt);
            }
