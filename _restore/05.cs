            if (mm1 < 0 || nodes[mm1].Operator != OperatorId.MatMul
                || nodes[mm1].Inputs[0] != (uint)qt) return null;
            // k: Transpose[0,1,3,2] -> MatMul1.B
            int kt = Sole(role[1]);
            if (kt < 0 || nodes[kt].Operator != OperatorId.Transpose || !Perm(kt, 0, 1, 3, 2)
                || nodes[mm1].Inputs[1] != nodes[kt].Outputs[0]) return null;
            int smx = Sole(checked((int)nodes[mm1].Outputs[0]));
            if (smx < 0 || nodes[smx].Operator != OperatorId.Softmax) return null;
            int ax = I32(_model.GetParameters(nodes[smx]), 4);
            if (ax != -1 && ax != 3) return null;
            int mm2 = Sole(checked((int)nodes[smx].Outputs[0]));
            if (mm2 < 0 || nodes[mm2].Operator != OperatorId.MatMul
                || nodes[mm2].Inputs[0] != nodes[smx].Outputs[0]
                || nodes[mm2].Inputs[1] != (uint)role[2] || Sole(role[2]) != mm2) return null;
            int ot = Sole(checked((int)nodes[mm2].Outputs[0]));
            if (ot < 0 || nodes[ot].Operator != OperatorId.Transpose || !Perm(ot, 0, 2, 1, 3))
                return null;
            mem.Add(kt); mem.Add(mm1); mem.Add(smx); mem.Add(mm2); mem.Add(ot);
            return (qkvT, checked((int)nodes[ot].Outputs[0]), T, H, D, scale);
        }

        // ---- pass 1: aliases + which tensors need arena slots ----
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            NodeRecord node = nodes[ni];
            int skip = _compiled.FusedSkip(ni);
            if (attnSkip.Contains(ni)) continue;
            // [n,1,a,b] -> [n,b,1,a] (perm 0,3,1,2): with a unit dim-1 the
            // row-major input flat order already equals the NHWC output's.
            if (node.Operator == OperatorId.Transpose
                && shapes[checked((int)node.Inputs[0])] is { Length: 4 } ts4 && ts4[1] == 1
                && _model.GetParameters(node) is ReadOnlySpan<byte> p4
                && U16(p4, 2) == 4 && I32(p4, 4) == 0 && I32(p4, 8) == 3
                && I32(p4, 12) == 1 && I32(p4, 16) == 2)
            {
                alias[checked((int)node.Outputs[0])] = checked((int)node.Inputs[0]);
                continue;
            }
            if (node.Operator is OperatorId.LayoutConvert or OperatorId.Squeeze
                or OperatorId.Unsqueeze or OperatorId.Reshape)
            {
                alias[checked((int)node.Outputs[0])] = checked((int)node.Inputs[0]);
                continue;
            }
            // rank-3 [n,c,w] -> [n,w,c] transpose is a no-op under the
            // channel-last physical layout: out flat (n*w'+c') index equals the
            // producer's (n*w+c) index because w'=w rows and c'=c channels swap
            // positions but keep the same flat offset.
            if (node.Operator == OperatorId.Transpose
                && shapes[checked((int)node.Inputs[0])].Length == 3
                && _model.GetParameters(node) is ReadOnlySpan<byte> p2
                && U16(p2, 2) == 3 && I32(p2, 4) == 0 && I32(p2, 8) == 2
                && I32(p2, 12) == 1)
            {
                alias[checked((int)node.Outputs[0])] = checked((int)node.Inputs[0]);
                continue;
            }
            foreach (uint inp in node.Inputs)
                if (inp != uint.MaxValue && !isConst[checked((int)inp)])
                    used[checked((int)inp)] = true;
            // the slot that gets written = the fused group's last node output
            int sink = checked((int)nodes[ni + skip].Outputs[0]);
            used[sink] = true;
            // fused members reading tensors produced BEFORE the group head
            // (residuals, dual-write pre-gelu taps) need real slots too —
            // in-group reads never touch the arena.
            for (int j = ni + 1; j <= ni + skip; j++)
                foreach (uint inp2 in nodes[j].Inputs)
                {
                    if (inp2 == uint.MaxValue) continue;
                    int ip = checked((int)inp2);
                    if (!isConst[ip] && prodNode[ip] >= 0 && prodNode[ip] < ni)
                        used[Phys(ip)] = true;
                }
            // conv+bias+gelu groups dual-write the pre-gelu tensor when it also
            // feeds a residual outside the group — reserve its slot
            if (skip == 6 && node.Operator == OperatorId.Conv
                && nodes[ni + 1].Operator == OperatorId.Add
                && nodes[ni + 5].Operator == OperatorId.Mul)
            {
                // preG has exactly 2 in-group consumers (Div ni+2, Mul ni+5);
                // anything beyond that is an external reader needing a slot
                int preG = checked((int)nodes[ni + 1].Outputs[0]);
                if (refCount[preG] > 2) used[preG] = true;
            }
            ni += skip;
        }
        foreach (uint g in _model.GraphOutputs) used[checked((int)g)] = true;
        foreach (var at in attnAt.Values) { used[at.outT] = true; used[Phys(at.qkv)] = true; }

        // graph input gets physically padded to 4 channels when Cin%4!=0 so the
        // input conv's im2col can use the vector path; weights pack zeros there.
        int inCinPad = shapes[inIdx][1] % 4 != 0 ? 4 : 0;

        var off = new long[nT];
        var slabElems = new long[nT];
        long cursor = 0;
        long maxIm2col = 0;   // shared scratch: max M*K_pad over emitted im2col convs
        for (int i = 0; i < nT; i++)
        {
            off[i] = -1;
            if (!used[i] || isConst[i] || alias[i] != i) continue;
            off[i] = cursor;
            // pad every slab to a 128-row tile multiple: coopmat reads pad rows;
            // slabs are 8-element (16 B) aligned: the sg32 coopmat stages
            // 16 B uvec4 loads straight from slab starts
            long sz = numel[i];
            if (i == inIdx && inCinPad != 0)
                sz = numel[i] / shapes[i][1] * inCinPad;
            slabElems[i] = (sz + 7) / 8 * 8 + 128 * 64;
            if (slabElems[i] * 2 > (long)_dev.MaxStorageRange)
                throw new NotSupportedException(
                    $"tensor {i} ({sz * 2} B) exceeds maxStorageBufferRange {_dev.MaxStorageRange}");
            cursor += slabElems[i];
        }
        // the im2col scratch sits past every slab; its size (and the arena
        // total) is settled after emit, so convs that take a direct or
        // implicit-GEMM path reserve nothing
        long im2colOff = (cursor + 63) / 64 * 64;
        // Role sentinels: the session binds its own grow-only buffers at
