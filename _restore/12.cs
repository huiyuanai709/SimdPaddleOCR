                                 (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                 (uint)inW, (uint)inH, flags],
                                Div256((long)outW * outH * cout));
                        }
                    }
                    else if (group == 1)
                    {
                        // dense kxk conv → explicit im2col (tap-major) + coopmat GEMM.
                        // graph-input channels are physically padded to 4 (cinIn).
                        int inT = Phys(checked((int)node.Inputs[0]));
                        int cinIn = (inT == inIdx && cin % 4 != 0) ? 4 : cin;
                        int K = cinIn * kH * kW, Kp = (K + 15) / 16 * 16;
                        uint M = (uint)(outH * outW);
                        // im2col-free direct conv: tap addressing inline in the
                        // dot kernel; skips materializing the K-expanded matrix.
                        // batched graphs always take it: the im2col fallback is batch-free
                        // sg32: implicit-GEMM coopmat conv instead (same tap
                        // addressing, tensor cores; any K, batched or not)
                        bool convkCm = _sg32 && cinIn % 8 == 0 && scalarBias == 0;
                        // the im2col matrix is one binding: past
                        // maxStorageBufferRange (Adreno: 128 MB) go direct
                        bool im2colFits = (long)M * Kp * 2 <= (long)_dev.MaxStorageRange;
                        // wide no-coopmat parts: direct at every K measured
                        // (Kp <= 2304 in these models), im2col+GEMM never won
                        if (convkCm || (cout % 4 == 0 && scalarBias == 0 && cinIn % 4 == 0
                            && (nb > 1 || !im2colFits || Kp <= (Environment.GetEnvironmentVariable(
                                "SIMD_OCR_CONVD_KMAX") is string km
                                ? int.Parse(km) : _ncWide ? int.MaxValue : 1024))
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NODCONV") == null))
                        {
                            // stem conv on the fp32 NCHW input: skip nchw2nhwc
                            // when this is the sole consumer of the graph input
                            bool f32In = !convkCm && inT == inIdx && cinIn == 4 && cin <= 4
                                && Environment.GetEnvironmentVariable(
                                    "SIMD_OCR_NOF32IN") == null;
                            if (f32In && (consumers[inIdx]?.Count ?? 0) == 1)
                                inF32Consumed = true;
                            // sole-consumer concat: write straight into its slice
                            long wO = off[outPhys]; uint dcv = 0, dco = 0;
                            {
                                List<int>? cs = consumers[outPhys];
                                if (!_noCatAbs && refCount[outPhys] == 1 && cs != null
                                    && cs.Count == 1
                                    && nodes[cs[0]].Operator == OperatorId.Concat)
                                {
                                    int cn = cs[0];
                                    int[] ospc = shapes[nodes[cn].Outputs[0]];
                                    int co = 0; bool hit = false;
                                    foreach (uint cin_ in nodes[cn].Inputs)
                                    {
                                        if (Phys(checked((int)cin_)) == outPhys)
                                        { hit = true; break; }
                                        co += shapes[checked((int)cin_)][1];
                                    }
                                    if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                                    {
                                        wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                        dcv = (uint)(ospc[1] / 4);
                                        dco = (uint)(co / 4);
                                        concatAbsorbed.Add(outPhys);
                                    }
                                }
                            }
                            if (convkCm)
                            {
                                var (kp, tm, tn) = CmTile(cout);
                                kp = kp == _pConv1x1N32 ? _pConvKN32!
                                   : kp == _pConv1x1N64 ? _pConvKN64! : _pConvK!;
                                uint Mt = M * (uint)nb;
                                Emit(kp, $"convkcm n{ni} {Mt}x{K}x{cout}",
                                    [(arena, SlotOf(node.Inputs[0]), 2),
                                     (ConstTapMajor(checked((int)node.Inputs[1]),
                                         cout, cin, kH, kW, Kp, cinPad: cinIn), 0, 2),
                                     (biasBuf, 0, 2),
                                     (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                     (arena, wO, 2),
                                     (arena, dualT != uint.MaxValue ? SlotOf(dualT) : 0, 2)],
                                    [Mt, (uint)cout, (uint)Kp, flags, (uint)outW, M,
                                     (uint)inW, (uint)inH, (uint)cinIn, (uint)kW,
                                     (uint)sH, (uint)sW, (uint)pt, (uint)pl, (uint)K,
                                     dcv * 4, dco * 4],
                                    (Mt + tm - 1) / tm, (uint)((cout + tn - 1) / tn));
                            }
                            else
                            Emit(f32In ? _pConvDF32 : _pConvD,
                                $"convd n{ni} {M}x{K}x{cout}",
                                [(f32In ? inF32 : arena,
                                  f32In ? 0 : SlotOf(node.Inputs[0]),
                                  f32In ? 4 : 2),
                                 (ConstKMajor(checked((int)node.Inputs[1]),
                                     cout, cin, kH * kW, Kp, cinIn), 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, wO, 2)],
                                f32In
                                ? [M, (uint)cout, (uint)outW, (uint)inW, (uint)inH,
                                   (uint)(cinIn / 4), (uint)(kH * kW), (uint)kW,
                                   (uint)sH, (uint)sW, (uint)pt, (uint)pl, flags,
                                   dcv, dco, (uint)cin, (uint)(inH * inW)]
                                : [M, (uint)cout, (uint)outW, (uint)inW, (uint)inH,
                                   (uint)(cinIn / 4), (uint)(kH * kW), (uint)kW,
                                   (uint)sH, (uint)sW, (uint)pt, (uint)pl, flags,
                                   dcv, dco],
                                Div256(((M + 3) / 4) * (long)(cout / 4)),
                                (uint)nb);
                        }
                        else
                        {
                        if (nb > 1)
                            throw new NotSupportedException(
                                $"im2col conv path lacks batch support (n{ni})");
                        if (!im2colFits)
                            throw new NotSupportedException(
                                $"im2col matrix {M}x{Kp} exceeds maxStorageBufferRange (n{ni})");
                        maxIm2col = Math.Max(maxIm2col, (long)M * Kp);
                        Emit(_pIm2col, $"im2col n{ni} M{M} K{K}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (arena, im2colOff, 2)],
                            [(uint)outW, (uint)outH, (uint)cinIn, (uint)kH, (uint)kW,
                             (uint)sH, (uint)sW, (uint)pt, (uint)pl,
