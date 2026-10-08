                            && (hasPs || !addpsSrc.ContainsKey(Phys(checked((int)node.Inputs[0])))))
                            dotOk = false;
                        // no-coopmat GEMM likewise; it has no SE-prescale variant
                        if (dotOk && _nocm && !hasPs && (long)M * cout > 1L << 16
                            && !addpsSrc.ContainsKey(Phys(checked((int)node.Inputs[0]))))
                            dotOk = false;
                        if (dotOk)
                        {
                            // small-K pointwise conv: direct dot kernel beats coopmat.
                            // prescale: read x (not the mul output) + per-channel se scale.
                            long aOff = hasPs ? SlotOf(psv.x) : SlotOf(node.Inputs[0]);
                            bool aps = addpsSrc.TryGetValue(
                                Phys(checked((int)node.Inputs[0])), out var ads)
                                && !hasPs;
                            if (aps) aOff = SlotOf(ads.x);
                            VkBuffer wk = ConstKMajor(checked((int)node.Inputs[1]),
                                cout, cin, 1, cinIn, cinIn);
                            // se tensors are per-batch [n, C] — mImg lets the
                            // kernel pick the batch's scale vector; 0 = batch-free.
                            uint mIm = hasPs || aps ? mImg : 0u;
                            uint dFlags = flags | (aps ? 512u : 0u);
                            long dTiles = ((M + 3) / 4) * (long)(cout / 4);
                            bool dSplit = cinIn >= 64 && dTiles <= 16384 && !s_noSplitK
                                && (dFlags & (8u | 512u | 1024u)) == 0;
                            Emit(dSplit ? _pDotSk : _pDot, $"conv1x1d{(dSplit ? "sk" : "")} n{ni} {M}x{cinIn}x{cout}",
                                [(arena, aOff, 2), (wk, 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, off[outPhys], 2),
                                 (arena, hasPs ? SlotOf(psv.se)
                                              : aps ? SlotOf(ads.se) : 0, 2),
                                 (arena, dualT != uint.MaxValue ? SlotOf(dualT) : 0, 2),
                                 (arena, aps ? SlotOf(ads.a) : 0, 2)],
                                [M, (uint)cout, (uint)cinIn, (uint)(cinIn / 4),
                                 dFlags, (uint)(cinIn / 4), mIm],
                                dSplit ? (uint)((dTiles + 15) / 16) : Div256(dTiles));
                        }
                        else
                        {
                            var (cp, tm, tn) = CmTile(cout, hasPs ? 0 : M, cinIn);
                            (VkBuffer, long, int)[] cb =
                                [(arena, hasPs ? SlotOf(psv.x) : SlotOf(node.Inputs[0]), 2), (wbuf, 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, off[outPhys], 2),
                                 (arena, dualT != uint.MaxValue ? SlotOf(dualT) : 0, 2)];
                            uint[] cpc = [M, (uint)cout, (uint)cinIn, flags];
                            if (hasPs && _pConvPs is not null)
                            {
                                cp = cp == _pConv1x1N32 ? _pConvPsN32!
                                   : cp == _pConv1x1N64 ? _pConvPsN64! : _pConvPs;
                                cb = [.. cb, (arena, SlotOf(psv.se), 2)];
                                cpc = [.. cpc, mImg];
                            }
                            Emit(NcTail(cp, flags), $"conv1x1 n{ni} {M}x{cin}x{cout}", cb, cpc,
                                (M + tm - 1) / tm, (uint)((cout + tn - 1) / tn));
                        }
                    }
                    else if (group == cin && cout == cin)
                    {
                        if (cin % 4 == 0)
                        {
                            if (DwTiled(kH, kW, sH, sW))
                            {
                                // shared-tile variant: ~kH*kW less global traffic
                                uint tx = (uint)((outW + 15) / 16),
                                     ty = (uint)((outH + 15) / 16);
                                bool aps = addpsSrc.TryGetValue(
                                    Phys(checked((int)node.Inputs[0])), out var ads);
                                Emit(_pDw4T, $"conv_dw4t n{ni} {cout}ch k{kH}s{sH}",
                                    [(arena, aps ? SlotOf(ads.x)
                                                : SlotOf(node.Inputs[0]), 2),
                                     (ConstDwTap(checked((int)node.Inputs[1]), cout, kH * kW), 0, 2),
                                     (biasBuf, 0, 2), (arena, off[outPhys], 2),
                                     (arena, aps ? SlotOf(ads.a) : 0, 2),
                                     (arena, aps ? SlotOf(ads.se) : 0, 2)],
                                    [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                     (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                     (uint)inW, (uint)inH,
                                     flags | (aps ? 256u : 0u)],
                                    tx * ty * (uint)(cout / 4), (uint)nb);
                            }
                            else if (DwFlatA(kH, kW, sH, sW))
                            {
                                bool aps = addpsSrc.TryGetValue(
                                    Phys(checked((int)node.Inputs[0])), out var ads);
                                Emit(_pDw4A!, $"conv_dw4a n{ni} {cout}ch k{kH}s{sH}",
                                    [(arena, aps ? SlotOf(ads.x)
                                                : SlotOf(node.Inputs[0]), 2),
                                     (ConstDwTap(checked((int)node.Inputs[1]), cout, kH * kW), 0, 2),
                                     (biasBuf, 0, 2), (arena, off[outPhys], 2),
                                     (arena, aps ? SlotOf(ads.a) : 0, 2),
                                     (arena, aps ? SlotOf(ads.se) : 0, 2)],
                                    [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                     (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                     (uint)inW, (uint)inH,
                                     flags | (aps ? 256u : 0u)],
                                    Div256((long)outW * outH * (cout / 4)), (uint)nb);
                            }
                            else
                            Emit(_pDw4, $"conv_dw4 n{ni} {cout}ch k{kH}s{sH}",
                                [(arena, SlotOf(node.Inputs[0]), 2),
                                 (ConstDwTap(checked((int)node.Inputs[1]), cout, kH * kW), 0, 2),
                                 (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                                [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                 (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                 (uint)inW, (uint)inH, flags],
                                Div256((long)outW * outH * (cout / 4)),
                                (uint)nb);
                        }
                        else
                        {
                            if (nb > 1)
                                throw new NotSupportedException(
                                    $"scalar dw conv lacks batch support (n{ni})");
                            Emit(_pDw, $"conv_dw n{ni} {cout}ch k{kH}s{sH}",
                                [(arena, SlotOf(node.Inputs[0]), 2),
                                 (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                                 (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                                [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
