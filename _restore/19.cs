                [(uint)numel[outIdx], outAct], Div256(numel[outIdx]));

        // head: fp32 NCHW input → fp16 NHWC (channel-padded to inCinPad if set).
        // Skipped entirely when the stem conv read the fp32 input itself.
        if (!inF32Consumed)
        {
            int[] isp = shapes[inIdx];
            int hw = isp[2] * isp[3], c = isp[1];
            uint cout = inCinPad != 0 ? (uint)inCinPad : (uint)c;
            recs.Insert(0, new GpuRec
            {
                Pipe = _pNchw,
                Binds = [new GpuBind(inF32, 0), new GpuBind(arena, (ulong)off[inIdx] * 2)],
                Gx = Div256((long)hw * cout), Gy = (uint)nb,
                Pc = Pcu((uint)hw, (uint)c, cout), Tag = "nchw2nhwc",
            });
        }

        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1")
            for (int ri = 0; ri < recs.Count; ri++)
                Console.Error.WriteLine($"rec[{ri}] {recs[ri].Tag} gx={recs[ri].Gx} gy={recs[ri].Gy}");

        // device-independent, but the sg16 (Arc) kernel set has not been
        // verified against packed slabs yet. The no-coopmat tile only binds
        // slab starts, same as the sg32 GEMM, so packing is safe there too.
        if (_sg32 || _nocm) PackSlabs(recs, arena, off, slabElems, ref im2colOff);
        // +128*512: coopmat A-tile reads pad rows past M
        long arenaElems = im2colOff + maxIm2col + 128 * 512 + (1L << 20);
        return new GpuSchedule
        {
            Recs = recs.ToArray(),
            OutElems = (int)numel[outIdx], InElems = numel[inIdx], ArenaElems = arenaElems,
            Off = off, Alias = alias, Numel = numel, Shapes = shapes,
            Im2colOff = im2colOff,
        };
    }

    /// <summary>
    /// Re-packs the one-slab-per-tensor arena by lifetime: a slab lives from
    /// the first to the last rec that binds it, and slabs with disjoint
    /// lifetimes share memory (every rec is followed by a barrier, so a later
    /// writer never races an earlier reader). Rewrites the arena bindings,
    /// <paramref name="off"/> and the im2col scratch offset in place. Leaves
    /// everything untouched when a rec binds the arena anywhere but a slab
    /// start or the scratch. Afterwards only slabs still live at the end of
    /// the schedule keep their values for the debug readers.
    /// </summary>
    private static void PackSlabs(List<GpuRec> recs, VkBuffer arena, long[] off,
        long[] slabElems, ref long im2colOff)
    {
        var slabAt = new Dictionary<long, int>();
        for (int i = 0; i < off.Length; i++)
            if (off[i] >= 0) slabAt[off[i]] = i;
        var first = new int[off.Length];
        var last = new int[off.Length];
        Array.Fill(first, -1);
        for (int r = 0; r < recs.Count; r++)
            foreach (GpuBind b in recs[r].Binds)
            {
                if (!ReferenceEquals(b.Buf, arena)) continue;
                long e = (long)(b.ByteOffset / 2);
                if (e == im2colOff) continue;
                if ((b.ByteOffset & 1) != 0 || !slabAt.TryGetValue(e, out int i)) return;
                if (first[i] < 0) first[i] = r;
                last[i] = r;
            }

        var order = new List<int>();
        for (int i = 0; i < off.Length; i++)
            if (first[i] >= 0) order.Add(i);
        order.Sort((a, b) => first[a] != first[b] ? first[a].CompareTo(first[b]) : a.CompareTo(b));
        var newOff = new long[off.Length];
        Array.Fill(newOff, -1L);
        var free = new List<(long Off, long Len)>();   // sorted by offset, coalesced
        var live = new List<int>();
        long top = 0;
        foreach (int i in order)
        {
            for (int k = live.Count - 1; k >= 0; k--)
            {
                int j = live[k];
                if (last[j] >= first[i]) continue;
                live.RemoveAt(k);
                Release(free, newOff[j], slabElems[j]);
            }
            long need = slabElems[i];
            int best = -1;
            for (int k = 0; k < free.Count; k++)
                if (free[k].Len >= need && (best < 0 || free[k].Len < free[best].Len)) best = k;
            if (best >= 0)
            {
                newOff[i] = free[best].Off;
                if (free[best].Len == need) free.RemoveAt(best);
                else free[best] = (free[best].Off + need, free[best].Len - need);
            }
            else if (free.Count > 0 && free[^1].Off + free[^1].Len == top)
            {
                newOff[i] = free[^1].Off;
                top = newOff[i] + need;
                free.RemoveAt(free.Count - 1);
            }
            else
            {
                newOff[i] = top;
                top += need;
            }
            live.Add(i);
        }

        long newIm2col = (top + 63) / 64 * 64;
        foreach (GpuRec rec in recs)
        {
            GpuBind[] bb = (GpuBind[])rec.Binds.Clone();
            for (int k = 0; k < bb.Length; k++)
            {
                if (!ReferenceEquals(bb[k].Buf, arena)) continue;
                long e = (long)(bb[k].ByteOffset / 2);
                bb[k] = new GpuBind(arena, (ulong)(e == im2colOff ? newIm2col : newOff[slabAt[e]]) * 2);
            }
            rec.Binds = bb;
