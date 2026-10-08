    // wave64): direct-load GEMM plus the routing measured on Adreno 750
    private readonly bool _ncWide;

    // fp16 copies of constant tensors (weights, biases, scalars), lazy
    private readonly Dictionary<(int, int, int, int), VkBuffer> _constF16 = new();
    private readonly Dictionary<int, VkBuffer> _constF32 = new();
    private readonly Dictionary<int, VkBuffer> _constTap = new();

    // conv weight repack to tap-major [Cout, (dy*kW+dx)*Cin+ci], rows padded to Kp
    private VkBuffer ConstTapMajor(int tensorIndex, int cout, int cin,
        int kH, int kW, int kp, int cinPad = 0)
    {
        if (cinPad == 0) cinPad = cin;
        if (_constTap.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int kSpan = kH * kW;
        int rowsPad = (cout + 127) / 128 * 128;
        Half[] h = new Half[rowsPad * kp];
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cinPad; ci++)
                for (int tap = 0; tap < kSpan; tap++)
                    if (ci < cin)
                        h[co * kp + tap * cinPad + ci] =
                            (Half)f32[co * cin * kSpan + ci * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constTap[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // depthwise weight repack to tap-major [tap*C + c] fp16 (vec4-readable)
    private readonly Dictionary<int, VkBuffer> _constDw = new();
    private VkBuffer ConstDwTap(int tensorIndex, int c, int kSpan)
    {
        if (_constDw.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[c * kSpan];
        for (int cc = 0; cc < c; cc++)
            for (int tap = 0; tap < kSpan; tap++)
                h[tap * c + cc] = (Half)f32[cc * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constDw[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // k-major weight repack for conv1x1_dot: [k*Cout + n], k = tap*cinPad + ci
    private readonly Dictionary<int, VkBuffer> _constKm = new();
    private VkBuffer ConstKMajor(int tensorIndex, int cout, int cin,
        int kSpan, int kp, int cinPad)
    {
        if (_constKm.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[kp * cout];
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cinPad; ci++)
                for (int tap = 0; tap < kSpan; tap++)
                    if (ci < cin)
                        h[(tap * cinPad + ci) * cout + co] =
                            (Half)f32[co * cin * kSpan + ci * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constKm[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // ConvTranspose weight repack [tap*Cin + ci, co] (tap=dy*2+dx) fp16
    private readonly Dictionary<int, VkBuffer> _constCt = new();
    private VkBuffer ConstConvT(int tensorIndex, int cin, int cout)
    {
        if (_constCt.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[4 * cin * cout];
        for (int tap = 0; tap < 4; tap++)
            for (int ci = 0; ci < cin; ci++)
                for (int co = 0; co < cout; co++)
                    h[(tap * cin + ci) * cout + co] =
                        (Half)f32[(ci * cout + co) * 4 + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constCt[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // reduce_hw4 phase-1 partials (fp32) + se_fused tickets: a fixed
    // session-owned scratch (S partitions × C channels × batch + ctr slots)
    // partials: fp32 [0, PartCtrElem); ticket counters from PartCtrElem on
    internal const int PartCtrElem = 1 << 18;
    internal const int PartBytes = PartCtrElem * 4 + 64 * 1024;
    private static VkBuffer PartBuf(int floats) => RolePart;
    private readonly List<VkBuffer> _allBufs = new();

    private GpuGraphModel(VkDevice dev, CompiledModel compiled)
    {
        _dev = dev;
        _compiled = compiled;
        _model = compiled.Model;
        _key = _model.ContentKey;
        _nocm = !dev.CoopGemm;
        // sg16-only coopmat shaders: NVIDIA (sg 32-32) and AMD wave64 cannot
        // satisfy requiredSubgroupSize=16 — swap in the sg32 variant.
        _sg32 = !_nocm && dev.Sg32Subgroup;
        if (_nocm)
        {
            // One tile; n64/n32 stay aliases so CmTile's sg16 cout split is
            // not reused here. 8 lanes keep its 64 fp32 accumulators in
            // registers (UHD 770: SIMD16 spills, 4x slower). Without an
            // 8-lane pin (wave64 Adreno) the register-prefetch form is
            // miscompiled there; the direct-load build is exact and faster.
            bool sg8 = dev.ComputeSubgroupSize && dev.SubgroupMin <= 8 && dev.SubgroupMax >= 8;
            _ncWide = !sg8;
            _pConv1x1 = sg8 ? Pipe("gemm_nc", 6, 16, 8u) : Pipe("gemm_nc_d", 6, 16);
