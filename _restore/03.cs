            h = new Half[alloc];
            for (int i = 0; i < n; i++) h[i] = (Half)f32[i];
        }
        VkBuffer buf = _dev.NewStorageBuffer((ulong)alloc * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)alloc * 2); }
        _constF16[key] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // fp32 device copy of a constant tensor (small tensors needing exact math).
    private VkBuffer ConstF32(int tensorIndex)
    {
        if (_constF32.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        VkBuffer buf = _dev.NewStorageBuffer((ulong)f32.Length * 4, hostVisible: false);
        unsafe { fixed (float* p = f32) _dev.Upload(buf, p, (ulong)f32.Length * 4); }
        _constF32[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    private ReadOnlySpan<float> CstF32(int tensorIndex) =>
        MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);

    // fp16 device copy of a computed vector (e.g. folded BN affine params)
    // keyed by (node, slot) so re-compiling another shape reuses the upload
    private readonly Dictionary<(int, int), VkBuffer> _vecF16 = new();
    private VkBuffer VecF16((int, int) key, float[] v)
    {
        if (_vecF16.TryGetValue(key, out VkBuffer? cached)) return cached;
        Half[] h = new Half[(v.Length + 7) / 8 * 8];
        for (int i = 0; i < v.Length; i++) h[i] = (Half)v[i];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _vecF16[key] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // MatMul weight repack [K,N] row-major -> [N,K] fp16 (coopmat B layout),
    // N padded to a 128-row tile.
    private readonly Dictionary<int, VkBuffer> _constGw = new();
    private VkBuffer ConstGemmW(int tensorIndex, int K, int N)
    {
        if (_constGw.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int nPad = (N + 127) / 128 * 128;
        Half[] h = new Half[nPad * K];
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h[n * K + k] = (Half)f32[k * N + n];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constGw[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // Managed schedule cache: pure metadata (no Vulkan objects), shared by
    // every session of this model. The LRU bound only caps managed memory —
    // eviction never touches a GPU resource and nothing is ever invalidated.
    private readonly Dictionary<GpuSchedule.Key, GpuSchedule> _schedules = new();
    private const int MaxSchedules = 512;
    private long _tick;

    internal static GpuSchedule.Key KeyOf(int[] inputShape, int nodeLimit, int outTensor)
        => new(inputShape[0], inputShape[2], inputShape[3], nodeLimit, outTensor);

    // SIMD_OCR_NOCATABS=1 disables sole-consumer concat-slice writes —
    // producers then write their own slot so DebugStats stays accurate.
    private static readonly bool _noCatAbs =
        Environment.GetEnvironmentVariable("SIMD_OCR_NOCATABS") != null;
    private static readonly bool s_dbgTime =
        Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TIME") == "1";
    // SIMD_OCR_NOSK=1: narrow-M MatMuls use the plain dot kernel (bisecting)
    private static readonly bool s_noSplitK =
        Environment.GetEnvironmentVariable("SIMD_OCR_NOSK") != null;
    /// <summary>Schedule for (shape, nodeLimit, outTensor); compiled on first
    /// use. nodeLimit truncates the emit loop and outTensor overrides the
    /// readback tensor (REC stops before the vocab projection).</summary>
    internal GpuSchedule GetSchedule(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        GpuSchedule.Key key = KeyOf(inputShape, nodeLimit, outTensor);
        lock (this)
        {
            if (!_schedules.TryGetValue(key, out GpuSchedule? s))
            {
                if (_schedules.Count >= MaxSchedules)
                {
                    GpuSchedule.Key oldest = default;
                    long min = long.MaxValue;
                    foreach ((GpuSchedule.Key k, GpuSchedule v) in _schedules)
                        if (v.LastTick < min) { min = v.LastTick; oldest = k; }
                    _schedules.Remove(oldest);
                }
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                s = BuildPlan(inputShape, nodeLimit, outTensor);
                _schedules[key] = s;
                if (s_dbgTime)
                    Console.WriteLine($"[t] compile {key} {System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F2} ms");
            }
            s.LastTick = ++_tick;
            return s;
        }
    }

    internal GpuSchedule? TryGetSchedule(GpuSchedule.Key key)
    {
        lock (this) return _schedules.TryGetValue(key, out GpuSchedule? s) ? s : null;
    }

    private unsafe GpuSchedule BuildPlan(int[] inputShape, int nodeLimit, int outTensor)
    {
        int[][] shapes = _compiled.ResolveShapesFor(inputShape);
        NodeRecord[] nodes = _model.Nodes;
        int nT = shapes.Length;
        int outIdx = outTensor >= 0 ? outTensor
