            _pGemmNcS = sg8 ? Pipe("gemm_nc_s", 6, 16, 8u) : Pipe("gemm_nc_ds", 6, 16);
            _pConv1x1N64 = _pConv1x1;
            _pConv1x1N32 = _pConv1x1;
        }
        else if (_sg32)
        {
            // the sg32 lane mapping is hard-wired: without a pinned 32-lane
            // compute subgroup (a wave64 default) or the 16x16x16 fp16 MMA
            // these pipes would silently compute garbage
            if (!dev.CoopMatrix || !dev.Coop16x16x16
                || !(dev.ComputeSubgroupSize || (dev.SubgroupMin == 32 && dev.SubgroupMax == 32)))
                throw new NotSupportedException(
                    "Vulkan: sg32 coopmat path needs 16x16x16 fp16 coopmat and a 32-lane compute subgroup");
            // wave64-capable parts (RDNA) or a shared-memory cap below the
            // 48 KB the staged kernels take: 28 KB single-buffered staging,
            // scalar epilogue, grouped tile raster, plus direct global loads
            // for plain GEMMs (sg32l / sg32d)
            _lite = dev.SubgroupMax >= 64 || dev.MaxSharedMemory < 48 * 1024;
            string f = _lite ? "sg32l" : "sg32";
            _pConv1x1 = Pipe($"conv1x1_cm_{f}", 6, 16, 32);
            _pConv1x1N64 = Pipe($"conv1x1_cm_{f}_n64", 6, 16, 32);
            _pConv1x1N32 = Pipe($"conv1x1_cm_{f}_n32", 6, 16, 32);
            _pConvPs = Pipe($"conv1x1_cm_{f}_ps", 7, 20, 32);
            _pConvPsN64 = Pipe($"conv1x1_cm_{f}_ps_n64", 7, 20, 32);
            _pConvPsN32 = Pipe($"conv1x1_cm_{f}_ps_n32", 7, 20, 32);
            _pConvK = Pipe($"convk_cm_{f}", 6, 68, 32);
            _pConvKN64 = Pipe($"convk_cm_{f}_n64", 6, 68, 32);
            _pConvKN32 = Pipe($"convk_cm_{f}_n32", 6, 68, 32);
            if (_lite)
            {
                _pCmD = Pipe("conv1x1_cm_sg32d", 6, 16, 32);
                _pCmDN64 = Pipe("conv1x1_cm_sg32d_n64", 6, 16, 32);
                _pCmDN32 = Pipe("conv1x1_cm_sg32d_n32", 6, 16, 32);
                _pDw4A = Pipe("conv_dw4a", 6, 48);
            }
        }
        else
        {
            _pConv1x1 = Pipe("conv1x1_cm", 6, 16, 16);
            _pConv1x1N64 = Pipe("conv1x1_cm_n64", 6, 16, 16);
            _pConv1x1N32 = Pipe("conv1x1_cm_n32", 6, 16, 16);
        }
        _pDot = Pipe("conv1x1_dot", 8, 28);
        _pDw = Pipe("conv_dw", 4, 48);
        _pDw4 = Pipe("conv_dw4", 4, 48);
        _pDw4T = Pipe("conv_dw4t", 6, 48);
        _pDense = Pipe("conv_dense", 4, 52);
        _pIm2col = Pipe("im2col", 2, 56);
        _pConvT = Pipe("convt2s2", 5, 24);
        _pConvT4 = Pipe("convt4", 4, 24);
        _pReduce4 = Pipe("reduce_hw4", 2, 16);
        _pReduce4b = Pipe("reduce_hw4b", 2, 12);
        _pPool4 = Pipe("maxpool4", 2, 20);
        _pResize4 = Pipe("resize4", 4, 32);
        _pResize4Add = Pipe("resize4add", 5, 24);
        _pConcat4 = Pipe("concat4", 4, 20);
        _pCatRes = Pipe("catresize", 13, 76);
        _pElem = Pipe("elem", 3, 24);
        _pElem4 = Pipe("elem4", 3, 24);
        _pAddPs = Pipe("addps", 4, 12);
        _pSeF = Pipe("se_fused", 8, 32);
        _pConvD = Pipe("convk_dot", 5, 60);
        _pConvDF32 = Pipe("convk_f32n", 5, 68);
        _pReduce = Pipe("reduce_hw", 2, 8);
        _pPool = Pipe("maxpool2e", 2, 12);
        _pResize = Pipe("resize_nn", 2, 20);
        _pConcat = Pipe("concat_c", 2, 16);
        _pNchw = Pipe("nchw2nhwc", 2, 12);
        _pOut = Pipe("sigmoid_out", 2, 8);
        _pAvg4 = Pipe("avgpool4", 2, 44);
        _pAffine = Pipe("affine4", 4, 8);
        _pSoftmax = Pipe("softmax", 2, 8);
        _pLn = Pipe("layernorm", 4, 12);
        _pAttn = Pipe("attn", 2, 16);
        _pDotSk = Pipe("mmdot_sk", 8, 28);
        // Pipelines are in the process-wide cache now; persist it so the next
        // process does not compile the same shaders again.
        _dev.SavePipelineCache();
    }

    private VkPipeline Pipe(string name, int bindings, int pcBytes, uint reqSg = 0)
        => _dev.GetPipeline(name, () => LoadSpv(name), bindings, pcBytes, reqSg);

    private static byte[] LoadSpv(string name)
    {
        var asm = typeof(GpuGraphModel).Assembly;
        using Stream s = asm.GetManifestResourceStream(
            $"Sdcb.SimdPaddleOCR.Backends.Vulkan.Shaders.{name}.spv")
            ?? throw new FileNotFoundException(name + ".spv");
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }

    // fp16 device copy of a constant tensor; padded to padRows*rowElemsPad
    // elements. rowPad>rowElems repacks [rows, rowElems] into [rows, rowPad].
    private VkBuffer ConstF16(int tensorIndex, int padRows = 0, int rowElems = 0,
        int rowPad = 0)
    {
        var key = (tensorIndex, padRows, rowElems, rowPad);
        if (_constF16.TryGetValue(key, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int n = f32.Length;
        int alloc;
        Half[] h;
        if (rowPad > rowElems && rowElems > 0)
        {
            int rows = Math.Max(n / rowElems, padRows);
            alloc = rows * rowPad;
            h = new Half[alloc];
            for (int r = 0; r * rowElems < n; r++)
                for (int c = 0; c < rowElems && r * rowElems + c < n; c++)
                    h[r * rowPad + c] = (Half)f32[r * rowElems + c];
        }
        else
        {
            // whole 16 B: kernels read scalars as a packed half pair / f16vec4,
            // and Adreno bounds-checks every load against the binding range
            alloc = (Math.Max(n, padRows * rowElems) + 7) / 8 * 8;
