# SimdPaddleOCR 性能基准测试与优化分析报告 (Medium 模型, 4 Workers)

## 1. 测试场景与命令行

### 场景一：标准 100 图测试
**OpenVINO:**
```bash
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --workers 4 --model medium --engine openvino --input dataset --out bench-out/openvino-medium-4w.json
```
**C# (Sharp):**
```bash
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --workers 4 --model medium --engine sharp --input dataset --out bench-out/sharp-medium-4w.json
```

### 场景二：0912客户13图测试
**OpenVINO:**
```bash
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --workers 4 --model medium --engine openvino --input <0912客户13图目录> --out bench-out/openvino-medium-4w-0912.json
```
**C# (Sharp):**
```bash
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --workers 4 --model medium --engine sharp --input <0912客户13图目录> --out bench-out/sharp-medium-4w-0912.json
```

---

## 2. 测试结果总结

### 2.1 标准 100 图测试结果
| 引擎 | 端到端平均耗时 | 吞吐量 (img/s) | 准确率 (CER) | det_graph (ms) | cls_graph (ms) | rec_graph (ms) |
|---|---|---|---|---|---|---|
| **OpenVINO** | **989.7 ms** | **1.01** | 1.51% | **233.07** | **5.31** | **645.35** |
| **C# (Sharp)**| 1269.9 ms | 0.79 | **0.68%** | 554.61 | 17.24 | 2062.82 |

*注：C# 引擎在准确率上表现更好，但 OpenVINO 在推理速度上快约 28%。*

### 2.2 0912客户13图测试结果
| 引擎 | 端到端平均耗时 | 吞吐量 (img/s) | det_graph (ms) | cls_graph (ms) | rec_graph (ms) |
|---|---|---|---|---|---|
| **OpenVINO** | **1170.7 ms** | **0.85** | **410.13** | **16.48** | **708.47** |
| **C# (Sharp)**| 1278.5 ms | 0.78 | 750.01 | 33.00 | 1585.52 |

*注：OpenVINO 端到端速度快约 9%。C# 的非推理图阶段（如内存管理、多线程调度）效率较高，拉回了部分差距，但纯推理图（尤其是 rec_graph）耗时仍是 OpenVINO 的两倍以上。*

---

## 3. 性能瓶颈分析与 C# 优化方向 (To 优化 AI)

本节提炼核心差异与可行的 C# 优化点，供后续优化参考（已精简以节省 Token）：

### 核心差异 (Why OpenVINO is faster in Medium model)
1. **全局阻塞内存布局 (Blocked Memory Layout):** OpenVINO 贯穿使用 `nChw8c`/`nChw16c` 格式，完美适配 AVX 顺序访存。C# 引擎维持 `NCHW`，在 `Conv1x1` 内部动态转置为 `NHWC` 再转回，导致 Medium 模型下极大的内存带宽开销。
2. **JIT 运行时汇编 (Runtime Assembly):** OpenVINO 基于 Xbyak 动态生成硬编码步长和维度的汇编，消除循环开销，并实现完美的寄存器分块。C# 依赖预编译的泛型 Intrinsics，存在循环控制开销。
3. **极进的算子融合 (Epilogue Fusion):** OpenVINO 将 `Conv + BN + Act` 融合，中间结果不写回主存。C# 的后处理仍有部分需要遍历数组写回主存。

### C# 引擎优化建议 (Action Items)
1. **图级别内存布局重构 (高收益):** 消除算子内部的 `NCHW` <-> `NHWC` 转置。重构推理图，使张量在层间传递时直接保持 Blocked 格式 (如 `nChw8c`)，需同步重写 Pooling、Eltwise 等算子。
2. **深度算子融合 (高收益):** 利用 C# 11 `static abstract` 或函数指针，将激活函数 (HardSwish, ReLU) 内联到 `Conv1x1` 和 `Conv3x3` 的最内层计算循环中，避免数据落回主存。
3. **基于 Source Generator 的 AOT 展开 (中收益):** 针对 PaddleOCR 固定的网络 Shape，在编译期生成无 `for` 循环、硬编码步长的 C# 代码，逼近 JIT 汇编性能。
4. **动态 L2 Cache 块大小 (中收益):** 优化 `Parallel.For` 的切分逻辑（目前如 `Conv1x1.Avx.cs` 中硬编码 `tileSpatial = 64`），根据运行时的 L2 Cache 大小动态调整，提升缓存命中率。

---

## 4. 重构结果（2026-09-14，同一台 5800X / AVX2）

采纳了上面第 1、2 条（图级别 NHWC 布局 + 算子融合到 epilogue），并顺带做了神经网络 neck 部分的融合与线程调度调整。
所有对比均为同机、同数据、同一次会话内背靠背运行；文本输出（每行识别结果与 `hash`）与重构前 **逐位一致**。

### 4.1 端到端

| 用例 | 重构前 | 重构后 | OpenVINO（同机） |
|---|---|---|---|
| medium 4w，100 图 mean / median | 1076.6 / 1062.9 ms | **635.2 / 638.5 ms**（-41%） | 1013.7 / 988.8 ms |
| medium 4w，0912 客户 13 图 mean | 1278.5 ms（原报告） | **570.9 ms** | 1170.7 ms（原报告） |
| medium 1w，10 图 mean | 1931.7 ms | **~920 ms** | – |
| tiny 4w / 1w，100 图 mean | 114.2 / 188.1 ms | **~85 / ~134 ms** | – |
| small 4w，40 图 mean | 292.6 ms | **239.5 ms** | – |
| tiny 4w，netstandard2.0 库（Vector 路径，未启用 NHWC） | 214.1 ms | **177.2 ms** | – |

medium 4w 分阶段：`det_graph` 474.7 → 232.5 ms，`lines_wall` 585.6 → 386.6 ms。准确率不变（exact_lines 992/1026，CER 0.68%）。

### 4.2 做了什么

- **图级别 NHWC 布局段**（`OnnxSharp/LayoutPlanner.cs`）：模型加载时把连续的、全部算子都有 channels-last 实现的子图切成 NHWC 段，在边界插入 `LayoutConvert` 节点；张量逻辑形状仍为 NCHW，仅存储顺序不同（`TensorNhwc` 标记）。medium det 全图只剩输入端 1 次转换，rec 主干只在进入 SVTR neck 前转换 1 次。仅在 AVX2+FMA 下启用（`PPOCR_NHWC=0` 可关闭）；其他 ISA / netstandard2.0 走原 NCHW 内核，不受影响。
- **NHWC 内核**（`Kernels/Nhwc*.cs`）：一个 6 像素 × 16 输出通道的 FMA 微内核同时服务 pointwise、任意 KxK dense（隐式 GEMM，输入通道→tap 的归约顺序与参考实现一致）和 2x2/s2 ConvTranspose；depthwise（行条带累加，规避 NHWC 行距为 4 KiB 整数倍时的 L1 组冲突）；池化、最近邻 Resize、Concat、ReduceMean、BatchNorm、通道广播二元算子。1x1 512↔1024 @6×320 单次从 8.85 ms（4 线程）降到 ~1.6 ms（8 线程），接近 FMA 峰值。
- **Epilogue 融合**：bias、残差、ReLU、HardSwish、GELU（与 `SimdKernels.Gelu` 同一多项式）直接在累加器上完成，7.9 MB 的激活只写一次。
- **Neck 融合与快速路径**：LayerNorm 9 节点融合为一个内核；`Transpose`/`Slice` 改为步长遍历；末轴 `ReduceMean`、行标量广播快速路径；中等尺寸常量 `MatMul` 走多线程 NHWC GEMM（累加顺序与 `MatMulRows4` 相同）。
- **调度**：REC 行按估计宽度降序 + 共享游标动态分配（LPT），缩短 4 worker 的尾部；单 worker 时 REC intra-op 上限 4 → 8；NHWC 启用时 DET 使用全部逻辑核（SMT 对 FMA 密集内核有 8~25% 增益）。

### 4.3 工具

- `PPOCR_DUMP_NODES=1`：按节点（含融合组）累计耗时到 stderr，`bench-out/tools/agg_nodes.py` 聚合。
- `PPOCR_DUMP_GRAPH=1`：打印布局规划后的图（哪些张量是 `@nhwc`、插入了哪些 `LayoutConvert`）。
- `dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --kernel-bench [threads] [repeats]`：单独测 NHWC 内核并交替比较调参项（5800X 在持续 AVX 负载下会热降频，同一进程内交替测量才可比）。
- 基准程序现在接受任意 JPG 目录（无 `metadata.json` 时跳过准确率统计），可直接跑客户图。