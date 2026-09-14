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