# SimdPaddleOCR — WebAssembly (WASM) 4w 性能与基准测试报告

测试环境：
* **宿主硬件**：AMD Ryzen 7 5800X（8C/16T）、64 GB 内存，Windows 11
* **运行时框架**：.NET 10.0 (`net10.0` / `browser-wasm`)，`wasm-tools-net10` 负载（Emscripten 3.1.56）
* **浏览器引擎**：Microsoft Edge 155（Chromium 内核，Headless 模式）
* **SIMD 特性**：WASM SIMD128，`PackedSimd.IsSupported = True`（`Vector<float>.Count = 4`）
* **编译模式**：LLVM AOT 静态编译 (`<RunAOTCompilation>true</RunAOTCompilation>`)
* **多线程机制**：Web Workers + `SharedArrayBuffer`（由本地服务端注入 COOP: `same-origin` 与 COEP: `require-corp` 安全响应头）
* **并发参数**：`--workers 4`（`EffectiveLineWorkerCount = 4`）
* **测试工程**：`test/Sdcb.SimdPaddleOCR.WasmBench`，评测集为 `dataset/` 全量 **100 张标准图像**（1 轮预热 + 99 轮实测，共 1,026 行标注）

---

## 一、全量 100 图端到端性能 (4 Workers)

CPU 内核派发顺序为 **AVX-512 → AVX2 → AdvSIMD → PackedSimd → `Vector<float>` → 标量**。浏览器中命中 `PackedSimd`（WebAssembly SIMD128）专用内核，实测如下：

| 模型 | 端到端平均耗时 (Mean ms) | 中位数 (Median ms) | P95 延迟 (ms) | 吞吐量 (img/s) |
| :--- | :---: | :---: | :---: | :---: |
| **ChineseV6Tiny (tiny)** | **113.3 ms** | **113.1 ms** | **141.3 ms** | **8.83 img/s** |
| **ChineseV6Small (small)** | **409.9 ms** | **412.0 ms** | **523.4 ms** | **2.44 img/s** |
| **ChineseV6Medium (medium)**| **1,360.4 ms** | **1,371.0 ms** | **1,732.6 ms** | **0.74 img/s** |

> *注：以上数据均为开启 LLVM AOT 静态编译的实测指标。若在未开启 AOT 的纯 JIT（Jiterpreter）模式下运行，性能会慢 8~30 倍，JIT 模式仅建议作为跳过编译等待的本地快速调试手段，生产环境必须使用 AOT。*

在浏览器多线程环境下，`tiny 4w` 运行全量 100 张图平均耗时 **113.3 ms**（吞吐量 **8.83 张/秒**），同机桌面原生（`--engine sharp --workers 4`，tiny 中位 45.9 ms）约为其 2.5 倍，满足网页前端纯离线、低延迟 OCR 的需要。

### PackedSimd 与 `Vector<float>` 回退路径对比（同机 A/B）

同一台机器、同一浏览器交替运行，对比仅走 `Vector<float>` 的旧版本（tiny/small 为两轮平均，medium 为单轮）：

| 模型 | `Vector<float>` Mean | PackedSimd Mean | 提速 | 文本 Hash |
| :--- | :---: | :---: | :---: | :---: |
| tiny | 128.1 ms | 112.9 ms | **1.13×** | 100 张完全一致 |
| small | 474.6 ms | 411.3 ms | **1.15×** | 100 张完全一致 |
| medium | 1,654.6 ms | 1,360.4 ms | **1.22×** | 100 张完全一致 |

PackedSimd 内核与 `Vector<float>` 路径逐 lane 保持相同的运算顺序（无 FMA，乘加分离），因此在有限输入下结果逐位一致（ReLU / HardSwish 用的 `PseudoMax` / `PseudoMin` 遇到 NaN 时返回 0，而 `Vector.Max` 会传播 NaN）。主要收益来源：

* **寄存器分块重排**：V8 在 x64 上最多只给 v128 分配 15 个 XMM 寄存器，原分块会在内层循环溢出：NHWC 卷积在 `Vector<float>.Count == 4` 时是 6 行 × 8 通道分两次（12 个累加器），packed MatMul 是 4 行 × 16 列（16 个累加器）。PackedSimd 的 NHWC pointwise / dense 卷积与 MatMul 改为 4 像素 × 8 输出通道（8 个累加器），单核微基准 **1.4~1.9×**。
* **Dense 卷积预计算 k 偏移表**：去掉内层循环里的 tap/通道簿记，3×3 dense 卷积 **1.5~1.9×**。
* **MatMul + ArgMax（CTC 头）**：原 `Vector<float>` 实现把乘加累加器放在 `stackalloc` 数组中，PackedSimd 版本的累加器驻留寄存器（每行的最大值 / 下标状态仍在栈上，每 8 列更新一次），**2.3×**。
* Depthwise、Erf/GELU 等访存或分支受限的算子收益较小（约 1.0~1.15×）。

---

## 二、模型识别准确率 (全量 100 图 Accuracy & CER)

以 `dataset/metadata.json` 真实标注为基准计算字符错误率（CER）与行绝对匹配率：

| 模型 | 字符准确率 (Char Accuracy) | 字符错误率 (CER) | 行绝对匹配 (Line Exact) | 行匹配率 |
| :--- | :---: | :---: | :---: | :---: |
| **ChineseV6Tiny (tiny)** | 97.70% | 2.30% | 734 / 1,026 | 71.54% |
| **ChineseV6Small (small)** | 99.58% | 0.42% | 940 / 1,026 | 91.62% |
| **ChineseV6Medium (medium)**| **99.86%** | **0.14%** | **994 / 1,026** | **96.88%** |

> **准确率观察**：WASM 下全托管推理引擎 `OnnxSharp` 达成极高数值确定性。全量 100 张图实测下，Medium 模型的 CER 仅为 0.14%（字符准确率 99.86%），行绝对匹配率高达 96.88%，PackedSimd 与 `Vector<float>` 路径文本 Hash 完全一致，没有任何精度退化。

---

## 三、分阶段耗时剖析 (Mean ms/图)

| 阶段 | ChineseV6Tiny (4w) | ChineseV6Small (4w) | ChineseV6Medium (4w) | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **det_preprocess** | 1.8 ms | 1.8 ms | 1.6 ms | 缩放、归一化与通道重排 |
| **det_graph** | **54.1 ms** | **192.6 ms** | **404.8 ms** | 文本检测网络推断 |
| **det_postprocess** | 3.2 ms | 3.9 ms | 2.9 ms | 概率图二值化、轮廓与多边形拟合 |
| **crop** | 3.5 ms | 4.3 ms | 2.8 ms | 仿射透视校正提取文字行 |
| **cls_graph** | 29.8 ms | 32.1 ms | 24.3 ms | 文本行方向分类 (180° 翻转检测) |
| **rec_preprocess** | 8.1 ms | 8.9 ms | 6.9 ms | 动态宽高归一化 |
| **rec_graph (总和)** | 141.4 ms | 712.0 ms | 3,513.7 ms | 所有文字行识别网络的算力耗时总和 |
| **lines_wall (实测墙钟)** | **50.2 ms** | **206.8 ms** | **947.8 ms** | **4 个 Web Worker 并发执行后的实际耗时** |
| **rec_reshape** | 2.5 ms | 5.7 ms | 5.5 ms | 动态宽度重塑与权重预备 |

> **多线程并行收益**：
> 在 4 个并发 Worker 的协同下：
> * `tiny` 的文字识别墙钟耗时从 `rec_graph` 141.4 ms 压缩到 `lines_wall` 50.2 ms（**加速比 ~2.82×**）；
> * `small` 从 712.0 ms 压缩到 206.8 ms（**加速比 ~3.44×**）；
> * `medium` 从 3,513.7 ms 压缩到 947.8 ms（**加速比 ~3.71×**，非常接近 4× 理论上限）。
> 充分证明了 WebAssembly 多线程与任务并行调度在行识别阶段的高效并发。

---

## 四、内存占用与工作集 (Working Set)

在 AOT 静态编译模式下，方法元数据和字节码直接编译为 native wasm 代码段，托管堆内存占用极低：

| 模型 | 平均托管内存占用 | 峰值托管内存占用 | 内存安全评价 (WASM 2GB 限制) |
| :--- | :---: | :---: | :--- |
| **ChineseV6Tiny (tiny)** | **60.9 MB** | **74.0 MB** | 极低，适合低内存移动端浏览器 |
| **ChineseV6Small (small)** | **126.8 MB** | **158.0 MB** | 极低且平稳 |
| **ChineseV6Medium (medium)**| **383.5 MB** | **458.8 MB** | 远低于 2GB/4GB 寻址上限，无内存溢出风险 |

---

## 五、WebAssembly 运行要点与 Harness 实现

### 1. 多线程与跨域隔离 (COOP/COEP)
WebAssembly 原生不支持传统 OS 级轻量进程，.NET 10 的 `<WasmEnableThreads>true</WasmEnableThreads>` 依靠浏览器底层的 **Web Workers** 和 **SharedArrayBuffer**：
* 浏览器出于安全规范，要求页面必须运行在跨域隔离上下文（`crossOriginIsolated == true`）。
* 因此，Harness 的本地服务端（`server.mjs`）配置了：
  ```http
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```
* 若在纯 Node.js 宿主执行带线程的 WASM，Mono 运行时会直接触发 `Assert failed: This build of dotnet is multi-threaded, it doesn't support shell environments like V8 or NodeJS` 保护异常。

### 2. AOT 编译机制
* 在工程配置中默认开启 `<RunAOTCompilation>true</RunAOTCompilation>`。
* 需要本机安装 `wasm-tools-net10` 负载（通过 `dotnet workload install wasm-tools-net10`）。
* 编译时 .NET 自动调用 Emscripten 3.1.56 + LLVM 工具链生成高度优化的单一 WebAssembly 原生模块。

---

## 六、复现与基准测试运行指南

本项目提供了完整的自动化 Harness，位于 `test/Sdcb.SimdPaddleOCR.WasmBench`：

```powershell
# 1. 运行 tiny 模型 4 线程全量基准测试 (默认 100 张图，AOT 模式)
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model tiny -Count 100

# 2. 运行 small 模型 4 线程全量基准测试
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model small -Count 100

# 3. 运行 medium 模型 4 线程全量基准测试
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model medium -Count 100

# 4. 如需快速调试（跳过耗时的 AOT 编译），可加 -NoAot 参数使用 JIT 模式：
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model tiny -NoAot

# 5. 打开浏览器交互式仪表盘（实时可视化进度条、指标卡片与文本 Hash）
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model tiny -Open
```
