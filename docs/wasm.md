# SimdPaddleOCR — WebAssembly (WASM) 4w 性能与基准测试报告

测试环境：
* **宿主硬件**：与 [`vulkan-uhd770.md`](vulkan-uhd770.md) 同一台实测机（x64 桌面 CPU），Windows 11
* **运行时框架**：.NET 10.0 (`net10.0` / `browser-wasm`)
* **浏览器引擎**：Microsoft Edge (Chromium 内核，Headless 模式)
* **SIMD 特性**：WASM SIMD128 (`Vector<float>.Count = 4`, `Vector.IsHardwareAccelerated = True`)
* **编译模式**：LLVM AOT 静态编译 (`<RunAOTCompilation>true</RunAOTCompilation>`)
* **多线程机制**：Web Workers + `SharedArrayBuffer`（由本地服务端注入 COOP: `same-origin` 与 COEP: `require-corp` 安全响应头）
* **并发参数**：`--workers 4`（`EffectiveLineWorkerCount = 4`）
* **测试工程**：`test/Sdcb.SimdPaddleOCR.WasmBench`，评测集为 `dataset/` 全量 **100 张标准图像**（1 轮预热 + 99 轮实测，共 1,026 行标注）

---

## 一、全量 100 图端到端性能 (4 Workers)

在开启 LLVM AOT 静态全量编译后，C# 算子和指针操作直接转为原生 WebAssembly 二进制机器指令，全量 100 张图的实测表现如下：

| 模型 | 端到端平均耗时 (Mean ms) | 中位数 (Median ms) | P95 延迟 (ms) | 吞吐量 (img/s) |
| :--- | :---: | :---: | :---: | :---: |
| **ChineseV6Tiny (tiny)** | **102.8 ms** | **103.3 ms** | **141.5 ms** | **9.73 img/s** |
| **ChineseV6Small (small)** | **355.8 ms** | **362.6 ms** | **466.0 ms** | **2.81 img/s** |
| **ChineseV6Medium (medium)**| **2,935.8 ms** | **2,162.3 ms** | **13,886.5 ms** | **0.34 img/s** |

> *注：以上数据均为开启 LLVM AOT 静态编译的实测指标。若在未开启 AOT 的纯 JIT（Jiterpreter）模式下运行，性能会慢 8~30 倍，JIT 模式仅建议作为跳过编译等待的本地快速调试手段，生产环境必须使用 AOT。*

在浏览器多线程环境下，`tiny 4w` 运行全量 100 张图平均耗时仅需 **102.8 ms**（吞吐量高达 **9.73 张/秒**），相比同机桌面原生（`--engine sharp`，tiny 中位 ~44ms，见 [`vulkan-uhd770.md`](vulkan-uhd770.md)）仅相差约 2.3 倍，已完全达到商用网页前端纯离线、低延迟实时 OCR 的性能标准。

---

## 二、模型识别准确率 (全量 100 图 Accuracy & CER)

以 `dataset/metadata.json` 真实标注为基准计算字符错误率（CER）与行绝对匹配率：

| 模型 | 字符准确率 (Char Accuracy) | 字符错误率 (CER) | 行绝对匹配 (Line Exact) | 行匹配率 |
| :--- | :---: | :---: | :---: | :---: |
| **ChineseV6Tiny (tiny)** | 97.70% | 2.30% | 734 / 1,026 | 71.54% |
| **ChineseV6Small (small)** | 99.58% | 0.42% | 940 / 1,026 | 91.62% |
| **ChineseV6Medium (medium)**| **99.86%** | **0.14%** | **994 / 1,026** | **96.88%** |

> **准确率观察**：WASM 下全托管推理引擎 `OnnxSharp` 达成极高数值确定性。全量 100 张图实测下，Medium 模型的 CER 仅为 0.14%（字符准确率 99.86%），行绝对匹配率高达 96.88%，文本 Hash 与桌面端完全一致，没有任何精度退化。

---

## 三、分阶段耗时剖析 (Mean ms/图)

| 阶段 | ChineseV6Tiny (4w) | ChineseV6Small (4w) | ChineseV6Medium (4w) | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **det_preprocess** | 2.1 ms | 2.0 ms | 2.6 ms | 缩放、归一化与通道重排 |
| **det_graph** | **53.7 ms** | **152.7 ms** | **814.6 ms** | 文本检测网络推断 |
| **det_postprocess** | 2.3 ms | 2.5 ms | 3.2 ms | 概率图二值化、轮廓与多边形拟合 |
| **crop** | 2.4 ms | 2.7 ms | 3.1 ms | 仿射透视校正提取文字行 |
| **cls_graph** | 20.9 ms | 30.7 ms | 33.4 ms | 文本行方向分类 (180° 翻转检测) |
| **rec_preprocess** | 4.1 ms | 5.8 ms | 7.2 ms | 动态宽高归一化 |
| **rec_graph (总和)** | 126.7 ms | 675.5 ms | 7,854.8 ms | 所有文字行识别网络的算力耗时总和 |
| **lines_wall (实测墙钟)** | **42.0 ms** | **195.5 ms** | **2,111.9 ms** | **4 个 Web Worker 并发执行后的实际耗时** |
| **rec_reshape** | 2.1 ms | 5.8 ms | 22.2 ms | 动态宽度重塑与权重预备 |

> **多线程并行收益**：
> 在 4 个并发 Worker 的协同下：
> * `tiny` 的文字识别墙钟耗时从 `rec_graph` 126.7 ms 压缩到 `lines_wall` 42.0 ms（**加速比 ~3.02×**）；
> * `small` 从 675.5 ms 压缩到 195.5 ms（**加速比 ~3.46×**）；
> * `medium` 从 7,854.8 ms 压缩到 2,111.9 ms（**加速比 ~3.72×**，非常接近 4× 理论上限）。
> 充分证明了 WebAssembly 多线程与任务并行调度在行识别阶段的高效并发。

---

## 四、内存占用与工作集 (Working Set)

在 AOT 静态编译模式下，方法元数据和字节码直接编译为 native wasm 代码段，托管堆内存占用极低：

| 模型 | 平均托管内存占用 | 峰值托管内存占用 | 内存安全评价 (WASM 2GB 限制) |
| :--- | :---: | :---: | :--- |
| **ChineseV6Tiny (tiny)** | **62.0 MB** | **74.5 MB** | 极低，适合低内存移动端浏览器 |
| **ChineseV6Small (small)** | **133.5 MB** | **162.7 MB** | 极低且平稳 |
| **ChineseV6Medium (medium)**| **378.9 MB** | **457.0 MB** | 远低于 2GB/4GB 寻址上限，无内存溢出风险 |

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
