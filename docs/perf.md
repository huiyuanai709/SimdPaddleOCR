# Sdcb.SimdPaddleOCR 性能基准

## 1.4（`b784d28`，2026-09-17）

相对 **1.3.0**（`37fe1fd`），图级 NHWC 从「仅 AVX2+FMA」扩到：

| 路径                                        | 1.3                    | 1.4                                                                                |
| ------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| net10 AVX2+FMA                              | 图级 NHWC              | 同左；预处理直接写 NHWC，少一次输入 `LayoutConvert`                                |
| `netstandard2.0`（`tiny-4w-ns2`，`Vector`） | NCHW                   | **NHWC**                                                                           |
| x64 `DOTNET_EnableHWIntrinsic=0`（scalar）  | NCHW + 软件模拟 Vector | **专用 NHWC 标量 tile**（不再转 `*Vec`）                                           |
| net10 AdvSIMD（linux-arm64 / osx-arm64）    | NCHW                   | **手写 NHWC NEON tile**                                                            |
| 公开 `InferenceSession.Run`                 | 逻辑 NCHW              | 仍收逻辑 NCHW；图输入已标 NHWC 时入口自动转置                                      |
| 像素格式                                    | 只认紧排 BGR           | 默认仍 `Bgr24`；RGB24 / BGRA32 / RGBA32 在 resize / warp 就地 gather，不摊中间 BGR |

`PPOCR_NHWC=0` 仍可整图关回 NCHW。正确率不变。CI 不再跑 OpenVINO.NET。

**内存占用大幅下降。** tiny-4w 工作集峰值大约少 **300 MB**：win-x64 7763 **817 → 515 MB**，linux-arm64 N2 **840 → 572 MB**；跑图 Δ WS 从约 400 MB 降到约 100–160 MB。主要是 NHWC workspace 别名，以及预处理直接写 NHWC、不再为输入 `LayoutConvert` 留第二份缓冲。

### 怎么读

- **墙钟**默认去掉首张 warmup 后的 **median ms/图**（`--warmup 1`）。准确率和 ΔWS（初始化后 → 全部跑完）按满勤。win-x64 SIMD 先丢一次 25 张 tiny-4w 烤 VM（不上传），再跑默认 / noavx512 / ns2。noavx2 / noavx / scalar 只在 `smoke-win-x64-isa` 跑 20 张。下文 1.4 表仍是当时「首张 warmup、准确率也跳过首张」的尺子。
- 每个 replica 是一台独立的 GitHub-hosted VM。先在单 replica 内算比值，再只汇总 **同一 CPU**。
- GitHub `windows-2025` 会随机分到 EPYC 7763 / 9V74 / Xeon。**7763 没有 AVX-512**；9V74 / Xeon 有时走 AVX-512。这两类绝对时间不可比。
- 4 worker 下算子会并行重叠，**之和可以大于墙钟**，只适合看结构。
- 1.2 旧基线已从本文删除。相对 1.2 的数字见当时的 1.3 说明：AVX2 tiny 约 0.69×，本机 medium 约 0.43×。

### 数据来源

| 版本       | Actions                                                                       | 提交      | 说明                                          |
| ---------- | ----------------------------------------------------------------------------- | --------- | --------------------------------------------- |
| **1.3.0**  | [34818949921](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34818949921) | `37fe1fd` | 只 AVX2 NHWC；ns2 / ARM / scalar 仍 NCHW      |
| **1.4**    | [35241030966](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35241030966) | `b784d28` | 全路径 NHWC + 像素格式；下文「1.4」默认指这次 |
| 1.4 前一次 | [35217602432](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35217602432) | `67cf1fa` | 内核已是 1.4；用来给 win-x64 7763 补样本      |

数据集、模型、预解码 BGR、ISA 开关、runner 规格与以前相同：`.github/workflows/test.yml`，`dataset/` 固定种子合成 100 张 JPG。库 TFM 默认 `net10.0`；`tiny-4w-ns2` 把库编成 `netstandard2.0`，仍跑在 .NET 10 上。

主基线仍是 **win-x64 + EPYC 7763 + AVX2** 和 **linux-arm64 Neoverse N2**（`CPU part 0xd49`，6 replica 几乎一条直线）。osx-arm64 是 3 核 / 7 GB 虚拟 M1，只看同 replica 比值。

---

## 1.4 vs 1.3：各路径

正确率两边都是 sharp **757/1022、CER 3.53%**（含 scalar、ns2、全部 ISA）；c 仍是 759/1022、4.18%。smoke：tiny 152/208，small 188/208，medium 200/208。

### linux-arm64 N2（最稳，6 replica）

1.3 是 NCHW AdvSIMD；1.4 是手写 NHWC NEON。同一 `ubuntu-24.04-arm`、同一 `0xd49`。

| 用例             | 1.3 中位 | 1.4 中位 |      相对 | 1.3 同 replica 比 | 1.4 同 replica 比 |
| ---------------- | -------: | -------: | --------: | ----------------: | ----------------: |
| `tiny-4w`        |  **241** |  **180** | **0.75×** |              1.00 |              1.00 |
| `tiny-1w`        |      390 |  **291** | **0.75×** |              1.63 |              1.65 |
| `tiny-4w-ns2`    |      374 |  **295** | **0.79×** |              1.55 |              1.66 |
| `tiny-4w-scalar` |      984 |  **856** | **0.87×** |              4.16 |              4.75 |

读法：

- net10 AdvSIMD 是 1.4 最大的平台级收益：4w / 1w 都大约快 **25%**。
- ns2 在 ARM 上仍走 `Vector`（128-bit），没有手写 tile，但也吃到预处理直写 NHWC，大约快 **21%**。
- scalar 有专用 NHWC tile，大约快 **13%**。相对 4w 的倍数从 4.16 升到 4.75，是因为 4w 自己更快了，不是 scalar 变慢。
- 1.3 曾用 NCHW AdvSIMD vs NHWC `Vector` Count==4（约 230 vs 286）决定默认关闸；手写 NEON 之后已经翻过来。

工作集（`tiny-4w`）：peak **840 → 572 MB**，Δ WS **418 → 162 MB**。NHWC packed workspace + 别名，不是测法变了。

### win-x64 SIMD（只报 EPYC 7763）

1.3 这次 run 有 3 份 7763；1.4 最新 run 也是 3 份（另 3 份落到 Xeon / 9V74）。中位是各 replica median 的中位数。

| 用例               | 有效 ISA      | 1.3 中位 |      1.4 中位（范围） |          相对 | 1.3 同 replica 比 | 1.4 同 replica 比 |
| ------------------ | ------------- | -------- | --------------------: | ------------: | ----------------: | ----------------: |
| `tiny-4w`          | AVX2          | **167**  |    **184**（183–221） | ~1.1×（噪声） |              1.00 |              1.00 |
| `tiny-4w-noavx512` | AVX2          | 152      |    **140**（136–156） |         0.92× |              0.89 |              0.76 |
| `tiny-4w-noavx2`   | AVX           | 307      |    **328**（327–331） |         1.07× |              1.75 |              1.75 |
| `tiny-4w-noavx`    | Vector        | 481      |    **380**（376–390） |     **0.79×** |              2.76 |              2.00 |
| `tiny-4w-ns2`      | Vector / NHWC | **343**  |    **228**（220–229） |     **0.66×** |          **2.02** |          **1.24** |
| `tiny-4w-scalar`   | scalar        | 1368     | **1220**（1199–1278） |     **0.89×** |               7.7 |               6.3 |

读法：

- **AVX2 默认路径和 1.3 持平。** SIMD job 连续跑 6 个 case，7763 单次 183–221 都见过；引擎套件同机 4w 反而从 154 降到 142（见下）。不要用一份 221 喊回归。
- **ns2 是 x64 上 1.4 最大的收益**：343 → 228（约 **1.5×** 吞吐）。1.3 的 ns2 仍是 NCHW Vector，和关掉 AVX2 几乎同级（比值 ~2.0）；1.4 走到 NHWC，比值掉到 **1.24**。
- **noavx**（只留 `Vector`）同样吃到 NHWC：481 → 380（**0.79×**）。
- **scalar** 换成专用 16 路寄存器累加 tile，不再走软件模拟 Vector：1368 → 1220（**0.89×**）。本机 small-4w、`count=5`、`HWIntrinsic=0` 从 1750 降到 1237（约 **1.42×**），和 CI tiny 同方向、幅度更大（small 的 generic conv 更吃 NHWC）。
- `noavx512` 在 7763 上本来就没有 AVX-512，和 `tiny-4w` 同 ISA；两边差值是噪声地板。
- `noavx2`（只留 AVX）没有单独的 NHWC AVX tile，墙钟和 1.3 重叠。

并入 [35217602432](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35217602432) 的 4 份 7763 之后，1.4 `tiny-4w` 中位约 **191**（n=7，范围 183–221），ns2 仍是 221–241。结论不变。

工作集（7763 `tiny-4w`）：peak **817 → 515 MB**，Δ WS **398 → 107 MB**。

### win-x64 引擎套件（只报 EPYC 7763）

和 SIMD job 不是同一台 VM，绝对毫秒不要和上一张表硬接。1.3 CI 的 c 还不是 `20d0de6`；1.4 已钉到 [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)。1.4 不再跑 OpenVINO.NET。

| 用例            | 1.3 中位 | 1.4 中位 |         相对 | 1.3 vs sharp 4w | 1.4 vs sharp 4w |
| --------------- | -------: | -------: | -----------: | --------------: | --------------: |
| sharp `tiny-4w` |      154 |  **142** |        0.92× |            1.00 |            1.00 |
| sharp `tiny-1w` |      217 |  **200** |        0.92× |            1.41 |            1.41 |
| c `tiny-4w`     |      288 |      254 | （c 换 DLL） |            1.82 |        **1.76** |
| c `tiny-1w`     |      496 |      360 | （c 换 DLL） |            3.10 |            2.43 |
| OpenVINO.NET 4w |      216 |        — | 已从 CI 去掉 |        **1.38** |               — |

1.3 同 replica OpenVINO / sharp 是 **1.38**（本库反超）；这条只解释历史，不再新测。1.4 c/sharp 约 **1.7×**，c 更省内存（peak ~535 MB vs sharp ~513 MB）。

### osx-arm64（高噪声，只看比值）

虚拟 M1、3 逻辑核、7 GB。绝对毫秒 replica 之间可以差一倍，**不能**拿单次墙钟下结论。

| 用例             | 1.3 同 replica 比 | 1.4 同 replica 比 |
| ---------------- | ----------------: | ----------------: |
| `tiny-4w`        |              1.00 |              1.00 |
| `tiny-1w`        |              ~1.9 |              ~2.1 |
| `tiny-4w-ns2`    |              1.20 |          **1.67** |
| `tiny-4w-scalar` |              4.08 |              4.39 |

1.4 的 4w 自己变快之后，ns2（仍是 Vector）相对倍数会被拉开，和 linux-arm64 同一现象。正确率仍是 757/1022。只用来确认 AdvSIMD 路径能跑。

### 本机 5800X（AVX2，4 worker，n=99）

发布前本机复测（`97c1448`，2026-09-19）：HOME-MAIN，Ryzen 7 5800X / 16 逻辑核 / 64 GB，`.NET 10.0.11` 宿主。1.4 走当前树；**1.3 走 NuGet `Sdcb.SimdPaddleOCR` 1.3.0**（当时的临时 `-p:UsePublishedLibrary=1.3.0`，模型包仍是 1.0.0）。同一 `dataset/`、同一烤机（25 张 tiny）再各 100 张 `--warmup 1`。墙钟 n=99；准确率与 ΔWS 按满勤 100 张 / **1036** 行。

| 模型   | 引擎            |                mean |      loaded |           Δ WS |     exact_lines |           CER |
| ------ | --------------- | ------------------: | ----------: | -------------: | --------------: | ------------: |
| tiny   | 本库 net10 AVX2 |  86.0 → **63.1**（0.73×） | 415 → 401 MB | 389 → **113 MB** |        742/1036 |         2.78% |
| tiny   | 本库 ns2        | 203.1 → **96.5**（0.48×） | 417 → 404 MB | 342 → 117 MB |        742/1036 |         2.78% |
| small  | 本库 net10 AVX2 | 222.1 → **200.0**（0.90×） | 512 → 452 MB | 677 → **217 MB** |        950/1036 |         0.60% |
| small  | 本库 ns2        | 432.1 → **303.0**（0.70×） | 525 → 461 MB | 751 → 218 MB |        950/1036 |         0.60% |
| medium | 本库 net10 AVX2 |   628 → **585**（0.93×） | 959 → 699 MB | 1528 → **505 MB** | 1002 → **1004**/1036 | 0.67% → **0.26%** |
| medium | 本库 ns2        |  1606 → **874**（0.54×） | 986 → 726 MB | 1677 → 490 MB | 1002 → **1004**/1036 | 0.67% → **0.26%** |

本库 4w 工作集与阶段（1.3 NuGet 与 1.4 当前树，各烤机后再跑）：

| 模型   | TFM   |           loaded |              peak |            Δ WS |        img/s |          det / rec |
| ------ | ----- | ---------------: | ----------------: | --------------: | -----------: | -----------------: |
| tiny   | net10 | 415 → 401 MB | 804 → **515 MB** | 389 → **113 MB** | 11.6 → 15.8 | 28.9 / 100 → 21.5 / 83 |
| tiny   | ns2   | 417 → 404 MB | 766 → 522 MB | 342 → 117 MB |  4.9 → 10.4 | 111 / 158 → 34.6 / 122 |
| small  | net10 | 512 → 452 MB | 1195 → **674 MB** | 677 → **217 MB** |  4.5 → 5.0 | 89.2 / 385 → 73.4 / 386 |
| small  | ns2   | 525 → 461 MB | 1277 → 684 MB | 751 → 218 MB |  2.3 → 3.3 | 177 / 764 → 123 / 563 |
| medium | net10 | 959 → 699 MB | 2489 → **1206 MB** | 1528 → **505 MB** |  1.6 → 1.7 | 216 / 1428 → 201 / 1361 |
| medium | ns2   | 986 → 726 MB | 2663 → 1218 MB | 1677 → 490 MB |  0.6 → 1.1 | 768 / 2934 → 255 / 2194 |

读法：

- **1.3 本机数字已用 NuGet 1.3.0 同机重测替换**（旧表 87.8 / 237.5 / 640.5 是当时跳过首张、也没跑 ns2）。这次 tiny net10 **86.0** 和旧 87.8 重叠；small / medium 略快（222 / 628），仍是同一量级。
- net10 相对 1.3：**tiny 0.73×、small 0.90×、medium 0.93×**。tiny 吃到预处理直写 NHWC；small / medium 墙钟接近，收益主要在内存。
- **ns2 是 1.4 本机最大墙钟收益**：1.3 ns2 仍是 NCHW `Vector`（相对当时 net10 约 2.0–2.6×）；1.4 ns2 走到 NHWC 后 tiny **0.48×**、small **0.70×**、medium **0.54×**。1.4 的 ns2 / net10 比值收成 **1.50–1.53×**。
- **工作集大约腰斩**：tiny peak 804 → **515 MB**（Δ WS 389 → 113）；small 1195 → **674**；medium 2489 → **1206**。ns2 和 net10 同一量级，1.3 并不更省。
- tiny / small 行精确与 CER 在 1.3 / 1.4、net10 / ns2 之间 **完全相同**（742/1036、950/1036）。medium 1.3 是 1002/1036、CER 0.67%；1.4 是 **1004/1036、0.26%**，和 `1c06dee`（ClsResizeImg keep-aspect）同方向。
- 4 worker 下算子之和可以大于墙钟，只适合看结构。

1.4 JSON：`bench-out/local-5800x-{tiny,small,medium}-4w.json`、`bench-out/local-5800x-ns2-{tiny,small,medium}-4w.json`。1.3 NuGet JSON：`bench-out/local-5800x-v13-net10-{tiny,small,medium}-4w.json`、`bench-out/local-5800x-v13-ns2-{tiny,small,medium}-4w.json`。

#### 同机 lw.PPOCR.C 4w（`20d0de6`）

当前树 harness、`--engine c`、`--c-assets bench-out/c-runtime`，DLL [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)。同一 `dataset/`、先 25 张 tiny 烤机，再各 100 张 `--warmup 1`。C 引擎没有 stage / operator 剖析。

| 模型   |   mean | median |    p95 | img/s |  loaded |        peak |    last |    Δ WS |   exact_lines |   exact_img |       CER |
| ------ | -----: | -----: | -----: | ----: | ------: | ----------: | ------: | ------: | ------------: | ----------: | --------: |
| tiny   |  201.9 |  202.7 |  264.4 |   5.0 | 396 MB |  **541 MB** | 498 MB | 102 MB |      752/1036 |        7/100 |     4.09% |
| small  |  541.7 |  538.5 |  692.5 |   1.8 | 434 MB |  **760 MB** | 705 MB | 271 MB |      936/1036 |       39/100 |     1.08% |
| medium | 2319.5 | 2362.3 | 2924.0 |   0.4 | 585 MB | **1479 MB** | 1359 MB | 774 MB | **1014/1036** |       80/100 | **0.24%** |

JSON：`bench-out/local-5800x-c-{tiny,small,medium}-4w.json`。

### 平台 smoke（20 张，只看覆盖）

median **不能**和 100 张 bench 比。CPU 每次都会变。1.4 [35241030966](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35241030966) 行精确与 1.3 相同。

| RID              | 1.4 这次 CPU | ISA     | 中位 ms |  行精确 |
| ---------------- | ------------ | ------- | ------: | ------: |
| linux-arm64      | N2           | AdvSimd |     227 | 152/208 |
| linux-x64 tiny   | 7763         | AVX2    | 324–352 | 152/208 |
| linux-x64 small  | 7763         | AVX2    |     958 | 188/208 |
| linux-x64 medium | 7763         | AVX2    |    5200 | 200/208 |
| win-x64          | 9V45         | AVX-512 |     138 | 152/208 |
| win-x86          | 9V74         | AVX2    |     406 | 152/208 |
| win-arm64        | Cobalt 100   | AdvSimd |     213 | 152/208 |
| osx-arm64        | M1 Virtual   | AdvSimd |     324 | 152/208 |
| osx-x64          | i7-8700B     | AVX2    |     217 | 152/208 |

small 大约是 tiny 的 3 倍墙钟，medium 大约是 tiny 的 15–17 倍；CER 从 3.7% → 1.6% → 0.34%。

---

## 1.4 工作集（大幅下降）

| 场景                         | 1.3 peak |    1.4 peak | 1.4 Δ WS |
| ---------------------------- | -------: | ----------: | -------: |
| win-x64 7763 sharp tiny 4w   |  ~817 MB | **~515 MB** |  ~107 MB |
| linux-arm64 N2 sharp tiny 4w |  ~840 MB | **~572 MB** |  ~162 MB |
| osx-arm64 sharp tiny 4w      |  ~715 MB |     ~704 MB |  ~289 MB |
| win-x64 lw.PPOCR.C tiny 4w   |  ~586 MB |     ~535 MB |  ~105 MB |

x64 / ARM64 本库 peak 大约少 **300 MB**，主要是 NHWC workspace 别名和不再为输入 `LayoutConvert` 留第二份缓冲。c 仍然更省，但更慢。OpenVINO.NET 1.3 时 tiny 已近 2.6 GB，不再新测。

像素格式不增加整图缓冲：gather 写的是已经要做的双线性 / cubic BGR scratch。

---

## 以后怎么用这份基线

1. **回归判定（x64）**：只看 win-x64 **EPYC 7763** 的 SIMD 套件。`tiny-4w` 中位大约 **180–210 ms**；`tiny-4w-ns2` 同 replica 比值大约 **1.02–1.30**。单次 180–221 都出现过，不要用一份 replica 绝对时间喊回归。
2. **回归判定（ARM64）**：只看 linux-arm64 N2。`tiny-4w` 应在 **175–185 ms**；ns2 比值 **1.63–1.68**。这个平台比 Windows 更适合做自动阈值。
3. **引擎对比**：只用 win-x64 引擎套件、同一 replica。c/sharp 4w 大约 **1.67–1.80**（`20d0de6`）。不要再和 OpenVINO.NET 比新数。
4. **不要**：把 9V74/Xeon 的 AVX-512 和 7763 的 AVX2 写成「优化了 30%」；不要用 osx-arm64 / osx-x64 的绝对时间做门禁；不要拿 20 张 smoke 和 100 张 bench 比快慢。

复现：推送或手动触发 [`.github/workflows/test.yml`](../.github/workflows/test.yml)，下载 `perf-report` artifact。本地同一套数据可用 `test/Sdcb.SimdPaddleOCR.Tests` 的 `--benchmark` / `--summarize`。
