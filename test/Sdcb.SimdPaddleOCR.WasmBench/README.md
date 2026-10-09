# Sdcb.SimdPaddleOCR.WasmBench

WebAssembly (WASM) Benchmark Harness for `Sdcb.SimdPaddleOCR`.

Runs PP-OCR inference in a multi-threaded WebAssembly environment (`browser-wasm` targeting .NET 10.0), measuring end-to-end performance, stage breakdowns, and recognition accuracy on the dataset.

---

## Why a Dedicated Harness is Needed for Tiny 4w

In `Sdcb.SimdPaddleOCR`, `--workers 4` runs 4 line workers concurrently (`Parallel.For` over REC sessions):
1. **Multi-threaded WebAssembly**: .NET requires Web Workers and `SharedArrayBuffer` when `<WasmEnableThreads>true</WasmEnableThreads>` is active.
2. **COOP / COEP Headers**: Browsers restrict `SharedArrayBuffer` unless the web server serves the page with:
   - `Cross-Origin-Opener-Policy: same-origin`
   - `Cross-Origin-Embedder-Policy: require-corp`
3. **Headless Execution**: A local Node.js server (`server.mjs`) serves the WASM artifacts and dataset with required security headers and drives Microsoft Edge in headless mode, collecting results and saving standard benchmark JSON.

---

## Quick Start

### 1. Run via PowerShell
```powershell
# Run 10 samples smoke benchmark with 4 workers:
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model tiny -Count 10

# Run full 100-sample benchmark:
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Model tiny -Count 100 -Out bench-out/wasm-tiny-4w.json

# Open interactive browser dashboard:
.\test\Sdcb.SimdPaddleOCR.WasmBench\run.ps1 -Workers 4 -Open
```

### 2. Run via Node.js
```bash
# Build the WASM bundle
dotnet publish test/Sdcb.SimdPaddleOCR.WasmBench -c Release

# Run benchmark in Headless Edge
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model tiny --count 100 --out bench-out/wasm-tiny-4w.json
```

---

## Command Line Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--workers` | int | `4` | Number of line recognizer workers. In WASM with threads, spawns Web Workers up to `navigator.hardwareConcurrency`. |
| `--model` | string | `tiny` | OCR Model to benchmark: `tiny` (ChineseV6Tiny) or `small` (ChineseV6Small). |
| `--count` | int | `0` | Number of dataset images to run (`0` = run all available). |
| `--out` | string | `bench-out/wasm-{model}-{workers}w.json` | Output JSON result path. |
| `--open` | switch | `false` | Open in visible browser window instead of headless mode. |
