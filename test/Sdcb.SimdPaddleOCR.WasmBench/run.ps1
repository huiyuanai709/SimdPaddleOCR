[CmdletBinding()]
param(
    [int]$Workers = 4,
    [string]$Model = "tiny",
    [int]$Count = 0,
    [string]$Out = "",
    [switch]$Open,
    [switch]$NoBuild,
    [switch]$NoAot
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir

if (-not $NoBuild) {
    Write-Host "[WasmBench] Building and publishing WebAssembly app (AOT: $(-not $NoAot))..." -ForegroundColor Cyan
    $PublishArgs = @($ProjectDir, "-c", "Release", "--nologo")
    if ($NoAot) {
        $PublishArgs += @("-p:RunAOTCompilation=false")
    }
    dotnet publish @PublishArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to publish WasmBench project."
    }
}

$ServerScript = Join-Path $ProjectDir "server.mjs"
$NodeArgs = @($ServerScript, "--workers", $Workers, "--model", $Model)
if ($Count -gt 0) {
    $NodeArgs += @("--count", $Count)
}
if ($Out -ne "") {
    $NodeArgs += @("--out", $Out)
}
if ($Open) {
    $NodeArgs += @("--open")
}

Write-Host "[WasmBench] Launching benchmark harness..." -ForegroundColor Cyan
& node @NodeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark execution failed with exit code $LASTEXITCODE"
}
