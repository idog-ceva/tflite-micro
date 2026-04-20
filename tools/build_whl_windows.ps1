<#
.SYNOPSIS
    Builds a tflite-micro Python wheel for Windows x64.

.DESCRIPTION
    Standalone script that configures the environment, validates prerequisites,
    runs bazel clean --expunge, and builds a cp312-cp312-win_amd64 wheel.

    The output wheel is placed under:
        bazel-bin/python/tflite_micro/whl_dist/

    Prerequisites (install via Visual Studio 2022 Installer):
      - MSVC v143 C++ x64/x86 build tools
      - C++ Clang Compiler for Windows (clang-cl)
      - C++ Clang-cl for v143 build tools (x64/x86)
      - Git for Windows  (provides bash.exe)
      - Python 3.12      (default: C:\Python\Python312)
      - Bazel             (version must match .bazelversion)

.PARAMETER PythonVersion
    Python version tag: cp310, cp311, cp312, cp313.  Default: cp312

.PARAMETER LlvmPath
    Path to the LLVM toolchain shipped with Visual Studio.
    Default: C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\Llvm

.PARAMETER BazelSh
    Path to bash.exe used by Bazel.
    Default: C:\Program Files\Git\bin\bash.exe

.PARAMETER SkipClean
    Skip 'bazel clean --expunge' to speed up incremental builds.

.EXAMPLE
    .\tools\build_whl_windows.ps1
    .\tools\build_whl_windows.ps1 -PythonVersion cp313
    .\tools\build_whl_windows.ps1 -SkipClean
#>

[CmdletBinding()]
param(
    [ValidateSet("cp310", "cp311", "cp312", "cp313")]
    [string]$PythonVersion = "cp312",

    [string]$LlvmPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\Llvm",

    [string]$BazelSh = "C:\Program Files\Git\bin\bash.exe",

    [switch]$SkipClean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: write a coloured status line
# ---------------------------------------------------------------------------
function Write-Step([string]$msg) {
    Write-Host "`n>>> $msg" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. Locate the repo root (script lives in tools/)
# ---------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $RepoRoot
Write-Step "Repository root: $RepoRoot"

try {
    # -----------------------------------------------------------------------
    # 2. Validate prerequisites
    # -----------------------------------------------------------------------
    Write-Step "Validating prerequisites"

    # LLVM / clang-cl
    if (-not (Test-Path "$LlvmPath\bin\clang-cl.exe")) {
        throw "clang-cl.exe not found at $LlvmPath\bin\clang-cl.exe.  Install the C++ Clang tools via Visual Studio Installer or pass -LlvmPath."
    }

    # bash.exe (Bazel needs it for genrules)
    if (-not (Test-Path $BazelSh)) {
        throw "bash.exe not found at $BazelSh.  Install Git for Windows or pass -BazelSh."
    }

    # Bazelisk (automatically downloads the Bazel version from .bazelversion)
    $bazelCmd = if (Get-Command bazelisk -ErrorAction SilentlyContinue) { "bazelisk" }
                elseif (Get-Command bazel   -ErrorAction SilentlyContinue) { "bazel" }
                else { throw "Neither bazelisk nor bazel found on PATH.  Install Bazelisk and ensure it is in your PATH." }

    $expectedVersion = (Get-Content "$RepoRoot\.bazelversion" -First 1).Trim()

    Write-Host "  clang-cl  : $LlvmPath\bin\clang-cl.exe" -ForegroundColor Green
    Write-Host "  bash.exe  : $BazelSh" -ForegroundColor Green
    Write-Host "  bazel cmd : $bazelCmd (will use version $expectedVersion from .bazelversion)" -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 3. Configure environment variables
    # -----------------------------------------------------------------------
    Write-Step "Configuring environment for x64 clang-cl build"

    $env:BAZEL_LLVM = $LlvmPath
    $env:BAZEL_SH   = $BazelSh
    $env:PATH        = "$LlvmPath\bin;$env:PATH"

    # Remove stale MSVC flags that could force 32-bit compilation
    Remove-Item Env:\_CL_ -ErrorAction SilentlyContinue
    Remove-Item Env:\CL   -ErrorAction SilentlyContinue

    # Ensure MSVC arch variables target x64
    $env:VSCMD_ARG_TGT_ARCH  = "x64"
    $env:VSCMD_ARG_HOST_ARCH = "x64"

    # -----------------------------------------------------------------------
    # 4. Clean (unless -SkipClean)
    # -----------------------------------------------------------------------
    if (-not $SkipClean) {
        Write-Step "Running bazel clean --expunge"
        & $bazelCmd --host_jvm_args=-Djavax.net.ssl.trustStoreType=Windows-ROOT clean --expunge
        if ($LASTEXITCODE -ne 0) { throw "bazel clean failed (exit code $LASTEXITCODE)" }
    } else {
        Write-Step "Skipping bazel clean (incremental build)"
    }

    # -----------------------------------------------------------------------
    # 5. Build the wheel
    # -----------------------------------------------------------------------
    $CompatTag = "${PythonVersion}_${PythonVersion}_win_amd64"
    Write-Step "Building wheel with compatibility tag: $CompatTag"

    $LlvmPathFwd = $LlvmPath -replace '\\', '/'

    # Bazel's embedded JVM does not use the Windows certificate store by
    # default, which causes TLS failures behind corporate proxies.  Tell the
    # JVM to use the Windows-ROOT keystore so it trusts the same CAs that
    # PowerShell / browsers trust.
    $startupArgs = @(
        "--host_jvm_args=-Djavax.net.ssl.trustStoreType=Windows-ROOT"
    )

    $bazelArgs = @(
        "build"
        "//python/tflite_micro:whl.dist"

        # Toolchain
        "--config=clangcl"
        "--incompatible_enable_cc_toolchain_resolution=false"

        # Wheel platform tag
        "--//python/tflite_micro:compatibility_tag=$CompatTag"

        # LLVM paths for action sandboxes
        "--action_env=BAZEL_LLVM=$LlvmPathFwd"
        "--action_env=PATH=$LlvmPathFwd/bin;$env:PATH"

        # C++ standard conformance (MSVC/clang-cl compatible)
        "--cxxopt=/Zc:__cplusplus"

        # Linker: force 64-bit output
        "--linkopt=/MACHINE:X64"
        "--host_linkopt=/MACHINE:X64"

        # Disable werror feature (Bazel feature flag, compiler-agnostic)
        "--features=-werror"

        # Stamping (embeds version info into the wheel)
        "--stamp"
        "--workspace_status_command=powershell -NoProfile -ExecutionPolicy Bypass -File tools/workspace_status.ps1"
    )

    Write-Host "$bazelCmd $($startupArgs -join ' ') $($bazelArgs -join ' ')" -ForegroundColor DarkGray
    & $bazelCmd @startupArgs @bazelArgs
    if ($LASTEXITCODE -ne 0) { throw "bazel build failed (exit code $LASTEXITCODE)" }

    # -----------------------------------------------------------------------
    # 6. Report result
    # -----------------------------------------------------------------------
    $whlDir = Join-Path $RepoRoot "bazel-bin\python\tflite_micro\whl_dist"
    Write-Step "Build succeeded!"
    if (Test-Path $whlDir) {
        Write-Host "Wheel(s) in $whlDir`:" -ForegroundColor Green
        Get-ChildItem $whlDir -Filter "*.whl" | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
    } else {
        Write-Warning "Expected output directory not found: $whlDir"
    }
}
finally {
    Pop-Location
}
