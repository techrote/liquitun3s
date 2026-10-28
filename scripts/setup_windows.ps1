$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    if ($env:PYTHON) {
        & $env:PYTHON -m venv .venv
    } else {
        & py -3.11 -m venv .venv
    }
}

$Py = Join-Path $PWD ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip setuptools wheel

Write-Host ""
Write-Host "Installing PyTorch/torchvision first."
Write-Host "If this machine needs a specific CUDA wheel, set PYTORCH_INDEX_URL before running."
Write-Host "Example: set it to the current CUDA wheel URL from pytorch.org."
Write-Host ""

if ($env:PYTORCH_INDEX_URL) {
    & $Py -m pip install torch torchvision --index-url $env:PYTORCH_INDEX_URL
} else {
    & $Py -m pip install torch torchvision
}

& $Py -m pip install -e ".[train,dev]"

Write-Host ""
Write-Host "=== Titan/CUDA preflight ==="
& $Py -m liquitun3s.cli preflight
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed. Fix the CUDA-enabled PyTorch installation before training."
}

Write-Host ""
Write-Host "=== Verify pinned primary VLM source ==="
& $Py -m liquitun3s.cli source vl
if ($LASTEXITCODE -ne 0) {
    throw "LFM3-VL source verification failed."
}

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then try: liquitun3s models"
