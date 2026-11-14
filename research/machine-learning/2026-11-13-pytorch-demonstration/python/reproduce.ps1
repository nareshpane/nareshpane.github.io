# PowerShell: CPU-first reproduction in an isolated temporary environment.
$ErrorActionPreference = 'Stop'
$assets = Split-Path $PSScriptRoot -Parent
$working = Split-Path $assets -Parent
$repository = Split-Path (Split-Path $working -Parent) -Parent
$venv = Join-Path $env:TEMP 'pytorch-monograph-repro'
python --version
python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$py = Join-Path $venv 'Scripts\python.exe'
function Run-Python {
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw 'Python command failed; stop before rebuilding dependent artifacts.' }
}
Run-Python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
# For a GPU environment use https://pytorch.org/get-started/locally/ instead.
Run-Python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
Run-Python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
$env:PYTHONDONTWRITEBYTECODE = '1'
Run-Python (Join-Path $PSScriptRoot 'run_all.py') --output (Join-Path $assets 'outputs\seed-42')
Run-Python (Join-Path $PSScriptRoot 'replicate_experiments.py') --output (Join-Path $assets 'outputs')
Run-Python (Join-Path $PSScriptRoot 'benchmark_experiments.py') --output (Join-Path $assets 'outputs\seed-42')
Run-Python (Join-Path $PSScriptRoot '05_framework_comparisons.py') --output (Join-Path $assets 'outputs\seed-42')
Run-Python (Join-Path $PSScriptRoot 'validate_experiments.py')
Run-Python (Join-Path $PSScriptRoot 'build_monograph.py')
Run-Python (Join-Path $PSScriptRoot 'generate_demo_assets.py')
Run-Python (Join-Path $PSScriptRoot 'validate_page.py')
Write-Host 'Open /research/machine-learning/2026-11-13-pytorch-demonstration.html on the local server below.'
Set-Location -LiteralPath $repository
Run-Python -m http.server 8765 --bind 127.0.0.1
