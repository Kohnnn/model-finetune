$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    python -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
& $pythonExe -m pip install --upgrade `
    "git+https://github.com/unslothai/unsloth.git" `
    "git+https://github.com/unslothai/unsloth-zoo.git"

Write-Host "GPU fine-tuning environment is ready at $venvPath"
