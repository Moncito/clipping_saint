# One-time setup for clipper on Windows. Run from this folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
Write-Host "== clipper setup ==" -ForegroundColor Cyan

# 1. Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Installing Python 3.11 via winget..."
    winget install -e --id Python.Python.3.11
    Write-Host "Close and reopen PowerShell, then run setup.ps1 again." -ForegroundColor Yellow
    exit 1
}
python --version

# 2. ffmpeg (needs NVENC support - the Gyan build has it)
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing ffmpeg via winget..."
    winget install -e --id Gyan.FFmpeg
    Write-Host "Close and reopen PowerShell so ffmpeg is on PATH, then run setup.ps1 again." -ForegroundColor Yellow
    exit 1
}
ffmpeg -hide_banner -encoders 2>$null | Select-String "h264_nvenc"

# 3. Virtual environment + packages
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. GPU check
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} else {
    Write-Host "nvidia-smi not found - update your NVIDIA driver (GeForce Experience / nvidia.com)." -ForegroundColor Yellow
}

# 5. Optional helpers
if (-not (Get-Command deno -ErrorAction SilentlyContinue)) {
    Write-Host "Tip: if YouTube downloads fail, install Deno (yt-dlp uses it for YouTube):  winget install -e --id DenoLand.Deno" -ForegroundColor DarkGray
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Optional: Ollama not on PATH - only needed for --picker ollama." -ForegroundColor DarkGray
}
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host "Optional: 'claude' CLI not on PATH - needed for --picker claude (best clip picks)." -ForegroundColor DarkGray
}

Write-Host "`nDone. Test it with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe clip.py run <youtube-url> --count 5"
