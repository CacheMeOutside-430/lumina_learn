$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "C:\Users\shehe\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
}
& $python (Join-Path $root "scripts\generate_seed_screenshots.py") --output (Join-Path $root "data\screenshots") --count 12 --clean
