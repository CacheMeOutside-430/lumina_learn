$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = @(
  "$root\apps\backend\src",
  "$root\packages\shared\src",
  "$root\packages\ai-core\src",
  "$root\packages\vision\src",
  "$root\packages\memory\src",
  "$root\packages\activity-classifier\src"
) -join ";"
uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload
