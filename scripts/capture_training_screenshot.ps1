param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("coding","reading","writing","video","browser","messaging","gaming","idle","unknown")]
  [string]$Activity,

  [Parameter(Mandatory=$true)]
  [ValidateSet("programming","math","science","language","research","exam_prep","note_taking","general")]
  [string]$Education
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}
& $python (Join-Path $root "scripts\capture_training_screenshot.py") --activity $Activity --education $Education --output (Join-Path $root "data\screenshots")
