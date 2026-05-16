param(
  [int]$Epochs = 12,
  [int]$BatchSize = 16,
  [string]$Source = "data\screenshots",
  [string]$Output = "models",
  [int]$Seed = 20260515,
  [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  throw "Python virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
}

$env:PYTHONPATH = @(
  (Join-Path $root "packages\shared\src"),
  (Join-Path $root "packages\activity-classifier\src")
) -join ";"

$manifestDir = Join-Path $root "data\manifests"
$sourcePath = Join-Path $root $Source
$outputPath = Join-Path $root $Output

& $python -m activity_classifier.prepare_dataset --source $sourcePath --output $manifestDir --val-ratio 0.15
$trainerArgs = @(
  "-m", "activity_classifier.trainer",
  "--manifest", (Join-Path $manifestDir "train.jsonl"),
  "--val-manifest", (Join-Path $manifestDir "val.jsonl"),
  "--output", $outputPath,
  "--epochs", $Epochs,
  "--batch-size", $BatchSize,
  "--seed", $Seed
)
if ($ResumeCheckpoint) {
  $trainerArgs += @("--resume-checkpoint", $ResumeCheckpoint)
}
& $python @trainerArgs
