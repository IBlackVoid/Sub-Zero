param(
  [string]$StreamsDir = "datasets/streams",
  [string]$CasesOutDir = "benchmarks/cases/generated/streams",
  [string]$CasesDir = "benchmarks/cases/generated/streams",
  [string]$RunDirBase = "benchmarks/runs",
  [string]$ReportsDir = "benchmarks/reports",
  [double]$TimeoutSecs = 3600,
  [int]$MaxCases = 0,
  [string]$PyWhisperModel = "",
  [switch]$TrainGate,
  [ValidateSet("pass","ranking_top1","ranking_topk")]
  [string]$GateLabel = "ranking_top1",
  [int]$GateTopK = 1,
  [switch]$KeepGoing
)

$ErrorActionPreference = "Stop"

function Resolve-Py {
  $venv = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
  if (Test-Path $venv) { return (Resolve-Path $venv).Path }
  return "python"
}

$py = Resolve-Py
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if ($PyWhisperModel) {
  # Forces Sub-Zero's python-whisper backend to avoid accidentally selecting a huge model
  # in strict retries (e.g. large-v3) on 8GB VRAM GPUs.
  $env:SUB_ZERO_PYWHISPER_MODEL = $PyWhisperModel
  $env:SUB_ZERO_PYWHISPER_MODEL_DIR = (Resolve-Path (Join-Path $repoRoot "models\\whisper")).Path
  Write-Host "Pinned python-whisper model: $PyWhisperModel"
}

if ($MaxCases -and $MaxCases -gt 0) {
  # If caller didn't explicitly set output/input dirs, isolate sample runs so we don't
  # accidentally run the entire previously-generated corpus.
  if (-not $PSBoundParameters.ContainsKey("CasesOutDir")) {
    $CasesOutDir = "benchmarks/cases/generated/streams_sample_max$MaxCases"
  }
  if (-not $PSBoundParameters.ContainsKey("CasesDir")) {
    $CasesDir = $CasesOutDir
  }
}

Write-Host "Generating cases from $StreamsDir -> $CasesOutDir"
$genArgs = @(
  (Join-Path $repoRoot "scripts\bench\generate_stream_cases.py"),
  "--streams-dir", $StreamsDir,
  "--out-dir", $CasesOutDir
)
if ($MaxCases -and $MaxCases -gt 0) { $genArgs += @("--max", "$MaxCases") }
& $py @genArgs

Write-Host "Running corpus from $CasesDir (timeout=$TimeoutSecs sec)"
$args = @(
  (Join-Path $repoRoot "scripts\bench\run_corpus.py"),
  "--cases-dir", $CasesDir,
  "--run-dir-base", $RunDirBase,
  "--reports-dir", $ReportsDir,
  "--timeout-secs", "$TimeoutSecs"
)

if ($KeepGoing) { $args += "--keep-going" }
if ($TrainGate) {
  $args += "--train-gate"
  $args += "--gate-label"; $args += $GateLabel
  $args += "--gate-top-k"; $args += "$GateTopK"
}

& $py @args
