param(
  [int]$CueCount = 2000,
  [int]$Iterations = 3,
  [string]$OutputDir = "",
  [switch]$Release = $true
)

$ErrorActionPreference = "Stop"

function Format-SrtTimestamp {
  param([double]$Seconds)
  if (-not [double]::IsFinite($Seconds) -or $Seconds -lt 0) {
    $Seconds = 0
  }
  $totalMs = [long][Math]::Round($Seconds * 1000.0)
  $ms = $totalMs % 1000
  $totalS = [long]($totalMs / 1000)
  $s = $totalS % 60
  $totalM = [long]($totalS / 60)
  $m = $totalM % 60
  $h = [long]($totalM / 60)
  return ("{0:00}:{1:00}:{2:00},{3:000}" -f $h, $m, $s, $ms)
}

function New-SyntheticSrt {
  param(
    [string]$Path,
    [int]$CueCount
  )

  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $CueCount; $i++) {
    $index = $i + 1
    $start = [double]($i * 2)
    $end = $start + 1.0
    $timing = "$(Format-SrtTimestamp $start) --> $(Format-SrtTimestamp $end)"
    $text = if (($i % 2) -eq 0) { "こんにちは" } else { "ありがとう" }
    [void]$sb.AppendLine($index)
    [void]$sb.AppendLine($timing)
    [void]$sb.AppendLine($text)
    [void]$sb.AppendLine("")
  }
  [System.IO.File]::WriteAllText($Path, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $repoRoot ("benchmarks\reports\local_trace_bench_{0}" -f $stamp)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host ("trace bench: cues={0} iterations={1} out={2}" -f $CueCount, $Iterations, $OutputDir)

if ($Release) {
  Write-Host "building release..."
  cargo build --release | Out-Host
}

$exe = Join-Path $repoRoot "target\release\voidex.exe"
$useExe = Test-Path $exe
if (-not $useExe) {
  Write-Host "note: release exe not found; falling back to cargo run --release"
}

for ($iter = 1; $iter -le $Iterations; $iter++) {
  $caseStem = "trace_case_{0:00}" -f $iter
  $input = Join-Path $OutputDir ("{0}.srt" -f $caseStem)
  New-SyntheticSrt -Path $input -CueCount $CueCount

  Write-Host ("run {0}/{1}: {2}" -f $iter, $Iterations, $input)

  if ($useExe) {
    & $exe $input --offline --phrase-table --no-doom-qlock --trace-runtime | Out-Host
  } else {
    cargo run --release -- $input --offline --phrase-table --no-doom-qlock --trace-runtime | Out-Host
  }
}

$traceFiles = Get-ChildItem -Path $OutputDir -Filter "*.voidex.trace.json" | Select-Object -ExpandProperty FullName
if ($traceFiles.Count -eq 0) {
  throw "no trace files were produced in $OutputDir"
}

Write-Host "summarizing traces..."
$summary = Join-Path $OutputDir "trace_summary.txt"
python scripts\bench\summarize_trace.py @traceFiles | Tee-Object -FilePath $summary | Out-Host
Write-Host ("wrote {0}" -f $summary)

