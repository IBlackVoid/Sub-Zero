param(
  [string]$EnginePath    = "Pixel-Ripper_V16/ascii_engine.exe",
  [string]$MappingPath   = "scripts/tui/state_mapping.json",
  [string]$OutDir        = "assets/ascii",
  [string]$FallbackImage = "Pixel-Ripper_V16/Master.png",
  [switch]$Force,
  [switch]$NoFallback
)

# Pipeline: clips on disk  →  ascii_engine.exe  →  assets/ascii/<state>.jsonl
#
# Reads scripts/tui/state_mapping.json to learn which source file feeds
# which TUI state. Sources are looked up by stem (no extension) across
# every search_dirs entry; the first match wins.
#
# Re-run with -Force to overwrite an existing pack.
# Pass -NoFallback to fail loudly when a mapped source is missing.

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

# --- Resolve paths and toolchain ---
$enginePath = Resolve-Path $EnginePath -ErrorAction Stop
$mappingAbs = Resolve-Path $MappingPath -ErrorAction Stop
$outDirAbs  = Join-Path $repoRoot $OutDir
$null = New-Item -ItemType Directory -Force -Path $outDirAbs

$mingwPaths = @("C:\msys64\mingw64\bin", "C:\msys64\usr\bin") | Where-Object { Test-Path $_ }
if ($mingwPaths) {
  $env:PATH = ($mingwPaths -join ';') + ';' + $env:PATH
}

# --- Parse mapping ---
$mapping = Get-Content $mappingAbs -Raw | ConvertFrom-Json
$searchDirs = @($mapping.search_dirs)
$extensions = @($mapping.extensions)
$width      = [int]$mapping.output_width

function Find-SourceFile {
  param([string]$Stem)
  foreach ($dir in $searchDirs) {
    $abs = Join-Path $repoRoot $dir
    if (-not (Test-Path $abs)) { continue }
    foreach ($ext in $extensions) {
      $candidate = Join-Path $abs ($Stem + $ext)
      if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
  }
  return $null
}

function Convert-OneClip {
  param(
    [string]$InputPath,
    [string]$OutPath,
    [double]$Brightness,
    [double]$Saturate,
    [double]$Edge
  )
  if (-not $Force -and (Test-Path $OutPath)) {
    Write-Host ("skip: {0,-30} (use -Force to overwrite)" -f (Split-Path -Leaf $OutPath))
    return
  }
  $args = @(
    $InputPath,
    "--width", $width,
    "--output-format", "cells",
    "--output", $OutPath,
    "--brightness", $Brightness,
    "--saturate", $Saturate,
    "--edge", $Edge
  )
  Write-Host ("convert: {0,-40}  ->  {1}" -f (Split-Path -Leaf $InputPath), (Split-Path -Leaf $OutPath))
  & $enginePath @args | Out-Null
  if (-not (Test-Path $OutPath) -or (Get-Item $OutPath).Length -eq 0) {
    Write-Warning "no output produced for $InputPath"
  }
}

# --- Process each state ---
$mapping.states.PSObject.Properties | ForEach-Object {
  $stateName = $_.Name
  $entry     = $_.Value
  $stem      = $entry.source
  $out       = Join-Path $outDirAbs ("{0}.jsonl" -f $stateName)

  $src = Find-SourceFile $stem
  if (-not $src) {
    if ($NoFallback) {
      throw "missing source for state '$stateName' (stem '$stem' not found in: $($searchDirs -join ', '))"
    }
    $fallbackAbs = Join-Path $repoRoot $FallbackImage
    if (Test-Path $fallbackAbs) {
      Write-Warning ("source '{0}' missing; falling back to {1}" -f $stem, $FallbackImage)
      $src = (Resolve-Path $fallbackAbs).Path
    } else {
      Write-Warning ("source '{0}' missing and fallback {1} also missing; skipping" -f $stem, $FallbackImage)
      return
    }
  }

  Convert-OneClip -InputPath $src -OutPath $out `
                  -Brightness ([double]$entry.brightness) `
                  -Saturate   ([double]$entry.saturate) `
                  -Edge       ([double]$entry.edge)
}

Write-Host ""
Write-Host "animation pack contents:"
Get-ChildItem $outDirAbs -Filter *.jsonl | ForEach-Object {
  $frames = (Get-Content $_.FullName | Measure-Object -Line).Lines
  Write-Host ("  {0,-20} {1,8} bytes  {2,4} frames" -f $_.Name, $_.Length, $frames)
}
