# Build the app and install it on one device, then check what landed.
#
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Serial emulator-5554
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Serial 100.111.75.60:5555
#
# The serial is required: never whichever device adb picks. It refuses a version that differs
# from the desktop's (the desktop's scripts\bump_version.py sets both), a device that is absent
# or offline, and one where the capture service runs -- a coaching session is live there, and an
# install kills it -- unless -Force. The `phone` build type is the one the phone runs (gradlew
# :app:installPhone): the debug build's in all but being debuggable, signed with the same debug
# key, so it installs over a debug install and keeps the app's files and settings
# (app/build.gradle.kts). The engine's network is pushed by tools\test.ps1 for the tests, and
# the app fetches its own.
#
# -Voice also puts the faster Kokoro (the desktop's tools\kokoro_fast.py makes it, into its
# data\models\kokoro) in the app's external files, where the voice adopts it at its next load
# (KokoroTts.adoptFast); skipped when the phone holds that file already.

param(
    [Parameter(Mandatory = $true)][string]$Serial,
    [switch]$Force,
    [switch]$Voice
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot

# One version for every platform: a phone build that disagrees with the desktop beside it is
# refused (Tai, 2026-09-25: synced "even though sometimes we update only 1").
$gradle = Get-Content (Join-Path $root "app\build.gradle.kts") -Raw
$phone = [regex]::Match($gradle, 'versionName = "([^"]+)"').Groups[1].Value
$pyproject = Join-Path (Split-Path -Parent $root) "chess-coaching-companion\pyproject.toml"
if (Test-Path $pyproject) {
    $desktop = [regex]::Match((Get-Content $pyproject -Raw), '(?m)^version = "([^"]+)"').Groups[1].Value
    if ($phone -ne $desktop) {
        throw "the phone is $phone and the desktop ${desktop}: run the desktop's scripts\bump_version.py <version>"
    }
} else {
    Write-Host "no desktop repo beside this one: the versions are not checked"
}

$state = ((& $adb -s $Serial get-state 2>&1) -join " ").Trim()
if ($state -ne "device") { throw "$Serial is not connected ($state)" }

$services = (& $adb -s $Serial shell dumpsys activity services com.sidecoach) -join "`n"
if (-not $Force -and $services -match "CaptureService") {
    throw "a coaching session is running on $Serial (CaptureService): not installing over it. Stop it first, or -Force."
}

Write-Host "building and installing $phone on $Serial..."
$env:ANDROID_SERIAL = $Serial
& (Join-Path $root "gradlew.bat") -p $root -q :app:installPhone --console=plain
if ($LASTEXITCODE -ne 0) { throw "the install failed" }

$package = (& $adb -s $Serial shell dumpsys package com.sidecoach) -join "`n"
$version = [regex]::Match($package, "versionName=(\S+)").Groups[1].Value
$updated = [regex]::Match($package, "lastUpdateTime=([^\r\n]+)").Groups[1].Value
$commit = (& git -C $root log --oneline -1)
Write-Host "installed com.sidecoach $version on $Serial at $updated ($commit)"

if ($Voice) {
    $fast = Join-Path (Split-Path -Parent $root) "chess-coaching-companion\data\models\kokoro\kokoro-v1.0-q8.onnx"
    if (-not (Test-Path $fast)) { throw "no $fast: run the desktop's tools\kokoro_fast.py first" }
    $dir = "/sdcard/Android/data/com.sidecoach/files/models/kokoro"
    $size = (Get-Item $fast).Length
    $there = "$(& $adb -s $Serial shell "stat -c %s $dir/kokoro-v1.0-q8.onnx 2>/dev/null || echo none")".Trim()
    if ($there -eq "$size") {
        Write-Host "the faster voice is on $Serial already"
    } else {
        Write-Host "pushing the faster voice ($([int]($size / 1MB)) MB): adopted at the voice's next load"
        & $adb -s $Serial shell "mkdir -p $dir"
        & $adb -s $Serial push $fast "$dir/kokoro-v1.0-q8.onnx" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "the push of the faster voice failed" }
    }
}
