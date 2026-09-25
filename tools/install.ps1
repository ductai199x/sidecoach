# Build the app and install it on one device, then check what landed.
#
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Serial emulator-5554
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Serial 100.111.75.60:5555
#
# The serial is required: never whichever device adb picks. It refuses while the capture
# service runs -- a coaching session is live on that device, and an install kills it -- unless
# -Force. The debug build is the one the phone runs (gradlew :app:installDebug); the engine's
# network is pushed by tools\test.ps1 for the test APK, and the app fetches its own.

param(
    [Parameter(Mandatory = $true)][string]$Serial,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot

$state = ((& $adb -s $Serial get-state 2>&1) -join " ").Trim()
if ($state -ne "device") { throw "$Serial is not connected ($state)" }

$services = (& $adb -s $Serial shell dumpsys activity services com.sidecoach) -join "`n"
if (-not $Force -and $services -match "CaptureService") {
    throw "a coaching session is running on $Serial (CaptureService): not installing over it. Stop it first, or -Force."
}

Write-Host "building and installing on $Serial..."
$env:ANDROID_SERIAL = $Serial
& (Join-Path $root "gradlew.bat") -p $root -q :app:installDebug --console=plain
if ($LASTEXITCODE -ne 0) { throw "the install failed" }

$package = (& $adb -s $Serial shell dumpsys package com.sidecoach) -join "`n"
$version = [regex]::Match($package, "versionName=(\S+)").Groups[1].Value
$updated = [regex]::Match($package, "lastUpdateTime=([^\r\n]+)").Groups[1].Value
$commit = (& git -C $root log --oneline -1)
Write-Host "installed com.sidecoach $version on $Serial at $updated ($commit)"
