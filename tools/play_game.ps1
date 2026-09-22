# Play a rendered game on the device and read the log the watcher writes.
#
#   uv run --no-sync python tools/make_game.py     (from the desktop repo)
#   powershell -ExecutionPolicy Bypass -File tools\play_game.ps1
#
# Installs the app, grants what it needs, taps Start (the consent dialog included), plays
# the game from the debug APK's own assets, and prints what the watcher made of it.

param(
    [string]$Serial = "emulator-5556",
    [int]$Hold = 2500,
    [int]$Lead = 25000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot
$frames = Join-Path $root "app/src/debug/assets/game"

if (-not (Test-Path $frames)) { throw "no frames in $frames; run tools/make_game.py first" }

if (-not $SkipInstall) {
    Write-Host "installing..."
    & (Join-Path $root "gradlew.bat") -p $root :app:installDebug | Out-Null
}
& $adb -s $Serial shell pm grant com.sidecoach android.permission.POST_NOTIFICATIONS 2>$null

& $adb -s $Serial logcat -c
& $adb -s $Serial shell am force-stop com.sidecoach
& $adb -s $Serial shell am start -n com.sidecoach/.MainActivity | Out-Null
Start-Sleep -Seconds 3

# Start watching: the button, then the system's consent dialog.
function Tap-Text([string]$pattern, [int]$Tries = 1) {
    # The panel is a WebView: for the first seconds of a cold start its accessibility tree
    # is empty, so a button that is certainly coming is worth waiting for.
    $match = $null
    for ($try = 0; $try -lt $Tries; $try++) {
        if ($try -gt 0) { Start-Sleep -Seconds 2 }
        & $adb -s $Serial shell uiautomator dump /sdcard/ui.xml | Out-Null
        $xml = (& $adb -s $Serial shell cat /sdcard/ui.xml) -join ""
        $match = [regex]::Match($xml, "text=`"[^`"]*$pattern[^`"]*`"[^>]*bounds=`"\[(\d+),(\d+)\]\[(\d+),(\d+)\]`"")
        if (-not $match.Success) {
            $match = [regex]::Match($xml, "bounds=`"\[(\d+),(\d+)\]\[(\d+),(\d+)\]`"[^>]*text=`"[^`"]*$pattern[^`"]*`"")
        }
        if ($match.Success) { break }
    }
    if (-not $match.Success) { Write-Host "  (no '$pattern' on screen)"; return $false }
    $x = ([int]$match.Groups[1].Value + [int]$match.Groups[3].Value) / 2
    $y = ([int]$match.Groups[2].Value + [int]$match.Groups[4].Value) / 2
    & $adb -s $Serial shell input tap $x $y | Out-Null
    return $true
}

Write-Host "starting the watcher..."
Tap-Text "Start" -Tries 10 | Out-Null   # the panel page's own button
Start-Sleep -Seconds 2
foreach ($label in @("Entire screen", "Start now", "START NOW", "Allow")) {
    if (Tap-Text $label) { Start-Sleep -Seconds 1 }
}
Start-Sleep -Seconds 2

Write-Host "playing the game..."
& $adb -s $Serial shell am start -n com.sidecoach/.debug.BoardActivity -e game game --ei hold $Hold --ei lead $Lead | Out-Null

$frameCount = (Get-ChildItem $frames -Filter *.png).Count
Start-Sleep -Seconds ([int]($Lead / 1000 + ($frameCount + 2) * $Hold / 1000))

Write-Host ""
Write-Host "--- what the watcher saw ---"
& $adb -s $Serial logcat -d "sidecoach.game:I" "sidecoach.capture:I" "sidecoach.debug:I" "*:S"
