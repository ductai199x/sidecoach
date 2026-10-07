# Start watching from the panel as it stands and play the rendered game, without restarting
# the app: for what a running app does with a Start -- the kept engine, the book fetched
# after the warm-up -- which play_game.ps1, restarting the app first, cannot show.
#
#   powershell -ExecutionPolicy Bypass -File tools\watch_game.ps1

param(
    [string]$Serial = "emulator-5556",
    [int]$Hold = 1300,
    [int]$Lead = 8000
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

function Tap-Exact([string]$label, [int]$Tries = 10) {
    for ($try = 0; $try -lt $Tries; $try++) {
        if ($try -gt 0) { Start-Sleep -Seconds 1 }
        & $adb -s $Serial shell uiautomator dump /sdcard/ui.xml | Out-Null
        $xml = (& $adb -s $Serial shell cat /sdcard/ui.xml) -join ""
        $m = [regex]::Match($xml, "text=`"$label`"[^>]*bounds=`"\[(\d+),(\d+)\]\[(\d+),(\d+)\]`"")
        if ($m.Success) {
            $x = ([int]$m.Groups[1].Value + [int]$m.Groups[3].Value) / 2
            $y = ([int]$m.Groups[2].Value + [int]$m.Groups[4].Value) / 2
            & $adb -s $Serial shell input tap $x $y | Out-Null
            return $true
        }
    }
    return $false
}

& $adb -s $Serial logcat -c
if (-not (Tap-Exact "Start")) { throw "no Start button on the panel" }
Start-Sleep -Seconds 2
foreach ($label in @("Entire screen", "Start now")) { Tap-Exact $label 2 | Out-Null }
Start-Sleep -Seconds 2
& $adb -s $Serial shell am start -n com.sidecoach/.debug.BoardActivity -e game game --ei hold $Hold --ei lead $Lead | Out-Null
do {
    Start-Sleep -Seconds 3
    $over = & $adb -s $Serial logcat -d "sidecoach.debug:I" "*:S" | Select-String "the game is over"
} until ($over)
& $adb -s $Serial shell input keyevent 4
Start-Sleep -Seconds 3
Write-Host "played; the panel is in front"
