# Install, start watching, and walk the debug boards past the vision, then print what it
# read. The emulator has no chess app, so the boards are the desktop renderer's own
# (tools/make_fixtures.py) shown full screen by the debug activity.
#
#     powershell -File tools\run_boards.ps1 [-Device emulator-5556]

param(
    [string]$Device = "emulator-5556",
    [string[]]$Boards = @("start-green", "italian-green", "start-merida", "italian-alpha", "endgame-blue", "start-brown"),
    [int]$Hold = 5
)

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$apk = "D:\sidecoach-android\app\build\outputs\apk\debug\app-debug.apk"

& $adb -s $Device install -r $apk | Select-Object -Last 1
& $adb -s $Device shell am force-stop com.sidecoach
& $adb -s $Device shell pm grant com.sidecoach android.permission.POST_NOTIFICATIONS
& $adb -s $Device shell appops set com.sidecoach PROJECT_MEDIA allow
& $adb -s $Device logcat -c
& $adb -s $Device shell am start -n com.sidecoach/.MainActivity | Out-Null
Start-Sleep 4

# The Start button, wherever the layout puts it today.
& $adb -s $Device shell uiautomator dump /sdcard/ui.xml | Out-Null
& $adb -s $Device pull /sdcard/ui.xml "$env:TEMP\ui.xml" | Out-Null
$xml = Get-Content "$env:TEMP\ui.xml" -Raw
$hit = [regex]::Match($xml, 'clickable="true"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
if (-not $hit.Success) { "no button on screen; is the app up?"; exit 1 }
$x = ([int]$hit.Groups[1].Value + [int]$hit.Groups[3].Value) / 2
$y = ([int]$hit.Groups[2].Value + [int]$hit.Groups[4].Value) / 2
& $adb -s $Device shell input tap $x $y
Start-Sleep 5

foreach ($board in $Boards) {
    & $adb -s $Device shell am start -n com.sidecoach/.debug.BoardActivity -e board $board | Out-Null
    Start-Sleep $Hold
}

"--- what it read ---"
& $adb -s $Device logcat -d | Select-String "sidecoach\.(vision|capture)" | ForEach-Object { ($_ -split "sidecoach\.")[1] }
