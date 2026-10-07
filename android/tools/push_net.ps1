# Put Stockfish's network where the app can read it, for development.
#
#   python tools/fetch_stockfish.py --net
#   powershell -ExecutionPolicy Bypass -File tools\push_net.ps1
#
# The app will fetch the network itself; this is the shortcut while there is no fetcher, and
# the only way the instrumented engine test can find one. It goes through /data/local/tmp
# because a file adb pushes into the app's own directories belongs to the shell, and the app
# cannot read it; `run-as` copies it across as the app.

param(
    [string]$Serial = "emulator-5556",
    [string]$Package = "com.sidecoach"
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot
$net = Get-ChildItem (Join-Path $root "third_party") -Filter "*.nnue" | Select-Object -First 1
if (-not $net) { throw "no .nnue in third_party; run tools/fetch_stockfish.py --net" }

$name = $net.Name
Write-Host "pushing $name ($([int]($net.Length / 1MB)) MB)..."
& $adb -s $Serial push $net.FullName "/data/local/tmp/$name" | Out-Null
& $adb -s $Serial shell chmod 644 "/data/local/tmp/$name"
# The app may not have run yet, so its files directory may not exist.
& $adb -s $Serial shell "run-as $Package mkdir -p files"
& $adb -s $Serial shell "run-as $Package sh -c 'cat /data/local/tmp/$name > files/$name'"
& $adb -s $Serial shell rm "/data/local/tmp/$name"
$size = & $adb -s $Serial shell "run-as $Package stat -c %s files/$name"
Write-Host "in place: files/$name ($size bytes)"
