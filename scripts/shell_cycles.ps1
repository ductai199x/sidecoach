# Launch/teardown cycles of the desktop shell (docs/app-plan.md, section 5, step 6): every
# cycle starts the shell, waits for the sidecar's health through the lock file, ends it one
# of three ways, and checks what was left behind.
#
#   quit  the window closed with the shell setting "quit": the shell stops the sidecar
#         and exits; nothing may remain.
#   tray  the window closed with the setting "tray": the shell stays, its window hidden,
#         the sidecar still answering; then the shell is killed and nothing may remain
#         (the sidecar's parent watchdog).
#   kill  the shell killed outright; the watchdog takes the sidecar with it.
#
#   pwsh scripts/shell_cycles.ps1 -Exe desktop/src-tauri/target/release/sidecoach-desktop.exe -Cycles 6
#
# The shell setting is written to shell.json in the app's config directory before each
# launch and restored afterwards. A second launch while a shell runs is answered by the
# running one (single instance), so every cycle first makes sure none is left.
param(
    [string]$Exe = "desktop/src-tauri/target/debug/sidecoach-desktop.exe",
    [int]$Cycles = 6,
    [string]$Lock = "data/panel.lock",
    [string]$ConfigDir = (Join-Path $env:APPDATA "dev.sidecoach.desktop")
)
$ErrorActionPreference = "Stop"
Add-Type @"
using System; using System.Runtime.InteropServices;
public class Win { [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l); [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h); }
"@
$shellJson = Join-Path $ConfigDir "shell.json"
$saved = if (Test-Path $shellJson) { Get-Content $shellJson -Raw } else { $null }
New-Item -ItemType Directory -Force $ConfigDir | Out-Null

function Sidecars { Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like "*sidecoach*panel*" } }
function Shells { Get-Process -Name sidecoach-desktop -ErrorAction SilentlyContinue }
function Clear-Left {
    Shells | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    Sidecars | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 500
}
function Wait-Gone([int]$Port, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $listening = if ($Port) { Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue } else { $null }
        if (-not (Sidecars) -and -not (Shells) -and -not $listening) { return $true }
    }
    return $false
}

$results = @()
$kinds = @("quit", "tray", "kill")
try {
    for ($i = 1; $i -le $Cycles; $i++) {
        $how = $kinds[($i - 1) % $kinds.Count]
        Clear-Left
        Remove-Item $Lock -ErrorAction SilentlyContinue
        Set-Content -Path $shellJson -Value ('{"close":"' + $(if ($how -eq "tray") { "tray" } else { "quit" }) + '","autostart":false}')
        $t0 = Get-Date
        # No splash: the cycles close the main window by its handle, which must be the main one.
        $env:SIDECOACH_NO_SPLASH = "1"
        $p = Start-Process -FilePath $Exe -PassThru
        $port = $null
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 250
            if (Test-Path $Lock) {
                try { $port = (Get-Content $Lock | ConvertFrom-Json).port } catch { $port = $null }
                if ($port) {
                    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$port/api/health").StatusCode -eq 200) { break } } catch {}
                }
            }
        }
        $up = ((Get-Date) - $t0).TotalSeconds
        $healthy = [bool]($port -and ((Get-Date) -lt $deadline))
        Start-Sleep 2
        $p.Refresh()
        $trayOk = $null
        $t1 = Get-Date
        if ($how -eq "kill") {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        } else {
            if ($p.MainWindowHandle -ne 0) { [Win]::PostMessage($p.MainWindowHandle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null } else { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
            if ($how -eq "tray") {
                # The window hides, the shell and the sidecar stay; then the shell is killed.
                Start-Sleep 3
                $alive = -not (Get-Process -Id $p.Id -ErrorAction SilentlyContinue).HasExited
                $hidden = -not [Win]::IsWindowVisible($p.MainWindowHandle)
                $answering = $false
                try { $answering = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$port/api/health").StatusCode -eq 200 } catch {}
                $trayOk = $alive -and $hidden -and $answering
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            }
        }
        $gone = Wait-Gone -Port $port -Seconds 15
        $down = ((Get-Date) - $t1).TotalSeconds
        $results += [pscustomobject]@{ cycle = $i; how = $how; port = $port; up_s = [math]::Round($up, 1); healthy = $healthy; tray = $trayOk; clean = $gone; down_s = [math]::Round($down, 1) }
        "{0,2} {1,-5} port {2,-6} up {3,5}s healthy {4,-5} tray {5,-5} clean {6,-5} down {7,4}s" -f $i, $how, $port, [math]::Round($up, 1), $healthy, $(if ($null -eq $trayOk) { "-" } else { $trayOk }), $gone, [math]::Round($down, 1)
    }
} finally {
    Clear-Left
    if ($null -ne $saved) { Set-Content -Path $shellJson -Value $saved -NoNewline } else { Remove-Item $shellJson -ErrorAction SilentlyContinue }
}
$results | Format-Table -AutoSize
$ok = ($results | Where-Object { $_.healthy -and $_.clean -and ($null -eq $_.tray -or $_.tray) }).Count
"healthy: $(($results | Where-Object healthy).Count)/$Cycles  clean: $(($results | Where-Object clean).Count)/$Cycles  tray: $(($results | Where-Object { $_.tray }).Count)/$(($results | Where-Object { $null -ne $_.tray }).Count)  up p50 $([math]::Round(($results.up_s | Sort-Object)[[int]($Cycles/2)], 1))s  down max $(($results.down_s | Measure-Object -Maximum).Maximum)s"
if ($ok -ne $Cycles) { exit 1 }
