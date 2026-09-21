# Launch and quit the desktop shell N times and check that nothing is left behind
# (docs/app-plan.md, section 4, "Lifecycle"): after each quit no sidecar python is
# running, the announced port is free, and the next launch comes up on its own.
#
#   pwsh scripts/shell_cycles.ps1 -Exe desktop/src-tauri/target/debug/sidecoach-desktop.exe -Cycles 20
#
# The shell is ended with a forced kill on even cycles (the parent-pid watchdog must
# take the sidecar down) and with WM_CLOSE on odd ones (the shell's own exit path).
param(
    [string]$Exe = "desktop/src-tauri/target/debug/sidecoach-desktop.exe",
    [int]$Cycles = 20,
    [string]$Lock = "data/panel.lock"
)
$ErrorActionPreference = "Stop"
Add-Type @"
using System; using System.Runtime.InteropServices;
public class Win { [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l); }
"@
$results = @()
for ($i = 1; $i -le $Cycles; $i++) {
    Remove-Item $Lock -ErrorAction SilentlyContinue
    $t0 = Get-Date
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
    $healthy = $port -and ((Get-Date) -lt $deadline)
    Start-Sleep 2
    $how = if ($i % 2 -eq 0) { "kill" } else { "close" }
    if ($how -eq "kill") {
        Stop-Process -Id $p.Id -Force
    } else {
        $p.Refresh()
        if ($p.MainWindowHandle -ne 0) { [Win]::PostMessage($p.MainWindowHandle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null } else { Stop-Process -Id $p.Id -Force }
    }
    $t1 = Get-Date
    $gone = $false
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $python = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like "*sidecoach*panel*" }
        $shell = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
        $listening = if ($port) { Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue } else { $null }
        if (-not $python -and -not $shell -and -not $listening) { $gone = $true; break }
    }
    $down = ((Get-Date) - $t1).TotalSeconds
    $results += [pscustomobject]@{ cycle = $i; how = $how; port = $port; up_s = [math]::Round($up, 1); healthy = $healthy; clean = $gone; down_s = [math]::Round($down, 1) }
    "{0,2} {1,-5} port {2,-6} up {3,5}s healthy {4,-5} clean {5,-5} down {6,4}s" -f $i, $how, $port, [math]::Round($up, 1), $healthy, $gone, [math]::Round($down, 1)
}
$results | Format-Table -AutoSize
"healthy: $(($results | Where-Object healthy).Count)/$Cycles  clean: $(($results | Where-Object clean).Count)/$Cycles  up p50 $([math]::Round(($results.up_s | Sort-Object)[[int]($Cycles/2)], 1))s  down max $(($results.down_s | Measure-Object -Maximum).Maximum)s"
