# Shrink every live-session recording under a folder to 720p in place, once a move into
# it has finished (the `done` marker robocopy's wrapper writes). Each session becomes
# `sidecoach shrink <session> <session>-720p` plus its logs copied over, then the original
# goes and the shrunk copy takes its name. Skips sessions already at or under 720p.
#
#   pwsh scripts/shrink_sessions.ps1 -Root C:\Users\Tai\sidecoach-recordings
param(
    [string]$Root = "C:\Users\Tai\sidecoach-recordings",
    [string]$Repo = "D:\chess-coaching-companion",
    [int]$Short = 720
)
$ErrorActionPreference = "Continue"
$log = Join-Path $Root "shrink.log"
$marker = Join-Path $Root "move.log"
while (-not (Test-Path $marker) -or -not (Select-String -Path $marker -Pattern "done" -Quiet)) { Start-Sleep 15 }
"$(Get-Date -Format s) move done; shrinking" | Out-File -Append $log
Set-Location $Repo
foreach ($dir in Get-ChildItem $Root -Directory | Where-Object { $_.Name -match '^\d{8}-\d{6}$' }) {
    $meta = Join-Path $dir.FullName "meta.json"
    if (-not (Test-Path $meta)) { continue }
    $m = Get-Content $meta | ConvertFrom-Json
    $shortSide = [math]::Min($m.width, $m.height)
    if ($shortSide -le $Short) { "$($dir.Name): already ${shortSide}p" | Out-File -Append $log; continue }
    $out = "$($dir.FullName)-720p"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    $t0 = Get-Date
    & uv run --no-sync sidecoach shrink $out $dir.FullName --short $Short 2>&1 | Out-File -Append $log
    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $out "frames.mkv"))) {
        foreach ($f in Get-ChildItem $dir.FullName -File | Where-Object { $_.Name -notin @("frames.mkv", "index.jsonl", "meta.json") }) {
            Copy-Item $f.FullName (Join-Path $out $f.Name) -Force
        }
        $before = (Get-ChildItem $dir.FullName -Recurse -File | Measure-Object Length -Sum).Sum
        $after = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum
        Remove-Item -Recurse -Force $dir.FullName
        Rename-Item $out $dir.Name
        "$($dir.Name): $([math]::Round($before/1MB)) MB -> $([math]::Round($after/1MB)) MB in $([math]::Round(((Get-Date) - $t0).TotalSeconds)) s" | Out-File -Append $log
    } else {
        "$($dir.Name): shrink failed (exit $LASTEXITCODE); original kept" | Out-File -Append $log
        if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    }
}
"$(Get-Date -Format s) all done" | Out-File -Append $log
