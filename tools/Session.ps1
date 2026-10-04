# What every script here that builds or talks to a device does first and last, dot-sourced
# (`. (Join-Path $PSScriptRoot "Session.ps1")`).
#
# A script leaves nothing running behind it (Tai, 2026-10-04: "can you make everything
# graceful"). Gradle keeps its daemon alive after a build, and on Windows the daemon holds the
# console of whatever started it: a caller waiting for the script's output to end waited as long
# as the daemon lived (install.ps1 done at 13:44, its caller still waiting at 14:15). So the
# script stops the daemon and adb's server in a `finally`, on success, failure and Ctrl-C alike;
# Kotlin compiles inside Gradle's daemon (gradle.properties), so there is no second daemon to
# stop. A device on the network is connected again at the start: adb's server, stopped by the
# last script, forgets it.

# adb writes its notices ("daemon not running; starting now") to stderr, which a script under
# $ErrorActionPreference = "Stop" takes for a failure: in these two, what adb says is read, not
# thrown.
function Connect-Device([string]$adb, [string]$serial) {
    $ErrorActionPreference = "Continue"
    & $adb start-server *> $null
    $state = ((& $adb -s $serial get-state 2>&1) -join " ").Trim()
    if ($state -ne "device" -and $serial -match '^[\w\.\-]+:\d+$') {
        & $adb connect $serial 2>&1 | Out-Null
        $state = ((& $adb -s $serial get-state 2>&1) -join " ").Trim()
    }
    if ($state -ne "device") { throw "$serial is not connected ($state)" }
}

function Stop-Helpers([string]$root, [string]$adb) {
    # Quiet and never failing: this runs in `finally`, after whatever went wrong.
    $ErrorActionPreference = "Continue"
    & (Join-Path $root "gradlew.bat") -p $root --stop *> $null
    & $adb kill-server *> $null
}
