# Run the whole suite: the device's tests (app/src/androidTest) on a device, with the engine's
# network in place, and beside them the host's (app/src/test: the tests that need no Android,
# on the JVM, Gradle's testPhoneUnitTest). The exit code is 0 only if every test of both passed.
#
#   powershell -ExecutionPolicy Bypass -File tools\test.ps1
#   powershell -ExecutionPolicy Bypass -File tools\test.ps1 -Class com.sidecoach.analysis.EngineTest
#
# The default run is the whole suite: every case of every test (Tai, 2026-10-02). `-Class` runs
# that device class alone; a host class runs with `gradlew :app:testPhoneUnitTest --tests <class>`.
#
# The tests run against the `phone` build type, the build the phone runs: not debuggable, so
# the code runs as fast as it does there (app/build.gradle.kts). Gradle's
# `connectedPhoneAndroidTest` would uninstall the app when it is done, which takes the engine's
# 98 MB network and the voice's models with it. So the two APKs are installed and the
# instrumentation driven directly: the files are pushed once and survive every run until the
# app is uninstalled.

param(
    [string]$Serial = "emulator-5556",
    [string]$Class = "",
    [switch]$SkipBuild,
    # Every class one after another, as before the lanes (androidTest Lanes.kt): to tell a
    # lane's fault from a test's, and for each test's own CPU in app/build/test-timings.tsv.
    [switch]$OneAtATime
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot
$gradle = Join-Path $root "gradlew.bat"
$clock = [Diagnostics.Stopwatch]::StartNew()
function At { "[{0,6:N1} s]" -f $clock.Elapsed.TotalSeconds }

if (-not $SkipBuild) {
    Write-Host "$(At) building..."
    # The two APKs and the host tests' classes, on at most eight of the host's workers.
    & $gradle -p $root --max-workers=8 :app:assemblePhone :app:assemblePhoneAndroidTest :app:compilePhoneUnitTestSources | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the build failed" }
}

# The host's tests, started now and run while the device runs its own: Gradle's daemon is free
# once the build is done, and the tests' two JVMs take a quarter of the host at most
# (app/build.gradle.kts). Their results are Gradle's XML, read at the end.
$hostRun = $null
$hostLog = Join-Path $root "app/build/test-host.log"
$hostResults = Join-Path $root "app/build/test-results/testPhoneUnitTest"
if (-not $Class) {
    if (Test-Path $hostResults) { Get-ChildItem $hostResults -Filter "*.xml" | Remove-Item }
    $hostRun = Start-Process -FilePath $gradle -NoNewWindow -PassThru `
        -ArgumentList @("-p", $root, "--max-workers=4", "-q", ":app:testPhoneUnitTest") `
        -RedirectStandardOutput $hostLog -RedirectStandardError "$hostLog.err"
    $null = $hostRun.Handle # held now, or PowerShell has no exit code to give at the end
}

$app = Join-Path $root "app/build/outputs/apk/phone/app-phone.apk"
$test = Join-Path $root "app/build/outputs/apk/androidTest/phone/app-phone-androidTest.apk"
foreach ($apk in @($app, $test)) {
    if (-not (Test-Path $apk)) { throw "no APK at $apk" }
}

# The files the tests load, each as (local file, folder under the app's files, what skips
# without it): the engine's network from third_party; the voice's model files (Kokoro and its
# voices) from the desktop's data folder; the recogniser's (Parakeet int8, its vocabulary and
# config) and the voice gate's (Silero) from the desktop's Hugging Face cache, where Setup puts
# them on the phone. Missing here, the tests that need one say so and skip.
$files = @()
$net = Get-ChildItem (Join-Path $root "third_party") -Filter "*.nnue" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($net) { $files += , @($net.FullName, "") } else { Write-Host "no network in third_party; the engine tests will say so and skip" }
$voiceDir = "D:\chess-coaching-companion\data\models\kokoro"
foreach ($name in @("kokoro-v1.0.onnx", "voices-v1.0.bin")) {
    $file = Join-Path $voiceDir $name
    if (Test-Path $file) { $files += , @($file, "models/kokoro") } else { Write-Host "no $name in $voiceDir; the voice tests will skip" }
}
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$pushes = @(
    @("models--OpenVoiceOS--nvidia-parakeet-tdt_ctc-110m-onnx", "encoder-model.int8.onnx", "models/parakeet-tdt_ctc-110m"),
    @("models--OpenVoiceOS--nvidia-parakeet-tdt_ctc-110m-onnx", "decoder_joint-model.int8.onnx", "models/parakeet-tdt_ctc-110m"),
    @("models--OpenVoiceOS--nvidia-parakeet-tdt_ctc-110m-onnx", "vocab.txt", "models/parakeet-tdt_ctc-110m"),
    @("models--OpenVoiceOS--nvidia-parakeet-tdt_ctc-110m-onnx", "config.json", "models/parakeet-tdt_ctc-110m"),
    @("models--istupakov--silero-vad-onnx", "silero_vad.onnx", "models/silero")
)
foreach ($push in $pushes) {
    $repo, $name, $dest = $push
    $file = Get-ChildItem (Join-Path $hub "$repo\snapshots") -Recurse -Filter $name -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($file) { $files += , @($file.FullName, $dest) } else { Write-Host "no $name in the Hugging Face cache; the listening tests will skip" }
}
# Each one's path under the app's files.
$paths = $files | ForEach-Object { $name = Split-Path $_[0] -Leaf; if ($_[1]) { "$($_[1])/$name" } else { $name } }

# Which of them the device has, in one look: the list the tests' listener writes of the app's
# files at each run's start (com.sidecoach.FilesIn); before the first run of the phone build, the
# files as the debuggable build installed then holds them (`run-as`, which only it allows); on a
# fresh device, none.
$outside = "/sdcard/Android/data/com.sidecoach/files"
$held = @(& $adb -s $Serial shell "cat $outside/files.txt 2>/dev/null" | ForEach-Object { ($_.Trim() -split " ")[0] } |
    Where-Object { $_ })
if ($held.Count -eq 0) {
    $held = @(& $adb -s $Serial shell "run-as com.sidecoach sh -c 'cd files && for f in $($paths -join ' '); do [ -f `$f ] && echo `$f; done' 2>/dev/null" |
        ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

# When each package was last installed on the device, as its package manager says.
function Installed-At {
    $times = & $adb -s $Serial shell "dumpsys package com.sidecoach 2>/dev/null | grep lastUpdateTime; dumpsys package com.sidecoach.test 2>/dev/null | grep lastUpdateTime"
    return (($times | ForEach-Object { $_.Trim() }) -join "|")
}

# The two APKs installed only when they changed since this checkout last installed them here: the
# build leaves an APK alone when nothing changed, and the install was seconds of every run. The
# stamp holds the APKs' sizes and times and the device's own install times, so an install from
# anywhere else (another branch, tools\install.ps1) since then is seen and replaced. Over a debug
# build it keeps the app's files and settings: the same package, signed with the same key.
$stamp = Join-Path $root "app/build/test-installed-$($Serial -replace '[^A-Za-z0-9]', '_').txt"
$apks = (@($app, $test) | ForEach-Object { $item = Get-Item $_; "$($item.Length):$($item.LastWriteTimeUtc.Ticks)" }) -join "|"
$wanted = "$apks|$(Installed-At)"
if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $wanted)) {
    Write-Host "$(At) installed already (the APKs have not changed)"
} else {
    Write-Host "$(At) installing..."
    # -d: this checkout's build, even over a newer version another branch installed on a shared
    # emulator (a version bump elsewhere made the install fail, and the tests ran on that build).
    & $adb -s $Serial install -r -t -d $app | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the app did not install" }
    & $adb -s $Serial install -r -t -d $test | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the tests did not install" }
    Set-Content -Path $stamp -Value "$apks|$(Installed-At)" -NoNewline
}

# What is missing goes to the app's external files, which the shell may write, and the tests'
# listener moves it into the app's own files when the run starts.
for ($i = 0; $i -lt $files.Count; $i++) {
    if ($held -contains $paths[$i]) { continue }
    $local = $files[$i][0]
    Write-Host "pushing $(Split-Path $local -Leaf) ($([int]((Get-Item $local).Length / 1MB)) MB)..."
    # The folder made by the shell first: `adb push` makes its folders with an owner it may not
    # set under Android/data ("secure_mkdirs failed: Operation not permitted").
    $folder = "$outside/push/$($paths[$i])" -replace '/[^/]+$', ''
    & $adb -s $Serial shell "mkdir -p $folder"
    & $adb -s $Serial push $local "$outside/push/$($paths[$i])" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the push of $($paths[$i]) failed" }
}

& $adb -s $Serial logcat -c
& $adb -s $Serial shell "rm -f $outside/timings.tsv"
$args = @("-w", "-e", "listener", "androidx.test.internal.runner.listener.LogRunListener,com.sidecoach.FilesIn")
if ($Class) { $args += @("-e", "class", $Class) }
if ($OneAtATime) { $args += @("-e", "lanes", "off") }
$args += "com.sidecoach.test/androidx.test.runner.SidecoachRunner"
Write-Host "$(At) running..."
$deviceOut = @(& $adb -s $Serial shell am instrument @args | ForEach-Object { Write-Host $_; $_ })
$devicePassed = [bool]($deviceOut | Where-Object { $_ -match "^OK \(\d+ tests?\)" })

Write-Host ""
Write-Host "$(At) --- what the tests logged ---"
& $adb -s $Serial logcat -d "sidecoach.chess:I" "sidecoach.vision:I" "sidecoach.state:I" "sidecoach.engine:I" "sidecoach.voice:I" "sidecoach.lanes:I" "*:S"

# Each test's lane and times (com.sidecoach.Lanes), for a look at where the time goes.
& $adb -s $Serial pull "$outside/timings.tsv" (Join-Path $root "app/build/test-timings.tsv") | Out-Null

$hostPassed = $true
if ($hostRun) {
    $hostRun.WaitForExit()
    $tests = 0; $failed = 0; $skipped = 0
    foreach ($xml in Get-ChildItem $hostResults -Filter "*.xml" -ErrorAction SilentlyContinue) {
        $suite = ([xml](Get-Content $xml.FullName -Raw)).testsuite
        $tests += [int]$suite.tests; $failed += [int]$suite.failures + [int]$suite.errors; $skipped += [int]$suite.skipped
    }
    $hostPassed = ($hostRun.ExitCode -eq 0) -and ($failed -eq 0) -and ($tests -gt 0)
    Write-Host ""
    Write-Host "$(At) --- the host's tests ---"
    Write-Host "$tests tests, $failed failed, $skipped skipped (gradle exit $($hostRun.ExitCode); app/build/test-host.log)"
    if (-not $hostPassed) { Get-Content $hostLog, "$hostLog.err" -ErrorAction SilentlyContinue | Select-Object -Last 60 }
}
if (-not ($devicePassed -and $hostPassed)) {
    Write-Host "$(At) FAILED: device $(if ($devicePassed) { 'passed' } else { 'failed' }), host $(if ($hostPassed) { 'passed' } else { 'failed' })"
    exit 1
}
Write-Host "$(At) PASSED: device and host"
