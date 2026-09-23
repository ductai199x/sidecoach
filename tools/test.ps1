# Run the parity suite on a device, with the engine's network in place.
#
#   powershell -ExecutionPolicy Bypass -File tools\test.ps1
#   powershell -ExecutionPolicy Bypass -File tools\test.ps1 -Class com.sidecoach.analysis.EngineTest
#
# Gradle's `connectedDebugAndroidTest` uninstalls the app when it is done, which takes the
# engine's 98 MB network with it and leaves the engine tests with nothing to load. So the
# two APKs are installed and the instrumentation driven directly: the network is pushed
# once after the install and survives every run until the app is uninstalled.

param(
    [string]$Serial = "emulator-5556",
    [string]$Class = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$root = Split-Path -Parent $PSScriptRoot

if (-not $SkipBuild) {
    Write-Host "building..."
    & (Join-Path $root "gradlew.bat") -p $root :app:assembleDebug :app:assembleDebugAndroidTest | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the build failed" }
}

$app = Join-Path $root "app/build/outputs/apk/debug/app-debug.apk"
$test = Join-Path $root "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
foreach ($apk in @($app, $test)) {
    if (-not (Test-Path $apk)) { throw "no APK at $apk" }
}

Write-Host "installing..."
& $adb -s $Serial install -r -t $app | Out-Null
& $adb -s $Serial install -r -t $test | Out-Null

# The network, if it is not already there. `adb push` writes as the shell, and the app
# cannot read what the shell owns inside its own directories, so it goes by way of
# /data/local/tmp and `run-as`.
$net = Get-ChildItem (Join-Path $root "third_party") -Filter "*.nnue" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($net) {
    $there = & $adb -s $Serial shell "run-as com.sidecoach ls files/$($net.Name) 2>/dev/null"
    if (-not $there) {
        Write-Host "pushing $($net.Name) ($([int]($net.Length / 1MB)) MB)..."
        & $adb -s $Serial push $net.FullName "/data/local/tmp/$($net.Name)" | Out-Null
        & $adb -s $Serial shell chmod 644 "/data/local/tmp/$($net.Name)"
        & $adb -s $Serial shell "run-as com.sidecoach mkdir -p files"
        & $adb -s $Serial shell "run-as com.sidecoach sh -c 'cat /data/local/tmp/$($net.Name) > files/$($net.Name)'"
        & $adb -s $Serial shell rm "/data/local/tmp/$($net.Name)"
    }
} else {
    Write-Host "no network in third_party; the engine tests will say so and skip"
}

# The voice's model files (Kokoro int8 and its voices), from the desktop's data folder, the
# same way: what the voice tests synthesise with. Missing, those tests say so and skip.
$voiceDir = "D:\chess-coaching-companion\data\models\kokoro"
foreach ($name in @("kokoro-v1.0.onnx", "voices-v1.0.bin")) {
    $file = Join-Path $voiceDir $name
    if (-not (Test-Path $file)) { Write-Host "no $name in $voiceDir; the voice tests will skip"; continue }
    $there = & $adb -s $Serial shell "run-as com.sidecoach ls files/models/kokoro/$name 2>/dev/null"
    if (-not $there) {
        Write-Host "pushing $name ($([int]((Get-Item $file).Length / 1MB)) MB)..."
        & $adb -s $Serial push $file "/data/local/tmp/$name" | Out-Null
        & $adb -s $Serial shell chmod 644 "/data/local/tmp/$name"
        & $adb -s $Serial shell "run-as com.sidecoach mkdir -p files/models/kokoro"
        & $adb -s $Serial shell "run-as com.sidecoach sh -c 'cat /data/local/tmp/$name > files/models/kokoro/$name'"
        & $adb -s $Serial shell rm "/data/local/tmp/$name"
    }
}

# The recogniser's files (Parakeet int8, its vocabulary and config) and the voice gate
# (Silero), from the desktop's Hugging Face cache, where Setup puts them on the phone.
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
    if (-not $file) { Write-Host "no $name in the Hugging Face cache; the listening tests will skip"; continue }
    $there = & $adb -s $Serial shell "run-as com.sidecoach ls files/$dest/$name 2>/dev/null"
    if (-not $there) {
        Write-Host "pushing $name ($([int]($file.Length / 1MB)) MB)..."
        & $adb -s $Serial push $file.FullName "/data/local/tmp/$name" | Out-Null
        & $adb -s $Serial shell chmod 644 "/data/local/tmp/$name"
        & $adb -s $Serial shell "run-as com.sidecoach mkdir -p files/$dest"
        & $adb -s $Serial shell "run-as com.sidecoach sh -c 'cat /data/local/tmp/$name > files/$dest/$name'"
        & $adb -s $Serial shell rm "/data/local/tmp/$name"
    }
}

& $adb -s $Serial logcat -c
$args = @("-w", "-e", "listener", "androidx.test.internal.runner.listener.LogRunListener")
if ($Class) { $args += @("-e", "class", $Class) }
$args += "com.sidecoach.test/androidx.test.runner.AndroidJUnitRunner"
Write-Host "running..."
& $adb -s $Serial shell am instrument @args

Write-Host ""
Write-Host "--- what the tests logged ---"
& $adb -s $Serial logcat -d "sidecoach.chess:I" "sidecoach.vision:I" "sidecoach.state:I" "sidecoach.engine:I" "sidecoach.voice:I" "*:S"
