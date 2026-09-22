# Sidecoach for Android

The chess coaching companion on the phone: open it once, grant screen capture, switch to
the chess app and play; the coach speaks. Phase 3 of `docs/app-plan.md` in the desktop
repo (`D:\chess-coaching-companion`), which is the source of truth for scope and order.

The desktop app is Python behind a Tauri shell. The phone cannot run that interpreter, so
the pipeline is ported to Kotlin module by module, each with parity tests against the
Python package's fixtures. Nothing is ported before the spikes below say the phone can
carry it.

## Where it stands

- **Spike A -- capture.** A foreground service with MediaProjection into an `ImageReader`
  at half the screen's width and height, throttled to a target rate, with the frame
  interval, this process's CPU, the battery and the thermal status logged. What it must
  answer: can the phone hand us frames steadily for half an hour with a chess app in
  front, and what does it cost. **In progress.**
- **Spike B -- the building blocks.** onnxruntime-android with the 6 MB piece embedder,
  sherpa-onnx for Kokoro and Parakeet, Stockfish's Android build under the NDK; then all
  four at once for thirty minutes with the thermal state logged.
- **The port**, in pipeline order: chess, vision, state, analysis, coach, profile.

## Building

The SDK is the one Android Studio installs (`%LOCALAPPDATA%\Android\Sdk`); `local.properties`
points at it and is not committed. JDK 17.

```
gradlew.bat assembleDebug          # the APK
gradlew.bat installDebug           # onto the phone over adb
```

The phone needs developer options and USB debugging; `adb devices` must list it.

## Layout

```
app/            the application: the capture service and the Compose screens
  capture/      MediaProjection, the virtual display, the frame ring, the spike's stats
```

Modules for the ported pipeline (`vision`, `chess`, `analysis`, `coach`, `profile`) are
added as each one lands.
