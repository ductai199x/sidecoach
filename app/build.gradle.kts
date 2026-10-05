plugins {
    // AGP 9 compiles Kotlin itself (the built-in support) and configures the Compose
    // compiler from `buildFeatures.compose`: no Kotlin plugins in this block.
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

/** The device tests' extra processes (the `apart` build types; androidTest Lanes.kt). */
val APART_PROCESSES = 3

android {
    namespace = "com.sidecoach"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.sidecoach"
        // 33 is where POST_NOTIFICATIONS became a runtime permission; below it the
        // foreground-service and projection rules this app is built around do not hold.
        minSdk = 33
        targetSdk = 37
        versionCode = 177
        versionName = "0.1.77"
        // AndroidJUnitRunner with the classes in lanes side by side (androidTest Lanes.kt).
        testInstrumentationRunner = "androidx.test.runner.SidecoachRunner"
        ndk {
            // OpenCV and onnxruntime each ship four ABIs and the APK came to 323 MB. The
            // phone is arm64 and the emulator x86_64; a release build splits per ABI.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    // Stockfish, from the source tools/fetch_stockfish.py puts in third_party/. Without
    // it the app builds and everything but the engine works, so a fresh clone is not
    // blocked on a download.
    if (file("src/main/cpp/CMakeLists.txt").exists() &&
        file("../third_party/stockfish/src/uci.cpp").exists()
    ) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }

    androidResources {
        // The panel's page keeps its chunks in `_app/`, and the default pattern skips any
        // directory whose name begins with an underscore: without this the page loads and
        // every script it asks for comes back as index.html.
        ignoreAssetsPatterns += listOf()
        ignoreAssetsPattern = "!.svn:!.git:!.ds_store:!*.scc:.*:!CVS:!thumbs.db:!picasa.ini:!*~"
    }

    buildTypes {
        debug {
            buildConfigField("boolean", "DEV_TOOLS", "true")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            buildConfigField("boolean", "DEV_TOOLS", "false")
        }
        // The build Tai's phone runs and the tests test (tools/install.ps1, tools/test.ps1):
        // debug's in all but being debuggable. A debuggable app's code runs without inlining or
        // intrinsics and calls native code through generic trampolines -- the parity suite ran
        // several times slower for it, and so did the app on the phone. Signed with the debug
        // key, so it installs over a debug build and keeps its files and settings (Tai,
        // 2026-10-01). No R8 yet: one change at a time.
        create("phone") {
            initWith(getByName("debug"))
            isDebuggable = false
            isJniDebuggable = false
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
            // The developer's tools the debug build had, kept on: the capture window's numbers
            // in files/spike, and the panel's WebView open to chrome://inspect.
            buildConfigField("boolean", "DEV_TOOLS", "true")
        }
        // The phone's build again under other package names, for the device's tests alone: each
        // installed app is a process of its own with a Java heap of its own, and the test classes
        // that churn through the heap (the vision replays allocate gigabytes) run in them beside
        // the main run instead of sharing its one 192 MB heap and its one collector
        // (androidTest Lanes.kt, `Lane.process`; tools/test.ps1). Never installed on the phone.
        for (n in 1..APART_PROCESSES) {
            create("apart$n") {
                initWith(getByName("phone"))
                applicationIdSuffix = ".apart$n"
                matchingFallbacks += listOf("phone", "debug")
            }
        }
    }
    // The instrumentation tests run against the phone's build, not the debuggable one.
    testBuildType = "phone"
    // The helpers both kinds of test use (Parallel, Workers): one copy, compiled into the host's
    // unit tests (src/test) and the device's (src/androidTest).
    sourceSets {
        getByName("test").kotlin.srcDir("src/sharedTest/java")
        getByName("androidTest").kotlin.srcDir("src/sharedTest/java")
        // The vision bench's game (VisionBenchTest), written by the desktop's
        // `tools/bench_vision.py --frames app/build/bench/bench-vision`: in the test APK only
        // when it has been written, so the suite's own runs carry nothing extra.
        getByName("androidTest").assets.srcDir("build/bench")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    androidResources {
        // The net is read straight out of the APK; compressing it would only cost time.
        noCompress += "onnx"
    }
    packaging {
        // The engine is an executable in the library directory, and a file can only be
        // executed if it exists as a file: with the libraries left in the APK there is
        // nothing to run. The cost is the unpacked copy on disk.
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.onnxruntime.android)
    implementation(libs.opencv)
    // The Android build of the library (its .aar), with the decoder for each ABI.
    implementation(libs.zstd.jni) { artifact { type = "aar" } }
    debugImplementation(libs.androidx.ui.tooling)

    testImplementation(libs.junit)
    // Android's own org.json (AOSP libcore's, libcore-json/), ahead of android.jar's stubs: the
    // code under test builds and reads JSON as it does on the phone -- insertion-ordered objects,
    // Android's number formatting and escapes.
    testImplementation(project(":libcore-json"))
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
}

// The pure-logic tests, on the host (src/test): the chess, the coach, the reconciler, the
// analysis helpers and the model client, none of which needs the Android runtime. A call into
// android.jar fails loudly ("not mocked"; no isReturnDefaultValues), so a test that needs the
// phone cannot pass here by accident. Run with `gradlew :app:testPhoneUnitTest`.
tasks.withType<Test>().configureEach {
    // Two test JVMs of two processors each (the tests' own pools follow: Parallel.cores is
    // availableProcessors()). Measured, a JVM of two processors keeps about three host threads
    // busy -- its two workers, the thread feeding them, its two compilers in bursts: two such
    // JVMs peak at a quarter of the 24 threads, within the eight the host tests may take while
    // the device's suite runs (2026-10-02: three peaked at 35 %, four at 47 %, for 1-2 s less).
    maxParallelForks = 2
    forkEvery = 0
    minHeapSize = "256m"
    maxHeapSize = "1g"
    jvmArgs(
        "-XX:ActiveProcessorCount=2",
        // No collector or compiler threads beyond the test's own: the serial collector runs on
        // the thread that allocates, and two compiler threads are the fewest tiered compilation
        // takes.
        "-XX:+UseSerialGC",
        "-XX:CICompilerCount=2",
        // As on the phone: UTF-8 everywhere and an English locale for String.format.
        "-Dfile.encoding=UTF-8",
        "-Dstdout.encoding=UTF-8",
        "-Duser.language=en",
        "-Duser.country=US",
    )
    // The fixtures stay where the device's tests read them (androidTest/assets); the app's own
    // assets are main/assets.
    systemProperty("sidecoach.assets", file("src/androidTest/assets").absolutePath)
    systemProperty("sidecoach.appAssets", file("src/main/assets").absolutePath)
    // Every run runs every test: never UP-TO-DATE, never FROM-CACHE.
    outputs.upToDateWhen { false }
    outputs.cacheIf { false }
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showStandardStreams = false
    }
}
