plugins {
    // AGP 9 compiles Kotlin itself (the built-in support) and configures the Compose
    // compiler from `buildFeatures.compose`: no Kotlin plugins in this block.
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.sidecoach"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.sidecoach"
        // 33 is where POST_NOTIFICATIONS became a runtime permission; below it the
        // foreground-service and projection rules this app is built around do not hold.
        minSdk = 33
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // OpenCV and onnxruntime each ship four ABIs and the APK came to 323 MB. The
            // phone is arm64 and the emulator x86_64; a release build splits per ABI.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // The net is read straight out of the APK; compressing it would only cost time.
        noCompress += "onnx"
    }
    packaging {
        // OpenCV ships every ABI; a phone needs one, and the emulator the other.
        jniLibs.useLegacyPackaging = false
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
    debugImplementation(libs.androidx.ui.tooling)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
}
