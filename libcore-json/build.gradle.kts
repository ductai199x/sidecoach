// Android's org.json, for the app's unit tests on the host: AOSP libcore's json/src/main/java/org/json,
// verbatim from platform/libcore main at de876a01b29230b877c9f408348c38a90ee724a5 (2025-03-26),
// Apache 2.0. android.jar has only stubs of it (they throw), and the libraries published to Maven
// differ: json.org's orders keys by hash, and the 2013 snapshot of Android's (android-json
// 0.0.20131108.vaadin1) still kept its keys in a HashMap. The phone's keeps insertion order.
//
// The sources carry AOSP's API annotations; src/stubs declares them for the compiler only, so
// nothing but org.json reaches the tests' classpath.
plugins {
    `java-library`
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

val stubs: SourceSet = sourceSets.create("stubs")

dependencies {
    compileOnly(stubs.output)
}
