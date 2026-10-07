package androidx.test.runner;

/**
 * {@link SidecoachRunner} for the phone build's second copy (com.sidecoach.apart2): the same
 * runner under its own name, since a test APK declares each instrumentation by its class.
 * Which classes run here is Lanes.kt's ({@code Lane.process}).
 */
public class SidecoachRunnerApart2 extends SidecoachRunner {
}
