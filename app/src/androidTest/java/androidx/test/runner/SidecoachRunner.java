package androidx.test.runner;

import android.os.Bundle;
import androidx.test.internal.runner.RunnerArgs;
import com.sidecoach.Lanes;
import org.junit.runner.Request;

/**
 * AndroidJUnitRunner with its test classes run in lanes side by side ({@link Lanes}): one class
 * at a time per lane, the lanes at once. Everything else -- the arguments, the filters, the
 * listeners and what they print -- is AndroidJUnitRunner's own.
 *
 * <p>It lives in AndroidJUnitRunner's package for the one method it changes: {@code buildRequest}
 * is package-private there, and Java (not Kotlin) can override a package-private method.
 */
public class SidecoachRunner extends AndroidJUnitRunner {

    @Override
    Request buildRequest(RunnerArgs runnerArgs, Bundle bundleArgs) {
        return Lanes.request(super.buildRequest(runnerArgs, bundleArgs), bundleArgs);
    }
}
