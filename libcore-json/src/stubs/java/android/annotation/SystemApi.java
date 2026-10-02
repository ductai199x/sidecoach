package android.annotation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** Compile-only stand-in for AOSP's annotation of the same name: it marks, and does nothing. */
@Retention(RetentionPolicy.SOURCE)
public @interface SystemApi {
    enum Client {
        PRIVILEGED_APPS,
        MODULE_LIBRARIES,
        SYSTEM_SERVER,
    }

    Client client() default Client.PRIVILEGED_APPS;
}
