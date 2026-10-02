package android.compat.annotation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** Compile-only stand-in for AOSP's annotation of the same name: it marks, and does nothing. */
@Retention(RetentionPolicy.SOURCE)
public @interface UnsupportedAppUsage {
    int maxTargetSdk() default Integer.MAX_VALUE;

    long trackingBug() default 0;

    String expectedSignature() default "";

    String implicitMember() default "";

    String publicAlternatives() default "";
}
