package libcore.util;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** Compile-only stand-in for AOSP's annotation of the same name: it marks, and does nothing. */
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE_USE})
public @interface Nullable {
}
