package com.sidecoach

import androidx.test.platform.app.InstrumentationRegistry

/**
 * The sweeps' other cases, as the desktop's `--sweep` adds them (tests/conftest.py): a test that
 * holds one behaviour to many cases of the same kind -- every piece set -- runs a representative
 * few by default and the rest when asked, with `tools/test.ps1 -All` (`-e sweep true`). The
 * cases a sweep leaves out are marked `sweep` in its fixture (Tai, 2026-10-01: "SLIM DOWN THE
 * SUITE AND MAKE IT BLAZING FAST").
 */
internal object Sweep {
    val all: Boolean = InstrumentationRegistry.getArguments().getString("sweep") == "true"
}
