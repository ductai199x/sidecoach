package com.sidecoach

/**
 * The threads a test's cell embedder runs on (`CellEmbedder.load(context, threads)`): one, as the
 * desktop's suite runs every net (onnxruntime's `intra_op_num_threads=1` in conftest's
 * `_one_thread_each`). The app reads a board on four ([com.sidecoach.vision.CellEmbedder.load]'s
 * default, measured on the phone); in the suite the classes side by side and the replays' own
 * pools are the parallelism, and a net four threads wide beside them only queued for the
 * emulator's cores, its idle threads spinning. A net's answers do not depend on its threads: the
 * parity tests hold them to the desktop's, which ran on one.
 */
object TestNets {
    const val THREADS = 1
}
