cmake_minimum_required(VERSION 3.22)
project(sidecoach_native C CXX)

# Stockfish, built from the source `tools/fetch_stockfish.py` fetches.
#
# Two things differ from an ordinary Stockfish build. It is built without its embedded
# network (NNUE_EMBEDDING_OFF): the network is 98 MB and would go into the APK once per ABI,
# so it is a data file the app fetches instead, given to the engine through the EvalFile
# option. And the executable is named libstockfish.so so that the packager puts it in the
# APK's library directory, which is the only place an app may execute a file from on
# Android 10 and later.

set(SF ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/stockfish/src)

set(SOURCES
    ${SF}/attacks.cpp
    ${SF}/benchmark.cpp
    ${SF}/bitboard.cpp
    ${SF}/engine.cpp
    ${SF}/evaluate.cpp
    ${SF}/main.cpp
    ${SF}/memory.cpp
    ${SF}/misc.cpp
    ${SF}/movegen.cpp
    ${SF}/movepick.cpp
    ${SF}/position.cpp
    ${SF}/score.cpp
    ${SF}/search.cpp
    ${SF}/syzygy/tbprobe.cpp
    ${SF}/thread.cpp
    ${SF}/timeman.cpp
    ${SF}/tt.cpp
    ${SF}/tune.cpp
    ${SF}/uci.cpp
    ${SF}/ucioption.cpp
    ${SF}/nnue/network.cpp
    ${SF}/nnue/nnue_accumulator.cpp
    ${SF}/nnue/nnue_misc.cpp
    ${SF}/nnue/features/half_ka_v2_hm.cpp
    ${SF}/nnue/features/full_threats.cpp
    ${SF}/nnue/features/pp_3wide.cpp
)

add_executable(stockfish ${SOURCES})

# libstockfish.so: an executable the packager treats as a native library (see above).
set_target_properties(stockfish PROPERTIES PREFIX "lib" SUFFIX ".so")

target_include_directories(stockfish PRIVATE ${SF})

target_compile_features(stockfish PRIVATE cxx_std_17)
target_compile_options(stockfish PRIVATE
    -O3 -fno-exceptions -fno-rtti -flto=thin
    -Wno-unused-parameter -Wno-deprecated-declarations
)
target_link_options(stockfish PRIVATE -flto=thin -pie)

# The flags Stockfish's own Makefile sets for each architecture.
target_compile_definitions(stockfish PRIVATE
    NNUE_EMBEDDING_OFF
    IS_64BIT
    USE_PREFETCH
    USE_POPCNT
    NDEBUG
)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # armv8: NEON, and the Int8 dot product every phone since 2019 has. The Pixel has it;
    # `ARCH=armv8-dotprod` is what Stockfish's Makefile calls this.
    target_compile_definitions(stockfish PRIVATE USE_NEON=8 USE_NEON_DOTPROD)
    target_compile_options(stockfish PRIVATE -march=armv8.2-a+dotprod)
elseif(ANDROID_ABI STREQUAL "x86_64")
    # The emulator: SSE up to 4.1, which is what `ARCH=x86-64-sse41-popcnt` builds.
    target_compile_definitions(stockfish PRIVATE USE_SSE41 USE_SSSE3 USE_SSE2)
    target_compile_options(stockfish PRIVATE -msse4.1 -mpopcnt)
endif()

find_library(log-lib log)
target_link_libraries(stockfish ${log-lib})

# ---- espeak-ng, for the voice's phonemes ------------------------------------------------
#
# The desktop's voice turns text into phonemes with espeak-ng 1.52.0 (kokoro-onnx asks it
# through phonemizer). The same tag, from the source `tools/fetch_espeak.py` fetches, is built
# here as espeak-ng's own CMake builds its library -- the library only: its data is compiled
# by running the tool it builds, which a cross-compile cannot do, so the app carries the
# desktop's compiled English data in its assets instead. What the options switch off (async
# speech, Klatt, speech-player, MBROLA, sonic, pcaudio) is sound synthesis; the app uses
# espeak-ng for phonemes only. The JNI bridge (espeak_jni.c) is linked into the same library.

set(ESPEAK ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/espeak-ng)

if(EXISTS ${ESPEAK}/src/libespeak-ng/speech.c)
    set(HAVE_MKSTEMP 1)
    set(USE_ASYNC 0)
    set(USE_KLATT 0)
    set(USE_LIBPCAUDIO 0)
    set(USE_LIBSONIC 0)
    set(USE_MBROLA 0)
    set(USE_SPEECHPLAYER 0)
    set(PROJECT_VERSION 1.52.0)
    configure_file(${ESPEAK}/src/libespeak-ng/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/espeak/config.h)

    set(ESPEAK_LIB ${ESPEAK}/src/libespeak-ng)
    set(UCD ${ESPEAK}/src/ucd-tools/src)
    add_library(sidecoach_espeak SHARED
        ${ESPEAK_LIB}/common.c
        ${ESPEAK_LIB}/mnemonics.c
        ${ESPEAK_LIB}/error.c
        ${ESPEAK_LIB}/ieee80.c
        ${ESPEAK_LIB}/compiledata.c
        ${ESPEAK_LIB}/compiledict.c
        ${ESPEAK_LIB}/dictionary.c
        ${ESPEAK_LIB}/encoding.c
        ${ESPEAK_LIB}/intonation.c
        ${ESPEAK_LIB}/langopts.c
        ${ESPEAK_LIB}/numbers.c
        ${ESPEAK_LIB}/phoneme.c
        ${ESPEAK_LIB}/phonemelist.c
        ${ESPEAK_LIB}/readclause.c
        ${ESPEAK_LIB}/setlengths.c
        ${ESPEAK_LIB}/soundicon.c
        ${ESPEAK_LIB}/spect.c
        ${ESPEAK_LIB}/ssml.c
        ${ESPEAK_LIB}/synthdata.c
        ${ESPEAK_LIB}/synthesize.c
        ${ESPEAK_LIB}/tr_languages.c
        ${ESPEAK_LIB}/translate.c
        ${ESPEAK_LIB}/translateword.c
        ${ESPEAK_LIB}/voices.c
        ${ESPEAK_LIB}/wavegen.c
        ${ESPEAK_LIB}/speech.c
        ${ESPEAK_LIB}/espeak_api.c
        ${UCD}/case.c
        ${UCD}/categories.c
        ${UCD}/ctype.c
        ${UCD}/proplist.c
        ${UCD}/scripts.c
        ${UCD}/tostring.c
        espeak_jni.c
    )
    target_include_directories(sidecoach_espeak PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/espeak
        ${ESPEAK}/src/include
        ${ESPEAK}/src/include/compat
        ${UCD}/include
        ${ESPEAK_LIB}
    )
    target_compile_definitions(sidecoach_espeak PRIVATE
        LIBESPEAK_NG_EXPORT=1
        PATH_ESPEAK_DATA="/nonexistent"
    )
    target_compile_options(sidecoach_espeak PRIVATE
        -O2 -fwrapv -fno-exceptions
        -Wno-unused-parameter -Wno-missing-prototypes -Wno-unused-but-set-variable
    )
    target_link_libraries(sidecoach_espeak m ${log-lib})
else()
    # The voice loads this library at start: built without it, the app shut down after its
    # splash on every launch (0.1.80 and 0.1.81, the source gone from third_party/).
    message(FATAL_ERROR "espeak-ng source missing at ${ESPEAK}: run tools/fetch_espeak.py")
endif()
