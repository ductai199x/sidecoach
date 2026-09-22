cmake_minimum_required(VERSION 3.22)
project(sidecoach_engine CXX)

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
