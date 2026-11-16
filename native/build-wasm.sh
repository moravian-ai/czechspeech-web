#!/usr/bin/env bash
# Builds the CzechSpeech WebAssembly modules (whisper.cpp + ggml + native/czechspeech_wasm.c) with Emscripten:
#
#   src/wasm/czechspeech-mt.{js,wasm}           pthreads + SIMD; used when the page is cross-origin isolated
#   src/wasm/czechspeech-st.{js,wasm}           single-threaded + SIMD; fallback, and the VAD-only capture worker
#   src/wasm/czechspeech-webgpu.{js,wasm}       single-threaded + SIMD + ggml's WebGPU backend through JSPI; the
#                                               recogniser when the browser offers WebGPU (emdawnwebgpu is fetched
#                                               by Emscripten's built-in port on the first build, hash-checked).
#                                               Built from a copy of whisper.cpp with native/patches/ggml-webgpu-*.patch
#                                               applied; the submodule itself and the CPU modules stay unpatched.
#   build/wasm-node/czechspeech-{mt,st}-node.*  the same objects linked for Node.js (integration tests only)
# Each browser module's JS glue is post-processed by scripts/fix-glue.mjs: its pthread workers start from the module's
# own URL instead of its file name, which bundlers change (for modules built before: node scripts/fix-glue.mjs ...).
#
# Usage (Linux or WSL):  native/build-wasm.sh [all|mt|st|webgpu]      (default: all)
# Environment:
#   EMSDK                       emsdk checkout with EMSCRIPTEN_VERSION installed (default ~/emsdk)
#   JOBS                        parallel compile jobs (default 8)
#   CZECHSPEECH_WASM_BUILD_DIR  CMake build trees (default ~/.cache/czechspeech-web-build; keep it off /mnt/c); the
#                               patched sources of the webgpu variant go to <that>-src/whisper.cpp
set -euo pipefail

WHAT=${1:-all}
# Not named EMSDK_*: emsdk_env.sh unsets every exported EMSDK_* variable it does not manage of its own.
EMSCRIPTEN_VERSION=6.0.13
WHISPER_COMMIT=482279b8a18855ba741d7b88c30fcb3874691f59
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EMSDK=${EMSDK:-$HOME/emsdk}
JOBS=${JOBS:-8}
BUILD=${CZECHSPEECH_WASM_BUILD_DIR:-$HOME/.cache/czechspeech-web-build}
WHISPER=$ROOT/third_party/whisper.cpp
OUT=$ROOT/src/wasm
OUT_NODE=$ROOT/build/wasm-node
WEBGPU_SRC=$BUILD-src/whisper.cpp
# Strip local paths from __FILE__ (ggml asserts) so the modules carry no build-machine paths and are reproducible.
PREFIX_MAP="-ffile-prefix-map=$ROOT/= -ffile-prefix-map=$BUILD/=build/ -ffile-prefix-map=$EMSDK/=emsdk/"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$WHISPER/include/whisper.h" ] || die "third_party/whisper.cpp is empty; run: git submodule update --init"
actual=$(git -C "$WHISPER" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$actual" = "$WHISPER_COMMIT" ] || die "third_party/whisper.cpp is at $actual, expected $WHISPER_COMMIT"
[ -f "$EMSDK/emsdk_env.sh" ] || die "emsdk not found at $EMSDK (git clone https://github.com/emscripten-core/emsdk; ./emsdk install $EMSCRIPTEN_VERSION; ./emsdk activate $EMSCRIPTEN_VERSION)"
# shellcheck disable=SC1091
source "$EMSDK/emsdk_env.sh" > /dev/null 2>&1
emcc --version | head -1 | grep -qF " $EMSCRIPTEN_VERSION " || die "emcc is not $EMSCRIPTEN_VERSION: $(emcc --version | head -1)"
command -v ninja > /dev/null || die "ninja not found"
command -v rsync > /dev/null && command -v patch > /dev/null || die "rsync and patch are needed"
command -v node > /dev/null || die "node not found (emsdk_env.sh puts Emscripten's Node.js on the PATH)"

# The webgpu variant's sources: a copy of the pinned whisper.cpp with the patches applied, in file-name order.
prepare_webgpu_sources() {
    local p
    mkdir -p "$WEBGPU_SRC"
    rsync -a --delete --exclude .git "$WHISPER/" "$WEBGPU_SRC/"
    for p in "$ROOT"/native/patches/ggml-webgpu-*.patch; do
        [ -f "$p" ] || continue
        patch -d "$WEBGPU_SRC" -p1 --forward --batch --quiet < "$p" > /dev/null || die "$(basename "$p") does not apply"
        echo "   patch $(basename "$p")"
    done
}

build_variant() {  # mt|st|webgpu
    local v=$1 dir=$BUILD/$1 threads webgpu=OFF launcher=() targets source=()
    local cflags="-msimd128 $PREFIX_MAP" cxxflags="-msimd128 $PREFIX_MAP"
    targets=("czechspeech-$v" "czechspeech-$v-node")
    if [ "$v" = mt ]; then
        threads=ON
    else
        threads=OFF
        launcher=(-DCMAKE_C_COMPILER_LAUNCHER="$ROOT/native/strip-pthread.sh"
                  -DCMAKE_CXX_COMPILER_LAUNCHER="$ROOT/native/strip-pthread.sh"
                  -DCMAKE_C_LINKER_LAUNCHER="$ROOT/native/strip-pthread.sh"
                  -DCMAKE_CXX_LINKER_LAUNCHER="$ROOT/native/strip-pthread.sh")
    fi
    if [ "$v" = webgpu ]; then
        webgpu=ON
        targets=("czechspeech-$v")   # browsers only: Node.js has no WebGPU
        prepare_webgpu_sources
        source=(-DWHISPER_CPP_DIR="$WEBGPU_SRC")
        # the copy's paths are mapped to the submodule's, so __FILE__ strings read as in an unpatched build
        cflags="$cflags -ffile-prefix-map=$WEBGPU_SRC/=third_party/whisper.cpp/"
        # ggml-webgpu is compiled with wasm exceptions (its WGSL preprocessor throws); whisper.cpp's own try/catch
        # blocks (state allocation) then work too instead of escaping to JavaScript.
        cxxflags="$cflags -fwasm-exceptions"
        # a build tree configured for another source tree is started afresh
        if [ -f "$dir/CMakeCache.txt" ] && ! grep -qxF "WHISPER_CPP_DIR:PATH=$WEBGPU_SRC" "$dir/CMakeCache.txt"; then
            rm -rf "$dir"
        fi
    fi
    echo "== czechspeech-$v (threads=$threads, webgpu=$webgpu)"
    mkdir -p "$dir"
    # EMSCRIPTEN_SYSTEM_PROCESSOR=wasm32: Emscripten.cmake reports "x86" by default, which makes ggml fall back to
    # GGML_CPU_GENERIC (scalar kernels); "wasm32" selects ggml-cpu/arch/wasm/quants.c with the SIMD128 dot products.
    emcmake cmake -S "$ROOT/native" -B "$dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm32 \
        -DCMAKE_C_FLAGS="$cflags" -DCMAKE_CXX_FLAGS="$cxxflags" "${source[@]}" \
        -DCZECHSPEECH_THREADS=$threads -DCZECHSPEECH_WEBGPU=$webgpu -DCZECHSPEECH_NAME="czechspeech-$v" "${launcher[@]}" \
        > "$dir.configure.log" 2>&1 || { tail -40 "$dir.configure.log"; die "configure $v failed"; }
    cmake --build "$dir" --target "${targets[@]}" -j "$JOBS" > "$dir.build.log" 2>&1 \
        || { tail -60 "$dir.build.log"; die "build $v failed"; }
    mkdir -p "$OUT" "$OUT_NODE"
    cp "$dir/czechspeech-$v.js" "$dir/czechspeech-$v.wasm" "$OUT/"
    node "$ROOT/scripts/fix-glue.mjs" "$OUT/czechspeech-$v.js" || die "czechspeech-$v.js could not be post-processed"
    if [ "$v" != webgpu ]; then
        cp "$dir/czechspeech-$v-node.js" "$dir/czechspeech-$v-node.wasm" "$OUT_NODE/"
    fi
}

check_variant() {  # mt|st|webgpu
    local v=$1 f files
    files=("$OUT/czechspeech-$v.js" "$OUT/czechspeech-$v.wasm")
    [ "$v" = webgpu ] || files+=("$OUT_NODE/czechspeech-$v-node.js" "$OUT_NODE/czechspeech-$v-node.wasm")
    for f in "${files[@]}"; do
        # no build-machine paths (user names, home directories, /mnt drives) in the artifacts
        if strings -n 4 "$f" | grep -qF -e "$ROOT" -e "$BUILD" -e "$HOME" -e "/mnt/" -e "Users/" -e 'C:\'; then
            strings -n 4 "$f" | grep -F -e "$ROOT" -e "$BUILD" -e "$HOME" -e "/mnt/" -e "Users/" -e 'C:\' | head -5
            die "$(basename "$f") contains local build paths"
        fi
    done
    local js=$OUT/czechspeech-$v.js
    node "$ROOT/scripts/fix-glue.mjs" --check "$js" || die "czechspeech-$v.js loads its threads by a fixed file name"
    if [ "$v" = mt ]; then
        grep -q "em-pthread" "$js" || die "czechspeech-mt.js has no pthread support"
        grep -q "czechspeechTerminateRuntime" "$js" && grep -q "clearMailboxAwait" "$js" ||
            die "czechspeech-mt.js cannot tear its runtime down (native/threads-teardown.js)"
    else
        if grep -q -e "em-pthread" -e "SharedArrayBuffer" "$js"; then die "czechspeech-$v.js uses threads"; fi
    fi
    if [ "$v" = webgpu ]; then
        grep -q "WebAssembly.promising" "$js" || die "czechspeech-webgpu.js does not use JSPI"
        grep -q "requestAdapter" "$js" || die "czechspeech-webgpu.js has no WebGPU bindings"
        grep -qa "mul_mat_q4_0_tile" "$OUT/czechspeech-$v.wasm" || die "czechspeech-webgpu.wasm lacks the patched Q4_0 kernel"
    elif grep -q "requestAdapter" "$js"; then
        die "czechspeech-$v.js contains WebGPU bindings"
    fi
    printf '   %-28s %9d bytes\n' "czechspeech-$v.wasm" "$(stat -c %s "$OUT/czechspeech-$v.wasm")"
    printf '   %-28s %9d bytes\n' "czechspeech-$v.js" "$(stat -c %s "$js")"
}

mkdir -p "$BUILD"
case $WHAT in
    all) build_variant mt; build_variant st; build_variant webgpu
         echo "== checks"; check_variant mt; check_variant st; check_variant webgpu ;;
    mt|st|webgpu) build_variant "$WHAT"; echo "== checks"; check_variant "$WHAT" ;;
    *) die "unknown target '$WHAT' (all|mt|st|webgpu)" ;;
esac
echo DONE
