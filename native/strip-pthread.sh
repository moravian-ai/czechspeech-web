#!/bin/sh
# Compiler/linker launcher for the single-threaded build: runs the command without "-pthread".
# whisper.cpp's CMakeLists adds -pthread to every compile under Emscripten; this keeps the ST objects plain wasm
# (no atomics, no shared memory) without patching the whisper.cpp tree.
n=$#
i=0
while [ "$i" -lt "$n" ]; do
    a=$1
    shift
    i=$((i + 1))
    [ "$a" = "-pthread" ] || set -- "$@" "$a"
done
exec "$@"
