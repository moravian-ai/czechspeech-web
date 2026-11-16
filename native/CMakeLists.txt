# CzechSpeech WebAssembly modules: whisper.cpp + ggml compiled with Emscripten into one ES module per variant.
#
# native/build-wasm.sh drives it with one build tree per variant:
#   CZECHSPEECH_THREADS=ON   pthreads + SIMD (needs a cross-origin isolated page: SharedArrayBuffer)
#   CZECHSPEECH_THREADS=OFF  single-threaded + SIMD (fallback, and the VAD-only capture worker)
#   CZECHSPEECH_WEBGPU=ON    single-threaded + SIMD + ggml's WebGPU backend (Dawn's emdawnwebgpu bindings) with
#                            JSPI: the exports that wait for the GPU return Promises (see CS_JSPI_EXPORTS)
# The CPU trees link the same objects twice: czechspeech-<variant>.js for browsers (ENVIRONMENT=worker, shipped) and
# czechspeech-<variant>-node.js for the Node.js integration tests (not shipped). The WebGPU tree links the browser
# module only: Node.js has no WebGPU.
cmake_minimum_required(VERSION 3.22)
project(czechspeech_wasm C CXX)

if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "Configure with emcmake (Emscripten); see native/build-wasm.sh")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CZECHSPEECH_THREADS "pthreads build (SharedArrayBuffer)" ON)
option(CZECHSPEECH_WEBGPU "ggml WebGPU backend (JSPI), single-threaded" OFF)
set(CZECHSPEECH_NAME "czechspeech" CACHE STRING "Output base name")
if (CZECHSPEECH_WEBGPU AND CZECHSPEECH_THREADS)
    message(FATAL_ERROR "CZECHSPEECH_WEBGPU needs CZECHSPEECH_THREADS=OFF (the GPU does the heavy work)")
endif()
set(WHISPER_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/whisper.cpp" CACHE PATH "whisper.cpp source tree")

if (NOT EXISTS "${WHISPER_CPP_DIR}/include/whisper.h")
    message(FATAL_ERROR "whisper.cpp not found in ${WHISPER_CPP_DIR}; run: git submodule update --init")
endif()

# whisper.cpp and ggml as static libraries, no OpenMP, no dynamic backend loading; the CPU backend always, the WebGPU
# backend in the WebGPU variant.
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL           OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2           OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(WHISPER_WASM_SINGLE_FILE OFF CACHE BOOL "" FORCE)
set(GGML_WASM_SINGLE_FILE  OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL        OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS  OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE            OFF CACHE BOOL "" FORCE)
set(GGML_METAL             OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN            OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL            OFF CACHE BOOL "" FORCE)
set(GGML_CUDA              OFF CACHE BOOL "" FORCE)
if (CZECHSPEECH_WEBGPU)
    # ggml-webgpu waits for the GPU (adapter, device, queue, buffer maps) with wgpuInstanceWaitAny, which under
    # Emscripten needs JSPI (default) or Asyncify. JSPI keeps the wasm unchanged and fast; it needs a browser with
    # WebAssembly.Suspending / WebAssembly.promising.
    set(GGML_WEBGPU        ON  CACHE BOOL "" FORCE)
    set(GGML_WEBGPU_JSPI   ON  CACHE BOOL "" FORCE)
    set(GGML_WEBGPU_DEBUG  OFF CACHE BOOL "" FORCE)
    # Development only (never shipped): ggml-webgpu's per-shader GPU timings, printed to stdout when the backend is
    # freed. Configure by hand with -DCZECHSPEECH_WEBGPU_PROFILE=ON into a separate build tree.
    option(CZECHSPEECH_WEBGPU_PROFILE "ggml-webgpu GPU/CPU profiling (development)" OFF)
    set(GGML_WEBGPU_CPU_PROFILE ${CZECHSPEECH_WEBGPU_PROFILE} CACHE BOOL "" FORCE)
    set(GGML_WEBGPU_GPU_PROFILE ${CZECHSPEECH_WEBGPU_PROFILE} CACHE BOOL "" FORCE)
else()
    set(GGML_WEBGPU        OFF CACHE BOOL "" FORCE)
endif()

# whisper.cpp's CMakeLists adds -pthread to every compile under Emscripten. The single-threaded variant removes it
# again with a compiler launcher (native/strip-pthread.sh, set by build-wasm.sh), so its objects are plain wasm.
add_subdirectory("${WHISPER_CPP_DIR}" whisper.cpp EXCLUDE_FROM_ALL)

set(CS_EXPORTS
    _malloc _free
    _cs_init _cs_audio_ctx _cs_set_verbose _cs_system_info _cs_zero_free _cs_backend_init
    _cs_whisper_init _cs_whisper_free _cs_whisper_transcribe _cs_whisper_timings _cs_whisper_tokens _cs_whisper_capped
    _cs_whisper_language
    _cs_text_ptr _cs_text_len _cs_text_release
    _cs_vad_init _cs_vad_free _cs_vad_reset _cs_vad_probs _cs_vad_speech_segments)
list(JOIN CS_EXPORTS "," CS_EXPORTS_CSV)
# WebGPU variant: the exports that can reach wgpuInstanceWaitAny. They return Promises (WebAssembly.promising); every
# other export stays synchronous. cs_backend_init must run first: it builds ggml's backend registry, which requests
# the adapter and device (later registry lookups, e.g. by the VAD, then return at once without waiting).
set(CS_JSPI_EXPORTS cs_backend_init cs_whisper_init cs_whisper_transcribe cs_whisper_free)
list(JOIN CS_JSPI_EXPORTS "," CS_JSPI_EXPORTS_CSV)

set(CS_LINK_FLAGS
    -O3
    -msimd128
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createCzechSpeechModule
    -sFILESYSTEM=0
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=64MB
    -sMAXIMUM_MEMORY=2GB
    -sSTACK_SIZE=5MB
    -sDYNAMIC_EXECUTION=0
    -sASSERTIONS=0
    -sEXPORTED_FUNCTIONS=${CS_EXPORTS_CSV}
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,UTF8ToString,wasmMemory
)
if (CZECHSPEECH_THREADS)
    # Pool size comes from the page: Module.czechspeechPool (threads + 1), set by src/engine-wasm.js. ggml creates
    # and joins its worker threads for every graph; joins return the Workers to the pool synchronously, so a pool
    # of that size is never exhausted. Exhaustion would be a hard EAGAIN instead of a deadlock (STRICT=2).
    # threads-teardown.js gives src/engine-wasm.js Emscripten's runtime teardown for when it releases the module (the
    # workers and the main thread's mailbox wait hold the module's shared memory; without it the memory outlives the
    # engine).
    list(APPEND CS_LINK_FLAGS
        -pthread
        -sPTHREAD_POOL_SIZE=Module.czechspeechPool
        -sPTHREAD_POOL_SIZE_STRICT=2
        -sDEFAULT_PTHREAD_STACK_SIZE=1MB
        --post-js=${CMAKE_CURRENT_SOURCE_DIR}/threads-teardown.js)
endif()
if (CZECHSPEECH_WEBGPU)
    # ggml-webgpu adds --use-port=emdawnwebgpu, -sJSPI and -fwasm-exceptions to the link (INTERFACE options).
    list(APPEND CS_LINK_FLAGS -sJSPI -sJSPI_EXPORTS=${CS_JSPI_EXPORTS_CSV} -fwasm-exceptions)
    set(CS_ENVS worker)
else()
    set(CS_ENVS worker node)
endif()

foreach (CS_ENV ${CS_ENVS})
    if (CS_ENV STREQUAL "worker")
        set(CS_TARGET "${CZECHSPEECH_NAME}")
    else()
        set(CS_TARGET "${CZECHSPEECH_NAME}-node")
    endif()
    add_executable(${CS_TARGET} czechspeech_wasm.c)
    target_link_libraries(${CS_TARGET} PRIVATE whisper)
    set_target_properties(${CS_TARGET} PROPERTIES SUFFIX ".js")
    if (CZECHSPEECH_THREADS)
        target_compile_options(${CS_TARGET} PRIVATE -pthread)
    endif()
    target_compile_options(${CS_TARGET} PRIVATE -O3 -msimd128 -Wall -Wextra)
    target_link_options(${CS_TARGET} PRIVATE ${CS_LINK_FLAGS} -sENVIRONMENT=${CS_ENV})
endforeach()
