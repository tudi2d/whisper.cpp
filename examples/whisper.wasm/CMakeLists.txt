#
# libmain
#

set(TARGET libmain)

add_executable(${TARGET}
    emscripten.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    whisper
    )

unset(EXTRA_FLAGS)

if (WHISPER_WASM_SINGLE_FILE)
    set(EXTRA_FLAGS "-s SINGLE_FILE=1")
    message(STATUS "Embedding WASM inside main.js")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/libmain.js
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/whisper.wasm/main.js
        )
endif()

# Build flags: Enforces optimisation with `-O3` improving runtime performance
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    -std=c++11 \
    -flto \
    -O3 \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=2GB \
    -s TOTAL_MEMORY=2GB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\" \
    ${EXTRA_FLAGS} \
    ")

#
# whisper.wasm
#

set(TARGET whisper.wasm)