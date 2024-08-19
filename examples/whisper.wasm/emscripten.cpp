#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>
#include <thread>

std::thread g_worker;

std::vector<std::vector<std::string>> result;
std::vector<struct whisper_context *> g_contexts(4, nullptr);

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

// Largest power of 2, which is less/equal to n
static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p/2;
}


EMSCRIPTEN_BINDINGS(whisper) {
    
    /**
     * Initializes whisper model from given model file.
     *
     * @param path_model The path to the model file.
     * @return The index of the newly created context, or 0 if no context could be created.
     */
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    /**
     * Free up allocated memory.
     *
     * @param index (optional) Free up specific context at index
     * @return The index of the newly created context, or 0 if no context could be created.
     */
    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index < g_contexts.size()) {
            whisper_free(g_contexts[index]);
            g_contexts[index] = nullptr;
        }
    }));


    /**
     * Run initialized Whisper model on input audio with given parameters.
     *
     * @param index The index of the context to use.
     * @param audio Reference to JavaScript Float32Array of Audio Channel Data
     * @param lang Language code for the language of the resulting text.
     * @param nthreads The number of threads used for running the model.
     * @param translate Whether translation is need for the transcription.
     * @param max_len The maximum character length of a segment.
     * @param tdrz Whether to support speaker detection when "tinydiarize" model is used.
     * @return 0 if transcription was started successfully, -1 if the context index is out of range, -2 if the context at the given index is null.
     */
    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate, int max_len, bool tdrz) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index >= g_contexts.size()) {
            return -1;
        }

        if (g_contexts[index] == nullptr) {
            return -2;
        }

        // Create parameter reference struct to be passed to `whisper_full
        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        // Apply input paramters from user interface
        params.print_realtime   = false;
        params.print_timestamps = true;
        params.token_timestamps = true; // required for `max_len`
        params.max_len          = max_len;
        params.print_special    = false;
        params.translate        = translate;
        params.language         = "auto";
        params.n_threads        = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms        = 0;
        params.split_on_word    = true;
        params.tdrz_enable      = tdrz;

        // Callback runs on each new segment
        if (!params.print_realtime) {
            params.new_segment_callback_user_data = nullptr;
            params.new_segment_callback           = [](struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
                
                // Get all segments from currently running inference 
                const int n_segments = whisper_full_n_segments(ctx);

                int64_t t0 = 0;
                int64_t t1 = 0;

                const int s0 = n_segments - n_new;

                // Only go over the newly added segments & print them out
                for (int i = s0; i < n_segments; i++) {

                    // Segment timestamp
                    t0 = whisper_full_get_segment_t0(ctx, i);
                    t1 = whisper_full_get_segment_t1(ctx, i);
                    printf("TIME##%lld##%lld\n", t0, t1);
                
                    // Segment tokens
                    for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                        const char * text = whisper_full_get_token_text(ctx, i, j);

                        // Token precision 
                        const float p     = whisper_full_get_token_p(ctx, i, j);

                        printf("TEXT##%s##%f\n", text, p);
                    }
                }
            };
        }

        // Parse the Float32Array audio data to `pcmf32` for compatibiliy

        std::vector<float> pcmf32;
        const int n = audio["length"].as<int>();

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memory = heap["buffer"];

        pcmf32.resize(n);

        emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
        memoryView.call<void>("set", audio);

        // print system information
        {
            printf("system_info: n_threads = %d / %d | %s\n",
                    params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());

            printf("%s: processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s ..., max_len = %d\n",
                    __func__, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, 1,
                    params.language,
                    params.translate ? "translate" : "transcribe",
                    params.max_len
                    );
        }

        // run the worker
        {
            g_worker = std::thread([index, params, pcmf32 = std::move(pcmf32)]() {
                whisper_reset_timings(g_contexts[index]);
                // run whisper model
                whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());
                whisper_print_timings(g_contexts[index]);
                printf("DONE##");
            });
        }

        return 0;
    }));
}
