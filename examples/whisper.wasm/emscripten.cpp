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

static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p/2;
}

EMSCRIPTEN_BINDINGS(whisper) {
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
    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate, int max_len) {
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

        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        params.print_realtime   = false;
        params.print_timestamps = true;
        params.token_timestamps = true; // required for `max_len`
        params.max_len          = 1;
        params.print_special    = false;
        params.translate        = translate;
        params.language         = whisper_is_multilingual(g_contexts[index]) ? lang.c_str() : "en";
        params.n_threads        = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms        = 0;
        params.split_on_word    = true;

        // this callback is called on each new segment
        if (!params.print_realtime) {
            params.new_segment_callback_user_data = nullptr;
            params.new_segment_callback           = [](struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
                const int n_segments = whisper_full_n_segments(ctx);

                int64_t t0 = 0;
                int64_t t1 = 0;

                // print the last n_new segments
                const int s0 = n_segments - n_new;

                if (s0 == 0) {
                    printf("\n");
                }

                for (int i = s0; i < n_segments; i++) {
                    t0 = whisper_full_get_segment_t0(ctx, i);
                    t1 = whisper_full_get_segment_t1(ctx, i);
                    printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
                    printf("\n");
                
                    for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                        const char * text = whisper_full_get_token_text(ctx, i, j);
                        const float  p     = whisper_full_get_token_p(ctx, i, j);

                        printf("text: %s; token.p: %f\n", text, p);
                    }

                    printf("\n");
                }
            };
        }


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

            printf("\n");
        }

        // run the worker
        {
            g_worker = std::thread([index, params, pcmf32 = std::move(pcmf32)]() {
                whisper_reset_timings(g_contexts[index]);
                whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());

                const int n_segments = whisper_full_n_segments(g_contexts[index]);
                printf("n_segments: %d\n", n_segments);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(g_contexts[index], i);

                    int prob_n = 0;
                    float prob = 0.0f;

                    // n_tokens = 1 for max_len = 1
                    const int n_tokens = whisper_full_n_tokens(g_contexts[index], i);
                    for (int j = 0; j < n_tokens; ++j) {
                        const auto token = whisper_full_get_token_data(g_contexts[index], i, j);
                        prob += token.p;
                        ++prob_n;
                    }

                    printf("text: %s; token.p: %f\n", text, prob/prob_n);
                }
                
                whisper_print_timings(g_contexts[index]);
            });
        }

        return 0;
    }));
}
