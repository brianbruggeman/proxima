// llama-ngram fixture generator.
//
// Calls llama.cpp's own n-gram drafter functions directly against real
// gemma4-tokenized text, so proxima's ports can be tested token-for-token
// against the incumbent. See ../README.md for the fixture schema.

#include "llama.h"
#include "common.h"
#include "log.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "ngram-cache.h"
#include "speculative.h"
#include "json_writer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <sstream>

namespace fs = std::filesystem;

static const char * LLAMA_CPP_COMMIT = "f1ea20621";
static const char * GENERATOR_VERSION = "llama-ngram-fixturegen v1";

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "fatal: could not open %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static std::vector<int32_t> to_i32(const llama_tokens & tokens) {
    return std::vector<int32_t>(tokens.begin(), tokens.end());
}

struct token_stream {
    int32_t id;
    std::string source;
    llama_tokens tokens;
};

static llama_tokens tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special) {
    return common_tokenize(vocab, text, add_special, /*parse_special=*/false);
}

// ngram-mod's shared hash table only needs enough distinct 24-token windows
// to push occupancy past its 0.25 threshold (README.md's "ngram-mod reset
// mechanism") -- the warmup's actual content is never compared against
// anything, so a synthetic high-entropy stream reproduces the reset without
// storing millions of real token ids in the fixture. SplitMix64 (Vigna,
// public domain) is deterministic from {seed, length, vocab_size} alone, and
// this generator's Rust port (`ngram_mod.rs`'s `synthetic_occupancy_warmup`)
// implements the identical constants so both sides produce the same stream.
static const uint64_t OCCUPANCY_WARMUP_SEED = 1;
static const uint32_t OCCUPANCY_WARMUP_LENGTH = 2'200'000;

static uint64_t splitmix64_next(uint64_t & state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static llama_tokens generate_occupancy_warmup_tokens(int32_t vocab_size, uint64_t seed, uint32_t length) {
    llama_tokens tokens;
    tokens.reserve(length);
    uint64_t state = seed;
    for (uint32_t index = 0; index < length; ++index) {
        const uint64_t value = splitmix64_next(state);
        tokens.push_back((llama_token) (value % (uint64_t) vocab_size));
    }
    return tokens;
}

static int32_t longest_accepted_prefix(
        const llama_tokens & draft, const llama_tokens & stream, size_t continuation_start) {
    int32_t accepted = 0;
    for (size_t index = 0; index < draft.size() && continuation_start + index < stream.size(); ++index) {
        if (draft[index] != stream[continuation_start + index]) {
            break;
        }
        ++accepted;
    }
    return accepted;
}

struct case_record {
    int32_t stream_id;
    int32_t position;
    int32_t sampled;
    llama_tokens draft;
    int32_t accepted;
};

static void write_cases_file(
        const std::string & path,
        const std::string & drafter_type,
        const std::string & params_json,
        const std::vector<std::string> & sources,
        const std::vector<case_record> & cases,
        const std::string & extra_header_json = "") {
    FILE * file = fopen(path.c_str(), "w");
    if (!file) {
        fprintf(stderr, "fatal: could not open %s for write\n", path.c_str());
        std::exit(1);
    }
    json_writer writer(file);
    writer.raw("{\n");
    writer.raw("  \"llama_cpp_commit\": \"" + std::string(LLAMA_CPP_COMMIT) + "\",\n");
    writer.raw("  \"generator\": \"" + std::string(GENERATOR_VERSION) + "\",\n");
    writer.raw("  \"drafter\": \"" + drafter_type + "\",\n");
    writer.raw("  \"params\": " + params_json + ",\n");
    writer.raw("  \"sources\": " + json_writer::string_array(sources) + ",\n");
    if (!extra_header_json.empty()) {
        writer.raw("  " + extra_header_json + ",\n");
    }
    writer.raw("  \"cases\": [\n");
    for (size_t index = 0; index < cases.size(); ++index) {
        const case_record & record = cases[index];
        writer.raw("    {");
        writer.raw("\"stream_id\": " + std::to_string(record.stream_id) + ", ");
        writer.raw("\"position\": " + std::to_string(record.position) + ", ");
        writer.raw("\"sampled\": " + std::to_string(record.sampled) + ", ");
        writer.raw("\"draft\": " + json_writer::int_array(to_i32(record.draft)) + ", ");
        writer.raw("\"accepted\": " + std::to_string(record.accepted));
        writer.raw("}");
        writer.raw(index + 1 < cases.size() ? ",\n" : "\n");
    }
    writer.raw("  ]\n");
    writer.raw("}\n");
    fclose(file);
}

static void write_streams_file(const std::string & path, const std::string & vocab_gguf, const std::vector<token_stream> & streams) {
    FILE * file = fopen(path.c_str(), "w");
    if (!file) {
        fprintf(stderr, "fatal: could not open %s for write\n", path.c_str());
        std::exit(1);
    }
    json_writer writer(file);
    writer.raw("{\n");
    writer.raw("  \"vocab_gguf\": \"" + vocab_gguf + "\",\n");
    writer.raw("  \"streams\": [\n");
    for (size_t index = 0; index < streams.size(); ++index) {
        const token_stream & stream = streams[index];
        writer.raw("    {");
        writer.raw("\"id\": " + std::to_string(stream.id) + ", ");
        writer.raw("\"source\": \"" + json_writer::escape(stream.source) + "\", ");
        writer.raw("\"tokens\": " + json_writer::int_array(to_i32(stream.tokens)));
        writer.raw("}");
        writer.raw(index + 1 < streams.size() ? ",\n" : "\n");
    }
    writer.raw("  ]\n");
    writer.raw("}\n");
    fclose(file);
}

// per-stream, not per-file: without this, a single long low-match-density
// prose stream processed first exhausts the whole file's budget before a
// short, match-rich stream (the repeated sentence, the ngram-mod trap) is
// ever reached -- measured: 600 total cases, 0 from the repeated stream.
static const size_t CASE_CAP_PER_STREAM = 400;

// ngram-simple: stateless, one call per position.
static std::vector<case_record> replay_ngram_simple(
        const std::vector<token_stream> & streams, const common_ngram_simple_config & config) {
    std::vector<case_record> cases;
    const size_t min_len = (size_t) config.size_ngram + config.size_mgram + 2;

    for (const auto & stream : streams) {
        size_t i = min_len;
        size_t stream_cases = 0;
        while (i + 1 < stream.tokens.size() && stream_cases < CASE_CAP_PER_STREAM) {
            llama_tokens history(stream.tokens.begin(), stream.tokens.begin() + i);
            const llama_token sampled = stream.tokens[i];

            llama_tokens draft = common_ngram_simple_draft(config, history, sampled);
            const int32_t accepted = longest_accepted_prefix(draft, stream.tokens, i + 1);

            cases.push_back({stream.id, (int32_t) i, sampled, draft, accepted});
            ++stream_cases;
            i += (size_t) accepted + 1;
        }
    }
    return cases;
}

// ngram-map: stateful (key-only or key+values), begin once per stream then draft/accept per position.
static std::vector<case_record> replay_ngram_map(
        const std::vector<token_stream> & streams, uint16_t size_key, uint16_t size_value,
        bool key_only, uint16_t min_hits) {
    std::vector<case_record> cases;
    const size_t min_len = (size_t) 2 * size_key + size_value + 2;

    for (const auto & stream : streams) {
        if (stream.tokens.size() <= min_len) {
            continue;
        }
        common_ngram_map map(size_key, size_value, key_only, min_hits);

        const size_t prompt_len = std::min(min_len, stream.tokens.size() / 2);
        llama_tokens prompt(stream.tokens.begin(), stream.tokens.begin() + prompt_len);
        common_ngram_map_begin(map, prompt);

        size_t i = std::max(prompt_len, min_len);
        size_t stream_cases = 0;
        while (i + 1 < stream.tokens.size() && stream_cases < CASE_CAP_PER_STREAM) {
            llama_tokens history(stream.tokens.begin(), stream.tokens.begin() + i);
            const llama_token sampled = stream.tokens[i];

            llama_tokens draft;
            common_ngram_map_draft(map, history, sampled, draft);
            const int32_t accepted = longest_accepted_prefix(draft, stream.tokens, i + 1);

            cases.push_back({stream.id, (int32_t) i, sampled, draft, accepted});
            common_ngram_map_accept(map, (uint16_t) accepted);
            ++stream_cases;
            i += (size_t) accepted + 1;
        }
    }
    return cases;
}

// ngram-cache: mirrors common_speculative_impl_ngram_cache::draft_one exactly
// (common/speculative.cpp:2095-2138), calling only the public
// common_ngram_cache_update / common_ngram_cache_draft functions.
static std::vector<case_record> replay_ngram_cache(
        const std::vector<token_stream> & streams, uint16_t n_draft, common_ngram_cache & nc_static) {
    std::vector<case_record> cases;

    for (const auto & stream : streams) {
        if (stream.tokens.size() <= (size_t) LLAMA_NGRAM_MAX + 2) {
            continue;
        }
        common_ngram_cache nc_context;
        common_ngram_cache nc_dynamic;
        size_t cache_size = 0;

        size_t i = LLAMA_NGRAM_MAX + 1;
        size_t stream_cases = 0;
        while (i + 1 < stream.tokens.size() && stream_cases < CASE_CAP_PER_STREAM) {
            if (cache_size < i + 1) {
                llama_tokens tokens_new(stream.tokens.begin() + cache_size, stream.tokens.begin() + i);
                tokens_new.push_back(stream.tokens[i]);
                common_ngram_cache_update(nc_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, tokens_new, (int) tokens_new.size(), false);
                cache_size = i + 1;
            }

            llama_tokens inp(stream.tokens.begin(), stream.tokens.begin() + i);
            inp.push_back(stream.tokens[i]);

            llama_tokens draft;
            draft.push_back(stream.tokens[i]);
            common_ngram_cache_draft(inp, draft, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, nc_context, nc_dynamic, nc_static);
            if (!draft.empty()) {
                draft.erase(draft.begin());
            }

            const int32_t accepted = longest_accepted_prefix(draft, stream.tokens, i + 1);
            cases.push_back({stream.id, (int32_t) i, stream.tokens[i], draft, accepted});
            ++stream_cases;
            i += (size_t) accepted + 1;
        }
    }
    return cases;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <vocab-gguf> <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string vocab_gguf_path = argv[1];
    const std::string output_dir = argv[2];
    fs::create_directories(output_dir);

    const std::string log_path = output_dir + "/generator.log";
    setenv("LLAMA_TRACE", "1", 1);
    common_log_set_file(common_log_main(), log_path.c_str());
    common_log_set_verbosity_thold(LOG_LEVEL_TRACE);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_gguf_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "fatal: failed to load vocab-only model from %s\n", vocab_gguf_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    const std::vector<std::pair<std::string, std::string>> case_sources = {
        {"llama.cpp README.md",       "/Users/brianbruggeman/repos/others/llama.cpp/README.md"},
        {"proxima README.md",         "/Users/brianbruggeman/repos/slot-0/proxima/README.md"},
        {"llama.cpp ngram-map.cpp",   "/Users/brianbruggeman/repos/others/llama.cpp/common/ngram-map.cpp"},
        {"proxima vocab.rs",          "/Users/brianbruggeman/repos/slot-0/proxima/proxima-tokenizer/src/vocab.rs"},
    };

    std::vector<token_stream> streams;
    int32_t next_id = 0;
    for (const auto & entry : case_sources) {
        const std::string text = read_file(entry.second);
        llama_tokens tokens = tokenize_text(vocab, text, /*add_special=*/true);
        streams.push_back({next_id++, entry.second, tokens});
        fprintf(stderr, "[stream %d] %s -> %zu tokens\n", streams.back().id, entry.second.c_str(), tokens.size());
    }

    {
        std::string repeated;
        for (int repeat = 0; repeat < 700; ++repeat) {
            repeated += "Repeat exactly five times: the quick brown fox jumps over the lazy dog. ";
        }
        llama_tokens tokens = tokenize_text(vocab, repeated, /*add_special=*/true);
        streams.push_back({next_id++, "repeated sentence (real text, 700x)", tokens});
        fprintf(stderr, "[stream %d] repeated sentence -> %zu tokens\n", streams.back().id, tokens.size());
    }

    // constructed to force ngram-mod's single-slot-overwrite weakness: the
    // same >=24 token COMMON prefix precedes seven distinct real-sentence
    // tails, so each successive occurrence's drafted continuation (chained
    // through the PREVIOUS tail, still sitting in the shared hash table)
    // mismatches the actual next tail -- see README.md "ngram-mod reset
    // mechanism" for why this is deterministic, not chance-based.
    {
        const std::string common_prefix =
            "According to the shared hash table every preceding window of exactly "
            "twenty four tokens determines which single token the drafter predicts "
            "will come next in the generated sequence. ";
        const std::vector<std::string> tails = {
            "Zebras migrate across the savanna in tightly coordinated herds, using subtle vocal cues and visual stripe patterns to keep the group together even when visibility drops during the dry season storms that periodically sweep through the region each year without much warning to the animals living nearby. ",
            "Quantum computers manipulate superposed qubits through carefully tuned microwave pulses, exploiting entanglement to perform certain calculations exponentially faster than any classical machine could achieve using the same amount of physical hardware and comparable levels of ambient thermal noise during the experiment. ",
            "Volcanic eruptions release enormous quantities of ash and sulfur dioxide into the stratosphere, occasionally cooling the planet for several years afterward as the fine particles reflect incoming sunlight back into space before eventually settling out of the atmosphere entirely over subsequent seasons. ",
            "Orchestral conductors rely on subtle hand gestures and eye contact to guide dozens of musicians through tempo changes that would be nearly impossible to communicate verbally during a live performance in front of a large and attentive audience of paying subscribers. ",
            "Coral reefs support roughly a quarter of all known marine species despite covering less than one percent of the ocean floor, making their rapid decline from warming waters a significant concern for global biodiversity researchers studying long term ecological trends worldwide. ",
            "Mountain glaciers retreat measurably each decade as global temperatures rise, exposing bare rock that had remained hidden beneath thick ice for many thousands of years before modern instruments began recording these dramatic seasonal changes in ice mass. ",
            "Deep sea vents host chemosynthetic bacteria that convert dissolved minerals directly into usable energy, forming the base of an entire food web that survives without any sunlight whatsoever at crushing pressures far below the reach of surface expeditions. ",
        };
        std::string trap;
        for (const auto & tail : tails) {
            trap += common_prefix;
            trap += tail;
        }
        llama_tokens tokens = tokenize_text(vocab, trap, /*add_special=*/true);
        streams.push_back({next_id++, "ngram-mod low-acceptance trap (constructed, real sentences)", tokens});
        fprintf(stderr, "[stream %d] ngram-mod trap -> %zu tokens\n", streams.back().id, tokens.size());
    }

    // forces ngram-map key_only (R5) and k4v (R13) to diverge. Two prior
    // constructions measured and rejected: a 2-way A/B alternation (the
    // n+m+1=61 token minimum match-search gap, common/ngram-map.cpp:278-312,
    // skips the immediately preceding opposite-tail occurrence and lands two
    // cycles back -- SAME parity -- so match_pos always matched reality: 1
    // /144 differed) and a 4-way A/B/C/D rotation (ambiguity is only a
    // TRANSIENT tie right as a second distinct continuation first appears;
    // once one value slot's count pulls ahead the guard stops firing: 2/258
    // differed). Each distinct 12-token key gets its own independent
    // common_ngram_map_key with its own value slots, so instead of trying to
    // sustain one long ambiguous run, this repeats the transient-tie trap
    // across 40 INDEPENDENT keys: key_i + tail_A once, then key_i + tail_B
    // three times. At key_i's second occurrence, common_ngram_map_draft's
    // value-tally loop (:412-456) has just created slot0=tail_A(count=1)
    // and slot1=tail_B(count=1) -- a tie -- so the ambiguity guard
    // `sum_occur > 0 && max_occur < 2*sum_occur` (:495-499) fires for k4v
    // (drafts nothing) while key_only (:382-398) has no such guard and
    // drafts match_pos's content regardless.
    {
        const std::vector<std::string> tails = {
            "Freshwater eels travel thousands of kilometers back to a single spawning ground in the ocean, guided by magnetic and olfactory cues that researchers still do not fully understand despite decades of dedicated field studies across multiple continents. ",
            "Suspension bridges distribute enormous structural loads through high tension cables anchored deep into bedrock, allowing a roadway to span distances that a rigid beam of the same material could never support on its own. ",
            "Beekeepers inspect each hive frame by frame during the warmer months, checking for a healthy laying pattern from the queen and removing any comb that shows early signs of disease before it can spread to neighboring colonies. ",
            "Pipe organs route pressurized air through thousands of individually tuned pipes of varying length and material, each producing a single pitch and timbre that the organist selects by pulling a labeled stop on the console. ",
        };
        // the matched "key" is only ever the LAST size_key=12 tokens before
        // the tail begins (common/ngram-map.cpp:246-252). A standalone
        // size_key=3 diagnostic (scratch/diag.cpp) reproduced the exact
        // tie-breaking divergence cleanly in isolation on a single fresh
        // common_ngram_map: at a key's SECOND occurrence (the moment
        // slot0=tail_a(count=1) and slot1=tail_b(count=1) first tie), k4v's
        // ambiguity guard (common/ngram-map.cpp:495-499) suppresses the
        // draft while key_only (:382-398) does not. Two prior constructions
        // measured and rejected: concatenating 250 scenarios into ONE stream
        // (so they share one accumulating map, since replay_ngram_map below
        // constructs one map per STREAM) produced only 8 divergences no
        // matter the scenario count (80 or 250, identically 8) -- something
        // in the shared map's long-run state suppresses the effect after the
        // first handful; and one stream PER scenario produced zero, because
        // the match search requires `size_last_begin > n+m+1=61`
        // (common/ngram-map.cpp:278) and a growing gap beyond that
        // (:298-312), so a ~113-token single-scenario stream never grows
        // large enough for any match to be findable at all. The fix
        // combines both lessons: batch several scenarios per stream (long
        // enough, and with a real `size_last_begin` from begin(), for
        // matches to become findable) but keep each batch SHORT enough that
        // it stays within the handful of scenarios that reliably tie before
        // the shared-map falloff.
        for (int batch = 0; batch < 6; ++batch) {
            std::string batch_text;
            for (int scenario = 0; scenario < 10; ++scenario) {
                const int global_scenario = batch * 10 + scenario;
                std::string key_phrase =
                    "for lookup and matching purposes the recurring context marker "
                    "number is " + std::to_string(global_scenario) + " right here now. ";
                const std::string & tail_a = tails[global_scenario % tails.size()];
                const std::string & tail_b = tails[(global_scenario + 1) % tails.size()];
                batch_text += key_phrase + tail_a + key_phrase + tail_b;
            }
            llama_tokens tokens = tokenize_text(vocab, batch_text, /*add_special=*/true);
            streams.push_back({next_id++, "ngram-map k vs k4v divergence, batch " + std::to_string(batch) + " (constructed, real sentences)", tokens});
        }
        fprintf(stderr, "[streams 6..%d] ngram-map divergence batches\n", next_id - 1);
    }

    // the streams every drafter type replays over. the ngram-mod occupancy
    // warmup is generated synthetically further below and is never appended
    // here -- it carries no distinguishing content, so it is not part of
    // streams.json at all (see generate_occupancy_warmup_tokens above).
    const std::vector<token_stream> primary_streams(streams.begin(), streams.end());
    write_streams_file(output_dir + "/streams.json", vocab_gguf_path, streams);

    // ngram-simple (R4): defaults from common_params_speculative_ngram_map
    // used for ngram_simple (common/common.h:361-365), size_n/size_m only.
    {
        common_ngram_simple_config config { /* .size_ngram = */ 12, /* .size_mgram = */ 48 };
        auto cases = replay_ngram_simple(primary_streams, config);
        std::string params_json = "{\"size_ngram\": 12, \"size_mgram\": 48}";
        write_cases_file(output_dir + "/ngram_simple.json", "ngram-simple", params_json, {}, cases);
        fprintf(stderr, "[ngram-simple] cases = %zu\n", cases.size());
    }

    // ngram-map-k (R5): key_only = true.
    {
        auto cases = replay_ngram_map(primary_streams, 12, 48, /*key_only=*/true, /*min_hits=*/1);
        std::string params_json = "{\"size_key\": 12, \"size_value\": 48, \"key_only\": true, \"min_hits\": 1}";
        write_cases_file(output_dir + "/ngram_map_k.json", "ngram-map-k", params_json, {}, cases);
        fprintf(stderr, "[ngram-map-k] cases = %zu\n", cases.size());
    }

    // ngram-map-k4v (R13): key_only = false.
    {
        auto cases = replay_ngram_map(primary_streams, 12, 48, /*key_only=*/false, /*min_hits=*/1);
        std::string params_json = "{\"size_key\": 12, \"size_value\": 48, \"key_only\": false, \"min_hits\": 1}";
        write_cases_file(output_dir + "/ngram_map_k4v.json", "ngram-map-k4v", params_json, {}, cases);
        fprintf(stderr, "[ngram-map-k4v] cases = %zu\n", cases.size());
    }

    // ngram-cache (R7): nc_dynamic/nc_static empty, matching the default
    // (no --lookup-cache-static/--lookup-cache-dynamic) common_speculative
    // configuration at common/speculative.cpp:2189-2203.
    {
        common_ngram_cache nc_static_empty;
        auto cases = replay_ngram_cache(primary_streams, /*n_draft=*/8, nc_static_empty);
        std::string params_json = "{\"n_draft\": 8, \"ngram_min\": " + std::to_string(LLAMA_NGRAM_MIN) +
            ", \"ngram_max\": " + std::to_string(LLAMA_NGRAM_MAX) + ", \"nc_dynamic\": \"empty\", \"nc_static\": \"empty\"}";
        write_cases_file(output_dir + "/ngram_cache.json", "ngram-cache", params_json, {}, cases);
        fprintf(stderr, "[ngram-cache] cases = %zu\n", cases.size());

        // save/load round trip fixture: a real static cache built from stream 2 (C++ source).
        common_ngram_cache nc_for_save;
        llama_tokens save_source = primary_streams[2].tokens;
        common_ngram_cache_update(nc_for_save, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, save_source, (int) save_source.size(), false);
        common_ngram_cache_save(nc_for_save, output_dir + "/ngram_cache_static.bin");
        fprintf(stderr, "[ngram-cache] saved static cache with %zu ngram entries from stream %d\n", nc_for_save.size(), primary_streams[2].id);
    }

    // ngram-mod (R6): the SAME spec instance whose draft cases below are
    // recorded into ngram_mod.json first absorbs a synthetic occupancy-warmup
    // stream (generate_occupancy_warmup_tokens above) via begin() on this
    // instance -- not a throwaway instance -- so `ngram_mod.json`'s own
    // recorded cases come from a table that has genuinely undergone
    // llama.cpp's own occupancy reset (`begin()`'s `mod.reset()` at
    // `common/speculative.cpp:~1912`). begin() unconditionally zeroes
    // `sinfo.i_last`/`n_draft_last` on entry (mirrored by this port's own
    // `ngram_mod_begin`), and a reset wipes the table back to fully EMPTY,
    // so the constructed trap stream (streams[5]) below still starts from
    // the near-empty table its own construction comment requires -- the
    // warmup changes NOTHING about the recorded draft content, only proves
    // the reset fired on the instance under test. The constructed trap
    // stream then forces the low-acceptance reset deterministically (see
    // its construction comment above); the other streams contribute
    // general non-empty coverage once begin() is given a real half-stream
    // prompt so the periodic streams train the shared table in one pass
    // instead of waiting on the 32-token incremental-add lag in draft_one
    // (common/speculative.cpp:1940-1946).
    {
        common_params_speculative_ngram_mod mod_params;
        mod_params.n_match = 24;
        mod_params.n_max = 64;
        mod_params.n_min = 48;

        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_MOD };
        params.ngram_mod = mod_params;
        common_speculative * spec = common_speculative_init(params, 1);

        const llama_tokens occupancy_warmup = generate_occupancy_warmup_tokens(
            vocab_size, OCCUPANCY_WARMUP_SEED, OCCUPANCY_WARMUP_LENGTH);
        common_speculative_begin(spec, 0, occupancy_warmup);
        fprintf(stderr, "[ngram-mod] fed synthetic occupancy warmup (%u tokens, seed=%llu, vocab_size=%d) into the recorded instance\n",
                OCCUPANCY_WARMUP_LENGTH, (unsigned long long) OCCUPANCY_WARMUP_SEED, vocab_size);

        std::vector<token_stream> ordered = {
            primary_streams[5], primary_streams[2], primary_streams[0],
            primary_streams[1], primary_streams[3], primary_streams[4],
        };

        std::vector<case_record> cases;
        for (const auto & stream : ordered) {
            if (stream.tokens.size() <= (size_t) mod_params.n_match + 2) {
                continue;
            }
            // the trap stream must start from a near-empty table so its first
            // COMMON occurrence trains rather than pre-resolves; every other
            // stream gets a real half-stream prompt so begin() trains it in
            // one pass instead of relying on the slow incremental-add lag.
            const size_t prompt_len = stream.id == 5
                ? std::min((size_t) mod_params.n_match + 1, stream.tokens.size() / 4)
                : stream.tokens.size() / 2;
            llama_tokens prompt(stream.tokens.begin(), stream.tokens.begin() + prompt_len);
            common_speculative_begin(spec, 0, prompt);

            size_t i = prompt_len;
            size_t stream_cases = 0;
            while (i + 1 < stream.tokens.size() && stream_cases < CASE_CAP_PER_STREAM) {
                llama_tokens history(stream.tokens.begin(), stream.tokens.begin() + i);

                common_speculative_draft_params & dparams = common_speculative_get_draft_params(spec, 0);
                dparams.drafting = true;
                dparams.n_max = -1;
                dparams.pos0 = (llama_pos) i;
                dparams.id_last = stream.tokens[i];
                dparams.prompt = &history;

                llama_tokens draft;
                dparams.result = &draft;

                common_speculative_draft(spec);

                const int32_t accepted = longest_accepted_prefix(draft, stream.tokens, i + 1);
                cases.push_back({stream.id, (int32_t) i, stream.tokens[i], draft, accepted});

                common_speculative_accept(spec, 0, (uint16_t) accepted);
                ++stream_cases;
                i += (size_t) accepted + 1;
            }
        }
        common_speculative_free(spec);

        std::string params_json = "{\"n_match\": " + std::to_string(mod_params.n_match) +
            ", \"n_max\": " + std::to_string(mod_params.n_max) +
            ", \"n_min\": " + std::to_string(mod_params.n_min) +
            ", \"table_size\": 4194304, \"occupancy_threshold\": 0.25, \"low_accept_threshold\": 0.25, \"low_accept_streak\": 5}";
        std::string occupancy_warmup_json = "\"occupancy_warmup\": {\"prng\": \"splitmix64\", \"seed\": " +
            std::to_string(OCCUPANCY_WARMUP_SEED) + ", \"length\": " + std::to_string(OCCUPANCY_WARMUP_LENGTH) +
            ", \"vocab_size\": " + std::to_string(vocab_size) + "}";
        write_cases_file(output_dir + "/ngram_mod.json", "ngram-mod", params_json, {}, cases, occupancy_warmup_json);
        fprintf(stderr, "[ngram-mod] cases = %zu\n", cases.size());
    }

    common_log_flush(common_log_main());

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "done. log at %s\n", log_path.c_str());
    return 0;
}
