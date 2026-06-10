// WHY insert-start/writeback/end are null: pipeann's insert_in_place is a
// single synchronous call with no internal event hooks, so the three-phase
// timing that fnct-hermes exposes does not exist here.

#include <pthread.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// Forward-declare friend struct before pipeann headers pull in pq_nbr.h.
struct Sift100mPipeannBench;

#include "arrow/api.h"
#include "arrow/io/file.h"
#include "dynamic_index.h"
#include "linux_aligned_file_reader.h"
#include "parquet/arrow/writer.h"
#include "utils/index_build_utils.h"

// ── constants ────────────────────────────────────────────────────────────────

static constexpr uint32_t DIM            = 128;
static constexpr uint64_t N              = 100'000'000ULL;
static constexpr uint64_t QLEN           = 10'000;
static constexpr uint32_t MAX_DEGREE     = 32;
static constexpr float    ALPHA          = 1.2f;
static constexpr uint32_t L_BUILD        = 128;
static constexpr uint32_t PQ_CHUNKS      = 32;
static constexpr uint32_t PQ_K           = 256;
static constexpr uint64_t PQ_TRAIN_SIZE  = 256'000;
static constexpr uint32_t NUM_KMEANS_REP = 15;

struct SearchCfg { uint32_t k; uint32_t beam; uint32_t starts; };
static constexpr SearchCfg SEARCH_CFGS[] = {
    {1, 64, 1}, {1, 128, 1}, {10, 128, 1}, {10, 256, 1}
};
static constexpr int N_CFGS = (int)(sizeof(SEARCH_CFGS)/sizeof(SEARCH_CFGS[0]));

// ── pinning ──────────────────────────────────────────────────────────────────

static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs);
    cpu %= (int)std::thread::hardware_concurrency();
    CPU_SET((size_t)cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// ── RSS ──────────────────────────────────────────────────────────────────────

static double rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            double kb = 0; sscanf(line.c_str() + 6, "%lf", &kb);
            return kb / 1024.0;
        }
    }
    return 0.0;
}

// ── watch-dog (mirrors fnct-bench/src/lib.rs:425) ────────────────────────────

static void watch_dog(const char *tag,
                      const std::atomic<uint64_t> &done, uint64_t total,
                      std::mutex &mu, std::condition_variable &cv,
                      const bool &stop) {
    uint64_t prev = 0;
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::seconds(10), [&]{ return stop; });
            if (stop) break;
        }
        uint64_t d = done.load(std::memory_order_relaxed);
        if (d >= total) break;
        double ips = (double)(d - prev) / 10.0;
        fprintf(stderr, "[%s] %llu/%llu %.0f/s rss=%.0fMB\n",
                tag, (unsigned long long)d, (unsigned long long)total,
                ips, rss_mb());
        prev = d;
    }
    uint64_t d = done.load(std::memory_order_relaxed);
    fprintf(stderr, "[%s] %llu/%llu done rss=%.0fMB\n",
            tag, (unsigned long long)d, (unsigned long long)total, rss_mb());
}

// ── PQ friend — calls private generate_pq_pivots / generate_pq_data_from_pivots ──

struct Sift100mPipeannBench {
    static void train_pq_on_prefix(pipeann::PQNeighbor<uint8_t> &nbr,
                                   const float *train_float, uint64_t n_train,
                                   const std::string &pq_pivots_path,
                                   const std::string &batch0_bin,
                                   const std::string &pq_compressed_path) {
        nbr.generate_pq_pivots(train_float, n_train, DIM, PQ_K, PQ_CHUNKS,
                               NUM_KMEANS_REP, pq_pivots_path);
        nbr.generate_pq_data_from_pivots(batch0_bin, PQ_K, PQ_CHUNKS,
                                         pq_pivots_path, pq_compressed_path, 0);
        nbr.npoints = N;
    }
};

// ── wrapper: prevents build_disk_index from re-training PQ ───────────────────

struct PretrainedPQ : pipeann::AbstractNeighbor<uint8_t> {
    pipeann::PQNeighbor<uint8_t> *real_;
    explicit PretrainedPQ(pipeann::PQNeighbor<uint8_t> *r)
        : pipeann::AbstractNeighbor<uint8_t>(r->metric), real_(r) {
        npoints = r->npoints;
    }
    uint64_t query_ctx_size() override { return real_->query_ctx_size(); }
    std::string get_name() override { return real_->get_name(); }
    pipeann::AbstractNeighbor<uint8_t> *shuffle(
        const libcuckoo::cuckoohash_map<uint32_t,uint32_t> &m,
        uint64_t n, uint32_t t) override { return real_->shuffle(m, n, t); }
    void initialize_query(const uint8_t *q, pipeann::QueryBuffer *b) override {
        real_->initialize_query(q, b); }
    void compute_dists(pipeann::QueryBuffer *b,
                       const uint32_t *ids, uint64_t n) override {
        real_->compute_dists(b, ids, n); }
    void compute_dists(uint32_t qid, const uint32_t *ids, uint64_t n,
                       float *d, uint8_t *s) override {
        real_->compute_dists(qid, ids, n, d, s); }
    void load(const char *p) override { real_->load(p); }
    void save(const char *p) override { real_->save(p); }
    // WHY no-op: PQ pivots were trained on exactly the first PQ_TRAIN_SIZE
    // points; block automatic re-training that build_disk_index would trigger.
    void build(const std::string &, const std::string &, uint32_t) override {
        npoints = N; real_->npoints = N; }
    void insert(uint8_t *pt, uint32_t loc) override { real_->insert(pt, loc); }
    ~PretrainedPQ() override = default;
};

// ── write pipeann .bin (4-byte npts + 4-byte dim header) ─────────────────────

static void write_bin(const std::string &raw_path, uint64_t begin, uint64_t n,
                      uint32_t dim, const std::string &dst) {
    std::ifstream src(raw_path, std::ios::binary);
    if (!src) {
        fprintf(stderr, "ERROR: open %s\n", raw_path.c_str());
        std::exit(1);
    }
    src.seekg((std::streamoff)(begin * dim), std::ios::beg);
    std::vector<uint8_t> buf(n * dim);
    src.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)(n * dim));
    std::ofstream out(dst, std::ios::binary);
    uint32_t n32 = (uint32_t)n, d32 = dim;
    out.write(reinterpret_cast<const char *>(&n32), 4);
    out.write(reinterpret_cast<const char *>(&d32), 4);
    out.write(reinterpret_cast<const char *>(buf.data()),
              (std::streamsize)(n * dim));
}

// ── Arrow/Parquet output ─────────────────────────────────────────────────────

static std::shared_ptr<arrow::Schema> make_schema() {
    auto rf = arrow::FieldVector{
        arrow::field("dist",  arrow::float64(), false),
        arrow::field("id",    arrow::uint32(),  false),
        arrow::field("value", arrow::uint64(),  false),
    };
    auto rtype = arrow::struct_(rf);
    return arrow::schema({
        arrow::field("tag",              arrow::utf8(),   false),
        arrow::field("op",               arrow::utf8(),   false),
        arrow::field("index",            arrow::uint64(), false),
        arrow::field("tid",              arrow::uint32(), false),
        arrow::field("results",
            arrow::list(arrow::field("item", rtype, true)), true),
        arrow::field("insert-start",     arrow::uint64(), true),
        arrow::field("insert-writeback", arrow::uint64(), true),
        arrow::field("insert-end",       arrow::uint64(), true),
        arrow::field("batch-elapsed-ns", arrow::uint64(), true),
    });
}

struct Row {
    const char *tag = nullptr;
    const char *op  = nullptr;
    uint64_t    idx = 0;
    uint32_t    tid = 0;
    std::vector<std::tuple<double,uint32_t,uint64_t>> results;
    bool        has_results      = false;
    bool        has_elapsed      = false;
    uint64_t    batch_elapsed_ns = 0;
};

static void flush(std::shared_ptr<parquet::arrow::FileWriter> &w,
                  const std::vector<Row> &rows,
                  const std::shared_ptr<arrow::Schema> &schema) {
    if (rows.empty()) return;
    int64_t n = (int64_t)rows.size();
    auto pool = arrow::default_memory_pool();

    arrow::StringBuilder tag_b, op_b;
    arrow::UInt64Builder idx_b;
    arrow::UInt32Builder tid_b;
    for (auto &r : rows) {
        tag_b.Append(r.tag).ok(); op_b.Append(r.op).ok();
        idx_b.Append(r.idx).ok(); tid_b.Append(r.tid).ok();
    }
    std::shared_ptr<arrow::Array> tag_a, op_a, idx_a, tid_a;
    tag_b.Finish(&tag_a).ok(); op_b.Finish(&op_a).ok();
    idx_b.Finish(&idx_a).ok(); tid_b.Finish(&tid_a).ok();

    auto rf = arrow::FieldVector{
        arrow::field("dist",  arrow::float64(), false),
        arrow::field("id",    arrow::uint32(),  false),
        arrow::field("value", arrow::uint64(),  false),
    };
    auto dist_b = std::make_shared<arrow::DoubleBuilder>();
    auto id_b   = std::make_shared<arrow::UInt32Builder>();
    auto val_b  = std::make_shared<arrow::UInt64Builder>();
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> ch{dist_b, id_b, val_b};
    auto sb = std::make_shared<arrow::StructBuilder>(
        arrow::struct_(rf), pool, ch);
    auto lb = std::make_shared<arrow::ListBuilder>(
        pool, sb, arrow::field("item", arrow::struct_(rf), true));
    for (auto &r : rows) {
        if (r.has_results) {
            lb->Append().ok();
            for (auto &[d,id,v] : r.results) {
                dist_b->Append(d).ok();
                id_b->Append(id).ok();
                val_b->Append(v).ok();
                sb->Append(true).ok();
            }
        } else {
            lb->AppendNull().ok();
        }
    }
    std::shared_ptr<arrow::Array> res_a; lb->Finish(&res_a).ok();

    arrow::UInt64Builder is_b, iw_b, ie_b;
    is_b.AppendNulls(n).ok(); iw_b.AppendNulls(n).ok(); ie_b.AppendNulls(n).ok();
    std::shared_ptr<arrow::Array> is_a, iw_a, ie_a;
    is_b.Finish(&is_a).ok(); iw_b.Finish(&iw_a).ok(); ie_b.Finish(&ie_a).ok();

    arrow::UInt64Builder be_b;
    for (auto &r : rows) {
        if (r.has_elapsed) be_b.Append(r.batch_elapsed_ns).ok();
        else               be_b.AppendNull().ok();
    }
    std::shared_ptr<arrow::Array> be_a; be_b.Finish(&be_a).ok();

    auto batch = arrow::RecordBatch::Make(schema, n,
        {tag_a, op_a, idx_a, tid_a, res_a, is_a, iw_a, ie_a, be_a});
    w->WriteRecordBatch(*batch).ok();
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    auto req_env = [](const char *name) -> std::string {
        const char *v = std::getenv(name);
        if (!v || v[0] == '\0') {
            fprintf(stderr, "ERROR: %s must be set\n", name);
            std::exit(1);
        }
        return v;
    };

    const std::string data_path    = req_env("DATA");
    const std::string queries_path = req_env("QUERIES");
    const std::string index_prefix = req_env("INDEX");
    const std::string output_path  = req_env("OUTPUT");

    uint64_t BATCH = 20'000'000ULL;
    if (const char *b = std::getenv("BATCH")) BATCH = std::stoull(b);

    const uint32_t threads   = std::max(1u, std::thread::hardware_concurrency());
    const uint64_t n_batches = N / BATCH;

    auto schema = make_schema();
    auto arrow_file =
        arrow::io::FileOutputStream::Open(output_path).ValueOrDie();
    parquet::WriterProperties::Builder pb;
    pb.compression(parquet::Compression::SNAPPY);
    auto pq_writer = parquet::arrow::FileWriter::Open(
        *schema, arrow::default_memory_pool(),
        arrow_file, pb.build()).ValueOrDie();

    // ── STEP 1: PQ training on FIRST PQ_TRAIN_SIZE points ───────────────────
    // This is the FIRST data touch — before any disk index build or insert.
    const std::string batch0_bin         = index_prefix + "_batch0.bin";
    const std::string pq_pivots_path     = index_prefix + "_pq_pivots.bin";
    const std::string pq_compressed_path = index_prefix + "_pq_compressed.bin";

    {
        std::vector<float> train_float(PQ_TRAIN_SIZE * DIM);
        {
            std::ifstream df(data_path, std::ios::binary);
            if (!df) {
                fprintf(stderr, "ERROR: cannot open DATA=%s\n",
                        data_path.c_str());
                return 1;
            }
            std::vector<uint8_t> raw(PQ_TRAIN_SIZE * DIM);
            df.read(reinterpret_cast<char *>(raw.data()),
                    (std::streamsize)(PQ_TRAIN_SIZE * DIM));
            for (uint64_t i = 0; i < PQ_TRAIN_SIZE * DIM; i++)
                train_float[i] = static_cast<float>(raw[i]);
        }

        // write batch-0 as pipeann .bin for build_disk_index and PQ encoding
        write_bin(data_path, 0, BATCH, DIM, batch0_bin);

        auto *pq_nbr = new pipeann::PQNeighbor<uint8_t>(pipeann::Metric::L2);
        Sift100mPipeannBench::train_pq_on_prefix(
            *pq_nbr, train_float.data(), PQ_TRAIN_SIZE,
            pq_pivots_path, batch0_bin, pq_compressed_path);
        // reload so the table is populated for search distance queries
        pq_nbr->load(index_prefix.c_str());

        // ── STEP 2: offline build SSDIndex on [0, BATCH) ────────────────────
        auto build_t0 = std::chrono::steady_clock::now();
        auto *wrapper = new PretrainedPQ(pq_nbr);
        pipeann::build_disk_index<uint8_t, uint32_t>(
            batch0_bin.c_str(), index_prefix.c_str(),
            MAX_DEGREE, L_BUILD, /*M=*/0, threads, PQ_CHUNKS,
            pipeann::Metric::L2, /*tag_file=*/nullptr,
            wrapper, /*attr_writer=*/nullptr);
        uint64_t build_elapsed_ns =
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - build_t0).count();

        // ── STEP 3: open SSDIndex for streaming ─────────────────────────────
        auto reader = std::make_shared<LinuxAlignedFileReader>();
        auto *search_pq = new pipeann::PQNeighbor<uint8_t>(pipeann::Metric::L2);
        pipeann::IndexBuildParameters idx_params;
        idx_params.R           = MAX_DEGREE;
        idx_params.L           = L_BUILD;
        idx_params.alpha       = ALPHA;
        idx_params.num_threads = threads;
        auto ssd = std::make_unique<pipeann::SSDIndex<uint8_t, uint32_t>>(
            pipeann::Metric::L2, reader, search_pq,
            /*tags=*/true, &idx_params);
        ssd->load(index_prefix.c_str(), /*enable_writes=*/true);

        // load queries (raw uint8, no header)
        std::vector<uint8_t> queries(QLEN * DIM);
        {
            std::ifstream qf(queries_path, std::ios::binary);
            if (!qf) {
                fprintf(stderr, "ERROR: open QUERIES=%s\n",
                        queries_path.c_str());
                return 1;
            }
            qf.read(reinterpret_cast<char *>(queries.data()),
                    (std::streamsize)(QLEN * DIM));
        }

        // ── batch loop ───────────────────────────────────────────────────────
        for (uint64_t b = 0; b < n_batches; b++) {
            uint64_t base_offset = b * BATCH;
            // LEAK tag string — pointer stays valid for the lifetime of the run
            const char *ins_tag =
                new std::string("insert_" + std::to_string(b))->c_str();

            std::vector<Row> batch_rows;

            if (b == 0) {
                // batch 0 was built offline; emit insert rows without inserting
                std::atomic<uint64_t> done(0);
                std::mutex mu; std::condition_variable cv; bool stop = false;
                std::thread wd(
                    [&]{ watch_dog(ins_tag, done, BATCH, mu, cv, stop); });

                batch_rows.resize(BATCH);
                std::vector<std::thread> ts;
                for (uint32_t tid = 0; tid < threads; tid++) {
                    ts.emplace_back([&, tid]() {
                        pin((int)tid);
                        uint64_t lo = BATCH * tid / threads;
                        uint64_t hi = BATCH * (tid+1) / threads;
                        for (uint64_t i = lo; i < hi; i++) {
                            batch_rows[i] = {ins_tag, "insert", i, tid};
                            done.fetch_add(1, std::memory_order_relaxed);
                        }
                    });
                }
                for (auto &t : ts) t.join();
                { std::lock_guard<std::mutex> lk(mu); stop = true; }
                cv.notify_one(); wd.join();

                Row summ; summ.tag = ins_tag; summ.op = "batch-summary";
                summ.idx = 0; summ.tid = 0;
                summ.has_elapsed = true;
                summ.batch_elapsed_ns = build_elapsed_ns;
                batch_rows.push_back(summ);

            } else {
                // stream-insert [b*BATCH, (b+1)*BATCH) via insert_in_place
                std::vector<uint8_t> batch_data(BATCH * DIM);
                {
                    std::ifstream df(data_path, std::ios::binary);
                    df.seekg((std::streamoff)(base_offset * DIM), std::ios::beg);
                    df.read(reinterpret_cast<char *>(batch_data.data()),
                            (std::streamsize)(BATCH * DIM));
                }

                auto ins_t0 = std::chrono::steady_clock::now();
                std::atomic<uint64_t> done(0);
                std::mutex mu; std::condition_variable cv; bool stop = false;
                std::thread wd(
                    [&]{ watch_dog(ins_tag, done, BATCH, mu, cv, stop); });

                batch_rows.resize(BATCH);
                std::vector<std::thread> ts;
                for (uint32_t tid = 0; tid < threads; tid++) {
                    ts.emplace_back([&, tid]() {
                        pin((int)tid);
                        uint64_t lo = BATCH * tid / threads;
                        uint64_t hi = BATCH * (tid+1) / threads;
                        for (uint64_t i = lo; i < hi; i++) {
                            uint32_t gid = (uint32_t)(base_offset + i);
                            ssd->insert_in_place(
                                batch_data.data() + i * DIM, gid);
                            batch_rows[i] = {
                                ins_tag, "insert", base_offset + i, tid};
                            done.fetch_add(1, std::memory_order_relaxed);
                        }
                    });
                }
                for (auto &t : ts) t.join();
                { std::lock_guard<std::mutex> lk(mu); stop = true; }
                cv.notify_one(); wd.join();

                uint64_t elapsed_ns =
                    (uint64_t)std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - ins_t0).count();
                Row summ; summ.tag = ins_tag; summ.op = "batch-summary";
                summ.idx = 0; summ.tid = 0;
                summ.has_elapsed = true;
                summ.batch_elapsed_ns = elapsed_ns;
                batch_rows.push_back(summ);
            }

            flush(pq_writer, batch_rows, schema);
            batch_rows.clear();

            // search batch_id is incremented BEFORE search (matches Rust script)
            uint64_t sbid = b + 1;

            for (int ci = 0; ci < N_CFGS; ci++) {
                auto cfg = SEARCH_CFGS[ci];
                // LEAK tag string
                const char *stag = new std::string(
                    "search_b" + std::to_string(sbid) +
                    "_" + std::to_string(cfg.k) +
                    "_" + std::to_string(cfg.beam) +
                    "_" + std::to_string(cfg.starts)
                )->c_str();

                std::atomic<uint64_t> done(0);
                std::mutex mu; std::condition_variable cv; bool stop = false;
                std::thread wd(
                    [&]{ watch_dog(stag, done, QLEN, mu, cv, stop); });

                std::vector<Row> srows(QLEN);
                std::vector<std::thread> ts;
                for (uint32_t tid = 0; tid < threads; tid++) {
                    ts.emplace_back([&, tid, cfg, stag]() {
                        pin((int)tid);
                        uint64_t lo = QLEN * tid / threads;
                        uint64_t hi = QLEN * (tid+1) / threads;
                        std::vector<uint32_t> rids(cfg.k);
                        std::vector<float>    rdists(cfg.k);
                        for (uint64_t qi = lo; qi < hi; qi++) {
                            const uint8_t *q = queries.data() + qi * DIM;
                            ssd->beam_search(q, cfg.k, /*mem_L=*/0, cfg.beam,
                                             rids.data(), rdists.data(),
                                             cfg.beam);
                            std::vector<std::tuple<double,uint32_t,uint64_t>>
                                rv;
                            rv.reserve(cfg.k);
                            for (uint32_t j = 0; j < cfg.k; j++) {
                                // value = id (pipeann has no separate value)
                                rv.emplace_back((double)rdists[j], rids[j],
                                               (uint64_t)rids[j]);
                            }
                            Row r; r.tag = stag; r.op = "search";
                            r.idx = qi; r.tid = tid;
                            r.results = std::move(rv); r.has_results = true;
                            srows[qi] = std::move(r);
                            done.fetch_add(1, std::memory_order_relaxed);
                        }
                    });
                }
                for (auto &t : ts) t.join();
                { std::lock_guard<std::mutex> lk(mu); stop = true; }
                cv.notify_one(); wd.join();

                flush(pq_writer, srows, schema);
            }
        }
    }

    pq_writer->Close().ok();
    arrow_file->Close().ok();
    return 0;
}
