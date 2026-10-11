// AdsRank C++ inference benchmark.
//
// Replays real fine-ranking requests (one query x K candidates) through the
// exported DeepFM graph with ONNX Runtime and reports per-request latency
// percentiles as JSON lines.
//
//   bench <model.onnx> <requests.bin> <mode> <intra_threads> [iters] [warmup]
//
// mode = naive   default SessionOptions; every request builds a fresh input
//                vector + tensor and lets Run() allocate the output
//        tuned   explicit intra-op threads, inter-op 1, sequential executor,
//                spin-wait on, and one pinned input/output buffer pair bound
//                through IoBinding and reused for every request
//
// intra_threads = 0 leaves ORT's default (one per physical core).

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Requests {
  int64_t n = 0, batch = 0, fields = 0;
  std::vector<int64_t> data;
  const int64_t* at(int64_t i) const { return data.data() + (i % n) * batch * fields; }
  size_t elems() const { return static_cast<size_t>(batch * fields); }
};

Requests load_requests(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Requests r;
  int64_t hdr[3];
  f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
  r.n = hdr[0], r.batch = hdr[1], r.fields = hdr[2];
  r.data.resize(static_cast<size_t>(r.n * r.batch * r.fields));
  f.read(reinterpret_cast<char*>(r.data.data()), r.data.size() * sizeof(int64_t));
  if (!f) throw std::runtime_error("short read on " + path);
  return r;
}

double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q * (v.size() - 1);
  size_t lo = static_cast<size_t>(pos);
  size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

using Clock = std::chrono::steady_clock;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s model.onnx requests.bin naive|tuned threads [iters] [warmup]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1], req_path = argv[2], mode = argv[3];
  const int threads = std::stoi(argv[4]);
  const int iters = argc > 5 ? std::stoi(argv[5]) : 2000;
  const int warmup = argc > 6 ? std::stoi(argv[6]) : 100;
  const bool tuned = mode == "tuned";

  Requests reqs = load_requests(req_path);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "adsrank");
  Ort::SessionOptions so;
  if (threads > 0) so.SetIntraOpNumThreads(threads);
  if (tuned) {
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.AddConfigEntry("session.intra_op.allow_spinning", "1");
  }
  Ort::Session session(env, model.c_str(), so);

  const char* in_names[] = {"x"};
  const char* out_names[] = {"p"};
  const std::array<int64_t, 2> in_shape{reqs.batch, reqs.fields};
  const std::array<int64_t, 1> out_shape{reqs.batch};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // tuned-mode state: allocated once, reused for every request
  std::vector<int64_t> in_buf(reqs.elems());
  std::vector<float> out_buf(static_cast<size_t>(reqs.batch));
  Ort::Value in_t = Ort::Value::CreateTensor<int64_t>(mem, in_buf.data(), in_buf.size(), in_shape.data(), 2);
  Ort::Value out_t = Ort::Value::CreateTensor<float>(mem, out_buf.data(), out_buf.size(), out_shape.data(), 1);
  Ort::IoBinding binding(session);
  binding.BindInput("x", in_t);
  binding.BindOutput("p", out_t);
  Ort::RunOptions run_opts;

  double checksum = 0.0;
  auto run_one = [&](int64_t i) {
    const int64_t* src = reqs.at(i);
    if (tuned) {
      std::memcpy(in_buf.data(), src, reqs.elems() * sizeof(int64_t));
      session.Run(run_opts, binding);
      checksum += out_buf[0];
    } else {
      std::vector<int64_t> input(src, src + reqs.elems());
      Ort::Value x = Ort::Value::CreateTensor<int64_t>(mem, input.data(), input.size(), in_shape.data(), 2);
      auto out = session.Run(run_opts, in_names, &x, 1, out_names, 1);
      checksum += out[0].GetTensorData<float>()[0];
    }
  };

  for (int i = 0; i < warmup; ++i) run_one(i);
  std::vector<double> lat;
  lat.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    run_one(i);
    lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  double mean = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
  std::printf(
      "{\"mode\":\"%s\",\"threads\":%d,\"batch\":%lld,\"iters\":%d,"
      "\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f,\"mean\":%.4f,\"checksum\":%.6f}\n",
      mode.c_str(), threads, static_cast<long long>(reqs.batch), iters, pct(lat, 0.50), pct(lat, 0.95),
      pct(lat, 0.99), mean, checksum);
  return 0;
}
