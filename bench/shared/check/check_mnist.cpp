// Real-MNIST convergence gate — GPU only. Trains a 784-256-10 ReLU MLP with
// cross-entropy to a real test-accuracy target on the actual MNIST test set.
// Unlike bench_mlp_train (synthetic, fixed-step throughput), this is a
// correctness/quality gate: it asserts the whole GPU op stack (GEMM + relu +
// softmax + axis-sum + broadcast-add, hand-rolled backprop) can drive a real
// model to convergence, not just run fast. A few seconds on the RTX 3090.
//
// Follows the check_cuda.cpp skip-as-pass convention: no GPU (e.g. CI) -> print
// and return 0. Data is fetched once via curl into a cache dir and reused;
// if the fetch fails (offline), skip-as-pass too — the gate only bites where a
// GPU AND data are both present, which is exactly the dev box. Deterministic
// (fixed init seed + fixed data order), so the accuracy is reproducible and the
// threshold is not flaky.
//
//   check_mnist [data_dir]   (default ./mnist-data)

#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kIn = 784;
constexpr int64_t kHid = 256;
constexpr int64_t kOut = 10;
constexpr int64_t kBatch = 256;
constexpr int kEpochs = 20;
constexpr float kLr = 0.1f;
constexpr float kThreshold = 0.96f;  // observed ~0.97-0.98; margin for atomics

using tl::array;

// MNIST IDX: big-endian magic + dims, then raw uint8. Images -> [N,784] in
// [0,1]; labels -> [N] ints. Returns false if the file is absent/short.
uint32_t be32(std::ifstream& f) {
  unsigned char b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

bool load_images(const std::string& path, std::vector<float>& out, int64_t& n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  if (be32(f) != 2051) return false;
  n = be32(f);
  int64_t rows = be32(f), cols = be32(f);
  int64_t px = rows * cols;
  std::vector<unsigned char> raw(n * px);
  f.read(reinterpret_cast<char*>(raw.data()), raw.size());
  if (!f) return false;
  out.resize(n * px);
  for (int64_t i = 0; i < n * px; i++) out[i] = raw[i] / 255.0f;
  return true;
}

bool load_labels(const std::string& path, std::vector<int>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  if (be32(f) != 2049) return false;
  int64_t n = be32(f);
  std::vector<unsigned char> raw(n);
  f.read(reinterpret_cast<char*>(raw.data()), n);
  if (!f) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

// Fetch the four IDX files into dir via curl+gunzip if missing. Best-effort;
// caller re-checks for the files afterward.
void ensure_data(const std::string& dir) {
  const char* files[] = {"train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"};
  std::ifstream probe(dir + "/" + files[0], std::ios::binary);
  if (probe.good()) return;  // already cached
  if (std::system("command -v curl >/dev/null 2>&1") != 0) return;
  std::printf("[mnist] fetching MNIST into %s (once) ...\n", dir.c_str());
  std::string mk = "mkdir -p '" + dir + "'";
  if (std::system(mk.c_str()) != 0) return;
  const char* base = "https://storage.googleapis.com/cvdf-datasets/mnist/";
  for (const char* fn : files) {
    std::string cmd = "curl -sSL -o '" + dir + "/" + fn + ".gz' " + base + fn +
                      ".gz && gunzip -f '" + dir + "/" + fn + ".gz'";
    if (std::system(cmd.c_str()) != 0)
      std::printf("[mnist] fetch failed for %s\n", fn);
  }
}

array from_vec(std::vector<float> v, tl::shape_t s) {
  return array::from(std::move(v), std::move(s));
}

array he_init(tl::shape_t shape, unsigned seed, int64_t fan_in) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / fan_in));
  size_t n = 1;
  for (auto x : shape) n *= static_cast<size_t>(x);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return from_vec(std::move(v), std::move(shape));
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "mnist-data";

  if (!tl::gpu_available()) {
    std::printf("[mnist] no GPU device — skipping convergence gate (pass)\n");
    return 0;
  }
  ensure_data(dir);

  std::vector<float> Xtr_h, Xte_h;
  std::vector<int> ytr, yte;
  int64_t ntr = 0, nte = 0;
  if (!load_images(dir + "/train-images-idx3-ubyte", Xtr_h, ntr) ||
      !load_labels(dir + "/train-labels-idx1-ubyte", ytr) ||
      !load_images(dir + "/t10k-images-idx3-ubyte", Xte_h, nte) ||
      !load_labels(dir + "/t10k-labels-idx1-ubyte", yte)) {
    std::printf("[mnist] data unavailable in %s — skipping gate (pass)\n",
                dir.c_str());
    return 0;
  }

  // Deterministic shuffle of the train set once (fixed order every epoch).
  std::vector<int> perm(ntr);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 g(1234);
  std::shuffle(perm.begin(), perm.end(), g);
  std::vector<float> Xs(ntr * kIn);
  std::vector<float> Ys(ntr * kOut, 0.0f);  // one-hot
  for (int64_t i = 0; i < ntr; i++) {
    int src = perm[i];
    std::copy_n(&Xtr_h[src * kIn], kIn, &Xs[i * kIn]);
    Ys[i * kOut + ytr[src]] = 1.0f;
  }

  tl::use_gpu();
  array X = from_vec(std::move(Xs), {ntr, kIn});      // resident on device
  array Y = from_vec(std::move(Ys), {ntr, kOut});
  array Xte = from_vec(std::move(Xte_h), {nte, kIn});

  array W1 = he_init({kIn, kHid}, 1, kIn);
  array b1 = from_vec(std::vector<float>(kHid, 0.0f), {1, kHid});
  array W2 = he_init({kHid, kOut}, 3, kHid);
  array b2 = from_vec(std::vector<float>(kOut, 0.0f), {1, kOut});

  const int64_t steps = ntr / kBatch;
  const float invB = 1.0f / static_cast<float>(kBatch);
  auto t0 = std::chrono::steady_clock::now();
  for (int e = 0; e < kEpochs; e++) {
    for (int64_t s = 0; s < steps; s++) {
      array xb = X.slice(s * kBatch, kBatch);  // [B,784] contiguous view
      array yb = Y.slice(s * kBatch, kBatch);  // [B,10]

      array z1 = xb.dot(W1) + b1;
      array h = z1.relu();
      array z2 = h.dot(W2) + b2;
      array p = z2.softmax();

      array dz2 = (p - yb) * invB;
      array dW2 = h.transpose().dot(dz2);
      array db2 = dz2.sum(0, true);
      array dh = dz2.dot(W2.transpose());
      array dz1 = dh * (z1 > 0.0f);  // relu'(z1)
      array dW1 = xb.transpose().dot(dz1);
      array db1 = dz1.sum(0, true);

      W1 = (W1 - dW1 * kLr).eval();
      b1 = (b1 - db1 * kLr).eval();
      W2 = (W2 - dW2 * kLr).eval();
      b2 = (b2 - db2 * kLr).eval();
    }
  }
  double train_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  // Test accuracy over the full 10k held-out set.
  array pred = ((Xte.dot(W1) + b1).relu().dot(W2) + b2).argmax(1);
  const float* pd = pred.data();
  int correct = 0;
  for (int64_t i = 0; i < nte; i++)
    if (static_cast<int>(pd[i] + 0.5f) == yte[i]) correct++;
  double acc = static_cast<double>(correct) / static_cast<double>(nte);
  tl::use_cpu();

  std::printf(
      "[mnist] 784-256-10 ReLU/CE, %d epochs, batch %lld: test acc %.4f "
      "(%d/%lld) in %.2fs\n",
      kEpochs, (long long)kBatch, acc, correct, (long long)nte, train_s);

  if (acc < kThreshold) {
    std::printf("[mnist] FAIL: acc %.4f < threshold %.2f\n", acc, kThreshold);
    return 1;
  }
  std::printf("[mnist] PASS (>= %.2f)\n", kThreshold);
  return 0;
}
