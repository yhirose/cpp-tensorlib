// MNIST-shaped MLP training bench — a real end-to-end dev workload with a
// correctness signal (loss down, accuracy up), fast enough to iterate on.
//
// This library owns the graph/fusion/execution but NOT autograd (culebra keeps
// VJP; a real MNIST *training* run lives there — benchmarks/mnist). So the
// backward pass here is hand-rolled: a 784-128-10 sigmoid MLP with cross-entropy
// is shallow enough that its gradients are a handful of GEMMs + elementwise ops,
// which is exactly the point — it exercises GEMM + sigmoid + softmax + axis-sum
// + broadcast-add end to end on the GPU, the full op mix a trainer hits.
//
// Fast-iterating by construction: SYNTHETIC separable data (no download), a
// single fixed batch (overfits → loss→0, a clean "it learns" signal), and a
// fixed STEPS count — throughput, not convergence. ~sub-second. Backend-agnostic
// array API: runs on CUDA off-Apple, Metal on Apple, CPU otherwise.
//
// Optional args:  bench_mlp_train [steps] [batch]

#include <tensorlib.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int64_t kIn = 784;   // 28*28, MNIST input dim
constexpr int64_t kHid = 128;
constexpr int64_t kOut = 10;   // classes
constexpr float kLr = 0.5f;

using tl::array;

array from_vec(std::vector<float> v, tl::shape_t shape) {
  return array::from(std::move(v), std::move(shape));
}

// Small random init (Gaussian-ish via summed uniforms), deterministic.
array init(tl::shape_t shape, unsigned seed, float scale) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-scale, scale);
  size_t n = 1;
  for (auto x : shape) n *= static_cast<size_t>(x);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return from_vec(std::move(v), std::move(shape));
}

// Synthetic separable dataset: one random prototype per class, samples =
// prototype[label] + noise. A linear-ish model can fit it, so loss must drop.
struct dataset {
  array X, Y;                   // [B,kIn], one-hot [B,kOut]
  std::vector<int> labels;      // for accuracy
};

dataset make_data(int64_t B, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> proto(-1.0f, 1.0f);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  std::vector<std::vector<float>> protos(kOut, std::vector<float>(kIn));
  for (auto& p : protos)
    for (auto& x : p) x = proto(g);
  std::vector<float> X(B * kIn), Y(B * kOut, 0.0f);
  std::vector<int> labels(B);
  for (int64_t i = 0; i < B; i++) {
    int lbl = static_cast<int>(g() % kOut);
    labels[i] = lbl;
    for (int64_t j = 0; j < kIn; j++) X[i * kIn + j] = protos[lbl][j] + noise(g);
    Y[i * kOut + lbl] = 1.0f;
  }
  return {from_vec(std::move(X), {B, kIn}), from_vec(std::move(Y), {B, kOut}),
          std::move(labels)};
}

struct mlp {
  array W1, b1, W2, b2;
  array forward(const array& X) const {
    array h = (X.dot(W1) + b1).sigmoid();  // [B,kHid], broadcast bias
    return (h.dot(W2) + b2).softmax();     // [B,kOut]
  }
};

}  // namespace

int main(int argc, char** argv) {
  int64_t STEPS = argc > 1 ? std::atoll(argv[1]) : 100;
  int64_t B = argc > 2 ? std::atoll(argv[2]) : 256;

  const bool gpu = tl::gpu_available();
  if (gpu)
    tl::use_gpu();
  else
    tl::use_cpu();

  dataset ds = make_data(B, 7);
  mlp m{init({kIn, kHid}, 1, 0.05f), init({1, kHid}, 2, 0.0f),
        init({kHid, kOut}, 3, 0.05f), init({1, kOut}, 4, 0.0f)};

  // loss (cross-entropy) + accuracy over the batch; forces a host read (sync).
  auto evaluate = [&](float& loss, float& acc) {
    array p = m.forward(ds.X);
    array ce = ds.Y * (p + 1e-7f).log();     // [B,kOut]
    loss = -ce.sum() / static_cast<float>(B);
    array pred = p.argmax(1);                // [B] class indices as F32
    const float* pd = pred.data();
    int correct = 0;
    for (int64_t i = 0; i < B; i++)
      if (static_cast<int>(pd[i] + 0.5f) == ds.labels[i]) correct++;
    acc = static_cast<float>(correct) / static_cast<float>(B);
  };

  // One SGD step: forward, hand-rolled backward, in-place param update. eval()
  // materializes each param so the lazy graph doesn't grow across steps.
  const float invB = 1.0f / static_cast<float>(B);
  auto step = [&] {
    array z1 = ds.X.dot(m.W1) + m.b1;
    array h = z1.sigmoid();
    array z2 = h.dot(m.W2) + m.b2;
    array p = z2.softmax();

    array dz2 = (p - ds.Y) * invB;                     // [B,kOut]
    array dW2 = h.transpose().dot(dz2);                // [kHid,kOut]
    array db2 = dz2.sum(0, true);                      // [1,kOut]
    array dh = dz2.dot(m.W2.transpose());              // [B,kHid]
    array dz1 = dh * h * (1.0f - h);                   // sigmoid'(z1) = h(1-h)
    array dW1 = ds.X.transpose().dot(dz1);             // [kIn,kHid]
    array db1 = dz1.sum(0, true);                      // [1,kHid]

    m.W1 = (m.W1 - dW1 * kLr).eval();
    m.b1 = (m.b1 - db1 * kLr).eval();
    m.W2 = (m.W2 - dW2 * kLr).eval();
    m.b2 = (m.b2 - db2 * kLr).eval();
  };

  // FLOPs per step: the five GEMMs (fwd 2 + bwd 3), 2*M*N*K each.
  double step_flops = 2.0 * (B * kIn * kHid + B * kHid * kOut +   // forward
                             kHid * B * kOut + B * kOut * kHid +   // dW2, dh
                             kIn * B * kHid);                      // dW1

  std::printf("tensorlib MLP train bench — %s backend, synthetic MNIST-shape\n",
              gpu ? "GPU" : "CPU");
  std::printf("mlp %lld-%lld-%lld  batch=%lld  steps=%lld  lr=%.2f\n\n",
              (long long)kIn, (long long)kHid, (long long)kOut, (long long)B,
              (long long)STEPS, kLr);

  float loss0, acc0, loss1, acc1;
  evaluate(loss0, acc0);
  for (int i = 0; i < 3; i++) step();  // warmup (not timed)

  auto t0 = std::chrono::steady_clock::now();
  for (int64_t s = 0; s < STEPS; s++) step();
  auto t1 = std::chrono::steady_clock::now();

  evaluate(loss1, acc1);
  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double per_step = total_ms / static_cast<double>(STEPS);

  std::printf("  loss   %7.4f -> %7.4f\n", loss0, loss1);
  std::printf("  acc    %6.1f%% -> %6.1f%%\n", acc0 * 100, acc1 * 100);
  std::printf("  step   %7.3f ms   (%.1f GFLOP/s, %.0f steps/s)\n", per_step,
              step_flops / (per_step * 1e6), 1000.0 / per_step);

  if (gpu) tl::use_cpu();
  return 0;
}
