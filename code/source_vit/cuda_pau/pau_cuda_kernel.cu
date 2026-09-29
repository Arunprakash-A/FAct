// Custom CUDA kernel for PAU (fourier_layers.py's Pade Activation Unit):
//
//   f(t) = P(t) / Q(t),  P(t) = sum_{j=0..M} a_j t^j,
//                        Q(t) = 1 + sum_{k=1..N} |b_k| t^k
//
// order (M, N) = (5, 4) -- the paper's own default and the only order this
// study uses -- fixed as compile-time constants exactly like fact_k2/k9's
// kernels fix K, so the per-element loops unroll and every coefficient
// tensor has a known static shape. Coefficients are either literally global
// (P=1, the "pau_global" variant every PAU study in this repo uses) or
// per-feature (P=M_features, shared=False), same P-dimension convention as
// fact_k2/k9's a0/a/b and PAU's own `shared` flag.
//
// No trig, no w: this is a plain rational function of t itself, so forward
// is just Horner-style power accumulation, not a sincos/recurrence kernel --
// the closest analogue to fact_k2/k9_cuda_kernel.cu's design is the backward
// PARAMETER reduction (same block-shared-memory-then-atomicAdd strategy),
// not the forward math.
//
// GRADIENTS (matches fourier_layers.PAU.forward()'s autograd graph exactly,
// verified numerically in test_pau_cuda.py against that pure-PyTorch class):
//   d f/d t     = (P'(t) Q(t) - P(t) Q'(t)) / Q(t)^2
//                 P'(t) = sum_{j=1..M} j a_j t^(j-1)
//                 Q'(t) = sum_{k=1..N} k |b_k| t^(k-1)
//   d f/d a_j   = t^j / Q(t)                              (P depends linearly on a_j, Q does not)
//   d f/d b_k   = -(P(t) t^k / Q(t)^2) * sign(b_k)         (chain rule through |b_k|; forward()
//                                                            takes b.abs(), so this rederives
//                                                            PyTorch's own abs() subgradient,
//                                                            sign(0) := 0)
//
// PRECISION / DTYPES: same policy as fact_k2/k9 -- per-element math runs in
// the input's own scalar_t (fp32 during AMP-autocast-disabled fp32 upcast,
// see pau_module.py), and the {a, b} gradient reduction accumulates in
// double regardless of input dtype (same many-summands-into-few-scalars
// argument as the Fourier kernels).
#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int M_ORDER = 5;      // numerator degree -> M_ORDER+1 = 6 coeffs (a_0..a_5)
constexpr int N_ORDER = 4;      // denominator degree -> N_ORDER = 4 coeffs (b_1..b_4)
constexpr int NA = M_ORDER + 1; // 6
constexpr int NB = N_ORDER;     // 4
constexpr int THREADS = 256;
constexpr int MAX_REDUCE_BLOCKS = 2048;
// NA grad_a slots + NB grad_b slots per thread in the shared-memory reduction.
constexpr int NPARAMS = NA + NB;

inline int64_t num_blocks(int64_t total, int64_t threads) {
  return (total + threads - 1) / threads;
}

template <typename T>
__device__ __forceinline__ T sign_t(T x) {
  return (x > (T)0) ? (T)1 : ((x < (T)0) ? (T)-1 : (T)0);
}

// tpow[0..M_ORDER] = t^0 .. t^M_ORDER (M_ORDER=5 always >= N_ORDER=4 here,
// so this one array covers both P's and Q's powers).
template <typename acc_t>
__device__ __forceinline__ void powers(acc_t t, acc_t* tpow) {
  tpow[0] = (acc_t)1;
  #pragma unroll
  for (int i = 1; i <= M_ORDER; ++i) tpow[i] = tpow[i - 1] * t;
}

// ---------------------------------------------------------------------- //
// Forward
// ---------------------------------------------------------------------- //
template <typename scalar_t, typename acc_t>
__global__ void pau_forward_kernel(
    const scalar_t* __restrict__ input,
    const acc_t* __restrict__ a,   // (P, NA)
    const acc_t* __restrict__ b,   // (P, NB)
    acc_t* __restrict__ output,
    const int64_t total, const int64_t M, const int64_t P) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const int64_t p = (P == 1) ? 0 : feat;
  const acc_t t = static_cast<acc_t>(input[idx]);
  acc_t tpow[M_ORDER + 1];
  powers<acc_t>(t, tpow);

  acc_t P_val = (acc_t)0;
  #pragma unroll
  for (int j = 0; j <= M_ORDER; ++j) P_val += a[p * NA + j] * tpow[j];

  acc_t Q_val = (acc_t)1;
  #pragma unroll
  for (int k = 1; k <= N_ORDER; ++k) Q_val += fabs(b[p * NB + (k - 1)]) * tpow[k];

  output[idx] = P_val / Q_val;
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt input (elementwise, no reduction)
// ---------------------------------------------------------------------- //
template <typename scalar_t>
__global__ void pau_backward_input_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ grad_input,
    const int64_t total, const int64_t M, const int64_t P) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const int64_t p = (P == 1) ? 0 : feat;
  const scalar_t t = input[idx];
  scalar_t tpow[M_ORDER + 1];
  powers<scalar_t>(t, tpow);

  scalar_t P_val = (scalar_t)0, Pprime = (scalar_t)0;
  #pragma unroll
  for (int j = 0; j <= M_ORDER; ++j) {
    const scalar_t aj = a[p * NA + j];
    P_val += aj * tpow[j];
    if (j >= 1) Pprime += (scalar_t)j * aj * tpow[j - 1];
  }

  scalar_t Q_val = (scalar_t)1, Qprime = (scalar_t)0;
  #pragma unroll
  for (int k = 1; k <= N_ORDER; ++k) {
    const scalar_t bk_abs = fabs(b[p * NB + (k - 1)]);
    Q_val += bk_abs * tpow[k];
    Qprime += (scalar_t)k * bk_abs * tpow[k - 1];
  }

  const scalar_t dfdt = (Pprime * Q_val - P_val * Qprime) / (Q_val * Q_val);
  grad_input[idx] = grad_output[idx] * dfdt;
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt {a, b}, P == 1 (shared/global) case.
// One block-level shared-memory reduction over NPARAMS=10 running sums per
// thread, then a single atomicAdd per slot per block -- same strategy as
// fact_k2/k9's params-shared kernel, except `a` must also be passed in here
// (unlike those kernels' b-gradient, PAU's d f/d b_k needs P(t), which needs
// `a`).
// ---------------------------------------------------------------------- //
template <typename scalar_t>
__global__ void pau_backward_params_shared_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a,   // (1, NA)
    const scalar_t* __restrict__ b,   // (1, NB)
    double* __restrict__ grad_a,      // (NA,)
    double* __restrict__ grad_b,      // (NB,)
    const int64_t total) {
  extern __shared__ double sdata[];  // NPARAMS * blockDim.x doubles

  const int tid = threadIdx.x;
  double local[NPARAMS];
  #pragma unroll
  for (int i = 0; i < NPARAMS; ++i) local[i] = 0.0;

  scalar_t bsign[NB];
  #pragma unroll
  for (int k = 0; k < NB; ++k) bsign[k] = sign_t<scalar_t>(b[k]);

  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < total; idx += stride) {
    const scalar_t go = grad_output[idx];
    const scalar_t t = input[idx];
    scalar_t tpow[M_ORDER + 1];
    powers<scalar_t>(t, tpow);

    scalar_t P_val = (scalar_t)0;
    #pragma unroll
    for (int j = 0; j <= M_ORDER; ++j) P_val += a[j] * tpow[j];

    scalar_t Q_val = (scalar_t)1;
    #pragma unroll
    for (int k = 1; k <= N_ORDER; ++k) Q_val += fabs(b[k - 1]) * tpow[k];

    const scalar_t inv_Q = (scalar_t)1 / Q_val;
    const scalar_t inv_Q2_P = P_val * inv_Q * inv_Q;

    #pragma unroll
    for (int j = 0; j <= M_ORDER; ++j) local[j] += (double)(go * tpow[j] * inv_Q);
    #pragma unroll
    for (int k = 1; k <= N_ORDER; ++k) {
      local[NA + (k - 1)] += (double)(-go * inv_Q2_P * tpow[k] * bsign[k - 1]);
    }
  }
  #pragma unroll
  for (int i = 0; i < NPARAMS; ++i) sdata[i * blockDim.x + tid] = local[i];
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int i = 0; i < NPARAMS; ++i) sdata[i * blockDim.x + tid] += sdata[i * blockDim.x + tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    #pragma unroll
    for (int j = 0; j <= M_ORDER; ++j) atomicAdd(&grad_a[j], sdata[j * blockDim.x]);
    #pragma unroll
    for (int k = 1; k <= N_ORDER; ++k) atomicAdd(&grad_b[k - 1], sdata[(NA + (k - 1)) * blockDim.x]);
  }
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt {a, b}, P == M (per-feature) case.
// ---------------------------------------------------------------------- //
template <typename scalar_t>
__global__ void pau_backward_params_perfeature_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a,   // (M, NA)
    const scalar_t* __restrict__ b,   // (M, NB)
    double* __restrict__ grad_a,      // (M, NA)
    double* __restrict__ grad_b,      // (M, NB)
    const int64_t total, const int64_t M) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const scalar_t go = grad_output[idx];
  const scalar_t t = input[idx];
  scalar_t tpow[M_ORDER + 1];
  powers<scalar_t>(t, tpow);

  scalar_t P_val = (scalar_t)0;
  #pragma unroll
  for (int j = 0; j <= M_ORDER; ++j) P_val += a[feat * NA + j] * tpow[j];

  scalar_t Q_val = (scalar_t)1;
  #pragma unroll
  for (int k = 1; k <= N_ORDER; ++k) Q_val += fabs(b[feat * NB + (k - 1)]) * tpow[k];

  const scalar_t inv_Q = (scalar_t)1 / Q_val;
  const scalar_t inv_Q2_P = P_val * inv_Q * inv_Q;

  #pragma unroll
  for (int j = 0; j <= M_ORDER; ++j) atomicAdd(&grad_a[feat * NA + j], (double)(go * tpow[j] * inv_Q));
  #pragma unroll
  for (int k = 1; k <= N_ORDER; ++k) {
    const scalar_t bsign = sign_t<scalar_t>(b[feat * NB + (k - 1)]);
    atomicAdd(&grad_b[feat * NB + (k - 1)], (double)(-go * inv_Q2_P * tpow[k] * bsign));
  }
}

}  // namespace

// ---------------------------------------------------------------------- //
// Host launchers
// ---------------------------------------------------------------------- //
torch::Tensor pau_cuda_forward(torch::Tensor input, torch::Tensor a, torch::Tensor b) {
  const int64_t total = input.numel();
  const int64_t M = input.size(-1);
  const int64_t P = a.size(0);
  auto output = torch::empty_like(input);
  const int64_t blocks = num_blocks(total, THREADS);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pau_forward_cuda", ([&] {
    pau_forward_kernel<scalar_t><<<blocks, THREADS>>>(
        input.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), total, M, P);
  }));
  return output;
}

std::vector<torch::Tensor> pau_cuda_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor a, torch::Tensor b) {
  const int64_t total = input.numel();
  const int64_t M = input.size(-1);
  const int64_t P = a.size(0);
  const int64_t blocks = num_blocks(total, THREADS);

  auto grad_input = torch::empty_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pau_backward_input_cuda", ([&] {
    pau_backward_input_kernel<scalar_t><<<blocks, THREADS>>>(
        grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(), total, M, P);
  }));

  auto opts_d = torch::TensorOptions().dtype(torch::kFloat64).device(input.device());
  auto grad_a_d = torch::zeros({P, NA}, opts_d);
  auto grad_b_d = torch::zeros({P, NB}, opts_d);

  if (P == 1) {
    const int64_t reduce_blocks = std::min<int64_t>(blocks, MAX_REDUCE_BLOCKS);
    const size_t shmem = (size_t)NPARAMS * THREADS * sizeof(double);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pau_backward_params_shared_cuda", ([&] {
      pau_backward_params_shared_kernel<scalar_t><<<reduce_blocks, THREADS, shmem>>>(
          grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
          grad_a_d.data_ptr<double>(), grad_b_d.data_ptr<double>(), total);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pau_backward_params_perfeature_cuda", ([&] {
      pau_backward_params_perfeature_kernel<scalar_t><<<blocks, THREADS>>>(
          grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
          grad_a_d.data_ptr<double>(), grad_b_d.data_ptr<double>(), total, M);
    }));
  }

  return {grad_input, grad_a_d.to(a.scalar_type()), grad_b_d.to(b.scalar_type())};
}

// ---------------------------------------------------------------------- //
// Python bindings
// ---------------------------------------------------------------------- //
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor pau_forward(torch::Tensor input, torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(input); CHECK_INPUT(a); CHECK_INPUT(b);
  TORCH_CHECK(a.size(0) == b.size(0), "a, b must share the same leading (param) dimension P");
  TORCH_CHECK(a.size(-1) == NA, "a must have last dim 6 (order-5 numerator)");
  TORCH_CHECK(b.size(-1) == NB, "b must have last dim 4 (order-4 denominator)");
  TORCH_CHECK(a.size(0) == 1 || a.size(0) == input.size(-1),
              "a/b's param dim P must be 1 (shared) or equal to input.size(-1) (per-feature)");
  return pau_cuda_forward(input, a, b);
}

std::vector<torch::Tensor> pau_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(grad_output); CHECK_INPUT(input); CHECK_INPUT(a); CHECK_INPUT(b);
  return pau_cuda_backward(grad_output, input, a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pau_forward, "PAU forward (CUDA)");
  m.def("backward", &pau_backward, "PAU backward (CUDA)");
}
