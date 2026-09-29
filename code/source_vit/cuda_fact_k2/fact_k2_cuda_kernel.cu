// Custom CUDA kernel for FourierActivation at K=2 (fourier_layers.py):
//
//   phi(t) = a0 + a1*cos(w*t) + b1*sin(w*t) + a2*cos(2*w*t) + b2*sin(2*w*t)
//
// applied elementwise over the last dimension (size M) of an arbitrary-shape
// input, with coefficients {a0, a, b} either literally global (P=1, the
// "_global" variant the paper's shared-activation runs use) or
// per-feature (P=M, shared=False). w is a fixed python float (not learned --
// see true_fourier_coeffs), so it is passed as a plain scalar.
//
// Forward and the analytic backward (grad wrt input AND wrt a0/a/b) are both
// hand-written here instead of relying on autograd to unroll K=2 trig calls,
// which is the whole point of a dedicated kernel: one fused elementwise pass
// each way instead of ~8 separate elementwise CUDA kernels PyTorch's autograd
// would otherwise launch (cos, sin, four multiplies, two adds -- doubled for
// backward).
//
// PRECISION: the trig is evaluated in `acc_t` -- at::acc_type<scalar_t>, i.e.
// float for half/bfloat16/float inputs and double for double inputs -- rather
// than promoting everything to double. An earlier version did all four trig
// calls in double regardless of input dtype, which cost ~5.7x on hardware with
// a low FP64 rate -- an L4 runs FP64 at 1/64 of FP32, and sincos(double) is a
// software routine on top of that -- for no accuracy: the result is stored back
// as float anyway, so the max forward error against a float64 reference is
// 3.5e-7 either way (see bench_fact_k2_variants.py, which measures this and
// seven other ways of getting sin/cos, including [-pi,pi] lookup tables). The
// {a0, a, b} gradients still accumulate in double, because those reduce tens of
// millions of summands into five scalars and fp32 would lose the answer.
//
// DTYPES: the input/grad_input may be half, bfloat16, float or double
// (`scalar_t`), but the coefficients {a0, a, b} are ALWAYS acc_t -- fp32 for
// the three narrow types, fp64 for double -- and so is the output. This is not
// a kernel-side choice: it reproduces exactly what the reference
// fourier_layers.FourierActivation does under torch.amp.autocast. Its forward
// computes `ang = t.unsqueeze(-1) * (self.kvec * self.w)`, and `mul` carries no
// autocast policy, so ordinary type promotion against the fp32 kvec buffer
// lifts a half input to fp32 before any trig runs; the whole series, and the
// returned tensor, are fp32 there too. Feeding this kernel fp32 coefficients
// and handing back an fp32 result therefore keeps a CUDA-kernel run
// numerically comparable to the pure-PyTorch fact_k2_global runs it is being
// compared against, instead of silently doing the series in fp16.
#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int THREADS = 256;
// Cap on grid-stride blocks for the reduction kernels -- keeps the number of
// atomicAdd's into the 5 (or 5*M) accumulator slots bounded regardless of
// how huge `total` is, instead of one block per THREADS elements.
constexpr int MAX_REDUCE_BLOCKS = 2048;

inline int64_t num_blocks(int64_t total, int64_t threads) {
  return (total + threads - 1) / threads;
}

// Dispatch sin+cos to the intrinsic matching scalar_t, so a float tensor never
// pays for double-precision transcendentals. See the PRECISION note above.
__device__ __forceinline__ void sincos_t(float x, float* s, float* c) {
  sincosf(x, s, c);
}
__device__ __forceinline__ void sincos_t(double x, double* s, double* c) {
  sincos(x, s, c);
}

// ---------------------------------------------------------------------- //
// Forward
// ---------------------------------------------------------------------- //
template <typename scalar_t, typename acc_t>
__global__ void fact_k2_forward_kernel(
    const scalar_t* __restrict__ input,
    const acc_t* __restrict__ a0,   // (P,)
    const acc_t* __restrict__ a,    // (P, 2)
    const acc_t* __restrict__ b,    // (P, 2)
    acc_t* __restrict__ output,
    const int64_t total, const int64_t M, const int64_t P, const double w) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const int64_t p = (P == 1) ? 0 : feat;
  const acc_t wt = (acc_t)w * static_cast<acc_t>(input[idx]);
  acc_t s1, c1, s2, c2;
  sincos_t(wt, &s1, &c1);
  sincos_t((acc_t)2 * wt, &s2, &c2);
  output[idx] = a0[p]
      + a[p * 2 + 0] * c1 + b[p * 2 + 0] * s1
      + a[p * 2 + 1] * c2 + b[p * 2 + 1] * s2;
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt input (elementwise, no reduction)
// ---------------------------------------------------------------------- //
template <typename scalar_t>
__global__ void fact_k2_backward_input_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ grad_input,
    const int64_t total, const int64_t M, const int64_t P, const double w) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const int64_t p = (P == 1) ? 0 : feat;
  const scalar_t ws = (scalar_t)w;
  const scalar_t wt = ws * input[idx];
  scalar_t s1, c1, s2, c2;
  sincos_t(wt, &s1, &c1);
  sincos_t((scalar_t)2 * wt, &s2, &c2);
  const scalar_t a1 = a[p * 2 + 0], b1 = b[p * 2 + 0];
  const scalar_t a2 = a[p * 2 + 1], b2 = b[p * 2 + 1];
  // d(phi)/dt = w*(-a1*sin(wt) + b1*cos(wt)) + 2w*(-a2*sin(2wt) + b2*cos(2wt))
  const scalar_t dphidt = ws * (-a1 * s1 + b1 * c1)
                        + (scalar_t)2 * ws * (-a2 * s2 + b2 * c2);
  grad_input[idx] = grad_output[idx] * dphidt;
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt {a0, a, b}, P == 1 (shared/global) case.
// Block-level shared-memory reduction (5 scalars/thread) before a single
// atomicAdd per block -- avoids every one of `total` threads hammering the
// same 5 global accumulators, which is the common case here since every
// shared-activation (global) configuration uses shared=True.
// ---------------------------------------------------------------------- //
// NEED_W adds a sixth accumulator: d(phi)/dw for a LEARNABLE fundamental
// frequency (the fact_kK_global_lw variants). It is a compile-time template
// parameter, not a runtime flag, so every fixed-w call site keeps exactly the
// register/shared-memory footprint and instruction stream it had before -- the
// whole w branch is dead code the compiler removes. `a`/`b` are only read when
// NEED_W; grad wrt {a0, a, b} never needed them.
//
//   phi(t)     = a0 + sum_k [ a_k cos(k w t) + b_k sin(k w t) ]
//   dphi/dw    = sum_k k t [ -a_k sin(k w t) + b_k cos(k w t) ]
//
// w is ONE scalar for the whole network, so its gradient is a full reduction
// over every element -- the same block-reduce-then-one-atomicAdd shape the
// other five already use, riding the sincos this loop has already paid for.
template <typename scalar_t, bool NEED_W>
__global__ void fact_k2_backward_params_shared_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ a,     // (1, 2), read only when NEED_W
    const scalar_t* __restrict__ b,     // (1, 2), read only when NEED_W
    double* __restrict__ grad_a0,   // (1,)
    double* __restrict__ grad_a,    // (2,)
    double* __restrict__ grad_b,    // (2,)
    double* __restrict__ grad_w,    // (), written only when NEED_W
    const int64_t total, const double w) {
  extern __shared__ double sdata[];
  double* s_a0 = sdata;
  double* s_a1 = sdata + blockDim.x;
  double* s_b1 = sdata + 2 * blockDim.x;
  double* s_a2 = sdata + 3 * blockDim.x;
  double* s_b2 = sdata + 4 * blockDim.x;
  double* s_w  = sdata + 5 * blockDim.x;   // only allocated when NEED_W

  const int tid = threadIdx.x;
  // Hoisted out of the loop: shared (P==1) means these four are the same for
  // every element, and NEED_W is the only case that reads them at all.
  const scalar_t a1c = NEED_W ? a[0] : (scalar_t)0;
  const scalar_t a2c = NEED_W ? a[1] : (scalar_t)0;
  const scalar_t b1c = NEED_W ? b[0] : (scalar_t)0;
  const scalar_t b2c = NEED_W ? b[1] : (scalar_t)0;

  double la0 = 0.0, la1 = 0.0, lb1 = 0.0, la2 = 0.0, lb2 = 0.0, lw = 0.0;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < total; idx += stride) {
    const scalar_t go = grad_output[idx];
    const scalar_t ti = input[idx];
    const scalar_t wt = (scalar_t)w * ti;
    scalar_t s1, c1, s2, c2;
    sincos_t(wt, &s1, &c1);
    sincos_t((scalar_t)2 * wt, &s2, &c2);
    // trig at scalar_t precision, but the running sums stay in double --
    // `total` is routinely 1e7-1e8 here.
    la0 += (double)go;
    la1 += (double)(go * c1); lb1 += (double)(go * s1);
    la2 += (double)(go * c2); lb2 += (double)(go * s2);
    if (NEED_W) {
      const scalar_t dphidw = ti * (-a1c * s1 + b1c * c1)
                            + (scalar_t)2 * ti * (-a2c * s2 + b2c * c2);
      lw += (double)(go * dphidw);
    }
  }
  s_a0[tid] = la0; s_a1[tid] = la1; s_b1[tid] = lb1; s_a2[tid] = la2; s_b2[tid] = lb2;
  if (NEED_W) s_w[tid] = lw;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_a0[tid] += s_a0[tid + s];
      s_a1[tid] += s_a1[tid + s];
      s_b1[tid] += s_b1[tid + s];
      s_a2[tid] += s_a2[tid + s];
      s_b2[tid] += s_b2[tid + s];
      if (NEED_W) s_w[tid] += s_w[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(grad_a0, s_a0[0]);
    atomicAdd(&grad_a[0], s_a1[0]);
    atomicAdd(&grad_b[0], s_b1[0]);
    atomicAdd(&grad_a[1], s_a2[0]);
    atomicAdd(&grad_b[1], s_b2[0]);
    if (NEED_W) atomicAdd(grad_w, s_w[0]);
  }
}

// ---------------------------------------------------------------------- //
// Backward: grad wrt {a0, a, b}, P == M (per-feature) case.
// Contention is spread across M independent accumulator slots, so a direct
// atomicAdd per element (no block reduction) is adequate.
// ---------------------------------------------------------------------- //
template <typename scalar_t>
__global__ void fact_k2_backward_params_perfeature_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    double* __restrict__ grad_a0,   // (M,)
    double* __restrict__ grad_a,    // (M, 2)
    double* __restrict__ grad_b,    // (M, 2)
    const int64_t total, const int64_t M, const double w) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t feat = idx % M;
  const scalar_t go = grad_output[idx];
  const scalar_t wt = (scalar_t)w * input[idx];
  scalar_t s1, c1, s2, c2;
  sincos_t(wt, &s1, &c1);
  sincos_t((scalar_t)2 * wt, &s2, &c2);
  atomicAdd(&grad_a0[feat], (double)go);
  atomicAdd(&grad_a[feat * 2 + 0], (double)(go * c1));
  atomicAdd(&grad_b[feat * 2 + 0], (double)(go * s1));
  atomicAdd(&grad_a[feat * 2 + 1], (double)(go * c2));
  atomicAdd(&grad_b[feat * 2 + 1], (double)(go * s2));
}

}  // namespace

// ---------------------------------------------------------------------- //
// Host launchers
// ---------------------------------------------------------------------- //
torch::Tensor fact_k2_cuda_forward(
    torch::Tensor input, torch::Tensor a0, torch::Tensor a, torch::Tensor b, double w) {
  const int64_t total = input.numel();
  const int64_t M = input.size(-1);
  const int64_t P = a0.size(0);
  auto output = torch::empty_like(input);
  const int64_t blocks = num_blocks(total, THREADS);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fact_k2_forward_cuda", ([&] {
    fact_k2_forward_kernel<scalar_t><<<blocks, THREADS>>>(
        input.data_ptr<scalar_t>(), a0.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), total, M, P, w);
  }));
  return output;
}

std::vector<torch::Tensor> fact_k2_cuda_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor a0,
    torch::Tensor a, torch::Tensor b, double w, bool need_grad_w) {
  const int64_t total = input.numel();
  const int64_t M = input.size(-1);
  const int64_t P = a0.size(0);
  const int64_t blocks = num_blocks(total, THREADS);

  auto grad_input = torch::empty_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fact_k2_backward_input_cuda", ([&] {
    fact_k2_backward_input_kernel<scalar_t><<<blocks, THREADS>>>(
        grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(), total, M, P, w);
  }));

  auto opts_d = torch::TensorOptions().dtype(torch::kFloat64).device(input.device());
  auto grad_a0_d = torch::zeros({P}, opts_d);
  auto grad_a_d = torch::zeros({P, 2}, opts_d);
  auto grad_b_d = torch::zeros({P, 2}, opts_d);
  auto grad_w_d = torch::zeros({}, opts_d);

  // A learnable w is one scalar for the WHOLE network, which is only
  // meaningful for the shared/global parameterisation (P == 1) -- every
  // fact_kK_global_lw variant is shared=True by construction. Refuse rather
  // than silently returning a zero gradient for a per-feature caller.
  TORCH_CHECK(!need_grad_w || P == 1,
              "grad wrt w is only supported for the shared (P == 1) "
              "parameterisation; got P = ", P);

  if (P == 1) {
    const int64_t reduce_blocks = std::min<int64_t>(blocks, MAX_REDUCE_BLOCKS);
    const size_t shmem = (need_grad_w ? 6 : 5) * THREADS * sizeof(double);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fact_k2_backward_params_shared_cuda", ([&] {
      if (need_grad_w) {
        fact_k2_backward_params_shared_kernel<scalar_t, true><<<reduce_blocks, THREADS, shmem>>>(
            grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            grad_a0_d.data_ptr<double>(), grad_a_d.data_ptr<double>(),
            grad_b_d.data_ptr<double>(), grad_w_d.data_ptr<double>(), total, w);
      } else {
        fact_k2_backward_params_shared_kernel<scalar_t, false><<<reduce_blocks, THREADS, shmem>>>(
            grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            grad_a0_d.data_ptr<double>(), grad_a_d.data_ptr<double>(),
            grad_b_d.data_ptr<double>(), grad_w_d.data_ptr<double>(), total, w);
      }
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fact_k2_backward_params_perfeature_cuda", ([&] {
      fact_k2_backward_params_perfeature_kernel<scalar_t><<<blocks, THREADS>>>(
          grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          grad_a0_d.data_ptr<double>(), grad_a_d.data_ptr<double>(),
          grad_b_d.data_ptr<double>(), total, M, w);
    }));
  }

  return {grad_input, grad_a0_d.to(a0.scalar_type()),
          grad_a_d.to(a.scalar_type()), grad_b_d.to(b.scalar_type()),
          grad_w_d.to(a0.scalar_type())};
}

// ---------------------------------------------------------------------- //
// Python bindings
// ---------------------------------------------------------------------- //
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fact_k2_forward(
    torch::Tensor input, torch::Tensor a0, torch::Tensor a, torch::Tensor b, double w) {
  CHECK_INPUT(input); CHECK_INPUT(a0); CHECK_INPUT(a); CHECK_INPUT(b);
  TORCH_CHECK(a0.size(0) == a.size(0) && a0.size(0) == b.size(0),
              "a0, a, b must share the same leading (param) dimension P");
  TORCH_CHECK(a.size(-1) == 2 && b.size(-1) == 2, "a, b must have last dim 2 (K=2)");
  TORCH_CHECK(a0.size(0) == 1 || a0.size(0) == input.size(-1),
              "a0's param dim P must be 1 (shared) or equal to input.size(-1) (per-feature)");
  return fact_k2_cuda_forward(input, a0, a, b, w);
}

std::vector<torch::Tensor> fact_k2_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor a0,
    torch::Tensor a, torch::Tensor b, double w, bool need_grad_w) {
  CHECK_INPUT(grad_output); CHECK_INPUT(input); CHECK_INPUT(a0); CHECK_INPUT(a); CHECK_INPUT(b);
  return fact_k2_cuda_backward(grad_output, input, a0, a, b, w, need_grad_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fact_k2_forward, "FAct K=2 forward (CUDA)");
  m.def("backward", &fact_k2_backward, "FAct K=2 backward (CUDA)",
        py::arg("grad_output"), py::arg("input"), py::arg("a0"), py::arg("a"),
        py::arg("b"), py::arg("w"), py::arg("need_grad_w") = false);
}
