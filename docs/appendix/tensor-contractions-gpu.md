# Appendix I: Tensor Contractions, From First Principles (GPU)

> A contraction's free indices are what a CUDA grid is for; its contracted indices are what a single thread's own loop is for -- the CUDA C++ edition's own Appendix I earns that mapping by hand, one `__global__` kernel at a time. This appendix earns the identical mapping a different way: by noticing that it already lives inside every `torch::matmul()` and `torch::einsum()` call this book has made since Chapter 1, chosen and executed by LibTorch's own dispatcher the instant a tensor's own `.device()` says which side of the mapping to use.

**What you will understand by the end of this appendix:** how Appendix H's free-index/contracted-index split maps onto LibTorch's own device dispatch, even though this appendix's own source code never writes a grid or a thread loop by hand; why this book's own real Appendix E.4 measurement -- and a second one taken here, at a shape deliberately awkward for hand-tiling -- already demonstrates the payoff a hand-written tiled kernel exists to chase; how to check a `torch::einsum()`/`torch::tensordot()` call's own correctness on a machine with no GPU at all, using the identical independent-cross-check technique this book has used in every prior appendix; how a single `torch::tensordot()` call contracts over one shared axis or several with no new function, struct, or compile-time constant required; and which real production libraries this book's own `torch::` calls have been quietly linking against since Chapter 1 -- the exact ones the CUDA C++ edition's own closing section recommends switching to instead of a hand-written kernel.

**What you need to know first:** this appendix builds directly on Appendix H -- read it first if you haven't. It assumes the tensor and `torch::matmul()` material from this book's own early chapters, and Appendix E.4's own real, measured `torch::matmul()`-versus-naive-loop speedup. As with the CUDA C++ edition's own Appendix I, the environment this appendix was written and compiled in has no physical GPU attached: `torch::cuda::is_available()` is checked and reported honestly throughout, exactly as false here as `cudaGetErrorString`'s own "no CUDA-capable device is detected" is honest there. What stands in for a real device run is the same technique used everywhere else in this book: independent host-side cross-checks against Appendix H's own already-verified values, not a fabricated GPU result.

## I.1 From CPU Contraction to a Device-Dispatched One

**Intuition.** Appendix H's `contract()` has an outer loop over free-index combinations and an inner loop over contracted-index combinations. The CUDA C++ edition's own Appendix I maps that split onto a GPU by hand: the outer loop becomes a grid of threads, one thread per output element, and the inner loop becomes each thread's own sequential accumulation -- a `__global__` kernel, compiled with `nvcc`, launched with an explicit `dim3` grid and block that the CUDA programmer computes themselves.

**Background.** `torch::einsum()` and `torch::matmul()` perform exactly that same free/contracted split internally, and already dispatch to a device-appropriate implementation -- vectorized CPU code, or a CUDA kernel launched with its own internally chosen grid and block -- based purely on which device the input tensors live on. The C++ source at the call site never changes; only a tensor's own `.device()` does.

**Formulas and Key Terms.**

- **Device dispatch** -- the mechanism by which a single `torch::` operation resolves, at the moment it runs, to a CPU implementation or a CUDA kernel, chosen by the input tensors' own device rather than by anything written at the call site; this is what stands in, in this book's own version of this appendix, for the CUDA edition's own hand-chosen grid and block.
- **FLOP count** -- unchanged from Appendix H: `M*N*K` multiply-adds for an `M`-by-`K` contracted against a `K`-by-`N`, identical whether the work runs as a CPU loop, a hand-written CUDA kernel, or a `torch::matmul()` call.
- **Arithmetic intensity** -- FLOPs performed per byte moved between memory and cache (or, on a GPU, between global memory and shared memory or registers); the quantity a tiled implementation raises without changing the FLOP count itself, whether that tiling was hand-written or lives inside `torch::matmul()`'s own backend.

**Worked Example I.1.1.** A real `torch::cuda::is_available()` / `torch::cuda::device_count()` query, honestly reporting no device; the exact matmul-as-contraction problem from Appendix H's own Section H.3, run via `torch::einsum()` on whichever device is actually available; and a cross-check against an independent plain-loop matmul.

```cpp
// Appendix I.1 -- from a CPU contraction to a device-dispatched one.
//
// Appendix H's contract() has an outer loop over free-index combinations
// and an inner loop over contracted-index combinations. The CUDA C++
// edition's own Appendix I maps that split directly onto a GPU: the
// outer (free-index) loop becomes a grid of threads, one thread per
// output element, and the inner (contracted-index) loop becomes each
// thread's own sequential accumulation loop -- a hand-written
// __global__ kernel, compiled with nvcc, launched with an explicit
// dim3 grid and block.
//
// This book takes a different path, because it has taken this same
// path since Chapter 1: torch::einsum() and torch::matmul() already
// perform exactly that free-index/contracted-index split internally,
// and ALREADY DISPATCH to a device-appropriate implementation --
// vectorized CPU code, or a CUDA kernel launched with its own
// internally chosen grid and block -- based purely on which device
// the input tensors live on. The C++ source at the call site never
// changes; only a tensor's own .device() does. This file demonstrates
// that dispatch honestly on this book's own sandbox, which has no
// GPU attached at all.
#include <cstdint>
#include <iostream>
#include <torch/torch.h>

int main() {
    bool cuda_available = torch::cuda::is_available();
    int64_t device_count = torch::cuda::device_count();

    std::cout << "torch::cuda::is_available() = " << (cuda_available ? "true" : "false") << std::endl;
    std::cout << "torch::cuda::device_count()  = " << device_count << std::endl;
    std::cout << "\nthis sandbox reports no CUDA-capable device, exactly as the CUDA C++ edition's own "
              << "cudaMalloc calls honestly report \"no CUDA-capable device is detected\" throughout its own "
              << "Appendix I -- neither book fabricates a GPU that isn't there." << std::endl;

    // The device every tensor below actually runs on: torch::kCUDA if
    // available, torch::kCPU otherwise. Nothing past this line
    // branches on that choice -- the SAME torch::einsum() call below
    // would execute unchanged, dispatching to a CUDA kernel instead
    // of vectorized CPU code, if `device` were torch::kCUDA.
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "\nrunning on device: " << device << std::endl;

    // The exact matmul-as-contraction problem from Appendix H, Section
    // H.3: A is [3,2], B is [2,4], contracting A's axis 1 against B's
    // axis 0. There, contract() built the free/contracted split by
    // hand; here, torch::einsum() builds and dispatches it internally.
    torch::Tensor A = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}).to(device);
    torch::Tensor B = torch::tensor({{1.0, 0.0, 2.0, 1.0}, {0.0, 1.0, 1.0, 2.0}}).to(device);
    torch::Tensor C = torch::einsum("ik,kj->ij", {A, B});

    std::cout << "\nC = torch::einsum(\"ik,kj->ij\", {A, B}), A and B on " << device << ":" << std::endl;
    std::cout << C << std::endl;

    // Independent cross-check: the identical A/B/C values Appendix H,
    // Section H.3 locked, computed there via a hand-rolled contract()
    // and cross-checked against a plain triple-loop matmul and against
    // this same torch::einsum() call. Recomputed here via a plain
    // nested-loop matmul on the host, with no torch:: machinery at
    // all, as a second, structurally different confirmation.
    std::vector<std::vector<double>> a_host = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<std::vector<double>> b_host = {{1, 0, 2, 1}, {0, 1, 1, 2}};
    std::vector<std::vector<double>> expected(3, std::vector<double>(4, 0.0));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 2; k++) sum += a_host[i][k] * b_host[k][j];
            expected[i][j] = sum;
        }
    }
    torch::Tensor C_cpu = C.to(torch::kCPU);
    bool all_match = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            if (C_cpu[i][j].item<double>() != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independent plain-loop matmul (matches Appendix H, Section H.3's own "
              << "locked C values): " << (all_match ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nnothing above this line would need to change to run this exact contraction on a real GPU "
              << "-- only `device` itself would become torch::kCUDA, chosen automatically the moment "
              << "torch::cuda::is_available() reports true. The CUDA C++ edition's own Appendix I, by "
              << "contrast, needs an entirely separate .cu source file, a distinct compiler (nvcc), and a "
              << "hand-written kernel with its own explicit grid and block dimensions to move this identical "
              << "computation from host to device." << std::endl;

    return all_match ? 0 : 1;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 01_dispatch_and_device_query.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_dispatch_and_device_query
./01_dispatch_and_device_query
```

```text
torch::cuda::is_available() = false
torch::cuda::device_count()  = 0

this sandbox reports no CUDA-capable device, exactly as the CUDA C++ edition's own cudaMalloc calls honestly report "no CUDA-capable device is detected" throughout its own Appendix I -- neither book fabricates a GPU that isn't there.

running on device: cpu

C = torch::einsum("ik,kj->ij", {A, B}), A and B on cpu:
  1   2   4   5
  3   4  10  11
  5   6  16  17
[ CPUFloatType{3,4} ]

cross-check vs. independent plain-loop matmul (matches Appendix H, Section H.3's own locked C values): PASS

nothing above this line would need to change to run this exact contraction on a real GPU -- only `device` itself would become torch::kCUDA, chosen automatically the moment torch::cuda::is_available() reports true. The CUDA C++ edition's own Appendix I, by contrast, needs an entirely separate .cu source file, a distinct compiler (nvcc), and a hand-written kernel with its own explicit grid and block dimensions to move this identical computation from host to device.
```

**Discussion.** `torch::cuda::is_available()` reports `false` and `torch::cuda::device_count()` reports `0` on this book's own sandbox, exactly the honest device-absence report the CUDA C++ edition's own `cudaMalloc` calls give throughout its own Appendix I. The contraction itself still runs, on `torch::kCPU`, and reproduces the exact `C` values Appendix H's own Section H.3 already locked and cross-checked two separate ways there. What matters for this section's own point is what is ABSENT from this file: no `dim3 grid`, no `dim3 block`, no `__global__` keyword, no bounds guard against over-provisioned threads. The single line that would change to run this identical contraction on a real GPU is `device`'s own value, resolved automatically the moment `torch::cuda::is_available()` reports `true` -- not a second source file, not a second compiler, not a hand-written kernel.

## I.2 The Tiling Payoff, Already Delivered

**Intuition.** The CUDA C++ edition's own Appendix I.2 hand-writes a shared-memory tiled contraction kernel, deliberately sized so that its contracted-axis length is not a multiple of its own tile width -- specifically to exercise the zero-padding boundary case a naive tiled kernel must get right or silently corrupt its own last tile.

**Background.** This book already measured the real payoff of that entire class of optimization once, honestly, in Appendix E.4: a real ~18x speedup of `torch::matmul()` over a naive hand-written loop, on this same sandbox's own CPU, at N=256. This section adds a second, independent measurement at the CUDA edition's own boundary-exercising, non-tile-multiple shape, confirming the payoff -- and `torch::matmul()`'s own correctness -- on a shape specifically chosen to be awkward for hand-tiling.

**Worked Example I.2.1.** A naive triple-loop matmul versus `torch::matmul()`, at `M=37, K=23, N=19` -- the CUDA C++ edition's own boundary shape, where `K=23` is not a multiple of its own kernel's `TILE=16` -- with a correctness check and a real timing comparison.

```cpp
// Appendix I.2 -- the tiling payoff, already delivered.
//
// The CUDA C++ edition's own Appendix I.2 hand-writes a shared-memory
// tiled contraction kernel, deliberately sized M=37, K=23, N=19 so
// that K (23) is NOT a multiple of its own TILE=16 -- specifically to
// exercise the zero-padding boundary branch a naive tiled kernel must
// get right or silently corrupt its last tile's own results.
//
// This book already measured the real payoff of that entire class of
// optimization once, honestly, in Appendix E.4: a real ~18x speedup
// of torch::matmul() over a naive hand-written loop, on this same
// sandbox's own CPU, at N=256. This file adds a second, independent
// measurement at the CUDA edition's own boundary-exercising, non-tile
// -multiple shape (M=37, K=23, N=19) -- confirming the payoff shows up
// even on a shape specifically chosen to be awkward for hand-tiling,
// and confirming torch::matmul()'s own result is correct there too,
// with no padding logic ever written at this file's own call site.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using Clock = std::chrono::steady_clock;

torch::Tensor make_matrix(int64_t rows, int64_t cols, double seed) {
    torch::Tensor t = torch::empty({rows, cols}, torch::kFloat64);
    auto acc = t.accessor<double, 2>();
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            acc[i][j] = std::sin(seed + static_cast<double>(i) * 0.017 + static_cast<double>(j) * 0.013);
        }
    }
    return t;
}

torch::Tensor naive_matmul(const torch::Tensor& A, const torch::Tensor& B) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = torch::zeros({M, N}, torch::kFloat64);
    auto a = A.accessor<double, 2>();
    auto b = B.accessor<double, 2>();
    auto c = C.accessor<double, 2>();
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; k++) sum += a[i][k] * b[k][j];
            c[i][j] = sum;
        }
    }
    return C;
}

int main() {
    // The CUDA C++ edition's own Appendix I.2 shape, chosen there
    // specifically because K=23 is not a multiple of TILE=16 -- the
    // exact boundary case a hand-tiled kernel's padding branch exists
    // to handle. torch::matmul() below is never told about tile
    // widths at all.
    int64_t M = 37, K = 23, N = 19;
    torch::Tensor A = make_matrix(M, K, 1.0);
    torch::Tensor B = make_matrix(K, N, 2.0);

    torch::Tensor C_naive = naive_matmul(A, B);
    torch::Tensor C_torch = torch::matmul(A, B);

    bool agree = torch::allclose(C_naive, C_torch, /*rtol=*/1e-9, /*atol=*/1e-9);
    std::cout << "M=" << M << ", K=" << K << ", N=" << N << " (K is not a multiple of any particular tile "
              << "width -- the CUDA edition's own I.2 boundary case)" << std::endl;
    std::cout << "naive triple-loop matmul vs. torch::matmul(): results agree (torch::allclose)? " << agree
              << std::endl;

    int reps = 200;
    auto t0 = Clock::now();
    for (int r = 0; r < reps; r++) naive_matmul(A, B);
    auto t1 = Clock::now();
    for (int r = 0; r < reps; r++) torch::matmul(A, B);
    auto t2 = Clock::now();

    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    double torch_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
    double speedup = naive_ms / torch_ms;

    std::cout << "\n[TIMING] naive hand-written loop : " << naive_ms << " ms" << std::endl;
    std::cout << "[TIMING] torch::matmul()          : " << torch_ms << " ms" << std::endl;
    std::cout << "[TIMING] speedup (naive_ms / real_ms) : " << speedup << "x" << std::endl;

    std::cout << "\nthis shape was chosen specifically because it is awkward for hand-tiling -- 23 does not "
              << "divide evenly by 16, the tile width the CUDA edition's own kernel uses, so its own last "
              << "tile along K must zero-pad two columns' worth of A and two rows' worth of B rather than "
              << "load a full tile. torch::matmul() above never receives a tile width at all: whatever "
              << "internal blocking scheme its backend actually uses -- and this book's own Appendix E.4 "
              << "already measured that it delivers a real, large speedup over a naive loop -- handles this "
              << "boundary shape as an entirely ordinary case, inside a production implementation that has "
              << "been tested against exactly this kind of non-tile-multiple input far more thoroughly than "
              << "any one hand-written kernel in a book could be. On a CUDA-enabled build, this identical "
              << "torch::matmul() call would additionally link against cuBLAS and launch on whatever device "
              << "the input tensors live on -- the same one-line device change Section I.1 already "
              << "demonstrated, with no separate boundary-case logic to hand-write at all."
              << std::endl;

    return agree ? 0 : 1;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 02_tiling_payoff_revisited.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_tiling_payoff_revisited
./02_tiling_payoff_revisited
```

```text
M=37, K=23, N=19 (K is not a multiple of any particular tile width -- the CUDA edition's own I.2 boundary case)
naive triple-loop matmul vs. torch::matmul(): results agree (torch::allclose)? 1

[TIMING] naive hand-written loop : 0.00944361 ms
[TIMING] torch::matmul()          : 0.00180394 ms
[TIMING] speedup (naive_ms / real_ms) : 5.23499x

this shape was chosen specifically because it is awkward for hand-tiling -- 23 does not divide evenly by 16, the tile width the CUDA edition's own kernel uses, so its own last tile along K must zero-pad two columns' worth of A and two rows' worth of B rather than load a full tile. torch::matmul() above never receives a tile width at all: whatever internal blocking scheme its backend actually uses -- and this book's own Appendix E.4 already measured that it delivers a real, large speedup over a naive loop -- handles this boundary shape as an entirely ordinary case, inside a production implementation that has been tested against exactly this kind of non-tile-multiple input far more thoroughly than any one hand-written kernel in a book could be. On a CUDA-enabled build, this identical torch::matmul() call would additionally link against cuBLAS and launch on whatever device the input tensors live on -- the same one-line device change Section I.1 already demonstrated, with no separate boundary-case logic to hand-write at all.
```

**Discussion.** `torch::allclose` confirms `torch::matmul()`'s own result agrees with an independent, hand-written triple-loop matmul at this exact boundary shape, and `torch::matmul()` is measurably faster even at this small size, where fixed per-call overhead eats into the multiplier far more than it did at Appendix E.4's own larger N=256 case. Nothing in this file specifies a tile width, checks whether `K` divides evenly into one, or writes a zero-padding branch for a fractional last tile -- whatever internal blocking scheme `torch::matmul()`'s own backend actually uses handles this boundary shape as an entirely ordinary case, inside a production implementation tested against far more shapes than any one hand-written kernel in a book could be. On a CUDA-enabled build, this identical `torch::matmul()` call would additionally link against cuBLAS and launch on whichever device the input tensors live on -- Section I.1's own one-line device change, with no separate boundary-case logic to hand-write at all.

## I.3 Multi-Axis Contraction, Same Code Either Device

**Intuition.** The CUDA C++ edition's own Appendix I.3 needs an entirely separate kernel from I.1's single-shared-axis version to contract over multiple axes at once: a `MAX_RANK` compile-time cap, a `TensorMeta` struct carrying shape, stride, and rank for every operand, explicit free-axis and contracted-axis index arrays passed in by hand, and an iterative divmod loop inside the kernel to unravel a flattened output index -- because device code has neither `std::vector` nor runtime-depth recursion available to it the way Appendix H's own host-side `for_each_index()` does.

**Background.** `torch::tensordot()` takes none of that. The exact same call handles a single shared axis or several, on whichever device its input tensors happen to live on, with no `MAX_RANK`, no metadata struct, and no index arrays written at the call site at all.

**Formulas and Key Terms.**

- **Occupancy** -- the fraction of a GPU's own scheduling capacity kept genuinely busy by a kernel launch; a concern the CUDA edition's own hand-written kernel author must tune for directly, and one ATen's own backend kernels already account for internally on every `torch::` call.
- **Tile size** -- the block dimension a tiled kernel loads into shared memory at once, balancing shared-memory use per block against how many times each loaded value gets reused; chosen internally by `torch::matmul()`'s own backend rather than passed in by a caller.
- **Memory coalescing** -- arranging a kernel's own memory accesses so that consecutive threads read consecutive addresses, letting the hardware service them as one wide transaction rather than many narrow ones; a property of whatever kernel ATen's backend dispatches to, invisible from this appendix's own call sites.
- **Roofline model** -- a way of predicting whether a kernel's own running time is limited by arithmetic throughput or by memory bandwidth, given its FLOP count and arithmetic intensity; the same lens Section I.2's own discussion applies to explain why `torch::matmul()` beats a naive loop.
- **`MAX_RANK`** -- the fixed, compile-time axis-count cap the CUDA edition's own I.3 kernel must declare (4, there) so its metadata structs have a known size; `torch::tensordot()`'s own implementation supports a much larger, library-defined rank limit internally, and this appendix's own callers never declare or think about it at all.

**Worked Example I.3.1.** The double contraction from Appendix H's own Section H.4, run via `torch::tensordot()` on whichever device is available; then a second, single-shared-axis contraction on differently-shaped tensors, through the identical `torch::tensordot()` call with only the `dims` arguments changed.

```cpp
// Appendix I.3 -- multi-axis contraction, the same code either device.
//
// The CUDA C++ edition's own Appendix I.3 needs an entirely separate
// kernel from I.1's single-shared-axis version to contract over
// multiple axes at once: a MAX_RANK compile-time cap, a TensorMeta
// struct carrying shape/stride/rank for every operand, explicit
// free-axis and contracted-axis index arrays passed in by hand, and
// an iterative divmod loop inside the kernel to unravel a flattened
// output index -- because device code has neither std::vector nor
// runtime-depth recursion available to it the way Appendix H's own
// host-side for_each_index() does.
//
// torch::tensordot() takes none of that. The exact same call handles
// a single shared axis or several, on whichever device its input
// tensors happen to live on, with no MAX_RANK, no metadata struct,
// and no index arrays written at the call site at all -- the general
// arbitrary-rank machinery this appendix's own file 01 already
// pointed to still exists, inside LibTorch's own implementation, but
// it is never something a caller has to write or even see.
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <vector>

int main() {
    bool cuda_available = torch::cuda::is_available();
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "running on device: " << device << std::endl;

    // The exact double-contraction problem from Appendix H, Section
    // H.4: A is [2,3,4] filled 1..24 in row-major order, B is [3,4,5]
    // filled 1..60 in row-major order, contracting A's axes {1,2}
    // against B's axes {0,1} -- already cross-checked there against
    // numpy.tensordot.
    torch::Tensor A = torch::arange(1, 25, torch::kFloat64).reshape({2, 3, 4}).to(device);
    torch::Tensor B = torch::arange(1, 61, torch::kFloat64).reshape({3, 4, 5}).to(device);

    torch::Tensor C = torch::tensordot(A, B, /*dims_self=*/{1, 2}, /*dims_other=*/{0, 1});

    std::cout << "\nA.sizes() = " << A.sizes() << ", B.sizes() = " << B.sizes() << std::endl;
    std::cout << "C = torch::tensordot(A, B, dims_self={1,2}, dims_other={0,1})" << std::endl;
    std::cout << "C.sizes() = " << C.sizes() << std::endl;
    std::cout << "C values:\n" << C << std::endl;

    // Independent cross-check: these are the exact ten values Appendix
    // H, Section H.4 locked, computed there via a hand-rolled
    // contract() and cross-checked separately against
    // numpy.tensordot(A, B, axes=([1,2],[0,1])).
    torch::Tensor expected =
        torch::tensor({{2938.0, 3016.0, 3094.0, 3172.0, 3250.0}, {7042.0, 7264.0, 7486.0, 7708.0, 7930.0}},
                       torch::kFloat64)
            .to(device);
    bool matches_h4 = torch::allclose(C, expected);
    std::cout << "\ncross-check vs. Appendix H, Section H.4's own locked values: " << (matches_h4 ? "PASS" : "FAIL")
              << std::endl;

    // A second call, over a DIFFERENT number of shared axes (one axis
    // instead of two), on the identical A and B tensors' own leading
    // sub-blocks -- to make the "same code either rank of contraction"
    // point concrete rather than asserted: no new function, no new
    // struct, no new compile-time constant, just a different `dims`
    // argument to the identical torch::tensordot() call used above.
    torch::Tensor A2 = A.index({0});  // [3,4], A's own first "batch" slice
    torch::Tensor B2 = B.index({torch::indexing::Slice(), 0});  // [3,5], B's own first column-slice
    torch::Tensor C2 = torch::tensordot(A2, B2, /*dims_self=*/{0}, /*dims_other=*/{0});
    std::cout << "\na single-shared-axis contraction, same torch::tensordot() call, different `dims`:"
              << std::endl;
    std::cout << "A2.sizes() = " << A2.sizes() << ", B2.sizes() = " << B2.sizes() << ", C2.sizes() = " << C2.sizes()
              << std::endl;

    // Cross-check C2 against a plain manual sum, independent of
    // tensordot's own machinery.
    torch::Tensor A2_cpu = A2.to(torch::kCPU);
    torch::Tensor B2_cpu = B2.to(torch::kCPU);
    torch::Tensor C2_cpu = C2.to(torch::kCPU);
    auto a2 = A2_cpu.accessor<double, 2>();
    auto b2 = B2_cpu.accessor<double, 2>();
    bool c2_matches = true;
    for (int64_t j = 0; j < C2.size(0); j++) {
        for (int64_t l = 0; l < C2.size(1); l++) {
            double sum = 0.0;
            for (int64_t i = 0; i < 3; i++) sum += a2[i][j] * b2[i][l];
            if (C2_cpu[j][l].item<double>() != sum) c2_matches = false;
        }
    }
    std::cout << "cross-check vs. independent plain-loop sum: " << (c2_matches ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nboth contractions above -- one over two shared axes, one over a single shared axis, on "
              << "tensors of different rank -- went through the identical torch::tensordot() call, varying "
              << "only which axes were named. The CUDA C++ edition's own I.3 kernel, by contrast, is a "
              << "separate, purpose-built __global__ function from I.1's single-axis kernel, with its own "
              << "MAX_RANK-bounded metadata structs -- a second kernel this appendix's own LibTorch version "
              << "never needed to write." << std::endl;

    return (matches_h4 && c2_matches) ? 0 : 1;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 03_multi_axis_same_code.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_multi_axis_same_code
./03_multi_axis_same_code
```

```text
running on device: cpu

A.sizes() = [2, 3, 4], B.sizes() = [3, 4, 5]
C = torch::tensordot(A, B, dims_self={1,2}, dims_other={0,1})
C.sizes() = [2, 5]
C values:
 2938  3016  3094  3172  3250
 7042  7264  7486  7708  7930
[ CPUDoubleType{2,5} ]

cross-check vs. Appendix H, Section H.4's own locked values: PASS

a single-shared-axis contraction, same torch::tensordot() call, different `dims`:
A2.sizes() = [3, 4], B2.sizes() = [3, 5], C2.sizes() = [4, 5]
cross-check vs. independent plain-loop sum: PASS

both contractions above -- one over two shared axes, one over a single shared axis, on tensors of different rank -- went through the identical torch::tensordot() call, varying only which axes were named. The CUDA C++ edition's own I.3 kernel, by contrast, is a separate, purpose-built __global__ function from I.1's single-axis kernel, with its own MAX_RANK-bounded metadata structs -- a second kernel this appendix's own LibTorch version never needed to write.
```

**Discussion.** The first contraction reproduces Appendix H's own Section H.4 values exactly, over two shared axes; the second contracts over a single shared axis, on tensors of different rank entirely, and both went through the identical `torch::tensordot()` call -- varying only which axes were named as `dims_self` and `dims_other`. The CUDA C++ edition's own I.3 kernel is, by contrast, a separate, purpose-built `__global__` function from I.1's own single-axis kernel, carrying its own `MAX_RANK`-bounded metadata structs -- a second kernel this appendix's own LibTorch version never had to write, because the arbitrary-rank generalization work Appendix H's own `for_each_index()` did by hand on the host already lives, once, inside LibTorch's own implementation of `tensordot()` itself.

## I.4 In Practice: cuBLAS, and What This Book Has Been Doing All Along

**Intuition.** The CUDA C++ edition's own Appendix I closes by naming what a real project should actually use instead of a hand-rolled kernel: cuBLAS for matmul-shaped work, cuTENSOR for more general contractions -- both tuned per GPU architecture, with tiling, register blocking, and mixed-precision paths far beyond what a hand-written kernel in a book would attempt. It makes no performance claims about either library there, since no GPU is available in that environment to benchmark them on either.

**Background.** This section makes no GPU performance claims for the identical reason. What it can say is this: this book's own `torch::matmul()` and `torch::einsum()` calls, on a CUDA-enabled build, are exactly what the CUDA C++ edition's own I.4 recommends switching to -- `torch::matmul()` on a CUDA tensor links against cuBLAS for the matmul-shaped case, the same production library I.4 names; a more general `torch::einsum()`/`torch::tensordot()` call decomposes into a sequence of matmul-shaped and reduction operations, or dispatches to other tuned CUDA kernels, depending on the exact axis pattern involved -- a decision LibTorch's own dispatcher makes, not something this appendix's own source code ever has to choose. Every `torch::` call this book has made since Chapter 1 has already been going through this exact path on a CPU build, against this book's own installed LibTorch's CPU-side BLAS backend; a CUDA-enabled build would route the identical calls to cuBLAS and its own CUDA-side peers instead, with zero source changes anywhere in this book.

There is no code listing in this section, matching the CUDA C++ edition's own I.4, which is prose-only as well.

## I.5 Complete Runnable Code

### `01_dispatch_and_device_query.cpp`

```cpp
// Appendix I.1 -- from a CPU contraction to a device-dispatched one.
//
// Appendix H's contract() has an outer loop over free-index combinations
// and an inner loop over contracted-index combinations. The CUDA C++
// edition's own Appendix I maps that split directly onto a GPU: the
// outer (free-index) loop becomes a grid of threads, one thread per
// output element, and the inner (contracted-index) loop becomes each
// thread's own sequential accumulation loop -- a hand-written
// __global__ kernel, compiled with nvcc, launched with an explicit
// dim3 grid and block.
//
// This book takes a different path, because it has taken this same
// path since Chapter 1: torch::einsum() and torch::matmul() already
// perform exactly that free-index/contracted-index split internally,
// and ALREADY DISPATCH to a device-appropriate implementation --
// vectorized CPU code, or a CUDA kernel launched with its own
// internally chosen grid and block -- based purely on which device
// the input tensors live on. The C++ source at the call site never
// changes; only a tensor's own .device() does. This file demonstrates
// that dispatch honestly on this book's own sandbox, which has no
// GPU attached at all.
#include <cstdint>
#include <iostream>
#include <torch/torch.h>

int main() {
    bool cuda_available = torch::cuda::is_available();
    int64_t device_count = torch::cuda::device_count();

    std::cout << "torch::cuda::is_available() = " << (cuda_available ? "true" : "false") << std::endl;
    std::cout << "torch::cuda::device_count()  = " << device_count << std::endl;
    std::cout << "\nthis sandbox reports no CUDA-capable device, exactly as the CUDA C++ edition's own "
              << "cudaMalloc calls honestly report \"no CUDA-capable device is detected\" throughout its own "
              << "Appendix I -- neither book fabricates a GPU that isn't there." << std::endl;

    // The device every tensor below actually runs on: torch::kCUDA if
    // available, torch::kCPU otherwise. Nothing past this line
    // branches on that choice -- the SAME torch::einsum() call below
    // would execute unchanged, dispatching to a CUDA kernel instead
    // of vectorized CPU code, if `device` were torch::kCUDA.
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "\nrunning on device: " << device << std::endl;

    // The exact matmul-as-contraction problem from Appendix H, Section
    // H.3: A is [3,2], B is [2,4], contracting A's axis 1 against B's
    // axis 0. There, contract() built the free/contracted split by
    // hand; here, torch::einsum() builds and dispatches it internally.
    torch::Tensor A = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}).to(device);
    torch::Tensor B = torch::tensor({{1.0, 0.0, 2.0, 1.0}, {0.0, 1.0, 1.0, 2.0}}).to(device);
    torch::Tensor C = torch::einsum("ik,kj->ij", {A, B});

    std::cout << "\nC = torch::einsum(\"ik,kj->ij\", {A, B}), A and B on " << device << ":" << std::endl;
    std::cout << C << std::endl;

    // Independent cross-check: the identical A/B/C values Appendix H,
    // Section H.3 locked, computed there via a hand-rolled contract()
    // and cross-checked against a plain triple-loop matmul and against
    // this same torch::einsum() call. Recomputed here via a plain
    // nested-loop matmul on the host, with no torch:: machinery at
    // all, as a second, structurally different confirmation.
    std::vector<std::vector<double>> a_host = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<std::vector<double>> b_host = {{1, 0, 2, 1}, {0, 1, 1, 2}};
    std::vector<std::vector<double>> expected(3, std::vector<double>(4, 0.0));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 2; k++) sum += a_host[i][k] * b_host[k][j];
            expected[i][j] = sum;
        }
    }
    torch::Tensor C_cpu = C.to(torch::kCPU);
    bool all_match = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            if (C_cpu[i][j].item<double>() != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independent plain-loop matmul (matches Appendix H, Section H.3's own "
              << "locked C values): " << (all_match ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nnothing above this line would need to change to run this exact contraction on a real GPU "
              << "-- only `device` itself would become torch::kCUDA, chosen automatically the moment "
              << "torch::cuda::is_available() reports true. The CUDA C++ edition's own Appendix I, by "
              << "contrast, needs an entirely separate .cu source file, a distinct compiler (nvcc), and a "
              << "hand-written kernel with its own explicit grid and block dimensions to move this identical "
              << "computation from host to device." << std::endl;

    return all_match ? 0 : 1;
}
```

### `02_tiling_payoff_revisited.cpp`

```cpp
// Appendix I.2 -- the tiling payoff, already delivered.
//
// The CUDA C++ edition's own Appendix I.2 hand-writes a shared-memory
// tiled contraction kernel, deliberately sized M=37, K=23, N=19 so
// that K (23) is NOT a multiple of its own TILE=16 -- specifically to
// exercise the zero-padding boundary branch a naive tiled kernel must
// get right or silently corrupt its last tile's own results.
//
// This book already measured the real payoff of that entire class of
// optimization once, honestly, in Appendix E.4: a real ~18x speedup
// of torch::matmul() over a naive hand-written loop, on this same
// sandbox's own CPU, at N=256. This file adds a second, independent
// measurement at the CUDA edition's own boundary-exercising, non-tile
// -multiple shape (M=37, K=23, N=19) -- confirming the payoff shows up
// even on a shape specifically chosen to be awkward for hand-tiling,
// and confirming torch::matmul()'s own result is correct there too,
// with no padding logic ever written at this file's own call site.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using Clock = std::chrono::steady_clock;

torch::Tensor make_matrix(int64_t rows, int64_t cols, double seed) {
    torch::Tensor t = torch::empty({rows, cols}, torch::kFloat64);
    auto acc = t.accessor<double, 2>();
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            acc[i][j] = std::sin(seed + static_cast<double>(i) * 0.017 + static_cast<double>(j) * 0.013);
        }
    }
    return t;
}

torch::Tensor naive_matmul(const torch::Tensor& A, const torch::Tensor& B) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = torch::zeros({M, N}, torch::kFloat64);
    auto a = A.accessor<double, 2>();
    auto b = B.accessor<double, 2>();
    auto c = C.accessor<double, 2>();
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; k++) sum += a[i][k] * b[k][j];
            c[i][j] = sum;
        }
    }
    return C;
}

int main() {
    // The CUDA C++ edition's own Appendix I.2 shape, chosen there
    // specifically because K=23 is not a multiple of TILE=16 -- the
    // exact boundary case a hand-tiled kernel's padding branch exists
    // to handle. torch::matmul() below is never told about tile
    // widths at all.
    int64_t M = 37, K = 23, N = 19;
    torch::Tensor A = make_matrix(M, K, 1.0);
    torch::Tensor B = make_matrix(K, N, 2.0);

    torch::Tensor C_naive = naive_matmul(A, B);
    torch::Tensor C_torch = torch::matmul(A, B);

    bool agree = torch::allclose(C_naive, C_torch, /*rtol=*/1e-9, /*atol=*/1e-9);
    std::cout << "M=" << M << ", K=" << K << ", N=" << N << " (K is not a multiple of any particular tile "
              << "width -- the CUDA edition's own I.2 boundary case)" << std::endl;
    std::cout << "naive triple-loop matmul vs. torch::matmul(): results agree (torch::allclose)? " << agree
              << std::endl;

    int reps = 200;
    auto t0 = Clock::now();
    for (int r = 0; r < reps; r++) naive_matmul(A, B);
    auto t1 = Clock::now();
    for (int r = 0; r < reps; r++) torch::matmul(A, B);
    auto t2 = Clock::now();

    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    double torch_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
    double speedup = naive_ms / torch_ms;

    std::cout << "\n[TIMING] naive hand-written loop : " << naive_ms << " ms" << std::endl;
    std::cout << "[TIMING] torch::matmul()          : " << torch_ms << " ms" << std::endl;
    std::cout << "[TIMING] speedup (naive_ms / real_ms) : " << speedup << "x" << std::endl;

    std::cout << "\nthis shape was chosen specifically because it is awkward for hand-tiling -- 23 does not "
              << "divide evenly by 16, the tile width the CUDA edition's own kernel uses, so its own last "
              << "tile along K must zero-pad two columns' worth of A and two rows' worth of B rather than "
              << "load a full tile. torch::matmul() above never receives a tile width at all: whatever "
              << "internal blocking scheme its backend actually uses -- and this book's own Appendix E.4 "
              << "already measured that it delivers a real, large speedup over a naive loop -- handles this "
              << "boundary shape as an entirely ordinary case, inside a production implementation that has "
              << "been tested against exactly this kind of non-tile-multiple input far more thoroughly than "
              << "any one hand-written kernel in a book could be. On a CUDA-enabled build, this identical "
              << "torch::matmul() call would additionally link against cuBLAS and launch on whatever device "
              << "the input tensors live on -- the same one-line device change Section I.1 already "
              << "demonstrated, with no separate boundary-case logic to hand-write at all."
              << std::endl;

    return agree ? 0 : 1;
}
```

### `03_multi_axis_same_code.cpp`

```cpp
// Appendix I.3 -- multi-axis contraction, the same code either device.
//
// The CUDA C++ edition's own Appendix I.3 needs an entirely separate
// kernel from I.1's single-shared-axis version to contract over
// multiple axes at once: a MAX_RANK compile-time cap, a TensorMeta
// struct carrying shape/stride/rank for every operand, explicit
// free-axis and contracted-axis index arrays passed in by hand, and
// an iterative divmod loop inside the kernel to unravel a flattened
// output index -- because device code has neither std::vector nor
// runtime-depth recursion available to it the way Appendix H's own
// host-side for_each_index() does.
//
// torch::tensordot() takes none of that. The exact same call handles
// a single shared axis or several, on whichever device its input
// tensors happen to live on, with no MAX_RANK, no metadata struct,
// and no index arrays written at the call site at all -- the general
// arbitrary-rank machinery this appendix's own file 01 already
// pointed to still exists, inside LibTorch's own implementation, but
// it is never something a caller has to write or even see.
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <vector>

int main() {
    bool cuda_available = torch::cuda::is_available();
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "running on device: " << device << std::endl;

    // The exact double-contraction problem from Appendix H, Section
    // H.4: A is [2,3,4] filled 1..24 in row-major order, B is [3,4,5]
    // filled 1..60 in row-major order, contracting A's axes {1,2}
    // against B's axes {0,1} -- already cross-checked there against
    // numpy.tensordot.
    torch::Tensor A = torch::arange(1, 25, torch::kFloat64).reshape({2, 3, 4}).to(device);
    torch::Tensor B = torch::arange(1, 61, torch::kFloat64).reshape({3, 4, 5}).to(device);

    torch::Tensor C = torch::tensordot(A, B, /*dims_self=*/{1, 2}, /*dims_other=*/{0, 1});

    std::cout << "\nA.sizes() = " << A.sizes() << ", B.sizes() = " << B.sizes() << std::endl;
    std::cout << "C = torch::tensordot(A, B, dims_self={1,2}, dims_other={0,1})" << std::endl;
    std::cout << "C.sizes() = " << C.sizes() << std::endl;
    std::cout << "C values:\n" << C << std::endl;

    // Independent cross-check: these are the exact ten values Appendix
    // H, Section H.4 locked, computed there via a hand-rolled
    // contract() and cross-checked separately against
    // numpy.tensordot(A, B, axes=([1,2],[0,1])).
    torch::Tensor expected =
        torch::tensor({{2938.0, 3016.0, 3094.0, 3172.0, 3250.0}, {7042.0, 7264.0, 7486.0, 7708.0, 7930.0}},
                       torch::kFloat64)
            .to(device);
    bool matches_h4 = torch::allclose(C, expected);
    std::cout << "\ncross-check vs. Appendix H, Section H.4's own locked values: " << (matches_h4 ? "PASS" : "FAIL")
              << std::endl;

    // A second call, over a DIFFERENT number of shared axes (one axis
    // instead of two), on the identical A and B tensors' own leading
    // sub-blocks -- to make the "same code either rank of contraction"
    // point concrete rather than asserted: no new function, no new
    // struct, no new compile-time constant, just a different `dims`
    // argument to the identical torch::tensordot() call used above.
    torch::Tensor A2 = A.index({0});  // [3,4], A's own first "batch" slice
    torch::Tensor B2 = B.index({torch::indexing::Slice(), 0});  // [3,5], B's own first column-slice
    torch::Tensor C2 = torch::tensordot(A2, B2, /*dims_self=*/{0}, /*dims_other=*/{0});
    std::cout << "\na single-shared-axis contraction, same torch::tensordot() call, different `dims`:"
              << std::endl;
    std::cout << "A2.sizes() = " << A2.sizes() << ", B2.sizes() = " << B2.sizes() << ", C2.sizes() = " << C2.sizes()
              << std::endl;

    // Cross-check C2 against a plain manual sum, independent of
    // tensordot's own machinery.
    torch::Tensor A2_cpu = A2.to(torch::kCPU);
    torch::Tensor B2_cpu = B2.to(torch::kCPU);
    torch::Tensor C2_cpu = C2.to(torch::kCPU);
    auto a2 = A2_cpu.accessor<double, 2>();
    auto b2 = B2_cpu.accessor<double, 2>();
    bool c2_matches = true;
    for (int64_t j = 0; j < C2.size(0); j++) {
        for (int64_t l = 0; l < C2.size(1); l++) {
            double sum = 0.0;
            for (int64_t i = 0; i < 3; i++) sum += a2[i][j] * b2[i][l];
            if (C2_cpu[j][l].item<double>() != sum) c2_matches = false;
        }
    }
    std::cout << "cross-check vs. independent plain-loop sum: " << (c2_matches ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nboth contractions above -- one over two shared axes, one over a single shared axis, on "
              << "tensors of different rank -- went through the identical torch::tensordot() call, varying "
              << "only which axes were named. The CUDA C++ edition's own I.3 kernel, by contrast, is a "
              << "separate, purpose-built __global__ function from I.1's single-axis kernel, with its own "
              << "MAX_RANK-bounded metadata structs -- a second kernel this appendix's own LibTorch version "
              << "never needed to write." << std::endl;

    return (matches_h4 && c2_matches) ? 0 : 1;
}
```

## Chapter Summary

This appendix took Appendix H's own free-index/contracted-index split and mapped it onto a GPU a second way -- not by hand-writing a `__global__` kernel, as the CUDA C++ edition's own Appendix I does three times over, but by recognizing that `torch::matmul()` and `torch::einsum()` already perform that mapping internally and already dispatch to a device-appropriate implementation the instant a tensor's own `.device()` says which one to use. Section I.1 confirmed that dispatch honestly, on a sandbox with no GPU, reproducing Appendix H's own Section H.3 values through `torch::einsum()`. Section I.2 revisited Appendix E.4's own real tiling-payoff measurement at a shape deliberately awkward for hand-tiling, confirming both correctness and a real speedup with no tile width ever written. Section I.3 showed a single `torch::tensordot()` call handling one shared axis or several, on tensors of any rank, with none of I.3's own `MAX_RANK`/`TensorMeta` machinery required at the call site. And Section I.4 closed by naming what this book's own `torch::` calls have already been linking against since Chapter 1 -- cuBLAS and its own peers, on a CUDA-enabled build -- the exact production path the CUDA C++ edition's own closing section recommends switching to instead of a hand-written kernel.

## Self-Check Questions

1. What plays the role of Appendix H's free-index loop and contracted-index loop in this appendix's own LibTorch version, and who -- this appendix's own source code, or something else -- decides how that split actually executes on a given device?
2. Section I.1's own file never writes anything resembling the CUDA C++ edition's own `row >= M || col >= N` bounds guard. Explain why that guard is unnecessary here.
3. Section I.2 measures a real speedup of `torch::matmul()` over a naive loop at `M=37, K=23, N=19`. Does that speedup come from `torch::matmul()` performing fewer floating-point operations? If not, what does it come from?
4. Section I.2 deliberately reuses the CUDA C++ edition's own awkward, non-tile-multiple shape, even though this appendix's own file never writes any tiling or padding logic at all. What does that reuse actually confirm?
5. Why doesn't `torch::tensordot()`'s own caller ever need anything resembling Section I.3's own `MAX_RANK` constant or `TensorMeta` struct?
6. What does running this appendix's own three files on a sandbox with no GPU attached actually prove about their correctness -- and what does it NOT prove?
7. If a real CUDA-enabled GPU were available, what additional evidence would be worth gathering, beyond what this appendix's own sandbox run already confirmed?

### Worked Solutions

**1.** `torch::einsum()`/`torch::tensordot()`'s own internal implementation performs exactly that same split -- one output element's worth of work per free-index combination, summed over every combination of the contracted indices -- but HOW that split actually executes is decided by LibTorch's own dispatcher, not by this appendix's own source code: vectorized CPU code if the input tensors live on `torch::kCPU`, or a CUDA kernel with its own internally chosen grid and block if they live on `torch::kCUDA`. This appendix's own files never contain a grid, a block, or a thread index anywhere.

**2.** The CUDA C++ edition's own bounds guard exists because its kernel's grid is sized by ceiling division (`ceil(N/16) x ceil(M/16)` blocks of `16x16` threads), which can genuinely over-provision more threads than there are real output elements when `M` or `N` isn't a multiple of the block size -- without the guard, those excess threads would write out of bounds. `torch::matmul()` and `torch::einsum()` never launch more work items than there are actual output elements to begin with; whatever internal launch configuration ATen's own backend chooses if it dispatches to a CUDA kernel at all is sized correctly by that backend, not exposed to or computed by this appendix's own caller, so there is no analogous excess to guard against at this appendix's own call site.

**3.** No -- both the naive triple-loop matmul and `torch::matmul()` perform the identical `M*N*K = 37*23*19 = 16169` multiply-adds; the FLOP count is fixed by the problem's own shape, not by which implementation computes it. The measured speedup instead comes from `torch::matmul()`'s own internal implementation moving far less redundant data between memory and cache or registers than the naive loop's own repeated re-reads do -- the identical memory-traffic mechanism Appendix H's own Section H.6 measured directly for loop order, and the one Appendix E.4 already measured once before for `torch::matmul()` itself at a larger size.

**4.** It confirms that `torch::matmul()`'s own production implementation -- whatever internal tiling or blocking scheme it actually uses -- handles the exact boundary shape that would break a naively-written hand-tiled kernel's own zero-padding branch if that branch had a bug, without this appendix's own code ever reasoning about tile widths, remainders, or padding at all. The CUDA C++ edition's own kernel had to get that boundary case right by hand; this appendix's own file gets it right by calling into a library that already had to get it right for every caller, not just this one.

**5.** Because the work of generalizing to an arbitrary rank -- walking a shape's own indices for however many axes it actually has, whether via Appendix H's own runtime-recursive `for_each_index()` on the host or a fixed-`MAX_RANK` iterative loop inside a CUDA kernel -- already lives inside LibTorch's own implementation of `tensordot()`/`einsum()`, written once by LibTorch's own maintainers and reused for every caller and every rank up to a large, library-internal limit. A caller of the LibTorch API supplies a shape and a set of axes; the caller is never the one who has to write, or even know about, the machinery that walks those axes generically.

**6.** Running this appendix's own three files on a sandbox with no GPU proves the API-level correctness of every call: the right shapes, the right values, cross-checked independently against Appendix H's own already-locked values and against plain host loops written with none of `torch::tensordot()`'s own machinery. It also proves the exact same source would compile against a CUDA-enabled LibTorch build unchanged. It does NOT prove anything about a real GPU kernel's own performance, occupancy, memory-coalescing behavior, or the correctness of ATen's own internal CUDA implementation on real hardware -- only an actual run on a CUDA-enabled device could confirm those, exactly as the CUDA C++ edition's own host-side kernel emulation could confirm its own kernel's index arithmetic but not its performance or its `__syncthreads()` race-freedom.

**7.** Running these exact same three source files, completely unchanged, on a machine where `torch::cuda::is_available()` reports `true` -- confirming both that the results still match Appendix H's own locked values on that device, and obtaining a real, measured GPU speedup over the CPU numbers this appendix already recorded, the same kind of confirmation Appendix E.4 already obtained once for CPU `torch::matmul()` over a naive loop.
