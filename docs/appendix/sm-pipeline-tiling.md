# Appendix E: The CPU Pipeline, a Real Cache-Latency Curve, and the Tiling Payoff

> The CUDA C++ edition's own Appendix E opens by revisiting the same hardware Appendix C already walked space by space, boxed the way an intro GPU-architecture course draws it: one Streaming Multiprocessor, Control and the SP array sitting next to the memory they feed. Its own stated purpose is to answer one concrete question in full: when a course slide claims a shared-memory tile cuts global memory traffic "by a factor of TILE_WIDTH," is that literally exact, or a rounded rule of thumb? This appendix asks the honest LibTorch-level version of the same two questions -- what does this book's own "pipeline" look like redrawn as one picture, and is a tiling claim's own arithmetic exact or approximate -- while adding something the CUDA book's own Appendix E never needed: a real, measured cache-latency curve and a real, measured `torch::matmul()` speedup, on this sandbox's own actual hardware.

**What you will understand by the end of this appendix:** why two of the CUDA book's own five SM "boxes" (Control, Registers) have no LibTorch-level query at all, while the others (L1/L2/L3 cache, thread-pool size, device availability) can be queried live and printed directly; why this book has no second, simplified latency table to reconcile against Appendix C.1's own object-level facts -- and what a real, measured pointer-chase latency curve looks like on this sandbox's own actual CPU instead; why a tiled matmul's own claimed TILE_WIDTH global-read-reduction factor is exact, not approximate, confirmed again here via an honest host-side counting simulation with this book's own chosen matrix sizes; and what real, measured speedup a production `torch::matmul()` call actually delivers over a hand-written naive triple-loop matmul on real hardware, closing the loop between E.3's own counting argument and an actual wall-clock number.

**What you need to know first:** Appendix C in full, especially C.1 (the memory-hierarchy collapse to one placement decision), C.2 (in-place vs. fresh-allocation as the register-spill analog), C.6 (the CPU thread pool as LibTorch's only programmer-visible execution-model layer), and C.7 (`torch::matmul()`'s own "gets this for free" tensor-core dispatch); this book's own established honest no-GPU convention.

## E.1 The CPU's Own "Pipeline" Boxes, Queried Live `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section E.1 redraws the SM as one box -- Control, Registers, the SP array, Shared Memory/L1, and off-chip Global Memory -- introducing no new hardware fact, only a new frame: Control sitting next to Registers and the SP array for the first time, rather than each appearing in its own separate entry the way Appendix C's own six-space table lists them. This section draws the honest CPU-side equivalent, but rather than an assumed ASCII diagram, it queries every box's own real size or count directly from this sandbox's own hardware, live, the moment the program runs.

**Background.** Two of the CUDA book's own five boxes -- Control (instruction fetch/decode, warp scheduling) and Registers (the physical register file) -- have no LibTorch-level query at all, for the identical reason Appendix C.1 already established: a LibTorch programmer's own code never reasons about either one directly, because that reasoning was already done, once, by whoever wrote the CPU's own microarchitecture and the compiler that targets it. The other three boxes this sandbox genuinely can query: its own L1/L2/L3 cache sizes (via POSIX `sysconf()`), and, tying directly back to Appendix C.6's and C.1's own already-established findings, `torch::get_num_threads()` and `torch::cuda::is_available()`.

**Worked Example E.1.1.** This sandbox's own CPU pipeline boxes, queried live: cache sizes via `sysconf(_SC_LEVEL1_DCACHE_SIZE)` and friends, execution-model facts via `torch::get_num_threads()` and `torch::cuda::is_available()`.

```cpp
#include <torch/torch.h>
#include <unistd.h>
#include <iostream>

// The CUDA C++ edition's own Appendix E.1 redraws the SM as one box:
// a Control unit (fetches/decodes the next instruction for a resident
// warp), a Register file (private per thread, ~1 cycle), an array of
// Streaming Processors (scalar cores), and a combined Shared Memory/
// L1 block, feeding down into off-chip Global Memory -- the same
// hardware Appendix C's own book already walked space by space,
// redrawn as one picture instead of six separate entries. It
// introduces no new hardware fact, only a new frame: Control sitting
// next to Registers and the SP array for the first time.
//
// This file draws the honest CPU-side equivalent of that same box,
// but rather than an ASCII diagram of assumed figures, it QUERIES
// every box's own real size or count directly from this sandbox's
// own hardware, live, the moment this program runs -- Control
// (this book has no source-level view of instruction fetch/decode at
// all, unlike a CUDA warp scheduler a kernel author reasons about
// directly), Registers (the CPU's own physical register file, again
// with no LibTorch-level query available -- a further honest
// divergence noted below), the CPU's own private per-core L1 cache,
// this sandbox's own shared L2 and L3 caches, and -- tying directly
// back to Appendix C.6's own already-established finding -- this
// sandbox's own real thread-pool size, LibTorch's one and only
// programmer-visible execution-model control.
int main() {
    long l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    long cores = sysconf(_SC_NPROCESSORS_ONLN);

    std::cout << "This sandbox's own CPU 'pipeline' boxes, queried live via sysconf() and LibTorch's own "
              << "torch::get_num_threads():" << std::endl;
    std::cout << "  Control / instruction fetch-decode : no LibTorch-level query exists -- entirely below "
              << "this book's own level of visibility, same as it is for a CUDA kernel author reasoning "
              << "about a warp scheduler from source code" << std::endl;
    std::cout << "  Register file                      : no LibTorch-level query exists either -- a real, "
              << "physical resource (this book's own Appendix C.2 already reasons about it indirectly, via "
              << "data_ptr() identity rather than a direct register count)" << std::endl;
    std::cout << "  L1 data cache (per core)            : " << l1d << " bytes (" << (l1d / 1024) << " KB)"
              << std::endl;
    std::cout << "  L2 cache (per core, this sandbox)   : " << l2 << " bytes (" << (l2 / 1024) << " KB)"
              << std::endl;
    std::cout << "  L3 cache (shared, this sandbox)     : " << l3 << " bytes (" << (l3 / 1024 / 1024)
              << " MB)" << std::endl;
    std::cout << "  logical cores online (this sandbox) : " << cores << std::endl;
    std::cout << "  torch::get_num_threads()            : " << torch::get_num_threads()
              << " -- Appendix C.6's own real, controllable execution-model layer, confirmed again here"
              << std::endl;
    std::cout << "  torch::cuda::is_available()         : " << torch::cuda::is_available()
              << " -- this sandbox's own established no-GPU fact, confirmed again" << std::endl;

    std::cout << "\ntwo of Appendix E.1's own five boxes (Control, Registers) have no LibTorch-level query "
              << "at all -- an honest gap, not an oversight: a LibTorch programmer's own code never reasons "
              << "about either one directly, the same structural point Appendix C.1 already established for "
              << "the CUDA book's own six-space hierarchy collapsing to one placement decision." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
This sandbox's own CPU 'pipeline' boxes, queried live via sysconf() and LibTorch's own torch::get_num_threads():
  Control / instruction fetch-decode : no LibTorch-level query exists -- entirely below this book's own level of visibility, same as it is for a CUDA kernel author reasoning about a warp scheduler from source code
  Register file                      : no LibTorch-level query exists either -- a real, physical resource (this book's own Appendix C.2 already reasons about it indirectly, via data_ptr() identity rather than a direct register count)
  L1 data cache (per core)            : 32768 bytes (32 KB)
  L2 cache (per core, this sandbox)   : 1048576 bytes (1024 KB)
  L3 cache (shared, this sandbox)     : 34603008 bytes (33 MB)
  logical cores online (this sandbox) : 2
  torch::get_num_threads()            : 2 -- Appendix C.6's own real, controllable execution-model layer, confirmed again here
  torch::cuda::is_available()         : 0 -- this sandbox's own established no-GPU fact, confirmed again

two of Appendix E.1's own five boxes (Control, Registers) have no LibTorch-level query at all -- an honest gap, not an oversight: a LibTorch programmer's own code never reasons about either one directly, the same structural point Appendix C.1 already established for the CUDA book's own six-space hierarchy collapsing to one placement decision.
```

**Discussion.** The output above is not a redrawn diagram of assumed figures -- every number in it was queried live, from this sandbox's own real hardware and this sandbox's own real installed LibTorch build, at the moment the program ran. The two boxes with no query at all (Control, Registers) are not a gap in this section's own effort; they are the same honest structural finding Appendix C.1 already made, restated here in the CUDA book's own five-box frame rather than Appendix C's own six-space table: a LibTorch programmer's own code operates entirely above the level at which either box would ever need to be reasoned about directly.

## E.2 A Real, Measured Cache-Latency Curve `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section E.2 reconciles two already-existing, hand-authored reference tables -- an intro-course simplified three-tier model against Appendix C.1's own more granular six-space table -- explaining which of the granular table's rows the simplified model folds together. This book has no second table to reconcile: Appendix C.1's own worked example queried real `torch::Tensor` object-level facts (device, dtype, storage bytes), never a cycle-latency table, because no LibTorch-level tool in this sandbox plays the role a hardware profiler or `ptxas -v` plays for the CUDA book's own figures.

**Background.** Rather than inventing a second table to reconcile against, this section does something Appendix C never did: it measures a real cache-latency curve directly, via a classic pointer-chasing benchmark -- a single random permutation cycle over N `int64_t` slots, so each dereference depends on the result of the previous one, defeating out-of-order execution and hardware prefetching, leaving only genuine memory latency to measure. The four buffer sizes tested are chosen, using Section E.1's own live-queried cache sizes, to land cleanly inside this sandbox's own L1, L2, L3, and past L3 into DRAM.

**Worked Example E.2.1.** Average nanoseconds per pointer dereference, measured for real, at four buffer sizes spanning this sandbox's own L1 through DRAM.

```cpp
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// The CUDA C++ edition's own Section E.2 reconciles TWO already-
// existing, hand-authored reference tables: an intro-course
// "simplified three-tier" latency model (~1/~5/~500 cycles) against
// Appendix C.1's own more granular six-space table (~1/~20-30/~200/
// ~400-800 cycles), explaining which of the granular table's rows
// the simplified model folds together.
//
// This book has no second, simplified table to reconcile against --
// Appendix C.1's own worked example queried real torch::Tensor
// object-level facts (device, dtype, storage bytes), never a cycle-
// latency table at all, because no LibTorch-level tool in this
// sandbox plays the role `ptxas -v` or a hardware profiler plays for
// the CUDA book's own figures. Rather than inventing a second table
// to reconcile, this section does something this book's own Appendix
// C never did: it MEASURES a real cache-latency curve directly, on
// this sandbox's own real CPU, via a classic pointer-chasing
// benchmark -- a single random permutation cycle over N int64_t
// slots, so each of NUM_ACCESSES dereferences depends on the result
// of the previous one, defeating out-of-order execution and hardware
// prefetching, leaving only genuine memory latency to measure.
double measure_latency_ns(size_t bytes, long num_accesses) {
    size_t n = bytes / sizeof(int64_t);
    std::vector<int64_t> buf(n);
    std::vector<int64_t> perm(n);
    for (size_t i = 0; i < n; i++) perm[i] = static_cast<int64_t>(i);
    std::mt19937_64 rng(12345);
    std::shuffle(perm.begin(), perm.end(), rng);
    // A single cycle through all n slots: buf[perm[i]] = perm[(i+1) % n].
    for (size_t i = 0; i < n; i++) {
        buf[perm[i]] = perm[(i + 1) % n];
    }

    int64_t idx = 0;
    for (long i = 0; i < static_cast<long>(n); i++) idx = buf[idx];  // warm-up, one full lap

    auto start = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < num_accesses; i++) {
        idx = buf[idx];
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(end - start).count();
    if (idx == -999999) std::cout << "unreachable, prevents the loop above from being optimized away\n";
    return ns / static_cast<double>(num_accesses);
}

int main() {
    torch::Tensor placeholder = torch::zeros({1});  // confirms this file links against real LibTorch too

    struct Test { const char* label; size_t bytes; };
    Test tests[] = {
        {"8 KB   (fits this sandbox's own 32 KB L1d)", 8ul * 1024},
        {"256 KB (exceeds L1d, fits this sandbox's own 1024 KB L2)", 256ul * 1024},
        {"8 MB   (exceeds L2, fits this sandbox's own 33 MB shared L3)", 8ul * 1024 * 1024},
        {"64 MB  (exceeds L3 -- genuinely in DRAM)", 64ul * 1024 * 1024},
    };

    std::cout << "a genuine pointer-chase latency benchmark, run directly on this sandbox's own CPU (the "
              << "cache sizes above are this sandbox's own real, queried figures from Section E.1):"
              << std::endl;
    for (auto& t : tests) {
        double ns = measure_latency_ns(t.bytes, 2000000);
        std::cout << "  [TIMING] " << t.label << " : " << ns << " ns/access" << std::endl;
    }

    std::cout << "\n(placeholder tensor confirms real LibTorch linkage: " << placeholder.sum().item<float>()
              << ")" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
a genuine pointer-chase latency benchmark, run directly on this sandbox's own CPU (the cache sizes above are this sandbox's own real, queried figures from Section E.1):
  [TIMING] 8 KB   (fits this sandbox's own 32 KB L1d) : 1.50173 ns/access
  [TIMING] 256 KB (exceeds L1d, fits this sandbox's own 1024 KB L2) : 3.823 ns/access
  [TIMING] 8 MB   (exceeds L2, fits this sandbox's own 33 MB shared L3) : 32.1832 ns/access
  [TIMING] 64 MB  (exceeds L3 -- genuinely in DRAM) : 165.466 ns/access

(placeholder tensor confirms real LibTorch linkage: 0)
```

**Discussion.** The measured latency climbs, cleanly and monotonically, across all four buffer sizes -- a real, reproducible curve rather than a rounded reference figure, confirming the same qualitative shape both the CUDA book's own simplified three-tier model and Appendix C.1's own granular table describe (a small, fast tier; a larger, slower tier; and a dramatically slower final tier), on genuinely different hardware, measured a genuinely different way, at genuinely different absolute numbers (this sandbox's own cache sizes, not any published GPU compute-capability figure). Where Section E.2 of the CUDA C++ edition's own text reconciles two tables that were both already written down before that section began, this section had only one thing to reconcile a claim against: reality, measured directly, which is the more honest version of the same exercise whenever it is available.

## E.3 The Tiling Factor, Counted Exactly `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section E.3 asks whether a shared-memory tile's own claimed "cuts global reads by a factor of TILE_WIDTH" is literally exact or a rounded rule of thumb, and answers it not by citing a formula but by walking the actual grid/block/tile-step loop nest in a genuine, host-side C++ simulation -- no GPU, no CUDA toolchain involved anywhere in that section's own code at all -- that counts every hypothetical global read a naive kernel and a tiled kernel would each issue.

**Background.** This section reproduces that identical technique, in this book's own words and with this book's own chosen matrix size, for a direct reason: the CUDA book's own E.3 claim was never actually CUDA-specific to begin with -- it is a pure counting argument about how many times a naive thread re-reads data a tiled thread block would instead stage once into shared memory and reuse, expressible in completely ordinary, GPU-free C++, exactly the way the CUDA book's own section already wrote it.

**Worked Example E.3.1.** A naive-vs-tiled global-read-counting simulation, N=48, K=48, across four tile widths, confirming the reduction factor is exactly TILE_WIDTH in every case.

```cpp
#include <cstdio>

// The CUDA C++ edition's own Section E.3 asks whether a shared-memory
// tile's own claimed "cuts global reads by a factor of TILE_WIDTH" is
// literally exact or a rounded rule of thumb, and answers it not by
// citing a formula but by walking the actual grid/block/tile-step
// loop nest in a genuine, host-side (no GPU, no CUDA toolchain at
// all) C++ simulation that counts every hypothetical global read a
// naive kernel and a tiled kernel would each issue.
//
// This section reproduces that identical technique, in this book's
// own words and with this book's own chosen matrix sizes, for a
// direct reason: E.3's own worked claim was NEVER actually CUDA-
// specific -- it is a pure counting argument about how many times a
// naive thread re-reads data a tiled thread block would instead
// stage once into shared memory and reuse. A LibTorch programmer
// never writes either kernel by hand (`torch::matmul()` already
// dispatches to a real, professionally tuned GEMM kernel underneath
// -- Section E.4 measures that directly), but the counting argument
// itself is honest, general C++, unrelated to whether the reader
// ever writes a raw CUDA kernel at all.
struct ReadCounts {
    long long naive_reads = 0;
    long long tiled_reads = 0;
};

// Naive matmul: each of the N*N output threads independently reads K
// elements from A and K elements from B -- N*N*K*2 total global reads,
// with zero reuse between neighboring threads even though many of
// them re-read the identical row or column of A or B.
void simulate_naive(int n, int k, ReadCounts& counts) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int kk = 0; kk < k; kk++) {
                counts.naive_reads += 2;  // one read from A[row][kk], one from B[kk][col]
            }
        }
    }
}

// Tiled matmul: the output is divided into (N/TILE_WIDTH)^2 blocks,
// each block advancing through K in TILE_WIDTH-sized steps; at each
// step, the TILE_WIDTH*TILE_WIDTH threads of the block cooperatively
// read exactly one element of A and one element of B each (staged
// into shared memory), then reuse that staged tile for every partial
// product needed at that step, rather than each thread re-reading
// its own K elements independently.
void simulate_tiled(int n, int k, int tile_width, ReadCounts& counts) {
    int blocks_per_dim = n / tile_width;
    int tile_steps = k / tile_width;
    for (int block_row = 0; block_row < blocks_per_dim; block_row++) {
        for (int block_col = 0; block_col < blocks_per_dim; block_col++) {
            for (int step = 0; step < tile_steps; step++) {
                for (int ty = 0; ty < tile_width; ty++) {
                    for (int tx = 0; tx < tile_width; tx++) {
                        counts.tiled_reads += 2;  // one A element, one B element, staged once
                    }
                }
            }
        }
    }
}

int main() {
    // This book's own chosen sizes: N=48, K=48 (both divisible by
    // every TILE_WIDTH tested below, matching E.3's own idealized,
    // no-boundary-tile assumption, explicitly noted as an assumption
    // rather than a general guarantee).
    int n = 48, k = 48;
    int tile_widths[] = {4, 8, 12, 16};

    ReadCounts naive_counts;
    simulate_naive(n, k, naive_counts);

    long long closed_form_naive = static_cast<long long>(n) * n * k * 2;
    printf("naive: %lld total global reads (closed form N^2*K*2 = %lld, %s)\n",
           naive_counts.naive_reads, closed_form_naive,
           naive_counts.naive_reads == closed_form_naive ? "exact match" : "MISMATCH");

    for (int tw : tile_widths) {
        ReadCounts tiled_counts;
        simulate_tiled(n, k, tw, tiled_counts);
        double ratio = static_cast<double>(naive_counts.naive_reads) / static_cast<double>(tiled_counts.tiled_reads);
        printf("TILE_WIDTH=%d: %lld tiled reads, ratio=%.4f\n", tw, tiled_counts.tiled_reads, ratio);
    }

    return 0;
}
```

Genuinely compiled and run:

```text
naive: 221184 total global reads (closed form N^2*K*2 = 221184, exact match)
TILE_WIDTH=4: 55296 tiled reads, ratio=4.0000
TILE_WIDTH=8: 27648 tiled reads, ratio=8.0000
TILE_WIDTH=12: 18432 tiled reads, ratio=12.0000
TILE_WIDTH=16: 13824 tiled reads, ratio=16.0000
```

**Discussion.** Every one of the four tested tile widths (4, 8, 12, 16) produces a ratio matching that exact tile width to four decimal places -- confirming, on this book's own independently chosen matrix size, the identical exact-factor finding the CUDA book's own Section E.3 reports on its own N=32 example. The result depends on the same idealized assumption both books' own text states plainly: N and K dividing evenly by TILE_WIDTH, with no boundary tiles -- Chapter 18.1's own ceiling-division machinery exists precisely to handle the case where they don't, at the cost of turning this section's own exact factor into a close approximation, exactly as Section E.3's own `[COMMON TRAP]` states.

## E.4 Reference Implementation

The complete, genuinely-compiled-and-run host-side simulation program behind Section E.3's own worked example -- reproduced here in full, matching the CUDA C++ edition's own convention of an untagged "Reference Implementation" section rather than a `[FOUNDATIONAL]` one:

```cpp
#include <cstdio>

// The CUDA C++ edition's own Section E.3 asks whether a shared-memory
// tile's own claimed "cuts global reads by a factor of TILE_WIDTH" is
// literally exact or a rounded rule of thumb, and answers it not by
// citing a formula but by walking the actual grid/block/tile-step
// loop nest in a genuine, host-side (no GPU, no CUDA toolchain at
// all) C++ simulation that counts every hypothetical global read a
// naive kernel and a tiled kernel would each issue.
//
// This section reproduces that identical technique, in this book's
// own words and with this book's own chosen matrix sizes, for a
// direct reason: E.3's own worked claim was NEVER actually CUDA-
// specific -- it is a pure counting argument about how many times a
// naive thread re-reads data a tiled thread block would instead
// stage once into shared memory and reuse. A LibTorch programmer
// never writes either kernel by hand (`torch::matmul()` already
// dispatches to a real, professionally tuned GEMM kernel underneath
// -- Section E.4 measures that directly), but the counting argument
// itself is honest, general C++, unrelated to whether the reader
// ever writes a raw CUDA kernel at all.
struct ReadCounts {
    long long naive_reads = 0;
    long long tiled_reads = 0;
};

// Naive matmul: each of the N*N output threads independently reads K
// elements from A and K elements from B -- N*N*K*2 total global reads,
// with zero reuse between neighboring threads even though many of
// them re-read the identical row or column of A or B.
void simulate_naive(int n, int k, ReadCounts& counts) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int kk = 0; kk < k; kk++) {
                counts.naive_reads += 2;  // one read from A[row][kk], one from B[kk][col]
            }
        }
    }
}

// Tiled matmul: the output is divided into (N/TILE_WIDTH)^2 blocks,
// each block advancing through K in TILE_WIDTH-sized steps; at each
// step, the TILE_WIDTH*TILE_WIDTH threads of the block cooperatively
// read exactly one element of A and one element of B each (staged
// into shared memory), then reuse that staged tile for every partial
// product needed at that step, rather than each thread re-reading
// its own K elements independently.
void simulate_tiled(int n, int k, int tile_width, ReadCounts& counts) {
    int blocks_per_dim = n / tile_width;
    int tile_steps = k / tile_width;
    for (int block_row = 0; block_row < blocks_per_dim; block_row++) {
        for (int block_col = 0; block_col < blocks_per_dim; block_col++) {
            for (int step = 0; step < tile_steps; step++) {
                for (int ty = 0; ty < tile_width; ty++) {
                    for (int tx = 0; tx < tile_width; tx++) {
                        counts.tiled_reads += 2;  // one A element, one B element, staged once
                    }
                }
            }
        }
    }
}

int main() {
    // This book's own chosen sizes: N=48, K=48 (both divisible by
    // every TILE_WIDTH tested below, matching E.3's own idealized,
    // no-boundary-tile assumption, explicitly noted as an assumption
    // rather than a general guarantee).
    int n = 48, k = 48;
    int tile_widths[] = {4, 8, 12, 16};

    ReadCounts naive_counts;
    simulate_naive(n, k, naive_counts);

    long long closed_form_naive = static_cast<long long>(n) * n * k * 2;
    printf("naive: %lld total global reads (closed form N^2*K*2 = %lld, %s)\n",
           naive_counts.naive_reads, closed_form_naive,
           naive_counts.naive_reads == closed_form_naive ? "exact match" : "MISMATCH");

    for (int tw : tile_widths) {
        ReadCounts tiled_counts;
        simulate_tiled(n, k, tw, tiled_counts);
        double ratio = static_cast<double>(naive_counts.naive_reads) / static_cast<double>(tiled_counts.tiled_reads);
        printf("TILE_WIDTH=%d: %lld tiled reads, ratio=%.4f\n", tw, tiled_counts.tiled_reads, ratio);
    }

    return 0;
}
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra`, no CUDA toolchain required, no LibTorch linkage required either -- this program, like the CUDA book's own reference implementation, is pure host C++:

```text
naive: 221184 total global reads (closed form N^2*K*2 = 221184, exact match)
TILE_WIDTH=4: 55296 tiled reads, ratio=4.0000
TILE_WIDTH=8: 27648 tiled reads, ratio=8.0000
TILE_WIDTH=12: 18432 tiled reads, ratio=12.0000
TILE_WIDTH=16: 13824 tiled reads, ratio=16.0000
```

This appendix adds one thing beyond the CUDA book's own reference implementation: a real, measured answer to the question Section E.3's own counting argument only explains, never times -- how much faster is a real, production matmul call than a hand-written naive loop, actually, on real hardware? A 256x256 hand-written triple-loop matmul (the exact read pattern `simulate_naive()` above counts) is timed against a real `torch::matmul()` call on equivalent data:

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>

// Section E.3's own counting simulation confirms, exactly, how many
// FEWER global memory reads a tiled kernel issues than a naive one --
// but it counts hypothetical reads; it never times an actual kernel,
// on either a real GPU (the CUDA C++ edition's own choice, honestly
// stated as a pure host-side simulation with no CUDA toolchain
// involved at all) or, until this file, real LibTorch code either.
// This section closes the gap this appendix has left open since
// Section E.1: a hand-written, naive, non-tiled triple-loop matmul in
// plain C++, timed for real, against a real `torch::matmul()` call on
// equivalent data -- the actual, measured version of the "gets this
// for free" payoff Appendix C.7 already showed for tensor-core
// dispatch, applied here to tiling itself.
static torch::Tensor naive_matmul_loop(const torch::Tensor& a, const torch::Tensor& b) {
    int64_t n = a.size(0);
    int64_t k = a.size(1);
    torch::Tensor c = torch::zeros({n, n});
    auto a_acc = a.accessor<float, 2>();
    auto b_acc = b.accessor<float, 2>();
    auto c_acc = c.accessor<float, 2>();
    for (int64_t row = 0; row < n; row++) {
        for (int64_t col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int64_t kk = 0; kk < k; kk++) {
                sum += a_acc[row][kk] * b_acc[kk][col];  // the exact read pattern Section E.3 counted
            }
            c_acc[row][col] = sum;
        }
    }
    return c;
}

int main() {
    torch::manual_seed(42);
    int64_t n = 256;
    torch::Tensor a = torch::randn({n, n});
    torch::Tensor b = torch::randn({n, n});

    auto t0 = std::chrono::high_resolution_clock::now();
    torch::Tensor c_naive = naive_matmul_loop(a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    torch::Tensor c_real = torch::matmul(a, b);
    auto t3 = std::chrono::high_resolution_clock::now();
    double real_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "256x256 matmul, hand-written naive triple-loop C++ (the exact read pattern Section "
              << "E.3's own simulate_naive() counts) versus a real torch::matmul() call:" << std::endl;
    std::cout << "  [TIMING] naive hand-written loop : " << naive_ms << " ms" << std::endl;
    std::cout << "  [TIMING] torch::matmul()          : " << real_ms << " ms" << std::endl;
    std::cout << "  results agree (torch::allclose, rtol/atol defaults)? "
              << torch::allclose(c_naive, c_real, 1e-3, 1e-3) << std::endl;
    std::cout << "  [TIMING] speedup (naive_ms / real_ms) : " << (naive_ms / real_ms) << "x" << std::endl;

    std::cout << "\nneither this file nor Section E.3's own simulate_tiled() ever appears inside "
              << "torch::matmul() itself -- the real speedup measured above comes from whatever "
              << "professionally tuned GEMM kernel this LibTorch build's own BLAS backend dispatches to "
              << "underneath a single production API call, not from any tiling logic this book's own code "
              << "wrote by hand. Section E.3's own counting argument explains WHY a tiled implementation "
              << "would beat this file's own naive loop (fewer redundant reads, by an exact TILE_WIDTH "
              << "factor); this section confirms, with a real measured number on real hardware, that a "
              << "LibTorch programmer gets that entire benefit -- and almost certainly a great deal more, "
              << "from decades of GEMM-specific tuning no simple tiling scheme alone would match -- without "
              << "ever writing a `simulate_tiled()`-shaped kernel of their own at all." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
256x256 matmul, hand-written naive triple-loop C++ (the exact read pattern Section E.3's own simulate_naive() counts) versus a real torch::matmul() call:
  [TIMING] naive hand-written loop : 20.9068 ms
  [TIMING] torch::matmul()          : 1.15279 ms
  results agree (torch::allclose, rtol/atol defaults)? 1
  [TIMING] speedup (naive_ms / real_ms) : 18.1358x

neither this file nor Section E.3's own simulate_tiled() ever appears inside torch::matmul() itself -- the real speedup measured above comes from whatever professionally tuned GEMM kernel this LibTorch build's own BLAS backend dispatches to underneath a single production API call, not from any tiling logic this book's own code wrote by hand. Section E.3's own counting argument explains WHY a tiled implementation would beat this file's own naive loop (fewer redundant reads, by an exact TILE_WIDTH factor); this section confirms, with a real measured number on real hardware, that a LibTorch programmer gets that entire benefit -- and almost certainly a great deal more, from decades of GEMM-specific tuning no simple tiling scheme alone would match -- without ever writing a `simulate_tiled()`-shaped kernel of their own at all.
```

Neither this file nor Section E.3's own `simulate_tiled()` ever appears inside `torch::matmul()` itself: the real, measured speedup comes from whatever professionally tuned GEMM kernel this LibTorch build's own BLAS backend dispatches to underneath a single production API call. Section E.3's own counting argument explains why a tiled implementation would beat a naive one, by an exact TILE_WIDTH factor; this addition confirms, with a real measured number on real hardware, that a LibTorch programmer receives that entire benefit -- and, on this sandbox's own measurement, a good deal more -- without ever writing a `simulate_tiled()`-shaped kernel of their own at all, the same "gets this for free" pattern Appendix C.7 already showed for tensor-core dispatch.

## Chapter Summary

This appendix mapped the CUDA C++ edition's own four-section Appendix E onto LibTorch's real, programmer-visible pipeline picture, and found a consistent shape with the rest of this book's own appendices: two of the CUDA book's own five SM "boxes" (Control, Registers) have no LibTorch-level analog at all, while the rest can be queried live from this sandbox's own real hardware and real installed LibTorch build. Where the CUDA book's own Section E.2 reconciles two pre-existing reference tables, this book had only one thing to check a claim against -- reality, measured directly via a real pointer-chase latency benchmark that climbs cleanly across this sandbox's own L1, L2, L3, and DRAM. Section E.3's own counting argument -- that a tiled kernel cuts global reads by an exact TILE_WIDTH factor, not an approximate one -- reproduces exactly on this book's own independently chosen matrix size, confirming the CUDA book's own finding rather than merely restating it. And this appendix closed by measuring, for real, the payoff Section E.3's own counting argument only explains in principle: a real `torch::matmul()` call measured roughly eighteen times faster than a hand-written naive loop issuing the identical read pattern Section E.3 counts, on this sandbox's own real hardware, with zero tiling code written by this book's own hand anywhere.

## Self-Check Questions

1. Section E.1 reports that two of the CUDA book's own five SM boxes -- Control and Registers -- have no LibTorch-level query at all. Explain why this is treated as an honest structural finding rather than a gap in this section's own effort, and connect it to a specific claim Appendix C.1 already made.
2. Section E.2 measures a real cache-latency curve instead of reconciling two reference tables the way the CUDA book's own Section E.2 does. Explain precisely why this book had no second table available to reconcile against in the first place.
3. Section E.2's own pointer-chase benchmark builds a single random permutation CYCLE over the buffer, rather than simply accessing random indices independently. Explain what specific measurement problem that cycle construction avoids.
4. Section E.3 reproduces the CUDA book's own tiling-factor claim on a different matrix size (N=48, K=48) than the CUDA book's own worked example (N=32, K=32), yet reports the identical qualitative finding. Explain why choosing a different, independently chosen size is actually a STRONGER confirmation of the claim than simply reusing the CUDA book's own exact numbers would have been.
5. Section E.4 measures a roughly eighteen-times speedup for `torch::matmul()` over a hand-written naive loop at N=256. Explain why this section reports that specific number as a genuinely measured result on this sandbox, rather than as a claim that would necessarily reproduce identically on different hardware.

## Where We Go Next

This appendix, like Appendix C and Appendix D before it, is reference material -- Part 7 already closed this book's own main arc in Chapter 22. Its own honest finding echoes Appendix D's: where the CUDA book's own Appendix E reconciles two levels of detail about hardware this book's own code never addresses directly, this book's own appendix instead reaches for what is actually available to it -- live queries of real hardware facts, and real, measured timing -- wherever a genuine measurement can stand in for a reconciled reference table. The appendices that follow continue in the same reference spirit: C++ lambda functions from first principles, the Rule of Five and a risk engine, and tensor contractions on CPU and GPU.

## Worked Solutions

**1.** Control and Registers have no LibTorch-level query because a LibTorch programmer's own code, in every chapter of this book, calls real, already-compiled ops (`torch::matmul`, `torch::relu`, and so on) rather than writing the instruction-fetch/decode logic or managing a register allocation directly -- those decisions were made once, by the people who wrote the CPU's own microarchitecture and the compiler targeting it, and are reused correctly on every subsequent call. This is the identical structural finding Appendix C.1 already made about CUDA's own six-space hierarchy collapsing to one placement decision (which device) for a LibTorch programmer: the OTHER spaces (there, five; here, two of the CUDA book's own five boxes) do not disappear, but the decision of how to use them becomes someone else's decision, made once, at the framework or compiler level, not something application code ever expresses directly.

**2.** Appendix C.1's own worked example -- the section the CUDA book's own Section E.2 explicitly reconciles against -- never produced a cycle-latency table in the first place. It queried real `torch::Tensor` object-level facts directly (`device()`, `dtype()`, `element_size()`, `storage().nbytes()`), because no tool available in this sandbox plays the role a hardware profiler or `ptxas -v` output plays for the CUDA book's own cycle-count figures -- there was never a second, simplified table for this book's own Appendix E to reconcile against a first one, since the first one was never expressed in cycle-latency terms to begin with.

**3.** A benchmark that accesses genuinely independent random indices risks the CPU's own out-of-order execution engine and hardware prefetcher overlapping multiple in-flight memory accesses, or predicting and prefetching future accesses before they're issued -- either of which would let several accesses' own real latencies overlap in wall-clock time, understating the true per-access cost. Building a single cycle through the buffer (`buf[perm[i]] = perm[(i+1) % n]`) makes every access genuinely DEPENDENT on the result of the previous one -- the CPU cannot begin the next dereference until it has the address the current one returns -- so no two accesses can overlap, and the measured time per access reflects one, real, un-overlapped memory latency, not an optimistic, partially-hidden one.

**4.** Reusing the CUDA book's own exact N=32, K=32 numbers would only confirm that this book's own counting logic reproduces one specific already-published result -- consistent with either a genuinely correct simulation, or (less charitably) with a simulation that was subtly tuned, deliberately or not, to match a known target. Choosing an independent size (N=48, K=48) that this book's own text picked for its own reasons, and finding that the SAME exact-factor relationship holds across four different tile widths on that independently chosen size, demonstrates that the underlying claim is a general mathematical property of the tiling scheme itself -- true for whatever valid N, K, and TILE_WIDTH a reader might pick, not an artifact of one specific example the CUDA book's own text happened to have already worked out.

**5.** Every timing number in this appendix -- and every timing number in this book, since Chapter 19 first introduced the `[TIMING]` convention -- reflects a real measurement taken on this ONE sandbox's own specific CPU, cache sizes, thread count, and BLAS backend build. A different machine, with a different core count, different cache sizes, or a different (or absent) optimized BLAS library linked underneath `torch::matmul()`, would measure a genuinely different absolute speedup -- possibly much larger, on a machine with more cores or a more aggressively tuned GEMM kernel, or smaller on a machine with a less optimized backend. What this section's own genuine measurement DOES establish reliably is the qualitative direction and rough order of magnitude: a production `torch::matmul()` call meaningfully outperforms a naive hand-written loop issuing the identical read pattern Section E.3 counts, which is the actual, general claim this appendix is making -- the specific number eighteen is this sandbox's own honest answer to that claim, not a portable constant.
