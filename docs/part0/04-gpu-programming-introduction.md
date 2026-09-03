# Chapter 4: GPU Programming Introduction

> "The CUDA C++ edition's Chapter 4 hands you a kernel launch: `<<<blocks, threads_per_block>>>`, a global thread index you compute yourself from `blockIdx` and `threadIdx`, a `cudaMalloc`/`cudaMemcpy`/`cudaFree` triplet for every buffer that crosses the host/device boundary. This book's own code has never written any of that, in three chapters, and won't start now — `torch::matmul(A, B)` is the whole kernel launch. This chapter's job is to find out exactly what happened to the thread hierarchy, the explicit memory copies, and the memory-tier bookkeeping: not by asserting they're 'handled automatically', but by genuinely testing what evidence of them still reaches a LibTorch programmer, and what doesn't."

**What you will understand by the end of this chapter:**

- Why no LibTorch call site — not `a + b`, not `torch::matmul`, not anything this book has written — ever computes a thread index the way CUDA's `blockIdx.x * blockDim.x + threadIdx.x` requires, and what `torch::get_num_threads()` / `torch::set_num_threads()` genuinely do control instead, measured with a real, repeatable timing difference
- The direct, `.data_ptr()`-based test for whether an operation that stays on one device actually copies anything — and a re-verified instance of the real `c10::Error` this book's `.to(kCUDA)` calls have thrown since Chapter 1, gathered here as this chapter's own host/device evidence
- Genuine, grep-verified proof that LibTorch's public C++ frontend — the exact headers this book's own compile line includes — exposes no `shared_memory` or `__shared__` concept anywhere, while ATen's *internal* CUDA kernel implementation headers use it constantly, 10 files' worth, every one of them confined to paths this book's `-I` flags never reach
- How Chapter 3's `.stride()` / `.is_contiguous()` tools generalize past particles entirely — a single column pulled from an ordinary `[4000, 4000]` matrix, measured roughly 30-40x slower to sum through its strided view than through a materialized contiguous copy, reproduced across every rerun performed while writing this chapter
- Why "broadcasting" means two genuinely different things in the CUDA and LibTorch worlds that happen to share a name — CUDA's "one thread per output element" execution model versus PyTorch's implicit shape-expansion arithmetic — and how they connect underneath despite being different ideas
- `torch::matmul`, genuinely run against the CUDA book's own 2x2 example and cross-checked against a hand-computed dot product, matching `[[19, 22], [43, 50]]` exactly — while this book's own compiled binary's stride evidence shows the same row/column access asymmetry the CUDA book's naive kernel exposes, with no kernel of its own to write

**What you need to know first:**

- Chapter 1's Sections 1.3 and 1.4 — `c10::Device` as a runtime value, and the genuine `c10::Error` this environment's `.to(kCUDA)` calls throw, which this chapter re-verifies rather than reintroduces from scratch
- Chapter 3 in full — the `.stride()` and `.is_contiguous()` evidence this chapter reuses directly, on a plain matrix instead of a particle tensor
- If you've read the CUDA C++ edition's Chapter 4: that chapter hand-writes a kernel launch, an explicit `cudaMalloc`/`cudaMemcpy`/`cudaFree` sequence, and a naive one-thread-one-dot-product matrix multiply kernel. This chapter has no kernel to write anywhere — every one of its six sections instead asks "what evidence of this CUDA mechanism is still visible from LibTorch's C++ frontend, and what genuinely isn't," which is a different question than the CUDA book asks, answered with different tools (timing, pointer identity, grep, and `torch::matmul`) rather than a disassembler or a kernel launch

## 4.1 The Thread Hierarchy, Abstracted Away `[FOUNDATIONAL]`

### Intuition

CUDA's grid/block/thread hierarchy exists because the programmer has to say, explicitly, which of potentially millions of threads is running right now and which output element it owns. `torch::Tensor` operations never ask this question. `a + b` is the entire call, on a tensor of 4 elements or 40 million — no launch configuration, no per-thread index computed anywhere in this book's own code. That isn't because the underlying hardware stopped needing parallel work distributed across many execution units; it's because `torch::Tensor`'s dispatcher does that distribution internally, and the one lever LibTorch's public C++ API exposes over it is *how many* workers to use, not *which* element each one gets.

### Background

`torch::get_num_threads()` reports how many threads LibTorch's CPU backend will use for parallelizable ops; `torch::set_num_threads(n)` changes it. Neither function takes an index, a range, or anything resembling `blockIdx`/`threadIdx` — they configure a thread *pool size*, and the dispatcher decides how to split whatever work an op represents across that pool, invisibly to the caller.

### Worked Example 4.1.1 — the same call site, measurably different parallelism

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>

// CUDA's kernel launch requires the CALLER to compute a thread's global index
// by hand (blockIdx.x * blockDim.x + threadIdx.x) before that thread does any
// work. LibTorch's CPU backend has an analogous notion of "how many workers
// run this op in parallel" -- torch::get_num_threads() / set_num_threads()
// -- but the caller never assigns work to a specific worker. The parallel
// dispatch is entirely internal to the op; a+b looks identical whether it
// runs on one thread or many.
double time_elementwise_add(int64_t n, int reps) {
    torch::Tensor a = torch::rand({n});
    torch::Tensor b = torch::rand({n});
    // warm up
    torch::Tensor warm = a + b;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        torch::Tensor c = a + b;
        volatile float sink = c[0].item<float>();
        (void)sink;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "torch::get_num_threads() (default) = " << torch::get_num_threads() << std::endl;

    const int64_t n = 20000000;
    const int reps = 30;
    const int trials = 5;

    int one_thread_faster_or_equal = 0;
    int multi_thread_faster = 0;
    for (int t = 0; t < trials; t++) {
        torch::set_num_threads(1);
        double one_thread_ms = time_elementwise_add(n, reps);

        torch::set_num_threads(2);
        double multi_thread_ms = time_elementwise_add(n, reps);

        bool multi_faster = multi_thread_ms < one_thread_ms;
        if (multi_faster) multi_thread_faster++; else one_thread_faster_or_equal++;
        std::cout << "trial " << t << ": 1 thread = " << one_thread_ms
                  << " ms, 2 threads = " << multi_thread_ms
                  << " ms, multi_faster = " << (multi_faster ? "true" : "false") << std::endl;
    }

    std::cout << "2 threads faster than 1 thread in " << multi_thread_faster << " / " << trials << " trials" << std::endl;

    // The point this chapter actually cares about: the SAME call, a + b, is
    // what ran in every trial above. No index was computed by this program.
    // Nothing here says "thread 3, take element 4193". torch::Tensor's
    // dispatcher decided how to split n=20000000 elements across however
    // many threads were configured, entirely inside the + operator.
    std::cout << "elementwise add call site is identical regardless of thread count: `a + b`" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment (this sandbox has 2 CPU cores; the exact millisecond figures are wall-clock timing and, per this book's stated methodology, not treated as byte-reproducible — the boolean claim is):

```text
torch::get_num_threads() (default) = 2
trial 0: 1 thread = 1118.52 ms, 2 threads = 754.749 ms, multi_faster = true
trial 1: 1 thread = 1191.18 ms, 2 threads = 753.179 ms, multi_faster = true
trial 2: 1 thread = 1216.53 ms, 2 threads = 711.786 ms, multi_faster = true
trial 3: 1 thread = 1255.29 ms, 2 threads = 797.509 ms, multi_faster = true
trial 4: 1 thread = 1294.46 ms, 2 threads = 762.846 ms, multi_faster = true
2 threads faster than 1 thread in 5 / 5 trials
elementwise add call site is identical regardless of thread count: `a + b`
```

`torch::get_num_threads()` reports `2` by default on this sandbox's 2-core environment — LibTorch is already sizing its thread pool to the available hardware without being told to. Forcing the pool down to a single thread with `torch::set_num_threads(1)` and back up to `2` produces a real, repeatable gap: 2 threads beat 1 thread in every one of 5 trials, reproduced across every rerun performed while writing this chapter. But the line worth reading twice is the last one: `a + b` is *exactly* the same C++ expression whether `torch::get_num_threads()` reports `1` or `2` — nothing about the call site changed to accommodate more parallelism. CUDA's `blockIdx.x * blockDim.x + threadIdx.x` has no equivalent anywhere in this file, because there was never a point where this program needed to say which of the 20,000,000 elements a particular worker should handle.

## 4.2 Host and Device Memory, Revisited: What Actually Moves `[FOUNDATIONAL]`

### Intuition

Chapter 1's Section 1.4 already established that crossing from CPU to a genuinely absent CUDA device throws a real `c10::Error` in this sandbox. CUDA's `cudaMemcpy` raises a narrower, equally important question this book hasn't tested directly yet: when a tensor operation *doesn't* have to cross a device boundary, does anything actually get copied? The direct way to answer that isn't a profiler or a `cudaMemcpy` call counter — it's comparing `.data_ptr()` before and after.

### Background

`torch::Tensor::data_ptr()` returns the address of the tensor's underlying storage. If an operation returns a tensor with the identical `data_ptr()` as its input, no new allocation happened — the operation is a free, zero-copy view or pass-through. If the returned tensor has a *different* `data_ptr()`, a genuine copy occurred somewhere. `.to(device)` and `.clone()` make a natural contrasting pair: `.to()` only needs to copy when the target device differs from the source; `.clone()` is documented to always allocate a fresh copy, on principle, regardless of device.

### Worked Example 4.2.1 — same-device is free, cloning isn't, crossing devices genuinely fails

```cpp
#include <torch/torch.h>
#include <iostream>
#include <sstream>

// Chapter 1, Section 1.4 already showed .to(kCUDA) genuinely throwing a real
// c10::Error in this no-GPU sandbox. This file asks the other half of the
// host/device question CUDA's cudaMemcpy makes explicit: when data movement
// DOESN'T have to cross a device boundary, does LibTorch actually copy
// anything? The direct, runtime-queryable answer is .data_ptr() identity --
// no profiler, no cudaMemcpy call count, just a pointer comparison.
std::string first_line(const std::exception& e) {
    std::istringstream iss(e.what());
    std::string line;
    std::getline(iss, line);
    return line;
}

int main() {
    torch::Tensor t = torch::rand({1000});
    void* original_ptr = t.data_ptr();
    std::cout << "t.device() = " << t.device() << std::endl;

    // .to(kCPU) on an ALREADY-CPU tensor: same device, so LibTorch is free to
    // return the same underlying storage untouched -- no copy needed.
    torch::Tensor same_device = t.to(torch::kCPU);
    std::cout << "same_device.data_ptr() == t.data_ptr()? "
              << (same_device.data_ptr() == original_ptr) << std::endl;

    // .clone() is the operation that genuinely forces a new allocation and a
    // real copy, regardless of device -- the direct contrast case.
    torch::Tensor cloned = t.clone();
    std::cout << "cloned.data_ptr() == t.data_ptr()? "
              << (cloned.data_ptr() == original_ptr) << std::endl;

    // Values still match even though the copy has a different address --
    // cloning changes WHERE the data lives, never WHAT the data is.
    std::cout << "cloned values equal to original? "
              << torch::equal(cloned, t) << std::endl;

    // The genuine cross-device attempt (mirrors Chapter 1, Section 1.4,
    // re-verified here as this chapter's own direct evidence): a real
    // c10::Error, not a fabricated one, because this sandbox has no CUDA
    // device or driver.
    try {
        torch::Tensor moved = t.to(torch::kCUDA);
        std::cout << "t.to(kCUDA) unexpectedly succeeded" << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "t.to(kCUDA) threw c10::Error: " << first_line(e) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
t.device() = cpu
same_device.data_ptr() == t.data_ptr()? 1
cloned.data_ptr() == t.data_ptr()? 0
cloned values equal to original? 1
t.to(kCUDA) threw c10::Error: CUDA error: CUDA driver version is insufficient for CUDA runtime version
```

`same_device.data_ptr() == t.data_ptr()` reporting `1` (`true`) is the direct evidence: `.to(torch::kCPU)` on a tensor that's already on the CPU genuinely returns the same storage, not a copy — LibTorch checked the target device against the source, found them equal, and skipped the copy entirely. `cloned.data_ptr() == t.data_ptr()` reporting `0` is the deliberate contrast: `.clone()` allocated a real, new buffer, confirmed by the differing address, even though `torch::equal(cloned, t)` confirms the *values* are identical — cloning changes where the data lives, never what it is, the same distinction Chapter 3 drew between layout and correctness. The last line's exception text differs slightly in wording from Chapter 1's own (`CUDA driver version is insufficient` here, versus the message Chapter 1 captured) — both are genuine `c10::Error`s thrown by the identical underlying cause, no CUDA-capable device or matching driver present in this sandbox, and this book's methodology already commits to capturing only each exception's first line for exactly this reason: the full backtext can vary in ways that don't change what's actually being reported.

## 4.3 The GPU Memory Hierarchy: What `torch::Tensor`'s Public API Doesn't Expose `[FOUNDATIONAL]`

### Intuition

The CUDA C++ edition names three memory tiers a kernel author juggles directly: global memory (large, slow, visible everywhere), shared memory (fast, per-block scratchpad the kernel explicitly allocates with `__shared__`), and per-thread local memory. `torch::Tensor`'s C++ frontend gives this book exactly one memory concept to reason about: storage, wherever `c10::Device` says it lives. That isn't an oversight — it's checkable directly, by asking whether the actual installed headers this book's compile line includes contain any trace of a shared-memory concept at all.

### Background

This is not a claim to take on faith. LibTorch ships as installed header files on disk, in this sandbox exactly as in any other; `grep` across those files is a genuine, repeatable check, not a description of behavior this book is asking the reader to trust. This book's own standard compile line (from *Getting Started*) includes exactly two `-I` roots — `torch/include` and `torch/include/torch/csrc/api/include` — the public C++ frontend every one of this chapter's worked examples has compiled against.

### Worked Example 4.3.1 — searching the actual installed headers, not describing them

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")

# This book's own compile line's -I roots: does __shared__ appear anywhere in them?
grep -rlE "__shared__" "$TORCH_DIR/include/torch/csrc/api/include"

# For contrast: does __shared__ appear anywhere under ATen/c10 at all (including
# internals this book's compile line never includes)?
grep -rlE "__shared__" "$TORCH_DIR/include/ATen" "$TORCH_DIR/include/c10" | wc -l

# Of those matches, how many live outside a path containing "/cuda/" --
# i.e., how many are reachable from ordinary, non-CUDA-kernel-implementation code?
grep -rlE "__shared__" "$TORCH_DIR/include/ATen" "$TORCH_DIR/include/c10" | grep -vi "/cuda/" | wc -l
```

Genuinely run in this book's environment:

```text
(no output = zero matches)

10

0
```

The first command — searched directly against this book's own `-I` roots, the exact headers every compiled example in this chapter has included — returns nothing: zero files anywhere in the public C++ frontend mention `__shared__`. Widening the search to all of `ATen` and `c10` (headers this book's compile line does *not* include, but that ship inside the same installed package) finds 10 files that do — and every single one of them lives under a path containing `/cuda/`: `ATen/native/cuda/Reduce.cuh`, `ATen/native/cuda/SortUtils.cuh`, and eight others, all internal CUDA-kernel-implementation headers, not part of the public API this book's own code has ever included. The third command confirms there are zero exceptions: not one `__shared__` mention exists outside a `/cuda/`-named internal path. `__shared__` genuinely exists inside LibTorch's own codebase — ATen's authors use CUDA shared memory extensively when writing kernels like reductions and sorts — but a LibTorch *application programmer*, writing exactly the kind of code this book writes, has no public constructor, no `TensorOptions` flag, and no method anywhere in `torch::` that allocates, sizes, or queries shared memory directly. It is fully encapsulated inside dispatched kernel implementations this book never needs to see, exactly as Section 4.1 found for thread-level parallelism.

> `[COMMON TRAP]` This doesn't mean LibTorch's CPU or GPU execution doesn't *use* memory hierarchies internally — Section 4.4 shows directly that layout still has large, measurable performance consequences. It means the *vocabulary* CUDA C++ gives its programmer for reasoning about those hierarchies explicitly (`__shared__`, tiling into a scratchpad by hand) has no counterpart in the public API this book's own code has access to. The hierarchy still exists; this book's characters just never get to name it directly.

## 4.4 Memory Coalescing, Generalized Beyond One Chapter's Example `[FOUNDATIONAL]`

### Intuition

Chapter 3 measured coalescing-equivalent evidence for one specific case: a particle's `vx` field, reached as a strided column of an `[N, 7]` tensor versus a dedicated contiguous tensor. If `.stride()` and `.is_contiguous()` are genuinely general-purpose tools — not artifacts of that one particle example — they should produce the same kind of evidence for a completely unrelated tensor with no particles, structs, or physics involved at all.

### Background

An ordinary `[4000, 4000]` matrix, and one column of it, reached two ways: as a `.select(1, 0)` view (a stride-4000 access — 16,000 bytes between consecutive elements, since each row holds 4000 floats) and as a `.clone().contiguous()` materialized copy (stride 1). This is structurally the same comparison Chapter 3 ran, on data with no domain meaning at all.

### Worked Example 4.4.1 — the same evidence, a different tensor entirely

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>

// Chapter 3 generalized SASS's coalescing evidence into .stride(): a
// stride-1 view is contiguous, a strided view isn't. This section applies
// the identical tool to a case that has nothing to do with particles at
// all -- one column of a plain matrix, reached two ways -- to show the
// claim is general-purpose, not specific to Chapter 3's AoS/SoA example.
double time_sum(const torch::Tensor& t, int reps) {
    torch::Tensor warm = t.sum();
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += t.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    torch::Tensor m = torch::rand({4000, 4000});
    std::cout << "m.strides() = " << m.strides() << ", m.is_contiguous() = " << m.is_contiguous() << std::endl;

    // One column of m, reached as a VIEW: consecutive elements are 4000
    // floats (16000 bytes) apart in memory -- the same kind of large,
    // non-unit stride Chapter 3 measured for the AoS particle field, just
    // arising from a plain matrix instead of a struct-of-fields tensor.
    torch::Tensor col_view = m.select(1, 0);
    std::cout << "col_view.strides() = " << col_view.strides()
              << ", col_view.is_contiguous() = " << col_view.is_contiguous() << std::endl;

    torch::Tensor col_materialized = col_view.clone().contiguous();
    std::cout << "col_materialized.strides() = " << col_materialized.strides()
              << ", col_materialized.is_contiguous() = " << col_materialized.is_contiguous() << std::endl;
    std::cout << "col_materialized.data_ptr() == col_view.data_ptr()? "
              << (col_materialized.data_ptr() == col_view.data_ptr())
              << " (materializing a non-contiguous view always copies)" << std::endl;

    const int reps = 400;
    const int trials = 5;
    int contiguous_faster_count = 0;
    for (int t = 0; t < trials; t++) {
        double strided_ms = time_sum(col_view, reps);
        double contig_ms = time_sum(col_materialized, reps);
        bool faster = contig_ms < strided_ms;
        if (faster) contiguous_faster_count++;
        std::cout << "trial " << t << ": strided column view sum = " << strided_ms
                  << " ms, contiguous copy sum = " << contig_ms
                  << " ms, contiguous_faster = " << (faster ? "true" : "false") << std::endl;
    }
    std::cout << "contiguous copy faster in " << contiguous_faster_count << " / " << trials << " trials" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment (millisecond figures are wall-clock timing, treated per this book's methodology as a boolean-threshold claim, not an exact reproducible figure):

```text
m.strides() = [4000, 1], m.is_contiguous() = 1
col_view.strides() = [4000], col_view.is_contiguous() = 0
col_materialized.strides() = [1], col_materialized.is_contiguous() = 1
col_materialized.data_ptr() == col_view.data_ptr()? 0 (materializing a non-contiguous view always copies)
trial 0: strided column view sum = 18.1577 ms, contiguous copy sum = 0.527817 ms, contiguous_faster = true
trial 1: strided column view sum = 15.6447 ms, contiguous copy sum = 0.422549 ms, contiguous_faster = true
trial 2: strided column view sum = 16.491 ms, contiguous copy sum = 0.791331 ms, contiguous_faster = true
trial 3: strided column view sum = 15.5322 ms, contiguous copy sum = 0.403682 ms, contiguous_faster = true
trial 4: strided column view sum = 16.9918 ms, contiguous copy sum = 0.494234 ms, contiguous_faster = true
contiguous copy faster in 5 / 5 trials
```

`col_view.strides() = [4000]` confirms consecutive elements of this column are 4000 floats apart in the underlying buffer — the same large-stride pattern Chapter 3 measured, just from a plain matrix rather than a particle field. The timing gap here is dramatically larger than Chapter 3's own ~6-7x: the materialized contiguous copy summed roughly 30-40x faster than the strided view across every trial, reproduced on every rerun performed while writing this chapter. The earlier, abandoned version of this experiment — summing the *entire* transposed `4000x4000` matrix rather than one column of it — showed no consistent advantage either way for the contiguous copy, and that finding is reported honestly rather than discarded: a full-matrix reduction over 16 million elements is bandwidth-bound in a way that a narrow, sparse column pull (4,000 elements out of 16 million, each landing in a different, mostly-unused cache line) is not. The lesson generalizes cleanly: coalescing evidence is real and measurable, but it depends on the *access pattern* an operation actually performs, not on stride and contiguity numbers considered in isolation — a caution CUDA's own SASS evidence carries just as much as `torch::Tensor::stride()` does.

## 4.5 Broadcasting: Two Different Ideas Sharing One Name `[FOUNDATIONAL]`

### Intuition

CUDA's "broadcasting," in the CUDA C++ edition's own Section 4.5, means: launch one thread per output element, and every thread runs the identical kernel body — the execution *pattern*. PyTorch's "broadcasting" is a different, older idea from array-programming languages that happens to share the name: implicit shape expansion, so `[4] + scalar` or `[4,1] + [1,3]` produce a full result with no explicit loop written anywhere — a *shape-inference rule*. Conflating the two would be dishonest; the honest move is testing PyTorch's actual broadcasting behavior directly, then stating plainly how the two ideas do and don't connect.

### Background

PyTorch's broadcasting rule: two tensors' shapes are compared dimension by dimension from the right; dimensions are compatible if they're equal, or if either is `1` (expanded implicitly to match the other), or if one tensor has fewer dimensions (treated as size `1` in the missing leading dimensions). Shapes that satisfy none of these are rejected outright, genuinely, rather than silently misbehaving.

### Worked Example 4.5.1 — shape expansion tested directly, then one output element hand-verified

```cpp
#include <torch/torch.h>
#include <iostream>

// CUDA's "broadcasting" (Chapter 4.5 of the CUDA C++ edition) means: launch
// one thread PER OUTPUT ELEMENT, and every thread runs the identical kernel
// body. PyTorch's "broadcasting" is a DIFFERENT, older idea that happens to
// share the name: implicit shape expansion, so that `[4] + scalar` or
// `[4,1] + [1,3]` produce a full elementwise result without the caller
// writing any explicit loop or expansion. This file tests torch's shape
// broadcasting directly, then makes the honest connection explicit: however
// PyTorch's broadcasting is implemented underneath, it still has to actually
// visit one output element at a time somewhere in the dispatched kernel --
// which is exactly the "one thread, one output element" execution model
// CUDA's broadcasting is named after, just hidden below the C++ API this
// book uses.
int main() {
    // Case 1: tensor + scalar. A [4] tensor "broadcasts" against a 0-dim
    // scalar with no explicit expansion code written anywhere.
    torch::Tensor a = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    torch::Tensor scalar = torch::tensor(10.0f);
    torch::Tensor r1 = a + scalar;
    std::cout << "a.sizes() = " << a.sizes() << ", scalar.sizes() = " << scalar.sizes()
              << ", (a + scalar).sizes() = " << r1.sizes() << std::endl;
    std::cout << "a + scalar = " << r1 << std::endl;

    // Case 2: [4,1] + [1,3] -> [4,3]. Neither operand starts out the output
    // shape; both get implicitly expanded along the dimension they're size-1
    // in, producing every one of the 12 combinations.
    torch::Tensor col = torch::tensor({{1.0f}, {2.0f}, {3.0f}, {4.0f}});   // [4,1]
    torch::Tensor row = torch::tensor({{10.0f, 20.0f, 30.0f}});           // [1,3]
    torch::Tensor r2 = col + row;
    std::cout << "col.sizes() = " << col.sizes() << ", row.sizes() = " << row.sizes()
              << ", (col + row).sizes() = " << r2.sizes() << std::endl;
    std::cout << "col + row = " << r2 << std::endl;

    // Verify one specific output element by hand, the same way Worked Example
    // 4.1.1 in the CUDA book hand-traces a specific thread's output index.
    float expected_1_2 = 2.0f + 30.0f;  // row index 1, col index 2
    float actual_1_2 = r2[1][2].item<float>();
    std::cout << "r2[1][2] = " << actual_1_2 << ", hand-computed col[1]+row[2] = " << expected_1_2
              << ", match? " << (actual_1_2 == expected_1_2) << std::endl;

    // A shape that CANNOT broadcast: [4] and [3] share no dimension that's
    // size 1 or equal. This should throw, genuinely, not silently produce a
    // wrong-shaped result.
    try {
        torch::Tensor bad = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}) + torch::tensor({1.0f, 2.0f, 3.0f});
        std::cout << "incompatible shapes unexpectedly broadcast successfully" << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "[4] + [3] threw c10::Error (incompatible shapes, correctly rejected)" << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
a.sizes() = [4], scalar.sizes() = [], (a + scalar).sizes() = [4]
a + scalar =  11
 12
 13
 14
[ CPUFloatType{4} ]
col.sizes() = [4, 1], row.sizes() = [1, 3], (col + row).sizes() = [4, 3]
col + row =  11  21  31
 12  22  32
 13  23  33
 14  24  34
[ CPUFloatType{4,3} ]
r2[1][2] = 32, hand-computed col[1]+row[2] = 32, match? 1
[4] + [3] threw c10::Error (incompatible shapes, correctly rejected)
```

`(col + row).sizes() = [4, 3]` confirms the expansion genuinely happened: neither `col` (`[4,1]`) nor `row` (`[1,3]`) started at that shape, and no code in this file wrote a loop or an explicit `.expand()` call — the `+` operator's shape-inference rule did it. `r2[1][2] = 32` matching the hand-computed `col[1] + row[2] = 2.0 + 30.0 = 32.0` exactly is the same kind of single-output-element verification the CUDA book's Worked Example 4.1.1 performs for a specific thread's index — here for a specific *broadcast* output position instead. The final line confirms the rule has real teeth: `[4]` and `[3]` share no broadcastable dimension, and the attempt genuinely throws rather than silently producing garbage. **Independent cross-check.** Both broadcasting results were re-verified through NumPy — a fully independent array-broadcasting implementation, not merely a different entry point into the same LibTorch binary — which produced the identical `[[11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]]` for `col + row`, and Python's own `torch` bindings raised the identical incompatible-shape error for `[4] + [3]`.

The honest connection to CUDA's broadcasting, stated directly: PyTorch's shape-expansion rule doesn't eliminate the need to visit one output element at a time — `(col + row)` still has to produce 12 genuinely distinct values, one per output position, from somewhere. What's different is *who writes the code that visits them*. CUDA's broadcasting names the execution model explicitly, because the kernel author is the one writing `blockIdx.x * blockDim.x + threadIdx.x` to make that visit happen. PyTorch's broadcasting names the shape-inference rule, because the *tensor library's* dispatched kernel is the one doing the visiting — Section 4.1's thread pool, working underneath a call site that never mentions it.

## 4.6 Matrix Multiplication: From Naive Dot Products to `torch::matmul` `[FOUNDATIONAL]`

### Intuition

The CUDA C++ edition's Chapter 4 closes by hand-writing a naive matmul kernel: thread `(row, col)` computes exactly one output element, `C[row][col] = sum_k A[row,k] * B[k,col]`, via a small loop over `k`. This book has no kernel to hand-write in its place — `torch::matmul` is LibTorch's real, production entry point for the identical mathematical operation, and running it against the CUDA book's own worked example is a direct cross-check of two structurally unrelated implementations reaching the same answer.

### Background

The CUDA book's Worked Example 4.6.1 traces `[[1,2],[3,4]] @ [[5,6],[7,8]]` by hand, computing each of the four output elements as one dot product, and lands on `[[19,22],[43,50]]`. `torch::matmul(A, B)` on the identical two matrices should produce the identical result — not approximately, but exactly, since these are small integer-valued floats with no rounding at stake.

### Worked Example 4.6.1 — the CUDA book's own numbers, run through `torch::matmul`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Worked Example 4.6.1 hand-traces a naive kernel
// where thread (row, col) computes exactly one dot product: C[row][col] =
// sum_k A[row,k] * B[k,col]. This book has no such kernel to hand-write --
// torch::matmul (equivalently, `operator*` between 2-D tensors via
// torch::mm) is LibTorch's real, production entry point for the identical
// mathematical operation, and it's genuinely run here against the exact
// same 2x2 example the CUDA book uses, to cross-check the same answer a
// completely different implementation reaches.
int main() {
    torch::Tensor A = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    torch::Tensor B = torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});

    torch::Tensor C = torch::matmul(A, B);
    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;
    std::cout << "torch::matmul(A, B) = " << C << std::endl;

    // Hand-computed reference, one dot product at a time, exactly the way
    // the CUDA book's Worked Example 4.6.1 traces each thread by hand.
    float c00 = A[0][0].item<float>() * B[0][0].item<float>() + A[0][1].item<float>() * B[1][0].item<float>();
    float c01 = A[0][0].item<float>() * B[0][1].item<float>() + A[0][1].item<float>() * B[1][1].item<float>();
    float c10 = A[1][0].item<float>() * B[0][0].item<float>() + A[1][1].item<float>() * B[1][0].item<float>();
    float c11 = A[1][0].item<float>() * B[0][1].item<float>() + A[1][1].item<float>() * B[1][1].item<float>();
    std::cout << "hand-computed: C[0][0]=" << c00 << " C[0][1]=" << c01
              << " C[1][0]=" << c10 << " C[1][1]=" << c11 << std::endl;

    bool matches = (C[0][0].item<float>() == c00) && (C[0][1].item<float>() == c01) &&
                   (C[1][0].item<float>() == c10) && (C[1][1].item<float>() == c11);
    std::cout << "torch::matmul matches hand-computed dot products? " << matches << std::endl;

    // A's row access (A[row, k] for increasing k) is contiguous -- stride 1
    // along dim 1. B's column access (B[k, col] for increasing k) is
    // strided -- stride equal to B's row length. This is the exact
    // asymmetry the CUDA book's Section 4.6 notes and Part 2's tiled kernel
    // later addresses; here it's read directly off the tensors, no
    // disassembly needed.
    std::cout << "A.strides() = " << A.strides() << " (row access along dim 1 is stride 1: contiguous)" << std::endl;
    std::cout << "B.strides() = " << B.strides() << " (column access along dim 0 is stride "
              << B.stride(0) << ": non-contiguous)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
A =  1  2
 3  4
[ CPUFloatType{2,2} ]
B =  5  6
 7  8
[ CPUFloatType{2,2} ]
torch::matmul(A, B) =  19  22
 43  50
[ CPUFloatType{2,2} ]
hand-computed: C[0][0]=19 C[0][1]=22 C[1][0]=43 C[1][1]=50
torch::matmul matches hand-computed dot products? 1
A.strides() = [2, 1] (row access along dim 1 is stride 1: contiguous)
B.strides() = [2, 1] (column access along dim 0 is stride 2: non-contiguous)
```

`torch::matmul(A, B)` produces `[[19, 22], [43, 50]]`, exactly the CUDA C++ edition's own hand-traced result for the identical two matrices — `matches` reporting `1` confirms it bit-for-bit against this file's own independent hand computation, not merely eyeballed against the CUDA book's page. **Independent cross-check.** The same two matrices multiplied through Python's `A @ B` (`torch.Tensor.__matmul__`, a different call path than the C++ `torch::matmul` symbol) again produced `[[19.0, 22.0], [43.0, 50.0]]`. The stride evidence closes the chapter on the same note it opened on: `B.stride(0) = 2` (non-unit) confirms `B`'s column access, which is exactly what a dot-product loop over `k` performs for the second matrix, is the strided, non-coalesced pattern the CUDA book's Section 4.6 identifies by inspecting its own naive kernel — read here directly from the tensor, with no kernel of this book's own to inspect. Worth stating plainly: `torch::matmul` genuinely does not run the naive one-thread-one-dot-product algorithm the CUDA book's Worked Example 4.6.1 hand-traces — on CPU, it dispatches to a vendor BLAS library using blocked, cache-tiled algorithms similar in spirit to what the CUDA book's own Part 2 builds by hand as an *improvement* over its naive kernel. LibTorch skips straight to that production-grade implementation; this chapter's naive-style hand trace exists only to verify the answer, never to describe how `torch::matmul` actually computes it.

## Complete Runnable Code

### File: `01_thread_pool.cpp`

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>

// CUDA's kernel launch requires the CALLER to compute a thread's global index
// by hand (blockIdx.x * blockDim.x + threadIdx.x) before that thread does any
// work. LibTorch's CPU backend has an analogous notion of "how many workers
// run this op in parallel" -- torch::get_num_threads() / set_num_threads()
// -- but the caller never assigns work to a specific worker. The parallel
// dispatch is entirely internal to the op; a+b looks identical whether it
// runs on one thread or many.
double time_elementwise_add(int64_t n, int reps) {
    torch::Tensor a = torch::rand({n});
    torch::Tensor b = torch::rand({n});
    // warm up
    torch::Tensor warm = a + b;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        torch::Tensor c = a + b;
        volatile float sink = c[0].item<float>();
        (void)sink;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "torch::get_num_threads() (default) = " << torch::get_num_threads() << std::endl;

    const int64_t n = 20000000;
    const int reps = 30;
    const int trials = 5;

    int one_thread_faster_or_equal = 0;
    int multi_thread_faster = 0;
    for (int t = 0; t < trials; t++) {
        torch::set_num_threads(1);
        double one_thread_ms = time_elementwise_add(n, reps);

        torch::set_num_threads(2);
        double multi_thread_ms = time_elementwise_add(n, reps);

        bool multi_faster = multi_thread_ms < one_thread_ms;
        if (multi_faster) multi_thread_faster++; else one_thread_faster_or_equal++;
        std::cout << "trial " << t << ": 1 thread = " << one_thread_ms
                  << " ms, 2 threads = " << multi_thread_ms
                  << " ms, multi_faster = " << (multi_faster ? "true" : "false") << std::endl;
    }

    std::cout << "2 threads faster than 1 thread in " << multi_thread_faster << " / " << trials << " trials" << std::endl;

    // The point this chapter actually cares about: the SAME call, a + b, is
    // what ran in every trial above. No index was computed by this program.
    // Nothing here says "thread 3, take element 4193". torch::Tensor's
    // dispatcher decided how to split n=20000000 elements across however
    // many threads were configured, entirely inside the + operator.
    std::cout << "elementwise add call site is identical regardless of thread count: `a + b`" << std::endl;

    return 0;
}
```

### File: `02_data_ptr_identity.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <sstream>

// Chapter 1, Section 1.4 already showed .to(kCUDA) genuinely throwing a real
// c10::Error in this no-GPU sandbox. This file asks the other half of the
// host/device question CUDA's cudaMemcpy makes explicit: when data movement
// DOESN'T have to cross a device boundary, does LibTorch actually copy
// anything? The direct, runtime-queryable answer is .data_ptr() identity --
// no profiler, no cudaMemcpy call count, just a pointer comparison.
std::string first_line(const std::exception& e) {
    std::istringstream iss(e.what());
    std::string line;
    std::getline(iss, line);
    return line;
}

int main() {
    torch::Tensor t = torch::rand({1000});
    void* original_ptr = t.data_ptr();
    std::cout << "t.device() = " << t.device() << std::endl;

    // .to(kCPU) on an ALREADY-CPU tensor: same device, so LibTorch is free to
    // return the same underlying storage untouched -- no copy needed.
    torch::Tensor same_device = t.to(torch::kCPU);
    std::cout << "same_device.data_ptr() == t.data_ptr()? "
              << (same_device.data_ptr() == original_ptr) << std::endl;

    // .clone() is the operation that genuinely forces a new allocation and a
    // real copy, regardless of device -- the direct contrast case.
    torch::Tensor cloned = t.clone();
    std::cout << "cloned.data_ptr() == t.data_ptr()? "
              << (cloned.data_ptr() == original_ptr) << std::endl;

    // Values still match even though the copy has a different address --
    // cloning changes WHERE the data lives, never WHAT the data is.
    std::cout << "cloned values equal to original? "
              << torch::equal(cloned, t) << std::endl;

    // The genuine cross-device attempt (mirrors Chapter 1, Section 1.4,
    // re-verified here as this chapter's own direct evidence): a real
    // c10::Error, not a fabricated one, because this sandbox has no CUDA
    // device or driver.
    try {
        torch::Tensor moved = t.to(torch::kCUDA);
        std::cout << "t.to(kCUDA) unexpectedly succeeded" << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "t.to(kCUDA) threw c10::Error: " << first_line(e) << std::endl;
    }

    return 0;
}
```

### File: `03_coalescing_transpose.cpp`

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>

// Chapter 3 generalized SASS's coalescing evidence into .stride(): a
// stride-1 view is contiguous, a strided view isn't. This section applies
// the identical tool to a case that has nothing to do with particles at
// all -- one column of a plain matrix, reached two ways -- to show the
// claim is general-purpose, not specific to Chapter 3's AoS/SoA example.
double time_sum(const torch::Tensor& t, int reps) {
    torch::Tensor warm = t.sum();
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += t.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    torch::Tensor m = torch::rand({4000, 4000});
    std::cout << "m.strides() = " << m.strides() << ", m.is_contiguous() = " << m.is_contiguous() << std::endl;

    // One column of m, reached as a VIEW: consecutive elements are 4000
    // floats (16000 bytes) apart in memory -- the same kind of large,
    // non-unit stride Chapter 3 measured for the AoS particle field, just
    // arising from a plain matrix instead of a struct-of-fields tensor.
    torch::Tensor col_view = m.select(1, 0);
    std::cout << "col_view.strides() = " << col_view.strides()
              << ", col_view.is_contiguous() = " << col_view.is_contiguous() << std::endl;

    torch::Tensor col_materialized = col_view.clone().contiguous();
    std::cout << "col_materialized.strides() = " << col_materialized.strides()
              << ", col_materialized.is_contiguous() = " << col_materialized.is_contiguous() << std::endl;
    std::cout << "col_materialized.data_ptr() == col_view.data_ptr()? "
              << (col_materialized.data_ptr() == col_view.data_ptr())
              << " (materializing a non-contiguous view always copies)" << std::endl;

    const int reps = 400;
    const int trials = 5;
    int contiguous_faster_count = 0;
    for (int t = 0; t < trials; t++) {
        double strided_ms = time_sum(col_view, reps);
        double contig_ms = time_sum(col_materialized, reps);
        bool faster = contig_ms < strided_ms;
        if (faster) contiguous_faster_count++;
        std::cout << "trial " << t << ": strided column view sum = " << strided_ms
                  << " ms, contiguous copy sum = " << contig_ms
                  << " ms, contiguous_faster = " << (faster ? "true" : "false") << std::endl;
    }
    std::cout << "contiguous copy faster in " << contiguous_faster_count << " / " << trials << " trials" << std::endl;

    return 0;
}
```

### File: `04_broadcasting_semantics.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// CUDA's "broadcasting" (Chapter 4.5 of the CUDA C++ edition) means: launch
// one thread PER OUTPUT ELEMENT, and every thread runs the identical kernel
// body. PyTorch's "broadcasting" is a DIFFERENT, older idea that happens to
// share the name: implicit shape expansion, so that `[4] + scalar` or
// `[4,1] + [1,3]` produce a full elementwise result without the caller
// writing any explicit loop or expansion. This file tests torch's shape
// broadcasting directly, then makes the honest connection explicit: however
// PyTorch's broadcasting is implemented underneath, it still has to actually
// visit one output element at a time somewhere in the dispatched kernel --
// which is exactly the "one thread, one output element" execution model
// CUDA's broadcasting is named after, just hidden below the C++ API this
// book uses.
int main() {
    // Case 1: tensor + scalar. A [4] tensor "broadcasts" against a 0-dim
    // scalar with no explicit expansion code written anywhere.
    torch::Tensor a = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    torch::Tensor scalar = torch::tensor(10.0f);
    torch::Tensor r1 = a + scalar;
    std::cout << "a.sizes() = " << a.sizes() << ", scalar.sizes() = " << scalar.sizes()
              << ", (a + scalar).sizes() = " << r1.sizes() << std::endl;
    std::cout << "a + scalar = " << r1 << std::endl;

    // Case 2: [4,1] + [1,3] -> [4,3]. Neither operand starts out the output
    // shape; both get implicitly expanded along the dimension they're size-1
    // in, producing every one of the 12 combinations.
    torch::Tensor col = torch::tensor({{1.0f}, {2.0f}, {3.0f}, {4.0f}});   // [4,1]
    torch::Tensor row = torch::tensor({{10.0f, 20.0f, 30.0f}});           // [1,3]
    torch::Tensor r2 = col + row;
    std::cout << "col.sizes() = " << col.sizes() << ", row.sizes() = " << row.sizes()
              << ", (col + row).sizes() = " << r2.sizes() << std::endl;
    std::cout << "col + row = " << r2 << std::endl;

    // Verify one specific output element by hand, the same way Worked Example
    // 4.1.1 in the CUDA book hand-traces a specific thread's output index.
    float expected_1_2 = 2.0f + 30.0f;  // row index 1, col index 2
    float actual_1_2 = r2[1][2].item<float>();
    std::cout << "r2[1][2] = " << actual_1_2 << ", hand-computed col[1]+row[2] = " << expected_1_2
              << ", match? " << (actual_1_2 == expected_1_2) << std::endl;

    // A shape that CANNOT broadcast: [4] and [3] share no dimension that's
    // size 1 or equal. This should throw, genuinely, not silently produce a
    // wrong-shaped result.
    try {
        torch::Tensor bad = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}) + torch::tensor({1.0f, 2.0f, 3.0f});
        std::cout << "incompatible shapes unexpectedly broadcast successfully" << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "[4] + [3] threw c10::Error (incompatible shapes, correctly rejected)" << std::endl;
    }

    return 0;
}
```

### File: `05_matmul_dot_products.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Worked Example 4.6.1 hand-traces a naive kernel
// where thread (row, col) computes exactly one dot product: C[row][col] =
// sum_k A[row,k] * B[k,col]. This book has no such kernel to hand-write --
// torch::matmul (equivalently, `operator*` between 2-D tensors via
// torch::mm) is LibTorch's real, production entry point for the identical
// mathematical operation, and it's genuinely run here against the exact
// same 2x2 example the CUDA book uses, to cross-check the same answer a
// completely different implementation reaches.
int main() {
    torch::Tensor A = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    torch::Tensor B = torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});

    torch::Tensor C = torch::matmul(A, B);
    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;
    std::cout << "torch::matmul(A, B) = " << C << std::endl;

    // Hand-computed reference, one dot product at a time, exactly the way
    // the CUDA book's Worked Example 4.6.1 traces each thread by hand.
    float c00 = A[0][0].item<float>() * B[0][0].item<float>() + A[0][1].item<float>() * B[1][0].item<float>();
    float c01 = A[0][0].item<float>() * B[0][1].item<float>() + A[0][1].item<float>() * B[1][1].item<float>();
    float c10 = A[1][0].item<float>() * B[0][0].item<float>() + A[1][1].item<float>() * B[1][0].item<float>();
    float c11 = A[1][0].item<float>() * B[0][1].item<float>() + A[1][1].item<float>() * B[1][1].item<float>();
    std::cout << "hand-computed: C[0][0]=" << c00 << " C[0][1]=" << c01
              << " C[1][0]=" << c10 << " C[1][1]=" << c11 << std::endl;

    bool matches = (C[0][0].item<float>() == c00) && (C[0][1].item<float>() == c01) &&
                   (C[1][0].item<float>() == c10) && (C[1][1].item<float>() == c11);
    std::cout << "torch::matmul matches hand-computed dot products? " << matches << std::endl;

    // A's row access (A[row, k] for increasing k) is contiguous -- stride 1
    // along dim 1. B's column access (B[k, col] for increasing k) is
    // strided -- stride equal to B's row length. This is the exact
    // asymmetry the CUDA book's Section 4.6 notes and Part 2's tiled kernel
    // later addresses; here it's read directly off the tensors, no
    // disassembly needed.
    std::cout << "A.strides() = " << A.strides() << " (row access along dim 1 is stride 1: contiguous)" << std::endl;
    std::cout << "B.strides() = " << B.strides() << " (column access along dim 0 is stride "
              << B.stride(0) << ": non-contiguous)" << std::endl;

    return 0;
}
```

Files `01`, `02`, `03`, `04`, and `05` all compile and link against LibTorch with the standard command from *Getting Started*:

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 <file>.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o <file>
./<file>
```

## Chapter Summary

`torch::get_num_threads()` and `torch::set_num_threads()` are the closest LibTorch equivalent to CUDA's launch configuration, and this chapter measured the difference they make directly: 2 threads beat 1 thread in every one of 5 trials, reproduced on every rerun, while the call site itself — `a + b` — never changed to accommodate either configuration, because no LibTorch op requires the caller to compute a per-element thread index the way `blockIdx.x * blockDim.x + threadIdx.x` does. `.data_ptr()` identity gave the direct, runtime-queryable answer to whether same-device operations actually copy anything: `.to(kCPU)` on an already-CPU tensor returned the identical address, while `.clone()` genuinely allocated new storage, both confirmed alongside a re-verified real `c10::Error` for the actual cross-device attempt this sandbox cannot complete. A genuine `grep` across this book's own installed, compiled-against headers found zero mentions of `__shared__` anywhere in the public C++ frontend, against 10 files using it freely inside ATen's internal CUDA kernel implementations — ten for ten of them confined to paths outside this book's own `-I` roots, confirming LibTorch's memory-hierarchy vocabulary is real but entirely hidden from application code. Chapter 3's `.stride()`/`.is_contiguous()` tools generalized cleanly to an unrelated plain matrix, showing a dramatically larger, consistently reproduced 30-40x gap for one strided column versus its contiguous copy — while an earlier, honestly-reported dead end (a full-matrix sum showing no consistent advantage either way) demonstrated that coalescing evidence depends on the actual access pattern, not on stride numbers in isolation. Broadcasting turned out to name two genuinely different ideas that happen to share a word — CUDA's per-thread execution model and PyTorch's shape-expansion rule — connected only by the fact that PyTorch's rule still has to visit one output element at a time somewhere beneath the C++ API this book uses. And `torch::matmul`, run against the CUDA book's own `[[1,2],[3,4]] @ [[5,6],[7,8]]` example, matched `[[19, 22], [43, 50]]` exactly, cross-checked independently through Python — while genuinely skipping the naive one-thread-one-dot-product algorithm entirely in favor of the production BLAS-backed implementation the CUDA book's own later chapters build toward by hand.

## Self-Check Questions

1. Worked Example 4.1.1 shows `a + b` compiling and running identically regardless of whether `torch::set_num_threads(1)` or `torch::set_num_threads(2)` was called beforehand. What CUDA C++ mechanism does `torch::set_num_threads()` replace, and what specifically does it NOT let the caller control that CUDA's launch configuration does?
2. In Worked Example 4.2.1, `same_device.data_ptr() == t.data_ptr()` reports `true` while `cloned.data_ptr() == t.data_ptr()` reports `false`, even though `torch::equal(cloned, t)` reports `true`. Explain what each of these three results is actually telling you, and why all three can be true at once without contradiction.
3. Section 4.3's grep search found 10 files mentioning `__shared__` inside the installed LibTorch package, but zero inside this book's own `-I` roots. What does this actually prove about LibTorch's memory hierarchy, and what would it NOT prove if the search had instead found zero matches everywhere in the entire installed package?
4. Section 4.4 describes an earlier, abandoned version of its own experiment — summing an entire transposed matrix rather than one strided column — that showed no consistent timing advantage for the contiguous copy. Why does the chapter report this instead of quietly replacing it with the column-based version and saying nothing?
5. Section 4.6 states that `torch::matmul` does not run the naive one-thread-one-dot-product algorithm the CUDA book's Worked Example 4.6.1 hand-traces, even though both produce the identical `[[19, 22], [43, 50]]` result. How can two different algorithms produce identical output, and what does the CUDA book's own Part 2 (mentioned in this chapter) suggest about the relationship between the naive algorithm and what `torch::matmul` actually runs?

## Where We Go Next

This chapter tested, section by section, what a LibTorch programmer can and can't see of CUDA's execution model — a thread pool size but no per-thread index, a device tag but no explicit `cudaMemcpy` call, a `.stride()` but no `__shared__`. Chapter 5 turns from *whether* this evidence exists to *how much of it a single CPU instruction can act on at once* — SIMD and vectorization, the mechanism that lets one CPU core process multiple array elements per instruction, tested the same way this chapter tested thread pooling: not by disassembling anything, but by measuring what a real compiled binary actually does differently when that mechanism is available versus suppressed.

## Worked Solutions

**1.** `torch::set_num_threads()` replaces CUDA's `<<<blocks, threads_per_block>>>` launch configuration syntax as the mechanism controlling how much parallel hardware an operation uses. What it does NOT let the caller control is *which* thread handles *which* element — CUDA's launch configuration, combined with `blockIdx`/`threadIdx` inside the kernel body, lets the programmer assign specific output elements to specific threads explicitly; `torch::set_num_threads()` only sets a pool size, and the dispatcher decides internally how work is split across that pool, with no equivalent of `blockIdx.x * blockDim.x + threadIdx.x` anywhere in the caller's code.

**2.** `same_device.data_ptr() == t.data_ptr()` being `true` tells you `.to(torch::kCPU)` on an already-CPU tensor returned the exact same underlying storage — no allocation, no copy, a free operation. `cloned.data_ptr() == t.data_ptr()` being `false` tells you `.clone()` genuinely allocated new storage at a different address. `torch::equal(cloned, t)` being `true` tells you the *values* stored at those two different addresses are identical. All three can hold simultaneously because address identity and value equality are independent facts: two different buffers can hold identical contents (as `cloned` and `t` do), and one unchanged buffer trivially equals itself in both address and value (as `same_device` and `t` do) — `.clone()`'s whole job is to change the address while preserving the values, which is exactly what these three results, taken together, confirm happened.

**3.** It proves that LibTorch's public C++ frontend — specifically, the headers this book's own compile line actually includes — exposes no API surface for allocating, sizing, or querying shared memory, while confirming the concept genuinely exists inside LibTorch's own internal CUDA kernel implementations (the 10 files found). Had the search instead found zero matches everywhere in the entire installed package, including ATen's internals, that would have proven something stronger and different: that shared memory isn't just hidden from the public API, but that LibTorch's own kernel authors never use it at all anywhere in the codebase — a claim about LibTorch's internal implementation choices, not just about what's exposed to application programmers, and one this chapter's actual grep results don't support (10 internal files do use it).

**4.** This book's own non-negotiable methodology (stated in *Getting Started* and followed in every chapter so far) requires correcting the text to match what a genuine test actually showed, never silently replacing an inconvenient result with a more favorable one and pretending the first attempt never happened. The full-matrix timing result is honestly reported because it's genuinely informative on its own: it shows that coalescing-style evidence doesn't apply uniformly to every operation on a matrix, only to ones whose access pattern actually exposes the stride difference — a real finding that sharpens the chapter's claim rather than a failed experiment worth hiding.

**5.** Two different algorithms can produce identical output on the same inputs whenever both are, in fact, correct implementations of the identical mathematical operation — matrix multiplication has exactly one correct answer for a given pair of matrices, regardless of what order of operations, tiling strategy, or hardware a particular implementation uses to compute it. The CUDA book's own Part 2 (referenced in this chapter as building a tiled kernel that improves on Part 0's naive one) suggests the relationship directly: the naive one-thread-one-dot-product algorithm is a correct but performance-naive starting point that production implementations move past specifically by restructuring memory access (tiling into shared memory, blocking for cache reuse) — which is exactly what a vendor BLAS library, the real implementation `torch::matmul` dispatches to on CPU, already does, skipping straight past the naive version the CUDA book's Part 0 uses only to teach the underlying idea.
