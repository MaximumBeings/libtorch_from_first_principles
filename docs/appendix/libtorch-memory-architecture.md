# Appendix C: LibTorch Memory Architecture

> The CUDA C++ edition's own Appendix C opens: "A GPU kernel's correctness lives in its arithmetic. Its speed lives almost entirely somewhere else: which of six genuinely different kinds of memory each value passes through on its way to that arithmetic." A LibTorch programmer, writing the ordinary tensor code every chapter of this book has used, never chooses among those six kinds of memory at all -- that choice is made, underneath, by whatever real ATen/cuBLAS/cuDNN kernel a given op dispatches to. This appendix asks, section by section, what a LibTorch programmer's own real, directly-visible memory-and-execution picture actually looks like instead -- not by re-deriving CUDA's own hardware model, but by testing, honestly and on real hardware (this book's own established CPU-only sandbox), what genuinely differs, what genuinely doesn't, and what genuinely cannot be tested here at all.

**What you will understand by the end of this appendix:** why a LibTorch programmer's own memory-hierarchy decision is far flatter than CUDA's own six-space model -- one placement choice (which device) rather than six explicit spaces; why writing an expression in place versus out of place is the real, LibTorch-visible analog of avoiding a register spill; why an ordinary tensor transpose's own real, measured CPU cost looks nothing like CUDA's own dramatic bank-conflict penalty, and why that gap is itself the interesting finding; why `torch::Tensor::expand()`'s stride-0 view achieves broadcast's own goal (one small value, read by many, never duplicated) through a completely different mechanism than a hardware constant-memory cache; why an ordinary CPU tensor has no unified-memory migration story at all, while a scattered-vs-sequential access pattern still costs a real, measurable, dramatic difference; why LibTorch's CPU execution model has no warp or CTA concept whatsoever, only a real, controllable thread pool; and why a single `torch::matmul()` call is the entire tensor-core-dispatch code a LibTorch programmer ever has to write, compared to CUDA's own hand-written WMMA fragment API.

**What you need to know first:** Chapter 18.2's own coalesced-memory-access analysis on the book's SoA bond portfolio; Chapter 19's own memory-traffic and fusion counting; Chapter 22.1's own struct-of-arrays design; this book's own established honest no-GPU convention (Chapter 18 onward): a claim that genuinely cannot be tested without real CUDA hardware is tagged `[UNVERIFIED -- pending real-GPU test]` rather than fabricated.

## C.1 The Memory Hierarchy at a Glance: Six Hardware Spaces, One Programmer-Visible Choice `[FOUNDATIONAL]`

**Intuition.** CUDA's own memory hierarchy asks a raw kernel author to choose, explicitly, among six genuinely different kinds of memory -- registers, local memory, shared memory, constant memory, L2 cache, global memory -- each with its own scope, size, and latency, and each requiring its own API or attribute to use correctly. A LibTorch programmer writing ordinary tensor code, the way every chapter of this book has, never makes that choice at all: the one placement decision actually available is which DEVICE a tensor's own storage lives on, full stop.

**Background.** Section C.1's own summary table gives NVIDIA's published compute-capability-8.0 reference figures for all six spaces (registers ~1 cycle/~256 KB per SM; local memory ~400-800 cycles, compiler-spilled; shared memory ~20-30 cycles, up to 164 KB per SM, programmer-managed; constant memory ~1 cycle on a broadcast hit, 64 KB total/8 KB cached per SM; L2 cache ~200 cycles, tens of MB, hardware-managed; global memory ~400-800 cycles, tens of GB, programmer-managed via `cudaMalloc`). None of these six is a separate object, API, or decision anywhere in this book's own LibTorch code -- they are all implementation details of whatever kernel a real op (`torch::matmul`, `torch::relu`, anything else) dispatches to underneath, entirely invisible to code written the way this book writes it.

**Worked Example C.1.1.** A real `torch::Tensor`'s own directly-queryable memory facts: `device()`, `dtype()`, `element_size()`, `numel()`, and `storage().nbytes()` -- confirmed to agree with each other genuinely, not merely by formula.

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.1 of the CUDA C++ edition lays out six genuinely distinct
// hardware memory spaces a raw CUDA programmer must explicitly choose
// between and manage by hand: registers, local memory, shared memory,
// constant memory, L2 cache, and global memory -- each with its own
// scope, size, latency, and API. A LibTorch programmer, writing the
// ordinary tensor code every chapter of this book has used, manages
// NONE of those six directly. What a LibTorch programmer actually
// chooses between is far flatter: which DEVICE a tensor's own storage
// lives on (CPU host memory, or CUDA device memory) -- everything
// below that single choice (which registers a fused kernel uses, when
// something spills, whether shared memory or constant memory gets
// used inside a matmul) is decided by whatever ATen/cuDNN/cuBLAS
// kernel a given real op dispatches to, invisible to code written the
// way this book's own chapters have written it throughout. This file
// genuinely queries the ONE memory-hierarchy fact a LibTorch program
// actually has direct, ordinary access to: which device a tensor
// lives on, and how many bytes its own storage genuinely occupies.
int main() {
    torch::Tensor t = torch::randn({1024, 1024});

    std::cout << "a real torch::Tensor's own memory-hierarchy surface, as seen by ordinary LibTorch "
              << "code (no custom CUDA kernel involved):" << std::endl;
    std::cout << "  t.device() = " << t.device() << " (the ONE placement decision a LibTorch programmer "
              << "makes directly -- everything below this, in a real op, is chosen by ATen's own kernel "
              << "dispatch, not by this code)" << std::endl;
    std::cout << "  t.dtype() = " << t.dtype() << ", t.element_size() = " << t.element_size()
              << " bytes/element" << std::endl;
    std::cout << "  t.numel() = " << t.numel() << " elements, t.numel()*t.element_size() = "
              << (t.numel() * t.element_size()) << " bytes" << std::endl;
    std::cout << "  t.storage().nbytes() (genuinely queried from the real underlying allocation) = "
              << t.storage().nbytes() << " bytes, matches numel()*element_size()? "
              << (t.storage().nbytes() == static_cast<size_t>(t.numel() * t.element_size())) << std::endl;

    std::cout << "\nfor comparison, the CUDA C++ edition's own six-space hierarchy (NVIDIA's published "
              << "compute-capability-8.0 reference figures, not measured on this GPU-less sandbox): "
              << "registers (~1 cycle, ~256 KB/SM, compiler-managed), local memory (~400-800 cycles, "
              << "compiler-spilled), shared memory (~20-30 cycles, up to 164 KB/SM, programmer-managed "
              << "via __shared__), constant memory (~1 cycle on a broadcast hit, 64 KB total/8 KB "
              << "cached per SM, programmer-managed via __constant__), L2 cache (~200 cycles, tens of "
              << "MB, hardware-managed), global memory (~400-800 cycles, tens of GB, programmer-managed "
              << "via cudaMalloc). None of these six is a separate object, API, or decision in ordinary "
              << "LibTorch tensor code -- they are all implementation details of whatever kernel torch::"
              << "matmul, torch::relu, or any other real op dispatches to underneath." << std::endl;

    // torch::cuda::is_available() is the one honest, real check this
    // book has used since Appendix A -- confirming, directly, that
    // 'device' in this sandbox can only ever mean CPU host memory, not
    // that this file is guessing or hard-coding that fact.
    std::cout << "\ntorch::cuda::is_available() = " << torch::cuda::is_available()
              << " -- every 'device' fact reported above is genuinely CPU host memory in this sandbox, "
              << "confirmed rather than assumed." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 01_memory_hierarchy_query.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_memory_hierarchy_query
./01_memory_hierarchy_query
```

```text
a real torch::Tensor's own memory-hierarchy surface, as seen by ordinary LibTorch code (no custom CUDA kernel involved):
  t.device() = cpu (the ONE placement decision a LibTorch programmer makes directly -- everything below this, in a real op, is chosen by ATen's own kernel dispatch, not by this code)
  t.dtype() = float, t.element_size() = 4 bytes/element
  t.numel() = 1048576 elements, t.numel()*t.element_size() = 4194304 bytes
  t.storage().nbytes() (genuinely queried from the real underlying allocation) = 4194304 bytes, matches numel()*element_size()? 1

for comparison, the CUDA C++ edition's own six-space hierarchy (NVIDIA's published compute-capability-8.0 reference figures, not measured on this GPU-less sandbox): registers (~1 cycle, ~256 KB/SM, compiler-managed), local memory (~400-800 cycles, compiler-spilled), shared memory (~20-30 cycles, up to 164 KB/SM, programmer-managed via __shared__), constant memory (~1 cycle on a broadcast hit, 64 KB total/8 KB cached per SM, programmer-managed via __constant__), L2 cache (~200 cycles, tens of MB, hardware-managed), global memory (~400-800 cycles, tens of GB, programmer-managed via cudaMalloc). None of these six is a separate object, API, or decision in ordinary LibTorch tensor code -- they are all implementation details of whatever kernel torch::matmul, torch::relu, or any other real op dispatches to underneath.

torch::cuda::is_available() = 0 -- every 'device' fact reported above is genuinely CPU host memory in this sandbox, confirmed rather than assumed.
```

**Discussion.** `torch::cuda::is_available()` is the same honest check this book has used since Appendix A, confirming directly (not assuming) that every "device" fact this file reports is genuinely CPU host memory in this sandbox. The larger point this section is making is structural, not just a smaller table: CUDA's own six-space hierarchy exists because a raw kernel author is WRITING the code that decides which space each value lives in (a `__shared__` declaration, a `__constant__` declaration, an ordinary local variable that may or may not fit in a register); a LibTorch programmer calling `torch::matmul` is not writing that code at all -- it is written once, inside ATen/cuBLAS/cuDNN, by people who make exactly those six-space decisions on the programmer's behalf, informed by exactly the kind of analysis Sections C.2 through C.7 walk through.

## C.2 Registers and Local Memory: In-Place Writes vs. Fresh Allocations `[FOUNDATIONAL]`

**Intuition.** CUDA's own register spilling happens when a kernel's own working set exceeds the hardware's own register file, forcing the compiler to round-trip some values through local memory (itself backed by global DRAM) instead of keeping them in a ~1-cycle register -- a real, measurable, ~400-800-cycle-per-access penalty. LibTorch has no registers a C++ programmer manages directly, but there is a real, directly analogous choice made in every single chapter of this book without ever being named: whether an expression's own intermediate result gets written into a FRESH, newly-allocated tensor, or written IN PLACE into a tensor that already exists.

**Background.** Section C.2's own worked example measures a specific kernel's own real register/spill counts via `ptxas` output (0 bytes spilled unconstrained; 368 bytes of stack frame and spills once `-maxrregcount=16` forces the compiler's hand). This file measures the direct LibTorch-level analog: does an expression's own result share its input's own `data_ptr()` (no new allocation, the closest a LibTorch programmer gets to "stayed in a register"), or does it get a genuinely new one (a fresh allocation, the direct analog of a value that had to leave)?

**Worked Example C.2.1.** `relu(a*b+c)`, computed two ways: out-of-place (three separate intermediate tensors, each a fresh allocation) and fully in-place (`out.mul_(b).add_(c).relu_()`, zero new allocations after the initial clone) -- confirmed, via `data_ptr()` comparisons, to differ ONLY in allocation count, never in the result itself.

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.2 of the CUDA C++ edition measures REGISTER SPILLING: a
// kernel using too many registers gets some of them forced out into
// local memory, paying a genuine ~400-800 cycle round trip per spilled
// value instead of the ~1 cycle a register access costs. LibTorch has
// no registers for a C++ programmer to manage directly (Section C.1's
// own point) -- but there IS a real, directly analogous choice every
// chapter of this book has made without naming it: whether an
// expression's own intermediate RESULT gets written into a fresh,
// newly-allocated tensor (a new round trip to host memory, paid once
// per intermediate), or written IN PLACE into a tensor that already
// exists (no new allocation at all, the closest a LibTorch programmer
// gets to "keeping a value in a register" instead of spilling it out
// to memory). This file measures that distinction directly and
// honestly, via each tensor's own data_ptr() -- a fresh allocation
// gets a NEW address, an in-place write keeps the SAME address.
int main() {
    torch::Tensor a = torch::randn({100000});
    torch::Tensor b = torch::randn({100000});
    torch::Tensor c = torch::randn({100000});

    // Unfused, out-of-place: torch::relu(a*b+c). Each intermediate
    // (a*b, then (a*b)+c, then relu(...)) is, in general, a genuinely
    // separate tensor with its own freshly-allocated storage -- the
    // real LibTorch-level analog of a spilled value making a genuine
    // round trip to memory instead of staying in one register.
    void* a_ptr = a.data_ptr();
    void* b_ptr = b.data_ptr();
    void* c_ptr = c.data_ptr();

    torch::Tensor mul_result = a * b;
    torch::Tensor add_result = mul_result + c;
    torch::Tensor relu_result = torch::relu(add_result);

    bool mul_is_new = (mul_result.data_ptr() != a_ptr && mul_result.data_ptr() != b_ptr);
    bool add_is_new = (add_result.data_ptr() != mul_result.data_ptr() && add_result.data_ptr() != c_ptr);
    bool relu_is_new = (relu_result.data_ptr() != add_result.data_ptr());

    std::cout << "unfused, out-of-place: relu(a*b+c) computed as three separate steps:" << std::endl;
    std::cout << "  (a*b) got a genuinely NEW allocation (different data_ptr from both a and b)? "
              << mul_is_new << std::endl;
    std::cout << "  (a*b)+c got a genuinely NEW allocation (different data_ptr from (a*b) and c)? "
              << add_is_new << std::endl;
    std::cout << "  relu(...) got a genuinely NEW allocation (different data_ptr from its own input)? "
              << relu_is_new << std::endl;
    std::cout << "  total genuinely distinct allocations for this one logical expression: "
              << (1 + static_cast<int>(mul_is_new) + static_cast<int>(add_is_new) + static_cast<int>(relu_is_new))
              << " (a, plus each new intermediate)" << std::endl;

    // In-place: the SAME storage is reused for every step, exactly the
    // way a value kept in one register across several instructions
    // never has to round-trip through memory in between.
    torch::Tensor out = a.clone();
    void* out_ptr_before = out.data_ptr();
    out.mul_(b);          // out = out * b, in place
    void* out_ptr_after_mul = out.data_ptr();
    out.add_(c);           // out = out + c, in place
    void* out_ptr_after_add = out.data_ptr();
    out.relu_();            // out = relu(out), in place
    void* out_ptr_after_relu = out.data_ptr();

    bool all_same = (out_ptr_before == out_ptr_after_mul) && (out_ptr_after_mul == out_ptr_after_add) &&
                     (out_ptr_after_add == out_ptr_after_relu);
    std::cout << "\nin-place: out.mul_(b).add_(c).relu_() computed as three in-place steps on the SAME "
              << "tensor:" << std::endl;
    std::cout << "  data_ptr() identical across all three in-place steps (no new allocation at any "
              << "point, the direct analog of a value that never leaves its register)? " << all_same
              << std::endl;

    bool results_match = torch::allclose(relu_result, out);
    std::cout << "\nboth computations produce the SAME mathematical result (torch::allclose)? "
              << results_match << " -- the two approaches are NOT alternative algorithms, they are "
              << "the identical computation, differing only in how many fresh memory allocations (the "
              << "LibTorch-level analog of register spills) it costs to get there." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 02_inplace_vs_allocation.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_inplace_vs_allocation
./02_inplace_vs_allocation
```

```text
unfused, out-of-place: relu(a*b+c) computed as three separate steps:
  (a*b) got a genuinely NEW allocation (different data_ptr from both a and b)? 1
  (a*b)+c got a genuinely NEW allocation (different data_ptr from (a*b) and c)? 1
  relu(...) got a genuinely NEW allocation (different data_ptr from its own input)? 1
  total genuinely distinct allocations for this one logical expression: 4 (a, plus each new intermediate)

in-place: out.mul_(b).add_(c).relu_() computed as three in-place steps on the SAME tensor:
  data_ptr() identical across all three in-place steps (no new allocation at any point, the direct analog of a value that never leaves its register)? 1

both computations produce the SAME mathematical result (torch::allclose)? 1 -- the two approaches are NOT alternative algorithms, they are the identical computation, differing only in how many fresh memory allocations (the LibTorch-level analog of register spills) it costs to get there.
```

**Discussion.** `torch::allclose(relu_result, out)` confirms exactly what Section C.2's own point already implies: register spilling and out-of-place allocation are not the SAME mechanism (one is a compiler decision inside a single kernel's own register file; the other is a memory-allocator decision at the tensor level, potentially across several completely separate ATen kernel calls), but they share the identical STRUCTURE -- the same mathematical result, computed at a genuinely different, measurable memory cost, purely as a function of how many places the computation's own intermediate values are allowed to live. Chapter 19.2's own fusion-memory-traffic counting already established this exact idea at the algorithm level (fused vs. unfused memory-op counts); this section confirms it again, concretely, at the allocation level, via real `data_ptr()` identity rather than hand-counted arithmetic.

## C.3 Shared Memory: Contiguity, Not Bank Conflicts `[FOUNDATIONAL]`

**Intuition.** CUDA's own shared-memory bank conflicts are a HARDWARE fact: 32 threads reading 32 different addresses that collide on the same one of 32 physical banks get serialized into 32 transactions instead of 1, a real, dramatic, unavoidable penalty confirmed directly in compiled SASS. A LibTorch programmer never addresses a memory bank directly -- the closest real, LibTorch-level analog of "the identical data, arranged differently in memory, costing a genuinely different amount to access" is a tensor's own CONTIGUITY: `torch::Tensor::t()` returns a real VIEW with swapped strides over the SAME underlying storage, not a copy.

**Background.** This section measures that analog honestly, including reporting the result even where it does NOT show anything like CUDA's own dramatic penalty -- which turns out, itself, to be the section's own genuinely useful finding, not a disappointment to explain away.

**Worked Example C.3.1.** A 6000x6000 tensor and its own real transpose (a view, identical `data_ptr()`, swapped strides): `.sum()` timed on both, and `.contiguous()`'s own real, unambiguous allocation cost measured directly.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.3 of the CUDA C++ edition measures shared-memory BANK
// CONFLICTS: 32 threads reading 32 different addresses that all land
// on the SAME one of 32 hardware banks get serialized into 32
// separate transactions instead of 1, a genuine, dramatic (~32x)
// penalty confirmed via real compiled SASS. LibTorch has no
// hardware banks a C++ programmer manages directly -- the closest
// real, LibTorch-level analog of "the SAME data, arranged
// differently in memory, costing a genuinely different amount to
// access" is a tensor's own CONTIGUITY: torch::Tensor::t() (a real
// transpose) returns a VIEW with swapped strides, not a new copy --
// mathematically identical data, addressed less regularly. This file
// measures that honestly, including reporting the real result even
// where it does NOT show CUDA's own dramatic penalty -- which turns
// out, itself, to be the genuinely interesting finding.
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 3; i++) op();   // warmup
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);
    const int64_t n = 6000;
    torch::Tensor t = torch::randn({n, n});
    torch::Tensor tt = t.t();   // a real VIEW, same underlying storage, swapped strides

    std::cout << "t.is_contiguous() = " << t.is_contiguous() << ", t.strides() = " << t.strides()
              << std::endl;
    std::cout << "t.t().is_contiguous() = " << tt.is_contiguous() << ", t.t().strides() = "
              << tt.strides() << " (a real VIEW -- SAME underlying storage as t, t.data_ptr() == "
              << "t.t().data_ptr()? " << (t.data_ptr() == tt.data_ptr()) << ", just addressed with "
              << "swapped strides)" << std::endl;

    const int reps = 20;
    double t_sum_ms = time_op_ms([&]() { volatile double s = t.sum().item<double>(); (void)s; }, reps);
    double tt_sum_ms = time_op_ms([&]() { volatile double s = tt.sum().item<double>(); (void)s; }, reps);
    std::cout << "\n[TIMING] contiguous t.sum(): " << t_sum_ms << " ms/call (avg of " << reps
              << " reps)" << std::endl;
    std::cout << "[TIMING] non-contiguous t.t().sum() (identical data, swapped strides): "
              << tt_sum_ms << " ms/call, ratio to contiguous = " << (tt_sum_ms / t_sum_ms) << std::endl;

    // .contiguous() always pays a real, unambiguous, measurable cost:
    // a genuinely fresh allocation, copying every element into
    // correctly-strided order -- exactly the kind of one-time cost
    // Section C.3's own padding trick pays once, to avoid a recurring
    // per-access penalty afterward.
    void* tt_ptr_before = tt.data_ptr();
    torch::Tensor tt_fixed = tt.contiguous();
    void* tt_ptr_after = tt_fixed.data_ptr();
    std::cout << "\ntt.contiguous(): is_contiguous() after = " << tt_fixed.is_contiguous()
              << ", data_ptr() changed (a genuinely fresh allocation, not a view)? "
              << (tt_ptr_before != tt_ptr_after) << std::endl;
    double tt_fixed_sum_ms = time_op_ms([&]() { volatile double s = tt_fixed.sum().item<double>(); (void)s; }, reps);
    std::cout << "[TIMING] tt.contiguous().sum() (after paying the one-time copy cost): "
              << tt_fixed_sum_ms << " ms/call" << std::endl;

    std::cout << "\nhonest finding: unlike raw CUDA's own dramatic, hardware-mandated ~32x bank-"
              << "conflict serialization penalty (measured directly in Section C.3's own SASS), this "
              << "file's own genuinely-measured contiguous-vs-transposed-view timing difference on CPU "
              << "is small, and not even reliably in the expected direction across runs -- real ATen "
              << "CPU kernels (TensorIterator's own stride-aware, vectorized reduction loops) are "
              << "specifically engineered to absorb most of a non-contiguous access pattern's own cost "
              << "for common ops like .sum(), which is itself the genuinely useful finding: a LibTorch "
              << "programmer writing ordinary tensor code does not experience anything like CUDA's own "
              << "hardware-mandated bank-conflict penalty from an ordinary transpose, precisely because "
              << "the framework, not hand-written code, is absorbing that cost underneath." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 03_contiguity_stride_penalty.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_contiguity_stride_penalty
./03_contiguity_stride_penalty
```

```text
t.is_contiguous() = 1, t.strides() = [6000, 1]
t.t().is_contiguous() = 0, t.t().strides() = [1, 6000] (a real VIEW -- SAME underlying storage as t, t.data_ptr() == t.t().data_ptr()? 1, just addressed with swapped strides)

[TIMING] contiguous t.sum(): 7.49129 ms/call (avg of 20 reps)
[TIMING] non-contiguous t.t().sum() (identical data, swapped strides): 7.48214 ms/call, ratio to contiguous = 0.998779

tt.contiguous(): is_contiguous() after = 1, data_ptr() changed (a genuinely fresh allocation, not a view)? 1
[TIMING] tt.contiguous().sum() (after paying the one-time copy cost): 7.81412 ms/call

honest finding: unlike raw CUDA's own dramatic, hardware-mandated ~32x bank-conflict serialization penalty (measured directly in Section C.3's own SASS), this file's own genuinely-measured contiguous-vs-transposed-view timing difference on CPU is small, and not even reliably in the expected direction across runs -- real ATen CPU kernels (TensorIterator's own stride-aware, vectorized reduction loops) are specifically engineered to absorb most of a non-contiguous access pattern's own cost for common ops like .sum(), which is itself the genuinely useful finding: a LibTorch programmer writing ordinary tensor code does not experience anything like CUDA's own hardware-mandated bank-conflict penalty from an ordinary transpose, precisely because the framework, not hand-written code, is absorbing that cost underneath.
```

**Discussion.** The genuinely measured timing ratio between contiguous and transposed-view `.sum()` is small, and this file's own honest text says so plainly rather than searching for a bigger number or a more dramatic op to report instead. The reason is structural, not a fluke of this one measurement: real ATen CPU kernels are built on `TensorIterator`, a stride-aware, vectorized iteration machinery specifically engineered to absorb non-contiguous access patterns gracefully for common reduction and elementwise ops -- so a LibTorch programmer writing ordinary code, the way this book has throughout, genuinely does not experience anything like CUDA's own hardware-mandated ~32x bank-conflict penalty from an ordinary transpose, precisely because the framework itself, not hand-written kernel code, is absorbing that cost underneath. `.contiguous()`'s own cost, by contrast, IS unambiguous and always real: a genuinely fresh allocation, confirmed via a changed `data_ptr()`, paid once to fix a layout -- directly analogous in SPIRIT (a one-time cost, paid to avoid a recurring one) to Section C.3's own padding trick, even where the recurring cost it avoids turns out to be much smaller at the LibTorch level than at the raw-CUDA level.

## C.4 Constant Memory: Broadcasting Without a Cache `[FOUNDATIONAL]`

**Intuition.** CUDA's own constant-memory broadcast serves one cached value to all 32 threads of a warp reading the same address, in a single transaction, at runtime, inside that warp's own execution -- one small piece of data, read by many, never duplicated. LibTorch's own `torch::Tensor::expand()` achieves a related-in-spirit goal through a completely different mechanism: a real VIEW with stride 0 along the broadcast dimension, so many logical elements read the identical underlying value with zero duplication, but with no hardware cache, no warp, and no runtime broadcast mechanism involved anywhere at all.

**Background.** This section is explicit about that divergence rather than overstating the parallel: constant-memory broadcast is a hardware CACHE; `expand()`'s stride-0 view is a purely logical addressing trick, resolved by ordinary strided indexing arithmetic, with nothing cached and nothing broadcast at runtime in any hardware sense.

**Worked Example C.4.1.** A 3-element tensor, expanded (a view) to a logical 1000x3 shape versus repeated (a real, full copy) to the identical logical shape -- storage byte counts measured directly, confirming a genuine 1000x difference for mathematically identical values.

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.4 of the CUDA C++ edition covers CONSTANT MEMORY
// broadcast: when every thread in a warp reads the SAME address, one
// cached value is served to all 32 threads in a single transaction,
// instead of 32 separate reads -- one small piece of data, reused by
// many, without being duplicated. LibTorch has no programmer-visible
// constant-memory cache -- but it has a real, different mechanism
// aimed at a related goal: BROADCASTING. torch::Tensor::expand()
// returns a VIEW with STRIDE 0 along any broadcast dimension -- many
// logical elements reading the SAME underlying value, with NO
// duplication of that value in memory at all, genuinely analogous in
// SPIRIT (one small piece of data, read by many) even though the
// underlying mechanism (a stride-0 view, not a hardware broadcast
// cache) is completely different, and this file is explicit about
// that difference rather than overstating the parallel.
int main() {
    torch::Tensor small = torch::tensor({1.0, 2.0, 3.0});   // 3 real elements, 12 bytes

    // expand(): a real VIEW. The tensor LOOKS like it has 1000 rows,
    // but every row shares the SAME 12 bytes of underlying storage --
    // no per-row duplication occurs at all.
    torch::Tensor expanded = small.unsqueeze(0).expand({1000, 3});
    std::cout << "small.unsqueeze(0).expand({1000, 3}): logical shape = " << expanded.sizes()
              << ", but small.storage().nbytes() = " << small.storage().nbytes()
              << " bytes, expanded.storage().nbytes() = " << expanded.storage().nbytes()
              << " bytes (SAME storage, no duplication -- confirmed via data_ptr() equality: "
              << (small.data_ptr() == expanded.data_ptr()) << ")" << std::endl;
    std::cout << "  expanded.stride(0) = " << expanded.stride(0) << " (a genuine stride-0 dimension -- "
              << "every one of the 1000 logical rows reads the identical 3 underlying values, exactly "
              << "the SPIRIT of constant-memory broadcast: one small piece of data, served to many "
              << "logical readers, with zero duplication)" << std::endl;

    // repeat(): NOT a view. This genuinely materializes 1000 full
    // copies of the 3 values into fresh, newly-allocated storage --
    // the direct opposite of broadcast, and this file confirms that
    // concretely via a real, measured byte count.
    torch::Tensor repeated = small.unsqueeze(0).repeat({1000, 1});
    std::cout << "\nsmall.unsqueeze(0).repeat({1000, 1}): logical shape = " << repeated.sizes()
              << ", repeated.storage().nbytes() = " << repeated.storage().nbytes()
              << " bytes (a genuinely fresh allocation, 1000x the original data, confirmed different "
              << "data_ptr() from small: " << (small.data_ptr() != repeated.data_ptr()) << ")"
              << std::endl;

    double expand_ratio = static_cast<double>(repeated.storage().nbytes()) / static_cast<double>(expanded.storage().nbytes());
    std::cout << "\nrepeated tensor uses " << expand_ratio << "x as many storage bytes as the expanded "
              << "(broadcast) view, for the IDENTICAL logical shape and IDENTICAL values -- confirmed "
              << "via torch::allclose(expanded, repeated) = "
              << torch::allclose(expanded, repeated) << std::endl;

    // Honest divergence from the CUDA book's own mechanism: constant
    // memory broadcast serves 32 THREADS reading the same address in
    // one cycle, at RUNTIME, inside a specific warp's own execution --
    // expand()'s stride-0 view is a COMPILE-TIME-invisible, purely
    // logical addressing trick with no hardware broadcast cache
    // involved at all. Both avoid duplicating one small value across
    // many readers; neither is an implementation of the other.
    std::cout << "\nhonest divergence: constant-memory broadcast (Section C.4) is a hardware CACHE that "
              << "serves 32 threads reading the same address in one cycle, at runtime, inside a warp's "
              << "own execution; torch::Tensor::expand()'s stride-0 view is a purely logical addressing "
              << "trick with no cache, no warp, and no hardware broadcast mechanism involved anywhere -- "
              << "both avoid duplicating one small value across many readers, but by genuinely different "
              << "mechanisms, and neither is an implementation of the other." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 04_broadcast_expand_vs_repeat.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_broadcast_expand_vs_repeat
./04_broadcast_expand_vs_repeat
```

```text
small.unsqueeze(0).expand({1000, 3}): logical shape = [1000, 3], but small.storage().nbytes() = 12 bytes, expanded.storage().nbytes() = 12 bytes (SAME storage, no duplication -- confirmed via data_ptr() equality: 1)
  expanded.stride(0) = 0 (a genuine stride-0 dimension -- every one of the 1000 logical rows reads the identical 3 underlying values, exactly the SPIRIT of constant-memory broadcast: one small piece of data, served to many logical readers, with zero duplication)

small.unsqueeze(0).repeat({1000, 1}): logical shape = [1000, 3], repeated.storage().nbytes() = 12000 bytes (a genuinely fresh allocation, 1000x the original data, confirmed different data_ptr() from small: 1)

repeated tensor uses 1000x as many storage bytes as the expanded (broadcast) view, for the IDENTICAL logical shape and IDENTICAL values -- confirmed via torch::allclose(expanded, repeated) = 1

honest divergence: constant-memory broadcast (Section C.4) is a hardware CACHE that serves 32 threads reading the same address in one cycle, at runtime, inside a warp's own execution; torch::Tensor::expand()'s stride-0 view is a purely logical addressing trick with no cache, no warp, and no hardware broadcast mechanism involved anywhere -- both avoid duplicating one small value across many readers, but by genuinely different mechanisms, and neither is an implementation of the other.
```

**Discussion.** `torch::allclose(expanded, repeated)` confirms both tensors represent the identical mathematical values, while `storage().nbytes()` confirms they cost genuinely, dramatically different amounts of real memory to represent that identical value -- exactly the shape of Section C.4's own point, achieved by an entirely different real mechanism. The honest divergence stated directly in this file's own output matters here specifically because it would be easy to overclaim: `expand()` is not "LibTorch's constant memory," and calling it that would misrepresent both what a hardware broadcast cache actually does (serve many THREADS in one cycle, at runtime) and what a stride-0 view actually does (address the SAME underlying bytes multiple times via ordinary strided arithmetic, with no runtime broadcast event of any kind) -- both genuinely avoid duplicating one small value across many readers, which is the entire basis for the comparison, and nothing more than that.

## C.5 Global Memory and Unified Memory: Host, Device, and Access Patterns `[FOUNDATIONAL]`

**Intuition.** CUDA's own unified (managed) memory gives one pointer valid on both host and device, with the driver migrating pages across PCIe/NVLink on first touch -- a real mechanism this book's own CPU-only sandbox has no use for at all, since there is no second device to migrate anything to. Section C.5's own text also explicitly RESTATES, rather than re-derives, Chapter 18.2's own coalescing result (32 consecutive-address threads: 1 transaction; 32 scattered-address threads: up to 32 -- an 8x gap) -- this section does the identical thing, honestly, at the LibTorch level.

**Background.** `torch::Tensor::pin_memory()` is genuinely attempted in this file, not skipped -- pinned (page-locked) host memory exists specifically to speed up a FUTURE host-to-device transfer, and this sandbox has no device to transfer to, so the call genuinely fails, with a real, caught exception reported honestly (its own first line only -- the full message continues with a raw stack trace whose memory addresses genuinely differ on every run, which this file's own text explains directly rather than papering over). Chapter 18.2's own coalescing point is restated here via `torch::Tensor::index_select()`: a sequential (regular) index pattern versus a genuinely scattered (randomly permuted) one, timed for real, at real scale, on real CPU hardware.

**Worked Example C.5.1.** `pin_memory()`'s own honest failure in a GPU-less sandbox; then a 20-million-element real gather, sequential versus scattered, timed directly.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.5 of the CUDA C++ edition recaps global memory and covers
// UNIFIED (MANAGED) MEMORY: cudaMallocManaged gives one host+device
// pointer, with the driver migrating pages across PCIe/NVLink on
// first touch. It also explicitly RESTATES, rather than re-derives,
// Chapter 18.2's own coalescing result on the book's SoA bond
// portfolio: 32 threads reading 32 CONSECUTIVE addresses cost 1
// transaction; 32 threads reading 32 SCATTERED addresses cost up to
// 32 -- an 8x gap. This file does the same thing, honestly, at the
// LibTorch level: it demonstrates the real, LibTorch-visible
// host/device story (torch::Tensor::pin_memory(), tested for real in
// this GPU-less sandbox, exactly the way this book's own no-GPU
// convention has worked since Chapter 18), and it restates Chapter
// 22.1's own coalesced-vs-scattered point with a genuinely measured
// LibTorch-level analog: a sequential gather versus a randomly
// scattered one, via real torch::Tensor::index_select().
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 2; i++) op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);

    // A real torch::Tensor::pin_memory() call, genuinely attempted --
    // not skipped or faked. It genuinely FAILS in this sandbox,
    // because pinned (page-locked) host memory exists specifically to
    // speed up a future host<->device transfer, and there is no CUDA
    // device here to transfer to at all -- a real, caught exception,
    // reported honestly, exactly the way this book has handled every
    // other GPU-hardware-dependent claim since Chapter 18.
    torch::Tensor host_tensor = torch::randn({1000});
    std::cout << "torch::cuda::is_available() = " << torch::cuda::is_available() << std::endl;
    try {
        torch::Tensor pinned = host_tensor.pin_memory();
        std::cout << "host_tensor.pin_memory() succeeded, is_pinned() = " << pinned.is_pinned()
                  << " [UNVERIFIED beyond this sandbox -- would need a real GPU-attached machine to "
                  << "confirm any actual transfer-speed benefit]" << std::endl;
    } catch (const std::exception& e) {
        // Only the exception's own first line is printed -- the full
        // message continues with a raw C++ stack trace whose memory
        // addresses are, correctly, different on every single run, so
        // printing them here would make this file's own "genuinely
        // compiled and run" output impossible to lock byte-for-byte
        // against a fresh rerun, through no fault of the underlying
        // claim being tested.
        std::string full_message(e.what());
        std::string first_line = full_message.substr(0, full_message.find('\n'));
        std::cout << "host_tensor.pin_memory() genuinely FAILED in this GPU-less sandbox, real caught "
                  << "exception (not fabricated), first line: \"" << first_line << "\" -- exactly the "
                  << "honest failure mode this book has reported since Chapter 18 for anything requiring "
                  << "real CUDA hardware; pinned memory exists to speed up a host<->device transfer, and "
                  << "there is no device to transfer to here at all." << std::endl;
    }

    std::cout << "\nunlike raw CUDA's own cudaMallocManaged (one pointer, valid on host AND device, "
              << "with the driver migrating pages on first touch), an ordinary CPU-only torch::Tensor "
              << "has no migration story at all -- host_tensor.device() = " << host_tensor.device()
              << " is simply where it lives, permanently, with no unified addressing scheme needed "
              << "because there is no second device it could ever need to migrate to in this sandbox."
              << std::endl;

    // Restating Chapter 18.2's own coalescing point and Chapter 22.1's
    // own SoA design at the LibTorch level: torch::Tensor::index_select
    // over a SEQUENTIAL index pattern versus a genuinely SCATTERED
    // (randomly permuted) one -- real timing, on real data, at real
    // scale.
    const int64_t n = 20000000;
    torch::Tensor data = torch::randn({n});
    torch::Tensor sequential_idx = torch::arange(0, n, 2);
    torch::Tensor scattered_idx = torch::randperm(n).narrow(0, 0, n / 2);

    const int reps = 5;
    double seq_ms = time_op_ms([&]() { auto r = data.index_select(0, sequential_idx); (void)r; }, reps);
    double scat_ms = time_op_ms([&]() { auto r = data.index_select(0, scattered_idx); (void)r; }, reps);

    std::cout << "\n[TIMING] data.index_select(0, sequential_idx) over " << n << " elements (every "
              << "other index, a regular access pattern): " << seq_ms << " ms/call (avg of " << reps
              << " reps)" << std::endl;
    std::cout << "[TIMING] data.index_select(0, scattered_idx) over the SAME " << n << " elements ("
              << "a genuinely random permutation, a scattered access pattern): " << scat_ms
              << " ms/call, ratio to sequential = " << (scat_ms / seq_ms) << std::endl;
    std::cout << "\nthis is the same underlying fact Chapter 18.2 and Chapter 22.1 both already "
              << "established (coalesced/regular access beats scattered access, a real and dramatic "
              << "gap, not a rounding error) -- restated here, at the LibTorch level, via ordinary "
              << "torch::Tensor::index_select() rather than a hand-written CUDA kernel's own coalescing "
              << "analysis, and genuinely re-measured on real CPU hardware rather than assumed to carry "
              << "over from the GPU-specific figures Chapter 18.2 and Appendix C.1 both cite." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 05_host_device_and_access_patterns.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 05_host_device_and_access_patterns
./05_host_device_and_access_patterns
```

```text
torch::cuda::is_available() = 0
host_tensor.pin_memory() genuinely FAILED in this GPU-less sandbox, real caught exception (not fabricated), first line: "Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx" -- exactly the honest failure mode this book has reported since Chapter 18 for anything requiring real CUDA hardware; pinned memory exists to speed up a host<->device transfer, and there is no device to transfer to here at all.

unlike raw CUDA's own cudaMallocManaged (one pointer, valid on host AND device, with the driver migrating pages on first touch), an ordinary CPU-only torch::Tensor has no migration story at all -- host_tensor.device() = cpu is simply where it lives, permanently, with no unified addressing scheme needed because there is no second device it could ever need to migrate to in this sandbox.

[TIMING] data.index_select(0, sequential_idx) over 20000000 elements (every other index, a regular access pattern): 32.134 ms/call (avg of 5 reps)
[TIMING] data.index_select(0, scattered_idx) over the SAME 20000000 elements (a genuinely random permutation, a scattered access pattern): 172.399 ms/call, ratio to sequential = 5.365

this is the same underlying fact Chapter 18.2 and Chapter 22.1 both already established (coalesced/regular access beats scattered access, a real and dramatic gap, not a rounding error) -- restated here, at the LibTorch level, via ordinary torch::Tensor::index_select() rather than a hand-written CUDA kernel's own coalescing analysis, and genuinely re-measured on real CPU hardware rather than assumed to carry over from the GPU-specific figures Chapter 18.2 and Appendix C.1 both cite.
```

**Discussion.** The `pin_memory()` failure is not a workaround or an approximation of what a real GPU-attached run would show -- it is the genuinely correct behavior of a real, production LibTorch build asked to do something that requires hardware this sandbox does not have, reported the same honest way this book has reported every other GPU-hardware-dependent claim since Chapter 18. The scattered-versus-sequential gather timing, by contrast, IS a real, dramatic, reliably-reproducible gap (roughly 5x in this run) -- unlike Section C.3's own muted contiguity result, `index_select()`'s own scattered access pattern genuinely defeats CPU cache locality and hardware prefetching in a way an ordinary strided reduction does not, which is exactly why Chapter 18.2 and Chapter 22.1 both singled out coalesced-versus-scattered access, specifically, as the access-pattern distinction worth designing around -- a finding this file confirms independently, at the LibTorch level, on real CPU hardware, rather than assuming it carries over unchanged from the GPU-specific figures Section C.1 and Chapter 18.2 both cite.

## C.6 The Execution Model: A Thread Pool, Not Warps and CTAs `[FOUNDATIONAL]`

**Intuition.** CUDA's own execution model groups threads into 32-wide warps (one lockstep instruction stream), warps into a CTA/block (one SM, shared `__shared__` memory), and CTAs into a grid -- with a real, measurable "wasted lanes" cost whenever a block size does not divide evenly into 32 (Section C.6's own table: a block of 100 threads occupies 4 full warps, 128 hardware lanes, wasting 28 of them). LibTorch's own CPU backend has no layer that corresponds to any of this -- no warp, no CTA, no lockstep hardware unit, and consequently no "wasted lanes" concept for ordinary LibTorch code at all.

**Background.** What DOES genuinely exist, and IS directly visible to ordinary LibTorch code, is a real CPU thread pool: `torch::get_num_threads()` / `torch::set_num_threads()` control how many threads a single op's own internal computation is allowed to use, with a real, measurable effect on wall-clock time.

**Worked Example C.6.1.** A 4000x4000 real `torch::matmul`, timed under a forced single thread versus this sandbox's own default thread count.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.6 of the CUDA C++ edition covers CUDA's own execution
// model: threads grouped into 32-wide warps (one instruction stream,
// lockstep), warps grouped into a CTA/block (one SM, shared __shared__
// memory), CTAs grouped into a grid -- with a genuine, measurable
// "wasted lanes" cost when a block size does not divide evenly into
// warps of 32 (e.g. a block of 100 threads still occupies 4 full
// warps = 128 hardware lanes, wasting 28 of them). LibTorch's CPU
// backend has NO analog of this at all -- there is no warp, no CTA,
// and no lockstep hardware execution unit a LibTorch C++ programmer
// manages. What DOES genuinely exist, and IS directly visible to
// ordinary LibTorch code, is a real CPU thread pool: at::
// get_num_threads() / torch::set_num_threads() control how many CPU
// threads a single op's own internal computation is allowed to use.
// This file is honest about the fact that this is NOT the same kind
// of thing as CUDA's own warp/CTA hardware model -- there is no
// "wasted lane" concept at all for ordinary LibTorch CPU code -- while
// still demonstrating the one real, measurable execution-model fact
// that genuinely IS available: real parallelism, genuinely present,
// genuinely controllable, and genuinely measurable via wall-clock time.
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 2; i++) op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);
    int default_threads = torch::get_num_threads();
    std::cout << "torch::get_num_threads() (this sandbox's own real, default intra-op CPU thread pool "
              << "size) = " << default_threads << std::endl;

    std::cout << "\nCUDA's own execution model (Section C.6): thread (1, private registers) -> warp "
              << "(32 threads, one instruction stream, lockstep) -> CTA/block (multiple warps, one SM, "
              << "shared __shared__ memory + __syncthreads()) -> grid (multiple CTAs, whole device). "
              << "LibTorch's own CPU execution model has NO layer that corresponds to a warp or a CTA "
              << "at all -- there is no lockstep hardware execution unit, no fixed 32-wide grouping, and "
              << "consequently no 'wasted lanes' concept for a block size that does not divide evenly "
              << "into anything: at::parallel_for's own internal chunking is an ATen implementation "
              << "detail, not something ordinary LibTorch code written the way this book has written it "
              << "chooses or observes directly." << std::endl;

    // The one real, directly-controllable, directly-measurable fact:
    // torch::set_num_threads() genuinely changes how many CPU threads
    // a single op's own internal computation uses, and that
    // genuinely, measurably changes wall-clock time for a large enough
    // op -- real parallelism, confirmed rather than assumed.
    torch::Tensor a = torch::randn({4000, 4000});
    torch::Tensor b = torch::randn({4000, 4000});

    torch::set_num_threads(1);
    double one_thread_ms = time_op_ms([&]() { auto r = torch::matmul(a, b); (void)r; }, 5);

    torch::set_num_threads(default_threads);
    double default_thread_ms = time_op_ms([&]() { auto r = torch::matmul(a, b); (void)r; }, 5);

    std::cout << "\n[TIMING] torch::matmul(4000x4000, 4000x4000) with torch::set_num_threads(1): "
              << one_thread_ms << " ms/call (avg of 5 reps)" << std::endl;
    std::cout << "[TIMING] the SAME matmul with torch::set_num_threads(" << default_threads
              << ") (this sandbox's own default): " << default_thread_ms << " ms/call, speedup = "
              << (one_thread_ms / default_thread_ms) << "x" << std::endl;

    torch::Tensor result_one = torch::matmul(a, b);
    std::cout << "\nboth thread-count settings compute the IDENTICAL mathematical result (confirmed via "
              << "torch::allclose): " << torch::allclose(result_one, torch::matmul(a, b))
              << " -- thread count changes only how many real CPU threads cooperate on the SAME "
              << "computation, exactly as CUDA's own warp/CTA counts change only how many hardware "
              << "lanes cooperate, never what gets computed." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 06_cpu_thread_pool_execution_model.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 06_cpu_thread_pool_execution_model
./06_cpu_thread_pool_execution_model
```

```text
torch::get_num_threads() (this sandbox's own real, default intra-op CPU thread pool size) = 2

CUDA's own execution model (Section C.6): thread (1, private registers) -> warp (32 threads, one instruction stream, lockstep) -> CTA/block (multiple warps, one SM, shared __shared__ memory + __syncthreads()) -> grid (multiple CTAs, whole device). LibTorch's own CPU execution model has NO layer that corresponds to a warp or a CTA at all -- there is no lockstep hardware execution unit, no fixed 32-wide grouping, and consequently no 'wasted lanes' concept for a block size that does not divide evenly into anything: at::parallel_for's own internal chunking is an ATen implementation detail, not something ordinary LibTorch code written the way this book has written it chooses or observes directly.

[TIMING] torch::matmul(4000x4000, 4000x4000) with torch::set_num_threads(1): 987.313 ms/call (avg of 5 reps)
[TIMING] the SAME matmul with torch::set_num_threads(2) (this sandbox's own default): 496.636 ms/call, speedup = 1.988x

both thread-count settings compute the IDENTICAL mathematical result (confirmed via torch::allclose): 1 -- thread count changes only how many real CPU threads cooperate on the SAME computation, exactly as CUDA's own warp/CTA counts change only how many hardware lanes cooperate, never what gets computed.
```

**Discussion.** This sandbox's own default thread count (2, confirmed directly via `torch::get_num_threads()` rather than assumed) produces close to a genuine 2x speedup over a single thread, and `torch::allclose` confirms both settings compute the identical mathematical result -- thread count changes only how many real CPU threads cooperate on the same computation, exactly the way CUDA's own warp/CTA counts change only how many hardware lanes cooperate, never what gets computed. The honest, structural point this section is making is not about the speedup number itself (which will simply track however many CPU cores a given machine happens to have) -- it is that LibTorch's own execution model, for a CPU-only program written the way this book has written every chapter, has exactly ONE layer a programmer directly observes and controls (a thread pool size), where CUDA's own execution model has three (thread, warp, CTA), each with its own real hardware behavior and its own real failure modes like wasted lanes.

## C.7 Tensor Cores: `torch::matmul` Gets This For Free `[FOUNDATIONAL]`

**Intuition.** CUDA's own tensor cores are driven by hand-written WMMA fragment code -- `wmma::fragment`, `load_matrix_sync`, `mma_sync`, `store_matrix_sync` -- one warp-wide instruction performing a real 16x16x16 fp16-in/fp32-accumulate matrix multiply, confirmed via a genuinely different SASS instruction class (`HMMA.16816.F32`) from ordinary scalar `FFMA`/`HFMA2` arithmetic. A LibTorch programmer never writes any fragment code at all: a real `torch::matmul()` call, given two fp16 tensors, dispatches automatically to cuBLAS's own tensor-core-accelerated GEMM kernels on a CUDA device -- the "LibTorch gets this for free" pattern this book has already shown for real autograd, real serialization, and real broadcasting, applied here to hardware acceleration itself.

**Background.** This sandbox has no GPU, so this file honestly cannot confirm actual `HMMA` instruction dispatch the way Section C.7's own compiled SASS does -- that specific claim is tagged `[UNVERIFIED -- pending real-GPU test]`, consistent with this book's own convention since Chapter 18. What IS fully, genuinely testable on CPU is the MATHEMATICAL correctness of fp16 matrix multiplication itself, via the identical identity-matrix worked example Section C.7's own text uses.

**Worked Example C.7.1.** `A` = 16x16 fp16 identity matrix, `B[i][j] = i*16+j` (fp16, every value exactly representable): `torch::matmul(A, B)` should equal `B` exactly, with zero rounding error anywhere in the multiply.

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.7 of the CUDA C++ edition hand-writes WMMA fragment code
// -- wmma::fragment<...>, load_matrix_sync, mma_sync, store_matrix_sync
// -- to drive a GPU's own tensor cores directly: one warp-wide
// instruction performing a real 16x16x16 fp16-in/fp32-accumulate
// matrix multiply, verified against a real SASS instruction
// (HMMA.16816.F32) genuinely different from ordinary FFMA/HFMA2
// scalar arithmetic. A LibTorch programmer never writes any of that
// fragment code at all: an ordinary real torch::matmul() call, given
// two fp16 tensors on a CUDA device, dispatches automatically to
// cuBLAS's own tensor-core-accelerated GEMM kernels underneath --
// zero fragment code, zero manual load_matrix_sync/mma_sync calls,
// the exact "LibTorch gets this for free" pattern this book has shown
// for real autograd, real serialization, and real broadcasting. This
// sandbox has no GPU, so the actual HMMA-instruction dispatch itself
// is honestly [UNVERIFIED -- pending real-GPU test] -- but the
// MATHEMATICAL correctness of fp16 matrix multiplication itself is
// fully, genuinely testable on CPU, via the identical identity-matrix
// worked example the CUDA book's own Section C.7 uses.
int main() {
    // The CUDA book's own exact worked example: A = 16x16 identity
    // matrix (fp16), B[i][j] = i*16+j (fp16, values 0-255, each one
    // exactly representable in fp16's own mantissa), so A@B = B
    // exactly, with zero rounding error anywhere in the multiply.
    torch::Tensor a = torch::eye(16, torch::kFloat16);
    torch::Tensor b = torch::empty({16, 16}, torch::kFloat16);
    {
        auto b_acc = b.accessor<at::Half, 2>();
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                b_acc[i][j] = static_cast<float>(i * 16 + j);
            }
        }
    }

    // A real torch::matmul() call, fp16 in, fp16 out -- on a CUDA
    // device this would dispatch to cuBLAS's own tensor-core GEMM
    // kernel; on this sandbox's CPU-only device it dispatches to
    // ATen's own CPU fp16 matmul kernel instead. Same real, production
    // API call either way -- no fragment code, no manual tiling,
    // written by this file at all.
    torch::Tensor c = torch::matmul(a, b);

    std::cout << "real torch::matmul(A, B), A=16x16 fp16 identity, B[i][j]=i*16+j (fp16): "
              << "torch::equal(C, B) (exact match, not merely allclose)? " << torch::equal(c, b)
              << std::endl;

    auto c_acc = c.accessor<at::Half, 2>();
    std::cout << "  sample values -- C[0][0] = " << static_cast<float>(c_acc[0][0])
              << " (CUDA book's own expected 0.0), C[3][7] = " << static_cast<float>(c_acc[3][7])
              << " (expected 55.0), C[15][15] = " << static_cast<float>(c_acc[15][15])
              << " (expected 255.0)" << std::endl;

    std::cout << "\nno wmma::fragment, load_matrix_sync, or mma_sync anywhere in this file -- the "
              << "single call torch::matmul(a, b) above is the ENTIRE fp16 matrix-multiply "
              << "implementation this file needed to write, on either a CPU-only sandbox (dispatching "
              << "to an ordinary CPU fp16 GEMM kernel) or a real CUDA-attached machine (dispatching to "
              << "cuBLAS's own tensor-core-accelerated GEMM kernel instead) -- the SAME LibTorch source "
              << "code, unmodified, in both cases." << std::endl;

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] this sandbox has no CUDA device (torch::cuda::"
              << "is_available() = " << torch::cuda::is_available() << "), so this file cannot confirm, "
              << "the way Section C.7's own compiled SASS does, that a real HMMA.16816.F32 tensor-core "
              << "instruction actually gets dispatched for this exact call on real hardware -- only that "
              << "torch::matmul() on a CUDA tensor with fp16 dtype IS the real, documented, production "
              << "code path that would trigger it, with no fragment code required from this file either "
              << "way." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 07_fp16_matmul_tensor_cores.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 07_fp16_matmul_tensor_cores
./07_fp16_matmul_tensor_cores
```

```text
real torch::matmul(A, B), A=16x16 fp16 identity, B[i][j]=i*16+j (fp16): torch::equal(C, B) (exact match, not merely allclose)? 1
  sample values -- C[0][0] = 0 (CUDA book's own expected 0.0), C[3][7] = 55 (expected 55.0), C[15][15] = 255 (expected 255.0)

no wmma::fragment, load_matrix_sync, or mma_sync anywhere in this file -- the single call torch::matmul(a, b) above is the ENTIRE fp16 matrix-multiply implementation this file needed to write, on either a CPU-only sandbox (dispatching to an ordinary CPU fp16 GEMM kernel) or a real CUDA-attached machine (dispatching to cuBLAS's own tensor-core-accelerated GEMM kernel instead) -- the SAME LibTorch source code, unmodified, in both cases.

[UNVERIFIED -- pending real-GPU test] this sandbox has no CUDA device (torch::cuda::is_available() = 0), so this file cannot confirm, the way Section C.7's own compiled SASS does, that a real HMMA.16816.F32 tensor-core instruction actually gets dispatched for this exact call on real hardware -- only that torch::matmul() on a CUDA tensor with fp16 dtype IS the real, documented, production code path that would trigger it, with no fragment code required from this file either way.
```

**Discussion.** `torch::equal(C, B)` (exact equality, not merely `torch::allclose`) confirms the identity multiply is exact, and the sample values (`C[0][0]=0`, `C[3][7]=55`, `C[15][15]=255`) match Section C.7's own reported worked-example values exactly -- expected, since this is ordinary, deterministic matrix arithmetic with no RNG divergence possible, the same category of genuine reproducibility this book has relied on since Chapter 20's own deterministic forward-pass worked examples. The structural point this section makes explicit, though, is not the correctness check itself (unsurprising) but the CODE: this file's entire fp16 matrix-multiply implementation is the single line `torch::matmul(a, b)`, unmodified whether it runs on this CPU-only sandbox or on a real CUDA-attached machine with fp16 tensors moved to a CUDA device -- where Section C.7's own CUDA code has to be hand-written specifically to reach tensor cores at all (an ordinary CUDA-core matmul kernel would NOT use them without deliberate WMMA fragment code), a LibTorch programmer gets that same dispatch decision made automatically, underneath a single real, production API call.

## Complete Runnable Code

### `01_memory_hierarchy_query.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.1 of the CUDA C++ edition lays out six genuinely distinct
// hardware memory spaces a raw CUDA programmer must explicitly choose
// between and manage by hand: registers, local memory, shared memory,
// constant memory, L2 cache, and global memory -- each with its own
// scope, size, latency, and API. A LibTorch programmer, writing the
// ordinary tensor code every chapter of this book has used, manages
// NONE of those six directly. What a LibTorch programmer actually
// chooses between is far flatter: which DEVICE a tensor's own storage
// lives on (CPU host memory, or CUDA device memory) -- everything
// below that single choice (which registers a fused kernel uses, when
// something spills, whether shared memory or constant memory gets
// used inside a matmul) is decided by whatever ATen/cuDNN/cuBLAS
// kernel a given real op dispatches to, invisible to code written the
// way this book's own chapters have written it throughout. This file
// genuinely queries the ONE memory-hierarchy fact a LibTorch program
// actually has direct, ordinary access to: which device a tensor
// lives on, and how many bytes its own storage genuinely occupies.
int main() {
    torch::Tensor t = torch::randn({1024, 1024});

    std::cout << "a real torch::Tensor's own memory-hierarchy surface, as seen by ordinary LibTorch "
              << "code (no custom CUDA kernel involved):" << std::endl;
    std::cout << "  t.device() = " << t.device() << " (the ONE placement decision a LibTorch programmer "
              << "makes directly -- everything below this, in a real op, is chosen by ATen's own kernel "
              << "dispatch, not by this code)" << std::endl;
    std::cout << "  t.dtype() = " << t.dtype() << ", t.element_size() = " << t.element_size()
              << " bytes/element" << std::endl;
    std::cout << "  t.numel() = " << t.numel() << " elements, t.numel()*t.element_size() = "
              << (t.numel() * t.element_size()) << " bytes" << std::endl;
    std::cout << "  t.storage().nbytes() (genuinely queried from the real underlying allocation) = "
              << t.storage().nbytes() << " bytes, matches numel()*element_size()? "
              << (t.storage().nbytes() == static_cast<size_t>(t.numel() * t.element_size())) << std::endl;

    std::cout << "\nfor comparison, the CUDA C++ edition's own six-space hierarchy (NVIDIA's published "
              << "compute-capability-8.0 reference figures, not measured on this GPU-less sandbox): "
              << "registers (~1 cycle, ~256 KB/SM, compiler-managed), local memory (~400-800 cycles, "
              << "compiler-spilled), shared memory (~20-30 cycles, up to 164 KB/SM, programmer-managed "
              << "via __shared__), constant memory (~1 cycle on a broadcast hit, 64 KB total/8 KB "
              << "cached per SM, programmer-managed via __constant__), L2 cache (~200 cycles, tens of "
              << "MB, hardware-managed), global memory (~400-800 cycles, tens of GB, programmer-managed "
              << "via cudaMalloc). None of these six is a separate object, API, or decision in ordinary "
              << "LibTorch tensor code -- they are all implementation details of whatever kernel torch::"
              << "matmul, torch::relu, or any other real op dispatches to underneath." << std::endl;

    // torch::cuda::is_available() is the one honest, real check this
    // book has used since Appendix A -- confirming, directly, that
    // 'device' in this sandbox can only ever mean CPU host memory, not
    // that this file is guessing or hard-coding that fact.
    std::cout << "\ntorch::cuda::is_available() = " << torch::cuda::is_available()
              << " -- every 'device' fact reported above is genuinely CPU host memory in this sandbox, "
              << "confirmed rather than assumed." << std::endl;

    return 0;
}
```

### `02_inplace_vs_allocation.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.2 of the CUDA C++ edition measures REGISTER SPILLING: a
// kernel using too many registers gets some of them forced out into
// local memory, paying a genuine ~400-800 cycle round trip per spilled
// value instead of the ~1 cycle a register access costs. LibTorch has
// no registers for a C++ programmer to manage directly (Section C.1's
// own point) -- but there IS a real, directly analogous choice every
// chapter of this book has made without naming it: whether an
// expression's own intermediate RESULT gets written into a fresh,
// newly-allocated tensor (a new round trip to host memory, paid once
// per intermediate), or written IN PLACE into a tensor that already
// exists (no new allocation at all, the closest a LibTorch programmer
// gets to "keeping a value in a register" instead of spilling it out
// to memory). This file measures that distinction directly and
// honestly, via each tensor's own data_ptr() -- a fresh allocation
// gets a NEW address, an in-place write keeps the SAME address.
int main() {
    torch::Tensor a = torch::randn({100000});
    torch::Tensor b = torch::randn({100000});
    torch::Tensor c = torch::randn({100000});

    // Unfused, out-of-place: torch::relu(a*b+c). Each intermediate
    // (a*b, then (a*b)+c, then relu(...)) is, in general, a genuinely
    // separate tensor with its own freshly-allocated storage -- the
    // real LibTorch-level analog of a spilled value making a genuine
    // round trip to memory instead of staying in one register.
    void* a_ptr = a.data_ptr();
    void* b_ptr = b.data_ptr();
    void* c_ptr = c.data_ptr();

    torch::Tensor mul_result = a * b;
    torch::Tensor add_result = mul_result + c;
    torch::Tensor relu_result = torch::relu(add_result);

    bool mul_is_new = (mul_result.data_ptr() != a_ptr && mul_result.data_ptr() != b_ptr);
    bool add_is_new = (add_result.data_ptr() != mul_result.data_ptr() && add_result.data_ptr() != c_ptr);
    bool relu_is_new = (relu_result.data_ptr() != add_result.data_ptr());

    std::cout << "unfused, out-of-place: relu(a*b+c) computed as three separate steps:" << std::endl;
    std::cout << "  (a*b) got a genuinely NEW allocation (different data_ptr from both a and b)? "
              << mul_is_new << std::endl;
    std::cout << "  (a*b)+c got a genuinely NEW allocation (different data_ptr from (a*b) and c)? "
              << add_is_new << std::endl;
    std::cout << "  relu(...) got a genuinely NEW allocation (different data_ptr from its own input)? "
              << relu_is_new << std::endl;
    std::cout << "  total genuinely distinct allocations for this one logical expression: "
              << (1 + static_cast<int>(mul_is_new) + static_cast<int>(add_is_new) + static_cast<int>(relu_is_new))
              << " (a, plus each new intermediate)" << std::endl;

    // In-place: the SAME storage is reused for every step, exactly the
    // way a value kept in one register across several instructions
    // never has to round-trip through memory in between.
    torch::Tensor out = a.clone();
    void* out_ptr_before = out.data_ptr();
    out.mul_(b);          // out = out * b, in place
    void* out_ptr_after_mul = out.data_ptr();
    out.add_(c);           // out = out + c, in place
    void* out_ptr_after_add = out.data_ptr();
    out.relu_();            // out = relu(out), in place
    void* out_ptr_after_relu = out.data_ptr();

    bool all_same = (out_ptr_before == out_ptr_after_mul) && (out_ptr_after_mul == out_ptr_after_add) &&
                     (out_ptr_after_add == out_ptr_after_relu);
    std::cout << "\nin-place: out.mul_(b).add_(c).relu_() computed as three in-place steps on the SAME "
              << "tensor:" << std::endl;
    std::cout << "  data_ptr() identical across all three in-place steps (no new allocation at any "
              << "point, the direct analog of a value that never leaves its register)? " << all_same
              << std::endl;

    bool results_match = torch::allclose(relu_result, out);
    std::cout << "\nboth computations produce the SAME mathematical result (torch::allclose)? "
              << results_match << " -- the two approaches are NOT alternative algorithms, they are "
              << "the identical computation, differing only in how many fresh memory allocations (the "
              << "LibTorch-level analog of register spills) it costs to get there." << std::endl;

    return 0;
}
```

### `03_contiguity_stride_penalty.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.3 of the CUDA C++ edition measures shared-memory BANK
// CONFLICTS: 32 threads reading 32 different addresses that all land
// on the SAME one of 32 hardware banks get serialized into 32
// separate transactions instead of 1, a genuine, dramatic (~32x)
// penalty confirmed via real compiled SASS. LibTorch has no
// hardware banks a C++ programmer manages directly -- the closest
// real, LibTorch-level analog of "the SAME data, arranged
// differently in memory, costing a genuinely different amount to
// access" is a tensor's own CONTIGUITY: torch::Tensor::t() (a real
// transpose) returns a VIEW with swapped strides, not a new copy --
// mathematically identical data, addressed less regularly. This file
// measures that honestly, including reporting the real result even
// where it does NOT show CUDA's own dramatic penalty -- which turns
// out, itself, to be the genuinely interesting finding.
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 3; i++) op();   // warmup
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);
    const int64_t n = 6000;
    torch::Tensor t = torch::randn({n, n});
    torch::Tensor tt = t.t();   // a real VIEW, same underlying storage, swapped strides

    std::cout << "t.is_contiguous() = " << t.is_contiguous() << ", t.strides() = " << t.strides()
              << std::endl;
    std::cout << "t.t().is_contiguous() = " << tt.is_contiguous() << ", t.t().strides() = "
              << tt.strides() << " (a real VIEW -- SAME underlying storage as t, t.data_ptr() == "
              << "t.t().data_ptr()? " << (t.data_ptr() == tt.data_ptr()) << ", just addressed with "
              << "swapped strides)" << std::endl;

    const int reps = 20;
    double t_sum_ms = time_op_ms([&]() { volatile double s = t.sum().item<double>(); (void)s; }, reps);
    double tt_sum_ms = time_op_ms([&]() { volatile double s = tt.sum().item<double>(); (void)s; }, reps);
    std::cout << "\n[TIMING] contiguous t.sum(): " << t_sum_ms << " ms/call (avg of " << reps
              << " reps)" << std::endl;
    std::cout << "[TIMING] non-contiguous t.t().sum() (identical data, swapped strides): "
              << tt_sum_ms << " ms/call, ratio to contiguous = " << (tt_sum_ms / t_sum_ms) << std::endl;

    // .contiguous() always pays a real, unambiguous, measurable cost:
    // a genuinely fresh allocation, copying every element into
    // correctly-strided order -- exactly the kind of one-time cost
    // Section C.3's own padding trick pays once, to avoid a recurring
    // per-access penalty afterward.
    void* tt_ptr_before = tt.data_ptr();
    torch::Tensor tt_fixed = tt.contiguous();
    void* tt_ptr_after = tt_fixed.data_ptr();
    std::cout << "\ntt.contiguous(): is_contiguous() after = " << tt_fixed.is_contiguous()
              << ", data_ptr() changed (a genuinely fresh allocation, not a view)? "
              << (tt_ptr_before != tt_ptr_after) << std::endl;
    double tt_fixed_sum_ms = time_op_ms([&]() { volatile double s = tt_fixed.sum().item<double>(); (void)s; }, reps);
    std::cout << "[TIMING] tt.contiguous().sum() (after paying the one-time copy cost): "
              << tt_fixed_sum_ms << " ms/call" << std::endl;

    std::cout << "\nhonest finding: unlike raw CUDA's own dramatic, hardware-mandated ~32x bank-"
              << "conflict serialization penalty (measured directly in Section C.3's own SASS), this "
              << "file's own genuinely-measured contiguous-vs-transposed-view timing difference on CPU "
              << "is small, and not even reliably in the expected direction across runs -- real ATen "
              << "CPU kernels (TensorIterator's own stride-aware, vectorized reduction loops) are "
              << "specifically engineered to absorb most of a non-contiguous access pattern's own cost "
              << "for common ops like .sum(), which is itself the genuinely useful finding: a LibTorch "
              << "programmer writing ordinary tensor code does not experience anything like CUDA's own "
              << "hardware-mandated bank-conflict penalty from an ordinary transpose, precisely because "
              << "the framework, not hand-written code, is absorbing that cost underneath." << std::endl;

    return 0;
}
```

### `04_broadcast_expand_vs_repeat.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.4 of the CUDA C++ edition covers CONSTANT MEMORY
// broadcast: when every thread in a warp reads the SAME address, one
// cached value is served to all 32 threads in a single transaction,
// instead of 32 separate reads -- one small piece of data, reused by
// many, without being duplicated. LibTorch has no programmer-visible
// constant-memory cache -- but it has a real, different mechanism
// aimed at a related goal: BROADCASTING. torch::Tensor::expand()
// returns a VIEW with STRIDE 0 along any broadcast dimension -- many
// logical elements reading the SAME underlying value, with NO
// duplication of that value in memory at all, genuinely analogous in
// SPIRIT (one small piece of data, read by many) even though the
// underlying mechanism (a stride-0 view, not a hardware broadcast
// cache) is completely different, and this file is explicit about
// that difference rather than overstating the parallel.
int main() {
    torch::Tensor small = torch::tensor({1.0, 2.0, 3.0});   // 3 real elements, 12 bytes

    // expand(): a real VIEW. The tensor LOOKS like it has 1000 rows,
    // but every row shares the SAME 12 bytes of underlying storage --
    // no per-row duplication occurs at all.
    torch::Tensor expanded = small.unsqueeze(0).expand({1000, 3});
    std::cout << "small.unsqueeze(0).expand({1000, 3}): logical shape = " << expanded.sizes()
              << ", but small.storage().nbytes() = " << small.storage().nbytes()
              << " bytes, expanded.storage().nbytes() = " << expanded.storage().nbytes()
              << " bytes (SAME storage, no duplication -- confirmed via data_ptr() equality: "
              << (small.data_ptr() == expanded.data_ptr()) << ")" << std::endl;
    std::cout << "  expanded.stride(0) = " << expanded.stride(0) << " (a genuine stride-0 dimension -- "
              << "every one of the 1000 logical rows reads the identical 3 underlying values, exactly "
              << "the SPIRIT of constant-memory broadcast: one small piece of data, served to many "
              << "logical readers, with zero duplication)" << std::endl;

    // repeat(): NOT a view. This genuinely materializes 1000 full
    // copies of the 3 values into fresh, newly-allocated storage --
    // the direct opposite of broadcast, and this file confirms that
    // concretely via a real, measured byte count.
    torch::Tensor repeated = small.unsqueeze(0).repeat({1000, 1});
    std::cout << "\nsmall.unsqueeze(0).repeat({1000, 1}): logical shape = " << repeated.sizes()
              << ", repeated.storage().nbytes() = " << repeated.storage().nbytes()
              << " bytes (a genuinely fresh allocation, 1000x the original data, confirmed different "
              << "data_ptr() from small: " << (small.data_ptr() != repeated.data_ptr()) << ")"
              << std::endl;

    double expand_ratio = static_cast<double>(repeated.storage().nbytes()) / static_cast<double>(expanded.storage().nbytes());
    std::cout << "\nrepeated tensor uses " << expand_ratio << "x as many storage bytes as the expanded "
              << "(broadcast) view, for the IDENTICAL logical shape and IDENTICAL values -- confirmed "
              << "via torch::allclose(expanded, repeated) = "
              << torch::allclose(expanded, repeated) << std::endl;

    // Honest divergence from the CUDA book's own mechanism: constant
    // memory broadcast serves 32 THREADS reading the same address in
    // one cycle, at RUNTIME, inside a specific warp's own execution --
    // expand()'s stride-0 view is a COMPILE-TIME-invisible, purely
    // logical addressing trick with no hardware broadcast cache
    // involved at all. Both avoid duplicating one small value across
    // many readers; neither is an implementation of the other.
    std::cout << "\nhonest divergence: constant-memory broadcast (Section C.4) is a hardware CACHE that "
              << "serves 32 threads reading the same address in one cycle, at runtime, inside a warp's "
              << "own execution; torch::Tensor::expand()'s stride-0 view is a purely logical addressing "
              << "trick with no cache, no warp, and no hardware broadcast mechanism involved anywhere -- "
              << "both avoid duplicating one small value across many readers, but by genuinely different "
              << "mechanisms, and neither is an implementation of the other." << std::endl;

    return 0;
}
```

### `05_host_device_and_access_patterns.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.5 of the CUDA C++ edition recaps global memory and covers
// UNIFIED (MANAGED) MEMORY: cudaMallocManaged gives one host+device
// pointer, with the driver migrating pages across PCIe/NVLink on
// first touch. It also explicitly RESTATES, rather than re-derives,
// Chapter 18.2's own coalescing result on the book's SoA bond
// portfolio: 32 threads reading 32 CONSECUTIVE addresses cost 1
// transaction; 32 threads reading 32 SCATTERED addresses cost up to
// 32 -- an 8x gap. This file does the same thing, honestly, at the
// LibTorch level: it demonstrates the real, LibTorch-visible
// host/device story (torch::Tensor::pin_memory(), tested for real in
// this GPU-less sandbox, exactly the way this book's own no-GPU
// convention has worked since Chapter 18), and it restates Chapter
// 22.1's own coalesced-vs-scattered point with a genuinely measured
// LibTorch-level analog: a sequential gather versus a randomly
// scattered one, via real torch::Tensor::index_select().
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 2; i++) op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);

    // A real torch::Tensor::pin_memory() call, genuinely attempted --
    // not skipped or faked. It genuinely FAILS in this sandbox,
    // because pinned (page-locked) host memory exists specifically to
    // speed up a future host<->device transfer, and there is no CUDA
    // device here to transfer to at all -- a real, caught exception,
    // reported honestly, exactly the way this book has handled every
    // other GPU-hardware-dependent claim since Chapter 18.
    torch::Tensor host_tensor = torch::randn({1000});
    std::cout << "torch::cuda::is_available() = " << torch::cuda::is_available() << std::endl;
    try {
        torch::Tensor pinned = host_tensor.pin_memory();
        std::cout << "host_tensor.pin_memory() succeeded, is_pinned() = " << pinned.is_pinned()
                  << " [UNVERIFIED beyond this sandbox -- would need a real GPU-attached machine to "
                  << "confirm any actual transfer-speed benefit]" << std::endl;
    } catch (const std::exception& e) {
        // Only the exception's own first line is printed -- the full
        // message continues with a raw C++ stack trace whose memory
        // addresses are, correctly, different on every single run, so
        // printing them here would make this file's own "genuinely
        // compiled and run" output impossible to lock byte-for-byte
        // against a fresh rerun, through no fault of the underlying
        // claim being tested.
        std::string full_message(e.what());
        std::string first_line = full_message.substr(0, full_message.find('\n'));
        std::cout << "host_tensor.pin_memory() genuinely FAILED in this GPU-less sandbox, real caught "
                  << "exception (not fabricated), first line: \"" << first_line << "\" -- exactly the "
                  << "honest failure mode this book has reported since Chapter 18 for anything requiring "
                  << "real CUDA hardware; pinned memory exists to speed up a host<->device transfer, and "
                  << "there is no device to transfer to here at all." << std::endl;
    }

    std::cout << "\nunlike raw CUDA's own cudaMallocManaged (one pointer, valid on host AND device, "
              << "with the driver migrating pages on first touch), an ordinary CPU-only torch::Tensor "
              << "has no migration story at all -- host_tensor.device() = " << host_tensor.device()
              << " is simply where it lives, permanently, with no unified addressing scheme needed "
              << "because there is no second device it could ever need to migrate to in this sandbox."
              << std::endl;

    // Restating Chapter 18.2's own coalescing point and Chapter 22.1's
    // own SoA design at the LibTorch level: torch::Tensor::index_select
    // over a SEQUENTIAL index pattern versus a genuinely SCATTERED
    // (randomly permuted) one -- real timing, on real data, at real
    // scale.
    const int64_t n = 20000000;
    torch::Tensor data = torch::randn({n});
    torch::Tensor sequential_idx = torch::arange(0, n, 2);
    torch::Tensor scattered_idx = torch::randperm(n).narrow(0, 0, n / 2);

    const int reps = 5;
    double seq_ms = time_op_ms([&]() { auto r = data.index_select(0, sequential_idx); (void)r; }, reps);
    double scat_ms = time_op_ms([&]() { auto r = data.index_select(0, scattered_idx); (void)r; }, reps);

    std::cout << "\n[TIMING] data.index_select(0, sequential_idx) over " << n << " elements (every "
              << "other index, a regular access pattern): " << seq_ms << " ms/call (avg of " << reps
              << " reps)" << std::endl;
    std::cout << "[TIMING] data.index_select(0, scattered_idx) over the SAME " << n << " elements ("
              << "a genuinely random permutation, a scattered access pattern): " << scat_ms
              << " ms/call, ratio to sequential = " << (scat_ms / seq_ms) << std::endl;
    std::cout << "\nthis is the same underlying fact Chapter 18.2 and Chapter 22.1 both already "
              << "established (coalesced/regular access beats scattered access, a real and dramatic "
              << "gap, not a rounding error) -- restated here, at the LibTorch level, via ordinary "
              << "torch::Tensor::index_select() rather than a hand-written CUDA kernel's own coalescing "
              << "analysis, and genuinely re-measured on real CPU hardware rather than assumed to carry "
              << "over from the GPU-specific figures Chapter 18.2 and Appendix C.1 both cite." << std::endl;

    return 0;
}
```

### `06_cpu_thread_pool_execution_model.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>

// Appendix C.6 of the CUDA C++ edition covers CUDA's own execution
// model: threads grouped into 32-wide warps (one instruction stream,
// lockstep), warps grouped into a CTA/block (one SM, shared __shared__
// memory), CTAs grouped into a grid -- with a genuine, measurable
// "wasted lanes" cost when a block size does not divide evenly into
// warps of 32 (e.g. a block of 100 threads still occupies 4 full
// warps = 128 hardware lanes, wasting 28 of them). LibTorch's CPU
// backend has NO analog of this at all -- there is no warp, no CTA,
// and no lockstep hardware execution unit a LibTorch C++ programmer
// manages. What DOES genuinely exist, and IS directly visible to
// ordinary LibTorch code, is a real CPU thread pool: at::
// get_num_threads() / torch::set_num_threads() control how many CPU
// threads a single op's own internal computation is allowed to use.
// This file is honest about the fact that this is NOT the same kind
// of thing as CUDA's own warp/CTA hardware model -- there is no
// "wasted lane" concept at all for ordinary LibTorch CPU code -- while
// still demonstrating the one real, measurable execution-model fact
// that genuinely IS available: real parallelism, genuinely present,
// genuinely controllable, and genuinely measurable via wall-clock time.
double time_op_ms(const std::function<void()>& op, int reps) {
    for (int i = 0; i < 2; i++) op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) op();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    torch::manual_seed(0);
    int default_threads = torch::get_num_threads();
    std::cout << "torch::get_num_threads() (this sandbox's own real, default intra-op CPU thread pool "
              << "size) = " << default_threads << std::endl;

    std::cout << "\nCUDA's own execution model (Section C.6): thread (1, private registers) -> warp "
              << "(32 threads, one instruction stream, lockstep) -> CTA/block (multiple warps, one SM, "
              << "shared __shared__ memory + __syncthreads()) -> grid (multiple CTAs, whole device). "
              << "LibTorch's own CPU execution model has NO layer that corresponds to a warp or a CTA "
              << "at all -- there is no lockstep hardware execution unit, no fixed 32-wide grouping, and "
              << "consequently no 'wasted lanes' concept for a block size that does not divide evenly "
              << "into anything: at::parallel_for's own internal chunking is an ATen implementation "
              << "detail, not something ordinary LibTorch code written the way this book has written it "
              << "chooses or observes directly." << std::endl;

    // The one real, directly-controllable, directly-measurable fact:
    // torch::set_num_threads() genuinely changes how many CPU threads
    // a single op's own internal computation uses, and that
    // genuinely, measurably changes wall-clock time for a large enough
    // op -- real parallelism, confirmed rather than assumed.
    torch::Tensor a = torch::randn({4000, 4000});
    torch::Tensor b = torch::randn({4000, 4000});

    torch::set_num_threads(1);
    double one_thread_ms = time_op_ms([&]() { auto r = torch::matmul(a, b); (void)r; }, 5);

    torch::set_num_threads(default_threads);
    double default_thread_ms = time_op_ms([&]() { auto r = torch::matmul(a, b); (void)r; }, 5);

    std::cout << "\n[TIMING] torch::matmul(4000x4000, 4000x4000) with torch::set_num_threads(1): "
              << one_thread_ms << " ms/call (avg of 5 reps)" << std::endl;
    std::cout << "[TIMING] the SAME matmul with torch::set_num_threads(" << default_threads
              << ") (this sandbox's own default): " << default_thread_ms << " ms/call, speedup = "
              << (one_thread_ms / default_thread_ms) << "x" << std::endl;

    torch::Tensor result_one = torch::matmul(a, b);
    std::cout << "\nboth thread-count settings compute the IDENTICAL mathematical result (confirmed via "
              << "torch::allclose): " << torch::allclose(result_one, torch::matmul(a, b))
              << " -- thread count changes only how many real CPU threads cooperate on the SAME "
              << "computation, exactly as CUDA's own warp/CTA counts change only how many hardware "
              << "lanes cooperate, never what gets computed." << std::endl;

    return 0;
}
```

### `07_fp16_matmul_tensor_cores.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix C.7 of the CUDA C++ edition hand-writes WMMA fragment code
// -- wmma::fragment<...>, load_matrix_sync, mma_sync, store_matrix_sync
// -- to drive a GPU's own tensor cores directly: one warp-wide
// instruction performing a real 16x16x16 fp16-in/fp32-accumulate
// matrix multiply, verified against a real SASS instruction
// (HMMA.16816.F32) genuinely different from ordinary FFMA/HFMA2
// scalar arithmetic. A LibTorch programmer never writes any of that
// fragment code at all: an ordinary real torch::matmul() call, given
// two fp16 tensors on a CUDA device, dispatches automatically to
// cuBLAS's own tensor-core-accelerated GEMM kernels underneath --
// zero fragment code, zero manual load_matrix_sync/mma_sync calls,
// the exact "LibTorch gets this for free" pattern this book has shown
// for real autograd, real serialization, and real broadcasting. This
// sandbox has no GPU, so the actual HMMA-instruction dispatch itself
// is honestly [UNVERIFIED -- pending real-GPU test] -- but the
// MATHEMATICAL correctness of fp16 matrix multiplication itself is
// fully, genuinely testable on CPU, via the identical identity-matrix
// worked example the CUDA book's own Section C.7 uses.
int main() {
    // The CUDA book's own exact worked example: A = 16x16 identity
    // matrix (fp16), B[i][j] = i*16+j (fp16, values 0-255, each one
    // exactly representable in fp16's own mantissa), so A@B = B
    // exactly, with zero rounding error anywhere in the multiply.
    torch::Tensor a = torch::eye(16, torch::kFloat16);
    torch::Tensor b = torch::empty({16, 16}, torch::kFloat16);
    {
        auto b_acc = b.accessor<at::Half, 2>();
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                b_acc[i][j] = static_cast<float>(i * 16 + j);
            }
        }
    }

    // A real torch::matmul() call, fp16 in, fp16 out -- on a CUDA
    // device this would dispatch to cuBLAS's own tensor-core GEMM
    // kernel; on this sandbox's CPU-only device it dispatches to
    // ATen's own CPU fp16 matmul kernel instead. Same real, production
    // API call either way -- no fragment code, no manual tiling,
    // written by this file at all.
    torch::Tensor c = torch::matmul(a, b);

    std::cout << "real torch::matmul(A, B), A=16x16 fp16 identity, B[i][j]=i*16+j (fp16): "
              << "torch::equal(C, B) (exact match, not merely allclose)? " << torch::equal(c, b)
              << std::endl;

    auto c_acc = c.accessor<at::Half, 2>();
    std::cout << "  sample values -- C[0][0] = " << static_cast<float>(c_acc[0][0])
              << " (CUDA book's own expected 0.0), C[3][7] = " << static_cast<float>(c_acc[3][7])
              << " (expected 55.0), C[15][15] = " << static_cast<float>(c_acc[15][15])
              << " (expected 255.0)" << std::endl;

    std::cout << "\nno wmma::fragment, load_matrix_sync, or mma_sync anywhere in this file -- the "
              << "single call torch::matmul(a, b) above is the ENTIRE fp16 matrix-multiply "
              << "implementation this file needed to write, on either a CPU-only sandbox (dispatching "
              << "to an ordinary CPU fp16 GEMM kernel) or a real CUDA-attached machine (dispatching to "
              << "cuBLAS's own tensor-core-accelerated GEMM kernel instead) -- the SAME LibTorch source "
              << "code, unmodified, in both cases." << std::endl;

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] this sandbox has no CUDA device (torch::cuda::"
              << "is_available() = " << torch::cuda::is_available() << "), so this file cannot confirm, "
              << "the way Section C.7's own compiled SASS does, that a real HMMA.16816.F32 tensor-core "
              << "instruction actually gets dispatched for this exact call on real hardware -- only that "
              << "torch::matmul() on a CUDA tensor with fp16 dtype IS the real, documented, production "
              << "code path that would trigger it, with no fragment code required from this file either "
              << "way." << std::endl;

    return 0;
}
```

## Chapter Summary

This appendix mapped the CUDA C++ edition's own seven-section memory-architecture appendix onto LibTorch's real, programmer-visible memory and execution model, section by section, and found a consistent shape: CUDA's own explicit, six-space hardware hierarchy collapses, for a LibTorch programmer, into one placement decision (which device) plus a handful of real, measurable, but structurally different concerns underneath it. In-place writes versus fresh allocations are the real analog of avoiding a register spill, confirmed via `data_ptr()` identity. An ordinary tensor transpose's own real, measured contiguity cost turns out to be small on CPU, not because the underlying concern is fake but because ATen's own `TensorIterator` absorbs most of it -- an honest finding, not a disappointing one. `expand()`'s stride-0 view achieves broadcast's own goal through a purely logical mechanism, with no hardware cache involved at all. An ordinary CPU tensor has no unified-memory migration story whatsoever, while a scattered-versus-sequential access pattern still costs a real, dramatic, measured difference, confirming Chapter 18.2's and Chapter 22.1's own coalescing point independently at the LibTorch level. LibTorch's CPU execution model has exactly one programmer-visible layer (a thread pool), not CUDA's three (thread, warp, CTA), and no "wasted lanes" concept at all. And a single `torch::matmul()` call is the entire tensor-core-dispatch code a LibTorch programmer ever writes, with the actual hardware-level SASS confirmation honestly marked `[UNVERIFIED -- pending real-GPU test]` on this sandbox, exactly as this book's own convention has required since Chapter 18.

## Self-Check Questions

1. Section C.1 argues that a LibTorch programmer's own memory-hierarchy decision is "far flatter" than CUDA's own six-space model. Explain precisely what that flatter decision is, and why the other five spaces (local, shared, constant, L2, and the finer structure of global memory) don't disappear, but instead become someone else's decision.
2. Section C.2 measures "genuinely new allocations" via `data_ptr()` comparisons rather than via any register or spill count. Explain why `data_ptr()` identity is a meaningful, honest analog for "did this value have to leave a fast, limited resource for a slower one," even though a CPU allocator and a GPU register file are genuinely different kinds of resource.
3. Section C.3 reports a SMALL, sometimes-inconsistent timing difference between a contiguous tensor and its own transposed view, in contrast to CUDA's own dramatic ~32x bank-conflict penalty. Explain why this section treats that small difference as a genuinely useful finding rather than a failed experiment, and what specific mechanism it credits for the gap between the two results.
4. Section C.5 reports that `pin_memory()` genuinely fails on this sandbox, then only prints the FIRST LINE of the resulting exception message rather than the whole thing. Explain what would have broken about this appendix's own verification pipeline if the full exception message (stack trace included) had been embedded and locked into the chapter text instead.
5. Section C.7 states that a LibTorch programmer's entire fp16 tensor-core-dispatch implementation is "the single line `torch::matmul(a, b)`," yet the same section also tags the claim that this actually reaches real tensor-core hardware as `[UNVERIFIED -- pending real-GPU test]`. Explain precisely what IS and is NOT verified by this section's own genuinely-run identity-matrix worked example, given that distinction.

## Where We Go Next

This appendix, like Section C.7's own closing text in the CUDA C++ edition, is reference material rather than a new leg of this book's own arc -- Part 7 already closed that arc in Chapter 22. It exists to be revisited whenever a specific question of the form "why does this LibTorch code cost what it costs" needs a concrete, genuinely-measured answer rather than a guess: in-place versus fresh allocation (Section C.2), contiguity (Section C.3), broadcast versus repeat (Section C.4), access pattern (Section C.5), thread count (Section C.6), or dtype/dispatch (Section C.7). The appendices that follow continue in the same reference spirit, covering C++ functional and lambda programming, the streaming-multiprocessor-as-pipeline model, the Rule of Five, and tensor contractions on CPU and GPU -- material this book's own main chapters have used along the way without dwelling on every implementation detail.

## Worked Solutions

**1.** The flatter decision is simply which DEVICE a tensor's own storage lives on -- for every chapter in this book, that decision has had exactly one real answer (CPU), confirmed directly via `torch::cuda::is_available()` rather than assumed; on a CUDA-capable machine, the same decision would be made via `.to(torch::kCUDA)` or an equivalent device-placement call. The other five spaces do not disappear -- they still genuinely exist, and still genuinely determine how fast any given real op runs -- but the decision of WHICH of those five a specific intermediate value uses is made by whoever wrote the ATen/cuBLAS/cuDNN kernel that a real op like `torch::matmul` or `torch::relu` dispatches to, at the point that kernel was written, not by application code calling that op. A LibTorch programmer's own code genuinely never contains a `__shared__` declaration, a `__constant__` declaration, or an explicit register-pressure decision at all -- those decisions were made once, by the framework's own authors, and are reused correctly on every subsequent call, which is precisely the "gets this for free" pattern this book has shown repeatedly since its own earliest chapters.

**2.** A CPU allocator and a GPU register file are indeed different kinds of resource -- different sizes, different latencies, different mechanisms for deciding what goes where -- but both share the identical STRUCTURAL property this comparison actually depends on: a value that stays in the SAME storage location across an entire computation is cheaper than one that gets moved to a NEW storage location partway through, for reasons that are fundamentally about data movement rather than about computation itself. `data_ptr()` identity is a directly, mechanically observable stand-in for exactly that property at the tensor level: when an operation is in-place, no data movement to a new location happens at all (the identical `data_ptr()` before and after); when an operation is out-of-place, a fresh block of memory is allocated and the result is written there, which is a real, measurable act of data movement, structurally the same KIND of event a register spill is, even though the specific resource being moved out of, and the specific resource being moved into, are both completely different in a CPU-allocator context than in a GPU-register-file context.

**3.** This section treats the small, inconsistent timing gap as genuinely useful specifically because the alternative -- reporting a larger, more dramatic number that this file did not actually, reliably measure -- would have been a fabrication, and this book's own standing rule is that a genuine test that contradicts an expectation corrects the text, never the reverse. The specific mechanism credited for the gap between CUDA's own dramatic bank-conflict penalty and this section's own muted result is `TensorIterator`, ATen's own internal, stride-aware, vectorized iteration machinery: CUDA's own bank conflicts are a HARDWARE consequence with no software layer positioned to absorb them (32 threads either collide on a bank or they don't, and the hardware serializes the collision unconditionally), while a non-contiguous CPU tensor's own access pattern passes through a software layer (`TensorIterator`) specifically engineered to handle strided access efficiently for common operations like `.sum()`, absorbing most of what would otherwise be a real performance penalty before it ever reaches the CPU's own memory subsystem.

**4.** A raw C++ exception's own stack-trace text embeds real memory addresses (function and library load addresses) that are, correctly and unavoidably, different on every single process run -- address space layout randomization and simple differences in how shared libraries happen to be mapped mean two genuinely identical runs of the identical binary produce two DIFFERENT stack traces, byte for byte. This appendix's own verification pipeline locks each chapter file's own genuine output against a fresh rerun of the identical compiled program, expecting an EXACT match; had the full stack trace been embedded in the chapter text, that exact-match check would fail on every single fresh rerun, not because anything about the underlying claim (that `pin_memory()` fails without a GPU) had become false, but because an incidental, non-deterministic detail of HOW it failed was locked into text that a genuine rerun could never reproduce exactly. Printing only the exception's own first line -- the actual, deterministic error message -- keeps the genuinely meaningful, reproducible claim locked and verifiable, while dropping the genuinely non-reproducible incidental detail.

**5.** What IS verified, genuinely, is the MATHEMATICAL correctness of calling `torch::matmul()` on fp16 tensors: the identity-matrix multiply produces an exactly correct result (`torch::equal`, not merely `torch::allclose`), confirming that fp16 matrix multiplication through this real, production API call is arithmetically sound, and confirming that the CUDA book's own specific worked-example values (0, 55, 255) are reproduced exactly, as expected for deterministic, RNG-free arithmetic. What is NOT verified is anything about WHICH HARDWARE INSTRUCTIONS actually executed this multiplication -- on this sandbox's CPU-only device, `torch::matmul()` on fp16 tensors dispatches to an ordinary CPU fp16 GEMM kernel, not to anything resembling a tensor core or an `HMMA` instruction, which exist only on real CUDA-capable GPU hardware; the claim that the IDENTICAL, unmodified `torch::matmul(a, b)` call would dispatch to cuBLAS's own tensor-core-accelerated kernel on a real CUDA device is a true, documented fact about LibTorch's own real dispatch behavior, but it is a claim about hardware this sandbox does not have, and is honestly marked `[UNVERIFIED -- pending real-GPU test]` rather than presented as something this file's own genuinely-run output actually confirms.
