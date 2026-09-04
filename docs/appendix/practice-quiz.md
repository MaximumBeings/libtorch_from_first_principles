# Appendix B: Practice Quiz

This appendix is a cumulative review across the whole book: two questions per chapter, in reading order, plus one question drawn from each of the math- and systems-focused appendices (C through I). Every question is built from a specific, real fact this book's own chapters and appendices established by actually running code -- a measured number, a genuine API name, an honest divergence from the CUDA C++ edition -- not from general PyTorch knowledge. The intent is for a reader to attempt a Part's own questions right after finishing that Part, then check answers against the Answer Key at the end.

Each question is multiple choice with one correct answer. Questions are numbered continuously (1 through 51) so they can be checked against the single Answer Key at the very end of this appendix, matching the numbering style this book's own Self-Check Questions use throughout.

## Part 0 — LibTorch Foundations

### Chapter 1: Variables and Types

**1.** What real, measured fact does Chapter 1 establish about `torch::TensorOptions`?

A. It occupies exactly 8 bytes, bitfield-packed, despite exposing chained setters like `.dtype()`, `.device()`, and `.requires_grad()`
B. It heap-allocates a new object on every chained setter call
C. Its copy constructor is deleted, forcing move-only semantics
D. It is exactly 32 bytes -- one machine word per field, with no packing

**2.** Chapter 1 demonstrates that a C++ struct's size can shrink purely by:

A. Removing data members
B. Reordering fields to reduce alignment padding, with no data removed
C. Adding a custom destructor
D. Switching from a `struct` to a `class`

### Chapter 2: Struct Design Patterns

**3.** `torch::TensorAccessor<T,N>` is best described, per Chapter 2, as:

A. A non-owning wrapper around a data pointer plus sizes/strides arrays, for unchecked multi-dimensional indexing
B. A deep copy of tensor data, optimized for cache locality
C. A garbage-collected smart pointer to tensor storage
D. A CUDA-only accessor with no CPU equivalent

**4.** Chapter 2's comparison of array-of-structs (AoS) versus struct-of-arrays (SoA) layout:

A. Shows no measurable performance difference on modern CPUs
B. Is a purely theoretical distinction, with no code demonstrated
C. Shows a real, reproducible timing gap tied to cache-line access patterns
D. Only matters for GPU code, not CPU

### Chapter 3: Memory Layout Strategies

**5.** For a `(2,3)` row-major CPU tensor, Chapter 3 reports the strides as:

A. `[1,3]`
B. `[3,1]`
C. `[2,3]`
D. `[6,1]`

**6.** Chapter 3's key finding about `.expand()` versus `.repeat()`:

A. Both physically copy data
B. Both are zero-copy
C. `.expand()` uses stride-0 dimensions (zero-copy); `.repeat()` physically copies data
D. `.expand()` copies; `.repeat()` is zero-copy

### Chapter 4: GPU Programming Introduction

**7.** The correct, real LibTorch APIs to check GPU presence, per Chapter 4, are:

A. `torch::has_cuda()` and `torch::gpu_count()`
B. `torch::cuda::is_available()` and `torch::cuda::device_count()`
C. `torch::device_available()` and `torch::cuda::count()`
D. `c10::cuda::present()` and `c10::cuda::n_devices()`

**8.** Chapter 4's common-trap finding about device placement:

A. Constructing directly with `.device(kCUDA)` and building on CPU then calling `.to(kCUDA)` always behave identically
B. They are two different code paths, with distinct error behavior when CUDA is unavailable
C. `.to(kCUDA)` is deprecated in favor of `.device(kCUDA)`
D. Neither method can fail if CUDA is unavailable

### Chapter 5: SIMD and Vectorization

**9.** Chapter 5's real compiled-instruction evidence for vectorization includes:

A. `vaddps` (vectorized add) versus `addss` (scalar add), confirmed via actual disassembly
B. Only a claimed 4x speedup, with no disassembly shown
C. ARM NEON instructions only, since no x86 machine was available
D. A Python-level timing comparison, with no assembly inspection

**10.** Chapter 5 treats loop unrolling and kernel fusion as:

A. The same optimization under two different names
B. Two separate optimizations, each requiring its own separate demonstration
C. Mutually exclusive techniques that cannot both apply to one loop
D. GPU-only optimizations, with no CPU relevance

## Part 1 — Core Tensor Infrastructure

### Chapter 6: The Tensor

**11.** Chapter 6's Section 6.5 "honest divergence" finding about `torch::Tensor`'s copy constructor:

A. It is deleted, forcing move-only semantics
B. It performs a deep copy, unlike the CUDA C++ edition's own hand-rolled `Tensor`
C. It is not deleted, and performs a shallow copy sharing underlying storage
D. It triggers a compile error unless `std::move` is used

**12.** Mutating an element through a shallow copy, in Chapter 6's worked example:

A. Has no effect on the original tensor
B. Is visible in the original tensor, proving shared storage
C. Throws a runtime error
D. Silently creates a private copy first (copy-on-write)

### Chapter 7: Memory Layout Design

**13.** `c10::GetCPUAllocator()`, per Chapter 7's real measurement, guarantees CPU allocations aligned to:

A. 16 bytes
B. 32 bytes
C. 64 bytes
D. 128 bytes

**14.** `torch::MemoryFormat::ChannelsLast`, per Chapter 7:

A. Is identical to standard contiguous layout, just a different name
B. Is a real, distinct memory format, with different stride ordering for the same logical shape
C. Only exists in the Python frontend, not LibTorch
D. Is deprecated and unavailable in current LibTorch

### Chapter 8: Tensor Creation Functions

**15.** `torch::arange(start, stop)` and `torch::linspace(start, end, n)`, per Chapter 8, are respectively:

A. Both inclusive of their endpoint
B. `arange` is stop-exclusive; `linspace` is endpoint-inclusive
C. `arange` is stop-inclusive; `linspace` is endpoint-exclusive
D. Both stop-exclusive

**16.** Chapter 8 confirms `torch::randperm` and `torch::multinomial` internally use:

A. A fixed lookup table, not real randomness
B. Fisher-Yates shuffle (`randperm`) and inverse-CDF sampling (`multinomial`), confirmed via statistical testing
C. Rejection sampling, for both
D. The identical algorithm, for both functions

### Chapter 9: Specialized Tensor Types

**17.** For `torch::sparse_coo_tensor(...)`, Chapter 9 shows that a meaningful (deduplicated) `._nnz()` count requires:

A. Calling `.coalesce()` first, to merge duplicate indices
B. No special handling -- `._nnz()` is always deduplicated
C. Converting to dense first
D. Calling `.to_sparse_csr()` first

**18.** `c10::Half` and `c10::BFloat16`, per Chapter 9:

A. Are the same 16-bit format under two different names
B. Are distinct 16-bit float types, with different exponent/mantissa splits
C. Are both 32-bit formats despite the name
D. Are the same type; only `BFloat16` is real, `Half` is fictional

### Chapter 10: Device Abstraction Layer

**19.** On a CPU-only machine, `torch::cuda::device_count()` returning `0`, per Chapter 10, is:

A. A bug that should be reported
B. The correct, expected real behavior
C. Only possible if the CUDA headers are missing entirely
D. Impossible -- the call always throws instead

**20.** Chapter 10's device-aware allocator fallback discussion reuses which earlier chapter's finding as a cross-check?

A. Chapter 3's stride formula
B. Chapter 7's 64-byte CPU alignment finding
C. Chapter 5's disassembly evidence
D. Chapter 1's `TensorOptions` byte size

### Chapter 11: Memory Management System

**21.** `torch::Tensor::use_count()`, per Chapter 11's worked example, traces a real reference-count lifecycle of:

A. `1 -> 2 -> 1`
B. `1 -> 2 -> 3 -> 2 -> 1`
C. `0 -> 1 -> 0`
D. Always exactly `1`, since `Tensor` is not reference-counted

**22.** In Chapter 11's `Arena` alignment example, a debug build's `assert()` on overflow aborts with exit code `134`. An identical `-DNDEBUG` release build instead:

A. Also aborts with exit code `134`
B. Silently writes out of bounds and exits `0`
C. Throws a C++ exception instead
D. Refuses to compile

## Part 2 — Basic Tensor Operations

### Chapter 12: Element-wise Operations

**23.** At `a=2, b=5`, Chapter 12 confirms `d(a/b)/da` and `d(a/b)/db` via autograd as:

A. `0.2` and `-0.08`
B. `0.4` and `-0.16`
C. `0.5` and `-0.2`
D. `0.2` and `0.08`

**24.** For `(A+B).sum()` where `B` (shape `(3,)`) broadcasts across `A`'s 2 rows, Chapter 12 shows `B.grad()` is:

A. `[1,1,1]`, since each element of `B` is used once
B. `[2,2,2]`, since each element of `B` contributes to the sum once per row
C. `[0,0,0]`, since gradients don't flow through broadcasting
D. Undefined -- broadcasting breaks autograd

### Chapter 13: Matrix Operations

**25.** Calling `torch::matmul` on shape-mismatched tensors, per Chapter 13, is an honest divergence from the CUDA C++ edition because LibTorch:

A. Silently returns a wrong-but-plausible result, just like the hand-rolled kernel
B. Throws a real `c10::Error`, rather than reading garbage
C. Crashes the process with a segmentation fault
D. Auto-reshapes the tensors to make them compatible

**26.** Chapter 13 shows `.reshape({2,3})` on a transposed tensor `At` (from `A=[[1,2,3],[4,5,6]]`) produces a result that mismatches `A` at:

A. 0 of 6 positions (they match)
B. Exactly 4 of 6 positions
C. All 6 positions
D. Exactly 2 of 6 positions

### Chapter 14: Reduction Operations

**27.** `torch::var(x)`'s default behavior on `[2,4,4,4,5,5,7,9]`, per Chapter 14, gives:

A. `4.0` (population variance)
B. `4.57143` (Bessel-corrected, divides by `n-1`)
C. `4.57143` divided by `n`
D. Exactly `4.5`

**28.** `torch::nn::utils::clip_grad_norm_` on gradient `[3,4]` with `max_norm=2.0`, per Chapter 14, gives a scaled result of:

A. `[1.2, 1.6]`, with the clipped vector's norm exactly `2.0`
B. `[3,4]` unchanged, since the norm was already under the max
C. `[1.5, 2.0]`
D. `[0.6, 0.8]`

## Part 3 — Computational Graph Foundation

### Chapter 15: Graph Node Architecture

**29.** Chapter 15 confirms every real differentiable LibTorch operation is wired to its own backward `Node` type:

A. Via a runtime, string-keyed registry lookup
B. Statically, at compile time, with no runtime lookup -- confirmed across ops like `AddBackward0`, `MulBackward0`, and `ExpBackward0`
C. Via Python-side dispatch only
D. Only for a small subset of operations; most fall back to numerical differentiation

**30.** Chapter 15's gating rule for whether a graph node (`.grad_fn()`) is created:

A. ALL operands must have `requires_grad=true`
B. ANY operand having `requires_grad=true` is enough
C. It depends on tensor dtype, not `requires_grad`
D. A node is always created, regardless of `requires_grad`

## Part 4 — Automatic Differentiation Engine

### Chapter 16: Backward Function Implementation

**31.** Chapter 16 shows `torch::autograd::Function<T>` custom functions can supply gradients via:

A. Finite-difference approximation only
B. The implicit function theorem -- useful for solvers with no explicit forward formula
C. Manual chain-rule differentiation of every possible input
D. Symbolic differentiation via a computer algebra system

**32.** `retain_graph=true`, per Chapter 16, specifically enables:

A. Suppressing an error, with no other behavioral effect
B. Gradient accumulation across multiple `.backward()` calls on overlapping graph structure
C. Automatic graph pruning after each backward pass
D. Switching from reverse-mode to forward-mode autodiff

### Chapter 17: Gradient Computation Engine

**33.** `torch::autograd::grad(..., create_graph=true)`, per Chapter 17, enables:

A. Discarding the graph immediately after use, saving memory
B. Second-order derivatives (Hessians / Hessian-vector products), by keeping the backward pass itself differentiable
C. Parallel execution of independent backward passes
D. Skipping gradient computation for frozen parameters

**34.** Chapter 17 confirms that `AddBackward0` producing aliased/shared tensor references during backward is:

A. A real bug that corrupts gradients
B. Confirmed as harmless -- a tested finding about how the engine reuses tensors internally
C. Only safe on CPU, unsafe on GPU
D. Undefined behavior that the caller must avoid

## Part 5 — GPU Acceleration and Performance

### Chapter 18: GPU Kernel Implementation

**35.** Chapter 18 tags GPU-hardware-dependent claims that could not be run on real hardware as:

A. `[ASSUMED]`
B. `[UNVERIFIED -- pending real-GPU test]`
C. `[TODO]`
D. It silently omits them instead

**36.** Chapter 18's benchmark harness discipline requires, before reporting any performance number:

A. A single, untimed run
B. Warmup runs plus timed runs, averaged
C. Exactly one timed run, with no warmup
D. Only a theoretical FLOP-count estimate, no actual run

### Chapter 19: Performance Optimization Techniques

**37.** Chapter 19 confirms, via disassembly, that C++ template-based compile-time specialization has:

A. A small but nonzero runtime dispatch cost
B. Zero runtime dispatch cost -- though the chapter is explicit this doesn't mean zero cost overall (compile time, code size, etc. still apply)
C. Worse runtime performance than a runtime `if`/`else` dispatch
D. No measurable effect at all, positive or negative

**38.** Chapter 19 treats loop unrolling and kernel fusion, continuing Chapter 5's own distinction, as:

A. Interchangeable terms for the same optimization
B. Two distinct, separately-measurable optimizations, each with its own before/after evidence
C. Obsolete techniques superseded by autograd
D. GPU-only techniques, irrelevant on CPU

## Part 6 — Neural Network Building Blocks

### Chapter 20: Neural Network Layers

**39.** Chapter 20 checks `torch::nn::Linear`/`Sequential` initial-weight statistics against:

A. A fixed constant, regardless of layer size
B. The theoretical He and Xavier initialization formulas
C. No theoretical baseline -- purely qualitative inspection
D. The CUDA C++ edition's own hand-rolled initializer values

**40.** The "loss/gradient scale trap," a bug pattern the CUDA C++ edition warns about, is shown in Chapter 20 to be:

A. Equally present in LibTorch's real autograd engine
B. Structurally impossible under LibTorch's real autograd engine
C. Only avoidable by manual gradient clipping
D. Worse in LibTorch, due to automatic mixed precision

### Chapter 21: Advanced Features

**41.** Chapter 21 contrasts a hand-rolled binary save/load format against:

A. JSON serialization, which LibTorch uses exclusively
B. The real `torch::save`/`torch::load` functions, with concrete format/compatibility differences shown
C. Pickling via Python, since LibTorch has no native serialization
D. Nothing -- no real LibTorch equivalent exists

**42.** Chapter 21 verifies online softmax (the core Flash Attention algorithm) against:

A. A random baseline, since no ground truth exists
B. The naive full-softmax computation, for bit/tolerance-level agreement
C. Only a theoretical proof, with no numeric run
D. PyTorch's Python-level softmax only, not a from-scratch comparison

## Part 7 — Financial Computing Applications

### Chapter 22: Quantitative Finance Examples

**43.** Chapter 22 computes bond pricing sensitivities (DV01) via:

A. Finite-difference bumping only
B. Autodiff (`.backward()` through the pricing formula), cross-checked against a manual bump-and-reprice calculation
C. A closed-form analytic derivative, with no numeric verification
D. Monte Carlo simulation exclusively

**44.** Chapter 22's credit-spread bisection solver has a specific silent-failure trap:

A. It always converges exactly, with no failure mode
B. Hitting an iteration cap before convergence can return a plausible-looking but wrong spread
C. It throws an unrecoverable exception if it doesn't converge in one step
D. It defaults to a spread of `0` on any failure

## Appendices

### Appendix C: LibTorch Memory Architecture

**45.** Per Appendix C, what does dispatching a matmul to tensor cores require of a LibTorch programmer?

A. Manually writing a WMMA fragment API call, same as CUDA
B. A single `torch::matmul()` call -- no hand-written dispatch code at all
C. Setting a CMake flag, but still requiring a custom kernel
D. Nothing -- tensor cores are not accessible from LibTorch at all

### Appendix D: C++ Functional and Lambda Programming

**46.** Appendix D's key honest finding about CUDA's own restriction against reference captures in extended device lambdas:

A. It is a purely artificial restriction, with no real benefit
B. It is inconvenient, but genuinely prevents a real bug class (dangling references) that ordinary host C++ lambdas remain fully exposed to -- demonstrated via a genuine AddressSanitizer catch
C. LibTorch enforces the identical restriction on all host lambdas
D. It only applies to lambdas capturing `torch::Tensor` objects

### Appendix E: The SM as a Pipeline and the Tiling Payoff

**47.** Appendix E.4's real, measured comparison of `torch::matmul()` against a hand-written naive triple-loop matmul, at N=256, found a speedup of approximately:

A. 2x
B. 18x
C. 100x
D. No measurable difference

### Appendix F: C++ Lambda Functions, From First Principles

**48.** Appendix F confirms, via real `sizeof` measurements, that an empty-capture, one-int-capture, and three-int-capture lambda have sizes of:

A. `0, 4, 12` bytes
B. `1, 4, 12` bytes
C. `4, 8, 16` bytes
D. All exactly `1` byte, since captures don't affect size

### Appendix G: The Rule of Five, and a Risk Engine to Exercise It

**49.** Appendix G's central honest-divergence finding is that:

A. `torch::Tensor` requires the exact same hand-written Rule of Five as CUDA's own `GPUPathBuffer`
B. `torch::Tensor`'s own `intrusive_ptr`-based reference counting already IS a correctly-implemented Rule of Five/Zero for device memory, for free
C. `torch::Tensor` has no copy constructor at all
D. The Rule of Five is irrelevant to LibTorch programming

### Appendix H: Tensor Contractions, From First Principles (CPU)

**50.** Appendix H shows that ordinary matrix multiplication is:

A. An entirely separate operation from tensor contraction
B. The p=1 special case of tensor contraction -- a contraction over exactly one shared axis
C. Only expressible as a contraction when both tensors are square
D. Impossible to express via `torch::einsum()`

### Appendix I: Tensor Contractions, From First Principles (GPU)

**51.** Appendix I's approach to the GPU version of tensor contraction differs from the CUDA C++ edition's own hand-written kernels because:

A. It writes an equivalent set of `.cu` kernels, but never compiles them
B. `torch::tensordot()`/`torch::einsum()` already dispatch to a device-appropriate implementation, so the same call -- with no `MAX_RANK` constant or metadata struct -- handles CPU or CUDA
C. LibTorch cannot perform tensor contractions on a GPU at all
D. It requires the reader to have real GPU hardware to follow along

## Answer Key

**1.** A -- `torch::TensorOptions` is 8 bytes, bitfield-packed, regardless of how many setters are chained onto it.

**2.** B -- Chapter 1 shrinks a struct's size purely by reordering fields to eliminate alignment padding; no data is removed.

**3.** A -- `torch::TensorAccessor<T,N>` is a non-owning view: a data pointer plus sizes/strides arrays, giving unchecked multi-dimensional indexing with no copy.

**4.** C -- AoS versus SoA shows a real, reproducible timing gap in Chapter 2, tied directly to cache-line access patterns.

**5.** B -- Row-major strides for shape `(2,3)` are `[3,1]`: moving one row skips 3 elements; moving one column skips 1.

**6.** C -- `.expand()` is a zero-copy stride-0 broadcast view; `.repeat()` allocates and physically copies data.

**7.** B -- `torch::cuda::is_available()` and `torch::cuda::device_count()` are the real APIs Chapter 4 uses throughout.

**8.** B -- Building on CPU then calling `.to(kCUDA)` and constructing directly with `.device(kCUDA)` are genuinely different code paths, with different error behavior when no CUDA device exists.

**9.** A -- Chapter 5 shows real disassembly: `vaddps` (vectorized) versus `addss` (scalar), not just a claimed speedup.

**10.** B -- Chapter 5 treats loop unrolling and kernel fusion as two separate optimizations, each independently demonstrated.

**11.** C -- `torch::Tensor`'s copy constructor is not deleted; it performs a shallow copy that shares underlying storage, an honest divergence from the CUDA C++ edition's own hand-rolled `Tensor`.

**12.** B -- Mutating through the shallow copy is visible in the original, because both share the same storage.

**13.** C -- `c10::GetCPUAllocator()` guarantees 64-byte-aligned CPU allocations, confirmed by real measurement.

**14.** B -- `torch::MemoryFormat::ChannelsLast` is a real, distinct memory format with its own stride ordering for the same logical shape.

**15.** B -- `torch::arange` is stop-exclusive; `torch::linspace` is endpoint-inclusive, and correctly handles `n=1`.

**16.** B -- `torch::randperm` uses a genuine Fisher-Yates shuffle; `torch::multinomial` uses inverse-CDF sampling, both confirmed statistically.

**17.** A -- `._nnz()` only reports a meaningful, deduplicated count after `.coalesce()` merges duplicate indices.

**18.** B -- `c10::Half` and `c10::BFloat16` are both real, distinct 16-bit float types, with different exponent/mantissa splits.

**19.** B -- `torch::cuda::device_count()` returning `0` on a CPU-only machine is the correct, expected, real behavior.

**20.** B -- Chapter 10 reuses Chapter 7's own 64-byte CPU alignment finding as a cross-check for its device-aware allocator discussion.

**21.** B -- `use_count()` traces a real `1 -> 2 -> 3 -> 2 -> 1` lifecycle in Chapter 11's worked example.

**22.** B -- The `-DNDEBUG` release build silently writes out of bounds and exits `0`, since `assert()` is compiled out.

**23.** A -- `d(a/b)/da = 1/b = 0.2` and `d(a/b)/db = -a/b^2 = -0.08` at `a=2, b=5`.

**24.** B -- `B.grad()` is `[2,2,2]`, because each element of `B` is reused once per row under broadcasting, and gradients sum across the broadcast dimension.

**25.** B -- `torch::matmul` throws a real `c10::Error` on shape mismatch, an honest divergence from the CUDA C++ edition's own unchecked kernel.

**26.** B -- The mismatch occurs at exactly 4 of the 6 positions, since `.reshape()` on non-contiguous strides silently copies data in flat memory order rather than raising an error.

**27.** B -- `torch::var(x)`'s default applies Bessel's correction (divides by `n-1`), giving `4.57143`, not the population-variance `4.0`.

**28.** A -- Scaling `[3,4]` (norm 5) down to `max_norm=2.0` gives a scale factor of `0.4`, producing `[1.2, 1.6]`, whose norm is exactly `2.0`.

**29.** B -- Every real differentiable op is statically wired at compile time to its own backward `Node` type, with no runtime lookup, confirmed across six real ops.

**30.** B -- A graph node is created whenever ANY operand has `requires_grad=true`, not only when all operands do.

**31.** B -- `torch::autograd::Function<T>` can supply gradients via the implicit function theorem, useful when there is no explicit forward formula to differentiate.

**32.** B -- `retain_graph=true` specifically enables gradient accumulation across multiple `.backward()` calls on overlapping graph structure.

**33.** B -- `create_graph=true` keeps the backward pass itself differentiable, enabling second-order derivatives.

**34.** B -- Aliased/shared tensor references produced by `AddBackward0` during backward are confirmed harmless -- an internal reuse detail, not a bug.

**35.** B -- Chapter 18 tags every claim that genuinely could not be run on real hardware as `[UNVERIFIED -- pending real-GPU test]`.

**36.** B -- The benchmark harness requires warmup runs plus timed runs, averaged, before any performance number is reported.

**37.** B -- Disassembly confirms zero runtime dispatch cost for template-based compile-time specialization -- though the chapter is explicit that other costs (compile time, code size) still apply.

**38.** B -- Loop unrolling and kernel fusion are treated as two distinct, separately-measured optimizations, continuing Chapter 5's own distinction.

**39.** B -- Initial-weight statistics are checked against the theoretical He and Xavier initialization formulas.

**40.** B -- The loss/gradient scale trap is shown to be structurally impossible under LibTorch's real autograd engine.

**41.** B -- Chapter 21 contrasts a hand-rolled format against the real `torch::save`/`torch::load` functions, with concrete differences shown.

**42.** B -- Online softmax is verified against the naive full-softmax computation, for bit/tolerance-level agreement.

**43.** B -- DV01 is computed via autodiff through the pricing formula, cross-checked against a manual bump-and-reprice calculation.

**44.** B -- Hitting the bisection solver's iteration cap before convergence can silently return a plausible-looking but wrong spread.

**45.** B -- A single `torch::matmul()` call is the entire tensor-core-dispatch code a LibTorch programmer ever writes, compared to CUDA's own hand-written WMMA fragment API.

**46.** B -- The restriction against reference captures in extended device lambdas is inconvenient but genuinely prevents a real dangling-reference bug class, demonstrated via a genuine AddressSanitizer catch in ordinary host C++.

**47.** B -- `torch::matmul()` measured roughly 18x faster than a hand-written naive triple-loop matmul at N=256.

**48.** B -- Lambda `sizeof` grows `1, 4, 12` bytes for zero, one, and three captured `int`s, matching an ordinary struct's own layout rules.

**49.** B -- `torch::Tensor`'s own `intrusive_ptr`-based reference counting already is a correctly-implemented Rule of Five/Zero for device memory, for free -- exactly what CUDA's own `GPUPathBuffer` had to hand-write.

**50.** B -- Matrix multiplication is the p=1 special case of tensor contraction: a contraction over exactly one shared axis.

**51.** B -- `torch::tensordot()`/`torch::einsum()` already dispatch to a device-appropriate implementation, so the identical call handles one shared axis or several, on CPU or CUDA, with no `MAX_RANK` constant or metadata struct ever written at the call site.
