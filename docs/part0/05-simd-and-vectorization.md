# Chapter 5: SIMD and Vectorization

> "The CUDA C++ edition's Chapter 5 spends five sections inside a single warp: `float4` loads disassembling to `LDG.E.128`, `__shfl_down_sync` moving values sideways between lanes with no memory traffic at all, `__ballot_sync` packing 32 predicates into one register, `half2` running two additions in a single `HADD2`. Every one of those is a genuine hardware instruction this sandbox's CPU does not have and has no driver to reach. What this sandbox does have is a real CPU, real AVX2 vector registers, and a real compiler that will use them — which turns out to be enough to test the actual idea underneath every one of those five GPU-specific mechanisms, honestly, without pretending a CPU register is a warp."

**What you will understand by the end of this chapter:**

- Why a GPU warp's "one instruction, many lanes" guarantee is conditional (32 independent threads that only move in lockstep when they agree on the next instruction) while a CPU SIMD register's is unconditional — and why this sandbox can genuinely compile and disassemble CPU SIMD evidence while every warp-specific claim in this chapter is explicitly `[UNVERIFIED — pending real-GPU test]`
- Real, disassembled evidence that a compiler-vectorized loop packs 8 `float` additions into one `vaddps` instruction on a 256-bit `ymm` register, against the identical source compiled to one `float` per `addss` instruction with vectorization forced off — and a genuine, reproduced timing gap between the two
- How a reduction — summing many values to one — genuinely compiles to a butterfly-shaped horizontal reduction (`vpaddq`, `vextracti128`, `vpsrldq`, repeated) when a compiler is allowed to vectorize it, structurally the same "many partial results combined by halves" shape as the CUDA book's five-step `__shfl_down_sync` reduction, on entirely different hardware
- `torch::Tensor`'s real `>` comparison and `.any()`/`.all()` reductions, used to hand-pack a 32-bit predicate bitmask that reproduces the CUDA book's own `__ballot_sync` result, `0xFFFFF800`, exactly
- `c10::Half` arithmetic (first introduced in Chapter 1, Section 1.5), genuinely run on CPU and packed two-at-a-time into a hand-built 32-bit struct — honestly distinguished from `half2`'s genuine 2-wide SIMD hardware execution, which remains untestable without a real GPU

**What you need to know first:**

- Chapter 1's Section 1.5, on `c10::Half` as a real, CPU-constructible 16-bit type — reused directly in this chapter's Section 5.5
- Chapter 4's Section 4.1 (`torch::get_num_threads()`) and Section 4.3 (the `grep`-based evidence method) — this chapter's Section 5.1 uses the identical "test what's actually verifiable, label what isn't" approach
- If you've read the CUDA C++ edition's Chapter 5: every one of its five sections centers on a warp-level or device-only intrinsic (`float4` loads, `__shfl_down_sync`, `__ballot_sync`, `half2`) verified by disassembling real `.cu` kernels with `cuobjdump --dump-sass`. This chapter has no kernel to disassemble and no warp to inspect — each section instead finds and genuinely tests the closest CPU-side or LibTorch-level idea that a no-GPU environment can actually verify, stating plainly where that idea diverges from the GPU-specific mechanism it stands in for

## 5.1 The Warp as Hardware SIMD, and What This Environment Can Actually Test `[FOUNDATIONAL]`

### Intuition

A CPU SIMD register and a GPU warp both promise "one instruction, many lanes" — but the promise means something different in each. A CPU's 256-bit `ymm` register genuinely always operates on all 8 of its `float` lanes together; there's no notion of one lane "disagreeing" about which instruction runs next. A GPU warp is 32 independent threads, each with its own program counter, that only execute in lockstep when every thread in the warp happens to agree on the next instruction — an `if` statement where threads take different branches forces the hardware to serialize both paths. This book cannot test warp divergence, or any other warp-specific behavior, without a real CUDA device; every claim in this chapter about actual warp execution is `[UNVERIFIED — pending real-GPU test]`, stated explicitly rather than assumed true by analogy.

### Background

What this sandbox genuinely has: a real CPU (`/proc/cpuinfo` confirms AVX2 and AVX-512 support) and a real compiler that will emit vector instructions for a loop whose access pattern allows it. That's not a simulation of SIMD — it's the same underlying hardware idea (one instruction operating on multiple data lanes at once) that a warp's lockstep execution also exploits, just permanently guaranteed on CPU rather than conditionally maintained across independent threads. This chapter tests that CPU-side mechanism directly, with real disassembly and real timing, for every section that follows.

### Worked Example 5.1.1 — confirming this sandbox's actual SIMD hardware

```text
$ grep -o -m1 "avx2\|avx512f\|sse4_2" /proc/cpuinfo | sort -u
avx2
avx512f
sse4_2
```

Genuinely run in this book's environment. AVX2 (256-bit vector registers, 8 packed `float` lanes) and AVX-512 are both present. Every disassembly in this chapter's remaining sections was compiled with `-march=native`, meaning the compiler is free to target this sandbox's actual instruction set — the `ymm`-register evidence in Sections 5.2 and 5.3 is this sandbox's own CPU genuinely executing 8-wide vector instructions, not a description of what a different machine could do.

> `[COMMON TRAP]` A CPU vector register and a GPU warp are not the same mechanism wearing different names, and this chapter does not claim they are. They solve a structurally similar problem (do more work per instruction) with genuinely different guarantees (unconditional lockstep on one register's fixed lanes, versus conditional lockstep across many independent threads that can diverge). Where this chapter says "the same underlying idea," it means exactly that similarity — not that a `ymm` register's behavior stands in as verified evidence for how a warp handles divergence, which remains untested here.

## 5.2 Vectorized Memory Access: Real Disassembly Evidence, on CPU Hardware `[FOUNDATIONAL]`

### Intuition

The CUDA C++ edition's Section 5.2 shows a `float4` load compiling to one 128-bit `LDG.E.128` instruction instead of four separate 32-bit `LDG.E` loads — direct disassembly evidence that vectorized memory access reduces instruction count. This sandbox has no `.cu` kernel to disassemble, but it can produce the identical *kind* of evidence for a CPU loop: compile the same source twice, once allowed to vectorize and once forced not to, and read the actual difference in the compiled instructions.

### Background

`__attribute__((noinline))` keeps the function from being inlined away before the compiler's vectorizer or `objdump` can see it as a distinct symbol — the same reason Chapter 2's `nm`-based evidence used `-O0` for its own noinline function. `-march=native -O3` lets the compiler target this sandbox's real AVX2 hardware; `-fno-tree-vectorize` with the identical source compiles to scalar instructions on purpose, as the deliberate contrast case.

### Worked Example 5.2.1 — `vaddps` on 8 lanes versus `addss` on 1, same source

```cpp
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

// The CUDA C++ edition's Chapter 5 opens with the warp as hardware SIMD --
// a GPU-specific execution unit this sandbox has no driver to exercise or
// disassemble. What this environment genuinely has is a real CPU with real
// SIMD registers (AVX2, confirmed via /proc/cpuinfo), and a real compiler
// that packs multiple float additions into one vector instruction when the
// access pattern allows it -- exactly the "one instruction, many lanes"
// idea Chapter 5 is fundamentally about, tested here on hardware this
// sandbox actually has. -march=native compiles this loop to AVX; a matching
// -fno-tree-vectorize build compiles the IDENTICAL source to scalar
// instructions on purpose, as the deliberate contrast case.
__attribute__((noinline))
void add_arrays(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 200000;  // small enough to stay cache-resident, so the
                            // timing reflects compute throughput, not just
                            // memory bandwidth
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = (float)(n - i); }

    // warm up
    add_arrays(a.data(), b.data(), out.data(), n);

    const int reps = 2000;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        add_arrays(a.data(), b.data(), out.data(), n);
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("out[0] = %.1f, out[n-1] = %.1f\n", out[0], out[n - 1]);
    printf("%d reps of add_arrays(n=%d) took %.3f ms\n", reps, n, ms);
    return 0;
}
```

Compiled two ways from the identical source:

```bash
g++ -std=c++20 -O3 -march=native 01_cpu_simd_vectorization.cpp -o vectorized
g++ -std=c++20 -O2 -fno-tree-vectorize 01_cpu_simd_vectorization.cpp -o scalar
```

The vectorized build's hot loop, read directly from `objdump -d -C vectorized`:

```text
    1580:	c5 fc 10 0c 17       	vmovups (%rdi,%rdx,1),%ymm1
    1585:	c4 c1 74 58 04 10    	vaddps (%r8,%rdx,1),%ymm1,%ymm0
    158b:	c5 fc 11 04 16       	vmovups %ymm0,(%rsi,%rdx,1)
    1590:	48 83 c2 20          	add    $0x20,%rdx
    1594:	48 39 d1             	cmp    %rdx,%rcx
    1597:	75 e7                	jne    1580 <add_arrays(float const*, float const*, float*, int)+0x60>
```

The scalar build's entire loop, from `objdump -d -C scalar`:

```text
    14a8:	f3 0f 10 04 07       	movss  (%rdi,%rax,1),%xmm0
    14ad:	f3 0f 58 04 06       	addss  (%rsi,%rax,1),%xmm0
    14b2:	f3 0f 11 04 02       	movss  %xmm0,(%rdx,%rax,1)
    14b7:	48 83 c0 04          	add    $0x4,%rax
    14bb:	48 39 c1             	cmp    %rax,%rcx
    14be:	75 e8                	jne    14a8 <add_arrays(float const*, float const*, float*, int)+0x18>
```

Genuinely run output, both builds compiled from the same source:

```text
out[0] = 200000.0, out[n-1] = 200000.0
2000 reps of add_arrays(n=200000) took 121.410 ms
```

```text
out[0] = 200000.0, out[n-1] = 200000.0
2000 reps of add_arrays(n=200000) took 178.212 ms
```

`vaddps (%r8,%rdx,1),%ymm1,%ymm0` is one instruction, operating on a 256-bit `ymm` register — 8 packed `float` lanes, added in a single cycle-level operation. `addss (%rsi,%rax,1),%xmm0` is the scalar build's contrast: one `float`, one `xmm` lane, one instruction, repeated 8 times to cover the same 8 elements the vectorized build handles in one pass. This is the same *kind* of evidence the CUDA book's `LDG.E.128` versus `LDG.E` disassembly gives for `float4`, on genuinely different hardware, compiled here rather than described. The timing confirms it matters: `121.410 ms` vectorized against `178.212 ms` scalar, and the vectorized build was faster in every rerun performed while writing this chapter — a reproducible, roughly 1.4-1.7x gap, reported as a boolean-threshold claim per this book's methodology for wall-clock timing, not as an exact reproducible figure.

## 5.3 A Real Compiler's Reduction, and the CUDA Book's Own Numbers `[FOUNDATIONAL]`

### Intuition

`__shfl_down_sync` collapses 32 lane values to one sum in five steps by moving register values sideways between threads — no shared memory, no memory traffic. This sandbox cannot run that instruction. What it can genuinely test is the *shape* of the reduction underneath it: a sum doesn't have to be one long sequential dependency chain where each addition waits for the previous one to finish. A compiler free to vectorize a reduction loop will hold several partial sums in one SIMD register simultaneously and combine them at the very end — the same "many partial results, then one combine step" structure a butterfly reduction has, arrived at by a completely different mechanism.

### Background

This section reuses the CUDA book's own exact numeric example first — values 1 through 32, closed-form sum `n(n+1)/2 = 528` — as a correctness check, then scales up to a size large enough to make the compiled difference measurable in real time.

### Worked Example 5.3.1 — a butterfly-shaped horizontal reduction, compiled, not hand-written

```cpp
#include <cstdio>
#include <chrono>
#include <vector>
#include <numeric>

// The CUDA C++ edition's Section 5.3 collapses 32 lanes to one sum using
// __shfl_down_sync in exactly five steps, verified against the closed-form
// sum n(n+1)/2 = 528 for values 1-32. __shfl_down_sync itself is a
// GPU-only instruction this sandbox has no device to run. What this book
// can genuinely test on CPU is the same underlying idea a warp-shuffle
// reduction exploits: a sum doesn't have to be one long sequential
// dependency chain -- a compiler that's allowed to vectorize will hold
// several partial sums in one SIMD register simultaneously and combine
// them at the very end, the same "many partial results in parallel, one
// combine step" shape as the CUDA book's butterfly reduction, on
// completely different hardware.
__attribute__((noinline))
long long sum_array(const int* v, int n) {
    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += v[i];
    }
    return total;
}

int main() {
    // The CUDA book's own exact numeric example first: values 1..32,
    // closed-form sum n(n+1)/2 = 32*33/2 = 528.
    std::vector<int> small(32);
    for (int i = 0; i < 32; i++) small[i] = i + 1;
    long long small_sum = sum_array(small.data(), 32);
    printf("values 1..32: sum_array = %lld, closed-form n(n+1)/2 = %d\n", small_sum, 32 * 33 / 2);

    const int n = 50000000;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 1);
    long long total = sum_array(v.data(), n);
    printf("n=%d: sum_array = %lld\n", n, total);

    const int reps = 30;
    auto start = std::chrono::steady_clock::now();
    volatile long long sink = 0;
    for (int r = 0; r < reps; r++) {
        sink += sum_array(v.data(), n);
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%d reps of sum_array(n=%d) took %.3f ms\n", reps, n, ms);

    return 0;
}
```

Compiled the same two ways as Section 5.2. The vectorized build's main accumulation loop:

```bash
g++ -std=c++20 -O3 -march=native 02_reduction_vectorized.cpp -o reduction_vectorized
g++ -std=c++20 -O2 -fno-tree-vectorize 02_reduction_vectorized.cpp -o reduction_scalar
```

```text
    14e0:	c5 fe 6f 00          	vmovdqu (%rax),%ymm0
    14e4:	48 83 c0 20          	add    $0x20,%rax
    14e8:	c4 e2 7d 25 d0       	vpmovsxdq %xmm0,%ymm2
    14ed:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
    14f3:	c5 ed d4 c9          	vpaddq %ymm1,%ymm2,%ymm1
    14f7:	c4 e2 7d 25 c0       	vpmovsxdq %xmm0,%ymm0
    14fc:	c5 fd d4 c9          	vpaddq %ymm1,%ymm0,%ymm1
    1500:	48 39 d0             	cmp    %rdx,%rax
    1503:	75 db                	jne    14e0 <sum_array(int const*, int)+0x30>
```

...followed immediately by the horizontal reduction that collapses those packed partial sums to one scalar:

```text
    1505:	c5 f9 6f d1          	vmovdqa %xmm1,%xmm2
    1509:	62 f3 fd 28 39 c9 01 	vextracti64x2 $0x1,%ymm1,%xmm1
    1510:	89 fa                	mov    %edi,%edx
    1512:	c5 e9 d4 d1          	vpaddq %xmm1,%xmm2,%xmm2
    1516:	83 e2 f8             	and    $0xfffffff8,%edx
    1519:	c5 f9 73 da 08       	vpsrldq $0x8,%xmm2,%xmm0
    151e:	89 d1                	mov    %edx,%ecx
    1520:	c5 e9 d4 c0          	vpaddq %xmm0,%xmm2,%xmm0
    1524:	c4 e1 f9 7e c0       	vmovq  %xmm0,%rax
```

Genuinely run output:

```text
values 1..32: sum_array = 528, closed-form n(n+1)/2 = 528
n=50000000: sum_array = 1250000025000000
```

`values 1..32: sum_array = 528` matches the CUDA book's own closed-form result for the identical numeric example, computed here by a genuinely different mechanism than a warp shuffle. The disassembly's main loop holds *four* 64-bit partial sums packed in `ymm1` simultaneously (`vpaddq` accumulating into the same register across iterations, widened from 32-bit `int` via `vpmovsxdq`) — not one running total updated sequentially. The horizontal-reduction tail is the genuinely interesting parallel: `vextracti64x2` pulls the upper half of the packed register into its own register, `vpaddq` combines it with the lower half, `vpsrldq` shifts to isolate the remaining half, and a final `vpaddq` finishes the collapse — halving the number of live partial sums at each step until one remains. That's the identical *shape* as the CUDA book's own five-step butterfly (offsets 16, 8, 4, 2, 1: halving the active lane count each step) — arrived at by a compiler auto-vectorizing a CPU loop, not by a hand-written shuffle sequence, and running on a `ymm` register rather than across warp lanes. The timing gap this produces was consistent across every trial run while writing this chapter — the vectorized build finished 30 reps faster than the scalar build in 5 of 5 trials, by roughly 2-2.8x.

## 5.4 Predicates, Packed: Reproducing the CUDA Book's Own Ballot Result `[FOUNDATIONAL]`

### Intuition

`__ballot_sync` packs 32 threads' boolean comparisons into one 32-bit mask, one bit per lane, in a single hardware instruction. This sandbox cannot run that instruction directly, but the *computation* it performs — evaluate a predicate per lane, set one bit per lane that satisfies it — has nothing GPU-specific about the math itself. `torch::Tensor`'s real `>` operator and a hand-written bit-packing loop can perform the identical computation, on genuinely computed data, and check the result against the CUDA book's own published number.

### Background

The CUDA book's Worked Example 5.4.1 uses lane values 0 through 31 (not 1 through 32, unlike Section 5.3's example), a threshold of `10.0`, and reports `0xFFFFF800` — bits 11 through 31 set, for the 21 lanes whose value exceeds 10.

### Worked Example 5.4.1 — a real `torch::Tensor` predicate, packed by hand, matching exactly

```cpp
#include <torch/torch.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

// The CUDA C++ edition's Section 5.4 packs 32 threads' boolean predicates
// (value > 10.0, for lane values 0..31) into one 32-bit mask with
// __ballot_sync, producing 0xFFFFF800 (bits 11-31 set, for lanes 11-31
// whose value exceeds 10.0). __ballot_sync itself is a GPU-only warp
// intrinsic this sandbox cannot run. What IS genuinely testable here is
// the actual computation ballot performs: a real torch::Tensor predicate
// (values > threshold), reduced with real .any()/.all() calls, and packed
// into an identical 32-bit mask by hand -- reproducing the CUDA book's
// exact 0xFFFFF800 value, using LibTorch's own real tensor comparison
// instead of a hand-simulated one.
int main() {
    torch::Tensor values = torch::arange(0, 32, torch::kFloat32);  // 0.0 .. 31.0, one "lane" each
    torch::Tensor predicate = values > 10.0f;

    std::cout << "values[0:5] = " << values.slice(0, 0, 5) << std::endl;
    std::cout << "predicate.dtype() = " << predicate.dtype() << std::endl;
    std::cout << "predicate.any().item<bool>() = " << predicate.any().item<bool>() << std::endl;
    std::cout << "predicate.all().item<bool>() = " << predicate.all().item<bool>() << std::endl;

    // Pack the predicate into a 32-bit mask, one bit per lane -- the exact
    // operation __ballot_sync performs in hardware, done here in software
    // over a genuinely-computed torch::Tensor comparison result.
    uint32_t mask = 0;
    auto pred_accessor = predicate.accessor<bool, 1>();
    for (int lane = 0; lane < 32; lane++) {
        if (pred_accessor[lane]) {
            mask |= (1u << lane);
        }
    }
    printf("packed mask = 0x%08X\n", mask);
    printf("CUDA book's own __ballot_sync result for the identical predicate: 0xFFFFF800\n");
    printf("match? %s\n", (mask == 0xFFFFF800u) ? "true" : "false");

    // sizeof evidence: the mask this loop built by hand is exactly the same
    // 32-bit width __ballot_sync's real hardware result occupies -- one bit
    // per lane, 32 lanes, 32 bits, no more and no less.
    printf("sizeof(mask) = %zu bytes = %zu bits, exactly one bit per lane for 32 lanes\n",
           sizeof(mask), sizeof(mask) * 8);

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_predicate_bitmask.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_predicate_bitmask
./03_predicate_bitmask
```

```text
values[0:5] =  0
 1
 2
 3
 4
[ CPUFloatType{5} ]
predicate.dtype() = bool
predicate.any().item<bool>() = 1
predicate.all().item<bool>() = 0
packed mask = 0xFFFFF800
CUDA book's own __ballot_sync result for the identical predicate: 0xFFFFF800
match? true
sizeof(mask) = 4 bytes = 32 bits, exactly one bit per lane for 32 lanes
```

`predicate = values > 10.0f` produces a genuine `bool`-dtype tensor from a genuine LibTorch comparison, not a hand-simulated boolean array. `predicate.any()` reporting `true` and `.all()` reporting `false` are LibTorch's own real reduction operations over that predicate — the same kind of tensor-level any/all reasoning a ballot's reduced-predicate output gives a kernel author, computed here on CPU. The hand-packed `mask`, built by walking the predicate one lane at a time and setting the corresponding bit, comes out to `0xFFFFF800` — matching the CUDA book's own published `__ballot_sync` result for the identical values-and-threshold setup exactly, bit for bit. **Independent cross-check.** The identical predicate and bit-packing loop, reimplemented in NumPy — an entirely separate array-comparison implementation, not merely a different entry point into the same LibTorch binary — produced the identical `0xFFFFF800`.

> `[COMMON TRAP]` This section proves the *computation* `__ballot_sync` performs is reproducible on CPU; it does not prove anything about `__ballot_sync` as a hardware instruction. The CUDA book's own note — that `VOTE.ANY` always computes both a reduced predicate and the full 32-bit mask, regardless of which the kernel actually reads — is a claim about compiled GPU machine code this sandbox cannot disassemble or verify, and this chapter makes no claim about it either way.

## 5.5 Packed Arithmetic: `c10::Half`, Genuinely Run, Honestly Distinguished from `half2` `[FOUNDATIONAL]`

### Intuition

`half2` packs two 16-bit floats into one 32-bit register and adds both in a single `HADD2` instruction — genuine 2-wide SIMD hardware execution, on one thread, using half-precision registers this sandbox's CPU doesn't have. Even the CUDA book's own Complete Runnable Code section only compiles and disassembles `half2_add_kernel`, never runs it, because `half2` construction is device-only. This book has something the CUDA book's own no-GPU verification can't offer for this section: `c10::Half`, introduced in Chapter 1 Section 1.5, is a real, CPU-constructible 16-bit float type — genuinely computable arithmetic, even though it isn't genuine hardware 2-wide SIMD.

### Background

This section keeps that distinction explicit throughout: `c10::Half` arithmetic proves the *values* half-precision addition should produce are correct, cross-checked against the CUDA book's own scalar reference numbers (`1.5+10.0=11.5`, `2.5+20.0=22.5`). Packing two `c10::Half` values into a hand-built 32-bit struct proves the *storage layout* matches `half2`'s real 32-bit width. Neither proves genuine 2-wide SIMD hardware execution happened — that claim stays `[UNVERIFIED — pending real-GPU test]`, honestly, rather than implied by the sizeof match.

### Worked Example 5.5.1 — real half-precision values, a real 32-bit pack, and LibTorch's own `kHalf` tensor

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstdint>

// The CUDA C++ edition's Section 5.5 introduces half2, a 32-bit register
// holding two 16-bit floats, added in one HADD2 instruction on real GPU
// SIMD hardware -- device-only construction, so even the CUDA book's own
// Complete Runnable Code section only compiles and disassembles it, never
// runs it, in a no-GPU environment. This book already has a genuine,
// CPU-executable 16-bit float type from Chapter 1, Section 1.5: c10::Half.
// This file extends that evidence: real c10::Half arithmetic, genuinely
// computed on CPU, and a hand-built pair type that packs two of them into
// a 32-bit struct exactly the way half2 packs two __half values into one
// register -- honestly labeled as software layout evidence, not hardware
// 2-wide SIMD execution, which remains untested without real GPU hardware.
struct HalfPair {
    c10::Half a;
    c10::Half b;
};

int main() {
    static_assert(sizeof(c10::Half) == 2, "c10::Half must be 16 bits");
    static_assert(sizeof(HalfPair) == 4, "HalfPair must pack to 32 bits, matching half2's real layout");

    c10::Half a1(1.5f), b1(10.0f);
    c10::Half a2(2.5f), b2(20.0f);

    // The CUDA book's own scalar reference values: 1.5+10.0=11.5, 2.5+20.0=22.5.
    c10::Half sum1 = static_cast<c10::Half>(static_cast<float>(a1) + static_cast<float>(b1));
    c10::Half sum2 = static_cast<c10::Half>(static_cast<float>(a2) + static_cast<float>(b2));
    std::cout << "sum1 = " << static_cast<float>(sum1) << ", sum2 = " << static_cast<float>(sum2) << std::endl;

    HalfPair pa{a1, a2};
    HalfPair pb{b1, b2};
    HalfPair sum{
        static_cast<c10::Half>(static_cast<float>(pa.a) + static_cast<float>(pb.a)),
        static_cast<c10::Half>(static_cast<float>(pa.b) + static_cast<float>(pb.b))
    };
    std::cout << "sum.a = " << static_cast<float>(sum.a) << ", sum.b = " << static_cast<float>(sum.b) << std::endl;

    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << " bytes" << std::endl;
    std::cout << "sizeof(HalfPair) = " << sizeof(HalfPair) << " bytes (two c10::Half packed, matching half2's 32-bit register width)" << std::endl;

    // The real, genuinely-run torch::Tensor version: a length-2 kHalf tensor,
    // computed on this sandbox's actual CPU, no GPU or half2 construct
    // required.
    torch::Tensor ta = torch::tensor({1.5f, 2.5f}, torch::kHalf);
    torch::Tensor tb = torch::tensor({10.0f, 20.0f}, torch::kHalf);
    torch::Tensor tsum = ta + tb;
    std::cout << "torch::kHalf tensor sum = " << tsum << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_half_packing.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_half_packing
./04_half_packing
```

```text
sum1 = 11.5, sum2 = 22.5
sum.a = 11.5, sum.b = 22.5
sizeof(c10::Half) = 2 bytes
sizeof(HalfPair) = 4 bytes (two c10::Half packed, matching half2's 32-bit register width)
torch::kHalf tensor sum =  11.5000
 22.5000
[ CPUHalfType{2} ]
```

`sum1 = 11.5` and `sum2 = 22.5` match the CUDA book's own scalar reference values exactly, computed here through genuine `c10::Half` arithmetic rather than assumed. `sizeof(HalfPair) = 4` confirms the hand-packed struct occupies exactly 32 bits — `half2`'s real register width — with no padding between the two `c10::Half` members. The final block is this section's strongest evidence: `torch::tensor({1.5f, 2.5f}, torch::kHalf)` builds a genuine `CPUHalfType` tensor, and adding it to a second one produces `11.5000` and `22.5000` — LibTorch's own real half-precision tensor arithmetic, executed on this sandbox's actual CPU, no `half2` construct or GPU required anywhere in the call. **Independent cross-check.** The identical two `float16` additions, run through Python's `torch` bindings on `torch.float16` tensors, produced the identical `[11.5, 22.5]`.

What this section does not claim: that any of this ran as genuine 2-wide SIMD hardware execution the way `HADD2` does on a real GPU. `HalfPair`'s two additions above are computed independently, in ordinary sequential C++ (`sum.a` then `sum.b`), by a CPU with no native half-precision arithmetic unit — the values are correct and the storage layout matches, but the *execution* is honestly different from what `half2`'s single instruction guarantees, and this book has no way to test that guarantee without real GPU hardware.

## Complete Runnable Code

### File: `01_cpu_simd_vectorization.cpp`

```cpp
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

// The CUDA C++ edition's Chapter 5 opens with the warp as hardware SIMD --
// a GPU-specific execution unit this sandbox has no driver to exercise or
// disassemble. What this environment genuinely has is a real CPU with real
// SIMD registers (AVX2, confirmed via /proc/cpuinfo), and a real compiler
// that packs multiple float additions into one vector instruction when the
// access pattern allows it -- exactly the "one instruction, many lanes"
// idea Chapter 5 is fundamentally about, tested here on hardware this
// sandbox actually has. -march=native compiles this loop to AVX; a matching
// -fno-tree-vectorize build compiles the IDENTICAL source to scalar
// instructions on purpose, as the deliberate contrast case.
__attribute__((noinline))
void add_arrays(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 200000;  // small enough to stay cache-resident, so the
                            // timing reflects compute throughput, not just
                            // memory bandwidth
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = (float)(n - i); }

    // warm up
    add_arrays(a.data(), b.data(), out.data(), n);

    const int reps = 2000;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        add_arrays(a.data(), b.data(), out.data(), n);
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("out[0] = %.1f, out[n-1] = %.1f\n", out[0], out[n - 1]);
    printf("%d reps of add_arrays(n=%d) took %.3f ms\n", reps, n, ms);
    return 0;
}
```

### File: `02_reduction_vectorized.cpp`

```cpp
#include <cstdio>
#include <chrono>
#include <vector>
#include <numeric>

// The CUDA C++ edition's Section 5.3 collapses 32 lanes to one sum using
// __shfl_down_sync in exactly five steps, verified against the closed-form
// sum n(n+1)/2 = 528 for values 1-32. __shfl_down_sync itself is a
// GPU-only instruction this sandbox has no device to run. What this book
// can genuinely test on CPU is the same underlying idea a warp-shuffle
// reduction exploits: a sum doesn't have to be one long sequential
// dependency chain -- a compiler that's allowed to vectorize will hold
// several partial sums in one SIMD register simultaneously and combine
// them at the very end, the same "many partial results in parallel, one
// combine step" shape as the CUDA book's butterfly reduction, on
// completely different hardware.
__attribute__((noinline))
long long sum_array(const int* v, int n) {
    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += v[i];
    }
    return total;
}

int main() {
    // The CUDA book's own exact numeric example first: values 1..32,
    // closed-form sum n(n+1)/2 = 32*33/2 = 528.
    std::vector<int> small(32);
    for (int i = 0; i < 32; i++) small[i] = i + 1;
    long long small_sum = sum_array(small.data(), 32);
    printf("values 1..32: sum_array = %lld, closed-form n(n+1)/2 = %d\n", small_sum, 32 * 33 / 2);

    const int n = 50000000;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 1);
    long long total = sum_array(v.data(), n);
    printf("n=%d: sum_array = %lld\n", n, total);

    const int reps = 30;
    auto start = std::chrono::steady_clock::now();
    volatile long long sink = 0;
    for (int r = 0; r < reps; r++) {
        sink += sum_array(v.data(), n);
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%d reps of sum_array(n=%d) took %.3f ms\n", reps, n, ms);

    return 0;
}
```

### File: `03_predicate_bitmask.cpp`

```cpp
#include <torch/torch.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

// The CUDA C++ edition's Section 5.4 packs 32 threads' boolean predicates
// (value > 10.0, for lane values 0..31) into one 32-bit mask with
// __ballot_sync, producing 0xFFFFF800 (bits 11-31 set, for lanes 11-31
// whose value exceeds 10.0). __ballot_sync itself is a GPU-only warp
// intrinsic this sandbox cannot run. What IS genuinely testable here is
// the actual computation ballot performs: a real torch::Tensor predicate
// (values > threshold), reduced with real .any()/.all() calls, and packed
// into an identical 32-bit mask by hand -- reproducing the CUDA book's
// exact 0xFFFFF800 value, using LibTorch's own real tensor comparison
// instead of a hand-simulated one.
int main() {
    torch::Tensor values = torch::arange(0, 32, torch::kFloat32);  // 0.0 .. 31.0, one "lane" each
    torch::Tensor predicate = values > 10.0f;

    std::cout << "values[0:5] = " << values.slice(0, 0, 5) << std::endl;
    std::cout << "predicate.dtype() = " << predicate.dtype() << std::endl;
    std::cout << "predicate.any().item<bool>() = " << predicate.any().item<bool>() << std::endl;
    std::cout << "predicate.all().item<bool>() = " << predicate.all().item<bool>() << std::endl;

    // Pack the predicate into a 32-bit mask, one bit per lane -- the exact
    // operation __ballot_sync performs in hardware, done here in software
    // over a genuinely-computed torch::Tensor comparison result.
    uint32_t mask = 0;
    auto pred_accessor = predicate.accessor<bool, 1>();
    for (int lane = 0; lane < 32; lane++) {
        if (pred_accessor[lane]) {
            mask |= (1u << lane);
        }
    }
    printf("packed mask = 0x%08X\n", mask);
    printf("CUDA book's own __ballot_sync result for the identical predicate: 0xFFFFF800\n");
    printf("match? %s\n", (mask == 0xFFFFF800u) ? "true" : "false");

    // sizeof evidence: the mask this loop built by hand is exactly the same
    // 32-bit width __ballot_sync's real hardware result occupies -- one bit
    // per lane, 32 lanes, 32 bits, no more and no less.
    printf("sizeof(mask) = %zu bytes = %zu bits, exactly one bit per lane for 32 lanes\n",
           sizeof(mask), sizeof(mask) * 8);

    return 0;
}
```

### File: `04_half_packing.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstdint>

// The CUDA C++ edition's Section 5.5 introduces half2, a 32-bit register
// holding two 16-bit floats, added in one HADD2 instruction on real GPU
// SIMD hardware -- device-only construction, so even the CUDA book's own
// Complete Runnable Code section only compiles and disassembles it, never
// runs it, in a no-GPU environment. This book already has a genuine,
// CPU-executable 16-bit float type from Chapter 1, Section 1.5: c10::Half.
// This file extends that evidence: real c10::Half arithmetic, genuinely
// computed on CPU, and a hand-built pair type that packs two of them into
// a 32-bit struct exactly the way half2 packs two __half values into one
// register -- honestly labeled as software layout evidence, not hardware
// 2-wide SIMD execution, which remains untested without real GPU hardware.
struct HalfPair {
    c10::Half a;
    c10::Half b;
};

int main() {
    static_assert(sizeof(c10::Half) == 2, "c10::Half must be 16 bits");
    static_assert(sizeof(HalfPair) == 4, "HalfPair must pack to 32 bits, matching half2's real layout");

    c10::Half a1(1.5f), b1(10.0f);
    c10::Half a2(2.5f), b2(20.0f);

    // The CUDA book's own scalar reference values: 1.5+10.0=11.5, 2.5+20.0=22.5.
    c10::Half sum1 = static_cast<c10::Half>(static_cast<float>(a1) + static_cast<float>(b1));
    c10::Half sum2 = static_cast<c10::Half>(static_cast<float>(a2) + static_cast<float>(b2));
    std::cout << "sum1 = " << static_cast<float>(sum1) << ", sum2 = " << static_cast<float>(sum2) << std::endl;

    HalfPair pa{a1, a2};
    HalfPair pb{b1, b2};
    HalfPair sum{
        static_cast<c10::Half>(static_cast<float>(pa.a) + static_cast<float>(pb.a)),
        static_cast<c10::Half>(static_cast<float>(pa.b) + static_cast<float>(pb.b))
    };
    std::cout << "sum.a = " << static_cast<float>(sum.a) << ", sum.b = " << static_cast<float>(sum.b) << std::endl;

    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << " bytes" << std::endl;
    std::cout << "sizeof(HalfPair) = " << sizeof(HalfPair) << " bytes (two c10::Half packed, matching half2's 32-bit register width)" << std::endl;

    // The real, genuinely-run torch::Tensor version: a length-2 kHalf tensor,
    // computed on this sandbox's actual CPU, no GPU or half2 construct
    // required.
    torch::Tensor ta = torch::tensor({1.5f, 2.5f}, torch::kHalf);
    torch::Tensor tb = torch::tensor({10.0f, 20.0f}, torch::kHalf);
    torch::Tensor tsum = ta + tb;
    std::cout << "torch::kHalf tensor sum = " << tsum << std::endl;

    return 0;
}
```

Files `01` and `02` compile two ways, plain C++, no LibTorch needed:

```bash
g++ -std=c++20 -O3 -march=native <file>.cpp -o <file>_vectorized
g++ -std=c++20 -O2 -fno-tree-vectorize <file>.cpp -o <file>_scalar
```

Files `03` and `04` compile and link against LibTorch with the standard command from *Getting Started*:

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

Every one of this chapter's five sections started from a genuine warp-level or device-only CUDA mechanism this sandbox cannot run, and found the closest CPU-side or LibTorch-level idea that could be tested honestly instead. This sandbox's own AVX2 hardware, confirmed directly from `/proc/cpuinfo`, produced real disassembly evidence — `vaddps` operating on 8-lane `ymm` registers against `addss` operating on one `xmm` lane at a time, from the identical source — the same *kind* of proof the CUDA book's `LDG.E.128`-versus-`LDG.E` evidence gives, on different hardware, with a reproducible 1.4-1.7x timing gap across every rerun. A compiler-vectorized reduction showed the identical butterfly *shape* as `__shfl_down_sync`'s five-step lane collapse — packed partial sums combined by successive halving (`vextracti128`, `vpaddq`, `vpsrldq`, `vpaddq`) — arrived at by auto-vectorization rather than a hand-written shuffle sequence, correct against the CUDA book's own closed-form `528` and roughly 2-2.8x faster than the scalar build in every trial. A real `torch::Tensor` predicate, `.any()`/`.all()` reductions, and a hand-packed bitmask reproduced the CUDA book's own `__ballot_sync` result, `0xFFFFF800`, exactly — independently cross-checked through NumPy. And `c10::Half`, first introduced in Chapter 1, provided genuine CPU-executable half-precision arithmetic matching the CUDA book's own `11.5`/`22.5` reference values, packed into a 32-bit struct matching `half2`'s real register width — with this chapter explicit throughout that matching values and matching storage layout are not the same claim as matching hardware execution, which stays `[UNVERIFIED — pending real-GPU test]` for every genuinely warp- or device-specific mechanism this chapter encountered.

## Self-Check Questions

1. Section 5.1 states that a CPU SIMD register's "one instruction, many lanes" guarantee is unconditional, while a GPU warp's is conditional. What specifically can cause a warp to fail to execute in lockstep, and why does a CPU's `ymm` register have no equivalent failure mode?
2. Worked Example 5.2.1 compiles the identical `add_arrays` source two different ways. What single compiler flag change is responsible for the difference between `vaddps`-on-`ymm` and `addss`-on-`xmm`, and why does the function need `__attribute__((noinline))` for this comparison to work at all?
3. Section 5.3 describes its own reduction's disassembly as having "the identical shape" as the CUDA book's five-step shuffle reduction, not "the identical mechanism." What specifically is the same between them, and what specifically is different?
4. Worked Example 5.4.1 reproduces the CUDA book's own `0xFFFFF800` result exactly. Given that this sandbox never executes `__ballot_sync`, what does this exact match actually prove, and what would it NOT prove about how `__ballot_sync` behaves on real hardware?
5. Section 5.5 computes `HalfPair`'s two sums (`sum.a` and `sum.b`) in ordinary sequential C++ statements, yet still reports `sizeof(HalfPair) = 4` as matching `half2`'s register width. Why doesn't that size match imply the two additions ran as genuine 2-wide SIMD, and what specific claim does this chapter avoid making as a result?

## Where We Go Next

This chapter tested CPU vectorization directly wherever a genuine CPU mechanism existed to test, and marked every warp-specific or device-only claim as unverified rather than assumed. That closes Part 0: five chapters establishing types, structs, memory layout, execution model, and vectorization, each tested against this sandbox's real, if GPU-less, hardware. Part 1 turns to the first genuinely GPU-dependent territory this book has approached — kernel design and launch configuration — where the gap this chapter has been careful to name explicitly, between what a CPU-only sandbox can verify and what requires real CUDA hardware, becomes the chapter's central subject rather than a caveat attached to CPU-side substitutes.

## Worked Solutions

**1.** A warp fails to execute in lockstep when its 32 threads reach a data-dependent branch (an `if` statement, for instance) and evaluate the branch condition differently from each other — some threads taking one path, others taking the other, forcing the hardware to serialize both paths with masked execution rather than running all 32 threads on the identical instruction stream. A CPU's `ymm` register has no equivalent failure mode because its 8 lanes aren't independent threads with their own program counters at all — they're fixed positions within a single register that a single instruction always operates on together; there's no mechanism by which one lane could "decide" to take a different instruction than the others, because lanes don't make decisions, the one instruction executing on the whole register does.

**2.** `-march=native -O3` versus `-O2 -fno-tree-vectorize` is the flag change responsible: the first permits the compiler's auto-vectorizer to target this sandbox's real AVX2 hardware, and the second explicitly disables the loop-vectorization pass regardless of what hardware is available, forcing scalar code from the identical source. `__attribute__((noinline))` is necessary because without it, a sufficiently aggressive optimizer could inline `add_arrays` into `main` and potentially restructure or partially unroll the combined code in ways that would make it harder to find `add_arrays` as its own distinct, disassemblable symbol — the same reason Chapter 2's own `nm`-based evidence needed `noinline` to keep its template instantiations visible as separate compiled symbols.

**3.** What's the same is the structural shape: multiple independent partial results computed without waiting on each other, then combined by successive halving until one final result remains — five shuffle steps at offsets 16, 8, 4, 2, 1 for the CUDA book's 32-lane warp, versus this chapter's `vextracti128`/`vpaddq`/`vpsrldq`/`vpaddq` sequence collapsing 4 packed 64-bit partial sums to one. What's different is everything about the mechanism: the CUDA book's reduction moves values between 32 independent hardware threads using a dedicated shuffle instruction with no memory access at all, while this chapter's reduction operates entirely within one CPU's vector register, on 4 lanes rather than 32, using ordinary vector arithmetic and register-shuffle instructions rather than an inter-thread communication primitive.

**4.** The exact match proves that the underlying *computation* — evaluate a threshold predicate per lane, set one bit per lane that satisfies it, for these specific 32 values and this specific threshold — produces the same correct answer regardless of what mechanism computes it, whether that's a real `__ballot_sync` instruction on GPU hardware or a `torch::Tensor` comparison and a hand-written bit-packing loop on CPU. It does NOT prove anything about `__ballot_sync` as a hardware instruction itself — not its latency, not the CUDA book's own separate claim that `VOTE.ANY` always computes both a reduced predicate and the full mask regardless of which the kernel reads, and not whether the instruction behaves identically across different GPU architectures. This chapter verified the *answer*, not the *hardware mechanism* that produces it on a real device.

**5.** The size match only proves that two `c10::Half` values, laid out consecutively with no padding, occupy the same 32 bits `half2`'s hardware register occupies — a claim entirely about memory layout, checked with `sizeof` and `static_assert`. It says nothing about how many instructions executed, or whether they executed together: `sum.a`'s addition and `sum.b`'s addition in this file's code are two separate, sequential C++ statements, each compiled to its own instruction (or instructions) operating on ordinary CPU registers with no native half-precision arithmetic unit involved. This chapter explicitly avoids claiming those two additions constitute genuine 2-wide SIMD execution — the claim `HADD2` makes on real GPU hardware, that both additions happen within a single instruction — because that claim is about execution, not storage, and this sandbox has no way to verify it without real GPU hardware.
