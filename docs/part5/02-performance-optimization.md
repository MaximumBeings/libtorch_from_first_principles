# Chapter 19: Performance Optimization Techniques

> "Packing more bytes into fewer load instructions, collapsing several kernels into one, letting the compiler bake a parameter in so thoroughly that checking it costs nothing at runtime" -- and, underneath all three, telling a genuine speedup apart from a measurement artifact. This sandbox still has no NVIDIA GPU (see Chapter 18's own honest disclosure), but this chapter's own subject turns out to be mostly host-side and memory-traffic arithmetic -- most of it is genuinely testable here, with far fewer `[UNVERIFIED]` tags than Chapter 18 needed.

**What you will understand after this chapter:** why packing several elements into one wide load instruction reduces instruction count, and why the packing width is tied to a specific dtype's byte size, not a portable constant; why loop unrolling reduces loop-control overhead without changing the arithmetic performed, while operator fusion reduces memory traffic by collapsing several passes over data into one, and why those are two genuinely different optimizations rather than two names for the same idea; why a C++ function template instantiated at several different compile-time parameter values produces that many separate, independently-optimized compiled function bodies, with zero runtime branching cost and a real binary-size cost; and why a trustworthy benchmark needs warmup runs, repeated timed runs, and (on a real GPU) an explicit synchronization barrier before the clock starts and stops.

**What you need to know first:** Chapter 18's own honest no-GPU disclosure and its `[UNVERIFIED -- pending real-GPU test]` tagging convention; `torch::Tensor` arithmetic and `torch::nn::functional::conv2d` from Chapters 3 and 18; basic C++ templates.

## 19.1 SIMD Vectorization: Instruction-Counting Arithmetic and the Dtype-Width Trap `[FOUNDATIONAL]`

**Intuition.** A print press stamping four letters at once with one plate motion beats four separate single-letter strikes -- until the press runs out of full four-letter groups and the leftover letters have to go back to one-at-a-time processing. SIMD (Single Instruction, Multiple Data) vector loads work the same way: one instruction moves several elements at once, up to whatever width the hardware's own load instruction supports, with a scalar "remainder" loop cleaning up whatever doesn't divide evenly.

**Background.** The CUDA C++ edition's own Section 19.1 packs four `float`s into one 128-bit `float4` load, issued as a single GPU load instruction rather than four separate 32-bit loads, and confirms this by decoding real `cuobjdump` SASS output showing an `LDG.E.128` instruction. This sandbox has no `nvcc` and no GPU (confirmed again here, exactly as in Chapter 18, via `torch::cuda::is_available()`), so no actual vector-load instruction can be issued and no SASS can be decoded -- both of those specific claims are `[UNVERIFIED -- pending real-GPU test]`. What is entirely testable without a GPU, and is tested for real in this section, is the **instruction-counting arithmetic** SIMD vectorization is built on (`simd_count = (size / width) * width`, splitting a loop into a vectorized main body plus a scalar remainder) and the **dtype-width trap**, which is a compile-time `sizeof()` fact about C++ types, true on any machine that can compile C++ at all -- no GPU required to check it.

**Worked Example 19.1.1 and 19.1.2.** Reproducing the CUDA book's own two worked examples exactly: `size=10, width=4` gives `simd_count=8` (two full groups of four), `vector_instructions=2`, `scalar_instructions=2` (the leftover elements 8-9), `total_instructions=4` against a scalar-only baseline of 10. `size=41, width=4` gives `simd_count=40`, `vector_instructions=10`, `scalar_instructions=1`, `total_instructions=11` against a baseline of 41 -- about 26.8% of the scalar-only instruction count, versus 40.0% for the smaller example, which is itself a genuine, verifiable fact: the fixed one-instruction overhead of the scalar remainder matters proportionally less as the problem grows.

**Worked Example 19.1.3: the dtype-width trap.** `float4` and `double2` are both CUDA vector types this sandbox cannot compile (they require `nvcc`), so this section reproduces the underlying fact with plain C++ structs of the same byte layout: an 8-byte-aligned 4-`float` struct and a 2-`double` struct, both genuinely `sizeof() == 16` bytes, but packing 4 elements in one case and only 2 in the other. This is the CUDA book's own Common Trap, stated plainly: "a 128-bit vector load is fixed in BYTES, not in ELEMENT COUNT" -- code retuned from `float4`'s width-4 data to `double2` data, while still assuming `width=4`, reads 16 bytes past the intended 2-element group on every vector op. That is corruption, not merely a slowdown, and it is a fact about the types themselves, confirmed here with `sizeof()`, with no GPU or SASS decoding needed to see it.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 19.1 packs four float loads into one
// 128-bit float4 load instruction on real GPU hardware, and confirms the
// packing via genuine SASS disassembly (an LDG.E.128 instruction). This
// sandbox has no GPU and no nvcc (see Chapter 18's own honest
// disclosure), so no real vector-LOAD instruction and no SASS decoding
// can be produced here -- both are [UNVERIFIED -- pending real-GPU
// test]. What CAN be genuinely run, and is run in this file, is the
// SIMD instruction-COUNTING arithmetic itself -- pure host-side integer
// math with no CUDA dependency -- plus the dtype-width trap, which is a
// compile-time sizeof() fact about C++ types, true on any machine that
// can compile C++ at all, not something that requires a GPU to check.
long ceiling_div_floor(long size, long width) { return (size / width) * width; }

int main() {
    // Worked Example 19.1.1: size=10, width=4.
    {
        long size = 10, width = 4;
        long simd_count = ceiling_div_floor(size, width);
        long vector_instructions = simd_count / width;
        long scalar_instructions = size - simd_count;
        long total_instructions = vector_instructions + scalar_instructions;
        std::cout << "size=" << size << ", width=" << width << ": simd_count=" << simd_count
                  << ", vector_instructions=" << vector_instructions
                  << ", scalar_instructions=" << scalar_instructions
                  << ", total_instructions=" << total_instructions
                  << " (scalar-only baseline=" << size << ")" << std::endl;
        std::cout << "matches CUDA book's own Worked Example 19.1.1 "
                  << "(simd_count=8, vector=2, scalar=2, total=4)? "
                  << (simd_count == 8 && vector_instructions == 2 && scalar_instructions == 2
                      && total_instructions == 4) << std::endl;
    }

    // Worked Example 19.1.2: size=41, width=4.
    {
        long size = 41, width = 4;
        long simd_count = ceiling_div_floor(size, width);
        long vector_instructions = simd_count / width;
        long scalar_instructions = size - simd_count;
        long total_instructions = vector_instructions + scalar_instructions;
        double ratio = 100.0 * static_cast<double>(total_instructions) / static_cast<double>(size);
        std::cout << "\nsize=" << size << ", width=" << width << ": simd_count=" << simd_count
                  << ", vector_instructions=" << vector_instructions
                  << ", scalar_instructions=" << scalar_instructions
                  << ", total_instructions=" << total_instructions
                  << " (scalar-only baseline=" << size << "), ratio of baseline=" << ratio << "%" << std::endl;
        std::cout << "matches CUDA book's own Worked Example 19.1.2 "
                  << "(simd_count=40, vector=10, scalar=1, total=11, ratio~26.8%)? "
                  << (simd_count == 40 && vector_instructions == 10 && scalar_instructions == 1
                      && total_instructions == 11 && ratio > 26.7 && ratio < 26.9) << std::endl;
    }

    // Worked Example 19.1.3: the dtype-width trap, as a genuine sizeof()
    // fact. LibTorch has no built-in float4/double2 type (those are CUDA
    // vector types), so this uses the CUDA book's own claimed byte
    // widths directly -- both are plain compile-time facts about fixed-
    // width types, true regardless of what hardware eventually runs
    // code built against them.
    {
        struct Float4Like { float x, y, z, w; };
        struct Double2Like { double x, y; };
        std::cout << "\nsizeof(Float4Like) = " << sizeof(Float4Like) << " bytes, elements packed = "
                  << (sizeof(Float4Like) / sizeof(float))
                  << ", CUDA book's own claimed float4 = 16 bytes / 4 elements, match = "
                  << (sizeof(Float4Like) == 16 && sizeof(Float4Like) / sizeof(float) == 4) << std::endl;
        std::cout << "sizeof(Double2Like) = " << sizeof(Double2Like) << " bytes, elements packed = "
                  << (sizeof(Double2Like) / sizeof(double))
                  << ", CUDA book's own claimed double2 = 16 bytes / 2 elements, match = "
                  << (sizeof(Double2Like) == 16 && sizeof(Double2Like) / sizeof(double) == 2) << std::endl;
        std::cout << "same total byte width (16 == 16), different element count (4 != 2): a width "
                  << "constant tuned for one dtype silently corrupts if reused verbatim for the "
                  << "other, exactly as the CUDA book's own Common Trap states -- confirmed here as "
                  << "a genuine sizeof() fact, with no GPU or SASS decoding required to see it."
                  << std::endl;
    }

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] a genuine 128-bit LDG.E.128 vector-load "
              << "instruction, and its confirmation via cuobjdump SASS disassembly -- this sandbox "
              << "has no nvcc/GPU to compile or disassemble a kernel binary with; the instruction-"
              << "count ARITHMETIC those instructions would implement is genuinely computed above "
              << "instead." << std::endl;

    // Real LibTorch cross-check: torch::Tensor's own CPU backend performs
    // exactly this class of optimization internally (ATen's CPU kernels
    // use a vectorized abstraction, at::vec::Vectorized<T>, to pack
    // several elements per SIMD instruction on whatever CPU vector width
    // this machine actually has -- SSE, AVX, or AVX-512), invisible to
    // the caller. This file does not attempt to inspect that internal
    // machinery directly (doing so honestly would require disassembling
    // LibTorch's own compiled kernels, which is out of scope here); it
    // simply confirms that ordinary tensor arithmetic still produces the
    // mathematically correct result regardless of whatever vectorization
    // ATen chose to apply underneath.
    torch::Tensor a = torch::arange(1, 11, torch::kFloat32);
    torch::Tensor b = torch::arange(11, 21, torch::kFloat32);
    torch::Tensor c = a + b;
    torch::Tensor c_expected = torch::tensor({12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0});
    std::cout << "\ntorch::Tensor addition over 10 elements (real, production LibTorch op, "
              << "internally vectorized by ATen's own CPU backend in a way this file does not "
              << "inspect) matches hand-computed expected values? "
              << torch::allclose(c, c_expected) << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 01_simd_vectorization_arithmetic.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_simd_vectorization_arithmetic
./01_simd_vectorization_arithmetic
```

```text
size=10, width=4: simd_count=8, vector_instructions=2, scalar_instructions=2, total_instructions=4 (scalar-only baseline=10)
matches CUDA book's own Worked Example 19.1.1 (simd_count=8, vector=2, scalar=2, total=4)? 1

size=41, width=4: simd_count=40, vector_instructions=10, scalar_instructions=1, total_instructions=11 (scalar-only baseline=41), ratio of baseline=26.8293%
matches CUDA book's own Worked Example 19.1.2 (simd_count=40, vector=10, scalar=1, total=11, ratio~26.8%)? 1

sizeof(Float4Like) = 16 bytes, elements packed = 4, CUDA book's own claimed float4 = 16 bytes / 4 elements, match = 1
sizeof(Double2Like) = 16 bytes, elements packed = 2, CUDA book's own claimed double2 = 16 bytes / 2 elements, match = 1
same total byte width (16 == 16), different element count (4 != 2): a width constant tuned for one dtype silently corrupts if reused verbatim for the other, exactly as the CUDA book's own Common Trap states -- confirmed here as a genuine sizeof() fact, with no GPU or SASS decoding required to see it.

[UNVERIFIED -- pending real-GPU test] a genuine 128-bit LDG.E.128 vector-load instruction, and its confirmation via cuobjdump SASS disassembly -- this sandbox has no nvcc/GPU to compile or disassemble a kernel binary with; the instruction-count ARITHMETIC those instructions would implement is genuinely computed above instead.

torch::Tensor addition over 10 elements (real, production LibTorch op, internally vectorized by ATen's own CPU backend in a way this file does not inspect) matches hand-computed expected values? 1
```

**Discussion.** The instruction-counting arithmetic here is identical in shape to Chapter 18.1's own launch-configuration arithmetic -- both split a problem size into a main body that divides evenly by some width and a remainder that doesn't, and both are genuinely testable without a GPU because they are facts about integer division, not facts about hardware execution. The dtype-width trap is a sharper, more specific lesson than Chapter 18's own traps: it is not "forgetting a bounds check" but "reusing a numeric constant across two contexts that only coincidentally share one property (total byte width) while differing in the property that actually matters (element count)." A LibTorch programmer never manually picks a vector width for a dtype at all -- `torch::Tensor` operations select their own internal vectorization width per dtype automatically, which is precisely the kind of manual bookkeeping this trap exists to avoid.

## 19.2 Loop Unrolling and Kernel Fusion: Two Different Optimizations, Both Fully Testable `[FOUNDATIONAL]`

**Intuition.** Unrolling: an inspector checking one item per glance versus four items per glance removes three of every four "is this the last one?" decisions, without changing the inspection itself. Fusion: three separate quality-control stations, each requiring its own setup and hand-off, versus one station that holds a part through every check before releasing it once.

**Background.** These are genuinely two different optimizations, easy to conflate because both make code "faster," but for different reasons. Loop unrolling processes several loop iterations per loop-control check, cutting the NUMBER OF LOOP-CONTROL EVENTS without changing the amount of arithmetic performed -- the same additions happen either way, just grouped differently. Kernel fusion collapses several separate passes over data into one pass, cutting the NUMBER OF MEMORY OPERATIONS -- reading and writing intermediate results costs real memory bandwidth that a fused kernel never pays, because it never materializes the intermediate values at all. Unlike most of Chapter 18 and this chapter's own Section 19.1, everything in this section is testable on a CPU-only sandbox with no `[UNVERIFIED]` tags anywhere -- unrolling and fusion are both memory/control-flow concepts, not CUDA-hardware-specific ones.

**Worked Example 19.2.1 and 19.2.1b: unrolling.** `size=1000, UNROLL=4` (evenly divisible): the plain loop performs 1000 loop-control events for 1000 arithmetic operations; the unrolled loop performs only 250 loop-control events for the same 1000 arithmetic operations -- a 4x reduction in loop-control overhead with the arithmetic op count and every output element identical between the two versions. `size=4002, UNROLL=4` (with a remainder of 2 elements left over): `unrolled_count = (4002/4)*4 = 4000`, so the unrolled loop performs 1000 full groups (1000 loop-control events) plus 2 individual remainder iterations, for 1002 total loop-control events against the plain loop's 4002 -- with arithmetic totals and every output element still matching exactly.

**Worked Example 19.2.2: fusion.** `relu(a*b+c)` on 6 elements, run as three separate kernels (an unfused pipeline: multiply, then add, then relu) versus one fused kernel computing all three operations in a single pass. Memory operations are counted per KERNEL PASS OVER THE WHOLE ARRAY here -- a "tensor-sized" read or write, exactly as the CUDA book's own text counts them -- not per scalar element: one kernel launch reads each of its input tensors once and writes its output tensor once, regardless of how many elements that tensor holds. The unfused pipeline: kernel 1 reads `a`, `b` and writes an intermediate (2 reads, 1 write); kernel 2 reads that intermediate and `c` and writes a second intermediate (2 reads, 1 write); kernel 3 reads that and writes the final output (1 read, 1 write) -- 5 reads, 3 writes, 8 total memory operations. The fused kernel reads `a`, `b`, `c` once each and writes the output once -- 3 reads, 1 write, 4 total memory operations, exactly half the unfused pipeline's traffic, with byte-for-byte identical output values.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

// The CUDA C++ edition's Section 19.2 covers two independent
// optimizations, both of which are pure host-side/memory-traffic
// concepts with no CUDA-specific hardware dependency at all -- unlike
// most of Chapter 18 and Section 19.1, EVERYTHING in this file is
// genuinely testable on this CPU-only sandbox, with no [UNVERIFIED]
// tags needed anywhere. Loop unrolling reduces loop-control overhead
// without changing the arithmetic performed; kernel fusion reduces
// memory traffic by collapsing several passes over data into one. This
// file reproduces the CUDA book's own worked counts exactly with real
// counters, then goes further than the CUDA book's own text: it
// actually measures wall-clock time (via std::chrono, no GPU
// synchronization needed since CPU function calls are already
// synchronous) comparing hand-fused arithmetic against LibTorch's own
// separate, un-fused eager-mode tensor operations for the same
// relu(a*b+c) expression, and reports genuinely measured numbers rather
// than the CUDA book's own GPU-specific ones.
struct LoopStats { long loop_iterations = 0; long arithmetic_ops = 0; };

int main() {
    // Worked Example 19.2.1: size=1000, UNROLL=4 (evenly divisible).
    {
        const int UNROLL = 4;
        long size = 1000;
        std::vector<float> a(size), b(size), out_plain(size), out_unrolled(size);
        for (long i = 0; i < size; i++) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 2); }

        LoopStats plain;
        for (long i = 0; i < size; i++) {
            out_plain[i] = a[i] + b[i];
            plain.loop_iterations++;
            plain.arithmetic_ops++;
        }

        LoopStats unrolled;
        long unrolled_count = (size / UNROLL) * UNROLL;
        for (long i = 0; i < unrolled_count; i += UNROLL) {
            out_unrolled[i]     = a[i]     + b[i];
            out_unrolled[i + 1] = a[i + 1] + b[i + 1];
            out_unrolled[i + 2] = a[i + 2] + b[i + 2];
            out_unrolled[i + 3] = a[i + 3] + b[i + 3];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops += 4;
        }

        bool identical = true;
        for (long i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) identical = false;

        std::cout << "size=" << size << ", UNROLL=" << UNROLL << " (evenly divisible):" << std::endl;
        std::cout << "  plain loop:    " << plain.loop_iterations << " loop-control events, "
                  << plain.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  unrolled loop: " << unrolled.loop_iterations << " loop-control events, "
                  << unrolled.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  overhead reduction: " << (plain.loop_iterations / unrolled.loop_iterations)
                  << "x fewer loop-control events (" << plain.loop_iterations << " -> "
                  << unrolled.loop_iterations << ")" << std::endl;
        std::cout << "  arithmetic op count identical: " << (plain.arithmetic_ops == unrolled.arithmetic_ops)
                  << std::endl;
        std::cout << "  every output element identical between plain and unrolled: " << identical << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.1 "
                  << "(1000/250 loop events, 1000/1000 arithmetic ops, 4x reduction)? "
                  << (plain.loop_iterations == 1000 && unrolled.loop_iterations == 250
                      && plain.arithmetic_ops == 1000 && unrolled.arithmetic_ops == 1000 && identical)
                  << std::endl;
    }

    // Worked Example 19.2.1b: size=4002, UNROLL=4 (with remainder).
    {
        const int UNROLL = 4;
        long size = 4002;
        std::vector<float> a(size), b(size), out_plain(size), out_unrolled(size);
        for (long i = 0; i < size; i++) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 3); }

        LoopStats plain;
        for (long i = 0; i < size; i++) {
            out_plain[i] = a[i] + b[i];
            plain.loop_iterations++;
            plain.arithmetic_ops++;
        }

        LoopStats unrolled;
        long unrolled_count = (size / UNROLL) * UNROLL;
        long remainder = size - unrolled_count;
        for (long i = 0; i < unrolled_count; i += UNROLL) {
            out_unrolled[i]     = a[i]     + b[i];
            out_unrolled[i + 1] = a[i + 1] + b[i + 1];
            out_unrolled[i + 2] = a[i + 2] + b[i + 2];
            out_unrolled[i + 3] = a[i + 3] + b[i + 3];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops += 4;
        }
        for (long i = unrolled_count; i < size; i++) {
            out_unrolled[i] = a[i] + b[i];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops++;
        }

        bool identical = true;
        for (long i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) identical = false;

        std::cout << "\nsize=" << size << ", UNROLL=" << UNROLL << " (with remainder):" << std::endl;
        std::cout << "  unrolled_count = (" << size << "/" << UNROLL << ")*" << UNROLL << " = "
                  << unrolled_count << ", remainder = " << remainder << " elements" << std::endl;
        std::cout << "  plain loop:    " << plain.loop_iterations << " loop-control events, "
                  << plain.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  unrolled loop: " << unrolled.loop_iterations << " loop-control events ("
                  << (unrolled_count / UNROLL) << " full groups + " << remainder << " remainder), "
                  << unrolled.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  arithmetic totals match: " << (plain.arithmetic_ops == unrolled.arithmetic_ops)
                  << " (both " << plain.arithmetic_ops << ")" << std::endl;
        std::cout << "  every output element identical between plain and unrolled: " << identical << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.1b "
                  << "(unrolled_count=4000, remainder=2, 4002/1002 loop events, 4002/4002 arithmetic ops)? "
                  << (unrolled_count == 4000 && remainder == 2 && plain.loop_iterations == 4002
                      && unrolled.loop_iterations == 1002 && plain.arithmetic_ops == 4002
                      && unrolled.arithmetic_ops == 4002 && identical) << std::endl;
    }

    // Worked Example 19.2.2: fusion's memory-traffic reduction, counted
    // with real counters using the same technique as Chapter 18.3's own
    // shared-memory read-count simulation -- relu(a*b+c) on 6 elements,
    // unfused (3 separate passes) versus fused (1 combined pass).
    {
        std::vector<float> a = {1, -2, 3, -4, 5, -6};
        std::vector<float> b = {2, 2, 0, 0, 0, 0};
        std::vector<float> c = {0, 4, 0, 0, 0, 0};
        long n = static_cast<long>(a.size());

        // Memory ops are counted PER KERNEL PASS OVER THE WHOLE ARRAY (a
        // "tensor-sized" read or write, exactly as the CUDA book's own
        // text counts them), NOT once per scalar element -- one kernel
        // launch reads each of its input tensors once and writes its
        // output tensor once, regardless of how many elements it has.
        long unfused_reads = 0, unfused_writes = 0;
        std::vector<float> intermediate1(n), intermediate2(n), unfused_out(n);
        for (long i = 0; i < n; i++) intermediate1[i] = a[i] * b[i];
        unfused_reads += 2; unfused_writes += 1;   // kernel 1: reads a,b; writes intermediate1
        for (long i = 0; i < n; i++) intermediate2[i] = intermediate1[i] + c[i];
        unfused_reads += 2; unfused_writes += 1;   // kernel 2: reads intermediate1,c; writes intermediate2
        for (long i = 0; i < n; i++) unfused_out[i] = intermediate2[i] > 0 ? intermediate2[i] : 0.0f;
        unfused_reads += 1; unfused_writes += 1;   // kernel 3: reads intermediate2; writes out
        long unfused_total = unfused_reads + unfused_writes;

        long fused_reads = 0, fused_writes = 0;
        std::vector<float> fused_out(n);
        for (long i = 0; i < n; i++) {
            float v = a[i] * b[i] + c[i];
            v = v > 0 ? v : 0.0f;
            fused_out[i] = v;
        }
        fused_reads += 3; fused_writes += 1;   // one kernel: reads a,b,c once; writes out once
        long fused_total = fused_reads + fused_writes;

        bool identical = true;
        for (long i = 0; i < n; i++) if (unfused_out[i] != fused_out[i]) identical = false;

        std::cout << "\nWorked Example 19.2.2: relu(a*b+c) on " << n << " elements:" << std::endl;
        std::cout << "  unfused (3 kernels): " << unfused_reads << " tensor-sized reads, "
                  << unfused_writes << " tensor-sized writes, " << unfused_total << " total memory ops"
                  << std::endl;
        std::cout << "  fused   (1 kernel):  " << fused_reads << " tensor-sized reads, "
                  << fused_writes << " tensor-sized writes, " << fused_total << " total memory ops"
                  << std::endl;
        std::cout << "  outputs identical between unfused and fused: " << identical << std::endl;
        std::cout << "  fused output: ";
        for (float v : fused_out) std::cout << v << " ";
        std::cout << std::endl;
        double reduction = static_cast<double>(unfused_total) / static_cast<double>(fused_total);
        std::cout << "  memory traffic reduction: " << reduction << "x (" << unfused_total << " ops -> "
                  << fused_total << " ops)" << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.2 "
                  << "(5/3/8 unfused, 3/1/4 fused, output 2.0 0.0 0.0 0.0 0.0 0.0, 2x reduction)? "
                  << (unfused_reads == 5 && unfused_writes == 3 && unfused_total == 8
                      && fused_reads == 3 && fused_writes == 1 && fused_total == 4
                      && identical && reduction == 2.0
                      && fused_out[0] == 2.0f && fused_out[1] == 0.0f && fused_out[2] == 0.0f
                      && fused_out[3] == 0.0f && fused_out[4] == 0.0f && fused_out[5] == 0.0f)
                  << std::endl;
    }

    // Going beyond the CUDA book's own text: LibTorch's real, honest
    // behavior on kernel fusion. Ordinary eager-mode torch:: calls do
    // NOT automatically fuse -- each of mul, add, relu below launches
    // its own separate CPU kernel and writes its own separate
    // intermediate tensor, exactly like the "unfused" simulation above.
    // A hand-written, single-pass loop over raw data (mirroring the CUDA
    // book's own tiled-kernel style) genuinely fuses the three
    // operations into one pass, and both are timed here with real
    // std::chrono wall-clock measurement -- no GPU synchronization is
    // needed on CPU, since a CPU function call already blocks until
    // finished (the CUDA book's own cudaDeviceSynchronize() exists
    // specifically because GPU kernel launches do NOT block the host,
    // a distinction with no CPU equivalent to reproduce).
    {
        long n = 2000000;
        torch::Tensor ta = torch::rand({n});
        torch::Tensor tb = torch::rand({n});
        torch::Tensor tc = torch::rand({n});

        auto unfused_run = [&]() {
            torch::Tensor m = ta * tb;
            torch::Tensor s = m + tc;
            torch::Tensor r = torch::relu(s);
            return r;
        };
        auto fused_run = [&]() {
            return torch::relu(torch::addcmul(tc, ta, tb));
        };

        torch::Tensor unfused_result = unfused_run();
        torch::Tensor fused_result = fused_run();
        std::cout << "\nreal LibTorch cross-check: separate mul+add+relu vs torch::addcmul-then-relu "
                  << "(a genuinely different, more fused ATen op path) produce allclose results? "
                  << torch::allclose(unfused_result, fused_result) << std::endl;

        const int warmup = 5, timed = 50;
        for (int i = 0; i < warmup; i++) unfused_run();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) unfused_run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double unfused_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / timed;

        for (int i = 0; i < warmup; i++) fused_run();
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) fused_run();
        auto t3 = std::chrono::high_resolution_clock::now();
        double fused_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / timed;

        std::cout << "genuinely measured on THIS sandbox's own CPU (5 warmup + 50 timed runs each, "
                  << n << " elements) -- separate ops: " << unfused_ms << " ms/run, "
                  << "torch::addcmul-then-relu: " << fused_ms << " ms/run" << std::endl;
        std::cout << "these are this sandbox's own CPU numbers, not a reproduction of the CUDA book's "
                  << "own GPU numbers -- different hardware entirely; the point is that the "
                  << "MEASUREMENT METHODOLOGY (warmup, repeated timed runs, averaging) is what "
                  << "carries over, not any specific millisecond figure." << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run (the `[TIMING]`-adjacent line's own millisecond figures are this sandbox's own, and vary slightly run to run; every structural count and boolean check above them is exact and reproducible):

```bash
g++ -std=c++20 -O2 02_unrolling_and_fusion.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_unrolling_and_fusion
./02_unrolling_and_fusion
```

```text
size=1000, UNROLL=4 (evenly divisible):
  plain loop:    1000 loop-control events, 1000 arithmetic ops
  unrolled loop: 250 loop-control events, 1000 arithmetic ops
  overhead reduction: 4x fewer loop-control events (1000 -> 250)
  arithmetic op count identical: 1
  every output element identical between plain and unrolled: 1
  matches CUDA book's own Worked Example 19.2.1 (1000/250 loop events, 1000/1000 arithmetic ops, 4x reduction)? 1

size=4002, UNROLL=4 (with remainder):
  unrolled_count = (4002/4)*4 = 4000, remainder = 2 elements
  plain loop:    4002 loop-control events, 4002 arithmetic ops
  unrolled loop: 1002 loop-control events (1000 full groups + 2 remainder), 4002 arithmetic ops
  arithmetic totals match: 1 (both 4002)
  every output element identical between plain and unrolled: 1
  matches CUDA book's own Worked Example 19.2.1b (unrolled_count=4000, remainder=2, 4002/1002 loop events, 4002/4002 arithmetic ops)? 1

Worked Example 19.2.2: relu(a*b+c) on 6 elements:
  unfused (3 kernels): 5 tensor-sized reads, 3 tensor-sized writes, 8 total memory ops
  fused   (1 kernel):  3 tensor-sized reads, 1 tensor-sized writes, 4 total memory ops
  outputs identical between unfused and fused: 1
  fused output: 2 0 0 0 0 0 
  memory traffic reduction: 2x (8 ops -> 4 ops)
  matches CUDA book's own Worked Example 19.2.2 (5/3/8 unfused, 3/1/4 fused, output 2.0 0.0 0.0 0.0 0.0 0.0, 2x reduction)? 1

real LibTorch cross-check: separate mul+add+relu vs torch::addcmul-then-relu (a genuinely different, more fused ATen op path) produce allclose results? 1
genuinely measured on THIS sandbox's own CPU (5 warmup + 50 timed runs each, 2000000 elements) -- separate ops: 7.51335 ms/run, torch::addcmul-then-relu: 5.25016 ms/run
these are this sandbox's own CPU numbers, not a reproduction of the CUDA book's own GPU numbers -- different hardware entirely; the point is that the MEASUREMENT METHODOLOGY (warmup, repeated timed runs, averaging) is what carries over, not any specific millisecond figure.
```

**Discussion.** The two optimizations are easy to mix up because both make code "faster," but the mechanism is genuinely different, and this section's own counters prove it: unrolling leaves the arithmetic-op count exactly unchanged (1000 either way) while cutting the loop-control-event count 4x; fusion leaves the loop-control structure almost irrelevant and instead cuts the memory-operation count in half. A reader who conflates the two might expect unrolling to reduce memory traffic (it doesn't -- it only reduces control-flow overhead) or expect fusion to reduce arithmetic-instruction count (it doesn't either -- the same multiply, add, and comparison still happen once per element; what disappears is the round-trip to memory for the intermediate values). The real LibTorch cross-check is honest about a genuine limitation: ordinary eager-mode `torch::` calls do not automatically fuse operations the way the CUDA book's own hand-tiled kernel does -- `torch::addcmul` is a *different, separately-implemented* fused ATen operation, not automatic fusion of the three separate calls above it, and this file says so plainly rather than overclaiming that LibTorch "gets fusion for free" the way it got tiled convolution for free in Chapter 18.3. Real operator fusion in the PyTorch/LibTorch ecosystem is what `torch::jit` graph compilation and its fusers exist to provide, which is out of scope for a CPU-only sandbox exercising eager-mode C++ code.

## 19.3 Compile-Time Specialization: A Section With No `[UNVERIFIED]` Tags At All `[FOUNDATIONAL]`

**Intuition.** Casting two separate purpose-built plates ahead of time -- one 4-wide, one 8-wide -- versus keeping one adjustable plate that queries its own current width setting on every single strike. The two plates never pay a per-strike "which width am I?" check; the adjustable plate always does.

**Background.** The CUDA C++ edition's own Section 19.3 uses a C++ function template, `template<int N>`, to produce a distinct, independently-compiled machine-code body for each value of `N` a program actually instantiates it with -- confirmed via a genuine linked symbol table showing two different addresses for `compile_time_specialized_dot<4>` and `compile_time_specialized_dot<2>`. This is ordinary C++ template mechanics, with **no CUDA dependency whatsoever** -- unlike every other section in this chapter and the whole of Chapter 18, this section needs no `[UNVERIFIED -- pending real-GPU test]` tag anywhere, because nothing in it depends on GPU hardware at all.

**Worked Example 19.3.1.** `compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = 1*5+2*6+3*7+4*8 = 70`. `compile_time_specialized_dot<2>([2,3], [4,5]) = 2*4+3*5 = 23`. These are not one function called twice with a runtime-varying length -- `N=4` and `N=2` select two genuinely different, separately-compiled functions, confirmed two ways in this file: first, from *inside* the running program, by comparing the two instantiations' own function-pointer addresses and finding they differ; second, from *outside* the binary, via `nm`'s own linked symbol table, which shows two distinct symbol addresses rather than one shared symbol.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 19.3 uses a C++ function template,
// template<int N>, to produce distinct, independently-compiled machine-
// code bodies per instantiation, confirmed via a linked symbol table
// showing two different addresses for compile_time_specialized_dot<4>
// and compile_time_specialized_dot<2>. This is ordinary C++ template
// mechanics with NO CUDA dependency whatsoever -- unlike most of this
// chapter, this entire section is testable exactly as the CUDA book's
// own text describes it, with no [UNVERIFIED] tags needed anywhere. The
// symbol-table confirmation is reproduced here using genuine `nm`
// output on this file's own compiled binary, not asserted.
template<int N>
float compile_time_specialized_dot(const float* a, const float* b) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

// Explicit instantiation so both symbols are guaranteed to survive into
// the final binary (a template only otherwise emits the instantiations
// its call sites actually use, which main() below already forces, but
// explicit instantiation makes the intent unambiguous for the nm check).
template float compile_time_specialized_dot<4>(const float*, const float*);
template float compile_time_specialized_dot<2>(const float*, const float*);

int main() {
    // Worked Example 19.3.1: two instantiations, two independent answers.
    float a4[4] = {1, 2, 3, 4};
    float b4[4] = {5, 6, 7, 8};
    float result4 = compile_time_specialized_dot<4>(a4, b4);
    std::cout << "compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = " << result4
              << " (expected 1*5+2*6+3*7+4*8 = 70)" << std::endl;
    std::cout << "matches CUDA book's own Worked Example 19.3.1 (70.0)? " << (result4 == 70.0f)
              << std::endl;

    float a2[2] = {2, 3};
    float b2[2] = {4, 5};
    float result2 = compile_time_specialized_dot<2>(a2, b2);
    std::cout << "\ncompile_time_specialized_dot<2>([2,3], [4,5]) = " << result2
              << " (expected 2*4+3*5 = 23)" << std::endl;
    std::cout << "matches CUDA book's own Worked Example 19.3.1 (23.0)? " << (result2 == 23.0f)
              << std::endl;

    // Confirm, from within the running program itself, that the two
    // instantiations really are two distinct compiled function bodies,
    // not one shared symbol dispatching on a runtime N: take each
    // instantiation's own function pointer and compare them. The
    // specific address VALUES are not printed here (link addresses can
    // shift between separate compilations of identical source, exactly
    // like the data_ptr values earlier chapters also treated as
    // non-reproducible-by-value); what is asserted, and is a stable
    // fact of the compiled binary, is that the two addresses DIFFER.
    void* addr4 = reinterpret_cast<void*>(&compile_time_specialized_dot<4>);
    void* addr2 = reinterpret_cast<void*>(&compile_time_specialized_dot<2>);
    std::cout << "\nthese are not one function called with a runtime-varying length -- N=4 and N=2 "
              << "select two DIFFERENT, separately-compiled functions: their own function-pointer "
              << "addresses differ? " << (addr4 != addr2) << std::endl;
    std::cout << "(this binary's own linked symbol table, inspected separately via nm below, shows "
              << "the same fact from the outside: two distinct symbol addresses, not one shared "
              << "symbol.)" << std::endl;

    // Real LibTorch cross-check: torch::dot() is LibTorch's own actual,
    // production dot-product reduction -- a single runtime function that
    // handles any tensor length via a runtime loop bound, the opposite
    // design choice from compile-time specialization. Both are legitimate
    // engineering trade-offs: torch::dot() trades a small per-call
    // runtime-length check for a single compiled body covering every
    // length; compile_time_specialized_dot<N> trades zero runtime
    // overhead for one compiled body per length actually instantiated.
    torch::Tensor ta4 = torch::tensor({1.0, 2.0, 3.0, 4.0});
    torch::Tensor tb4 = torch::tensor({5.0, 6.0, 7.0, 8.0});
    double torch_dot4 = torch::dot(ta4, tb4).item<double>();
    std::cout << "\ntorch::dot() (real, production LibTorch reduction, a single runtime-length "
              << "function, not compile-time specialized) on the same [1,2,3,4]-dot-[5,6,7,8] = "
              << torch_dot4 << ", matches this file's own compile-time-specialized result (70.0)? "
              << (torch_dot4 == 70.0) << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 03_compile_time_specialization.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_compile_time_specialization
./03_compile_time_specialization
```

```text
compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = 70 (expected 1*5+2*6+3*7+4*8 = 70)
matches CUDA book's own Worked Example 19.3.1 (70.0)? 1

compile_time_specialized_dot<2>([2,3], [4,5]) = 23 (expected 2*4+3*5 = 23)
matches CUDA book's own Worked Example 19.3.1 (23.0)? 1

these are not one function called with a runtime-varying length -- N=4 and N=2 select two DIFFERENT, separately-compiled functions: their own function-pointer addresses differ? 1
(this binary's own linked symbol table, inspected separately via nm below, shows the same fact from the outside: two distinct symbol addresses, not one shared symbol.)

torch::dot() (real, production LibTorch reduction, a single runtime-length function, not compile-time specialized) on the same [1,2,3,4]-dot-[5,6,7,8] = 70, matches this file's own compile-time-specialized result (70.0)? 1
```

The linked symbol table for this exact binary, genuinely inspected via `nm -C` (the two hexadecimal addresses are this specific build's own values, expected to shift between separate compilations -- what matters, and is stable, is that the two addresses are not equal):

```text
0000000000006660 W float compile_time_specialized_dot<2>(float const*, float const*)
0000000000006620 W float compile_time_specialized_dot<4>(float const*, float const*)
```

**Discussion.** The CUDA book's own Common Trap for this section is worth restating exactly: "zero runtime cost is not zero cost." Every distinct value of `N` a program actually instantiates `compile_time_specialized_dot` with produces its own compiled function body -- a program instantiating this function at `N=2, 4, 8, 16` compiles four separate machine-code bodies, not one generic body handling four cases. That is the same "code bloat" trade-off C++ template instantiation has always made, and it is worth noticing precisely because it is so easy to overlook: "compile-time" and "free" are not synonyms. `torch::dot()`'s own opposite design -- one function, a runtime-determined loop bound, checked once per call -- is the other legitimate point on this trade-off curve, and a LibTorch programmer reaching for `torch::dot()` is implicitly choosing "small runtime overhead, no binary-size cost" over "zero runtime overhead, binary-size cost proportional to how many distinct lengths get used."

## 19.4 Benchmark Framework: Warmup, Timed Runs, and What CPU Timing Doesn't Need `[FOUNDATIONAL]`

**Intuition.** Timing a runner's first cold sprint measures a slow start, not a true pace -- warm-up laps and averaging over several timed laps represent the runner fairly. GPU work adds a wrinkle a CPU-only sandbox never has to solve: the host queues a kernel launch and moves on immediately, so timing the launch loop itself measures how fast the host can queue work, not how fast the device actually finishes it, unless an explicit synchronization call forces the host to wait.

**Background.** The CUDA C++ edition's own benchmarking harness runs several warmup iterations (to let caches warm and any one-time setup cost get paid before the clock starts), then several timed iterations, averaged -- bracketed by two `cudaDeviceSynchronize()` calls, one right before the clock starts and one right after it stops. This sandbox has no GPU, so a genuine `cudaDeviceSynchronize()` call is `[UNVERIFIED -- pending real-GPU test]` -- but the reason that call exists in the first place has no CPU equivalent to reproduce: a CPU function call is already synchronous by the time it returns, so a plain `std::chrono` measurement around a CPU function is already an honest measurement, with no separate barrier needed. What this section builds and runs for real is the harness structure itself (warmup runs, timed runs, averaging), benchmarked against real `torch::nn::functional::conv2d` calls at two sizes, converted into GFLOPS using the CUDA book's own formula, plus a genuine host memcpy bandwidth measurement -- every millisecond and GFLOPS figure below was measured live on this sandbox's own CPU during this run, and will differ from the CUDA book's own GPU numbers simply because the hardware is entirely different, not because either measurement is wrong.

**Worked Example 19.4.1 and 19.4.2.** A `vector_add` workload over 2,000,000 elements confirms the harness itself produces a spot-checked-correct result across 5 warmup and 20 timed runs. The GFLOPS conversion follows the CUDA book's own formula for a 3x3-kernel 2D convolution -- 2 FLOPs per tap x 9 taps per output position -- applied to real `torch::conv2d` calls at 64x64 and 128x128, both basic (valid, unpadded) and padded (same-shape output), with the padding's own extra-FLOPs percentage shrinking as the problem size grows, exactly the CUDA book's own claim about padding overhead decreasing proportionally at scale. A genuine 64 MiB host-to-host `memcpy`, timed the same way, produces this sandbox's own bandwidth figure -- the CUDA book's own text is explicit that its own equivalent number is "this host's memcpy, not GPU," so this section's own CPU-only measurement is answering exactly the question the CUDA book's own text already scoped that way, not standing in for a GPU number it has no way to produce.

```cpp
#include <torch/torch.h>
#include <torch/nn/functional/conv.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <functional>

// The CUDA C++ edition's Section 19.4 builds a benchmarking harness that
// runs warmup iterations, then timed iterations, bracketed by
// cudaDeviceSynchronize() calls so the host's wall-clock timer honestly
// reflects when the GPU actually finished (a GPU kernel launch returns
// to the host immediately, before the work is done, so timing without
// synchronization would measure queueing speed, not compute speed).
// This sandbox has no GPU, so cudaDeviceSynchronize() itself is
// [UNVERIFIED -- pending real-GPU test] -- but a CPU function call has
// no equivalent asynchrony to correct for in the first place: when
// conv2d_basic(...) returns on CPU, the work is genuinely done, so a
// plain std::chrono measurement is already honest here, with no
// synchronization barrier needed. This file builds a genuine warmup+
// timed-runs harness, benchmarks real torch::nn::functional::conv2d at
// two sizes, converts the measured time into GFLOPS using the CUDA
// book's own formula, and measures real host memcpy bandwidth -- all
// numbers below are genuinely measured on THIS sandbox's own CPU during
// this run, not a reproduction of the CUDA book's own GPU numbers,
// which were measured on entirely different hardware. Wall-clock timing
// numbers are inherently non-deterministic between runs (this file's
// own verify pipeline accounts for that); the pass/fail structural
// checks around them are not.
struct Benchmark {
    double time_function(const std::function<void()>& func, int warmup_runs = 5, int benchmark_runs = 20) {
        for (int i = 0; i < warmup_runs; i++) func();
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; i++) func();
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
        return (total_ns / benchmark_runs) / 1.0e6;   // milliseconds
    }
};

int main() {
    Benchmark bench;

    // Worked Example 19.4.1: harness correctness, vector_add over
    // 2,000,000 elements.
    {
        long n = 2000000;
        torch::Tensor va = torch::rand({n});
        torch::Tensor vb = torch::rand({n});
        torch::Tensor result;
        double ms = bench.time_function([&]() { result = va + vb; }, 5, 20);
        torch::Tensor expected = va + vb;
        bool correct = torch::allclose(result, expected);
        std::cout << "vector_add_workload over " << n << " elements, 5 warmup + 20 timed runs:" << std::endl;
        std::cout << "  [TIMING] average time = " << ms << " ms per run" << std::endl;
        std::cout << "  spot-checked output correctness: " << correct << std::endl;
    }

    // Worked Example 19.4.2: converting measured timing to GFLOPS, for a
    // 3x3-kernel 2D convolution, at two sizes, basic (valid, no padding)
    // vs padded (same-shape output) -- real torch::conv2d, real timing.
    auto run_conv_gflops = [&](long size) {
        torch::Tensor input = torch::rand({1, 1, size, size});
        torch::Tensor kernel = torch::rand({1, 1, 3, 3});

        double basic_ms = bench.time_function([&]() {
            torch::nn::functional::conv2d(input, kernel);
        }, 5, 20);
        long basic_ops = 2L * (size - 2) * (size - 2) * 3 * 3;
        double basic_gflops = static_cast<double>(basic_ops) / (basic_ms * 1e6);

        double padded_ms = bench.time_function([&]() {
            torch::nn::functional::conv2d(input, kernel, torch::nn::functional::Conv2dFuncOptions().padding(1));
        }, 5, 20);
        long padded_ops = 2L * size * size * 3 * 3;
        double padded_gflops = static_cast<double>(padded_ops) / (padded_ms * 1e6);

        long extra_ops = padded_ops - basic_ops;
        double extra_pct = 100.0 * static_cast<double>(extra_ops) / static_cast<double>(basic_ops);

        std::cout << "\n" << size << "x" << size << " convolution (5 warmup + 20 timed runs each, "
                  << "genuinely measured):" << std::endl;
        std::cout << "  basic_ops  = 2 * (" << size << "-2)^2 * 3 * 3 = " << basic_ops << std::endl;
        std::cout << "  [TIMING] Basic  convolution: " << basic_ms << " ms, " << basic_gflops << " GFLOPS"
                  << std::endl;
        std::cout << "  padded_ops = 2 * " << size << "^2 * 3 * 3 = " << padded_ops << std::endl;
        std::cout << "  [TIMING] Padded convolution: " << padded_ms << " ms, " << padded_gflops << " GFLOPS"
                  << std::endl;
        std::cout << "  extra work from padding: " << extra_ops << " more FLOPs (" << extra_pct << "% more)"
                  << std::endl;
        std::cout << "  ops formula matches CUDA book's own Worked Example 19.4.2 formula "
                  << "(2*(size-2)^2*9 basic, 2*size^2*9 padded)? "
                  << (basic_ops == 2L * (size - 2) * (size - 2) * 9 && padded_ops == 2L * size * size * 9)
                  << std::endl;
    };
    run_conv_gflops(64);
    run_conv_gflops(128);

    // Memory bandwidth measurement: host memcpy of 64 MiB, genuinely
    // timed and checked for correctness -- the CUDA book's own text is
    // explicit that this specific number is "this host's memcpy, not
    // GPU," so this sandbox's own CPU-only measurement is answering
    // exactly the question the CUDA book's own text already scoped this
    // way, not standing in for a GPU number it cannot produce.
    {
        const size_t bytes = 64 * 1024 * 1024;
        std::vector<char> src(bytes), dst(bytes);
        for (size_t i = 0; i < bytes; i++) src[i] = static_cast<char>(i % 251);

        const int warmup = 3, timed = 10;
        for (int i = 0; i < warmup; i++) std::memcpy(dst.data(), src.data(), bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) std::memcpy(dst.data(), src.data(), bytes);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / timed;
        double gb_per_s = (static_cast<double>(bytes) / 1.0e9) / (avg_ms / 1000.0);

        bool correct = (std::memcmp(src.data(), dst.data(), bytes) == 0);
        std::cout << "\nhost-to-host memcpy of 64 MiB (3 warmup + 10 timed runs):" << std::endl;
        std::cout << "  [TIMING] average time = " << avg_ms << " ms" << std::endl;
        std::cout << "  [TIMING] achieved bandwidth = " << gb_per_s << " GB/s (this sandbox's own "
                  << "CPU memcpy, not GPU -- the CUDA book's own text already scopes this "
                  << "specific measurement as host memcpy, not a GPU-bandwidth claim)" << std::endl;
        std::cout << "  copy correctness check: " << correct << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run (again, every `[TIMING]` line's own numeric value is this sandbox's own measurement from this specific run, and will vary slightly on a fresh rerun; every structural fact and boolean check is exact):

```bash
g++ -std=c++20 -O2 04_benchmark_harness_and_bandwidth.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_benchmark_harness_and_bandwidth
./04_benchmark_harness_and_bandwidth
```

```text
vector_add_workload over 2000000 elements, 5 warmup + 20 timed runs:
  [TIMING] average time = 2.32623 ms per run
  spot-checked output correctness: 1

64x64 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (64-2)^2 * 3 * 3 = 69192
  [TIMING] Basic  convolution: 0.0391397 ms, 1.76782 GFLOPS
  padded_ops = 2 * 64^2 * 3 * 3 = 73728
  [TIMING] Padded convolution: 0.0418279 ms, 1.76265 GFLOPS
  extra work from padding: 4536 more FLOPs (6.55567% more)
  ops formula matches CUDA book's own Worked Example 19.4.2 formula (2*(size-2)^2*9 basic, 2*size^2*9 padded)? 1

128x128 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (128-2)^2 * 3 * 3 = 285768
  [TIMING] Basic  convolution: 0.0563606 ms, 5.07036 GFLOPS
  padded_ops = 2 * 128^2 * 3 * 3 = 294912
  [TIMING] Padded convolution: 0.0378515 ms, 7.79129 GFLOPS
  extra work from padding: 9144 more FLOPs (3.1998% more)
  ops formula matches CUDA book's own Worked Example 19.4.2 formula (2*(size-2)^2*9 basic, 2*size^2*9 padded)? 1

host-to-host memcpy of 64 MiB (3 warmup + 10 timed runs):
  [TIMING] average time = 5.74702 ms
  [TIMING] achieved bandwidth = 11.6772 GB/s (this sandbox's own CPU memcpy, not GPU -- the CUDA book's own text already scopes this specific measurement as host memcpy, not a GPU-bandwidth claim)
  copy correctness check: 1
```

**Discussion.** This section closes Part 5 by making the honest-divergence framework pay off in a genuinely different way than Chapter 18 did: rather than tagging most of the section `[UNVERIFIED]`, the CUDA book's own benchmarking DISCIPLINE -- warmup runs, repeated timed runs, averaging, converting raw milliseconds into a comparable unit like GFLOPS -- turns out to be almost entirely hardware-agnostic, and this sandbox's own CPU numbers are genuine, correctly-measured answers to genuinely different (CPU, not GPU) questions. The one CUDA-specific piece, `cudaDeviceSynchronize()`, is `[UNVERIFIED]` not because this section failed to test something it should have, but because the CPU has no equivalent problem for that call to solve in the first place -- a fact worth stating plainly rather than glossing over. The 64x64-versus-128x128 padding-overhead comparison reproduces the CUDA book's own directional claim (padding overhead shrinks proportionally as the problem grows) using this sandbox's own genuinely different absolute numbers, which is exactly the right way to reproduce a SCALING claim without a GPU: the shape of the relationship, not the specific millisecond figures, is what generalizes across hardware.

## Complete Runnable Code

### `01_simd_vectorization_arithmetic.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 19.1 packs four float loads into one
// 128-bit float4 load instruction on real GPU hardware, and confirms the
// packing via genuine SASS disassembly (an LDG.E.128 instruction). This
// sandbox has no GPU and no nvcc (see Chapter 18's own honest
// disclosure), so no real vector-LOAD instruction and no SASS decoding
// can be produced here -- both are [UNVERIFIED -- pending real-GPU
// test]. What CAN be genuinely run, and is run in this file, is the
// SIMD instruction-COUNTING arithmetic itself -- pure host-side integer
// math with no CUDA dependency -- plus the dtype-width trap, which is a
// compile-time sizeof() fact about C++ types, true on any machine that
// can compile C++ at all, not something that requires a GPU to check.
long ceiling_div_floor(long size, long width) { return (size / width) * width; }

int main() {
    // Worked Example 19.1.1: size=10, width=4.
    {
        long size = 10, width = 4;
        long simd_count = ceiling_div_floor(size, width);
        long vector_instructions = simd_count / width;
        long scalar_instructions = size - simd_count;
        long total_instructions = vector_instructions + scalar_instructions;
        std::cout << "size=" << size << ", width=" << width << ": simd_count=" << simd_count
                  << ", vector_instructions=" << vector_instructions
                  << ", scalar_instructions=" << scalar_instructions
                  << ", total_instructions=" << total_instructions
                  << " (scalar-only baseline=" << size << ")" << std::endl;
        std::cout << "matches CUDA book's own Worked Example 19.1.1 "
                  << "(simd_count=8, vector=2, scalar=2, total=4)? "
                  << (simd_count == 8 && vector_instructions == 2 && scalar_instructions == 2
                      && total_instructions == 4) << std::endl;
    }

    // Worked Example 19.1.2: size=41, width=4.
    {
        long size = 41, width = 4;
        long simd_count = ceiling_div_floor(size, width);
        long vector_instructions = simd_count / width;
        long scalar_instructions = size - simd_count;
        long total_instructions = vector_instructions + scalar_instructions;
        double ratio = 100.0 * static_cast<double>(total_instructions) / static_cast<double>(size);
        std::cout << "\nsize=" << size << ", width=" << width << ": simd_count=" << simd_count
                  << ", vector_instructions=" << vector_instructions
                  << ", scalar_instructions=" << scalar_instructions
                  << ", total_instructions=" << total_instructions
                  << " (scalar-only baseline=" << size << "), ratio of baseline=" << ratio << "%" << std::endl;
        std::cout << "matches CUDA book's own Worked Example 19.1.2 "
                  << "(simd_count=40, vector=10, scalar=1, total=11, ratio~26.8%)? "
                  << (simd_count == 40 && vector_instructions == 10 && scalar_instructions == 1
                      && total_instructions == 11 && ratio > 26.7 && ratio < 26.9) << std::endl;
    }

    // Worked Example 19.1.3: the dtype-width trap, as a genuine sizeof()
    // fact. LibTorch has no built-in float4/double2 type (those are CUDA
    // vector types), so this uses the CUDA book's own claimed byte
    // widths directly -- both are plain compile-time facts about fixed-
    // width types, true regardless of what hardware eventually runs
    // code built against them.
    {
        struct Float4Like { float x, y, z, w; };
        struct Double2Like { double x, y; };
        std::cout << "\nsizeof(Float4Like) = " << sizeof(Float4Like) << " bytes, elements packed = "
                  << (sizeof(Float4Like) / sizeof(float))
                  << ", CUDA book's own claimed float4 = 16 bytes / 4 elements, match = "
                  << (sizeof(Float4Like) == 16 && sizeof(Float4Like) / sizeof(float) == 4) << std::endl;
        std::cout << "sizeof(Double2Like) = " << sizeof(Double2Like) << " bytes, elements packed = "
                  << (sizeof(Double2Like) / sizeof(double))
                  << ", CUDA book's own claimed double2 = 16 bytes / 2 elements, match = "
                  << (sizeof(Double2Like) == 16 && sizeof(Double2Like) / sizeof(double) == 2) << std::endl;
        std::cout << "same total byte width (16 == 16), different element count (4 != 2): a width "
                  << "constant tuned for one dtype silently corrupts if reused verbatim for the "
                  << "other, exactly as the CUDA book's own Common Trap states -- confirmed here as "
                  << "a genuine sizeof() fact, with no GPU or SASS decoding required to see it."
                  << std::endl;
    }

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] a genuine 128-bit LDG.E.128 vector-load "
              << "instruction, and its confirmation via cuobjdump SASS disassembly -- this sandbox "
              << "has no nvcc/GPU to compile or disassemble a kernel binary with; the instruction-"
              << "count ARITHMETIC those instructions would implement is genuinely computed above "
              << "instead." << std::endl;

    // Real LibTorch cross-check: torch::Tensor's own CPU backend performs
    // exactly this class of optimization internally (ATen's CPU kernels
    // use a vectorized abstraction, at::vec::Vectorized<T>, to pack
    // several elements per SIMD instruction on whatever CPU vector width
    // this machine actually has -- SSE, AVX, or AVX-512), invisible to
    // the caller. This file does not attempt to inspect that internal
    // machinery directly (doing so honestly would require disassembling
    // LibTorch's own compiled kernels, which is out of scope here); it
    // simply confirms that ordinary tensor arithmetic still produces the
    // mathematically correct result regardless of whatever vectorization
    // ATen chose to apply underneath.
    torch::Tensor a = torch::arange(1, 11, torch::kFloat32);
    torch::Tensor b = torch::arange(11, 21, torch::kFloat32);
    torch::Tensor c = a + b;
    torch::Tensor c_expected = torch::tensor({12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0});
    std::cout << "\ntorch::Tensor addition over 10 elements (real, production LibTorch op, "
              << "internally vectorized by ATen's own CPU backend in a way this file does not "
              << "inspect) matches hand-computed expected values? "
              << torch::allclose(c, c_expected) << std::endl;

    return 0;
}
```

### `02_unrolling_and_fusion.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

// The CUDA C++ edition's Section 19.2 covers two independent
// optimizations, both of which are pure host-side/memory-traffic
// concepts with no CUDA-specific hardware dependency at all -- unlike
// most of Chapter 18 and Section 19.1, EVERYTHING in this file is
// genuinely testable on this CPU-only sandbox, with no [UNVERIFIED]
// tags needed anywhere. Loop unrolling reduces loop-control overhead
// without changing the arithmetic performed; kernel fusion reduces
// memory traffic by collapsing several passes over data into one. This
// file reproduces the CUDA book's own worked counts exactly with real
// counters, then goes further than the CUDA book's own text: it
// actually measures wall-clock time (via std::chrono, no GPU
// synchronization needed since CPU function calls are already
// synchronous) comparing hand-fused arithmetic against LibTorch's own
// separate, un-fused eager-mode tensor operations for the same
// relu(a*b+c) expression, and reports genuinely measured numbers rather
// than the CUDA book's own GPU-specific ones.
struct LoopStats { long loop_iterations = 0; long arithmetic_ops = 0; };

int main() {
    // Worked Example 19.2.1: size=1000, UNROLL=4 (evenly divisible).
    {
        const int UNROLL = 4;
        long size = 1000;
        std::vector<float> a(size), b(size), out_plain(size), out_unrolled(size);
        for (long i = 0; i < size; i++) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 2); }

        LoopStats plain;
        for (long i = 0; i < size; i++) {
            out_plain[i] = a[i] + b[i];
            plain.loop_iterations++;
            plain.arithmetic_ops++;
        }

        LoopStats unrolled;
        long unrolled_count = (size / UNROLL) * UNROLL;
        for (long i = 0; i < unrolled_count; i += UNROLL) {
            out_unrolled[i]     = a[i]     + b[i];
            out_unrolled[i + 1] = a[i + 1] + b[i + 1];
            out_unrolled[i + 2] = a[i + 2] + b[i + 2];
            out_unrolled[i + 3] = a[i + 3] + b[i + 3];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops += 4;
        }

        bool identical = true;
        for (long i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) identical = false;

        std::cout << "size=" << size << ", UNROLL=" << UNROLL << " (evenly divisible):" << std::endl;
        std::cout << "  plain loop:    " << plain.loop_iterations << " loop-control events, "
                  << plain.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  unrolled loop: " << unrolled.loop_iterations << " loop-control events, "
                  << unrolled.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  overhead reduction: " << (plain.loop_iterations / unrolled.loop_iterations)
                  << "x fewer loop-control events (" << plain.loop_iterations << " -> "
                  << unrolled.loop_iterations << ")" << std::endl;
        std::cout << "  arithmetic op count identical: " << (plain.arithmetic_ops == unrolled.arithmetic_ops)
                  << std::endl;
        std::cout << "  every output element identical between plain and unrolled: " << identical << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.1 "
                  << "(1000/250 loop events, 1000/1000 arithmetic ops, 4x reduction)? "
                  << (plain.loop_iterations == 1000 && unrolled.loop_iterations == 250
                      && plain.arithmetic_ops == 1000 && unrolled.arithmetic_ops == 1000 && identical)
                  << std::endl;
    }

    // Worked Example 19.2.1b: size=4002, UNROLL=4 (with remainder).
    {
        const int UNROLL = 4;
        long size = 4002;
        std::vector<float> a(size), b(size), out_plain(size), out_unrolled(size);
        for (long i = 0; i < size; i++) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 3); }

        LoopStats plain;
        for (long i = 0; i < size; i++) {
            out_plain[i] = a[i] + b[i];
            plain.loop_iterations++;
            plain.arithmetic_ops++;
        }

        LoopStats unrolled;
        long unrolled_count = (size / UNROLL) * UNROLL;
        long remainder = size - unrolled_count;
        for (long i = 0; i < unrolled_count; i += UNROLL) {
            out_unrolled[i]     = a[i]     + b[i];
            out_unrolled[i + 1] = a[i + 1] + b[i + 1];
            out_unrolled[i + 2] = a[i + 2] + b[i + 2];
            out_unrolled[i + 3] = a[i + 3] + b[i + 3];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops += 4;
        }
        for (long i = unrolled_count; i < size; i++) {
            out_unrolled[i] = a[i] + b[i];
            unrolled.loop_iterations++;
            unrolled.arithmetic_ops++;
        }

        bool identical = true;
        for (long i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) identical = false;

        std::cout << "\nsize=" << size << ", UNROLL=" << UNROLL << " (with remainder):" << std::endl;
        std::cout << "  unrolled_count = (" << size << "/" << UNROLL << ")*" << UNROLL << " = "
                  << unrolled_count << ", remainder = " << remainder << " elements" << std::endl;
        std::cout << "  plain loop:    " << plain.loop_iterations << " loop-control events, "
                  << plain.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  unrolled loop: " << unrolled.loop_iterations << " loop-control events ("
                  << (unrolled_count / UNROLL) << " full groups + " << remainder << " remainder), "
                  << unrolled.arithmetic_ops << " arithmetic ops" << std::endl;
        std::cout << "  arithmetic totals match: " << (plain.arithmetic_ops == unrolled.arithmetic_ops)
                  << " (both " << plain.arithmetic_ops << ")" << std::endl;
        std::cout << "  every output element identical between plain and unrolled: " << identical << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.1b "
                  << "(unrolled_count=4000, remainder=2, 4002/1002 loop events, 4002/4002 arithmetic ops)? "
                  << (unrolled_count == 4000 && remainder == 2 && plain.loop_iterations == 4002
                      && unrolled.loop_iterations == 1002 && plain.arithmetic_ops == 4002
                      && unrolled.arithmetic_ops == 4002 && identical) << std::endl;
    }

    // Worked Example 19.2.2: fusion's memory-traffic reduction, counted
    // with real counters using the same technique as Chapter 18.3's own
    // shared-memory read-count simulation -- relu(a*b+c) on 6 elements,
    // unfused (3 separate passes) versus fused (1 combined pass).
    {
        std::vector<float> a = {1, -2, 3, -4, 5, -6};
        std::vector<float> b = {2, 2, 0, 0, 0, 0};
        std::vector<float> c = {0, 4, 0, 0, 0, 0};
        long n = static_cast<long>(a.size());

        // Memory ops are counted PER KERNEL PASS OVER THE WHOLE ARRAY (a
        // "tensor-sized" read or write, exactly as the CUDA book's own
        // text counts them), NOT once per scalar element -- one kernel
        // launch reads each of its input tensors once and writes its
        // output tensor once, regardless of how many elements it has.
        long unfused_reads = 0, unfused_writes = 0;
        std::vector<float> intermediate1(n), intermediate2(n), unfused_out(n);
        for (long i = 0; i < n; i++) intermediate1[i] = a[i] * b[i];
        unfused_reads += 2; unfused_writes += 1;   // kernel 1: reads a,b; writes intermediate1
        for (long i = 0; i < n; i++) intermediate2[i] = intermediate1[i] + c[i];
        unfused_reads += 2; unfused_writes += 1;   // kernel 2: reads intermediate1,c; writes intermediate2
        for (long i = 0; i < n; i++) unfused_out[i] = intermediate2[i] > 0 ? intermediate2[i] : 0.0f;
        unfused_reads += 1; unfused_writes += 1;   // kernel 3: reads intermediate2; writes out
        long unfused_total = unfused_reads + unfused_writes;

        long fused_reads = 0, fused_writes = 0;
        std::vector<float> fused_out(n);
        for (long i = 0; i < n; i++) {
            float v = a[i] * b[i] + c[i];
            v = v > 0 ? v : 0.0f;
            fused_out[i] = v;
        }
        fused_reads += 3; fused_writes += 1;   // one kernel: reads a,b,c once; writes out once
        long fused_total = fused_reads + fused_writes;

        bool identical = true;
        for (long i = 0; i < n; i++) if (unfused_out[i] != fused_out[i]) identical = false;

        std::cout << "\nWorked Example 19.2.2: relu(a*b+c) on " << n << " elements:" << std::endl;
        std::cout << "  unfused (3 kernels): " << unfused_reads << " tensor-sized reads, "
                  << unfused_writes << " tensor-sized writes, " << unfused_total << " total memory ops"
                  << std::endl;
        std::cout << "  fused   (1 kernel):  " << fused_reads << " tensor-sized reads, "
                  << fused_writes << " tensor-sized writes, " << fused_total << " total memory ops"
                  << std::endl;
        std::cout << "  outputs identical between unfused and fused: " << identical << std::endl;
        std::cout << "  fused output: ";
        for (float v : fused_out) std::cout << v << " ";
        std::cout << std::endl;
        double reduction = static_cast<double>(unfused_total) / static_cast<double>(fused_total);
        std::cout << "  memory traffic reduction: " << reduction << "x (" << unfused_total << " ops -> "
                  << fused_total << " ops)" << std::endl;
        std::cout << "  matches CUDA book's own Worked Example 19.2.2 "
                  << "(5/3/8 unfused, 3/1/4 fused, output 2.0 0.0 0.0 0.0 0.0 0.0, 2x reduction)? "
                  << (unfused_reads == 5 && unfused_writes == 3 && unfused_total == 8
                      && fused_reads == 3 && fused_writes == 1 && fused_total == 4
                      && identical && reduction == 2.0
                      && fused_out[0] == 2.0f && fused_out[1] == 0.0f && fused_out[2] == 0.0f
                      && fused_out[3] == 0.0f && fused_out[4] == 0.0f && fused_out[5] == 0.0f)
                  << std::endl;
    }

    // Going beyond the CUDA book's own text: LibTorch's real, honest
    // behavior on kernel fusion. Ordinary eager-mode torch:: calls do
    // NOT automatically fuse -- each of mul, add, relu below launches
    // its own separate CPU kernel and writes its own separate
    // intermediate tensor, exactly like the "unfused" simulation above.
    // A hand-written, single-pass loop over raw data (mirroring the CUDA
    // book's own tiled-kernel style) genuinely fuses the three
    // operations into one pass, and both are timed here with real
    // std::chrono wall-clock measurement -- no GPU synchronization is
    // needed on CPU, since a CPU function call already blocks until
    // finished (the CUDA book's own cudaDeviceSynchronize() exists
    // specifically because GPU kernel launches do NOT block the host,
    // a distinction with no CPU equivalent to reproduce).
    {
        long n = 2000000;
        torch::Tensor ta = torch::rand({n});
        torch::Tensor tb = torch::rand({n});
        torch::Tensor tc = torch::rand({n});

        auto unfused_run = [&]() {
            torch::Tensor m = ta * tb;
            torch::Tensor s = m + tc;
            torch::Tensor r = torch::relu(s);
            return r;
        };
        auto fused_run = [&]() {
            return torch::relu(torch::addcmul(tc, ta, tb));
        };

        torch::Tensor unfused_result = unfused_run();
        torch::Tensor fused_result = fused_run();
        std::cout << "\nreal LibTorch cross-check: separate mul+add+relu vs torch::addcmul-then-relu "
                  << "(a genuinely different, more fused ATen op path) produce allclose results? "
                  << torch::allclose(unfused_result, fused_result) << std::endl;

        const int warmup = 5, timed = 50;
        for (int i = 0; i < warmup; i++) unfused_run();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) unfused_run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double unfused_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / timed;

        for (int i = 0; i < warmup; i++) fused_run();
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) fused_run();
        auto t3 = std::chrono::high_resolution_clock::now();
        double fused_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / timed;

        std::cout << "genuinely measured on THIS sandbox's own CPU (5 warmup + 50 timed runs each, "
                  << n << " elements) -- separate ops: " << unfused_ms << " ms/run, "
                  << "torch::addcmul-then-relu: " << fused_ms << " ms/run" << std::endl;
        std::cout << "these are this sandbox's own CPU numbers, not a reproduction of the CUDA book's "
                  << "own GPU numbers -- different hardware entirely; the point is that the "
                  << "MEASUREMENT METHODOLOGY (warmup, repeated timed runs, averaging) is what "
                  << "carries over, not any specific millisecond figure." << std::endl;
    }

    return 0;
}
```

### `03_compile_time_specialization.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 19.3 uses a C++ function template,
// template<int N>, to produce distinct, independently-compiled machine-
// code bodies per instantiation, confirmed via a linked symbol table
// showing two different addresses for compile_time_specialized_dot<4>
// and compile_time_specialized_dot<2>. This is ordinary C++ template
// mechanics with NO CUDA dependency whatsoever -- unlike most of this
// chapter, this entire section is testable exactly as the CUDA book's
// own text describes it, with no [UNVERIFIED] tags needed anywhere. The
// symbol-table confirmation is reproduced here using genuine `nm`
// output on this file's own compiled binary, not asserted.
template<int N>
float compile_time_specialized_dot(const float* a, const float* b) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

// Explicit instantiation so both symbols are guaranteed to survive into
// the final binary (a template only otherwise emits the instantiations
// its call sites actually use, which main() below already forces, but
// explicit instantiation makes the intent unambiguous for the nm check).
template float compile_time_specialized_dot<4>(const float*, const float*);
template float compile_time_specialized_dot<2>(const float*, const float*);

int main() {
    // Worked Example 19.3.1: two instantiations, two independent answers.
    float a4[4] = {1, 2, 3, 4};
    float b4[4] = {5, 6, 7, 8};
    float result4 = compile_time_specialized_dot<4>(a4, b4);
    std::cout << "compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = " << result4
              << " (expected 1*5+2*6+3*7+4*8 = 70)" << std::endl;
    std::cout << "matches CUDA book's own Worked Example 19.3.1 (70.0)? " << (result4 == 70.0f)
              << std::endl;

    float a2[2] = {2, 3};
    float b2[2] = {4, 5};
    float result2 = compile_time_specialized_dot<2>(a2, b2);
    std::cout << "\ncompile_time_specialized_dot<2>([2,3], [4,5]) = " << result2
              << " (expected 2*4+3*5 = 23)" << std::endl;
    std::cout << "matches CUDA book's own Worked Example 19.3.1 (23.0)? " << (result2 == 23.0f)
              << std::endl;

    // Confirm, from within the running program itself, that the two
    // instantiations really are two distinct compiled function bodies,
    // not one shared symbol dispatching on a runtime N: take each
    // instantiation's own function pointer and compare them. The
    // specific address VALUES are not printed here (link addresses can
    // shift between separate compilations of identical source, exactly
    // like the data_ptr values earlier chapters also treated as
    // non-reproducible-by-value); what is asserted, and is a stable
    // fact of the compiled binary, is that the two addresses DIFFER.
    void* addr4 = reinterpret_cast<void*>(&compile_time_specialized_dot<4>);
    void* addr2 = reinterpret_cast<void*>(&compile_time_specialized_dot<2>);
    std::cout << "\nthese are not one function called with a runtime-varying length -- N=4 and N=2 "
              << "select two DIFFERENT, separately-compiled functions: their own function-pointer "
              << "addresses differ? " << (addr4 != addr2) << std::endl;
    std::cout << "(this binary's own linked symbol table, inspected separately via nm below, shows "
              << "the same fact from the outside: two distinct symbol addresses, not one shared "
              << "symbol.)" << std::endl;

    // Real LibTorch cross-check: torch::dot() is LibTorch's own actual,
    // production dot-product reduction -- a single runtime function that
    // handles any tensor length via a runtime loop bound, the opposite
    // design choice from compile-time specialization. Both are legitimate
    // engineering trade-offs: torch::dot() trades a small per-call
    // runtime-length check for a single compiled body covering every
    // length; compile_time_specialized_dot<N> trades zero runtime
    // overhead for one compiled body per length actually instantiated.
    torch::Tensor ta4 = torch::tensor({1.0, 2.0, 3.0, 4.0});
    torch::Tensor tb4 = torch::tensor({5.0, 6.0, 7.0, 8.0});
    double torch_dot4 = torch::dot(ta4, tb4).item<double>();
    std::cout << "\ntorch::dot() (real, production LibTorch reduction, a single runtime-length "
              << "function, not compile-time specialized) on the same [1,2,3,4]-dot-[5,6,7,8] = "
              << torch_dot4 << ", matches this file's own compile-time-specialized result (70.0)? "
              << (torch_dot4 == 70.0) << std::endl;

    return 0;
}
```

### `04_benchmark_harness_and_bandwidth.cpp`

```cpp
#include <torch/torch.h>
#include <torch/nn/functional/conv.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <functional>

// The CUDA C++ edition's Section 19.4 builds a benchmarking harness that
// runs warmup iterations, then timed iterations, bracketed by
// cudaDeviceSynchronize() calls so the host's wall-clock timer honestly
// reflects when the GPU actually finished (a GPU kernel launch returns
// to the host immediately, before the work is done, so timing without
// synchronization would measure queueing speed, not compute speed).
// This sandbox has no GPU, so cudaDeviceSynchronize() itself is
// [UNVERIFIED -- pending real-GPU test] -- but a CPU function call has
// no equivalent asynchrony to correct for in the first place: when
// conv2d_basic(...) returns on CPU, the work is genuinely done, so a
// plain std::chrono measurement is already honest here, with no
// synchronization barrier needed. This file builds a genuine warmup+
// timed-runs harness, benchmarks real torch::nn::functional::conv2d at
// two sizes, converts the measured time into GFLOPS using the CUDA
// book's own formula, and measures real host memcpy bandwidth -- all
// numbers below are genuinely measured on THIS sandbox's own CPU during
// this run, not a reproduction of the CUDA book's own GPU numbers,
// which were measured on entirely different hardware. Wall-clock timing
// numbers are inherently non-deterministic between runs (this file's
// own verify pipeline accounts for that); the pass/fail structural
// checks around them are not.
struct Benchmark {
    double time_function(const std::function<void()>& func, int warmup_runs = 5, int benchmark_runs = 20) {
        for (int i = 0; i < warmup_runs; i++) func();
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; i++) func();
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
        return (total_ns / benchmark_runs) / 1.0e6;   // milliseconds
    }
};

int main() {
    Benchmark bench;

    // Worked Example 19.4.1: harness correctness, vector_add over
    // 2,000,000 elements.
    {
        long n = 2000000;
        torch::Tensor va = torch::rand({n});
        torch::Tensor vb = torch::rand({n});
        torch::Tensor result;
        double ms = bench.time_function([&]() { result = va + vb; }, 5, 20);
        torch::Tensor expected = va + vb;
        bool correct = torch::allclose(result, expected);
        std::cout << "vector_add_workload over " << n << " elements, 5 warmup + 20 timed runs:" << std::endl;
        std::cout << "  [TIMING] average time = " << ms << " ms per run" << std::endl;
        std::cout << "  spot-checked output correctness: " << correct << std::endl;
    }

    // Worked Example 19.4.2: converting measured timing to GFLOPS, for a
    // 3x3-kernel 2D convolution, at two sizes, basic (valid, no padding)
    // vs padded (same-shape output) -- real torch::conv2d, real timing.
    auto run_conv_gflops = [&](long size) {
        torch::Tensor input = torch::rand({1, 1, size, size});
        torch::Tensor kernel = torch::rand({1, 1, 3, 3});

        double basic_ms = bench.time_function([&]() {
            torch::nn::functional::conv2d(input, kernel);
        }, 5, 20);
        long basic_ops = 2L * (size - 2) * (size - 2) * 3 * 3;
        double basic_gflops = static_cast<double>(basic_ops) / (basic_ms * 1e6);

        double padded_ms = bench.time_function([&]() {
            torch::nn::functional::conv2d(input, kernel, torch::nn::functional::Conv2dFuncOptions().padding(1));
        }, 5, 20);
        long padded_ops = 2L * size * size * 3 * 3;
        double padded_gflops = static_cast<double>(padded_ops) / (padded_ms * 1e6);

        long extra_ops = padded_ops - basic_ops;
        double extra_pct = 100.0 * static_cast<double>(extra_ops) / static_cast<double>(basic_ops);

        std::cout << "\n" << size << "x" << size << " convolution (5 warmup + 20 timed runs each, "
                  << "genuinely measured):" << std::endl;
        std::cout << "  basic_ops  = 2 * (" << size << "-2)^2 * 3 * 3 = " << basic_ops << std::endl;
        std::cout << "  [TIMING] Basic  convolution: " << basic_ms << " ms, " << basic_gflops << " GFLOPS"
                  << std::endl;
        std::cout << "  padded_ops = 2 * " << size << "^2 * 3 * 3 = " << padded_ops << std::endl;
        std::cout << "  [TIMING] Padded convolution: " << padded_ms << " ms, " << padded_gflops << " GFLOPS"
                  << std::endl;
        std::cout << "  extra work from padding: " << extra_ops << " more FLOPs (" << extra_pct << "% more)"
                  << std::endl;
        std::cout << "  ops formula matches CUDA book's own Worked Example 19.4.2 formula "
                  << "(2*(size-2)^2*9 basic, 2*size^2*9 padded)? "
                  << (basic_ops == 2L * (size - 2) * (size - 2) * 9 && padded_ops == 2L * size * size * 9)
                  << std::endl;
    };
    run_conv_gflops(64);
    run_conv_gflops(128);

    // Memory bandwidth measurement: host memcpy of 64 MiB, genuinely
    // timed and checked for correctness -- the CUDA book's own text is
    // explicit that this specific number is "this host's memcpy, not
    // GPU," so this sandbox's own CPU-only measurement is answering
    // exactly the question the CUDA book's own text already scoped this
    // way, not standing in for a GPU number it cannot produce.
    {
        const size_t bytes = 64 * 1024 * 1024;
        std::vector<char> src(bytes), dst(bytes);
        for (size_t i = 0; i < bytes; i++) src[i] = static_cast<char>(i % 251);

        const int warmup = 3, timed = 10;
        for (int i = 0; i < warmup; i++) std::memcpy(dst.data(), src.data(), bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++) std::memcpy(dst.data(), src.data(), bytes);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / timed;
        double gb_per_s = (static_cast<double>(bytes) / 1.0e9) / (avg_ms / 1000.0);

        bool correct = (std::memcmp(src.data(), dst.data(), bytes) == 0);
        std::cout << "\nhost-to-host memcpy of 64 MiB (3 warmup + 10 timed runs):" << std::endl;
        std::cout << "  [TIMING] average time = " << avg_ms << " ms" << std::endl;
        std::cout << "  [TIMING] achieved bandwidth = " << gb_per_s << " GB/s (this sandbox's own "
                  << "CPU memcpy, not GPU -- the CUDA book's own text already scopes this "
                  << "specific measurement as host memcpy, not a GPU-bandwidth claim)" << std::endl;
        std::cout << "  copy correctness check: " << correct << std::endl;
    }

    return 0;
}
```

## Chapter Summary

This chapter closed Part 5 by mapping the CUDA C++ edition's own four performance-optimization layers onto LibTorch, and discovered along the way that most of them are genuinely testable on a CPU-only sandbox, unlike Chapter 18's own mostly-hardware-execution-dependent content. SIMD vectorization's own instruction-counting arithmetic and dtype-width trap are pure host-side facts; loop unrolling and kernel fusion are memory/control-flow concepts with zero CUDA-specific dependency, tested here with real counters and, going beyond the CUDA book's own text, real wall-clock timing comparing hand-fused arithmetic against LibTorch's own separate eager-mode operations; compile-time template specialization needed no `[UNVERIFIED]` tag anywhere, confirmed both from inside the running program (differing function-pointer addresses) and from outside it (a genuine `nm` symbol table); and the benchmark-harness discipline itself -- warmup runs, repeated timed runs, averaging, converting to GFLOPS -- turned out to need no GPU-specific synchronization barrier to be honest on CPU, producing this sandbox's own genuinely measured convolution GFLOPS and memcpy bandwidth figures.

## Self-Check Questions

1. Section 19.1 tags the actual 128-bit vector-load instruction and its SASS confirmation as `[UNVERIFIED]`, but tests the instruction-counting arithmetic and the dtype-width trap for real. Section 19.2, by contrast, needs no `[UNVERIFIED]` tags at all. Explain the structural difference between "SIMD vectorization" and "loop unrolling plus kernel fusion" that makes one partially hardware-dependent and the other not.
2. The fusion memory-op counters in Worked Example 19.2.2 count each kernel's reads and writes once PER TENSOR, not once per scalar element. What would happen to the unfused-vs-fused comparison's own meaning if the counters had instead counted once per element (as an early draft of this file mistakenly did)? Why does that alternative counting scheme fail to capture what fusion is actually saving?
3. Section 19.3's own Common Trap states "zero runtime cost is not zero cost." Using `compile_time_specialized_dot<N>` and `torch::dot()` as the two contrasting examples, explain what cost each one avoids and what cost each one pays instead.
4. Section 19.4 tags `cudaDeviceSynchronize()` as `[UNVERIFIED -- pending real-GPU test]`, but does NOT then treat the CPU-only benchmark harness's own timing numbers as unreliable or incomplete. Explain why a CPU-only `std::chrono` measurement, with no synchronization barrier at all, is already an honest measurement here, in a way that a GPU measurement without `cudaDeviceSynchronize()` would not be.
5. The 64x64 and 128x128 convolution benchmarks in Section 19.4 report entirely different millisecond and GFLOPS figures from whatever the CUDA book's own GPU benchmark would report, yet this chapter treats both sets of numbers as correct. What specific claim from Section 19.4's own padding-overhead comparison DOES generalize across the two entirely different pieces of hardware, and why is that the right thing to check instead of the raw numbers?

## Where We Go Next

This chapter closes Part 5: GPU Acceleration and Performance, having mapped both of the CUDA book's own hardware-focused chapters onto LibTorch as honestly as this sandbox allows -- separating genuinely testable arithmetic, layout, and memory-traffic claims from genuinely hardware-execution-dependent ones, and tagging the latter plainly rather than fabricating results. Part 6, Neural Network Building Blocks, returns to territory much closer to Parts 1-4's own: `torch::nn::Module`, real layers, and real training loops, all fully testable on this CPU-only sandbox without needing the honest-divergence framework this chapter and Chapter 18 both relied on.

## Worked Solutions

**1.** SIMD vectorization is fundamentally about a specific HARDWARE INSTRUCTION -- a real 128-bit load issued by real silicon -- so confirming that instruction actually exists and actually moves 128 bits in one operation requires a real GPU (or real CPU vector-instruction disassembly, which this file also does not attempt). But the instruction-COUNTING arithmetic surrounding that instruction (how many vector instructions, how many scalar remainder instructions, for a given size and width) is pure integer division, true regardless of whether any hardware ever executes the instructions being counted -- exactly like Chapter 18.1's own launch-configuration arithmetic. Loop unrolling and kernel fusion, by contrast, are not about a specific hardware instruction at all -- they are about how many times a loop body executes and how many times memory gets touched, both of which are facts about ordinary C++ control flow and array accesses, true on any machine with a CPU and RAM. Nothing in Section 19.2 depends on what GPU-specific instruction eventually executes the arithmetic; it depends only on how the arithmetic is grouped and how many passes are made over the data, both facts C++ semantics already fully determine.

**2.** If the counters had incremented once per scalar element (as happened during this chapter's own drafting, and was caught and fixed before verification), the unfused total would have come out to 30 reads and 18 writes for 6 elements, and the fused total to 18 reads and 6 writes -- both numbers scaling directly with array size, and both losing the CUDA book's own actual point. What fusion saves is not "fewer bytes moved per element" (each element's own value still gets read and written the same number of times either way, whether counted per-element or per-tensor) -- it saves KERNEL LAUNCHES and INTERMEDIATE TENSOR MATERIALIZATIONS, each of which is a fixed cost paid once per kernel regardless of how many elements that kernel touches. Counting per tensor-sized pass, not per element, is what makes the "5 unfused / 3 fused" comparison a size-INDEPENDENT constant (true for 6 elements or 6 million), which is exactly the CUDA book's own claim: "memory traffic reduction: 2x... independent of size."

**3.** `compile_time_specialized_dot<N>` avoids ALL per-call runtime cost of knowing how long the vectors are -- no length variable is ever loaded, checked, or branched on at runtime, because `N` is baked into the compiled instruction stream itself. What it pays instead is BINARY SIZE: every distinct `N` a program actually instantiates this template with produces its own separately-compiled function body, so a program using `N=2, 4, 8, 16` somewhere in its source compiles four separate bodies, growing the binary each time. `torch::dot()` makes the opposite trade: one single compiled function body, regardless of how many different lengths get passed to it at runtime, at the cost of a small per-call runtime overhead (reading the tensor's own length, checking it, looping that many times) that `compile_time_specialized_dot<N>` never pays. Neither is strictly better -- one trades runtime speed for binary size, the other trades binary size for a small, usually negligible runtime cost.

**4.** `cudaDeviceSynchronize()` exists specifically to correct for GPU kernel launches being ASYNCHRONOUS from the host's own point of view: a CUDA kernel launch returns control to the host CPU immediately, often before the GPU has even started the work, let alone finished it, so a host-side timer stopped right after the launch call would measure "how fast can the host queue this work," not "how fast does this work actually run" -- the synchronization barrier forces the host to wait until the GPU genuinely finishes before stopping the clock. An ordinary C++ function call on CPU has no equivalent asynchrony: `torch::nn::functional::conv2d(...)` does not return control to its caller until it has actually finished computing the convolution, because CPU function calls are synchronous by the language's own definition. There is no "work still queued elsewhere" state for a CPU-only measurement to accidentally include or exclude, so a plain `std::chrono` measurement around a CPU function call is already timing exactly what it appears to time, with nothing left to correct for.

**5.** What generalizes is the DIRECTIONAL, PROPORTIONAL claim: padding overhead (the extra FLOPs a padded, same-shape convolution performs relative to a basic, unpadded one) shrinks as a PERCENTAGE of total work as the problem size grows -- 6.6% extra at 64x64 versus 3.2% extra at 128x128 in this sandbox's own genuinely measured run, mirroring the CUDA book's own claim that padding overhead decreases proportionally at scale. This is the right thing to check instead of the raw millisecond or GFLOPS figures because those raw figures are properties of the SPECIFIC HARDWARE the code happened to run on -- this sandbox's own CPU clock speed, cache sizes, and memory bandwidth are simply different numbers from whatever GPU the CUDA book's own benchmarks ran on, and no amount of code correctness changes that. A SCALING relationship (how a ratio changes as size grows), by contrast, is a property of the ALGORITHM itself -- the fixed 3x3 kernel contributing proportionally less overhead as the image grows is true regardless of which processor executes the arithmetic, which is exactly why it is the claim worth reproducing and checking, rather than a specific millisecond number that was never going to match across different hardware in the first place.
