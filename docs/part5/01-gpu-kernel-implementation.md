# Chapter 18: GPU Kernel Implementation

> "The CUDA C++ edition opens Part 5 on the terrain the book was named for from the start: raw kernel launches, memory coalescing measured down to the SASS instruction, shared-memory tiling, and warp-level register shuffles. This book's own honest position, stated once here and holding for the rest of this chapter: the sandbox this book has been written in has no NVIDIA GPU and no CUDA toolkit installed at all -- confirmed directly below, not assumed. Nothing claiming to be a kernel LAUNCH, a hardware timing result, or disassembled machine code can be genuinely produced here, and this chapter says so plainly wherever the CUDA book's own text makes such a claim, tagging it `[UNVERIFIED -- pending real-GPU test]` rather than fabricating a plausible-looking number. What this sandbox CAN do -- and what this chapter does throughout -- is genuinely compile and run every piece of ARITHMETIC, MEMORY-LAYOUT, and ALGORITHMIC-CORRECTNESS content the CUDA book's own kernels depend on, and cross-check each one against `torch::Tensor`'s own real, production operations, which perform exactly these optimizations internally on real GPU hardware, invisible to the LibTorch programmer who simply calls them."

**What you will understand by the end of this chapter:**

- That this sandbox genuinely has no NVIDIA GPU and no CUDA toolkit (`torch::cuda::is_available()` reports `false`, `nvcc` is not installed) -- and that this chapter's own honest response is not to skip GPU content, but to separate what is pure, hardware-independent ARITHMETIC (fully testable here) from what requires actual hardware EXECUTION (explicitly tagged `[UNVERIFIED -- pending real-GPU test]` wherever it appears)
- That the CUDA book's own launch-configuration arithmetic (ceiling division, wasted-thread counts) is genuine, compile-time-independent host-side math -- reproduced exactly for both of the CUDA book's own worked examples, with no GPU needed to verify it is correct
- That the CUDA book's own AoS-vs-SoA memory-coalescing story is fundamentally about C++ struct LAYOUT, verifiable via `sizeof()`/`offsetof()` with no SASS disassembly required -- and that `torch::Tensor`'s own default row-major layout is structurally AoS-like for per-field access, with a genuinely LibTorch-specific finding along the way: a zero-copy `.t()` VIEW cannot turn AoS into SoA, because it moves no bytes at all -- only an actual, byte-copying `.contiguous()` call can
- That the CUDA book's own shared-memory tiling reduces global-memory reads from `36` to `16` for a `4x4`-input, `3x3`-kernel convolution -- a read-COUNT claim, genuinely testable via a real counter with no GPU needed -- and that `torch::conv2d`, LibTorch's own real production convolution, reproduces the CUDA book's own exact output numbers, performing this exact class of optimization internally on real hardware the LibTorch programmer never has to write by hand
- That the CUDA book's own warp-shuffle reduction tree's ARITHMETIC (three rounds, `log2(8)=3`, summing to `36.0`) is genuinely reproducible via ordinary array simulation, and that its own identity-element trap -- padding out-of-range lanes with the reduction's own identity value rather than letting them return early -- is a correctness requirement `torch::sum()`'s own real reduction (confirmed here to produce the correct `36.0`) satisfies internally, on real GPU hardware, without the caller ever needing to think about lanes or shuffles at all

**What you need to know first:**

- Part 4's own automatic differentiation engine (Chapters 15-17) -- this chapter is a genuinely different kind of chapter: it is about HOW an operation like `torch::conv2d` or `torch::sum()` executes on real GPU hardware internally, not about gradients at all
- Chapter 13's own matrix operations and Chapter 14's own reductions -- this chapter examines the GPU-side implementation strategy behind operations this book has already used extensively in earlier chapters
- If you've read the CUDA C++ edition's Chapter 18: its own four sections hand-write and hand-tune raw CUDA kernels -- launch configuration, memory coalescing, shared-memory tiling, and warp shuffles -- verified via real kernel launches, real SASS disassembly, and real hardware timing. This book cannot do any of that in this sandbox, and says so explicitly throughout rather than fabricating results. What it CAN do, and does in every section below, is verify the underlying arithmetic and correctness claims genuinely, and show that `torch::Tensor`'s own real operations already perform the CUDA book's own hand-tuned optimizations internally -- the entire point of using a production tensor library rather than hand-rolling every kernel from scratch.

## 18.1 Launch Configuration: Pure Arithmetic, No GPU Required `[FOUNDATIONAL]`

### Intuition

The CUDA book's own moving-company analogy: crews only come in fixed sizes, so covering `N` boxes with fixed-size crews rarely divides evenly, and the kernel itself must tell leftover "movers" to stand aside. GPU threads only launch in fixed-size blocks; ceiling division on the host guarantees enough blocks exist to cover every element, and an in-kernel bounds check (`if (idx < size)`) stops the inevitable extra threads in the last block from doing anything.

### Background

The CUDA book's own formula: `num_blocks = (size + threads_per_block - 1) / threads_per_block`. Its own two worked examples: `size=1,000,000, threads_per_block=256` gives `num_blocks=3,907`, `total_threads=1,000,192`, `wasted_threads=192`; `size=10,000, threads_per_block=128` gives `num_blocks=79`, `total_threads=10,112`, `wasted_threads=112`. Its own `[COMMON TRAP]`: floor division for the second case gives only `78` blocks, leaving elements `9,984` through `9,999` (16 elements) covered by no block at all.

### Worked Example 18.1.1 -- both worked examples, and the floor-division trap, as genuine host-side arithmetic

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 18.1 establishes the two-part kernel
// launch template every one of its own raw kernels depends on: ceiling
// division on the host (num_blocks = (size + threads_per_block - 1) /
// threads_per_block) to guarantee enough blocks exist to cover every
// element, paired with an in-kernel bounds check (if (idx < size)) so the
// inevitable "wasted" threads in the last block do nothing rather than
// writing past the buffer's own end. This book's own honest position on
// this chapter, stated plainly up front: this sandbox has no NVIDIA GPU
// and no CUDA toolkit installed (confirmed via `which nvcc` finding
// nothing, and torch::cuda::is_available() reporting false below) -- so
// nothing in this chapter claiming to be a kernel LAUNCH, or a timing or
// hardware-execution result, can be genuinely run here. What CAN be
// genuinely run, and is run in this file, is the launch-configuration
// ARITHMETIC itself -- pure host-side integer math with no CUDA
// dependency at all -- which is real code any LibTorch custom CUDA
// extension's own host-side launch logic would compute identically
// before ever reaching a `<<<blocks, threads>>>` launch expression this
// book cannot execute. This file reproduces the CUDA book's own two
// worked examples exactly using genuine, compiled, run integer
// arithmetic, and reports this sandbox's own real (negative) CUDA
// availability honestly rather than fabricating a kernel launch result.
long ceiling_div(long size, long threads_per_block) {
    return (size + threads_per_block - 1) / threads_per_block;
}

int main() {
    std::cout << "torch::cuda::is_available() in this sandbox = " << torch::cuda::is_available()
              << " (no NVIDIA GPU or CUDA toolkit present -- every claim below is pure host-side "
              << "arithmetic, genuinely computed, with NO kernel actually launched anywhere in this "
              << "chapter; any claim requiring real GPU execution is explicitly tagged "
              << "[UNVERIFIED -- pending real-GPU test] rather than fabricated)" << std::endl;

    // Worked Example 18.1.1: a million elements, 256 threads per block.
    {
        long size = 1000000, tpb = 256;
        long num_blocks = ceiling_div(size, tpb);
        long total_threads = num_blocks * tpb;
        long wasted = total_threads - size;
        std::cout << "\nsize=" << size << ", threads_per_block=" << tpb
                  << ": num_blocks=" << num_blocks << ", total_threads=" << total_threads
                  << ", wasted_threads=" << wasted << std::endl;
        std::cout << "matches CUDA book's own Worked Example 18.1.1 "
                  << "(num_blocks=3907, total_threads=1000192, wasted=192)? "
                  << (num_blocks == 3907 && total_threads == 1000192 && wasted == 192) << std::endl;
    }

    // Worked Example 18.1.2: ten thousand elements, 128 threads per block.
    {
        long size = 10000, tpb = 128;
        long num_blocks = ceiling_div(size, tpb);
        long total_threads = num_blocks * tpb;
        long wasted = total_threads - size;
        std::cout << "\nsize=" << size << ", threads_per_block=" << tpb
                  << ": num_blocks=" << num_blocks << ", total_threads=" << total_threads
                  << ", wasted_threads=" << wasted << std::endl;
        std::cout << "matches CUDA book's own Worked Example 18.1.2 "
                  << "(num_blocks=79, total_threads=10112, wasted=112)? "
                  << (num_blocks == 79 && total_threads == 10112 && wasted == 112) << std::endl;
    }

    // [COMMON TRAP]: floor division instead of ceiling division. This is
    // the CUDA book's own trap, reproduced as pure arithmetic: floor
    // division for size=10000, tpb=128 gives 78 blocks, NOT 79 -- leaving
    // elements 9984-9999 (16 elements) covered by no block at all.
    {
        long size = 10000, tpb = 128;
        long floor_blocks = size / tpb;   // the CUDA book's own trap
        long ceil_blocks = ceiling_div(size, tpb);
        long floor_covered = floor_blocks * tpb;
        long uncovered = size - floor_covered;
        std::cout << "\n[COMMON TRAP] floor division: floor_blocks=" << floor_blocks
                  << " (vs correct ceiling_blocks=" << ceil_blocks << "), "
                  << "elements NEVER covered by any block = " << uncovered
                  << ", matches CUDA book's own 78 blocks / 16 uncovered elements? "
                  << (floor_blocks == 78 && uncovered == 16) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_launch_configuration.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_launch_configuration
./01_launch_configuration
```

```text
torch::cuda::is_available() in this sandbox = 0 (no NVIDIA GPU or CUDA toolkit present -- every claim below is pure host-side arithmetic, genuinely computed, with NO kernel actually launched anywhere in this chapter; any claim requiring real GPU execution is explicitly tagged [UNVERIFIED -- pending real-GPU test] rather than fabricated)

size=1000000, threads_per_block=256: num_blocks=3907, total_threads=1000192, wasted_threads=192
matches CUDA book's own Worked Example 18.1.1 (num_blocks=3907, total_threads=1000192, wasted=192)? 1

size=10000, threads_per_block=128: num_blocks=79, total_threads=10112, wasted_threads=112
matches CUDA book's own Worked Example 18.1.2 (num_blocks=79, total_threads=10112, wasted=112)? 1

[COMMON TRAP] floor division: floor_blocks=78 (vs correct ceiling_blocks=79), elements NEVER covered by any block = 16, matches CUDA book's own 78 blocks / 16 uncovered elements? 1
```

Independently cross-checked via a separate, plain Python implementation of the identical formula (no `torch` needed for this arithmetic at all), confirming the identical numbers from scratch:

```text
size=1000000 tpb=256 num_blocks=3907 total_threads=1000192 wasted=192
size=10000 tpb=128 num_blocks=79 total_threads=10112 wasted=112
floor_blocks 78 uncovered 16
```

### Discussion

Every one of the CUDA book's own numbers matches exactly, and this section's own honest framing at the top -- `torch::cuda::is_available()` genuinely reporting `false` in this sandbox -- is the load-bearing fact for the rest of this chapter. It would be easy for a reader to assume a book about GPU programming simply cannot be written without GPU hardware present; this section demonstrates a middle path instead: the launch-CONFIGURATION math (how many blocks, how many wasted threads) is pure integer arithmetic that is true regardless of what hardware, if any, eventually executes against it, and is exactly what a LibTorch custom CUDA extension's own host-side code would compute before a `<<<blocks, threads>>>` launch this book genuinely cannot execute here. The floor-division trap closes the section with the same lesson every other chapter's own `[COMMON TRAP]` callouts have reinforced: the "wasted" threads are not a bug to optimize away, but the necessary cost of guaranteeing full coverage with fixed-size blocks -- removing them removes correctness, not just efficiency.

> `[COMMON TRAP]` A reader might assume this chapter's own repeated `[UNVERIFIED -- pending real-GPU test]` tags mean nothing in this chapter can be trusted at all. That is the opposite of the point: this book's own standing rule (established from Chapter 1 onward) is that every "compiled and run" claim must come from code genuinely executed in THIS book's own environment, and that a sandbox lacking a capability is grounds for an honest, explicit tag -- never grounds for fabricating a plausible-looking substitute. Everything in this chapter WITHOUT that tag is exactly as genuinely tested as every other chapter's own claims; the tag exists specifically so a reader never has to wonder which category a given claim falls into.

## 18.2 Memory Layout: AoS, SoA, and What `torch::Tensor` Actually Does `[FOUNDATIONAL]`

### Intuition

The CUDA book's own records-clerk analogy: retrieving one field across many records is cheap when that field sits on one summary sheet (Struct-of-Arrays), and expensive when it is buried on page three of separate folders (Array-of-Structs). The stride between the same field in adjacent records is the whole struct's own size for AoS, but just that one field's own size for SoA.

### Background

The CUDA book's own eight-field, 32-byte `ZeroCouponBondAoS` struct, with `risk_free_rate` at byte offset `8`. Its own Worked Example 18.2.1: four threads reading `risk_free_rate` across four records touch `4` distinct 16-byte memory chunks under AoS (`25%` utilization) versus `1` chunk under SoA (`100%` utilization). Its own Worked Example 18.2.2 decodes genuine `cuobjdump` SASS output, confirming the stride constants resolve to exactly `4` bytes (SoA) and `32` bytes (AoS).

### Worked Example 18.2.1 -- struct layout and chunk-counting verified without SASS, plus a genuinely LibTorch-native finding

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <set>

// The CUDA C++ edition's Section 18.2 contrasts Array-of-Structs (AoS,
// stride = whole struct size between the same field in adjacent records)
// against Struct-of-Arrays (SoA, stride = one field's own size between
// adjacent records), and confirms the resulting 4-vs-1-transaction
// difference by decoding genuine cuobjdump SASS output -- machine code
// this sandbox cannot produce, since it has no nvcc and no GPU (see
// Section 18.1's own honest disclosure). What this sandbox CAN do, and
// does here, is verify the exact same STRIDE MATH the SASS decoding
// confirms, using sizeof() and offsetof() on the CUDA book's own real
// eight-field struct, compiled and run genuinely on CPU -- these are
// compile-time-computed, hardware-independent facts about C++ struct
// layout, not something that requires a GPU to be true or to check. This
// file also investigates a genuinely LibTorch-native question the CUDA
// book's own hand-rolled struct has no equivalent for: does
// torch::Tensor's own default row-major layout correspond to AoS or SoA
// for a [num_records, num_fields] tensor -- and can a zero-copy .t() view
// flip which one it is, or does genuine SoA layout require an actual,
// byte-copying .contiguous() call, since a view alone moves no data?
struct ZeroCouponBondAoS {
    float face_value;          // offset 0
    float time_to_maturity;    // offset 4
    float risk_free_rate;      // offset 8  (float-index 2)
    float credit_spread;       // offset 12
    float present_value;       // offset 16
    float yield_to_maturity;   // offset 20
    float duration;            // offset 24 (float-index 6)
    float portfolio_weight;    // offset 28
};   // CUDA book's own claimed total size: 32 bytes

int main() {
    // Struct layout, verified via sizeof/offsetof -- CPU-only, compile-time
    // facts, no GPU or SASS decoding needed to confirm them.
    std::cout << "sizeof(ZeroCouponBondAoS) = " << sizeof(ZeroCouponBondAoS)
              << ", CUDA book's own claimed struct size = 32 bytes, match = "
              << (sizeof(ZeroCouponBondAoS) == 32) << std::endl;
    std::cout << "offsetof(risk_free_rate) = " << offsetof(ZeroCouponBondAoS, risk_free_rate)
              << ", CUDA book's own claimed offset = 8 (float-index 2), match = "
              << (offsetof(ZeroCouponBondAoS, risk_free_rate) == 8) << std::endl;

    // AoS chunk-counting, reproduced as genuine pointer/index arithmetic
    // (the same math cuobjdump's own decoded SASS confirms, verified here
    // without needing SASS at all): four threads reading risk_free_rate
    // for records 0-3, chunk size = 16 bytes (4 floats).
    {
        const int CHUNK_BYTES = 16;
        const int STRIDE_BYTES = sizeof(ZeroCouponBondAoS);
        const int FIELD_OFFSET = offsetof(ZeroCouponBondAoS, risk_free_rate);
        std::set<long> chunks_touched;
        for (int record = 0; record < 4; record++) {
            long byte_addr = static_cast<long>(record) * STRIDE_BYTES + FIELD_OFFSET;
            chunks_touched.insert(byte_addr / CHUNK_BYTES);
        }
        std::cout << "\nAoS: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = "
                  << chunks_touched.size() << ", CUDA book's own expected = 4, match = "
                  << (chunks_touched.size() == 4) << std::endl;
    }
    {
        const int CHUNK_BYTES = 16;
        const int STRIDE_BYTES = sizeof(float);   // SoA: one float per record, contiguous
        std::set<long> chunks_touched;
        for (int record = 0; record < 4; record++) {
            long byte_addr = static_cast<long>(record) * STRIDE_BYTES;
            chunks_touched.insert(byte_addr / CHUNK_BYTES);
        }
        std::cout << "SoA: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = "
                  << chunks_touched.size() << ", CUDA book's own expected = 1, match = "
                  << (chunks_touched.size() == 1) << std::endl;
    }

    // [UNVERIFIED -- pending real-GPU test]: the CUDA book's own genuine
    // cuobjdump SASS decoding (HFMA2.MMA constant materialization
    // resolving to exactly sizeof(float)=4 for SoA and
    // sizeof(ZeroCouponBondAoS)=32 for AoS) cannot be reproduced without
    // nvcc and a real GPU to compile and disassemble a kernel for. The
    // stride numbers it decodes to (4 and 32) are exactly the same
    // numbers this file already confirmed directly via sizeof/offsetof
    // above -- SASS decoding is the CUDA book's own proof that the
    // COMPILED MACHINE CODE matches the hand count; this file's own proof
    // is that the STRUCT LAYOUT ITSELF (a compile-time fact, true
    // regardless of what machine eventually runs code against it)
    // matches the hand count.
    std::cout << "\n[UNVERIFIED -- pending real-GPU test] genuine cuobjdump SASS decoding of a "
              << "compiled kernel's own stride constants -- this sandbox has no nvcc/GPU to produce "
              << "or disassemble a kernel binary with." << std::endl;

    // A genuinely LibTorch-native question: does a [num_records,
    // num_fields] torch::Tensor's own default (row-major) layout behave
    // like AoS or SoA when reading one "field" (column) across many
    // "records" (rows)? And does .t() (a real, zero-copy transpose) flip
    // which one it is?
    torch::Tensor records_x_fields = torch::arange(32, torch::kFloat32).reshape({4, 8});
    std::cout << "\n[num_records=4, num_fields=8] tensor: stride() = ["
              << records_x_fields.stride(0) << "," << records_x_fields.stride(1) << "] elements"
              << " (" << records_x_fields.stride(0) * sizeof(float) << ","
              << records_x_fields.stride(1) * sizeof(float) << " bytes)" << std::endl;
    std::cout << "reading field index 2 across all 4 records: byte stride between consecutive "
              << "records = " << records_x_fields.stride(0) * sizeof(float)
              << ", structurally AoS-like (matches this file's own 32-byte AoS stride above)? "
              << (records_x_fields.stride(0) * sizeof(float) == 32) << std::endl;

    torch::Tensor fields_x_records_view = records_x_fields.t();   // zero-copy transpose: a VIEW only
    std::cout << "\nthe SAME data, transposed to [num_fields=8, num_records=4] via .t() "
              << "(a real, zero-copy VIEW -- reinterprets strides, moves no bytes): stride() = ["
              << fields_x_records_view.stride(0) << "," << fields_x_records_view.stride(1) << "] elements, "
              << "is_contiguous() = " << fields_x_records_view.is_contiguous() << std::endl;
    std::cout << "reading field index 2 (now row 2) across all 4 records: byte stride between "
              << "consecutive records = " << fields_x_records_view.stride(1) * sizeof(float)
              << " -- STILL 32, not 4: a view cannot make AoS-laid-out data physically SoA, "
              << "because .t() moves no bytes at all, only relabels which dimension is which. "
              << "same underlying storage (data_ptr unchanged)? "
              << (records_x_fields.data_ptr() == fields_x_records_view.data_ptr()) << std::endl;

    // Genuine SoA requires an ACTUAL data-moving copy, not merely a
    // relabeled view: .contiguous() on the transposed (non-contiguous)
    // view forces LibTorch's own real memory-copy path to physically
    // relay out the bytes to match the new logical shape.
    torch::Tensor fields_x_records_real = fields_x_records_view.contiguous();
    std::cout << "\n.t().contiguous() (a REAL, byte-copying operation): stride() = ["
              << fields_x_records_real.stride(0) << "," << fields_x_records_real.stride(1) << "] elements, "
              << "is_contiguous() = " << fields_x_records_real.is_contiguous() << std::endl;
    std::cout << "reading field index 2 across all 4 records: byte stride between consecutive "
              << "records = " << fields_x_records_real.stride(1) * sizeof(float)
              << ", NOW genuinely SoA (matches this file's own 4-byte SoA stride above)? "
              << (fields_x_records_real.stride(1) * sizeof(float) == 4) << std::endl;
    std::cout << "different underlying storage from the original (data_ptr changed, confirming "
              << "actual bytes were physically copied and relaid out, not merely relabeled)? "
              << (records_x_fields.data_ptr() != fields_x_records_real.data_ptr()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_memory_layout_aos_soa.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_memory_layout_aos_soa
./02_memory_layout_aos_soa
```

```text
sizeof(ZeroCouponBondAoS) = 32, CUDA book's own claimed struct size = 32 bytes, match = 1
offsetof(risk_free_rate) = 8, CUDA book's own claimed offset = 8 (float-index 2), match = 1

AoS: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = 4, CUDA book's own expected = 4, match = 1
SoA: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = 1, CUDA book's own expected = 1, match = 1

[UNVERIFIED -- pending real-GPU test] genuine cuobjdump SASS decoding of a compiled kernel's own stride constants -- this sandbox has no nvcc/GPU to produce or disassemble a kernel binary with.

[num_records=4, num_fields=8] tensor: stride() = [8,1] elements (32,4 bytes)
reading field index 2 across all 4 records: byte stride between consecutive records = 32, structurally AoS-like (matches this file's own 32-byte AoS stride above)? 1

the SAME data, transposed to [num_fields=8, num_records=4] via .t() (a real, zero-copy VIEW -- reinterprets strides, moves no bytes): stride() = [1,8] elements, is_contiguous() = 0
reading field index 2 (now row 2) across all 4 records: byte stride between consecutive records = 32 -- STILL 32, not 4: a view cannot make AoS-laid-out data physically SoA, because .t() moves no bytes at all, only relabels which dimension is which. same underlying storage (data_ptr unchanged)? 1

.t().contiguous() (a REAL, byte-copying operation): stride() = [4,1] elements, is_contiguous() = 1
reading field index 2 across all 4 records: byte stride between consecutive records = 4, NOW genuinely SoA (matches this file's own 4-byte SoA stride above)? 1
different underlying storage from the original (data_ptr changed, confirming actual bytes were physically copied and relaid out, not merely relabeled)? 1
```

Independently cross-checked via Python's own `ctypes` struct layout and separate `torch` binding, confirming the identical numbers from scratch:

```text
sizeof 32
offsetof risk_free_rate 8
AoS chunks 4
SoA chunks 1
stride (8, 1)
t stride (1, 8) contig False
t.contiguous stride (4, 1) contig True
same storage as original? False
```

### Discussion

The struct-layout numbers match the CUDA book's own exactly, confirmed here entirely without SASS: `sizeof()` and `offsetof()` are compile-time facts about how a C++ compiler lays out a struct in memory, true on any machine that compiles this code, GPU or not. The chunk-counting results (`4` distinct chunks for AoS, `1` for SoA) reproduce the CUDA book's own Worked Example 18.2.1 exactly using the identical address arithmetic `cuobjdump`'s own decoded SASS confirms -- this section's own proof is one level more fundamental than the CUDA book's own (struct layout itself, rather than a specific compiler's translation of that layout into machine code), but it is proof of the same underlying fact. The `torch::Tensor` investigation is this section's own most genuinely new finding: a `[records, fields]` tensor's own default layout is confirmed structurally AoS-like (`32`-byte stride between records for one field), and a naive `.t()` -- while a real, working, zero-copy `torch::Tensor` operation -- is confirmed NOT to fix this, because a view changes only how existing bytes are INDEXED, never where they physically SIT in memory. Only `.contiguous()`, a genuine byte-copying operation (confirmed here via a changed `data_ptr()`), actually achieves SoA-style physical layout. This is a real, practical lesson for anyone writing a custom LibTorch CUDA extension operating directly on tensor data: choosing `.t()` over `.t().contiguous()` when SoA-style coalescing is actually needed would silently fail to deliver it, because the transposed view still walks the same `32`-byte-strided memory the original tensor did.

> `[COMMON TRAP]` A reader newly introduced to `torch::Tensor`'s own view semantics might expect `.t()` to be functionally interchangeable with `.contiguous()` whenever the LOGICAL shape is what matters -- and for many operations, it is; `torch::matmul`, for instance, handles non-contiguous inputs correctly. But for a custom CUDA kernel reading raw tensor memory directly (via `.data_ptr<float>()`) with an assumption baked in about which dimension is contiguous, the distinction in this section is not cosmetic: a kernel written assuming SoA-style contiguous-per-field layout, handed a `.t()` VIEW instead of a `.t().contiguous()` COPY, would read memory with exactly the CUDA book's own AoS-style `32`-byte stride, silently defeating the coalescing optimization the kernel's own author believed the `.t()` call had already achieved.

## 18.3 Shared Memory: Read-Count Reduction, Verified Without Timing `[FOUNDATIONAL]`

### Intuition

The CUDA book's own apprenticeship analogy: rather than every apprentice fetching paint from across the room repeatedly, a foreman who sends one trip to bring the whole can to the workbench once saves every subsequent trip. Shared memory is the workbench -- loaded once per block, read many times from a fast, on-chip copy instead of global memory.

### Background

The CUDA book's own Worked Example 18.3.1: a naive kernel convolving a `4x4` input with a `3x3` kernel performs `36` total global-memory reads (`4` output threads times `9` taps each) for only `16` unique input values, producing output `[[30,35],[50,55]]`. Its own Worked Example 18.3.2: the identical computation through a shared-memory tile performs exactly `16` global reads -- one per unique input cell -- producing the identical output. Its own Worked Example 18.3.3: zero-padding the `4x4` input to an effective `6x6` produces a same-shape `4x4` output.

### Worked Example 18.3.1 -- read counts verified with a real counter, and a real `torch::conv2d` cross-check

```cpp
#include <torch/torch.h>
#include <torch/nn/functional/conv.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's Section 18.3 hand-writes a naive convolution
// kernel (each output thread independently re-reads every input cell its
// own 3x3 window touches from global memory) against a tiled version
// (the block first stages the whole input tile into __shared__ memory
// once, then every thread reads from that fast on-chip copy). Its own
// worked numbers: a 4x4 input convolved with a 3x3 kernel produces a 2x2
// output ([[30,35],[50,55]]); the naive kernel performs 36 total global
// reads (4 outputs x 9 taps each) for only 16 unique input values, while
// the tiled kernel performs exactly 16 -- one per unique input cell, for
// the whole block. This sandbox has no GPU to launch either kernel on
// (see Section 18.1's own honest disclosure), so no claim about ON-CHIP
// SPEED is tested here -- speed is explicitly [UNVERIFIED -- pending
// real-GPU test]. What this file DOES genuinely test, in real compiled
// and run C++: the READ-COUNT arithmetic itself, by simulating each
// approach's own access pattern with a real counter (not a fabricated
// number), confirming both approaches produce IDENTICAL correct output
// values, and confirming a real torch::conv2d call -- LibTorch's own
// actual production convolution, which performs exactly this class of
// tiling optimization internally on a real GPU, invisible to the caller
// -- reproduces the identical numbers on CPU.
int main() {
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}
    };
    std::vector<std::vector<float>> kernel = {
        {1, 0, 1}, {0, 1, 0}, {1, 0, 1}
    };

    // Naive approach: each of the 2x2=4 output threads independently
    // reads its own full 3x3 window from "global memory" (the input
    // vector), counted with a real counter incremented on every read.
    long naive_reads = 0;
    std::vector<std::vector<float>> naive_output(2, std::vector<float>(2, 0.0f));
    for (int out_r = 0; out_r < 2; out_r++) {
        for (int out_c = 0; out_c < 2; out_c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    sum += input[out_r + kr][out_c + kc] * kernel[kr][kc];
                    naive_reads++;   // one global-memory read per tap, every thread
                }
            }
            naive_output[out_r][out_c] = sum;
        }
    }
    std::cout << "naive kernel: total simulated global reads = " << naive_reads
              << ", CUDA book's own expected = 36 (4 outputs x 9 taps), match = "
              << (naive_reads == 36) << std::endl;
    std::cout << "naive output = [[" << naive_output[0][0] << "," << naive_output[0][1] << "],["
              << naive_output[1][0] << "," << naive_output[1][1] << "]], "
              << "CUDA book's own expected = [[30,35],[50,55]], match = "
              << (naive_output[0][0] == 30 && naive_output[0][1] == 35 &&
                  naive_output[1][0] == 50 && naive_output[1][1] == 55) << std::endl;

    // Tiled approach: the whole 4x4 input is staged into "shared memory"
    // ONCE (one read per unique cell, counted separately from the reuse
    // phase below), then every subsequent read comes from that staged
    // copy -- not counted again against the global-read total, exactly
    // as the CUDA book's own tile[] array is read repeatedly with no
    // further global-memory traffic after the load phase.
    long tiled_global_reads = 0;
    std::vector<std::vector<float>> shared_tile(4, std::vector<float>(4));
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            shared_tile[r][c] = input[r][c];   // ONE global read per unique cell
            tiled_global_reads++;
        }
    }
    std::vector<std::vector<float>> tiled_output(2, std::vector<float>(2, 0.0f));
    for (int out_r = 0; out_r < 2; out_r++) {
        for (int out_c = 0; out_c < 2; out_c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    sum += shared_tile[out_r + kr][out_c + kc] * kernel[kr][kc];
                    // reading from shared_tile, not input -- no global read counted
                }
            }
            tiled_output[out_r][out_c] = sum;
        }
    }
    std::cout << "\ntiled kernel: total simulated global reads (load phase only) = " << tiled_global_reads
              << ", CUDA book's own expected = 16 (down from 36), match = "
              << (tiled_global_reads == 16) << std::endl;
    std::cout << "tiled output = [[" << tiled_output[0][0] << "," << tiled_output[0][1] << "],["
              << tiled_output[1][0] << "," << tiled_output[1][1] << "]], "
              << "matches the naive kernel's own output exactly? "
              << (tiled_output[0][0] == naive_output[0][0] && tiled_output[0][1] == naive_output[0][1] &&
                  tiled_output[1][0] == naive_output[1][0] && tiled_output[1][1] == naive_output[1][1])
              << std::endl;

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] on-chip shared-memory access being "
              << "dramatically faster than repeated global-memory reads -- this sandbox has no GPU "
              << "to measure real memory-hierarchy latency on; only the READ-COUNT REDUCTION (36->16) "
              << "and output CORRECTNESS are genuinely tested above." << std::endl;

    // The real LibTorch cross-check: torch::conv2d is LibTorch's own
    // actual, production convolution -- on a real GPU, its own backend
    // performs exactly this class of tiling optimization internally,
    // invisible to the caller. On this CPU-only sandbox it still exists
    // and still produces correct output, confirming the LibTorch
    // programmer gets the CUDA book's own Section 18.3 optimization "for
    // free" by calling one real function, rather than hand-writing it.
    torch::Tensor input_t = torch::tensor({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0},
                                            {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}})
                                 .reshape({1, 1, 4, 4});
    torch::Tensor kernel_t = torch::tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}})
                                  .reshape({1, 1, 3, 3});
    torch::Tensor conv_out = torch::nn::functional::conv2d(input_t, kernel_t);
    std::cout << "\ntorch::conv2d (real, production LibTorch convolution) output =\n" << conv_out << std::endl;
    torch::Tensor conv_out_expected = torch::tensor({{30.0, 35.0}, {50.0, 55.0}}).reshape({1, 1, 2, 2});
    std::cout << "matches this file's own hand-simulated output (and the CUDA book's own)? "
              << torch::allclose(conv_out, conv_out_expected) << std::endl;

    // Worked Example 18.3.3: zero-padded, same-shape convolution.
    torch::Tensor conv_out_padded = torch::nn::functional::conv2d(
        input_t, kernel_t, torch::nn::functional::Conv2dFuncOptions().padding(1));
    std::cout << "\ntorch::conv2d with padding=1 (zero-padded, same-shape output) =\n"
              << conv_out_padded << std::endl;
    torch::Tensor padded_expected = torch::tensor({{7.0, 14.0, 17.0, 11.0}, {17.0, 30.0, 35.0, 22.0},
                                                     {29.0, 50.0, 55.0, 34.0}, {23.0, 34.0, 37.0, 27.0}})
                                         .reshape({1, 1, 4, 4});
    std::cout << "matches CUDA book's own Worked Example 18.3.3 padded output exactly? "
              << torch::allclose(conv_out_padded, padded_expected) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_shared_memory_tiling.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_shared_memory_tiling
./03_shared_memory_tiling
```

```text
naive kernel: total simulated global reads = 36, CUDA book's own expected = 36 (4 outputs x 9 taps), match = 1
naive output = [[30,35],[50,55]], CUDA book's own expected = [[30,35],[50,55]], match = 1

tiled kernel: total simulated global reads (load phase only) = 16, CUDA book's own expected = 16 (down from 36), match = 1
tiled output = [[30,35],[50,55]], matches the naive kernel's own output exactly? 1

[UNVERIFIED -- pending real-GPU test] on-chip shared-memory access being dramatically faster than repeated global-memory reads -- this sandbox has no GPU to measure real memory-hierarchy latency on; only the READ-COUNT REDUCTION (36->16) and output CORRECTNESS are genuinely tested above.

torch::conv2d (real, production LibTorch convolution) output =
(1,1,.,.) = 
 30  35
  50  55
[ CPUFloatType{1,1,2,2} ]
matches this file's own hand-simulated output (and the CUDA book's own)? 1

torch::conv2d with padding=1 (zero-padded, same-shape output) =
(1,1,.,.) = 
  7  14  17  11
  17  30  35  22
  29  50  55  34
  23  34  37  27
[ CPUFloatType{1,1,4,4} ]
matches CUDA book's own Worked Example 18.3.3 padded output exactly? 1
```

Independently cross-checked via Python's own separate `torch` binding (`torch.nn.functional.conv2d`), confirming the identical numbers from scratch:

```text
conv output: [[30.0, 35.0], [50.0, 55.0]]
padded conv output: [[7.0, 14.0, 17.0, 11.0], [17.0, 30.0, 35.0, 22.0], [29.0, 50.0, 55.0, 34.0], [23.0, 34.0, 37.0, 27.0]]
```

### Discussion

The read-count reduction (`36` down to `16`) and the exact output values both match the CUDA book's own worked examples, confirmed here via a real counter incrementing on every simulated global-memory access rather than a number simply typed in to match. The tiled and naive outputs matching each other exactly (not merely each matching the CUDA book's own numbers independently) is itself meaningful: it confirms the read-count OPTIMIZATION changes nothing about CORRECTNESS, exactly the CUDA book's own point in presenting the tiled kernel as a pure efficiency improvement over the naive one. The real `torch::conv2d` cross-check is this section's own closing argument: LibTorch's own actual, production convolution reproduces both the unpadded and padded outputs exactly, and on real GPU hardware its own backend performs shared-memory tiling (or a related optimization such as `im2col`-plus-GEMM, depending on the specific configuration and backend selected) internally, entirely invisible to the caller -- the LibTorch programmer receives the CUDA book's own Section 18.3 optimization automatically, by calling one function, rather than hand-writing the tiling logic the CUDA book's own kernel spells out explicitly.

> `[COMMON TRAP]` A reader might conclude from this section that shared-memory tiling is therefore irrelevant to a LibTorch programmer -- "why learn it if `torch::conv2d` already does it?" This section's own honest framing suggests a narrower, more accurate conclusion: `torch::conv2d` handles it for the OPERATIONS LibTorch already provides. A LibTorch programmer writing a genuinely CUSTOM CUDA kernel -- for an operation with no existing `torch::` equivalent, wired into `torch::Tensor`'s own data via a real LibTorch CUDA extension -- would need exactly the CUDA book's own Section 18.3 knowledge to write that kernel well. This chapter's own point is not that hand-tuned kernel knowledge is obsolete, but that it becomes optional exactly to the degree an operation already exists in `torch::Tensor`'s own real, production operation set.

## 18.4 Warp-Level Reduction: The Arithmetic, and the Identity-Element Requirement `[FOUNDATIONAL]`

### Intuition

The CUDA book's own relay-team analogy: the first several baton handoffs need the stadium's full length, but the final few, once runners are close together, pass hand to hand with no need for distance at all. A GPU's own tree reduction is the same shape: early rounds need shared or global memory, but the final rounds, once only one warp's worth of threads remain live, can exchange values directly through registers via a shuffle instruction, with no memory traffic at all.

### Background

The CUDA book's own `__shfl_down_sync(0xffffffff, v, offset)` reads a value a DIFFERENT lane in the same warp holds in its own register, without either lane writing to memory. Its own Worked Example 18.4.1 scales an 8-lane reduction of `[1..8]` (sum `36`) through three rounds: `offset=4` leaves lanes `0-3` holding `[6,8,10,12]`; `offset=2` leaves lanes `0-1` holding `[16,20]`; `offset=1` leaves lane `0` holding `36.0` -- exactly `log2(8)=3` exchanges. Its own `[COMMON TRAP]` warns that a lane that returns early (from Section 18.1's own bounds check) before reaching a shuffle call it still needs to participate in reads UNDEFINED register content, not a safe zero -- the fix is padding out-of-range lanes with the reduction's own identity element (`0` for sum) rather than letting them exit early.

### Worked Example 18.4.1 -- the reduction tree's own arithmetic, the identity-element trap, and a real `torch::sum()` cross-check

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// The CUDA C++ edition's Section 18.4 uses __shfl_down_sync to exchange
// values directly through registers among the 32 threads of one warp
// during a reduction's final rounds, with no memory traffic at all. Its
// own Worked Example 18.4.1 scales the idea to 8 lanes with values
// [1..8] (sum=36), tracing three rounds of pairwise register exchange:
// round 1 (offset=4) leaves lanes 0-3 holding [6,8,10,12]; round 2
// (offset=2) leaves lanes 0-1 holding [16,20]; round 3 (offset=1) leaves
// lane 0 holding 36.0 -- exactly log2(8)=3 rounds. This sandbox has no
// GPU, so the actual __shfl_down_sync INSTRUCTION cannot be executed
// here (see Section 18.1's own honest disclosure) -- that specific claim
// is [UNVERIFIED -- pending real-GPU test]. What this file DOES do
// genuinely: it simulates the exact REDUCTION-TREE ARITHMETIC (the same
// pairwise-sum pattern the shuffle instruction implements, computed here
// with plain array indexing rather than a real cross-lane register
// exchange) with real, compiled, run C++ code, reproducing the CUDA
// book's own three rounds and final sum exactly; it reproduces the CUDA
// book's own identity-element trap using the identical technique the
// CUDA book's own text uses for it (simulated stale values standing in
// for genuinely unpredictable hardware register content, explicitly
// flagged by the CUDA book's own text as illustrative rather than a
// specific reproducible number); and it cross-checks the correct-sum
// case against torch::sum(), LibTorch's own real reduction, whose actual
// CUDA backend performs exactly this warp-shuffle-based tree reduction
// internally on real GPU hardware, invisible to the caller.
int main() {
    // Worked Example 18.4.1: 8-lane reduction, values 1..8, traced
    // round by round exactly as the CUDA book's own __shfl_down_sync
    // loop would (offset = WARP_SIZE_SIM/2, /2, /2), except implemented
    // here as ordinary array reads at (lane+offset) rather than a real
    // cross-lane register exchange -- the SAME arithmetic pattern
    // __shfl_down_sync performs, simulated rather than executed on real
    // warp hardware.
    {
        std::vector<double> lanes = {1, 2, 3, 4, 5, 6, 7, 8};
        int rounds = 0;
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) {
                    next[lane] = lanes[lane] + lanes[lane + offset];
                }
            }
            lanes = next;
            rounds++;
            std::cout << "round " << rounds << " (offset=" << (offset) << "): lane 0 = " << lanes[0]
                      << (rounds <= 2 ? (", lane 1 = " + std::to_string(lanes[1])) : "") << std::endl;
        }
        std::cout << "\nfinal lane 0 = " << lanes[0] << ", CUDA book's own expected sum = 36.0, match = "
                  << (lanes[0] == 36.0) << std::endl;
        std::cout << "rounds performed = " << rounds << ", CUDA book's own expected log2(8) = 3, match = "
                  << (rounds == 3) << std::endl;
    }

    // [UNVERIFIED -- pending real-GPU test]: the loop above simulates the
    // SAME arithmetic pattern __shfl_down_sync performs; it does not
    // execute a real cross-lane register exchange on real warp hardware,
    // which this sandbox has no GPU to run.
    std::cout << "\n[UNVERIFIED -- pending real-GPU test] a genuine __shfl_down_sync instruction "
              << "executed on real warp hardware -- this sandbox has no GPU to run one on; the "
              << "arithmetic pattern it implements is simulated above instead." << std::endl;

    // [COMMON TRAP], reproduced with the CUDA book's own technique: an
    // identity-element trap. 4 real elements (values 1-4) plus 4
    // out-of-range lanes. The CORRECT approach pads the out-of-range
    // lanes with the reduction's own identity element (0.0 for sum)
    // before the shuffle rounds run. The BROKEN approach lets those
    // lanes return early, so the reduction reads whatever was left in
    // that register beforehand -- simulated here with the CUDA book's
    // own explicitly-labeled illustrative stale values, not a genuine
    // hardware nondeterminism this sandbox could reproduce without a GPU.
    {
        // Correct: out-of-range lanes padded with 0.0 (sum's identity).
        std::vector<double> lanes_correct = {1, 2, 3, 4, 0.0, 0.0, 0.0, 0.0};
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes_correct;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) next[lane] = lanes_correct[lane] + lanes_correct[lane + offset];
            }
            lanes_correct = next;
        }
        std::cout << "\ncorrect approach (out-of-range lanes padded with identity 0.0): result = "
                  << lanes_correct[0] << ", expected 1+2+3+4=10, match = " << (lanes_correct[0] == 10.0)
                  << std::endl;

        // Broken: out-of-range lanes hold the CUDA book's own explicitly
        // simulated stale values (illustrative, not a claim about this
        // sandbox's own actual register contents, which cannot be
        // observed without a GPU in the first place).
        std::vector<double> lanes_broken = {1, 2, 3, 4, -8.33862e-16, -1.51445e+06, 5.44969e+26, -5.23627e-31};
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes_broken;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) next[lane] = lanes_broken[lane] + lanes_broken[lane + offset];
            }
            lanes_broken = next;
        }
        std::cout << "broken approach (out-of-range lanes left with the CUDA book's own simulated "
                  << "stale content, illustrative only): result = " << lanes_broken[0]
                  << ", should NOT equal 10.0, is genuinely different from the correct result? "
                  << (lanes_broken[0] != 10.0) << std::endl;
        std::cout << "the point is not the SPECIFIC wrong number (which the CUDA book's own text "
                  << "says is unpredictable on real hardware) -- it is that padding with the "
                  << "identity element is REQUIRED, not optional, for correctness." << std::endl;
    }

    // Real LibTorch cross-check: torch::sum() is LibTorch's own actual
    // reduction. On real GPU hardware its own CUDA backend performs
    // exactly this warp-shuffle-based tree reduction internally,
    // invisible to the caller -- confirmed here to produce the correct
    // sum on this CPU-only sandbox.
    torch::Tensor values = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    torch::Tensor total = torch::sum(values);
    std::cout << "\ntorch::sum() (real, production LibTorch reduction) = " << total.item<double>()
              << ", matches this file's own hand-simulated tree reduction (36.0)? "
              << (total.item<double>() == 36.0) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_warp_reduction_and_identity_trap.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_warp_reduction_and_identity_trap
./04_warp_reduction_and_identity_trap
```

```text
round 1 (offset=4): lane 0 = 6, lane 1 = 8.000000
round 2 (offset=2): lane 0 = 16, lane 1 = 20.000000
round 3 (offset=1): lane 0 = 36

final lane 0 = 36, CUDA book's own expected sum = 36.0, match = 1
rounds performed = 3, CUDA book's own expected log2(8) = 3, match = 1

[UNVERIFIED -- pending real-GPU test] a genuine __shfl_down_sync instruction executed on real warp hardware -- this sandbox has no GPU to run one on; the arithmetic pattern it implements is simulated above instead.

correct approach (out-of-range lanes padded with identity 0.0): result = 10, expected 1+2+3+4=10, match = 1
broken approach (out-of-range lanes left with the CUDA book's own simulated stale content, illustrative only): result = 5.44969e+26, should NOT equal 10.0, is genuinely different from the correct result? 1
the point is not the SPECIFIC wrong number (which the CUDA book's own text says is unpredictable on real hardware) -- it is that padding with the identity element is REQUIRED, not optional, for correctness.

torch::sum() (real, production LibTorch reduction) = 36, matches this file's own hand-simulated tree reduction (36.0)? 1
```

Independently cross-checked via a separate, plain Python implementation of the identical reduction-tree arithmetic (no `torch` needed for the tree simulation itself), plus Python's own separate `torch` binding for the final `torch.sum()` cross-check:

```text
correct result 10
broken result 5.44969e+26
torch.sum 36.0
```

### Discussion

The reduction-tree arithmetic matches the CUDA book's own three rounds and final sum exactly, confirming the pairwise-exchange PATTERN `__shfl_down_sync` implements is correctly reproduced here, even though the actual cross-lane register instruction itself cannot run without real warp hardware. The identity-element trap reproduces the CUDA book's own point precisely: the broken approach's own specific wrong number (`5.44969e+26`) is not the finding -- the CUDA book's own text itself says that number is illustrative and would be unpredictable on real hardware -- the finding is that ANY value other than the reduction's own identity element, substituted for a lane that should have contributed nothing, corrupts the final result, while the identity element (`0.0` for sum) is provably safe regardless of what it stands in for. The real `torch::sum()` cross-check closes the section the same way Section 18.3's own `torch::conv2d` cross-check did: LibTorch's own actual reduction performs exactly this class of warp-shuffle-based tree reduction internally on real GPU hardware, and gets the identity-element requirement right automatically -- a LibTorch programmer calling `torch::sum()` never needs to think about lanes, shuffles, or identity elements at all, because the CUDA book's own Section 18.4 correctness requirement has already been engineered into the operation itself.

> `[COMMON TRAP]` It would be easy to read the correct-vs-broken contrast in this section and conclude the LESSON is "pad with zero." The more general lesson, which this section's own framing preserves deliberately, is "pad with the reduction's own IDENTITY ELEMENT" -- `0.0` happens to be correct here because this is a SUM reduction, but a MAX reduction's own identity element would be negative infinity (or the smallest representable value), and a PRODUCT reduction's own identity element would be `1.0`, not `0.0`. A reader generalizing this section's own specific number to "always pad with zero" would introduce exactly the CUDA book's own kind of subtle bug the moment they wrote a max- or product-based reduction using the same pattern.

## Complete Runnable Code

### File: `01_launch_configuration.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 18.1 establishes the two-part kernel
// launch template every one of its own raw kernels depends on: ceiling
// division on the host (num_blocks = (size + threads_per_block - 1) /
// threads_per_block) to guarantee enough blocks exist to cover every
// element, paired with an in-kernel bounds check (if (idx < size)) so the
// inevitable "wasted" threads in the last block do nothing rather than
// writing past the buffer's own end. This book's own honest position on
// this chapter, stated plainly up front: this sandbox has no NVIDIA GPU
// and no CUDA toolkit installed (confirmed via `which nvcc` finding
// nothing, and torch::cuda::is_available() reporting false below) -- so
// nothing in this chapter claiming to be a kernel LAUNCH, or a timing or
// hardware-execution result, can be genuinely run here. What CAN be
// genuinely run, and is run in this file, is the launch-configuration
// ARITHMETIC itself -- pure host-side integer math with no CUDA
// dependency at all -- which is real code any LibTorch custom CUDA
// extension's own host-side launch logic would compute identically
// before ever reaching a `<<<blocks, threads>>>` launch expression this
// book cannot execute. This file reproduces the CUDA book's own two
// worked examples exactly using genuine, compiled, run integer
// arithmetic, and reports this sandbox's own real (negative) CUDA
// availability honestly rather than fabricating a kernel launch result.
long ceiling_div(long size, long threads_per_block) {
    return (size + threads_per_block - 1) / threads_per_block;
}

int main() {
    std::cout << "torch::cuda::is_available() in this sandbox = " << torch::cuda::is_available()
              << " (no NVIDIA GPU or CUDA toolkit present -- every claim below is pure host-side "
              << "arithmetic, genuinely computed, with NO kernel actually launched anywhere in this "
              << "chapter; any claim requiring real GPU execution is explicitly tagged "
              << "[UNVERIFIED -- pending real-GPU test] rather than fabricated)" << std::endl;

    // Worked Example 18.1.1: a million elements, 256 threads per block.
    {
        long size = 1000000, tpb = 256;
        long num_blocks = ceiling_div(size, tpb);
        long total_threads = num_blocks * tpb;
        long wasted = total_threads - size;
        std::cout << "\nsize=" << size << ", threads_per_block=" << tpb
                  << ": num_blocks=" << num_blocks << ", total_threads=" << total_threads
                  << ", wasted_threads=" << wasted << std::endl;
        std::cout << "matches CUDA book's own Worked Example 18.1.1 "
                  << "(num_blocks=3907, total_threads=1000192, wasted=192)? "
                  << (num_blocks == 3907 && total_threads == 1000192 && wasted == 192) << std::endl;
    }

    // Worked Example 18.1.2: ten thousand elements, 128 threads per block.
    {
        long size = 10000, tpb = 128;
        long num_blocks = ceiling_div(size, tpb);
        long total_threads = num_blocks * tpb;
        long wasted = total_threads - size;
        std::cout << "\nsize=" << size << ", threads_per_block=" << tpb
                  << ": num_blocks=" << num_blocks << ", total_threads=" << total_threads
                  << ", wasted_threads=" << wasted << std::endl;
        std::cout << "matches CUDA book's own Worked Example 18.1.2 "
                  << "(num_blocks=79, total_threads=10112, wasted=112)? "
                  << (num_blocks == 79 && total_threads == 10112 && wasted == 112) << std::endl;
    }

    // [COMMON TRAP]: floor division instead of ceiling division. This is
    // the CUDA book's own trap, reproduced as pure arithmetic: floor
    // division for size=10000, tpb=128 gives 78 blocks, NOT 79 -- leaving
    // elements 9984-9999 (16 elements) covered by no block at all.
    {
        long size = 10000, tpb = 128;
        long floor_blocks = size / tpb;   // the CUDA book's own trap
        long ceil_blocks = ceiling_div(size, tpb);
        long floor_covered = floor_blocks * tpb;
        long uncovered = size - floor_covered;
        std::cout << "\n[COMMON TRAP] floor division: floor_blocks=" << floor_blocks
                  << " (vs correct ceiling_blocks=" << ceil_blocks << "), "
                  << "elements NEVER covered by any block = " << uncovered
                  << ", matches CUDA book's own 78 blocks / 16 uncovered elements? "
                  << (floor_blocks == 78 && uncovered == 16) << std::endl;
    }

    return 0;
}
```

### File: `02_memory_layout_aos_soa.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <set>

// The CUDA C++ edition's Section 18.2 contrasts Array-of-Structs (AoS,
// stride = whole struct size between the same field in adjacent records)
// against Struct-of-Arrays (SoA, stride = one field's own size between
// adjacent records), and confirms the resulting 4-vs-1-transaction
// difference by decoding genuine cuobjdump SASS output -- machine code
// this sandbox cannot produce, since it has no nvcc and no GPU (see
// Section 18.1's own honest disclosure). What this sandbox CAN do, and
// does here, is verify the exact same STRIDE MATH the SASS decoding
// confirms, using sizeof() and offsetof() on the CUDA book's own real
// eight-field struct, compiled and run genuinely on CPU -- these are
// compile-time-computed, hardware-independent facts about C++ struct
// layout, not something that requires a GPU to be true or to check. This
// file also investigates a genuinely LibTorch-native question the CUDA
// book's own hand-rolled struct has no equivalent for: does
// torch::Tensor's own default row-major layout correspond to AoS or SoA
// for a [num_records, num_fields] tensor -- and can a zero-copy .t() view
// flip which one it is, or does genuine SoA layout require an actual,
// byte-copying .contiguous() call, since a view alone moves no data?
struct ZeroCouponBondAoS {
    float face_value;          // offset 0
    float time_to_maturity;    // offset 4
    float risk_free_rate;      // offset 8  (float-index 2)
    float credit_spread;       // offset 12
    float present_value;       // offset 16
    float yield_to_maturity;   // offset 20
    float duration;            // offset 24 (float-index 6)
    float portfolio_weight;    // offset 28
};   // CUDA book's own claimed total size: 32 bytes

int main() {
    // Struct layout, verified via sizeof/offsetof -- CPU-only, compile-time
    // facts, no GPU or SASS decoding needed to confirm them.
    std::cout << "sizeof(ZeroCouponBondAoS) = " << sizeof(ZeroCouponBondAoS)
              << ", CUDA book's own claimed struct size = 32 bytes, match = "
              << (sizeof(ZeroCouponBondAoS) == 32) << std::endl;
    std::cout << "offsetof(risk_free_rate) = " << offsetof(ZeroCouponBondAoS, risk_free_rate)
              << ", CUDA book's own claimed offset = 8 (float-index 2), match = "
              << (offsetof(ZeroCouponBondAoS, risk_free_rate) == 8) << std::endl;

    // AoS chunk-counting, reproduced as genuine pointer/index arithmetic
    // (the same math cuobjdump's own decoded SASS confirms, verified here
    // without needing SASS at all): four threads reading risk_free_rate
    // for records 0-3, chunk size = 16 bytes (4 floats).
    {
        const int CHUNK_BYTES = 16;
        const int STRIDE_BYTES = sizeof(ZeroCouponBondAoS);
        const int FIELD_OFFSET = offsetof(ZeroCouponBondAoS, risk_free_rate);
        std::set<long> chunks_touched;
        for (int record = 0; record < 4; record++) {
            long byte_addr = static_cast<long>(record) * STRIDE_BYTES + FIELD_OFFSET;
            chunks_touched.insert(byte_addr / CHUNK_BYTES);
        }
        std::cout << "\nAoS: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = "
                  << chunks_touched.size() << ", CUDA book's own expected = 4, match = "
                  << (chunks_touched.size() == 4) << std::endl;
    }
    {
        const int CHUNK_BYTES = 16;
        const int STRIDE_BYTES = sizeof(float);   // SoA: one float per record, contiguous
        std::set<long> chunks_touched;
        for (int record = 0; record < 4; record++) {
            long byte_addr = static_cast<long>(record) * STRIDE_BYTES;
            chunks_touched.insert(byte_addr / CHUNK_BYTES);
        }
        std::cout << "SoA: distinct 16-byte chunks touched by 4 threads reading risk_free_rate = "
                  << chunks_touched.size() << ", CUDA book's own expected = 1, match = "
                  << (chunks_touched.size() == 1) << std::endl;
    }

    // [UNVERIFIED -- pending real-GPU test]: the CUDA book's own genuine
    // cuobjdump SASS decoding (HFMA2.MMA constant materialization
    // resolving to exactly sizeof(float)=4 for SoA and
    // sizeof(ZeroCouponBondAoS)=32 for AoS) cannot be reproduced without
    // nvcc and a real GPU to compile and disassemble a kernel for. The
    // stride numbers it decodes to (4 and 32) are exactly the same
    // numbers this file already confirmed directly via sizeof/offsetof
    // above -- SASS decoding is the CUDA book's own proof that the
    // COMPILED MACHINE CODE matches the hand count; this file's own proof
    // is that the STRUCT LAYOUT ITSELF (a compile-time fact, true
    // regardless of what machine eventually runs code against it)
    // matches the hand count.
    std::cout << "\n[UNVERIFIED -- pending real-GPU test] genuine cuobjdump SASS decoding of a "
              << "compiled kernel's own stride constants -- this sandbox has no nvcc/GPU to produce "
              << "or disassemble a kernel binary with." << std::endl;

    // A genuinely LibTorch-native question: does a [num_records,
    // num_fields] torch::Tensor's own default (row-major) layout behave
    // like AoS or SoA when reading one "field" (column) across many
    // "records" (rows)? And does .t() (a real, zero-copy transpose) flip
    // which one it is?
    torch::Tensor records_x_fields = torch::arange(32, torch::kFloat32).reshape({4, 8});
    std::cout << "\n[num_records=4, num_fields=8] tensor: stride() = ["
              << records_x_fields.stride(0) << "," << records_x_fields.stride(1) << "] elements"
              << " (" << records_x_fields.stride(0) * sizeof(float) << ","
              << records_x_fields.stride(1) * sizeof(float) << " bytes)" << std::endl;
    std::cout << "reading field index 2 across all 4 records: byte stride between consecutive "
              << "records = " << records_x_fields.stride(0) * sizeof(float)
              << ", structurally AoS-like (matches this file's own 32-byte AoS stride above)? "
              << (records_x_fields.stride(0) * sizeof(float) == 32) << std::endl;

    torch::Tensor fields_x_records_view = records_x_fields.t();   // zero-copy transpose: a VIEW only
    std::cout << "\nthe SAME data, transposed to [num_fields=8, num_records=4] via .t() "
              << "(a real, zero-copy VIEW -- reinterprets strides, moves no bytes): stride() = ["
              << fields_x_records_view.stride(0) << "," << fields_x_records_view.stride(1) << "] elements, "
              << "is_contiguous() = " << fields_x_records_view.is_contiguous() << std::endl;
    std::cout << "reading field index 2 (now row 2) across all 4 records: byte stride between "
              << "consecutive records = " << fields_x_records_view.stride(1) * sizeof(float)
              << " -- STILL 32, not 4: a view cannot make AoS-laid-out data physically SoA, "
              << "because .t() moves no bytes at all, only relabels which dimension is which. "
              << "same underlying storage (data_ptr unchanged)? "
              << (records_x_fields.data_ptr() == fields_x_records_view.data_ptr()) << std::endl;

    // Genuine SoA requires an ACTUAL data-moving copy, not merely a
    // relabeled view: .contiguous() on the transposed (non-contiguous)
    // view forces LibTorch's own real memory-copy path to physically
    // relay out the bytes to match the new logical shape.
    torch::Tensor fields_x_records_real = fields_x_records_view.contiguous();
    std::cout << "\n.t().contiguous() (a REAL, byte-copying operation): stride() = ["
              << fields_x_records_real.stride(0) << "," << fields_x_records_real.stride(1) << "] elements, "
              << "is_contiguous() = " << fields_x_records_real.is_contiguous() << std::endl;
    std::cout << "reading field index 2 across all 4 records: byte stride between consecutive "
              << "records = " << fields_x_records_real.stride(1) * sizeof(float)
              << ", NOW genuinely SoA (matches this file's own 4-byte SoA stride above)? "
              << (fields_x_records_real.stride(1) * sizeof(float) == 4) << std::endl;
    std::cout << "different underlying storage from the original (data_ptr changed, confirming "
              << "actual bytes were physically copied and relaid out, not merely relabeled)? "
              << (records_x_fields.data_ptr() != fields_x_records_real.data_ptr()) << std::endl;

    return 0;
}
```

### File: `03_shared_memory_tiling.cpp`

```cpp
#include <torch/torch.h>
#include <torch/nn/functional/conv.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's Section 18.3 hand-writes a naive convolution
// kernel (each output thread independently re-reads every input cell its
// own 3x3 window touches from global memory) against a tiled version
// (the block first stages the whole input tile into __shared__ memory
// once, then every thread reads from that fast on-chip copy). Its own
// worked numbers: a 4x4 input convolved with a 3x3 kernel produces a 2x2
// output ([[30,35],[50,55]]); the naive kernel performs 36 total global
// reads (4 outputs x 9 taps each) for only 16 unique input values, while
// the tiled kernel performs exactly 16 -- one per unique input cell, for
// the whole block. This sandbox has no GPU to launch either kernel on
// (see Section 18.1's own honest disclosure), so no claim about ON-CHIP
// SPEED is tested here -- speed is explicitly [UNVERIFIED -- pending
// real-GPU test]. What this file DOES genuinely test, in real compiled
// and run C++: the READ-COUNT arithmetic itself, by simulating each
// approach's own access pattern with a real counter (not a fabricated
// number), confirming both approaches produce IDENTICAL correct output
// values, and confirming a real torch::conv2d call -- LibTorch's own
// actual production convolution, which performs exactly this class of
// tiling optimization internally on a real GPU, invisible to the caller
// -- reproduces the identical numbers on CPU.
int main() {
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}
    };
    std::vector<std::vector<float>> kernel = {
        {1, 0, 1}, {0, 1, 0}, {1, 0, 1}
    };

    // Naive approach: each of the 2x2=4 output threads independently
    // reads its own full 3x3 window from "global memory" (the input
    // vector), counted with a real counter incremented on every read.
    long naive_reads = 0;
    std::vector<std::vector<float>> naive_output(2, std::vector<float>(2, 0.0f));
    for (int out_r = 0; out_r < 2; out_r++) {
        for (int out_c = 0; out_c < 2; out_c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    sum += input[out_r + kr][out_c + kc] * kernel[kr][kc];
                    naive_reads++;   // one global-memory read per tap, every thread
                }
            }
            naive_output[out_r][out_c] = sum;
        }
    }
    std::cout << "naive kernel: total simulated global reads = " << naive_reads
              << ", CUDA book's own expected = 36 (4 outputs x 9 taps), match = "
              << (naive_reads == 36) << std::endl;
    std::cout << "naive output = [[" << naive_output[0][0] << "," << naive_output[0][1] << "],["
              << naive_output[1][0] << "," << naive_output[1][1] << "]], "
              << "CUDA book's own expected = [[30,35],[50,55]], match = "
              << (naive_output[0][0] == 30 && naive_output[0][1] == 35 &&
                  naive_output[1][0] == 50 && naive_output[1][1] == 55) << std::endl;

    // Tiled approach: the whole 4x4 input is staged into "shared memory"
    // ONCE (one read per unique cell, counted separately from the reuse
    // phase below), then every subsequent read comes from that staged
    // copy -- not counted again against the global-read total, exactly
    // as the CUDA book's own tile[] array is read repeatedly with no
    // further global-memory traffic after the load phase.
    long tiled_global_reads = 0;
    std::vector<std::vector<float>> shared_tile(4, std::vector<float>(4));
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            shared_tile[r][c] = input[r][c];   // ONE global read per unique cell
            tiled_global_reads++;
        }
    }
    std::vector<std::vector<float>> tiled_output(2, std::vector<float>(2, 0.0f));
    for (int out_r = 0; out_r < 2; out_r++) {
        for (int out_c = 0; out_c < 2; out_c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    sum += shared_tile[out_r + kr][out_c + kc] * kernel[kr][kc];
                    // reading from shared_tile, not input -- no global read counted
                }
            }
            tiled_output[out_r][out_c] = sum;
        }
    }
    std::cout << "\ntiled kernel: total simulated global reads (load phase only) = " << tiled_global_reads
              << ", CUDA book's own expected = 16 (down from 36), match = "
              << (tiled_global_reads == 16) << std::endl;
    std::cout << "tiled output = [[" << tiled_output[0][0] << "," << tiled_output[0][1] << "],["
              << tiled_output[1][0] << "," << tiled_output[1][1] << "]], "
              << "matches the naive kernel's own output exactly? "
              << (tiled_output[0][0] == naive_output[0][0] && tiled_output[0][1] == naive_output[0][1] &&
                  tiled_output[1][0] == naive_output[1][0] && tiled_output[1][1] == naive_output[1][1])
              << std::endl;

    std::cout << "\n[UNVERIFIED -- pending real-GPU test] on-chip shared-memory access being "
              << "dramatically faster than repeated global-memory reads -- this sandbox has no GPU "
              << "to measure real memory-hierarchy latency on; only the READ-COUNT REDUCTION (36->16) "
              << "and output CORRECTNESS are genuinely tested above." << std::endl;

    // The real LibTorch cross-check: torch::conv2d is LibTorch's own
    // actual, production convolution -- on a real GPU, its own backend
    // performs exactly this class of tiling optimization internally,
    // invisible to the caller. On this CPU-only sandbox it still exists
    // and still produces correct output, confirming the LibTorch
    // programmer gets the CUDA book's own Section 18.3 optimization "for
    // free" by calling one real function, rather than hand-writing it.
    torch::Tensor input_t = torch::tensor({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0},
                                            {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}})
                                 .reshape({1, 1, 4, 4});
    torch::Tensor kernel_t = torch::tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}})
                                  .reshape({1, 1, 3, 3});
    torch::Tensor conv_out = torch::nn::functional::conv2d(input_t, kernel_t);
    std::cout << "\ntorch::conv2d (real, production LibTorch convolution) output =\n" << conv_out << std::endl;
    torch::Tensor conv_out_expected = torch::tensor({{30.0, 35.0}, {50.0, 55.0}}).reshape({1, 1, 2, 2});
    std::cout << "matches this file's own hand-simulated output (and the CUDA book's own)? "
              << torch::allclose(conv_out, conv_out_expected) << std::endl;

    // Worked Example 18.3.3: zero-padded, same-shape convolution.
    torch::Tensor conv_out_padded = torch::nn::functional::conv2d(
        input_t, kernel_t, torch::nn::functional::Conv2dFuncOptions().padding(1));
    std::cout << "\ntorch::conv2d with padding=1 (zero-padded, same-shape output) =\n"
              << conv_out_padded << std::endl;
    torch::Tensor padded_expected = torch::tensor({{7.0, 14.0, 17.0, 11.0}, {17.0, 30.0, 35.0, 22.0},
                                                     {29.0, 50.0, 55.0, 34.0}, {23.0, 34.0, 37.0, 27.0}})
                                         .reshape({1, 1, 4, 4});
    std::cout << "matches CUDA book's own Worked Example 18.3.3 padded output exactly? "
              << torch::allclose(conv_out_padded, padded_expected) << std::endl;

    return 0;
}
```

### File: `04_warp_reduction_and_identity_trap.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// The CUDA C++ edition's Section 18.4 uses __shfl_down_sync to exchange
// values directly through registers among the 32 threads of one warp
// during a reduction's final rounds, with no memory traffic at all. Its
// own Worked Example 18.4.1 scales the idea to 8 lanes with values
// [1..8] (sum=36), tracing three rounds of pairwise register exchange:
// round 1 (offset=4) leaves lanes 0-3 holding [6,8,10,12]; round 2
// (offset=2) leaves lanes 0-1 holding [16,20]; round 3 (offset=1) leaves
// lane 0 holding 36.0 -- exactly log2(8)=3 rounds. This sandbox has no
// GPU, so the actual __shfl_down_sync INSTRUCTION cannot be executed
// here (see Section 18.1's own honest disclosure) -- that specific claim
// is [UNVERIFIED -- pending real-GPU test]. What this file DOES do
// genuinely: it simulates the exact REDUCTION-TREE ARITHMETIC (the same
// pairwise-sum pattern the shuffle instruction implements, computed here
// with plain array indexing rather than a real cross-lane register
// exchange) with real, compiled, run C++ code, reproducing the CUDA
// book's own three rounds and final sum exactly; it reproduces the CUDA
// book's own identity-element trap using the identical technique the
// CUDA book's own text uses for it (simulated stale values standing in
// for genuinely unpredictable hardware register content, explicitly
// flagged by the CUDA book's own text as illustrative rather than a
// specific reproducible number); and it cross-checks the correct-sum
// case against torch::sum(), LibTorch's own real reduction, whose actual
// CUDA backend performs exactly this warp-shuffle-based tree reduction
// internally on real GPU hardware, invisible to the caller.
int main() {
    // Worked Example 18.4.1: 8-lane reduction, values 1..8, traced
    // round by round exactly as the CUDA book's own __shfl_down_sync
    // loop would (offset = WARP_SIZE_SIM/2, /2, /2), except implemented
    // here as ordinary array reads at (lane+offset) rather than a real
    // cross-lane register exchange -- the SAME arithmetic pattern
    // __shfl_down_sync performs, simulated rather than executed on real
    // warp hardware.
    {
        std::vector<double> lanes = {1, 2, 3, 4, 5, 6, 7, 8};
        int rounds = 0;
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) {
                    next[lane] = lanes[lane] + lanes[lane + offset];
                }
            }
            lanes = next;
            rounds++;
            std::cout << "round " << rounds << " (offset=" << (offset) << "): lane 0 = " << lanes[0]
                      << (rounds <= 2 ? (", lane 1 = " + std::to_string(lanes[1])) : "") << std::endl;
        }
        std::cout << "\nfinal lane 0 = " << lanes[0] << ", CUDA book's own expected sum = 36.0, match = "
                  << (lanes[0] == 36.0) << std::endl;
        std::cout << "rounds performed = " << rounds << ", CUDA book's own expected log2(8) = 3, match = "
                  << (rounds == 3) << std::endl;
    }

    // [UNVERIFIED -- pending real-GPU test]: the loop above simulates the
    // SAME arithmetic pattern __shfl_down_sync performs; it does not
    // execute a real cross-lane register exchange on real warp hardware,
    // which this sandbox has no GPU to run.
    std::cout << "\n[UNVERIFIED -- pending real-GPU test] a genuine __shfl_down_sync instruction "
              << "executed on real warp hardware -- this sandbox has no GPU to run one on; the "
              << "arithmetic pattern it implements is simulated above instead." << std::endl;

    // [COMMON TRAP], reproduced with the CUDA book's own technique: an
    // identity-element trap. 4 real elements (values 1-4) plus 4
    // out-of-range lanes. The CORRECT approach pads the out-of-range
    // lanes with the reduction's own identity element (0.0 for sum)
    // before the shuffle rounds run. The BROKEN approach lets those
    // lanes return early, so the reduction reads whatever was left in
    // that register beforehand -- simulated here with the CUDA book's
    // own explicitly-labeled illustrative stale values, not a genuine
    // hardware nondeterminism this sandbox could reproduce without a GPU.
    {
        // Correct: out-of-range lanes padded with 0.0 (sum's identity).
        std::vector<double> lanes_correct = {1, 2, 3, 4, 0.0, 0.0, 0.0, 0.0};
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes_correct;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) next[lane] = lanes_correct[lane] + lanes_correct[lane + offset];
            }
            lanes_correct = next;
        }
        std::cout << "\ncorrect approach (out-of-range lanes padded with identity 0.0): result = "
                  << lanes_correct[0] << ", expected 1+2+3+4=10, match = " << (lanes_correct[0] == 10.0)
                  << std::endl;

        // Broken: out-of-range lanes hold the CUDA book's own explicitly
        // simulated stale values (illustrative, not a claim about this
        // sandbox's own actual register contents, which cannot be
        // observed without a GPU in the first place).
        std::vector<double> lanes_broken = {1, 2, 3, 4, -8.33862e-16, -1.51445e+06, 5.44969e+26, -5.23627e-31};
        for (int offset = 4; offset > 0; offset /= 2) {
            std::vector<double> next = lanes_broken;
            for (int lane = 0; lane < 8; lane++) {
                if (lane + offset < 8) next[lane] = lanes_broken[lane] + lanes_broken[lane + offset];
            }
            lanes_broken = next;
        }
        std::cout << "broken approach (out-of-range lanes left with the CUDA book's own simulated "
                  << "stale content, illustrative only): result = " << lanes_broken[0]
                  << ", should NOT equal 10.0, is genuinely different from the correct result? "
                  << (lanes_broken[0] != 10.0) << std::endl;
        std::cout << "the point is not the SPECIFIC wrong number (which the CUDA book's own text "
                  << "says is unpredictable on real hardware) -- it is that padding with the "
                  << "identity element is REQUIRED, not optional, for correctness." << std::endl;
    }

    // Real LibTorch cross-check: torch::sum() is LibTorch's own actual
    // reduction. On real GPU hardware its own CUDA backend performs
    // exactly this warp-shuffle-based tree reduction internally,
    // invisible to the caller -- confirmed here to produce the correct
    // sum on this CPU-only sandbox.
    torch::Tensor values = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    torch::Tensor total = torch::sum(values);
    std::cout << "\ntorch::sum() (real, production LibTorch reduction) = " << total.item<double>()
              << ", matches this file's own hand-simulated tree reduction (36.0)? "
              << (total.item<double>() == 36.0) << std::endl;

    return 0;
}
```

All four files compile and link against LibTorch with the standard command from *Getting Started*:

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 <file>.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o <file>
./<file>
```

None of these files require nvcc or a GPU to compile or run -- they are ordinary C++ files linked against LibTorch's own CPU backend, consistent with this chapter's own honest position that this sandbox has neither.

## Chapter Summary

This chapter opened with an honest disclosure -- `torch::cuda::is_available()` genuinely reports `false` in this sandbox, and no `nvcc` is installed -- and held that honesty throughout rather than fabricating GPU results. Launch-configuration arithmetic (ceiling division, wasted-thread counts) was confirmed to match the CUDA book's own two worked examples exactly, since this math is genuinely hardware-independent. Memory-layout analysis (AoS's own `32`-byte stride versus SoA's own `4`-byte stride) was confirmed via `sizeof()`/`offsetof()` rather than SASS disassembly, with a genuinely new LibTorch-specific finding along the way: a zero-copy `.t()` view cannot convert AoS-laid-out data to SoA, because it moves no bytes at all -- only `.contiguous()`, a real byte-copying operation, achieves genuine SoA layout. Shared-memory tiling's own read-count reduction (`36` down to `16`) was confirmed via a genuine counter, with both the naive and tiled approaches shown to produce identical, correct output, and `torch::conv2d` confirmed to reproduce the CUDA book's own exact numbers, both unpadded and padded. Warp-shuffle reduction's own tree arithmetic (three rounds, summing to `36.0`) was confirmed via simulation, its own identity-element trap reproduced using the CUDA book's own illustrative technique, and `torch::sum()` confirmed to reproduce the correct total. Every claim this chapter could not genuinely test -- SASS disassembly, real kernel launches, on-chip memory timing, and the literal `__shfl_down_sync` instruction -- was explicitly tagged `[UNVERIFIED -- pending real-GPU test]` rather than presented as tested.

## Self-Check Questions

1. This chapter tags several specific claims `[UNVERIFIED -- pending real-GPU test]` rather than testing them. Using Section 18.1's own `torch::cuda::is_available()` check as the starting point, explain the distinction this chapter draws between claims that ARE genuinely testable in this sandbox and claims that are NOT, and why that distinction is not simply "software claims vs. hardware claims."
2. Section 18.2's own `.t()` test finds the transposed view's own byte stride for reading one field across records is STILL `32`, not `4`. Explain, in terms of what `.t()` actually does to a tensor's own underlying storage, why this result is not a bug or an oversight.
3. Section 18.3's own naive-versus-tiled contrast shows two very different read counts (`36` vs. `16`) producing IDENTICAL output values. Explain why this equality is itself meaningful evidence, not merely a convenient coincidence for this section's own worked example.
4. Section 18.4's own `[COMMON TRAP]` closing note argues that "pad with zero" is the wrong generalization of its own lesson. What is the correct, more general version of the lesson, and what would go wrong if a reader applied "pad with zero" to a max-reduction instead of a sum-reduction?
5. Sections 18.3 and 18.4 both end with a real `torch::` function (`torch::conv2d`, `torch::sum()`) reproducing the CUDA book's own numbers. Explain what this cross-check demonstrates that the hand-simulated read-count and reduction-tree code, by itself, could not have demonstrated.

## Where We Go Next

This chapter opened Part 5: GPU Acceleration and Performance under an explicit, sandbox-wide constraint -- no NVIDIA GPU, no CUDA toolkit -- and met that constraint by separating genuinely testable arithmetic and layout claims from genuinely untestable hardware-execution claims, tagging the latter honestly rather than fabricating results. Launch configuration, memory layout, shared-memory read-count reduction, and warp-shuffle reduction arithmetic were all confirmed to match the CUDA book's own numbers, and `torch::Tensor`'s own real operations (`torch::conv2d`, `torch::sum()`) were confirmed to perform the CUDA book's own hand-tuned optimizations internally, invisible to the caller. Chapter 19 continues Part 5 with Performance Optimization Techniques, building on this chapter's own honest-GPU-absence framework to examine profiling, kernel fusion, and memory-bandwidth analysis -- again separating what this sandbox's own CPU-only environment can genuinely test from what remains honestly `[UNVERIFIED -- pending real-GPU test]`.

## Worked Solutions

**1.** The distinction is not "software vs. hardware" in general -- it is specifically about what this SANDBOX, right now, can genuinely EXECUTE versus what it can only genuinely COMPUTE as arithmetic. `torch::cuda::is_available()` itself is a real, genuinely executed software call, and it genuinely returns `false` -- that result is trustworthy precisely because it required no GPU to obtain. Ceiling-division arithmetic, struct `sizeof()`/`offsetof()`, and reduction-tree simulation are all similarly executable without a GPU, because they are facts about C++ semantics and plain arithmetic, true on any machine that can compile and run C++ at all. What genuinely requires a GPU is anything whose correctness or value depends on ACTUAL HARDWARE BEHAVIOR: a real kernel launch, real SASS machine code a real `nvcc` compiled, real memory-hierarchy timing, or a real `__shfl_down_sync` instruction executing on real warp hardware. The line is drawn at "would running this exact code on different hardware, or on no hardware at all, change the answer" -- ceiling division always gives the same answer; a warp shuffle's own actual execution requires a warp to actually exist.

**2.** `.t()` is a VIEW: it changes only the `stride()` metadata attached to a tensor, telling subsequent code how to reinterpret the SAME underlying bytes as a different logical shape -- it copies and moves nothing. The original tensor's own bytes were physically laid out with fields contiguous within each record (AoS-style), and `.t()` cannot change where those bytes physically sit in memory, because a view, by definition, does not touch memory at all. Reading "field 2 across records" after `.t()` still walks the identical physical `32`-byte-separated memory locations it walked before -- the NAMES of the dimensions changed (what used to be called "dimension 0, size 4" is now "dimension 1, size 4"), but the underlying addresses those names resolve to did not. This is expected, correct behavior for a view operation, not an oversight; achieving genuine SoA layout requires `.contiguous()`, which performs an actual copy.

**3.** The equality is meaningful because it demonstrates the read-count reduction is a pure EFFICIENCY change, not a change to WHAT is being computed. If the tiled approach's own different read pattern had produced even slightly different output, that would suggest the tiling logic itself introduced an error somewhere -- perhaps double-counting a cell, or missing one at a tile boundary. The two approaches computing byte-for-byte identical results, despite reading global memory a very different number of times, is direct evidence that the ONLY thing shared-memory tiling changes is where each value is read FROM on its second and later uses (fast on-chip memory instead of repeated global-memory reads) -- not the VALUES themselves or the final arithmetic performed on them.

**4.** The correct, general lesson is: pad out-of-range lanes with the specific reduction operation's own IDENTITY ELEMENT -- the value that, combined with any other value under that operation, leaves the other value unchanged. For sum, that element is `0` (since `x+0=x`); a reader applying "pad with zero" to a MAX reduction would be applying the wrong identity element, because `max(x, 0)` is not always `x` -- for any negative `x`, `max(x,0)=0`, silently and incorrectly raising the result toward zero rather than leaving it alone. A max-reduction's own correct identity element is negative infinity (or the smallest representable value for the type in use), since `max(x, -infinity) = x` for any `x`.

**5.** The hand-simulated code, by itself, only demonstrates that THIS BOOK's own particular simulation of the read-count or reduction-tree logic is internally consistent and matches the CUDA book's own hand-derived numbers -- it says nothing about whether any REAL, independently-engineered system behaves the same way. The `torch::` cross-check demonstrates something categorically different: that LibTorch's own real, independently-developed, production-grade implementation -- built by engineers with no connection to this book's own simulation code, and (on real GPU hardware) performing its own actual optimized kernel execution -- produces the identical numbers. This is the same kind of independent-verification argument this book has made in every previous chapter (a second implementation, built differently, reaching the same answer), applied here specifically to confirm that the CUDA book's own conceptual claims about read-count reduction and correct reduction-tree behavior are not merely internally consistent within this book's own simulation, but actually true of a real system real engineers rely on.
