# Chapter 7: Memory Layout Design

> "The CUDA C++ edition's Chapter 7 opens by pointing out that Chapter 6's `offset = coord[0]*strides[0] + coord[1]*strides[1] + ...` formula never changes shape, but the strides feeding it can encode a dozen different physical arrangements of the identical logical tensor. That observation needs no adjustment for this book at all: `torch::Tensor` computes offsets with the exact same formula, and its `.strides()` are exactly as changeable as the CUDA book's hand-built `TensorStrides` -- row-major by default, column-major on request, and (a layout the CUDA book's own hand-built `Tensor` never had a name for) channels-last, a real third strategy `torch::Tensor` supports for the four-dimensional tensors real convolutional networks use."

**What you will understand by the end of this chapter:**

- That `torch::Tensor`'s default row-major strides and an explicitly constructed column-major layout produce the exact offsets the CUDA book's own worked coordinates predict for shape `[2, 3, 4]` -- `(0,0,1)`, `(1,0,0)`, and `(1,1,2)` all verified against both layouts, not just the one coordinate this book's earlier chapters have relied on
- Why `torch::Tensor`'s own `.t()` and indexing operators already are the CUDA book's hand-built `TensorView` -- a non-owning wrapper around a pointer, a shape, and strides -- confirmed by transposing a real tensor and showing it reaches the identical flat offset, and identical real address, as the CUDA book's own coordinate swap
- That C++ struct field order changes `sizeof` through alignment padding alone, on this book's own compiler, reproducing the CUDA book's exact `16`-byte vs `12`-byte comparison -- and what `c10::GetCPUAllocator()`, this book's own real allocator, actually guarantees about the addresses it hands back, measured across many allocations rather than trusted from a single sample
- That broadcasting a `[3]` vector against a `[4, 3]` shape is a real, zero-copy `torch::Tensor` operation, `.expand()`, that produces the CUDA book's exact `[0, 1]` stride pair and can be shown, by real address, to make all four logical rows alias the identical three physical floats
- That `torch::Tensor` supports a genuine third layout strategy beyond row-major and column-major -- `torch::MemoryFormat::ChannelsLast` -- closing the promise Chapter 6 made about layout strategies this book had not yet tested

**What you need to know first:**

- Chapter 6's Section 6.2, which derived and verified the row-major stride formula, and Section 6.4, which verified the CUDA book's own offset formula against five real coordinates -- this chapter reuses both directly rather than re-deriving them
- Chapter 3's `.stride()` and `.is_contiguous()` tooling, and Chapter 4's `.data_ptr()` identity check for distinguishing a view from a copy -- this chapter's TensorView and broadcasting sections lean on both
- If you've read the CUDA C++ edition's Chapter 7: it hand-builds `TensorView` as a new, separate, non-owning struct because its `Tensor` type from Chapter 6 owns its buffer and can't safely alias someone else's. `torch::Tensor` never drew that distinction in the first place -- every `torch::Tensor` this book has constructed since Chapter 1 can already be a view or an owner, decided at construction time, so this chapter's job is to demonstrate that `torch::Tensor`'s existing view behavior already satisfies the CUDA book's own worked examples, not to build a second type alongside the first

## 7.1 Strides Revisited: One Formula, Many Layouts `[FOUNDATIONAL]`

### Intuition

Chapter 6 verified `torch::Tensor`'s default row-major strides for shape `[2, 3, 4]` against the CUDA book's own `TensorStrides::row_major()` formula. The CUDA book's Chapter 7 makes a second claim that Chapter 6 never tested: the *same* offset formula, fed a *different* set of strides, computes an entirely different physical layout for the identical logical shape. Column-major -- the layout cuBLAS inherits from Fortran BLAS, where the outermost dimension is the one that's contiguous -- is the CUDA book's own worked example of this.

### Background

The CUDA book's own numbers for shape `[2, 3, 4]`: row-major strides `[12, 4, 1]`, column-major strides `[1, 2, 6]`. Three coordinates get worked by hand in the CUDA book's own text: `(0,0,1)` lands at offset `1` row-major and `6` column-major, `(1,0,0)` lands at `12` row-major and `1` column-major, and `(1,1,2)` -- the coordinate Chapter 6's own offset formula was tested against -- lands at `18` row-major and `15` column-major. `torch::Tensor` never needs a second `col_major()` static method the way the CUDA book's `TensorStrides` does; `torch::empty_strided()` accepts any explicit stride tuple directly.

### Worked Example 7.1.1 -- both layouts, three coordinates, one real tensor pair

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.1 contrasts row-major strides (computed
// right-to-left, innermost dimension contiguous) with column-major strides
// (computed left-to-right, OUTERMOST dimension contiguous, the convention
// cuBLAS inherits from Fortran BLAS). For shape [2,3,4]: row-major yields
// [12,4,1], column-major yields [1,2,6], and the same coordinate (1,1,2)
// lands at flat offset 18 under one and 15 under the other. torch::Tensor
// defaults to row-major, but genuinely supports arbitrary explicit strides
// via torch::empty_strided() -- this file builds BOTH layouts for real and
// verifies both of the CUDA book's own offset numbers directly.
int main() {
    torch::Tensor row_major = torch::arange(0, 24, torch::kFloat32).reshape({2, 3, 4});
    std::cout << "row_major.strides() = " << row_major.strides() << std::endl;

    // Explicit column-major strides, computed left-to-right the way the
    // CUDA book's col_major() static method does: strides[0]=1,
    // strides[d] = strides[d-1] * dims[d-1].
    torch::Tensor col_major = torch::empty_strided({2, 3, 4}, {1, 2, 6});
    col_major.copy_(row_major);  // fill with the same logical values, different physical layout
    std::cout << "col_major.strides() = " << col_major.strides() << std::endl;

    // Offsets for three of the CUDA book's own worked coordinates under
    // each layout, computed via the identical formula the CUDA book itself
    // uses: offset = sum(coord[d]*stride[d]).
    auto offset_of = [](const torch::Tensor& t, int i0, int i1, int i2) {
        return i0 * t.stride(0) + i1 * t.stride(1) + i2 * t.stride(2);
    };
    struct Coord { int i0, i1, i2; int64_t expected_row; int64_t expected_col; };
    Coord coords[] = {
        {0, 0, 1, 1, 6},
        {1, 0, 0, 12, 1},
        {1, 1, 2, 18, 15},
    };
    for (const auto& c : coords) {
        int64_t rm = offset_of(row_major, c.i0, c.i1, c.i2);
        int64_t cm = offset_of(col_major, c.i0, c.i1, c.i2);
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << ") row-major offset = " << rm
                  << ", CUDA book's own expected = " << c.expected_row << ", match = " << (rm == c.expected_row) << std::endl;
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << ") col-major offset = " << cm
                  << ", CUDA book's own expected = " << c.expected_col << ", match = " << (cm == c.expected_col) << std::endl;
    }
    int64_t row_major_offset = offset_of(row_major, 1, 1, 2);
    int64_t col_major_offset = offset_of(col_major, 1, 1, 2);

    // Confirm these are real, physically different addresses -- not the
    // same buffer read two different ways.
    float* rm_elem = &row_major[1][1][2].data_ptr<float>()[0];
    float* rm_base = row_major.data_ptr<float>();
    float* cm_elem = &col_major[1][1][2].data_ptr<float>()[0];
    float* cm_base = col_major.data_ptr<float>();
    std::cout << "real address offset, row-major: " << (rm_elem - rm_base) << std::endl;
    std::cout << "real address offset, col-major: " << (cm_elem - cm_base) << std::endl;

    // Chapter 6's own closing line promised more than "row-major or not":
    // torch::Tensor genuinely supports a third named layout strategy real
    // convolutional networks use, torch::MemoryFormat::ChannelsLast, for a
    // 4-D [N,C,H,W] tensor. It is neither row_major() nor col_major() from
    // this file's own two functions above -- it keeps N and C at the
    // outside but moves C to the INNERMOST stride, so channel values for a
    // single pixel sit next to each other in memory instead of a whole
    // channel plane sitting together.
    torch::Tensor nchw = torch::arange(0, 24, torch::kFloat32).reshape({1, 2, 3, 4});
    std::cout << "nchw.sizes() = " << nchw.sizes() << ", nchw.strides() = " << nchw.strides() << std::endl;
    std::cout << "nchw.is_contiguous() = " << nchw.is_contiguous() << std::endl;
    std::cout << "nchw.is_contiguous(ChannelsLast) = " << nchw.is_contiguous(torch::MemoryFormat::ChannelsLast) << std::endl;

    torch::Tensor nhwc = nchw.contiguous(torch::MemoryFormat::ChannelsLast);
    std::cout << "nhwc.sizes() = " << nhwc.sizes() << ", nhwc.strides() = " << nhwc.strides() << std::endl;
    std::cout << "nhwc.is_contiguous(ChannelsLast) = " << nhwc.is_contiguous(torch::MemoryFormat::ChannelsLast) << std::endl;
    std::cout << "nhwc.data_ptr() == nchw.data_ptr()? " << (nhwc.data_ptr() == nchw.data_ptr())
              << " (contiguous() into a different format triggers a real physical copy, since the source isn't already laid out that way)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_row_col_major_strides.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_row_col_major_strides
./01_row_col_major_strides
```

```text
row_major.strides() = [12, 4, 1]
col_major.strides() = [1, 2, 6]
(0,0,1) row-major offset = 1, CUDA book's own expected = 1, match = 1
(0,0,1) col-major offset = 6, CUDA book's own expected = 6, match = 1
(1,0,0) row-major offset = 12, CUDA book's own expected = 12, match = 1
(1,0,0) col-major offset = 1, CUDA book's own expected = 1, match = 1
(1,1,2) row-major offset = 18, CUDA book's own expected = 18, match = 1
(1,1,2) col-major offset = 15, CUDA book's own expected = 15, match = 1
real address offset, row-major: 18
real address offset, col-major: 15
nchw.sizes() = [1, 2, 3, 4], nchw.strides() = [24, 12, 4, 1]
nchw.is_contiguous() = 1
nchw.is_contiguous(ChannelsLast) = 0
nhwc.sizes() = [1, 2, 3, 4], nhwc.strides() = [24, 1, 8, 2]
nhwc.is_contiguous(ChannelsLast) = 1
nhwc.data_ptr() == nchw.data_ptr()? 0 (contiguous() into a different format triggers a real physical copy, since the source isn't already laid out that way)
```

Independently cross-checked outside `torch::Tensor` entirely, using NumPy's own `order='C'`/`order='F'` reshape (which implement row-major and column-major exactly the way the CUDA book's two static methods do) and its own byte-stride reporting divided by `float32`'s 4-byte width:

```text
row-major element strides: (12, 4, 1) expected [12,4,1]
col-major element strides: (1, 2, 6) expected [1,2,6]
(1,1,2) row-major offset: 18 expected 18
(1,1,2) col-major offset: 15 expected 15
```

### Discussion

Every one of the CUDA book's own three worked coordinates lands at the exact offset it publishes, under both layouts, on a real `torch::Tensor` -- not a re-derivation of the formula, the identical formula, fed strides the CUDA book's own two static methods would have computed. `torch::empty_strided({2,3,4}, {1,2,6})` does something the CUDA book's hand-built `TensorStrides` struct cannot: it lets a caller hand `torch::Tensor` *any* stride tuple directly, without writing a new named method for every layout strategy a caller might want. Row-major and column-major are two specific points on a much larger space of strides `torch::Tensor` already supports -- and `torch::MemoryFormat::ChannelsLast` is a third, real point on that same space, with its own name and its own dedicated `.is_contiguous(format)` / `.contiguous(format)` pair of methods. For a `[1, 2, 3, 4]` tensor, channels-last strides come out `[24, 1, 8, 2]` -- neither the row-major `[24, 12, 4, 1]` nor a plain column-major reordering, but a genuinely third arrangement where the channel dimension (size 2, at index 1) gets the *innermost* stride of `1`, matching the hand-derived formula `stride_n = H*W*C, stride_c = 1, stride_h = W*C, stride_w = C` for the channels-last memory order directly.

> `[COMMON TRAP]` `nchw.contiguous(torch::MemoryFormat::ChannelsLast)` returning `nhwc.data_ptr() != nchw.data_ptr()` is not a bug -- it's the honest cost of the operation. `nchw` was never laid out in channels-last order to begin with, so producing a tensor that genuinely *is* contiguous in that format requires copying every element into a new physical arrangement. `.contiguous(format)` only returns the *same* tensor, with no copy, when the tensor already satisfies that format -- calling it on a tensor that already is channels-last returns the identical `data_ptr()`, exactly the same "no-op if already true" behavior Chapter 4's `.to(kCPU)` showed for a tensor already on the CPU.

## 7.2 TensorView: Reading a Tensor Without Copying It `[FOUNDATIONAL]`

### Intuition

The CUDA book's `Tensor` type from Chapter 6 owns its buffer, so it can't safely represent "a read-only window into someone else's data" without either copying or risking a dangling pointer. Its solution is a second, deliberately separate type, `TensorView`: a non-owning struct holding a raw pointer, a shape, and strides, with a trivial destructor and a transpose/slice pair of methods that only ever adjust the pointer and the strides, never the underlying bytes. `torch::Tensor` never drew this ownership line at construction time the way the CUDA book's two-type design does -- every `torch::Tensor` already can be either an owner or a view, decided by which operation produced it, so `.t()` and indexing already are the CUDA book's `TensorView`.

### Background

The CUDA book's own Worked Example for Section 7.2 builds a `[3, 4]` view, reads `view(1, 2)` at offset `6`, transposes it into a `[4, 3]` view sharing the same pointer, and reads `t(2, 1)` at the identical offset `6` -- proving the coordinate swap and the offset formula agree. It then slices row `1` and confirms `row_slice(1)`'s pointer has advanced by `offset(1, 0) = 4` while its strides are unchanged.

### Worked Example 7.2.1 -- transpose and row-slice, on a real `torch::Tensor`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.2 hand-builds TensorView, a non-owning
// struct wrapping a raw pointer, a shape, and strides, then implements
// transpose() (swap shape dims and strides in lockstep) and row_slice()
// (advance the pointer, shrink the shape, keep the strides) entirely by
// hand. torch::Tensor's own .t() and indexing operators already ARE
// non-owning views with exactly this behavior -- no separate TensorView
// type needed. This file verifies the CUDA book's own claim directly on a
// real tensor: transposing and reading back a swapped coordinate reaches
// the SAME flat offset as the original coordinate, and slicing a row
// shares the same underlying storage with no copy.
int main() {
    torch::Tensor view = torch::arange(0, 12, torch::kFloat32).reshape({3, 4});
    std::cout << "view.sizes() = " << view.sizes() << ", view.strides() = " << view.strides() << std::endl;

    // view(1,2): row 1, col 2.
    float* base = view.data_ptr<float>();
    int64_t view_offset = 1 * view.stride(0) + 2 * view.stride(1);
    std::cout << "view(1,2) offset = " << view_offset << std::endl;

    // transpose(): swap shape AND strides together, no data movement.
    torch::Tensor t = view.t();
    std::cout << "t.sizes() = " << t.sizes() << ", t.strides() = " << t.strides() << std::endl;
    std::cout << "t.data_ptr() == view.data_ptr()? " << (t.data_ptr() == view.data_ptr())
              << " (transpose shares the identical buffer, zero bytes moved)" << std::endl;

    // t(2,1): the coordinate with row and column swapped, matching the CUDA
    // book's own claim that this reaches the identical flat offset.
    int64_t t_offset = 2 * t.stride(0) + 1 * t.stride(1);
    std::cout << "t(2,1) offset = " << t_offset << ", matches view(1,2)? " << (t_offset == view_offset) << std::endl;

    // Real address confirmation, not just the arithmetic.
    float* view_elem = &view[1][2].data_ptr<float>()[0];
    float* t_elem = &t[2][1].data_ptr<float>()[0];
    std::cout << "view[1][2] and t[2][1] are the SAME real address? " << (view_elem == t_elem) << std::endl;
    std::cout << "view[1][2] value = " << view[1][2].item<float>() << ", t[2][1] value = " << t[2][1].item<float>() << std::endl;

    // row_slice(row=1): a real row-selecting view, no copy, correct offset.
    torch::Tensor row = view[1];
    std::cout << "row.sizes() = " << row.sizes() << ", row.data_ptr() == view.data_ptr()+offset? "
              << (row.data_ptr<float>() == base + 1 * view.stride(0)) << std::endl;
    std::cout << "row[2] value = " << row[2].item<float>() << ", matches view[1][2]? "
              << (row[2].item<float>() == view[1][2].item<float>()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_view_transpose_slice.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_view_transpose_slice
./02_view_transpose_slice
```

```text
view.sizes() = [3, 4], view.strides() = [4, 1]
view(1,2) offset = 6
t.sizes() = [4, 3], t.strides() = [1, 4]
t.data_ptr() == view.data_ptr()? 1 (transpose shares the identical buffer, zero bytes moved)
t(2,1) offset = 6, matches view(1,2)? 1
view[1][2] and t[2][1] are the SAME real address? 1
view[1][2] value = 6, t[2][1] value = 6
row.sizes() = [4], row.data_ptr() == view.data_ptr()+offset? 1
row[2] value = 6, matches view[1][2]? 1
```

Cross-checked independently via NumPy's own `.transpose()` and row indexing, which implement the identical pointer-and-stride-only view semantics:

```text
view strides: (4, 1) expected [4,1]
t strides: (1, 4) expected [1,4]
view(1,2) offset: 6 t(2,1) offset: 6 match: True
```

### Discussion

`view.strides() = [4, 1]` and `t.strides() = [1, 4]` -- the exact swap the CUDA book's `transpose()` performs by hand on `TensorView`'s `shape` and `strides` members -- and `t.data_ptr() == view.data_ptr()` confirms no bytes moved to produce it. `view(1,2)` and `t(2,1)` both land at offset `6`, matching the CUDA book's own claim exactly, and `view[1][2]`/`t[2][1]` are confirmed to be the *same real address*, not merely two arithmetic computations that happen to agree. `row = view[1]` -- `torch::Tensor`'s indexing operator, rather than a dedicated `row_slice()` method -- produces a `[4]`-shaped tensor whose `.data_ptr()` sits exactly `1 * view.stride(0) = 4` elements past `view`'s own base, the identical pointer-advancement the CUDA book's `row_slice()` performs by hand. Nothing about this required a second, non-owning type: `torch::Tensor`'s own indexing and `.t()` already produce values with `TensorView`'s exact contract -- a pointer, a shape, and strides, sharing storage rather than copying it -- because `torch::Tensor` was never restricted to only ever owning its buffer in the first place.

## 7.3 Alignment and Padding `[FOUNDATIONAL]`

### Intuition

C++ compilers insert padding bytes between struct members so that every member starts at an address satisfying its own alignment requirement -- a rule that has nothing to do with GPUs at all, and applies identically on this book's own CPU compiler. The CUDA book's own worked comparison, `InterleavedHeader` (`char, int, char, int`) against `GroupedHeader` (`int, int, char, char`), demonstrates that reordering identical fields changes `sizeof` purely through how much padding the compiler has to insert.

### Background

The CUDA book's own numbers: `sizeof(InterleavedHeader) = 16` (padding after each `char`, twice), `sizeof(GroupedHeader) = 12` (padding only once, at the very end) -- four fewer bytes from reordering alone. It separately notes `cudaMalloc`'s own documented alignment guarantee, at least 256 bytes on current implementations. This book's own real CPU allocator, `c10::GetCPUAllocator()` -- first opened in Chapter 2, Section 2.4 -- makes an equivalent promise about the addresses it hands back, and this section measures what that promise actually is.

### Worked Example 7.3.1 -- identical struct comparison, plus a measured allocator guarantee

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <cstdint>

// The CUDA C++ edition's Section 7.3 defines InterleavedHeader (char, int,
// char, int) and GroupedHeader (int, int, char, char) -- identical fields,
// different declaration order, different sizeof, purely from compiler
// alignment padding. That discipline has nothing GPU-specific about it; a
// CPU compiler enforces the identical rule. This file reproduces the exact
// same two-struct comparison on this book's own compiler, then asks the
// LibTorch-specific question the CUDA book answers with cudaMalloc's own
// 256-byte guarantee: what alignment does c10::GetCPUAllocator() -- this
// book's own real allocator from Chapter 2, Section 2.4 -- actually give?
struct InterleavedHeader {
    char a;
    int b;
    char c;
    int d;
};

struct GroupedHeader {
    int b;
    int d;
    char a;
    char c;
};

int main() {
    std::cout << "sizeof(InterleavedHeader) = " << sizeof(InterleavedHeader)
              << " (char,int,char,int -- padding after each char)" << std::endl;
    std::cout << "sizeof(GroupedHeader) = " << sizeof(GroupedHeader)
              << " (int,int,char,char -- padding only at the end)" << std::endl;
    std::cout << "identical fields, different sizeof purely from order? "
              << (sizeof(InterleavedHeader) != sizeof(GroupedHeader)) << std::endl;

    std::cout << "alignof(int) = " << alignof(int) << ", alignof(char) = " << alignof(char) << std::endl;

    // c10::Half, this book's own real 16-bit type from Chapter 1, has its
    // own genuine alignment requirement, verified the same way the CUDA
    // book verifies float4's.
    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << ", alignof(c10::Half) = " << alignof(c10::Half) << std::endl;

    // LibTorch's own real CPU allocator's alignment GUARANTEE: a single
    // allocation's address can be aligned to more than the allocator
    // actually promises, just by chance of where the heap happened to place
    // it. The real guarantee is the alignment every allocation satisfies,
    // so this takes the MINIMUM alignment across 50 separate allocations
    // per size -- the direct LibTorch-native analog of the CUDA book's
    // cudaMalloc 256-byte guarantee claim.
    auto* allocator = c10::GetCPUAllocator();
    for (size_t bytes : {4, 64, 256}) {
        std::vector<c10::DataPtr> ptrs;
        int guaranteed_alignment = 4096;
        for (int i = 0; i < 50; i++) {
            c10::DataPtr ptr = allocator->allocate(bytes);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.get());
            int this_alignment = 1;
            for (int a = 4096; a >= 1; a /= 2) {
                if (addr % a == 0) { this_alignment = a; break; }
            }
            if (this_alignment < guaranteed_alignment) guaranteed_alignment = this_alignment;
            ptrs.push_back(std::move(ptr));  // keep alive so addresses don't get reused
        }
        std::cout << "allocate(" << bytes << " bytes) x50: minimum observed alignment (the actual guarantee) = "
                  << guaranteed_alignment << " bytes" << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_alignment_padding.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_alignment_padding
./03_alignment_padding
```

```text
sizeof(InterleavedHeader) = 16 (char,int,char,int -- padding after each char)
sizeof(GroupedHeader) = 12 (int,int,char,char -- padding only at the end)
identical fields, different sizeof purely from order? 1
alignof(int) = 4, alignof(char) = 1
sizeof(c10::Half) = 2, alignof(c10::Half) = 2
allocate(4 bytes) x50: minimum observed alignment (the actual guarantee) = 64 bytes
allocate(64 bytes) x50: minimum observed alignment (the actual guarantee) = 64 bytes
allocate(256 bytes) x50: minimum observed alignment (the actual guarantee) = 64 bytes
```

Reproduced across three fresh reruns of the same binary, with identical `64`-byte minimums for all three allocation sizes every time, and independently cross-checked outside `torch::Tensor` entirely via Python's `ctypes.Structure`, which enforces the identical C-struct-layout alignment rules as this book's own C++ compiler:

```text
ctypes sizeof(InterleavedHeader): 16 expected 16
ctypes sizeof(GroupedHeader): 12 expected 12
```

### Discussion

`sizeof(InterleavedHeader) = 16` and `sizeof(GroupedHeader) = 12` match the CUDA book's own published numbers exactly, and `ctypes.Structure` -- a completely independent implementation of C struct layout rules, from Python's standard library rather than this book's own compiler -- reproduces both figures without sharing a single line of code with the C++ compiler that produced them. `alignof(c10::Half) = 2` -- this book's own real 16-bit type, sitting in for the CUDA book's `float4` example -- confirms `sizeof(c10::Half)` and `alignof(c10::Half)` agree at `2`, the same "self-aligned" property the CUDA book's `float4` shows at `16`, just at a different width.

The allocator measurement is the section's genuinely LibTorch-specific finding. A single `allocate(256)` call earlier in this chapter's own development gave a misleadingly large `256`-byte alignment reading purely by chance of where the heap happened to place that one address -- exactly the trap the CUDA book's own 256-byte `cudaMalloc` guarantee warns against confusing with what a single observed pointer happens to satisfy. Sampling 50 separate, simultaneously-alive allocations per size and keeping the *minimum* alignment observed removes that luck: `c10::GetCPUAllocator()` gives a genuinely reproducible `64`-byte floor -- the size of a cache line on this book's own CPU -- for every one of the three sizes tested, `4`, `64`, and `256` bytes, confirmed identical across three fresh reruns of the same binary in a row. This is the honest LibTorch-native analog of the CUDA book's own `cudaMalloc` guarantee: not a number this book assumed by reading a manual, but a number measured directly against the same allocator this book's own tensors have been using since Chapter 2.

> `[COMMON TRAP]` A single allocation's observed alignment is not the same thing as an allocator's *guaranteed* alignment. An address that happens to land on a 256-byte boundary by chance still only proves the allocator gives *at least* whatever its true floor is -- it does not prove the floor itself is 256. This section's first draft made exactly this mistake, and the fix -- sampling many allocations and keeping the minimum -- is the general technique for turning "I observed X once" into "X is genuinely guaranteed."

## 7.4 Broadcasting: One Small Buffer, Many Logical Coordinates `[FOUNDATIONAL]`

### Intuition

Setting a dimension's stride to `0` makes every logical position along that dimension read from the identical physical address -- the CUDA book's own mechanism for broadcasting a small buffer across a larger logical shape without ever copying it. `torch::Tensor` implements the identical trick under a real, named method: `.expand()`, which stretches a size-`1` dimension to any larger size by giving it stride `0`, with no new allocation.

### Background

The CUDA book's own worked example: a `[3]` vector, real buffer `[10.0, 20.0, 30.0]`, broadcast against shape `[4, 3]` with strides `[0, 1]`. Every one of the four logical rows reads offset `col` regardless of `row`, so all four rows resolve to the identical physical values `10.0, 20.0, 30.0`.

### Worked Example 7.4.1 -- `.expand()`, reproducing the CUDA book's exact shape and strides

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.4 explains broadcasting as a pure
// stride trick: a [3] vector broadcast against a [4,3] shape does NOT
// get physically replicated into 4 rows. Instead it is viewed with an
// extra leading dimension whose STRIDE IS ZERO -- so every one of the
// 4 logical rows reads from the identical 3 physical floats. The CUDA
// book's own worked example gives shape [4,3], strides [0,1]. This
// file reproduces that exact shape/stride pair on a real torch::Tensor
// via .expand(), then proves all four logical rows resolve to the
// same physical addresses -- not just matching strides on paper.
int main() {
    torch::Tensor v = torch::tensor({10.0f, 20.0f, 30.0f});
    std::cout << "v.sizes() = " << v.sizes() << ", v.strides() = " << v.strides() << std::endl;

    // unsqueeze(0): [3] -> [1,3], a real dimension of size 1, stride
    // irrelevant since it has only one position.
    torch::Tensor v2 = v.unsqueeze(0);
    std::cout << "v.unsqueeze(0).sizes() = " << v2.sizes() << std::endl;

    // expand({4,3}): stretches the size-1 leading dimension to 4
    // WITHOUT allocating new storage -- this is the real LibTorch
    // mechanism behind the CUDA book's stride-0 trick.
    torch::Tensor b = v2.expand({4, 3});
    std::cout << "b.sizes() = " << b.sizes() << ", b.strides() = " << b.strides()
              << ", CUDA book's own expected strides = [0, 1], match = "
              << (b.stride(0) == 0 && b.stride(1) == 1) << std::endl;

    // No new allocation: still the same 3 physical floats.
    std::cout << "b.data_ptr() == v.data_ptr()? " << (b.data_ptr() == v.data_ptr())
              << " (broadcast is a view, zero bytes copied)" << std::endl;
    std::cout << "b.numel() = " << b.numel()
              << " logical elements, but only " << v.numel()
              << " physical floats actually exist" << std::endl;

    // All four logical rows must resolve to the SAME physical address
    // per column, since the row stride is 0.
    float* base = v.data_ptr<float>();
    bool all_rows_same_addr = true;
    for (int64_t row = 0; row < 4; row++) {
        for (int64_t col = 0; col < 3; col++) {
            float* elem = &b[row][col].data_ptr<float>()[0];
            float* expected = base + col * v.stride(0);
            if (elem != expected) all_rows_same_addr = false;
        }
    }
    std::cout << "all 4 logical rows read from the same 3 physical addresses? "
              << all_rows_same_addr << std::endl;

    // Values: every row must read back [10, 20, 30], the identical
    // physical data, no matter which logical row is indexed.
    std::cout << "b[0] = [" << b[0][0].item<float>() << ", " << b[0][1].item<float>() << ", " << b[0][2].item<float>() << "]"
              << ", b[3] = [" << b[3][0].item<float>() << ", " << b[3][1].item<float>() << ", " << b[3][2].item<float>() << "]"
              << std::endl;

    // The real-world payoff, mirroring the CUDA book's own framing:
    // adding a [3] bias to a [4,3] matrix triggers this exact
    // broadcast internally, with the same stride-0 mechanism.
    torch::Tensor m = torch::arange(0, 12, torch::kFloat32).reshape({4, 3});
    torch::Tensor bias = torch::tensor({1.0f, 2.0f, 3.0f});
    torch::Tensor sum = m + bias;
    std::cout << "m + bias, row 0 = [" << sum[0][0].item<float>() << ", " << sum[0][1].item<float>() << ", " << sum[0][2].item<float>() << "]"
              << ", row 3 = [" << sum[3][0].item<float>() << ", " << sum[3][1].item<float>() << ", " << sum[3][2].item<float>() << "]"
              << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_broadcast_expand_strides.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_broadcast_expand_strides
./04_broadcast_expand_strides
```

```text
v.sizes() = [3], v.strides() = [1]
v.unsqueeze(0).sizes() = [1, 3]
b.sizes() = [4, 3], b.strides() = [0, 1], CUDA book's own expected strides = [0, 1], match = 1
b.data_ptr() == v.data_ptr()? 1 (broadcast is a view, zero bytes copied)
b.numel() = 12 logical elements, but only 3 physical floats actually exist
all 4 logical rows read from the same 3 physical addresses? 1
b[0] = [10, 20, 30], b[3] = [10, 20, 30]
m + bias, row 0 = [1, 3, 5], row 3 = [10, 12, 14]
```

Cross-checked independently via NumPy's own `np.broadcast_to()`, which implements the identical stride-0 view semantics without going through `torch::Tensor` at all:

```text
broadcast_to([4,3]) strides: (0, 1) expected [0,1]
b.base is vec (view, no copy)? True
b[0]: [10.0, 20.0, 30.0] b[3]: [10.0, 20.0, 30.0]
m+bias row0: [1.0, 3.0, 5.0] row3: [10.0, 12.0, 14.0]
```

### Discussion

`b.strides() = [0, 1]` matches the CUDA book's own published stride pair exactly, and `b.data_ptr() == v.data_ptr()` confirms `.expand()` allocated nothing new -- `b` reports `12` logical elements while only `3` physical floats genuinely exist in memory. The address-by-address loop is the strongest evidence here: every one of the `4 * 3 = 12` logical positions in `b` was checked against its expected physical address directly, and all twelve resolve to one of only three real floats, confirming `all 4 logical rows read from the same 3 physical addresses? 1` rather than merely trusting the stride tuple to imply it. `m + bias` -- an ordinary tensor addition, not a method with "broadcast" in its name anywhere -- triggers this identical stride-0 mechanism internally, which is why `row 0 = [1, 3, 5]` and `row 3 = [10, 12, 14]` both add the same three-element `bias` correctly despite `m` having four rows and `bias` having only one.

> `[COMMON TRAP]` The CUDA book's own chapter summary warns that broadcasting is "safe for reading but hazardous for direct writes." That warning applies to `torch::Tensor` too, for the identical reason: `b[0][0] = 99.0f` on the expanded tensor from Worked Example 7.4.1 would write through to the single shared physical float at that column -- visibly changing `b[1][0]`, `b[2][0]`, and `b[3][0]` in the same instant, since all four "different" elements are one element wearing four logical addresses. LibTorch's higher-level operations (like the `m + bias` addition above) are written to only ever *read* from a broadcast view and materialize a fresh, non-aliased result -- but a caller writing directly into an expanded tensor's elements gets no automatic protection from that hazard.

## Complete Runnable Code

### File: `01_layout_row_col_major.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.1 contrasts row-major strides (computed
// right-to-left, innermost dimension contiguous) with column-major strides
// (computed left-to-right, OUTERMOST dimension contiguous, the convention
// cuBLAS inherits from Fortran BLAS). For shape [2,3,4]: row-major yields
// [12,4,1], column-major yields [1,2,6], and the same coordinate (1,1,2)
// lands at flat offset 18 under one and 15 under the other. torch::Tensor
// defaults to row-major, but genuinely supports arbitrary explicit strides
// via torch::empty_strided() -- this file builds BOTH layouts for real and
// verifies both of the CUDA book's own offset numbers directly.
int main() {
    torch::Tensor row_major = torch::arange(0, 24, torch::kFloat32).reshape({2, 3, 4});
    std::cout << "row_major.strides() = " << row_major.strides() << std::endl;

    // Explicit column-major strides, computed left-to-right the way the
    // CUDA book's col_major() static method does: strides[0]=1,
    // strides[d] = strides[d-1] * dims[d-1].
    torch::Tensor col_major = torch::empty_strided({2, 3, 4}, {1, 2, 6});
    col_major.copy_(row_major);  // fill with the same logical values, different physical layout
    std::cout << "col_major.strides() = " << col_major.strides() << std::endl;

    // Offsets for three of the CUDA book's own worked coordinates under
    // each layout, computed via the identical formula the CUDA book itself
    // uses: offset = sum(coord[d]*stride[d]).
    auto offset_of = [](const torch::Tensor& t, int i0, int i1, int i2) {
        return i0 * t.stride(0) + i1 * t.stride(1) + i2 * t.stride(2);
    };
    struct Coord { int i0, i1, i2; int64_t expected_row; int64_t expected_col; };
    Coord coords[] = {
        {0, 0, 1, 1, 6},
        {1, 0, 0, 12, 1},
        {1, 1, 2, 18, 15},
    };
    for (const auto& c : coords) {
        int64_t rm = offset_of(row_major, c.i0, c.i1, c.i2);
        int64_t cm = offset_of(col_major, c.i0, c.i1, c.i2);
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << ") row-major offset = " << rm
                  << ", CUDA book's own expected = " << c.expected_row << ", match = " << (rm == c.expected_row) << std::endl;
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << ") col-major offset = " << cm
                  << ", CUDA book's own expected = " << c.expected_col << ", match = " << (cm == c.expected_col) << std::endl;
    }
    int64_t row_major_offset = offset_of(row_major, 1, 1, 2);
    int64_t col_major_offset = offset_of(col_major, 1, 1, 2);

    // Confirm these are real, physically different addresses -- not the
    // same buffer read two different ways.
    float* rm_elem = &row_major[1][1][2].data_ptr<float>()[0];
    float* rm_base = row_major.data_ptr<float>();
    float* cm_elem = &col_major[1][1][2].data_ptr<float>()[0];
    float* cm_base = col_major.data_ptr<float>();
    std::cout << "real address offset, row-major: " << (rm_elem - rm_base) << std::endl;
    std::cout << "real address offset, col-major: " << (cm_elem - cm_base) << std::endl;

    // Chapter 6's own closing line promised more than "row-major or not":
    // torch::Tensor genuinely supports a third named layout strategy real
    // convolutional networks use, torch::MemoryFormat::ChannelsLast, for a
    // 4-D [N,C,H,W] tensor. It is neither row_major() nor col_major() from
    // this file's own two functions above -- it keeps N and C at the
    // outside but moves C to the INNERMOST stride, so channel values for a
    // single pixel sit next to each other in memory instead of a whole
    // channel plane sitting together.
    torch::Tensor nchw = torch::arange(0, 24, torch::kFloat32).reshape({1, 2, 3, 4});
    std::cout << "nchw.sizes() = " << nchw.sizes() << ", nchw.strides() = " << nchw.strides() << std::endl;
    std::cout << "nchw.is_contiguous() = " << nchw.is_contiguous() << std::endl;
    std::cout << "nchw.is_contiguous(ChannelsLast) = " << nchw.is_contiguous(torch::MemoryFormat::ChannelsLast) << std::endl;

    torch::Tensor nhwc = nchw.contiguous(torch::MemoryFormat::ChannelsLast);
    std::cout << "nhwc.sizes() = " << nhwc.sizes() << ", nhwc.strides() = " << nhwc.strides() << std::endl;
    std::cout << "nhwc.is_contiguous(ChannelsLast) = " << nhwc.is_contiguous(torch::MemoryFormat::ChannelsLast) << std::endl;
    std::cout << "nhwc.data_ptr() == nchw.data_ptr()? " << (nhwc.data_ptr() == nchw.data_ptr())
              << " (contiguous() into a different format triggers a real physical copy, since the source isn't already laid out that way)" << std::endl;

    return 0;
}
```

### File: `02_tensor_view_slicing_transpose.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.2 hand-builds TensorView, a non-owning
// struct wrapping a raw pointer, a shape, and strides, then implements
// transpose() (swap shape dims and strides in lockstep) and row_slice()
// (advance the pointer, shrink the shape, keep the strides) entirely by
// hand. torch::Tensor's own .t() and indexing operators already ARE
// non-owning views with exactly this behavior -- no separate TensorView
// type needed. This file verifies the CUDA book's own claim directly on a
// real tensor: transposing and reading back a swapped coordinate reaches
// the SAME flat offset as the original coordinate, and slicing a row
// shares the same underlying storage with no copy.
int main() {
    torch::Tensor view = torch::arange(0, 12, torch::kFloat32).reshape({3, 4});
    std::cout << "view.sizes() = " << view.sizes() << ", view.strides() = " << view.strides() << std::endl;

    // view(1,2): row 1, col 2.
    float* base = view.data_ptr<float>();
    int64_t view_offset = 1 * view.stride(0) + 2 * view.stride(1);
    std::cout << "view(1,2) offset = " << view_offset << std::endl;

    // transpose(): swap shape AND strides together, no data movement.
    torch::Tensor t = view.t();
    std::cout << "t.sizes() = " << t.sizes() << ", t.strides() = " << t.strides() << std::endl;
    std::cout << "t.data_ptr() == view.data_ptr()? " << (t.data_ptr() == view.data_ptr())
              << " (transpose shares the identical buffer, zero bytes moved)" << std::endl;

    // t(2,1): the coordinate with row and column swapped, matching the CUDA
    // book's own claim that this reaches the identical flat offset.
    int64_t t_offset = 2 * t.stride(0) + 1 * t.stride(1);
    std::cout << "t(2,1) offset = " << t_offset << ", matches view(1,2)? " << (t_offset == view_offset) << std::endl;

    // Real address confirmation, not just the arithmetic.
    float* view_elem = &view[1][2].data_ptr<float>()[0];
    float* t_elem = &t[2][1].data_ptr<float>()[0];
    std::cout << "view[1][2] and t[2][1] are the SAME real address? " << (view_elem == t_elem) << std::endl;
    std::cout << "view[1][2] value = " << view[1][2].item<float>() << ", t[2][1] value = " << t[2][1].item<float>() << std::endl;

    // row_slice(row=1): a real row-selecting view, no copy, correct offset.
    torch::Tensor row = view[1];
    std::cout << "row.sizes() = " << row.sizes() << ", row.data_ptr() == view.data_ptr()+offset? "
              << (row.data_ptr<float>() == base + 1 * view.stride(0)) << std::endl;
    std::cout << "row[2] value = " << row[2].item<float>() << ", matches view[1][2]? "
              << (row[2].item<float>() == view[1][2].item<float>()) << std::endl;

    return 0;
}
```

### File: `03_alignment_padding.cpp`

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <cstdint>

// The CUDA C++ edition's Section 7.3 defines InterleavedHeader (char, int,
// char, int) and GroupedHeader (int, int, char, char) -- identical fields,
// different declaration order, different sizeof, purely from compiler
// alignment padding. That discipline has nothing GPU-specific about it; a
// CPU compiler enforces the identical rule. This file reproduces the exact
// same two-struct comparison on this book's own compiler, then asks the
// LibTorch-specific question the CUDA book answers with cudaMalloc's own
// 256-byte guarantee: what alignment does c10::GetCPUAllocator() -- this
// book's own real allocator from Chapter 2, Section 2.4 -- actually give?
struct InterleavedHeader {
    char a;
    int b;
    char c;
    int d;
};

struct GroupedHeader {
    int b;
    int d;
    char a;
    char c;
};

int main() {
    std::cout << "sizeof(InterleavedHeader) = " << sizeof(InterleavedHeader)
              << " (char,int,char,int -- padding after each char)" << std::endl;
    std::cout << "sizeof(GroupedHeader) = " << sizeof(GroupedHeader)
              << " (int,int,char,char -- padding only at the end)" << std::endl;
    std::cout << "identical fields, different sizeof purely from order? "
              << (sizeof(InterleavedHeader) != sizeof(GroupedHeader)) << std::endl;

    std::cout << "alignof(int) = " << alignof(int) << ", alignof(char) = " << alignof(char) << std::endl;

    // c10::Half, this book's own real 16-bit type from Chapter 1, has its
    // own genuine alignment requirement, verified the same way the CUDA
    // book verifies float4's.
    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << ", alignof(c10::Half) = " << alignof(c10::Half) << std::endl;

    // LibTorch's own real CPU allocator's alignment GUARANTEE: a single
    // allocation's address can be aligned to more than the allocator
    // actually promises, just by chance of where the heap happened to place
    // it. The real guarantee is the alignment every allocation satisfies,
    // so this takes the MINIMUM alignment across 50 separate allocations
    // per size -- the direct LibTorch-native analog of the CUDA book's
    // cudaMalloc 256-byte guarantee claim.
    auto* allocator = c10::GetCPUAllocator();
    for (size_t bytes : {4, 64, 256}) {
        std::vector<c10::DataPtr> ptrs;
        int guaranteed_alignment = 4096;
        for (int i = 0; i < 50; i++) {
            c10::DataPtr ptr = allocator->allocate(bytes);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.get());
            int this_alignment = 1;
            for (int a = 4096; a >= 1; a /= 2) {
                if (addr % a == 0) { this_alignment = a; break; }
            }
            if (this_alignment < guaranteed_alignment) guaranteed_alignment = this_alignment;
            ptrs.push_back(std::move(ptr));  // keep alive so addresses don't get reused
        }
        std::cout << "allocate(" << bytes << " bytes) x50: minimum observed alignment (the actual guarantee) = "
                  << guaranteed_alignment << " bytes" << std::endl;
    }

    return 0;
}
```

### File: `04_broadcast_strides.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 7.4 explains broadcasting as a pure
// stride trick: a [3] vector broadcast against a [4,3] shape does NOT
// get physically replicated into 4 rows. Instead it is viewed with an
// extra leading dimension whose STRIDE IS ZERO -- so every one of the
// 4 logical rows reads from the identical 3 physical floats. The CUDA
// book's own worked example gives shape [4,3], strides [0,1]. This
// file reproduces that exact shape/stride pair on a real torch::Tensor
// via .expand(), then proves all four logical rows resolve to the
// same physical addresses -- not just matching strides on paper.
int main() {
    torch::Tensor v = torch::tensor({10.0f, 20.0f, 30.0f});
    std::cout << "v.sizes() = " << v.sizes() << ", v.strides() = " << v.strides() << std::endl;

    // unsqueeze(0): [3] -> [1,3], a real dimension of size 1, stride
    // irrelevant since it has only one position.
    torch::Tensor v2 = v.unsqueeze(0);
    std::cout << "v.unsqueeze(0).sizes() = " << v2.sizes() << std::endl;

    // expand({4,3}): stretches the size-1 leading dimension to 4
    // WITHOUT allocating new storage -- this is the real LibTorch
    // mechanism behind the CUDA book's stride-0 trick.
    torch::Tensor b = v2.expand({4, 3});
    std::cout << "b.sizes() = " << b.sizes() << ", b.strides() = " << b.strides()
              << ", CUDA book's own expected strides = [0, 1], match = "
              << (b.stride(0) == 0 && b.stride(1) == 1) << std::endl;

    // No new allocation: still the same 3 physical floats.
    std::cout << "b.data_ptr() == v.data_ptr()? " << (b.data_ptr() == v.data_ptr())
              << " (broadcast is a view, zero bytes copied)" << std::endl;
    std::cout << "b.numel() = " << b.numel()
              << " logical elements, but only " << v.numel()
              << " physical floats actually exist" << std::endl;

    // All four logical rows must resolve to the SAME physical address
    // per column, since the row stride is 0.
    float* base = v.data_ptr<float>();
    bool all_rows_same_addr = true;
    for (int64_t row = 0; row < 4; row++) {
        for (int64_t col = 0; col < 3; col++) {
            float* elem = &b[row][col].data_ptr<float>()[0];
            float* expected = base + col * v.stride(0);
            if (elem != expected) all_rows_same_addr = false;
        }
    }
    std::cout << "all 4 logical rows read from the same 3 physical addresses? "
              << all_rows_same_addr << std::endl;

    // Values: every row must read back [10, 20, 30], the identical
    // physical data, no matter which logical row is indexed.
    std::cout << "b[0] = [" << b[0][0].item<float>() << ", " << b[0][1].item<float>() << ", " << b[0][2].item<float>() << "]"
              << ", b[3] = [" << b[3][0].item<float>() << ", " << b[3][1].item<float>() << ", " << b[3][2].item<float>() << "]"
              << std::endl;

    // The real-world payoff, mirroring the CUDA book's own framing:
    // adding a [3] bias to a [4,3] matrix triggers this exact
    // broadcast internally, with the same stride-0 mechanism.
    torch::Tensor m = torch::arange(0, 12, torch::kFloat32).reshape({4, 3});
    torch::Tensor bias = torch::tensor({1.0f, 2.0f, 3.0f});
    torch::Tensor sum = m + bias;
    std::cout << "m + bias, row 0 = [" << sum[0][0].item<float>() << ", " << sum[0][1].item<float>() << ", " << sum[0][2].item<float>() << "]"
              << ", row 3 = [" << sum[3][0].item<float>() << ", " << sum[3][1].item<float>() << ", " << sum[3][2].item<float>() << "]"
              << std::endl;

    return 0;
}
```

Files `01`-`04` all compile and link against LibTorch with the standard command from *Getting Started*:

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

The offset formula Chapter 6 verified stays fixed; only the strides feeding it change to produce a different physical layout, confirmed here against three of the CUDA book's own worked coordinates under both row-major and column-major strides for shape `[2, 3, 4]` -- `(0,0,1)`, `(1,0,0)`, and `(1,1,2)` all landing at the CUDA book's own published offsets, independently cross-checked via NumPy's `order='C'`/`order='F'` reshape. `torch::Tensor` also supports a genuine third layout, `torch::MemoryFormat::ChannelsLast`, closing the promise Chapter 6 made about layout strategies beyond row-major, with real, measured strides `[24, 1, 8, 2]` for a `[1, 2, 3, 4]` tensor matching a hand-derived formula independent of `torch::Tensor`'s own internals. The CUDA book's hand-built `TensorView` turned out to need no separate type at all: `torch::Tensor`'s own `.t()` and indexing already carry TensorView's exact non-owning, pointer-plus-shape-plus-strides contract, verified by transposing a real tensor and confirming the coordinate-swapped read lands at the identical real address as the original. Struct field reordering was confirmed to change `sizeof` through alignment padding alone, reproducing the CUDA book's exact `16`-vs-`12`-byte comparison and cross-checked via Python's independent `ctypes.Structure` implementation, and this book's own real CPU allocator was measured -- across 50 samples per size rather than trusted from one -- to guarantee a genuine `64`-byte alignment floor. Broadcasting was confirmed as a real, zero-copy `torch::Tensor` operation, `.expand()`, reproducing the CUDA book's exact `[0, 1]` stride pair for a `[3]` vector broadcast against `[4, 3]`, with every one of the twelve logical positions checked by real address against the three physical floats that actually back them.

## Self-Check Questions

1. Worked Example 7.1.1 tests three coordinates, `(0,0,1)`, `(1,0,0)`, and `(1,1,2)`, against both row-major and column-major strides, rather than just the one coordinate Chapter 6 already verified. What would testing only `(1,1,2)` have left unproven that testing all three coordinates confirms?
2. Section 7.1's `[COMMON TRAP]` explains why `nhwc.data_ptr() != nchw.data_ptr()` after `.contiguous(ChannelsLast)`. Under what circumstance would calling `.contiguous(ChannelsLast)` on a tensor return the *same* `data_ptr()` instead?
3. Section 7.2 claims `torch::Tensor`'s own `.t()` and indexing operators already satisfy the CUDA book's `TensorView` contract, so no separate view type is needed. Name the one specific property the CUDA book's `TensorView` struct has (stated in its own definition) that this section's Worked Example does not directly test on `torch::Tensor`.
4. Section 7.3 rejects a single allocation's observed alignment as evidence of an allocator's guarantee, in favor of the minimum alignment across 50 samples. Suppose only 2 samples had been taken per size instead of 50, and by chance both happened to land on 128-byte boundaries. What false conclusion would that have produced, and why does a larger sample size make it less likely?
5. Section 7.4's `[COMMON TRAP]` states that writing through a broadcast view is hazardous, using `b[0][0] = 99.0f` as the example. Using the same `b.strides() = [0, 1]` from Worked Example 7.4.1, exactly which OTHER logical positions in `b` (out of all twelve) would also visibly change if that write happened, and which would stay at their original values?

## Where We Go Next

This chapter confirmed `torch::Tensor`'s memory layout is genuinely as flexible as the CUDA book's Chapter 7 claims a well-designed tensor type should be: row-major, column-major, and channels-last are all real, testable stride arrangements of the identical logical shape, `TensorView`'s non-owning contract was already present in `torch::Tensor`'s own view operations, and broadcasting was confirmed as a real, zero-copy stride-0 trick rather than a hidden allocation. Chapter 8 turns from how a tensor's memory is *arranged* to how a tensor's memory is *born* in the first place: the CUDA book's own family of tensor creation functions, and this book's own `torch::zeros`, `torch::ones`, `torch::arange`, and their relatives, tested for exactly which numbers, dtypes, and layouts each one actually produces by default.

## Worked Solutions

**1.** Testing only `(1,1,2)` would leave unproven whether the offset formula's coefficients -- `strides[0]`, `strides[1]`, `strides[2]` -- are each individually correct, or whether some combination of compensating errors happens to produce the right answer for that one specific coordinate. `(0,0,1)` isolates `strides[2]` alone (the other two coordinates are zero), and `(1,0,0)` isolates `strides[0]` alone, so testing all three coordinates confirms each stride component independently rather than only their combined effect on a single point.

**2.** `.contiguous(ChannelsLast)` returns the same `data_ptr()` when the tensor calling it is *already* laid out in channels-last order -- i.e., when `t.is_contiguous(torch::MemoryFormat::ChannelsLast)` is already `true` before the call. In that case there is nothing to physically rearrange, so `.contiguous()` (with any format argument) is a genuine no-op, exactly like Chapter 4's `.to(kCPU)` returning the identical tensor when it's already on the CPU.

**3.** The CUDA book's `TensorView` struct is explicitly documented as having a trivial destructor -- "nothing to free" -- because it never owns the memory it points into. Worked Example 7.2.1 never directly tests this property: it never constructs a `torch::Tensor` view, lets it go out of scope, and confirms the underlying buffer is still valid and unaffected afterward. It relies instead on `torch::Tensor`'s general reference-counted storage model (established in earlier chapters) to make this safe, rather than proving it fresh in this chapter.

**4.** With only 2 samples both landing on 128-byte boundaries by chance, the minimum-of-2 computation would report `128` bytes as the "guaranteed" alignment -- a false floor, since the true guarantee (measured at `64` bytes with 50 samples) is lower, and a 128-byte reading from only 2 samples could easily be coincidental rather than a genuine property of every allocation the allocator makes. A larger sample size makes this less likely because it takes many separate, independent draws from the allocator all agreeing on a higher-than-actual floor for the false conclusion to survive — the more samples taken, the higher the chance that at least one exposes the allocator's real, lower minimum.

**5.** `b.strides() = [0, 1]` means the offset formula is `offset(row, col) = row*0 + col*1 = col`, independent of `row` entirely. `b[0][0]`, `b[1][0]`, `b[2][0]`, and `b[3][0]` all compute `offset = 0`, so they are all the SAME physical float — writing `b[0][0] = 99.0f` would also visibly change `b[1][0]`, `b[2][0]`, and `b[3][0]`, three positions total, none of which the write directly touched. Every position with `col = 1` or `col = 2` (`b[0][1]` through `b[3][1]`, and `b[0][2]` through `b[3][2]`, eight positions in total) would stay at their original values, because column has a nonzero stride of `1` — each column is a genuinely distinct, non-aliased physical float, and only the row dimension's stride-`0` aliasing is in play.
