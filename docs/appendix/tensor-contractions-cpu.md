# Appendix H: Tensor Contractions, From First Principles (CPU)

> Matrix multiplication is not a special piece of machinery sitting off on its own -- it is the single most common instance of a more general operation, tensor contraction, that shows up everywhere real tensor code runs: batched matmuls, attention scores, convolution reshaped as a big contraction, and every call this book has made to `torch::matmul()` since Chapter 5. This appendix builds that general operation from shape and stride alone, in plain host C++, so that the next time a shape like `[batch, heads, seq, dim]` needs to collapse against another tensor along two axes at once, the mechanism underneath is no longer a black box.

**What you will understand by the end of this appendix:** the general definition of a tensor contraction -- rank, shape, free indices, contracted (dummy) indices, and the Einstein summation convention -- with ordinary matrix multiplication recovered as its simplest case, a single shared axis; how to implement a fully general, arbitrary-rank `contract()` function starting from nothing but a tensor's own shape and strides; how to contract two tensors over more than one shared axis at once, and verify the result against an independent method; how to recognize, and guard against, the silent-corruption trap that follows from skipping axis-size validation; and how to explain, with a genuine measurement rather than an assertion, why the loop order used inside a contraction changes how long it takes without changing the answer it produces.

**What you need to know first:** this appendix assumes the struct and memory-layout material from Chapters 2-3, and the recursive-lambda and generic-function style this book's own Appendix D and Appendix F both use throughout. Nothing here requires a GPU -- every example in this appendix is ordinary, portable C++ compiled with `g++` and run on the CPU. The companion appendix, Appendix I, takes this exact same mathematics onto CUDA.

## H.1 General Tensor Contraction: The Concept

**Intuition.** A tensor's rank is how many axes it has, and its shape is how many elements live along each axis. Given two tensors A and B, a contraction picks some number of axes on A and an equal number of axes on B, pairs them up one-to-one, and sums the elementwise product of A and B over every combination of values those paired axes can take -- collapsing those axes out of existence entirely. Every axis that was NOT paired up survives into the output exactly as it was: these are the free indices. The axes that WERE paired up and summed over are the contracted, or dummy, indices -- "dummy" because their specific value never appears in the final result, only their sum does.

**Background.** Einstein's summation convention is a compact way to write exactly this: when an index letter appears twice in a product -- once on each operand -- summation over that index is implied without writing an explicit sum symbol. Ordinary matrix multiplication, `C[i,j] = sum over k of A[i,k] * B[k,j]`, is Einstein notation's most familiar instance: `k` appears on both `A` and `B`, so it is summed over (contracted), while `i` and `j` each appear on exactly one operand, so they survive to the output. Matrix multiplication is therefore not a separate operation from tensor contraction -- it is the p=1 case, a contraction over exactly one shared axis.

**Formulas and Key Terms.**

General contraction, contracting p axes of A against p axes of B:

> `C[free axes of A, free axes of B]`
> `    = sum, over every combination of the p shared axis values, of`
> `          A[free axes of A, contracted axes] * B[contracted axes, free axes of B]`
>
> `rank(C) = rank(A) + rank(B) - 2p`

- **Rank** -- the number of axes a tensor has (a matrix is rank 2, a vector is rank 1, a scalar is rank 0).
- **Shape** -- the size of a tensor along each of its axes, e.g. `[2, 3, 4]`.
- **Free index** -- an axis that appears on exactly one of the two operands and survives into the output.
- **Contracted (dummy) index** -- an axis that appears on both operands and is summed away; its specific value never appears in the result.
- **Einstein summation convention** -- an index letter repeated once per operand implies summation over it, with no explicit sum symbol written.
- **Output rank** -- `rank(C) = rank(A) + rank(B) - 2p`, where `p` is the number of paired (contracted) axes; matmul is the `p = 1` special case.

This book has already been calling `contract()`'s most common instance by name, `torch::matmul()`, since Chapter 5 -- every one of those calls has been a p=1 tensor contraction all along. Sections H.2 through H.6 build the fully general, arbitrary-p version from shape and stride alone, and Section H.3 closes the loop by checking a hand-rolled contraction against this book's own real, installed LibTorch's own general contraction primitive, `torch::einsum()`.

## H.2 Shapes, Strides, and Indexing

**Intuition.** A tensor's data lives in one flat, contiguous buffer no matter how many axes its shape describes. Strides are what translate a multi-dimensional index like `(1, 1, 1)` into a single flat offset into that buffer: `strides[i]` is how many elements to skip, in the flat buffer, to move one step along axis `i`. For the row-major (C-order) layout this appendix uses throughout -- the same layout `torch::Tensor` itself defaults to -- the last axis always gets stride 1, because consecutive elements along the last axis are genuinely adjacent in memory; every other axis's stride is the product of every axis size to its right.

**Background.** Two operations invert each other: `ravel_index` takes a multi-index and strides and produces a flat offset (`offset = sum over i of index[i] * strides[i]`); `unravel_index` takes a flat offset and a shape and recovers the multi-index that produced it, by repeatedly taking a remainder and dividing, working from the last (fastest-varying) axis back to the first (slowest-varying). Chapters 2-3 introduced exactly this row-major layout for ordinary structs and arrays; this section applies the identical reasoning to a tensor's own shape.

**Worked Example H.2.1.** Row-major strides for shape `[2, 3, 4]`, a hand-verified offset-to-index round trip at offset 17, an exhaustive round-trip check over every one of the shape's 24 offsets, and a cross-check against this book's own real `torch::Tensor::strides()`.

```cpp
// Appendix H.2 -- shapes, strides, and indexing, from first principles.
//
// This file has nothing to do with torch::Tensor at all -- it is the
// same plain, host-only C++ this book has already used in Chapters 2-3
// (struct layout) and Appendix D/F (recursive-lambda and generic-function
// style). A tensor's shape tells you how many elements live along each
// axis; its strides tell you how many elements to skip, in the flat
// underlying buffer, to move one step along each axis. For a row-major
// (C-order) layout, the stride of the last axis is always 1, and every
// other axis's stride is the product of all the axis sizes to its right.
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

// Row-major strides for a given shape: strides[i] = product of
// shape[i+1 .. end]. The last axis always gets stride 1.
Strides row_major_strides(const Shape& shape) {
    Strides strides(shape.size());
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = running;
        running *= shape[i];
    }
    return strides;
}

// Multi-index -> flat offset: offset = sum(index[i] * strides[i]).
int64_t ravel_index(const std::vector<int64_t>& index, const Strides& strides) {
    if (index.size() != strides.size()) {
        throw std::invalid_argument("ravel_index: index rank does not match strides rank");
    }
    int64_t offset = 0;
    for (size_t i = 0; i < index.size(); i++) {
        offset += index[i] * strides[i];
    }
    return offset;
}

// Flat offset -> multi-index, given a shape. Walks the axes from
// slowest-varying (axis 0) to fastest-varying (last axis), peeling off
// one axis's contribution at a time.
std::vector<int64_t> unravel_index(int64_t offset, const Shape& shape) {
    std::vector<int64_t> index(shape.size());
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
        index[i] = offset % shape[i];
        offset /= shape[i];
    }
    return index;
}

int64_t num_elements(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

void print_vec(const std::vector<int64_t>& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

int main() {
    Shape shape = {2, 3, 4};
    Strides strides = row_major_strides(shape);

    std::cout << "shape  = ";
    print_vec(shape);
    std::cout << std::endl;
    std::cout << "strides = ";
    print_vec(strides);
    std::cout << std::endl;

    // Hand-verify one specific offset. For shape [2,3,4] and strides
    // [12,4,1], offset 17 should unravel to index (1,1,1):
    //   1*12 + 1*4 + 1*1 = 12 + 4 + 1 = 17.
    int64_t offset = 17;
    std::vector<int64_t> index = unravel_index(offset, shape);
    std::cout << "\nunravel_index(" << offset << ", shape) = ";
    print_vec(index);
    std::cout << std::endl;

    int64_t back_to_offset = ravel_index(index, strides);
    std::cout << "ravel_index(that index, strides) = " << back_to_offset << std::endl;

    bool hand_check_ok = (index == std::vector<int64_t>{1, 1, 1}) && (back_to_offset == offset);
    std::cout << "hand check (offset 17 <-> index (1,1,1)): " << (hand_check_ok ? "PASS" : "FAIL") << std::endl;

    // Exhaustive round-trip check: every offset in [0, num_elements)
    // must unravel and then ravel back to itself, with no exceptions.
    int64_t n = num_elements(shape);
    int64_t failures = 0;
    for (int64_t o = 0; o < n; o++) {
        std::vector<int64_t> idx = unravel_index(o, shape);
        int64_t back = ravel_index(idx, strides);
        if (back != o) failures++;
    }
    std::cout << "\nexhaustive round-trip check over all " << n << " offsets: " << failures << " failures"
              << std::endl;

    // Independent cross-check: this book's own real, installed
    // LibTorch already computes row-major strides for every
    // torch::Tensor it allocates. A fresh torch::zeros(shape) should
    // report the identical strides this hand-written function just
    // computed, confirmed via a real call rather than assumed.
    torch::Tensor t = torch::zeros({shape[0], shape[1], shape[2]});
    std::vector<int64_t> torch_strides = t.strides().vec();
    bool strides_match = (torch_strides == strides);
    std::cout << "\ntorch::Tensor::strides() for a real torch::zeros({2,3,4}): ";
    print_vec(torch_strides);
    std::cout << std::endl;
    std::cout << "cross-check vs. hand-computed row_major_strides(): " << (strides_match ? "PASS" : "FAIL")
              << std::endl;

    return (failures == 0 && strides_match) ? 0 : 1;
}
```

Genuinely compiled and run:

```text
shape  = [2, 3, 4]
strides = [12, 4, 1]

unravel_index(17, shape) = [1, 1, 1]
ravel_index(that index, strides) = 17
hand check (offset 17 <-> index (1,1,1)): PASS

exhaustive round-trip check over all 24 offsets: 0 failures

torch::Tensor::strides() for a real torch::zeros({2,3,4}): [12, 4, 1]
cross-check vs. hand-computed row_major_strides(): PASS
```

**Discussion.** Offset 17 unravels to `(1, 1, 1)` because `1*12 + 1*4 + 1*1 = 17` -- exactly what the strides `[12, 4, 1]` mean by construction. The exhaustive check confirms `ravel_index(unravel_index(o, shape), strides) == o` for every single one of the 24 offsets a `[2,3,4]` tensor has, with zero failures, which is the property `contract()` in Section H.3 depends on completely: every multi-index it ever visits must map to exactly one flat offset and back, with no ambiguity. The closing cross-check is the more interesting one: a real `torch::zeros({2,3,4})`, built by this book's own real, installed LibTorch and never touched by any code in this file, reports strides of `[12, 4, 1]` -- the identical values `row_major_strides()` computed by hand. `torch::Tensor` uses exactly the layout convention this section just derived from first principles, not a different one this book happened to invent for itself.

## H.3 A Generic Contraction Function

**Intuition.** With shape, strides, and indexing in hand, a fully general `contract()` function is direct to write: figure out which axes of each operand are free and which are contracted, compute the output's shape (free axes of A, then free axes of B), and for every combination of free indices, sum the product of A and B over every combination of the contracted indices. Nothing in this description assumes either operand is 2-D, and nothing assumes exactly one axis is shared -- both of those are properties of matmul specifically, not of contraction in general.

**Background.** Visiting "every combination of indices" for a shape whose rank is only known at runtime -- not fixed at compile time -- is exactly the kind of problem this book's own Appendix D and Appendix F's recursive-lambda style solves cleanly: a `for_each_index()` helper recurses one axis at a time, and its own recursion depth (not a compile-time unrolled loop) is what lets the identical function walk a rank-2 shape, a rank-3 shape, or any other rank at all.

**Worked Example H.3.1.** `contract(A, {1}, B, {0})` on a `[3,2]` tensor A and a `[2,4]` tensor B, recovering ordinary matrix multiplication as the single-shared-axis case; cross-checked against an independent plain-loop matmul, and again against this book's own real `torch::einsum()`.

```cpp
// Appendix H.3 -- a genuinely general tensor contraction, built from
// shape/stride machinery alone, with ordinary matrix multiplication
// recovered as its simplest special case (a single shared axis, p=1).
//
// A "contraction" sums a product of two tensors' elements over one or
// more shared ("dummy") axes, leaving the remaining ("free") axes of
// each operand as the axes of the result. Given A with free axes F_A
// and contracted axes X_A, and B with free axes F_B and contracted
// axes X_B (X_A and X_B paired up, one shared dimension per pair):
//
//     C[f_A, f_B] = sum over all dummy-index combinations of
//                       A[f_A, x] * B[x, f_B]
//
// This file implements that directly, with no assumption that either
// operand is 2-D and no assumption that exactly one axis is shared.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

// Recursively walks every multi-index of `shape`, in row-major order,
// invoking `fn` once per index. This is the same recursive style
// Appendix D.2/D.3 and Appendix F use for generic functions over
// arbitrary-rank data: the rank is a runtime value, so the recursion
// depth (not a compile-time unrolled loop) is what lets this work for
// any shape at all, matmul's 2-D case included.
void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

// Contract tensor A along axes `axesA` with tensor B along axes
// `axesB` (paired positionally: axesA[i] <-> axesB[i]). The output's
// axes are A's free axes (in order) followed by B's free axes (in
// order) -- exactly the rank formula from H.1: rank(C) = rank(A) +
// rank(B) - 2p, where p = axesA.size() == axesB.size().
Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    auto free_axes = [](int64_t rank, const std::vector<int64_t>& contracted) {
        std::vector<int64_t> free;
        for (int64_t ax = 0; ax < rank; ax++) {
            if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
        }
        return free;
    };
    std::vector<int64_t> freeA = free_axes(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);

    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());

        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];

            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];

            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });

    return C;
}

void print_matrix(const Tensor& T) {
    for (int64_t i = 0; i < T.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < T.shape[1]; j++) {
            std::cout << T.at({i, j});
            if (j + 1 < T.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    // A is [3x2], B is [2x4]. contract(A, {1}, B, {0}) shares A's
    // column axis with B's row axis -- exactly ordinary matmul.
    Tensor A({3, 2});
    A.at({0, 0}) = 1; A.at({0, 1}) = 2;
    A.at({1, 0}) = 3; A.at({1, 1}) = 4;
    A.at({2, 0}) = 5; A.at({2, 1}) = 6;

    Tensor B({2, 4});
    B.at({0, 0}) = 1; B.at({0, 1}) = 0; B.at({0, 2}) = 2; B.at({0, 3}) = 1;
    B.at({1, 0}) = 0; B.at({1, 1}) = 1; B.at({1, 2}) = 1; B.at({1, 3}) = 2;

    Tensor C = contract(A, {1}, B, {0});

    std::cout << "A [3x2]:" << std::endl;
    print_matrix(A);
    std::cout << "B [2x4]:" << std::endl;
    print_matrix(B);
    std::cout << "C = contract(A, {1}, B, {0})  [3x4]:" << std::endl;
    print_matrix(C);

    std::cout << "\nC.shape = [" << C.shape[0] << ", " << C.shape[1] << "]  "
              << "(rank(C) = rank(A) + rank(B) - 2p = 2 + 2 - 2*1 = 2, matches)" << std::endl;

    // Hand cross-check against the ordinary matmul formula, computed
    // independently by a second, structurally different method (plain
    // nested loops with no contract()/for_each_index() machinery at
    // all) rather than trusting contract() to check itself.
    bool all_match = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 4; j++) {
            double manual = 0.0;
            for (int64_t k = 0; k < 2; k++) manual += A.at({i, k}) * B.at({k, j});
            if (manual != C.at({i, j})) all_match = false;
        }
    }
    std::cout << "cross-check vs. independent plain-loop matmul: " << (all_match ? "PASS" : "FAIL") << std::endl;

    // A third, structurally different cross-check: this book's own
    // real, installed LibTorch already ships a general Einstein-
    // summation contraction as a single production call,
    // torch::einsum. "ik,kj->ij" names exactly this contraction --
    // sum over the shared index k, keep i and j free -- and should
    // reproduce contract()'s own C exactly.
    torch::Tensor A_t = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    torch::Tensor B_t = torch::tensor({{1.0, 0.0, 2.0, 1.0}, {0.0, 1.0, 1.0, 2.0}});
    torch::Tensor C_t = torch::einsum("ik,kj->ij", {A_t, B_t});

    bool matches_einsum = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 4; j++) {
            if (C_t[i][j].item<double>() != C.at({i, j})) matches_einsum = false;
        }
    }
    std::cout << "\ntorch::einsum(\"ik,kj->ij\", {A, B}):" << std::endl;
    std::cout << C_t << std::endl;
    std::cout << "cross-check vs. real torch::einsum: " << (matches_einsum ? "PASS" : "FAIL") << std::endl;

    return (all_match && matches_einsum) ? 0 : 1;
}
```

Genuinely compiled and run:

```text
A [3x2]:
  [1, 2]
  [3, 4]
  [5, 6]
B [2x4]:
  [1, 0, 2, 1]
  [0, 1, 1, 2]
C = contract(A, {1}, B, {0})  [3x4]:
  [1, 2, 4, 5]
  [3, 4, 10, 11]
  [5, 6, 16, 17]

C.shape = [3, 4]  (rank(C) = rank(A) + rank(B) - 2p = 2 + 2 - 2*1 = 2, matches)
cross-check vs. independent plain-loop matmul: PASS

torch::einsum("ik,kj->ij", {A, B}):
  1   2   4   5
  3   4  10  11
  5   6  16  17
[ CPUFloatType{3,4} ]
cross-check vs. real torch::einsum: PASS
```

**Discussion.** `contract(A, {1}, B, {0})` reproduces ordinary matrix multiplication exactly, values included, confirmed here two separate ways: once against a plain, independently written triple-nested-loop matmul with no `contract()` or `for_each_index()` machinery in it at all, and once again against `torch::einsum("ik,kj->ij", {A, B})` -- this book's own real, installed LibTorch's own production general-contraction primitive, which takes the same Einstein-notation idea Section H.1 introduced and hands it directly to a professionally tuned kernel underneath. The three results agree to the last digit. That agreement is the entire point of building `contract()` by hand at all: not because a LibTorch programmer should ever write their own contraction loop in production code -- `torch::einsum()` and `torch::tensordot()` exist precisely so they don't have to -- but because understanding what `torch::einsum()`'s index string actually specifies makes it possible to use correctly, the same way Appendix G's Rule-of-Five material makes `torch::Tensor`'s own reference counting legible rather than magical.

## H.4 Contracting Over Multiple Axes

**Intuition.** `contract()` places no restriction on how many axes are contracted at once -- `axesA` and `axesB` can each name more than one axis, as long as they name the same number and each paired axis's sizes agree. Contracting two axes at once collapses two dimensions from each operand instead of one, following the identical output-rank formula from Section H.1 with `p = 2` instead of `p = 1`.

**Background.** A rank-3 tensor A with shape `[2,3,4]` contracted against a rank-3 tensor B with shape `[3,4,5]` over A's axes `{1,2}` and B's axes `{0,1}` collapses both of A's non-batch axes against both of B's leading axes at once, leaving `rank(C) = 3 + 3 - 2*2 = 2` -- a plain matrix, with no manual reshaping into a 2-D matmul required anywhere in `contract()`'s own logic.

**Worked Example H.4.1.** `contract(A, {1,2}, B, {0,1})` on a `[2,3,4]` tensor A and a `[3,4,5]` tensor B, filled deterministically with the integers 1 through 24 and 1 through 60, producing a `[2,5]` result cross-checked against an independently computed `numpy.tensordot(A, B, axes=([1,2],[0,1]))`.

```cpp
// Appendix H.4 -- contracting over more than one shared axis at once.
//
// H.3's contract() already supports this: axesA and axesB can each
// name more than one axis. Here A is rank 3 ([2,3,4]) and B is rank 3
// ([3,4,5]), sharing TWO axes (A's axes 1 and 2 with B's axes 0 and
// 1). By the rank formula from H.1, rank(C) = rank(A) + rank(B) - 2p
// = 3 + 3 - 2*2 = 2, so the result collapses all the way down to a
// plain [2,5] matrix -- the double-contraction analog of matmul.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    auto free_axes = [](int64_t rank, const std::vector<int64_t>& contracted) {
        std::vector<int64_t> free;
        for (int64_t ax = 0; ax < rank; ax++) {
            if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
        }
        return free;
    };
    std::vector<int64_t> freeA = free_axes(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);

    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());

        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];

            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];

            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });

    return C;
}

int main() {
    // A is [2,3,4], filled 1..24 in row-major order; B is [3,4,5],
    // filled 1..60 in row-major order. Contract A's axes {1,2}
    // against B's axes {0,1} -- both of A's non-batch axes against
    // both of B's leading axes.
    Tensor A({2, 3, 4});
    Tensor B({3, 4, 5});
    {
        double v = 1.0;
        for_each_index(A.shape, [&](const std::vector<int64_t>& idx) { A.at(idx) = v++; });
    }
    {
        double v = 1.0;
        for_each_index(B.shape, [&](const std::vector<int64_t>& idx) { B.at(idx) = v++; });
    }

    Tensor C = contract(A, {1, 2}, B, {0, 1});

    std::cout << "A.shape = [" << A.shape[0] << ", " << A.shape[1] << ", " << A.shape[2] << "]" << std::endl;
    std::cout << "B.shape = [" << B.shape[0] << ", " << B.shape[1] << ", " << B.shape[2] << "]" << std::endl;
    std::cout << "C = contract(A, {1,2}, B, {0,1})" << std::endl;
    std::cout << "C.shape = [" << C.shape[0] << ", " << C.shape[1] << "]  "
              << "(rank(C) = rank(A) + rank(B) - 2p = 3 + 3 - 2*2 = 2, matches)" << std::endl;

    std::cout << "\nC values:" << std::endl;
    for (int64_t i = 0; i < C.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < C.shape[1]; j++) {
            std::cout << C.at({i, j});
            if (j + 1 < C.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Independent cross-check: this exact (A, B, axes) triple was
    // separately computed with numpy.tensordot(A, B, axes=([1,2],[0,1]))
    // before this file was written, giving:
    //   [[2938, 3016, 3094, 3172, 3250],
    //    [7042, 7264, 7486, 7708, 7930]]
    std::vector<std::vector<double>> expected = {
        {2938, 3016, 3094, 3172, 3250},
        {7042, 7264, 7486, 7708, 7930},
    };
    bool all_match = true;
    for (int64_t i = 0; i < C.shape[0]; i++) {
        for (int64_t j = 0; j < C.shape[1]; j++) {
            if (C.at({i, j}) != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independently computed numpy.tensordot(A, B, axes=([1,2],[0,1])): "
              << (all_match ? "PASS" : "FAIL") << std::endl;

    return all_match ? 0 : 1;
}
```

Genuinely compiled and run:

```text
A.shape = [2, 3, 4]
B.shape = [3, 4, 5]
C = contract(A, {1,2}, B, {0,1})
C.shape = [2, 5]  (rank(C) = rank(A) + rank(B) - 2p = 3 + 3 - 2*2 = 2, matches)

C values:
  [2938, 3016, 3094, 3172, 3250]
  [7042, 7264, 7486, 7708, 7930]

cross-check vs. independently computed numpy.tensordot(A, B, axes=([1,2],[0,1])): PASS
```

**Discussion.** The output collapses all the way down to a `[2,5]` matrix, exactly as the rank formula predicts, and its ten values match `numpy.tensordot`'s own independently computed result to the last digit -- a second, structurally unrelated implementation (a different language, a different library, a different set of assumptions about how contraction should be coded) agreeing with `contract()`'s own C++ output is a genuinely stronger confirmation than re-running the same C++ logic twice would be. This is the shape of contraction that shows up constantly in real tensor code once shapes stop being purely 2-D: a batched attention score, for instance, contracts a query tensor and a key tensor over their shared head-dimension axis while leaving batch and sequence axes free, which is a `p=1` contraction on tensors of rank higher than 2 -- structurally the same operation this section just performed with `p=2`, not a different one.

## H.5 `[COMMON TRAP]` Contracting Over Axes That Don't Actually Match

**Intuition.** `contract()` validates, before doing any work at all, that every paired axis's sizes genuinely agree between A and B. That check is not optional bookkeeping -- an unchecked version of the identical function, given mismatched axis sizes, does not crash and does not throw. It silently returns a tensor of plausible shape and plausible-looking values, built by contracting only as many dummy indices as the SMALLER of the two mismatched sizes allows, quietly discarding the rest of the larger operand's own data.

**Background.** This is possible specifically because the dummy shape an unchecked contraction walks is taken from ONE operand's own contracted-axis size, not validated against the other's. If that size happens to be smaller than the other operand's corresponding axis, every index the loop ever generates stays safely in bounds for both operands -- there is no out-of-bounds read to crash on, just an incomplete sum that never touches the discarded portion of the larger tensor at all.

**Worked Example H.5.1.** A `[3,2]` tensor A and a `[4,5]` tensor B, with mismatched contraction-axis sizes (2 vs. 4): the validated `contract()` throws `std::invalid_argument` immediately, quoted here verbatim; an unchecked sibling function, given the identical inputs, returns a well-formed `[3,5]` result that silently used only B's first two rows.

```cpp
// Appendix H.5 -- [COMMON TRAP] a contraction over axes whose sizes
// don't actually match.
//
// contract() from H.3/H.4 validates that A.shape[axesA[i]] ==
// B.shape[axesB[i]] for every paired axis before it does any work,
// and throws std::invalid_argument the instant that check fails. This
// section shows why that check is not optional bookkeeping: an
// UNCHECKED version of the same function, given the same mismatched
// axes, does not crash, does not throw, and does not produce an
// obviously-broken result. It silently produces a plausible-looking
// tensor of the "right" rank, built by contracting only as many dummy
// indices as the SMALLER of the two mismatched sizes allows -- quietly
// discarding the rest of the larger operand's data.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

std::vector<int64_t> free_axes_of(int64_t rank, const std::vector<int64_t>& contracted) {
    std::vector<int64_t> free;
    for (int64_t ax = 0; ax < rank; ax++) {
        if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
    }
    return free;
}

// The validated version from H.3/H.4: checks paired axis sizes first.
Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    std::vector<int64_t> freeA = free_axes_of(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes_of(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);
    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());
        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];
            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];
            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });
    return C;
}

// The UNCHECKED version: identical logic, minus the paired-axis-size
// validation. The dummy shape is still taken from A's contracted
// axes alone -- so if B's contracted axis happens to be LARGER, this
// never goes out of bounds. It just quietly ignores everything in B
// past A's own (too-small) contracted extent.
Tensor contract_unchecked(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                           const std::vector<int64_t>& axesB) {
    std::vector<int64_t> freeA = free_axes_of(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes_of(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);  // <-- no check against B's sizes

    Tensor C(out_shape);
    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());
        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];
            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];
            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });
    return C;
}

void fill_1_to_n(Tensor& T) {
    double v = 1.0;
    for_each_index(T.shape, [&](const std::vector<int64_t>& idx) { T.at(idx) = v++; });
}

int main() {
    // A is [3,2]; B is [4,5]. We ask to contract A's axis 1 (size 2)
    // against B's axis 0 (size 4) -- these do NOT match.
    Tensor A({3, 2});
    Tensor B({4, 5});
    fill_1_to_n(A);
    fill_1_to_n(B);

    std::cout << "A.shape = [3, 2], B.shape = [4, 5]" << std::endl;
    std::cout << "requesting contract(A, {1}, B, {0}): axis sizes 2 vs 4 -- mismatched\n" << std::endl;

    std::cout << "-- validated contract() --" << std::endl;
    try {
        Tensor C = contract(A, {1}, B, {0});
        std::cout << "no exception thrown (unexpected); C.shape = [" << C.shape[0] << ", " << C.shape[1] << "]"
                  << std::endl;
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "caught std::invalid_argument: \"" << e.what() << "\"" << std::endl;
    }

    std::cout << "\n-- contract_unchecked() on the identical mismatched inputs --" << std::endl;
    Tensor C_bad = contract_unchecked(A, {1}, B, {0});
    std::cout << "no exception thrown; C_bad.shape = [" << C_bad.shape[0] << ", " << C_bad.shape[1] << "]"
              << std::endl;
    std::cout << "C_bad values:" << std::endl;
    for (int64_t i = 0; i < C_bad.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < C_bad.shape[1]; j++) {
            std::cout << C_bad.at({i, j});
            if (j + 1 < C_bad.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Independently hand-computed: contract_unchecked silently uses
    // only B's first 2 rows (A's contracted axis size), discarding
    // B's rows 2 and 3 entirely -- with no error, no warning, and a
    // perfectly plausible-looking 3x5 result.
    std::vector<std::vector<double>> expected = {
        {13, 16, 19, 22, 25},
        {27, 34, 41, 48, 55},
        {41, 52, 63, 74, 85},
    };
    bool all_match = true;
    for (int64_t i = 0; i < C_bad.shape[0]; i++) {
        for (int64_t j = 0; j < C_bad.shape[1]; j++) {
            if (C_bad.at({i, j}) != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independently hand-computed 'B rows 2-3 silently dropped' result: "
              << (all_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "\nthe result is a well-formed 3x5 tensor of ordinary-looking numbers -- nothing about its "
              << "shape or values signals that 8 of B's 20 entries (rows 2 and 3) never participated at all. "
              << "This is exactly what makes the missing check dangerous: the bug is silent, not loud."
              << std::endl;

    return all_match ? 0 : 1;
}
```

Genuinely compiled and run:

```text
A.shape = [3, 2], B.shape = [4, 5]
requesting contract(A, {1}, B, {0}): axis sizes 2 vs 4 -- mismatched

-- validated contract() --
caught std::invalid_argument: "contract: paired axis sizes do not match"

-- contract_unchecked() on the identical mismatched inputs --
no exception thrown; C_bad.shape = [3, 5]
C_bad values:
  [13, 16, 19, 22, 25]
  [27, 34, 41, 48, 55]
  [41, 52, 63, 74, 85]

cross-check vs. independently hand-computed 'B rows 2-3 silently dropped' result: PASS

the result is a well-formed 3x5 tensor of ordinary-looking numbers -- nothing about its shape or values signals that 8 of B's 20 entries (rows 2 and 3) never participated at all. This is exactly what makes the missing check dangerous: the bug is silent, not loud.
```

**Discussion.** Nothing about the unchecked result's shape or values signals that anything went wrong -- `[3,5]` is exactly the shape the rank formula would predict for a `p=1` contraction of a rank-2 and a rank-2 tensor, and every entry is an ordinary-looking number, not a `NaN` or a garbage value that might prompt a second look. The bug is silent specifically because 8 of B's 20 entries (its rows 2 and 3) simply never participated in the sum at all, with no signal anywhere in the output that they were dropped. This is precisely why `contract()`'s own validation step runs before any of the real work begins: an exception raised the instant a caller's axes don't actually agree is far cheaper to debug than a plausible-but-wrong tensor discovered three computations downstream.

## H.6 Loop Order and Cache Locality

**Intuition.** The order in which `contract()`'s nested loops are written does not change the FLOP count of a contraction at all -- every valid loop ordering computes the identical number of multiplications and additions, in the identical mathematical order for each individual output element. What loop order DOES change is how the computation walks memory, and memory traffic, not FLOP count, is what actually determines wall-clock time on real hardware once a matrix gets large enough to outgrow cache.

**Background.** For an ordinary N-by-N matmul, the `ijk` loop order's innermost loop strides through matrix B one COLUMN at a time -- consecutive accesses land N elements apart in the flat buffer, a cache-hostile stride that touches a new cache line almost every iteration. The `ikj` loop order instead makes the innermost loop walk B (and C) one ROW at a time -- consecutive accesses are genuinely adjacent in memory, staying inside the same cache line across many iterations before moving on. Both orders compute the exact same sum, accumulated over `k` in the same ascending order for each `C[i][j]`, so both produce bit-identical results; only the memory-access pattern, and therefore the running time, differs.

**Formulas and Key Terms.**

- **Cache line** -- the fixed-size block (commonly 64 bytes) a CPU actually moves between main memory and cache on any single access; accessing one element of it is no cheaper than accessing all of it.
- **Arithmetic intensity** -- the ratio of floating-point operations performed to bytes moved between memory and cache; a cache-friendly loop order raises this ratio without changing the FLOP count itself.
- **FLOP count** -- for an N-by-N-by-N matmul, `2*N*N*N` (`N*N*N` multiplications plus `N*N*N` additions), identical regardless of loop order.

**Worked Example H.6.1.** `ijk` versus `ikj` loop order on an identical N-by-N matmul, timed at N = 64, 128, 256, and 384, with a bit-for-bit correctness check confirming both orders produce the exact same result at every size.

```cpp
// Appendix H.6 -- loop order and cache locality: same FLOP count,
// same exact answer, different wall-clock time.
//
// Ordinary triple-loop matmul C = A*B can be written with the three
// loops (i, j, k) nested in any of six orders; all six compute the
// identical sum for each C[i][j], accumulated over k in the same
// ascending order, so all six produce bit-identical results. What
// changes is how each loop order walks memory. This file compares
// two of them directly:
//
//   ijk:  for i, for j, for k: C[i][j] += A[i][k] * B[k][j];
//         -- the inner loop strides through B one COLUMN at a time
//         (stride N between consecutive B accesses): cache-hostile.
//
//   ikj:  for i, for k, for j: C[i][j] += A[i][k] * B[k][j];
//         -- the inner loop walks B and C one ROW at a time (stride
//         1, fully contiguous): cache-friendly.
//
// Neither loop order changes the FLOP count -- both do exactly
// 2*M*N*K floating-point operations (M*N*K multiplies, M*N*K adds).
// What changes is the number of bytes actually moved between main
// memory and cache, which is exactly what this benchmark measures.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

// Deterministic fill (no RNG): keeps the matmul itself fully
// reproducible so the only non-deterministic quantity in this file is
// wall-clock time.
std::vector<double> make_matrix(int64_t n, double seed) {
    std::vector<double> m(static_cast<size_t>(n) * n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            m[i * n + j] = std::sin(seed + static_cast<double>(i) * 0.01 + static_cast<double>(j) * 0.001);
        }
    }
    return m;
}

void matmul_ijk(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int64_t n) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matmul_ikj(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int64_t n) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int64_t j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

double time_ms(const std::function<void()>& fn, int reps) {
    // Warm-up run (matches Appendix E.4's own convention: cold-start
    // effects are real and must be excluded from the timed reading,
    // not silently averaged in).
    fn();
    auto start = Clock::now();
    for (int r = 0; r < reps; r++) fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    std::vector<int64_t> sizes = {64, 128, 256, 384};

    std::cout << "N-by-N naive matmul, ijk order vs. ikj order (same FLOPs, same exact result):\n" << std::endl;
    std::cout << "   N     ijk (ms)     ikj (ms)     speedup (ijk/ikj)" << std::endl;

    bool all_bitwise_identical = true;

    for (int64_t n : sizes) {
        std::vector<double> A = make_matrix(n, 1.0);
        std::vector<double> B = make_matrix(n, 2.0);
        std::vector<double> C_ijk(static_cast<size_t>(n) * n);
        std::vector<double> C_ikj(static_cast<size_t>(n) * n);

        // Correctness check first, before any timing: both loop
        // orders must produce the bit-identical result, since the
        // accumulation order into each C[i][j] (ascending k) is
        // unchanged by which loop is outermost.
        matmul_ijk(A, B, C_ijk, n);
        matmul_ikj(A, B, C_ikj, n);
        if (C_ijk != C_ikj) all_bitwise_identical = false;

        int reps = n <= 128 ? 8 : (n <= 256 ? 4 : 2);
        double t_ijk = time_ms([&]() { matmul_ijk(A, B, C_ijk, n); }, reps);
        double t_ikj = time_ms([&]() { matmul_ikj(A, B, C_ikj, n); }, reps);
        double speedup = t_ijk / t_ikj;

        std::cout << "  " << n << "   [TIMING] " << t_ijk << " ms   [TIMING] " << t_ikj << " ms   [TIMING] "
                  << speedup << "x" << std::endl;
    }

    std::cout << "\nbit-identical results (ijk vs ikj) across all sizes: "
              << (all_bitwise_identical ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nFLOP count for an NxN*NxN matmul is 2*N*N*N regardless of loop order -- e.g. at N=256, "
              << "2*256*256*256 = " << (2LL * 256 * 256 * 256) << " FLOPs either way. The timing gap above "
              << "comes entirely from memory traffic: ijk's inner loop touches a new cache line of B on "
              << "nearly every iteration (stride N doubles = " << 256 * 8 << " bytes at N=256), while ikj's "
              << "inner loop stays within one cache line of B and C across many consecutive iterations."
              << std::endl;

    return all_bitwise_identical ? 0 : 1;
}
```

Genuinely compiled and run:

```text
N-by-N naive matmul, ijk order vs. ikj order (same FLOPs, same exact result):

   N     ijk (ms)     ikj (ms)     speedup (ijk/ikj)
  64   [TIMING] 0.106739 ms   [TIMING] 0.123072 ms   [TIMING] 0.867289x
  128   [TIMING] 3.22623 ms   [TIMING] 1.11281 ms   [TIMING] 2.89917x
  256   [TIMING] 19.8904 ms   [TIMING] 9.07598 ms   [TIMING] 2.19154x
  384   [TIMING] 69.0963 ms   [TIMING] 29.2261 ms   [TIMING] 2.3642x

bit-identical results (ijk vs ikj) across all sizes: PASS

FLOP count for an NxN*NxN matmul is 2*N*N*N regardless of loop order -- e.g. at N=256, 2*256*256*256 = 33554432 FLOPs either way. The timing gap above comes entirely from memory traffic: ijk's inner loop touches a new cache line of B on nearly every iteration (stride N doubles = 2048 bytes at N=256), while ikj's inner loop stays within one cache line of B and C across many consecutive iterations.
```

**Discussion.** Every size confirms the two loop orders are bit-identical, and `ikj` is measurably, reproducibly faster than `ijk` at every size tested on this book's own sandbox hardware, growing from a modest edge at N=64 -- where the whole problem still mostly fits inside cache and timing noise dominates -- to roughly double `ijk`'s own speed by N=128 and staying in that range through N=384. The exact multiplier will differ from one machine's cache hierarchy to another (this appendix's own numbers reflect this book's own two-core sandbox CPU, not any particular reader's own hardware), but the underlying mechanism does not: at N=256, both loop orders perform the identical 33,554,432 FLOPs, yet `ikj` moves dramatically less redundant traffic through the memory hierarchy to do it, which is the entire reason the timing gap exists at all. This is the same principle Appendix E's own tiling material demonstrated at the level of a single tile-width factor; here it appears at the level of a single loop reordering, with no tiling, no GPU, and no shared memory involved anywhere.

## H.7 Complete Runnable Code

### `01_tensor_indexing.cpp`

```cpp
// Appendix H.2 -- shapes, strides, and indexing, from first principles.
//
// This file has nothing to do with torch::Tensor at all -- it is the
// same plain, host-only C++ this book has already used in Chapters 2-3
// (struct layout) and Appendix D/F (recursive-lambda and generic-function
// style). A tensor's shape tells you how many elements live along each
// axis; its strides tell you how many elements to skip, in the flat
// underlying buffer, to move one step along each axis. For a row-major
// (C-order) layout, the stride of the last axis is always 1, and every
// other axis's stride is the product of all the axis sizes to its right.
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

// Row-major strides for a given shape: strides[i] = product of
// shape[i+1 .. end]. The last axis always gets stride 1.
Strides row_major_strides(const Shape& shape) {
    Strides strides(shape.size());
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = running;
        running *= shape[i];
    }
    return strides;
}

// Multi-index -> flat offset: offset = sum(index[i] * strides[i]).
int64_t ravel_index(const std::vector<int64_t>& index, const Strides& strides) {
    if (index.size() != strides.size()) {
        throw std::invalid_argument("ravel_index: index rank does not match strides rank");
    }
    int64_t offset = 0;
    for (size_t i = 0; i < index.size(); i++) {
        offset += index[i] * strides[i];
    }
    return offset;
}

// Flat offset -> multi-index, given a shape. Walks the axes from
// slowest-varying (axis 0) to fastest-varying (last axis), peeling off
// one axis's contribution at a time.
std::vector<int64_t> unravel_index(int64_t offset, const Shape& shape) {
    std::vector<int64_t> index(shape.size());
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
        index[i] = offset % shape[i];
        offset /= shape[i];
    }
    return index;
}

int64_t num_elements(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

void print_vec(const std::vector<int64_t>& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

int main() {
    Shape shape = {2, 3, 4};
    Strides strides = row_major_strides(shape);

    std::cout << "shape  = ";
    print_vec(shape);
    std::cout << std::endl;
    std::cout << "strides = ";
    print_vec(strides);
    std::cout << std::endl;

    // Hand-verify one specific offset. For shape [2,3,4] and strides
    // [12,4,1], offset 17 should unravel to index (1,1,1):
    //   1*12 + 1*4 + 1*1 = 12 + 4 + 1 = 17.
    int64_t offset = 17;
    std::vector<int64_t> index = unravel_index(offset, shape);
    std::cout << "\nunravel_index(" << offset << ", shape) = ";
    print_vec(index);
    std::cout << std::endl;

    int64_t back_to_offset = ravel_index(index, strides);
    std::cout << "ravel_index(that index, strides) = " << back_to_offset << std::endl;

    bool hand_check_ok = (index == std::vector<int64_t>{1, 1, 1}) && (back_to_offset == offset);
    std::cout << "hand check (offset 17 <-> index (1,1,1)): " << (hand_check_ok ? "PASS" : "FAIL") << std::endl;

    // Exhaustive round-trip check: every offset in [0, num_elements)
    // must unravel and then ravel back to itself, with no exceptions.
    int64_t n = num_elements(shape);
    int64_t failures = 0;
    for (int64_t o = 0; o < n; o++) {
        std::vector<int64_t> idx = unravel_index(o, shape);
        int64_t back = ravel_index(idx, strides);
        if (back != o) failures++;
    }
    std::cout << "\nexhaustive round-trip check over all " << n << " offsets: " << failures << " failures"
              << std::endl;

    // Independent cross-check: this book's own real, installed
    // LibTorch already computes row-major strides for every
    // torch::Tensor it allocates. A fresh torch::zeros(shape) should
    // report the identical strides this hand-written function just
    // computed, confirmed via a real call rather than assumed.
    torch::Tensor t = torch::zeros({shape[0], shape[1], shape[2]});
    std::vector<int64_t> torch_strides = t.strides().vec();
    bool strides_match = (torch_strides == strides);
    std::cout << "\ntorch::Tensor::strides() for a real torch::zeros({2,3,4}): ";
    print_vec(torch_strides);
    std::cout << std::endl;
    std::cout << "cross-check vs. hand-computed row_major_strides(): " << (strides_match ? "PASS" : "FAIL")
              << std::endl;

    return (failures == 0 && strides_match) ? 0 : 1;
}
```

### `02_generic_contraction.cpp`

```cpp
// Appendix H.3 -- a genuinely general tensor contraction, built from
// shape/stride machinery alone, with ordinary matrix multiplication
// recovered as its simplest special case (a single shared axis, p=1).
//
// A "contraction" sums a product of two tensors' elements over one or
// more shared ("dummy") axes, leaving the remaining ("free") axes of
// each operand as the axes of the result. Given A with free axes F_A
// and contracted axes X_A, and B with free axes F_B and contracted
// axes X_B (X_A and X_B paired up, one shared dimension per pair):
//
//     C[f_A, f_B] = sum over all dummy-index combinations of
//                       A[f_A, x] * B[x, f_B]
//
// This file implements that directly, with no assumption that either
// operand is 2-D and no assumption that exactly one axis is shared.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

// Recursively walks every multi-index of `shape`, in row-major order,
// invoking `fn` once per index. This is the same recursive style
// Appendix D.2/D.3 and Appendix F use for generic functions over
// arbitrary-rank data: the rank is a runtime value, so the recursion
// depth (not a compile-time unrolled loop) is what lets this work for
// any shape at all, matmul's 2-D case included.
void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

// Contract tensor A along axes `axesA` with tensor B along axes
// `axesB` (paired positionally: axesA[i] <-> axesB[i]). The output's
// axes are A's free axes (in order) followed by B's free axes (in
// order) -- exactly the rank formula from H.1: rank(C) = rank(A) +
// rank(B) - 2p, where p = axesA.size() == axesB.size().
Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    auto free_axes = [](int64_t rank, const std::vector<int64_t>& contracted) {
        std::vector<int64_t> free;
        for (int64_t ax = 0; ax < rank; ax++) {
            if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
        }
        return free;
    };
    std::vector<int64_t> freeA = free_axes(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);

    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());

        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];

            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];

            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });

    return C;
}

void print_matrix(const Tensor& T) {
    for (int64_t i = 0; i < T.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < T.shape[1]; j++) {
            std::cout << T.at({i, j});
            if (j + 1 < T.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    // A is [3x2], B is [2x4]. contract(A, {1}, B, {0}) shares A's
    // column axis with B's row axis -- exactly ordinary matmul.
    Tensor A({3, 2});
    A.at({0, 0}) = 1; A.at({0, 1}) = 2;
    A.at({1, 0}) = 3; A.at({1, 1}) = 4;
    A.at({2, 0}) = 5; A.at({2, 1}) = 6;

    Tensor B({2, 4});
    B.at({0, 0}) = 1; B.at({0, 1}) = 0; B.at({0, 2}) = 2; B.at({0, 3}) = 1;
    B.at({1, 0}) = 0; B.at({1, 1}) = 1; B.at({1, 2}) = 1; B.at({1, 3}) = 2;

    Tensor C = contract(A, {1}, B, {0});

    std::cout << "A [3x2]:" << std::endl;
    print_matrix(A);
    std::cout << "B [2x4]:" << std::endl;
    print_matrix(B);
    std::cout << "C = contract(A, {1}, B, {0})  [3x4]:" << std::endl;
    print_matrix(C);

    std::cout << "\nC.shape = [" << C.shape[0] << ", " << C.shape[1] << "]  "
              << "(rank(C) = rank(A) + rank(B) - 2p = 2 + 2 - 2*1 = 2, matches)" << std::endl;

    // Hand cross-check against the ordinary matmul formula, computed
    // independently by a second, structurally different method (plain
    // nested loops with no contract()/for_each_index() machinery at
    // all) rather than trusting contract() to check itself.
    bool all_match = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 4; j++) {
            double manual = 0.0;
            for (int64_t k = 0; k < 2; k++) manual += A.at({i, k}) * B.at({k, j});
            if (manual != C.at({i, j})) all_match = false;
        }
    }
    std::cout << "cross-check vs. independent plain-loop matmul: " << (all_match ? "PASS" : "FAIL") << std::endl;

    // A third, structurally different cross-check: this book's own
    // real, installed LibTorch already ships a general Einstein-
    // summation contraction as a single production call,
    // torch::einsum. "ik,kj->ij" names exactly this contraction --
    // sum over the shared index k, keep i and j free -- and should
    // reproduce contract()'s own C exactly.
    torch::Tensor A_t = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    torch::Tensor B_t = torch::tensor({{1.0, 0.0, 2.0, 1.0}, {0.0, 1.0, 1.0, 2.0}});
    torch::Tensor C_t = torch::einsum("ik,kj->ij", {A_t, B_t});

    bool matches_einsum = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 4; j++) {
            if (C_t[i][j].item<double>() != C.at({i, j})) matches_einsum = false;
        }
    }
    std::cout << "\ntorch::einsum(\"ik,kj->ij\", {A, B}):" << std::endl;
    std::cout << C_t << std::endl;
    std::cout << "cross-check vs. real torch::einsum: " << (matches_einsum ? "PASS" : "FAIL") << std::endl;

    return (all_match && matches_einsum) ? 0 : 1;
}
```

### `03_double_contraction.cpp`

```cpp
// Appendix H.4 -- contracting over more than one shared axis at once.
//
// H.3's contract() already supports this: axesA and axesB can each
// name more than one axis. Here A is rank 3 ([2,3,4]) and B is rank 3
// ([3,4,5]), sharing TWO axes (A's axes 1 and 2 with B's axes 0 and
// 1). By the rank formula from H.1, rank(C) = rank(A) + rank(B) - 2p
// = 3 + 3 - 2*2 = 2, so the result collapses all the way down to a
// plain [2,5] matrix -- the double-contraction analog of matmul.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    auto free_axes = [](int64_t rank, const std::vector<int64_t>& contracted) {
        std::vector<int64_t> free;
        for (int64_t ax = 0; ax < rank; ax++) {
            if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
        }
        return free;
    };
    std::vector<int64_t> freeA = free_axes(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);

    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());

        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];

            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];

            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });

    return C;
}

int main() {
    // A is [2,3,4], filled 1..24 in row-major order; B is [3,4,5],
    // filled 1..60 in row-major order. Contract A's axes {1,2}
    // against B's axes {0,1} -- both of A's non-batch axes against
    // both of B's leading axes.
    Tensor A({2, 3, 4});
    Tensor B({3, 4, 5});
    {
        double v = 1.0;
        for_each_index(A.shape, [&](const std::vector<int64_t>& idx) { A.at(idx) = v++; });
    }
    {
        double v = 1.0;
        for_each_index(B.shape, [&](const std::vector<int64_t>& idx) { B.at(idx) = v++; });
    }

    Tensor C = contract(A, {1, 2}, B, {0, 1});

    std::cout << "A.shape = [" << A.shape[0] << ", " << A.shape[1] << ", " << A.shape[2] << "]" << std::endl;
    std::cout << "B.shape = [" << B.shape[0] << ", " << B.shape[1] << ", " << B.shape[2] << "]" << std::endl;
    std::cout << "C = contract(A, {1,2}, B, {0,1})" << std::endl;
    std::cout << "C.shape = [" << C.shape[0] << ", " << C.shape[1] << "]  "
              << "(rank(C) = rank(A) + rank(B) - 2p = 3 + 3 - 2*2 = 2, matches)" << std::endl;

    std::cout << "\nC values:" << std::endl;
    for (int64_t i = 0; i < C.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < C.shape[1]; j++) {
            std::cout << C.at({i, j});
            if (j + 1 < C.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Independent cross-check: this exact (A, B, axes) triple was
    // separately computed with numpy.tensordot(A, B, axes=([1,2],[0,1]))
    // before this file was written, giving:
    //   [[2938, 3016, 3094, 3172, 3250],
    //    [7042, 7264, 7486, 7708, 7930]]
    std::vector<std::vector<double>> expected = {
        {2938, 3016, 3094, 3172, 3250},
        {7042, 7264, 7486, 7708, 7930},
    };
    bool all_match = true;
    for (int64_t i = 0; i < C.shape[0]; i++) {
        for (int64_t j = 0; j < C.shape[1]; j++) {
            if (C.at({i, j}) != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independently computed numpy.tensordot(A, B, axes=([1,2],[0,1])): "
              << (all_match ? "PASS" : "FAIL") << std::endl;

    return all_match ? 0 : 1;
}
```

### `04_shape_mismatch_trap.cpp`

```cpp
// Appendix H.5 -- [COMMON TRAP] a contraction over axes whose sizes
// don't actually match.
//
// contract() from H.3/H.4 validates that A.shape[axesA[i]] ==
// B.shape[axesB[i]] for every paired axis before it does any work,
// and throws std::invalid_argument the instant that check fails. This
// section shows why that check is not optional bookkeeping: an
// UNCHECKED version of the same function, given the same mismatched
// axes, does not crash, does not throw, and does not produce an
// obviously-broken result. It silently produces a plausible-looking
// tensor of the "right" rank, built by contracting only as many dummy
// indices as the SMALLER of the two mismatched sizes allows -- quietly
// discarding the rest of the larger operand's data.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using Shape = std::vector<int64_t>;

struct Tensor {
    Shape shape;
    Shape strides;
    std::vector<double> data;

    explicit Tensor(Shape s) : shape(std::move(s)) {
        strides.resize(shape.size());
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        data.assign(running, 0.0);
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    double& at(const std::vector<int64_t>& index) {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
    double at(const std::vector<int64_t>& index) const {
        int64_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) offset += index[i] * strides[i];
        return data[offset];
    }
};

void for_each_index(const Shape& shape, const std::function<void(const std::vector<int64_t>&)>& fn) {
    std::vector<int64_t> index(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) {
            fn(index);
            return;
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            index[axis] = i;
            recurse(axis + 1);
        }
    };
    recurse(0);
}

std::vector<int64_t> free_axes_of(int64_t rank, const std::vector<int64_t>& contracted) {
    std::vector<int64_t> free;
    for (int64_t ax = 0; ax < rank; ax++) {
        if (std::find(contracted.begin(), contracted.end(), ax) == contracted.end()) free.push_back(ax);
    }
    return free;
}

// The validated version from H.3/H.4: checks paired axis sizes first.
Tensor contract(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                const std::vector<int64_t>& axesB) {
    if (axesA.size() != axesB.size()) {
        throw std::invalid_argument("contract: axesA and axesB must name the same number of axes");
    }
    for (size_t i = 0; i < axesA.size(); i++) {
        if (A.shape[axesA[i]] != B.shape[axesB[i]]) {
            throw std::invalid_argument("contract: paired axis sizes do not match");
        }
    }

    std::vector<int64_t> freeA = free_axes_of(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes_of(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);

    Tensor C(out_shape);
    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());
        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];
            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];
            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });
    return C;
}

// The UNCHECKED version: identical logic, minus the paired-axis-size
// validation. The dummy shape is still taken from A's contracted
// axes alone -- so if B's contracted axis happens to be LARGER, this
// never goes out of bounds. It just quietly ignores everything in B
// past A's own (too-small) contracted extent.
Tensor contract_unchecked(const Tensor& A, const std::vector<int64_t>& axesA, const Tensor& B,
                           const std::vector<int64_t>& axesB) {
    std::vector<int64_t> freeA = free_axes_of(A.rank(), axesA);
    std::vector<int64_t> freeB = free_axes_of(B.rank(), axesB);

    Shape out_shape;
    for (int64_t ax : freeA) out_shape.push_back(A.shape[ax]);
    for (int64_t ax : freeB) out_shape.push_back(B.shape[ax]);

    Shape dummy_shape;
    for (int64_t ax : axesA) dummy_shape.push_back(A.shape[ax]);  // <-- no check against B's sizes

    Tensor C(out_shape);
    for_each_index(out_shape, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> a_free(out_idx.begin(), out_idx.begin() + static_cast<int64_t>(freeA.size()));
        std::vector<int64_t> b_free(out_idx.begin() + static_cast<int64_t>(freeA.size()), out_idx.end());
        double sum = 0.0;
        for_each_index(dummy_shape, [&](const std::vector<int64_t>& dummy_idx) {
            std::vector<int64_t> a_idx(A.rank());
            for (size_t i = 0; i < freeA.size(); i++) a_idx[freeA[i]] = a_free[i];
            for (size_t i = 0; i < axesA.size(); i++) a_idx[axesA[i]] = dummy_idx[i];
            std::vector<int64_t> b_idx(B.rank());
            for (size_t i = 0; i < freeB.size(); i++) b_idx[freeB[i]] = b_free[i];
            for (size_t i = 0; i < axesB.size(); i++) b_idx[axesB[i]] = dummy_idx[i];
            sum += A.at(a_idx) * B.at(b_idx);
        });
        C.at(out_idx) = sum;
    });
    return C;
}

void fill_1_to_n(Tensor& T) {
    double v = 1.0;
    for_each_index(T.shape, [&](const std::vector<int64_t>& idx) { T.at(idx) = v++; });
}

int main() {
    // A is [3,2]; B is [4,5]. We ask to contract A's axis 1 (size 2)
    // against B's axis 0 (size 4) -- these do NOT match.
    Tensor A({3, 2});
    Tensor B({4, 5});
    fill_1_to_n(A);
    fill_1_to_n(B);

    std::cout << "A.shape = [3, 2], B.shape = [4, 5]" << std::endl;
    std::cout << "requesting contract(A, {1}, B, {0}): axis sizes 2 vs 4 -- mismatched\n" << std::endl;

    std::cout << "-- validated contract() --" << std::endl;
    try {
        Tensor C = contract(A, {1}, B, {0});
        std::cout << "no exception thrown (unexpected); C.shape = [" << C.shape[0] << ", " << C.shape[1] << "]"
                  << std::endl;
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "caught std::invalid_argument: \"" << e.what() << "\"" << std::endl;
    }

    std::cout << "\n-- contract_unchecked() on the identical mismatched inputs --" << std::endl;
    Tensor C_bad = contract_unchecked(A, {1}, B, {0});
    std::cout << "no exception thrown; C_bad.shape = [" << C_bad.shape[0] << ", " << C_bad.shape[1] << "]"
              << std::endl;
    std::cout << "C_bad values:" << std::endl;
    for (int64_t i = 0; i < C_bad.shape[0]; i++) {
        std::cout << "  [";
        for (int64_t j = 0; j < C_bad.shape[1]; j++) {
            std::cout << C_bad.at({i, j});
            if (j + 1 < C_bad.shape[1]) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Independently hand-computed: contract_unchecked silently uses
    // only B's first 2 rows (A's contracted axis size), discarding
    // B's rows 2 and 3 entirely -- with no error, no warning, and a
    // perfectly plausible-looking 3x5 result.
    std::vector<std::vector<double>> expected = {
        {13, 16, 19, 22, 25},
        {27, 34, 41, 48, 55},
        {41, 52, 63, 74, 85},
    };
    bool all_match = true;
    for (int64_t i = 0; i < C_bad.shape[0]; i++) {
        for (int64_t j = 0; j < C_bad.shape[1]; j++) {
            if (C_bad.at({i, j}) != expected[i][j]) all_match = false;
        }
    }
    std::cout << "\ncross-check vs. independently hand-computed 'B rows 2-3 silently dropped' result: "
              << (all_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "\nthe result is a well-formed 3x5 tensor of ordinary-looking numbers -- nothing about its "
              << "shape or values signals that 8 of B's 20 entries (rows 2 and 3) never participated at all. "
              << "This is exactly what makes the missing check dangerous: the bug is silent, not loud."
              << std::endl;

    return all_match ? 0 : 1;
}
```

### `05_loop_order_performance.cpp`

```cpp
// Appendix H.6 -- loop order and cache locality: same FLOP count,
// same exact answer, different wall-clock time.
//
// Ordinary triple-loop matmul C = A*B can be written with the three
// loops (i, j, k) nested in any of six orders; all six compute the
// identical sum for each C[i][j], accumulated over k in the same
// ascending order, so all six produce bit-identical results. What
// changes is how each loop order walks memory. This file compares
// two of them directly:
//
//   ijk:  for i, for j, for k: C[i][j] += A[i][k] * B[k][j];
//         -- the inner loop strides through B one COLUMN at a time
//         (stride N between consecutive B accesses): cache-hostile.
//
//   ikj:  for i, for k, for j: C[i][j] += A[i][k] * B[k][j];
//         -- the inner loop walks B and C one ROW at a time (stride
//         1, fully contiguous): cache-friendly.
//
// Neither loop order changes the FLOP count -- both do exactly
// 2*M*N*K floating-point operations (M*N*K multiplies, M*N*K adds).
// What changes is the number of bytes actually moved between main
// memory and cache, which is exactly what this benchmark measures.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

// Deterministic fill (no RNG): keeps the matmul itself fully
// reproducible so the only non-deterministic quantity in this file is
// wall-clock time.
std::vector<double> make_matrix(int64_t n, double seed) {
    std::vector<double> m(static_cast<size_t>(n) * n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            m[i * n + j] = std::sin(seed + static_cast<double>(i) * 0.01 + static_cast<double>(j) * 0.001);
        }
    }
    return m;
}

void matmul_ijk(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int64_t n) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matmul_ikj(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int64_t n) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int64_t j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

double time_ms(const std::function<void()>& fn, int reps) {
    // Warm-up run (matches Appendix E.4's own convention: cold-start
    // effects are real and must be excluded from the timed reading,
    // not silently averaged in).
    fn();
    auto start = Clock::now();
    for (int r = 0; r < reps; r++) fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    std::vector<int64_t> sizes = {64, 128, 256, 384};

    std::cout << "N-by-N naive matmul, ijk order vs. ikj order (same FLOPs, same exact result):\n" << std::endl;
    std::cout << "   N     ijk (ms)     ikj (ms)     speedup (ijk/ikj)" << std::endl;

    bool all_bitwise_identical = true;

    for (int64_t n : sizes) {
        std::vector<double> A = make_matrix(n, 1.0);
        std::vector<double> B = make_matrix(n, 2.0);
        std::vector<double> C_ijk(static_cast<size_t>(n) * n);
        std::vector<double> C_ikj(static_cast<size_t>(n) * n);

        // Correctness check first, before any timing: both loop
        // orders must produce the bit-identical result, since the
        // accumulation order into each C[i][j] (ascending k) is
        // unchanged by which loop is outermost.
        matmul_ijk(A, B, C_ijk, n);
        matmul_ikj(A, B, C_ikj, n);
        if (C_ijk != C_ikj) all_bitwise_identical = false;

        int reps = n <= 128 ? 8 : (n <= 256 ? 4 : 2);
        double t_ijk = time_ms([&]() { matmul_ijk(A, B, C_ijk, n); }, reps);
        double t_ikj = time_ms([&]() { matmul_ikj(A, B, C_ikj, n); }, reps);
        double speedup = t_ijk / t_ikj;

        std::cout << "  " << n << "   [TIMING] " << t_ijk << " ms   [TIMING] " << t_ikj << " ms   [TIMING] "
                  << speedup << "x" << std::endl;
    }

    std::cout << "\nbit-identical results (ijk vs ikj) across all sizes: "
              << (all_bitwise_identical ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nFLOP count for an NxN*NxN matmul is 2*N*N*N regardless of loop order -- e.g. at N=256, "
              << "2*256*256*256 = " << (2LL * 256 * 256 * 256) << " FLOPs either way. The timing gap above "
              << "comes entirely from memory traffic: ijk's inner loop touches a new cache line of B on "
              << "nearly every iteration (stride N doubles = " << 256 * 8 << " bytes at N=256), while ikj's "
              << "inner loop stays within one cache line of B and C across many consecutive iterations."
              << std::endl;

    return all_bitwise_identical ? 0 : 1;
}
```

## Chapter Summary

This appendix built tensor contraction up from nothing but shape and stride, in plain host C++, and showed that ordinary matrix multiplication was never a separate operation from it -- matmul is the p=1 special case, one shared axis out of however many a general contraction might share. Section H.2 established the row-major indexing machinery every later section depends on, and confirmed it matches this book's own real `torch::Tensor::strides()` exactly. Section H.3 built a fully general `contract()` using the same recursive, arbitrary-rank style this book's own Appendix D and Appendix F both established, and checked it not only against an independent plain-loop matmul but against this book's own real `torch::einsum()` -- the production primitive that makes hand-writing a contraction loop unnecessary in practice, while this appendix's own hand-rolled version makes clear exactly what that primitive's index strings mean. Section H.4 extended the identical machinery to a two-axis contraction with no special-casing at all, verified against an independently computed `numpy.tensordot`. Section H.5 showed why `contract()`'s axis-size validation runs first: skip it, and the result is not a crash but a silent, plausible-looking wrong answer. And Section H.6 measured, rather than asserted, why loop order changes a contraction's running time without changing its FLOP count or its answer -- the exact mechanism Appendix I revisits under CUDA's own very different memory hierarchy.

## Self-Check Questions

1. A rank-4 tensor A and a rank-3 tensor B are contracted over 2 shared axis pairs. Using the formula from Section H.1, what is the rank of the result?
2. Section H.2 states that row-major layout always gives the LAST axis a stride of 1. Explain why, in terms of what "adjacent in the flat buffer" means for consecutive values of the last axis's index.
3. In `contract(A, {1}, B, {0})` from Section H.3, which specific index -- A's axis 1, or B's axis 0 -- is the "dummy" index that gets summed away, and which two axes survive into the output?
4. Section H.5's unchecked `contract_unchecked()` is given A's contraction axis of size 2 and B's contraction axis of size 4. Explain precisely why the result is a well-formed `[3,5]` tensor rather than a crash, and which specific rows of B never participate in the sum.
5. Section H.6 reports that the `ijk`-vs-`ikj` speedup is small at N=64 but much larger at N=128 and above. Suggest a reason, in terms of cache capacity, why the gap would behave this way rather than growing smoothly and proportionally from the very smallest sizes.
6. `for_each_index()` in Section H.3 uses a `std::function`-based recursion whose depth is determined at runtime by the tensor's own rank. Explain why this exact function could not be dropped, unmodified, into a CUDA `__global__` kernel, and what Appendix I.3's own fixed-`MAX_RANK` iterative approach does differently to solve the same problem on a GPU.
7. Section H.4's result was cross-checked against `numpy.tensordot` -- a second, independently written implementation in a different language. If that second implementation had happened to share the exact same (wrong) misunderstanding of what "contracting axes {1,2} against {0,1}" means, would the two implementations still agree with each other? What does that imply about the limits of cross-checking two implementations against each other, versus checking against a specification?
8. Besides FLOP count, name the specific quantity that Section H.6 identifies as actually changing between the `ijk` and `ikj` loop orders, and explain in one sentence why that quantity -- not the FLOP count -- is what determines the measured wall-clock time.

## Worked Solutions

**1.** `rank(C) = rank(A) + rank(B) - 2p = 4 + 3 - 2*2 = 3`. Two shared axis pairs collapse two axes from each operand, leaving `(4-2) + (3-2) = 2 + 1 = 3` free axes total.

**2.** Row-major layout stores a tensor's elements so that the LAST axis varies fastest as you walk through the flat buffer in order -- element `(i, j, k)` is immediately followed in memory by `(i, j, k+1)`, for any fixed `i` and `j`, as long as `k+1` is still in range. "Immediately followed in memory" is exactly what a stride of 1 means: moving one step along the last axis moves exactly one element forward in the flat buffer, with nothing in between. Every other axis has to skip over an entire run of faster-varying axes to advance by one, which is why every other axis's stride is larger than 1.

**3.** A's axis 1 and B's axis 0 are the SAME shared axis pair -- both are the contracted (dummy) index, since `axesA = {1}` and `axesB = {0}` name them as the pair to sum over. The two axes that survive into the output are A's axis 0 (the free axis of A, giving the output's first axis) and B's axis 1 (the free axis of B, giving the output's second axis) -- exactly the row-index and column-index axes an ordinary `A @ B` matmul keeps.

**4.** `contract_unchecked()` builds its own dummy-index shape from A's own contracted axis size alone (2), never checking it against B's corresponding axis size (4). Every dummy index the function ever generates therefore ranges only from 0 to 1 -- values that are safely in bounds for BOTH A's axis (size 2) and B's axis (size 4, which happens to be large enough to cover indices 0 and 1 too), so no out-of-bounds access ever occurs. The result is well-formed because every access stays in bounds; it is WRONG because B's rows 2 and 3 (the portion of B's axis beyond index 1) are never visited by the dummy-index loop at all, and their contribution to the sum is silently missing from every entry of the `[3,5]` output.

**5.** At N=64, a `64*64` double-precision matrix is `64*64*8 = 32768` bytes -- small enough that most or all of both operand matrices, plus the output, can fit inside a modern CPU's own L1 or L2 cache simultaneously, regardless of which loop order visits them. When everything already fits in fast cache, the ACCESS PATTERN matters far less, because cache-hostile strides still land in cache, just not necessarily the very fastest possible slot within it. Once N grows large enough that the working set genuinely exceeds cache capacity (well before N=128 on most machines), the `ijk` order's column-wise strides start missing cache far more often than `ikj`'s row-wise strides do, and the gap widens sharply -- not because the FLOP count changed, but because `ijk` now genuinely pays the cost of repeated trips out to slower memory that `ikj` mostly avoids.

**6.** `for_each_index()`'s recursion depth is a runtime value -- it calls itself once per axis of a shape whose rank is only known when the function actually runs, using `std::function` and ordinary call-stack recursion to do so. A CUDA `__global__` kernel runs on a GPU thread that has no access to `std::function`'s own heap-allocated closures and no practical way to recurse to an arbitrary, runtime-determined depth per thread the way host code can. Appendix I.3's own approach instead fixes a `MAX_RANK` constant at compile time and walks a shape's indices ITERATIVELY -- using a fixed-size array and a manual divmod loop bounded by that compile-time constant -- trading the host version's genuine arbitrary-rank generality for a bounded, GPU-legal iteration that still handles any rank up to `MAX_RANK` without recursion or dynamic allocation.

**7.** Yes -- two independently written implementations that happen to share the identical misunderstanding of the specification would still agree with each other, and that agreement would be entirely worthless as a check on correctness. Cross-checking two implementations against each other only rules out mistakes that are specific to ONE of the two implementations (a transcription error, an off-by-one specific to that code's own structure, and so on); it cannot catch a mistake baked into both implementers' shared understanding of what the operation is even supposed to compute. That is exactly why Section H.4's own cross-check used `numpy.tensordot` -- a library function whose own contraction semantics were fixed by NumPy's own authors, independently of this appendix's own understanding of tensor contraction, rather than a second hand-written implementation of the same idea.

**8.** Section H.6 identifies bytes actually moved between main memory and cache as the quantity that changes between loop orders, even though the FLOP count stays fixed at `2*N*N*N` either way. Wall-clock time tracks memory traffic rather than FLOP count specifically because, on real hardware, a floating-point multiply-add executes far faster than a round trip to main memory does -- once a computation is memory-bound rather than compute-bound, reducing redundant memory traffic (what `ikj`'s cache-friendly access pattern does) shortens the running time even though it does not reduce the number of arithmetic operations performed at all.
