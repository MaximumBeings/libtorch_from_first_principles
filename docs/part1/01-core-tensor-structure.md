# Chapter 6: The Tensor

> "Every chapter so far has used `torch::Tensor` without ever asking what it actually is. The CUDA C++ edition answers that question by building one: a `TensorShape` struct, a `TensorStrides` struct computed from it, a `Tensor` struct composing both around two separately-`cudaMalloc`'d buffers, a hand-written offset formula, and a deliberately deleted copy constructor. This book's job from here is different from every sibling chapter before it and every one that follows: not to re-derive that design from nothing, but to open the real thing and show that the numbers this book has been trusting since Chapter 1 — `.sizes()`, `.strides()`, `.numel()` — are exactly the numbers the CUDA book's hand-built version would have produced, computed by a genuinely different implementation the reader has never seen before now."

**What you will understand by the end of this chapter:**

- That `torch::Tensor`'s `.numel()` and `.strides()` are not just conceptually similar to the CUDA book's hand-built `TensorShape::numel()` and `TensorStrides::row_major()` — for the identical shape `[2, 3, 4]`, they produce the exact same numbers, `24` and `[12, 4, 1]`, verified here against a from-scratch reimplementation of the CUDA book's own row-major formula
- Why a tensor with `requires_grad(true)` genuinely owns two separate buffers, `.data_ptr()` and `.grad().data_ptr()`, confirmed different addresses — the real LibTorch analog of the CUDA book's own `Tensor` struct owning two independent `cudaMalloc`'d buffers, `data` and `grad`
- That the CUDA book's own hand-written offset formula, `i0*strides[0] + i1*strides[1] + i2*strides[2]`, correctly predicts the real address of five specific coordinates in an actual `torch::Tensor` — the same five coordinates and the same expected offsets (0, 1, 4, 12, 23) the CUDA book's own Worked Example 6.4.1 hand-verifies
- The one place this chapter's mapping honestly diverges from the CUDA book's own design: `torch::Tensor`'s copy constructor is not deleted, and copying one shares storage rather than deep-copying it — a real, deliberate difference from the CUDA book's `= delete`, tested directly rather than assumed away
- That `.clone()` — genuinely present on `torch::Tensor`, under the identical name — is the real method that performs the CUDA book's own deep-copy contract, and that moving a `torch::Tensor` leaves the source in a real, directly observable `defined() == false` state

**What you need to know first:**

- Chapter 1's Section 1.4, where this book's own code first constructed and inspected `torch::Tensor`, and Chapter 2's Section 2.1, where `torch::TensorOptions` was read directly from LibTorch's own headers — this chapter continues that same "open the real header, run the real code" discipline on `torch::Tensor` itself
- Chapter 3's stride and contiguity tooling (`.stride()`, `.is_contiguous()`) — this chapter derives the row-major stride *formula* those tools have been reporting the output of since Chapter 3
- If you've read the CUDA C++ edition's Chapter 6: that chapter is the first of Part 1 because it's where the CUDA book starts *building* the `Tensor` type its remaining twenty-one chapters depend on, from two structs and a pair of `cudaMalloc` calls upward. This book has nothing to build here — `torch::Tensor` already exists, fully formed, in every chapter this book has written so far — so this chapter's job inverts the CUDA book's: not construct a tensor type, but open the one this book has been using all along and confirm, section by section, that its real numbers match what the CUDA book's hand-built version would have produced

## 6.1 `numel()`: The Real Number, From the Real Formula `[FOUNDATIONAL]`

### Intuition

The CUDA book's `TensorShape::numel()` multiplies every dimension size together in a loop — `dims[0] * dims[1] * ... * dims[ndim-1]` — to compute how many elements a tensor's shape implies. `torch::Tensor::numel()` is not a reimplementation of that idea this book needs to trust on faith; it's checkable directly, by computing the identical product-of-dimensions by hand and comparing it against what `torch::Tensor` itself reports.

### Background

The CUDA book's own worked example uses shape `[2, 3, 4]`, reporting `numel() = 24`. This section reuses that exact shape.

### Worked Example 6.1.1 — the CUDA book's own numbers, from `torch::Tensor`'s real implementation

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.1/6.2 hand-build TensorShape (an
// int dims[MAX_DIMS] plus a numel() computed as a product-of-dimensions
// loop) and TensorStrides (int strides[MAX_DIMS], computed right-to-left:
// strides[d] = strides[d+1] * dims[d+1]) as two separate structs a Tensor
// later composes. torch::Tensor has never needed either hand-built --
// .sizes() and .strides() are real, already-computed member functions.
// This file verifies the exact numbers the CUDA book's own shape [2,3,4]
// example produces, from the real thing rather than a hand-rolled one.
int main() {
    torch::Tensor t = torch::zeros({2, 3, 4});

    std::cout << "t.sizes() = " << t.sizes() << std::endl;
    std::cout << "t.numel() = " << t.numel() << std::endl;
    std::cout << "t.strides() = " << t.strides() << std::endl;

    // Hand-compute numel the same way the CUDA book's TensorShape::numel()
    // does: product of dims.
    int64_t hand_numel = 1;
    for (int64_t d : t.sizes()) hand_numel *= d;
    std::cout << "hand-computed numel (product of dims) = " << hand_numel << std::endl;

    // Hand-compute strides the same way TensorStrides::row_major() does:
    // right-to-left, strides[d] = strides[d+1] * dims[d+1], innermost = 1.
    auto sizes = t.sizes();
    int ndim = sizes.size();
    std::vector<int64_t> hand_strides(ndim);
    hand_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        hand_strides[d] = hand_strides[d + 1] * sizes[d + 1];
    }
    std::cout << "hand-computed strides (row_major formula) = [";
    for (int i = 0; i < ndim; i++) std::cout << hand_strides[i] << (i + 1 < ndim ? ", " : "");
    std::cout << "]" << std::endl;

    bool numel_matches = (hand_numel == t.numel());
    bool strides_match = true;
    for (int i = 0; i < ndim; i++) if (hand_strides[i] != t.stride(i)) strides_match = false;
    std::cout << "hand-computed numel matches t.numel()? " << numel_matches << std::endl;
    std::cout << "hand-computed strides match t.strides()? " << strides_match << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
t.sizes() = [2, 3, 4]
t.numel() = 24
t.strides() = [12, 4, 1]
hand-computed numel (product of dims) = 24
hand-computed strides (row_major formula) = [12, 4, 1]
hand-computed numel matches t.numel()? 1
hand-computed strides match t.strides()? 1
```

`t.numel()` reports `24` for shape `[2, 3, 4]` — the identical figure the CUDA book's own `TensorShape::numel()` produces for the identical shape, and this file's own hand-written product-of-dimensions loop, computed independently of `torch::Tensor`'s internals, confirms it: `hand-computed numel` also comes out to `24`, and the two match. `torch::Tensor` never stores a separate `TensorShape` struct the way the CUDA book's design does — `.sizes()` and `.numel()` are computed from the tensor's own internal metadata — but the *number* the CUDA book cares about, "how many elements does this shape imply," is identical either way.

## 6.2 `strides()`: The Same Row-Major Formula, Verified `[FOUNDATIONAL]`

### Intuition

`TensorStrides::row_major()` computes strides right-to-left from a shape: the innermost dimension always has stride 1 (its elements are contiguous), and every dimension moving outward multiplies the previous stride by the size of the dimension it's stepping over. This is the exact formula Chapter 3 and Chapter 4 have been reading the *output* of via `.stride()` since this book's own particle and matrix examples — this section derives and verifies the formula itself, on `torch::Tensor`'s real strides, for the first time.

### Background

Worked Example 6.1.1 already computed both `t.strides()` and a hand-built `row_major` formula side by side, for shape `[2, 3, 4]`. The CUDA book's own Worked Example 6.2.1 reports strides `[12, 4, 1]` for that exact shape — meaning stepping one position along dimension 0 skips 12 flat elements, exactly the size of the `[3, 4]` sub-array each outer index selects.

### Discussion — reading the numbers Worked Example 6.1.1 already produced

`t.strides() = [12, 4, 1]` matches the CUDA book's own published figures for shape `[2, 3, 4]` exactly, and the hand-computed `row_major` formula this file implements independently — `strides[ndim-1] = 1`, then `strides[d] = strides[d+1] * sizes[d+1]` working backward — reproduces those same three numbers without ever calling `t.strides()` to get them. `strides[2] = 1`: the innermost dimension (size 4) is contiguous, one flat element per step. `strides[1] = strides[2] * sizes[2] = 1 * 4 = 4`: stepping one position along dimension 1 skips a full row of 4 elements. `strides[0] = strides[1] * sizes[1] = 4 * 3 = 12`: stepping one position along dimension 0 skips a full `[3, 4]` sub-array, 12 elements. This is the identical derivation the CUDA book's own `TensorStrides::row_major()` performs, reaching the identical three numbers, confirmed here against `torch::Tensor`'s real, independently-computed `.strides()`.

> `[COMMON TRAP]` "Row-major" is `torch::Tensor`'s *default* stride layout for a freshly created tensor, not a guarantee every tensor satisfies. Chapter 3 and Chapter 4 both showed tensors — `.select()` views, `.t()` transposes — with strides that don't follow this formula at all, because they're views into another tensor's storage rather than tensors with their own freshly row-major-computed layout. The formula this section verifies describes how `torch::zeros({2,3,4})` lays out a *new* allocation; it doesn't describe every tensor this book has constructed since Chapter 3.

## 6.3 Two Owned Buffers, Genuinely Separate `[FOUNDATIONAL]`

### Intuition

The CUDA book's `Tensor` struct composes a shape, a strides object, and two independently-`cudaMalloc`'d buffers: `float* data` for the tensor's values, `float* grad` for its gradient, allocated separately in the constructor and freed separately in the destructor. `torch::Tensor` makes the identical structural choice for a tensor with `requires_grad(true)`: its own values live in one buffer, and once autograd computes a gradient for it, that gradient lives in a second, genuinely separate buffer — not a second view into the first one.

### Background

Chapter 1's Section 1.4 already ran a `.backward()` call and read a gradient value out of it. This section goes further: it checks the two buffers' addresses directly, the same `.data_ptr()` identity tool Chapter 4 used to distinguish a free view from a genuine copy.

### Worked Example 6.3.1 — `.data_ptr()` and `.grad().data_ptr()`, confirmed as two allocations

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.3 composes a Tensor from a shape, a
// strides object, and TWO independently-owned buffers: `float* data` and
// `float* grad`, each allocated with its own cudaMalloc call in the
// constructor. torch::Tensor's real gradient storage works the same way --
// a tensor's .grad() is a genuinely SEPARATE tensor, with its own separate
// storage, only populated once autograd actually computes it. This file
// verifies that separation directly: .data_ptr() and .grad().data_ptr()
// are two different allocations, not two views into one buffer.
int main() {
    torch::Tensor t = torch::ones({2, 3, 4}, torch::requires_grad(true));

    std::cout << "t.grad().defined() before backward() = " << t.grad().defined() << std::endl;

    torch::Tensor loss = (t * t).sum();
    loss.backward();

    std::cout << "t.grad().defined() after backward() = " << t.grad().defined() << std::endl;
    std::cout << "t.sizes() = " << t.sizes() << ", t.grad().sizes() = " << t.grad().sizes() << std::endl;

    void* data_ptr = t.data_ptr();
    void* grad_ptr = t.grad().data_ptr();
    std::cout << "t.data_ptr() == t.grad().data_ptr()? " << (data_ptr == grad_ptr)
              << " (two genuinely separate allocations, exactly like the CUDA book's two cudaMalloc calls)" << std::endl;

    // d/dt (sum(t*t)) = 2t, so grad should be filled with 2.0 everywhere,
    // since t was initialized to all ones.
    torch::Tensor expected_grad = torch::full({2, 3, 4}, 2.0);
    bool grad_correct = torch::equal(t.grad(), expected_grad);
    std::cout << "t.grad() == 2*ones(2,3,4) (d/dt sum(t^2) = 2t)? " << grad_correct << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
t.grad().defined() before backward() = 0
t.grad().defined() after backward() = 1
t.sizes() = [2, 3, 4], t.grad().sizes() = [2, 3, 4]
t.data_ptr() == t.grad().data_ptr()? 0 (two genuinely separate allocations, exactly like the CUDA book's two cudaMalloc calls)
t.grad() == 2*ones(2,3,4) (d/dt sum(t^2) = 2t)? 1
```

`t.grad().defined()` reporting `false` before `.backward()` and `true` after is the direct evidence the CUDA book's own two-`cudaMalloc`-calls design implies at the struct level: the gradient buffer genuinely doesn't come into existence — isn't allocated — until something actually computes it, unlike `t`'s own data buffer, which existed from the moment `torch::ones` returned. `t.data_ptr() == t.grad().data_ptr()` reporting `false` confirms these are two separate allocations, not one buffer serving double duty — the same structural fact the CUDA book's own two-pointer `Tensor` struct encodes, verified here against real, dynamically-allocated LibTorch storage rather than assumed from the design alone. `t.grad()` correctly equaling `2 * ones(2,3,4)` — the calculus is `d/dt sum(t^2) = 2t`, and `t` was all ones — confirms the gradient buffer holds a genuinely correct value, not merely a genuinely separate one.

## 6.4 Indexing: The CUDA Book's Own Formula, on a Real Tensor `[FOUNDATIONAL]`

### Intuition

`offset(i0, i1, i2) = i0*strides[0] + i1*strides[1] + i2*strides[2]` is the CUDA book's own hand-written formula for turning a coordinate into a flat buffer position. `torch::Tensor` never asks a caller to write this formula out — `t[i0][i1][i2]` already does it internally — but the formula still genuinely governs where each element lives in memory, and this section tests it directly: computing the CUDA book's own five example coordinates by hand, then confirming those hand-computed offsets match the real distance between actual element addresses in an actual `torch::Tensor`.

### Background

The CUDA book's Worked Example 6.4.1 hand-verifies five coordinates for shape `[2,3,4]` with strides `[12,4,1]`: `(0,0,0) -> 0`, `(0,0,1) -> 1`, `(0,1,0) -> 4`, `(1,0,0) -> 12`, `(1,2,3) -> 23`. This section reuses those exact five coordinates and those exact expected offsets.

### Worked Example 6.4.1 — the CUDA book's own five coordinates, verified against real addresses

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.4 hand-writes offset(i0,i1,i2) = i0*strides[0]
// + i1*strides[1] + i2*strides[2], then hand-verifies five coordinates for
// shape [2,3,4] with strides [12,4,1]: (0,0,0)->0, (0,0,1)->1, (0,1,0)->4,
// (1,0,0)->12, (1,2,3)->23. torch::Tensor never needs this formula written
// out -- indexing with operator[] already does it -- but the formula still
// genuinely governs where each element lives, and this file verifies it
// against the CUDA book's own five coordinates by comparing real element
// addresses (data_ptr() differences) on an actual torch::Tensor.
int main() {
    torch::Tensor t = torch::arange(0, 24, torch::kFloat32).reshape({2, 3, 4});
    std::cout << "t.sizes() = " << t.sizes() << ", t.strides() = " << t.strides() << std::endl;

    float* base = t.data_ptr<float>();

    struct Coord { int i0, i1, i2; int64_t expected_offset; };
    Coord coords[] = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
        {0, 1, 0, 4},
        {1, 0, 0, 12},
        {1, 2, 3, 23},
    };

    bool all_match = true;
    for (const auto& c : coords) {
        // Hand-computed offset, the exact formula the CUDA book's offset()
        // member function implements.
        int64_t hand_offset = c.i0 * t.stride(0) + c.i1 * t.stride(1) + c.i2 * t.stride(2);

        // Real address evidence: how many elements this coordinate's real
        // address sits from the tensor's base pointer.
        float* elem_ptr = &t[c.i0][c.i1][c.i2].data_ptr<float>()[0];
        int64_t real_offset = elem_ptr - base;

        bool matches = (hand_offset == c.expected_offset) && (real_offset == c.expected_offset);
        all_match = all_match && matches;
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << "): hand-computed offset = "
                  << hand_offset << ", real address offset = " << real_offset
                  << ", CUDA book's own expected = " << c.expected_offset
                  << ", match = " << (matches ? "true" : "false") << std::endl;
    }
    std::cout << "all five coordinates match the CUDA book's own hand-verified offsets? "
              << (all_match ? "true" : "false") << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
t.sizes() = [2, 3, 4], t.strides() = [12, 4, 1]
(0,0,0): hand-computed offset = 0, real address offset = 0, CUDA book's own expected = 0, match = true
(0,0,1): hand-computed offset = 1, real address offset = 1, CUDA book's own expected = 1, match = true
(0,1,0): hand-computed offset = 4, real address offset = 4, CUDA book's own expected = 4, match = true
(1,0,0): hand-computed offset = 12, real address offset = 12, CUDA book's own expected = 12, match = true
(1,2,3): hand-computed offset = 23, real address offset = 23, CUDA book's own expected = 23, match = true
all five coordinates match the CUDA book's own hand-verified offsets? true
```

Every one of the CUDA book's own five example coordinates comes out correct against two entirely independent checks: `hand_offset`, computed from the formula's own arithmetic, and `real_offset`, measured as the actual pointer distance between `t[i0][i1][i2]`'s real element address and the tensor's base pointer. `(1,2,3): ... = 23` is the CUDA book's own most complex example — `1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23` — reproduced here as a genuine measurement of where LibTorch actually placed that element in memory, not a description of where the formula says it should be. **Independent cross-check.** The identical five coordinates, computed against a NumPy array of the same shape via NumPy's own `.ctypes.data` pointer arithmetic — an entirely separate implementation from LibTorch, sharing no code — produced the identical five offsets.

## 6.5 Copy, Clone, and Move: Where This Book's Mapping Honestly Diverges `[FOUNDATIONAL]`

### Intuition

The CUDA book makes a deliberate design choice here: delete `Tensor`'s copy constructor entirely, so any accidental copy of two owned buffers is a compile error rather than a silent double-free waiting to happen, and provide `clone()` as the only way to get a deep copy. `torch::Tensor` makes a genuinely different choice, and this section reports it honestly rather than forcing a false equivalence: copying a `torch::Tensor` is not deleted, compiles fine, and is a *shallow* copy — both tensors end up sharing the same underlying storage, the same reference semantics Python's own tensor variables use. `.clone()` still exists, under the identical name, and still performs the CUDA book's own deep-copy contract.

### Background

This divergence isn't a flaw in either design — it's a different tradeoff for a different context. The CUDA book's `Tensor` owns raw device pointers directly, where an accidental shallow copy really would risk a double-free; `torch::Tensor` wraps its storage in a reference-counted `intrusive_ptr` internally, so a shallow copy is memory-safe by construction, which is exactly what makes Python-style "assignment shares the tensor" semantics viable at the C++ level too.

### Worked Example 6.5.1 — shallow copy, real `.clone()`, and a genuinely tested move

```cpp
#include <torch/torch.h>
#include <iostream>
#include <utility>

// The CUDA C++ edition's Section 6.5 deletes Tensor's copy constructor
// entirely (any accidental copy is a compile error), forcing clone() as
// the only way to deep-copy two owned buffers, and hand-writes a move
// constructor that steals the pointers and nulls the source.
// torch::Tensor makes a genuinely DIFFERENT design choice, honestly
// reported here rather than assumed identical: its copy constructor is
// NOT deleted -- copying a torch::Tensor is a real, valid, everyday
// operation, and it's a SHALLOW copy (both tensors share the same
// underlying storage), matching Python's own tensor reference semantics.
// .clone() is still the real, genuinely-named method that performs the
// CUDA book's own deep-copy behavior. Move construction is real too, and
// this file tests what actually happens to the moved-from tensor.
int main() {
    torch::Tensor original = torch::tensor({1.0f, 2.0f, 3.0f});

    // torch::Tensor's copy constructor: NOT deleted, compiles fine, but
    // shares storage -- the honest divergence from the CUDA book's design.
    torch::Tensor shallow_copy = original;
    std::cout << "shallow_copy.data_ptr() == original.data_ptr()? "
              << (shallow_copy.data_ptr() == original.data_ptr())
              << " (torch::Tensor's copy constructor shares storage, unlike the CUDA book's deleted one)" << std::endl;

    // Mutating through the shallow copy is visible through the original --
    // direct proof they share one buffer, not two independent ones.
    shallow_copy[0] = 99.0f;
    std::cout << "after shallow_copy[0]=99: original[0] = " << original[0].item<float>()
              << " (mutation through the copy is visible in the original)" << std::endl;

    // .clone(): the CUDA book's own real deep-copy method, genuinely present
    // on torch::Tensor with the identical name and the identical contract.
    torch::Tensor original2 = torch::tensor({1.0f, 2.0f, 3.0f});
    torch::Tensor cloned = original2.clone();
    std::cout << "cloned.data_ptr() == original2.data_ptr()? "
              << (cloned.data_ptr() == original2.data_ptr())
              << " (clone() genuinely allocates new, independent storage)" << std::endl;
    cloned[0] = 99.0f;
    std::cout << "after cloned[0]=99: original2[0] = " << original2[0].item<float>()
              << " (mutation through the clone does NOT affect the original)" << std::endl;

    // Move construction: real on torch::Tensor. What state is the
    // moved-from tensor actually left in? Tested directly, not assumed.
    torch::Tensor to_move = torch::tensor({4.0f, 5.0f, 6.0f});
    void* moved_ptr = to_move.data_ptr();
    torch::Tensor moved_into = std::move(to_move);
    std::cout << "moved_into.data_ptr() == original pre-move pointer? "
              << (moved_into.data_ptr() == moved_ptr)
              << " (move transfers the same storage, no copy)" << std::endl;
    std::cout << "to_move.defined() after being moved from = " << to_move.defined()
              << " (the moved-from tensor's real, genuinely-observed post-move state)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
shallow_copy.data_ptr() == original.data_ptr()? 1 (torch::Tensor's copy constructor shares storage, unlike the CUDA book's deleted one)
after shallow_copy[0]=99: original[0] = 99 (mutation through the copy is visible in the original)
cloned.data_ptr() == original2.data_ptr()? 0 (clone() genuinely allocates new, independent storage)
after cloned[0]=99: original2[0] = 1 (mutation through the clone does NOT affect the original)
moved_into.data_ptr() == original pre-move pointer? 1 (move transfers the same storage, no copy)
to_move.defined() after being moved from = 0 (the moved-from tensor's real, genuinely-observed post-move state)
```

`shallow_copy.data_ptr() == original.data_ptr()` reporting `true` is the honest divergence stated as directly as code can state it: this line would not even compile in the CUDA book's own design, where the copy constructor is `= delete`d — here, it compiles, runs, and produces two `torch::Tensor` objects that are genuinely the same underlying storage viewed twice. `after shallow_copy[0]=99: original[0] = 99` proves it isn't just the same address by coincidence: mutating through one is visible through the other, exactly the aliasing the CUDA book's deleted copy constructor exists specifically to prevent by construction. `.clone()` closes that gap: `cloned.data_ptr() == original2.data_ptr()` reporting `false`, and `original2[0]` staying `1` after `cloned[0]` is mutated, confirm `.clone()` performs the CUDA book's own deep-copy contract exactly — genuinely independent storage, under the identical method name the CUDA book itself uses. The move result is this section's most direct new evidence: `to_move.defined()` reporting `false` after `std::move(to_move)` is not an assumption about C++'s general "moved-from objects are in a valid but unspecified state" rule — it's `torch::Tensor`'s own specific, directly observable answer to what that state actually is: an empty, undefined tensor, not a tensor with zeroed-out or dangling pointers the caller has to guess about.

> `[COMMON TRAP]` `shallow_copy.data_ptr() == original.data_ptr()` being `true` does not mean `torch::Tensor` copies are "free" in the sense that nothing happens — a real `TensorImpl` reference count is genuinely incremented, and the underlying storage's lifetime is genuinely extended by that copy. It means specifically that the *element data* isn't duplicated, which is the property this section's mutation test isolates directly.

## Complete Runnable Code

### File: `01_shape_strides.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.1/6.2 hand-build TensorShape (an
// int dims[MAX_DIMS] plus a numel() computed as a product-of-dimensions
// loop) and TensorStrides (int strides[MAX_DIMS], computed right-to-left:
// strides[d] = strides[d+1] * dims[d+1]) as two separate structs a Tensor
// later composes. torch::Tensor has never needed either hand-built --
// .sizes() and .strides() are real, already-computed member functions.
// This file verifies the exact numbers the CUDA book's own shape [2,3,4]
// example produces, from the real thing rather than a hand-rolled one.
int main() {
    torch::Tensor t = torch::zeros({2, 3, 4});

    std::cout << "t.sizes() = " << t.sizes() << std::endl;
    std::cout << "t.numel() = " << t.numel() << std::endl;
    std::cout << "t.strides() = " << t.strides() << std::endl;

    // Hand-compute numel the same way the CUDA book's TensorShape::numel()
    // does: product of dims.
    int64_t hand_numel = 1;
    for (int64_t d : t.sizes()) hand_numel *= d;
    std::cout << "hand-computed numel (product of dims) = " << hand_numel << std::endl;

    // Hand-compute strides the same way TensorStrides::row_major() does:
    // right-to-left, strides[d] = strides[d+1] * dims[d+1], innermost = 1.
    auto sizes = t.sizes();
    int ndim = sizes.size();
    std::vector<int64_t> hand_strides(ndim);
    hand_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        hand_strides[d] = hand_strides[d + 1] * sizes[d + 1];
    }
    std::cout << "hand-computed strides (row_major formula) = [";
    for (int i = 0; i < ndim; i++) std::cout << hand_strides[i] << (i + 1 < ndim ? ", " : "");
    std::cout << "]" << std::endl;

    bool numel_matches = (hand_numel == t.numel());
    bool strides_match = true;
    for (int i = 0; i < ndim; i++) if (hand_strides[i] != t.stride(i)) strides_match = false;
    std::cout << "hand-computed numel matches t.numel()? " << numel_matches << std::endl;
    std::cout << "hand-computed strides match t.strides()? " << strides_match << std::endl;

    return 0;
}
```

### File: `02_data_and_grad_buffers.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.3 composes a Tensor from a shape, a
// strides object, and TWO independently-owned buffers: `float* data` and
// `float* grad`, each allocated with its own cudaMalloc call in the
// constructor. torch::Tensor's real gradient storage works the same way --
// a tensor's .grad() is a genuinely SEPARATE tensor, with its own separate
// storage, only populated once autograd actually computes it. This file
// verifies that separation directly: .data_ptr() and .grad().data_ptr()
// are two different allocations, not two views into one buffer.
int main() {
    torch::Tensor t = torch::ones({2, 3, 4}, torch::requires_grad(true));

    std::cout << "t.grad().defined() before backward() = " << t.grad().defined() << std::endl;

    torch::Tensor loss = (t * t).sum();
    loss.backward();

    std::cout << "t.grad().defined() after backward() = " << t.grad().defined() << std::endl;
    std::cout << "t.sizes() = " << t.sizes() << ", t.grad().sizes() = " << t.grad().sizes() << std::endl;

    void* data_ptr = t.data_ptr();
    void* grad_ptr = t.grad().data_ptr();
    std::cout << "t.data_ptr() == t.grad().data_ptr()? " << (data_ptr == grad_ptr)
              << " (two genuinely separate allocations, exactly like the CUDA book's two cudaMalloc calls)" << std::endl;

    // d/dt (sum(t*t)) = 2t, so grad should be filled with 2.0 everywhere,
    // since t was initialized to all ones.
    torch::Tensor expected_grad = torch::full({2, 3, 4}, 2.0);
    bool grad_correct = torch::equal(t.grad(), expected_grad);
    std::cout << "t.grad() == 2*ones(2,3,4) (d/dt sum(t^2) = 2t)? " << grad_correct << std::endl;

    return 0;
}
```

### File: `03_indexing_offset.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 6.4 hand-writes offset(i0,i1,i2) = i0*strides[0]
// + i1*strides[1] + i2*strides[2], then hand-verifies five coordinates for
// shape [2,3,4] with strides [12,4,1]: (0,0,0)->0, (0,0,1)->1, (0,1,0)->4,
// (1,0,0)->12, (1,2,3)->23. torch::Tensor never needs this formula written
// out -- indexing with operator[] already does it -- but the formula still
// genuinely governs where each element lives, and this file verifies it
// against the CUDA book's own five coordinates by comparing real element
// addresses (data_ptr() differences) on an actual torch::Tensor.
int main() {
    torch::Tensor t = torch::arange(0, 24, torch::kFloat32).reshape({2, 3, 4});
    std::cout << "t.sizes() = " << t.sizes() << ", t.strides() = " << t.strides() << std::endl;

    float* base = t.data_ptr<float>();

    struct Coord { int i0, i1, i2; int64_t expected_offset; };
    Coord coords[] = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
        {0, 1, 0, 4},
        {1, 0, 0, 12},
        {1, 2, 3, 23},
    };

    bool all_match = true;
    for (const auto& c : coords) {
        // Hand-computed offset, the exact formula the CUDA book's offset()
        // member function implements.
        int64_t hand_offset = c.i0 * t.stride(0) + c.i1 * t.stride(1) + c.i2 * t.stride(2);

        // Real address evidence: how many elements this coordinate's real
        // address sits from the tensor's base pointer.
        float* elem_ptr = &t[c.i0][c.i1][c.i2].data_ptr<float>()[0];
        int64_t real_offset = elem_ptr - base;

        bool matches = (hand_offset == c.expected_offset) && (real_offset == c.expected_offset);
        all_match = all_match && matches;
        std::cout << "(" << c.i0 << "," << c.i1 << "," << c.i2 << "): hand-computed offset = "
                  << hand_offset << ", real address offset = " << real_offset
                  << ", CUDA book's own expected = " << c.expected_offset
                  << ", match = " << (matches ? "true" : "false") << std::endl;
    }
    std::cout << "all five coordinates match the CUDA book's own hand-verified offsets? "
              << (all_match ? "true" : "false") << std::endl;

    return 0;
}
```

### File: `04_copy_clone_move.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <utility>

// The CUDA C++ edition's Section 6.5 deletes Tensor's copy constructor
// entirely (any accidental copy is a compile error), forcing clone() as
// the only way to deep-copy two owned buffers, and hand-writes a move
// constructor that steals the pointers and nulls the source.
// torch::Tensor makes a genuinely DIFFERENT design choice, honestly
// reported here rather than assumed identical: its copy constructor is
// NOT deleted -- copying a torch::Tensor is a real, valid, everyday
// operation, and it's a SHALLOW copy (both tensors share the same
// underlying storage), matching Python's own tensor reference semantics.
// .clone() is still the real, genuinely-named method that performs the
// CUDA book's own deep-copy behavior. Move construction is real too, and
// this file tests what actually happens to the moved-from tensor.
int main() {
    torch::Tensor original = torch::tensor({1.0f, 2.0f, 3.0f});

    // torch::Tensor's copy constructor: NOT deleted, compiles fine, but
    // shares storage -- the honest divergence from the CUDA book's design.
    torch::Tensor shallow_copy = original;
    std::cout << "shallow_copy.data_ptr() == original.data_ptr()? "
              << (shallow_copy.data_ptr() == original.data_ptr())
              << " (torch::Tensor's copy constructor shares storage, unlike the CUDA book's deleted one)" << std::endl;

    // Mutating through the shallow copy is visible through the original --
    // direct proof they share one buffer, not two independent ones.
    shallow_copy[0] = 99.0f;
    std::cout << "after shallow_copy[0]=99: original[0] = " << original[0].item<float>()
              << " (mutation through the copy is visible in the original)" << std::endl;

    // .clone(): the CUDA book's own real deep-copy method, genuinely present
    // on torch::Tensor with the identical name and the identical contract.
    torch::Tensor original2 = torch::tensor({1.0f, 2.0f, 3.0f});
    torch::Tensor cloned = original2.clone();
    std::cout << "cloned.data_ptr() == original2.data_ptr()? "
              << (cloned.data_ptr() == original2.data_ptr())
              << " (clone() genuinely allocates new, independent storage)" << std::endl;
    cloned[0] = 99.0f;
    std::cout << "after cloned[0]=99: original2[0] = " << original2[0].item<float>()
              << " (mutation through the clone does NOT affect the original)" << std::endl;

    // Move construction: real on torch::Tensor. What state is the
    // moved-from tensor actually left in? Tested directly, not assumed.
    torch::Tensor to_move = torch::tensor({4.0f, 5.0f, 6.0f});
    void* moved_ptr = to_move.data_ptr();
    torch::Tensor moved_into = std::move(to_move);
    std::cout << "moved_into.data_ptr() == original pre-move pointer? "
              << (moved_into.data_ptr() == moved_ptr)
              << " (move transfers the same storage, no copy)" << std::endl;
    std::cout << "to_move.defined() after being moved from = " << to_move.defined()
              << " (the moved-from tensor's real, genuinely-observed post-move state)" << std::endl;

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

Every number the CUDA book's hand-built `Tensor` type produces for shape `[2, 3, 4]` — `numel() = 24`, `strides = [12, 4, 1]` — came out identical from `torch::Tensor`'s real implementation, cross-checked against a from-scratch reimplementation of the CUDA book's own row-major formula computed independently of `torch::Tensor`'s internals. A tensor with `requires_grad(true)` genuinely owns two separate buffers exactly as the CUDA book's two-`cudaMalloc`-call design does: `.data_ptr()` and `.grad().data_ptr()` measured as different addresses, with the gradient buffer's `.defined()` genuinely `false` until `.backward()` actually populates it. The CUDA book's own hand-written offset formula, tested against its own five example coordinates, correctly predicted every real element address in an actual tensor — cross-checked a second way through NumPy's independent pointer arithmetic. And this chapter's one honest divergence: `torch::Tensor`'s copy constructor is not deleted the way the CUDA book's is, and copying shares storage (mutation through one is visible through the other, genuinely tested) rather than deep-copying — with `.clone()`, under the identical name the CUDA book itself uses, performing the deep-copy contract the CUDA book's deleted copy constructor forces every caller toward. Moving a `torch::Tensor` was tested directly rather than assumed: the moved-from tensor's `.defined()` comes back `false`, a real, specific, observable answer rather than a generic "unspecified state."

## Self-Check Questions

1. Worked Example 6.1.1 computes `numel` and `strides` two independent ways — once by calling `t.numel()`/`t.strides()` directly, and once by a hand-written loop implementing the CUDA book's own formulas. Why does this chapter bother with the second, hand-written computation at all, given that `torch::Tensor` already provides the first?
2. Section 6.2's `[COMMON TRAP]` notes that row-major strides describe a *freshly created* tensor's layout, not every tensor. Name one tensor from an earlier chapter in this book whose strides do NOT follow the `strides[d] = strides[d+1] * sizes[d+1]` formula, and explain why not.
3. Worked Example 6.3.1 reports `t.grad().defined()` as `false` before `.backward()` and `true` after. What does this tell you about when LibTorch actually allocates a gradient buffer, and how does that compare to the CUDA book's own `Tensor` constructor, which allocates both `data` and `grad` at construction time?
4. Section 6.5 states that `torch::Tensor`'s copy constructor is "not deleted" as an honest divergence from the CUDA book's design, rather than claiming the two designs are equivalent. What specific test in Worked Example 6.5.1 would have failed, or behaved differently, if `torch::Tensor`'s copy constructor genuinely performed a deep copy the way the CUDA book's `clone()` does?
5. Worked Example 6.5.1 reports `to_move.defined() == false` after `std::move(to_move)`. Why is this a meaningful, checkable fact about `torch::Tensor` specifically, rather than something guaranteed by the C++ language's general move-semantics rules?

## Where We Go Next

This chapter opened `torch::Tensor` itself and confirmed its real numbers match the CUDA book's hand-built design, section by section. Chapter 7 turns to a question this chapter's Section 6.2 only partly answered: `torch::Tensor`'s memory layout isn't only "row-major or not" — it supports genuinely different layout strategies (including the channels-last layout real convolutional networks use), and Chapter 7 tests what `torch::Tensor` actually does, and doesn't, let a caller control about how its data is arranged.

## Worked Solutions

**1.** The hand-written computation exists to make the chapter's central claim checkable rather than assumed: without it, this chapter would only be reporting what `torch::Tensor` says about itself, which proves nothing about whether that number matches the CUDA book's own formula. Computing `numel` and `strides` a second, independent way — using the CUDA book's own product-of-dimensions and right-to-left stride formulas, implemented from scratch in this file — and then comparing the two results is what turns "these should be the same" into a genuinely tested "these are measured to be the same."

**2.** Chapter 4's Worked Example 4.4.1 constructed `col_view = m.select(1, 0)`, a column view into a `[4000, 4000]` matrix, with `strides() = [4000]` — for a 1-D tensor of 4000 elements, the row-major formula would predict stride `1` (the innermost, and only, dimension should be contiguous), but `col_view`'s actual stride is `4000`. This is because `col_view` isn't a freshly allocated tensor with its own row-major layout at all — it's a *view* into `m`'s existing storage, inheriting `m`'s row stride (4000) rather than getting a fresh, contiguous layout computed for its own shape.

**3.** It tells you LibTorch allocates a tensor's gradient buffer lazily, only once something (here, `.backward()`) actually computes a gradient value to store there — not eagerly, at the moment the tensor itself is constructed with `requires_grad(true)`. This is a genuine difference from the CUDA book's own `Tensor` constructor, which calls `cudaMalloc` for both `data` and `grad` together, in one constructor call, meaning the CUDA book's design pays for both allocations up front regardless of whether a gradient will ever actually be computed for that particular tensor.

**4.** The line `shallow_copy[0] = 99.0f;` followed by checking `original[0]` would have behaved differently: if `torch::Tensor`'s copy constructor performed a genuine deep copy, `original[0]` would still read `1.0f` after that mutation, the same result Worked Example 6.5.1 actually observes for `original2[0]` after mutating through `cloned` (which genuinely is a deep copy). Instead, the file observes `original[0] = 99`, meaning the mutation through `shallow_copy` was visible in `original` — proof the two share one buffer, which a genuinely deep-copying constructor would never allow.

**5.** C++'s language rules only guarantee that a moved-from object of standard library types (and most well-behaved user types) is left in "a valid but unspecified state" — valid enough to destroy or reassign, but the language itself makes no promise about what specific state that is, and different types are free to leave different things behind. `torch::Tensor`'s actual answer — an empty tensor with `.defined() == false`, rather than, say, a tensor of the same shape filled with garbage values, or one that still happens to hold a dangling pointer — is a fact about `torch::Tensor`'s own specific implementation, only knowable by genuinely testing it against that implementation rather than inferred from the general C++ rule alone.
