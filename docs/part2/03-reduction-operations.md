# Chapter 14: Reduction Operations

> "The CUDA C++ edition's Chapter 14 builds reductions -- operations that collapse an array down to fewer values -- around a tree-reduction pattern that halves the array each round via pairwise combination, a discipline the book's own text ties directly to floating-point addition's non-associativity. Its own Worked Example 14.1.3 is a stark bug finding: a kernel launch left as a comment produces a `tensor_sum_incomplete` that returns three DIFFERENT wrong answers across three real runs, never the correct one, because nothing ever actually ran and the result is whatever garbage sat in freshly allocated memory. `torch::sum`, `torch::max`, `torch::norm`, and `torch::var` are real, already-implemented reductions -- this chapter verifies the CUDA book's own exact worked numbers on all four, directly probes the CUDA book's own bug's failure mode by confirming genuine determinism across repeated runs, and finds this chapter's own most consequential honest divergence: `torch::var`'s own default does not match the CUDA book's own variance formula at all, for a real and well-motivated statistical reason."

**What you will understand by the end of this chapter:**

- That `torch::sum` and `torch::mean` reproduce the CUDA book's own `30` and `7.5` (and `24` for a second array) exactly, and that repeating the identical computation genuinely produces the same correct answer every time -- the direct opposite of the CUDA book's own `tensor_sum_incomplete`, whose missing kernel launch produced three different wrong answers (`-0.000000`, `0.000000`, `nan`) across three real runs
- That `torch::max` reproduces the CUDA book's own argmax results exactly, including a hand-traced tie-break case matching the CUDA book's own `left >= right` tree-reduction convention, and that it remains correct on an array far larger than any single reduction round the CUDA book's own examples use
- That `torch::norm` and the real, production `torch::nn::utils::clip_grad_norm_` reproduce the CUDA book's own gradient-clipping numbers (`scale=0.4`, result `[1.2,1.6]`, clipped norm exactly `2.0`) exactly, using LibTorch's own genuine training-loop utility rather than a hand-built stand-in
- That `torch::var`'s own DEFAULT behavior does NOT reproduce the CUDA book's own variance number at all -- it applies Bessel's correction, an unbiased sample-variance convention the CUDA book's own population-variance formula never uses -- and that passing `correction=0` recovers the CUDA book's own exact `4.0` precisely

**What you need to know first:**

- Chapter 12's element-wise operations -- this chapter's own two-pass variance computation in Section 14.4 builds directly on element-wise subtraction and squaring
- Chapter 13's `torch::matmul` and real shape-checking behavior -- the same "real function throws or behaves differently than a hand-built version" pattern recurs in this chapter's own Section 14.4 finding
- If you've read the CUDA C++ edition's Chapter 14: its own `sum_reduce_kernel` and `max_reduce_kernel` both implement the identical tree-reduction pattern, differing in one important way its own `[COMMON TRAP]` calls out directly -- `sum_reduce_kernel` explicitly zero-fills any thread with no work, while `max_reduce_kernel` has no equivalent else branch, silently leaving some output positions unwritten for an over-launched grid. `torch::sum` and `torch::max` are real, production reductions with neither gap; this chapter verifies the CUDA book's own exact worked numbers on the real implementations, then investigates each place their behavior actually differs from the CUDA book's own hand-built version.

## 14.1 Sum and Mean: Genuine Determinism Where the CUDA Book Has a Genuine Bug `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `sum_reduce_kernel` performs a tree reduction: each round, pairs of consecutive elements are combined via addition, halving the array's length, with any thread that has no real pair explicitly zero-filled so an over-launched grid still produces a correct result. `torch::sum` and `torch::mean` compute the identical mathematical quantities as real, already-implemented, production reductions.

### Background

The CUDA book's own Worked Example 14.1.1: `[1,4,9,16]` reduces to `30` over two rounds (`1+4=5, 9+16=25`, then `5+25=30`); mean `=7.5`. Its own Worked Example 14.1.2: `[2,6,3,8,5]` (an odd-length array) reduces to `24` across three rounds. Its own Worked Example 14.1.3 is a critical bug finding: `tensor_sum_incomplete` has its own kernel launch left as a comment -- the reduction never actually runs -- and three genuine runs returned `-0.000000`, `0.000000`, and `nan`, never the correct `30`, because the result is whatever uninitialized garbage happened to occupy freshly `malloc`'d memory.

### Worked Example 14.1.1 -- the CUDA book's own numbers, and a direct probe of its own bug's failure mode

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 14.1 hand-writes sum_reduce_kernel: a tree
// reduction that pairs consecutive elements and combines them via addition
// each round, explicitly zero-filling any thread with no work so an
// over-launched grid still produces a correct, deterministic result. Its
// own Worked Example 14.1.3 is a critical bug finding: tensor_sum_incomplete
// has its kernel LAUNCH left as a comment -- the reduction never actually
// runs -- so the "result" is just whatever garbage happened to be sitting
// in freshly malloc'd, never-initialized memory, genuinely observed across
// three real runs as -0.000000, 0.000000, and nan, never the correct 30.
// torch::sum and torch::mean are real, already-implemented, and this file
// verifies the CUDA book's own exact worked numbers, then directly probes
// the CUDA book's own bug's failure mode: repeating the identical
// computation many times and confirming it is genuinely deterministic and
// correct every single time, the opposite of what tensor_sum_incomplete
// produces.
int main() {
    // Worked Example 14.1.1: [1,4,9,16] reduces to 30 over two rounds
    // ((1+4=5, 9+16=25), then (5+25=30)); mean = 7.5.
    torch::Tensor a = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
    torch::Tensor sum_a = a.sum();
    torch::Tensor mean_a = a.mean();
    std::cout << "sum([1,4,9,16]) = " << sum_a.item<float>()
              << ", CUDA book's own expected = 30, match = " << (sum_a.item<float>() == 30.0f) << std::endl;
    std::cout << "mean([1,4,9,16]) = " << mean_a.item<float>()
              << ", CUDA book's own expected = 7.5, match = " << (mean_a.item<float>() == 7.5f) << std::endl;

    // Worked Example 14.1.2: [2,6,3,8,5] (odd length) reduces to 24.
    torch::Tensor b = torch::tensor({2.0f, 6.0f, 3.0f, 8.0f, 5.0f});
    torch::Tensor sum_b = b.sum();
    std::cout << "\nsum([2,6,3,8,5]) = " << sum_b.item<float>()
              << ", CUDA book's own expected = 24, match = " << (sum_b.item<float>() == 24.0f) << std::endl;

    // Direct probe of the CUDA book's own bug's failure mode: its own
    // tensor_sum_incomplete produced THREE DIFFERENT wrong answers across
    // three real runs (-0.000000, 0.000000, nan), because it read
    // uninitialized memory that happened to differ run to run.
    // torch::sum has no missing-launch equivalent -- it is a single real
    // function call that always actually runs the reduction. Repeating the
    // identical computation many times here confirms it is genuinely
    // deterministic, unlike the CUDA book's own broken version.
    bool all_identical_and_correct = true;
    for (int i = 0; i < 5; i++) {
        torch::Tensor a_fresh = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
        float result = a_fresh.sum().item<float>();
        if (result != 30.0f) all_identical_and_correct = false;
    }
    std::cout << "\nsum([1,4,9,16]) repeated 5 fresh times: every result exactly 30.0 "
              << "(no run producing -0.0, 0.0, or nan, unlike the CUDA book's own "
              << "tensor_sum_incomplete)? " << all_identical_and_correct << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_sum_mean.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_sum_mean
./01_sum_mean
```

```text
sum([1,4,9,16]) = 30, CUDA book's own expected = 30, match = 1
mean([1,4,9,16]) = 7.5, CUDA book's own expected = 7.5, match = 1

sum([2,6,3,8,5]) = 24, CUDA book's own expected = 24, match = 1

sum([1,4,9,16]) repeated 5 fresh times: every result exactly 30.0 (no run producing -0.0, 0.0, or nan, unlike the CUDA book's own tensor_sum_incomplete)? 1
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all:

```text
numpy sum: 30.0 mean: 7.5
numpy sum b: 24.0
```

### Discussion

`sum([1,4,9,16])` and `sum([2,6,3,8,5])` match the CUDA book's own numbers exactly, and `mean([1,4,9,16])` confirms the CUDA book's own `7.5`. The five-repetition test is this section's own strongest evidence, and it targets the CUDA book's own bug's failure mode directly rather than merely computing a correct answer once: `tensor_sum_incomplete`'s own bug is specifically about non-determinism arising from reading uninitialized memory -- the CUDA book's own three real runs each produced a DIFFERENT wrong value, because whatever bytes happened to occupy that memory differed run to run. `torch::sum` genuinely runs its own real reduction on every call, so five fresh, independent tensors summing to exactly `30.0` five times out of five is a direct, empirical confirmation that no equivalent gap exists -- there is no uninitialized buffer for `torch::sum` to accidentally return the contents of, because it is a single function call that performs the actual computation every time it is invoked.

> `[COMMON TRAP]` A reader might assume `tensor_sum_incomplete`'s bug is "obviously" going to produce zero, or some other predictable placeholder value, since the buffer was never written. The CUDA book's own three genuine results -- `-0.000000`, `0.000000`, and `nan` -- show this assumption is wrong: uninitialized memory contains whatever bit pattern happened to be left over from previous, unrelated use of that memory, which can be interpreted as a negative zero, a positive zero, or a bit pattern that is not a valid floating-point number at all (`nan`). There is no way to predict which of these (or countless other possible values) a given run will produce, which is exactly why this section's own test repeats the CORRECT computation five times to confirm determinism, rather than trying to predict what an incorrect one would output.

## 14.2 Min and Max: A Real Argmax, Tested Against a Hand-Traced Tie-Break `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `max_reduce_kernel` performs a tree reduction that tracks both the running maximum value and a parallel index buffer recording which original position holds it, using a `left >= right` comparison (not strict `>`) to decide ties. `torch::max(x, dim)` computes the identical maximum-and-its-index pair as a real, already-implemented, production reduction, returned as a single value/index pair from one function call.

### Background

The CUDA book's own Worked Example 14.2.1: `[3,7,2,9]` with indices `[0,1,2,3]` gives argmax `9.0` at index `3`. Its own Worked Example 14.2.2: `[5,2,9,1,7]` gives argmax `9.0` at index `2`. Its own `[COMMON TRAP]` notes `max_reduce_kernel` lacks `sum_reduce_kernel`'s own explicit zero-fill for over-launched threads -- an output position an over-launched thread never writes is silently left unset.

### Worked Example 14.2.1 -- the CUDA book's own numbers, a hand-traced tie-break, and an over-launch-scale probe

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 14.2 hand-writes max_reduce_kernel: a tree
// reduction that pairs elements and compares them, carrying a parallel
// index buffer so the position of the maximum survives the reduction
// alongside its value -- using a left >= right comparison (not strict >)
// to break ties. Its own [COMMON TRAP] notes max_reduce_kernel lacks
// sum_reduce_kernel's explicit zero-fill for over-launched threads: an
// output position an over-launched thread never writes is silently left
// unset. torch::max(x, dim) is real, already-implemented, and returns
// both the maximum value and its index in one call. This file verifies
// the CUDA book's own exact worked numbers, extends to a genuine tie case
// to compare tie-breaking behavior directly, then directly probes the
// over-launch failure mode with an array far larger than any single
// reduction round the CUDA book's own examples use.
int main() {
    // Worked Example 14.2.1: [3,7,2,9] with indices [0,1,2,3]: argmax
    // returns 9.0 at index 3.
    torch::Tensor a = torch::tensor({3.0f, 7.0f, 2.0f, 9.0f});
    auto [max_val_a, max_idx_a] = torch::max(a, 0);
    std::cout << "max([3,7,2,9]) = " << max_val_a.item<float>() << " at index " << max_idx_a.item<int64_t>()
              << ", CUDA book's own expected = 9.0 at index 3, match = "
              << (max_val_a.item<float>() == 9.0f && max_idx_a.item<int64_t>() == 3) << std::endl;

    // Worked Example 14.2.2: [5,2,9,1,7]: argmax returns 9.0 at index 2.
    torch::Tensor b = torch::tensor({5.0f, 2.0f, 9.0f, 1.0f, 7.0f});
    auto [max_val_b, max_idx_b] = torch::max(b, 0);
    std::cout << "\nmax([5,2,9,1,7]) = " << max_val_b.item<float>() << " at index " << max_idx_b.item<int64_t>()
              << ", CUDA book's own expected = 9.0 at index 2, match = "
              << (max_val_b.item<float>() == 9.0f && max_idx_b.item<int64_t>() == 2) << std::endl;

    // Extension: a genuine tie case, [4,12,7,12,3] (value 12 appears at
    // both index 1 and index 3). Hand-tracing the CUDA book's own
    // left >= right tie-break through its own tree structure -- round 1
    // pairs (4,12)->12@1 and (7,12)->12@3, leftover 3@4 passes through;
    // round 2 pairs (12@1, 12@3), a genuine tie, and left >= right keeps
    // the LEFT operand, so 12@1 survives, leftover 3@4 passes through;
    // round 3 pairs (12@1, 3@4), 12 wins -- final answer index 1. This is
    // an extension beyond the CUDA book's own two worked examples, neither
    // of which contains a tie.
    torch::Tensor c = torch::tensor({4.0f, 12.0f, 7.0f, 12.0f, 3.0f});
    auto [max_val_c, max_idx_c] = torch::max(c, 0);
    std::cout << "\nmax([4,12,7,12,3]) (a genuine tie between index 1 and index 3) = "
              << max_val_c.item<float>() << " at index " << max_idx_c.item<int64_t>()
              << ", hand-traced CUDA book's own left>=right tree tie-break expected = 12.0 at index 1, "
              << "match = " << (max_val_c.item<float>() == 12.0f && max_idx_c.item<int64_t>() == 1) << std::endl;

    // Direct probe of the CUDA book's own missing-else-branch bug's
    // failure mode: an array far larger than any single reduction round
    // the CUDA book's own worked examples use, confirming torch::max
    // correctly finds the true maximum and its true index at every scale,
    // with no unwritten output position possible.
    int64_t n = 100000;
    torch::Tensor big = torch::arange(0, n, torch::kFloat32);
    big[73421] = 999999.0f;  // plant a known maximum at a specific position
    auto [big_max_val, big_max_idx] = torch::max(big, 0);
    std::cout << "\nmax over n=" << n << " elements (far beyond any single CUDA reduction round), "
              << "with a planted maximum at index 73421: found value = " << big_max_val.item<float>()
              << " at index " << big_max_idx.item<int64_t>()
              << ", correct? " << (big_max_val.item<float>() == 999999.0f && big_max_idx.item<int64_t>() == 73421)
              << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_max_argmax.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_max_argmax
./02_max_argmax
```

```text
max([3,7,2,9]) = 9 at index 3, CUDA book's own expected = 9.0 at index 3, match = 1

max([5,2,9,1,7]) = 9 at index 2, CUDA book's own expected = 9.0 at index 2, match = 1

max([4,12,7,12,3]) (a genuine tie between index 1 and index 3) = 12 at index 1, hand-traced CUDA book's own left>=right tree tie-break expected = 12.0 at index 1, match = 1

max over n=100000 elements (far beyond any single CUDA reduction round), with a planted maximum at index 73421: found value = 999999 at index 73421, correct? 1
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all:

```text
numpy argmax [3,7,2,9]: 3 value 9.0
numpy argmax [5,2,9,1,7]: 2 value 9.0
```

### Discussion

Both of the CUDA book's own non-tied worked examples match exactly. The tie case is this section's own strongest evidence: hand-tracing the CUDA book's own `left >= right` tie-break rule through its own three-round tree structure for `[4,12,7,12,3]` predicts index `1` survives (round 1 produces two `12`s at indices `1` and `3`; round 2's tie between them keeps the LEFT operand, index `1`; round 3 confirms `12` beats the leftover `3`) -- and `torch::max` genuinely returns index `1` for this exact array, confirming the two tie-breaking conventions agree in this specific case. The over-launch-scale probe directly targets the CUDA book's own `[COMMON TRAP]`: a hand-built kernel with no else branch could leave some output position unwritten for a grid sized beyond the input, but `torch::max` is a single real function correctly handling the requested reduction regardless of size, confirmed here by planting a known maximum deep inside a 100,000-element array and finding it exactly.

> `[COMMON TRAP]` The tie-case match in this section might read as proof that `torch::max`'s own tie-breaking rule is identical to the CUDA book's own `left >= right` convention in general. This is not established here -- only that the two conventions AGREE for this one specific array's specific pairing structure. `torch::max`'s own documented behavior is to return the index of the FIRST occurrence of the maximum value in the tensor's own reading order, a simpler and more general rule than the CUDA book's own tree-structure-dependent tie-break (whose winner can depend on exactly how elements happen to get paired across rounds, not just their positions). The two rules can, and in general will, disagree on other tied arrays with a different structure -- this section's own test confirms agreement on one concrete case, not equivalence of the underlying rules.

## 14.3 Norm and Gradient Clipping: A Real Training-Loop Utility, Not a Stand-In `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `l2_norm` computes the square root of a sum of squares -- itself built on the reduction machinery from Section 14.1 -- and its own gradient clipping rescales a vector by `max_norm / actual_norm` whenever the actual norm exceeds `max_norm`, so the rescaled result's own norm becomes exactly `max_norm`. `torch::norm` computes the identical L2 norm, and `torch::nn::utils::clip_grad_norm_` is not a hand-built stand-in but the real, production gradient-clipping utility genuine neural network training code uses to prevent exploding gradients.

### Background

The CUDA book's own Worked Example 14.3.1: `l2_norm([3,4]) = 5.0` (a 3-4-5 right triangle). Its own Worked Example 14.3.2: gradient `[3,4]` clipped with `max_norm=2.0` gives scale factor `0.4`, result `[1.2,1.6]`, with the clipped result's own norm verified to be exactly `2.0`.

### Worked Example 14.3.1 -- the CUDA book's own numbers, via both a manual rescale and the real production utility

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 14.3 hand-writes l2_norm (sqrt of a sum of
// squares, itself built on the reduction machinery from 14.1) and gradient
// clipping: rescale a vector by max_norm/actual_norm whenever its norm
// exceeds max_norm, so the result's own norm becomes exactly max_norm.
// torch::norm computes the identical L2 norm as a real, already-implemented
// function, and torch::nn::utils::clip_grad_norm_ is a real, production
// gradient-clipping utility used throughout actual neural network training
// code -- not a hand-built stand-in but the genuine mechanism deep learning
// frameworks use to prevent exploding gradients. This file verifies the
// CUDA book's own exact numbers via both a manual rescale (mirroring the
// CUDA book's own hand-built version) and the real clip_grad_norm_ utility,
// confirming both agree.
int main() {
    // Worked Example 14.3.1: l2_norm([3,4]) = 5.0 (a 3-4-5 right triangle).
    torch::Tensor v = torch::tensor({3.0f, 4.0f});
    torch::Tensor norm_v = torch::norm(v);
    std::cout << "norm([3,4]) = " << norm_v.item<float>()
              << ", CUDA book's own expected = 5.0, match = " << (norm_v.item<float>() == 5.0f) << std::endl;

    // Worked Example 14.3.2: gradient [3,4] clipped with max_norm=2.0:
    // scale factor 0.4, result [1.2, 1.6], verified norm = 2.0 exactly.
    // First, the manual rescale, mirroring the CUDA book's own hand-built
    // clip_gradient function directly.
    float max_norm = 2.0f;
    float actual_norm = norm_v.item<float>();
    float scale = max_norm / actual_norm;
    torch::Tensor clipped_manual = v * scale;
    std::cout << "\nmanual clip: scale = " << scale << ", CUDA book's own expected = 0.4, match = "
              << (scale == 0.4f) << std::endl;
    std::cout << "clipped [3,4] (manual) = " << clipped_manual << std::endl;
    torch::Tensor expected_clip = torch::tensor({1.2f, 1.6f});
    std::cout << "matches CUDA book's own [1.2, 1.6] (allclose)? "
              << torch::allclose(clipped_manual, expected_clip, 1e-5) << std::endl;
    std::cout << "norm of clipped result = " << torch::norm(clipped_manual).item<float>()
              << ", CUDA book's own expected = exactly 2.0, match = "
              << (std::abs(torch::norm(clipped_manual).item<float>() - 2.0f) < 1e-5) << std::endl;

    // Now the real, production gradient-clipping utility:
    // torch::nn::utils::clip_grad_norm_ -- not hand-built, the identical
    // mechanism real training loops use to prevent exploding gradients.
    // It operates in-place on a parameter's own .grad() field, mirroring
    // how it is genuinely used during backpropagation.
    torch::Tensor param = torch::zeros({2}, torch::requires_grad(true));
    param.mutable_grad() = torch::tensor({3.0f, 4.0f});
    double total_norm_before = torch::nn::utils::clip_grad_norm_(param, max_norm);
    std::cout << "\ntorch::nn::utils::clip_grad_norm_: reported pre-clip norm = " << total_norm_before
              << ", CUDA book's own expected = 5.0, match = " << (std::abs(total_norm_before - 5.0) < 1e-4) << std::endl;
    std::cout << "param.grad() after clip_grad_norm_ = " << param.grad() << std::endl;
    std::cout << "matches the manual rescale's own [1.2, 1.6] (allclose)? "
              << torch::allclose(param.grad(), expected_clip, 1e-4) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_norm_clipping.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_norm_clipping
./03_norm_clipping
```

```text
norm([3,4]) = 5, CUDA book's own expected = 5.0, match = 1

manual clip: scale = 0.4, CUDA book's own expected = 0.4, match = 1
clipped [3,4] (manual) =  1.2000
 1.6000
[ CPUFloatType{2} ]
matches CUDA book's own [1.2, 1.6] (allclose)? 1
norm of clipped result = 2, CUDA book's own expected = exactly 2.0, match = 1

torch::nn::utils::clip_grad_norm_: reported pre-clip norm = 5, CUDA book's own expected = 5.0, match = 1
param.grad() after clip_grad_norm_ =  1.2000
 1.6000
[ CPUFloatType{2} ]
matches the manual rescale's own [1.2, 1.6] (allclose)? 1
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all:

```text
numpy norm [3,4]: 5.0
numpy clipped: [1.2 1.6]
```

### Discussion

`norm([3,4])` matches the CUDA book's own `5.0` exactly, and both verification paths -- the manual rescale mirroring the CUDA book's own hand-built formula, and `torch::nn::utils::clip_grad_norm_`, a real function this book has not built from scratch anywhere -- agree on the identical `[1.2, 1.6]` result the CUDA book's own Worked Example 14.3.2 reports. The second path is this section's own most significant confirmation: `clip_grad_norm_` is genuinely how real PyTorch and LibTorch training code prevents exploding gradients in practice, called here on an actual `torch::Tensor` with `requires_grad(true)` and a real `.grad()` field set directly, exactly the way it is genuinely invoked inside a real training loop's backward pass. Its own reported pre-clip norm (`5.0`) and its own resulting clipped gradient both match the CUDA book's own numbers, and also match the manual rescale computed independently -- three separate confirmations of the identical result.

> `[COMMON TRAP]` A reader might assume `clip_grad_norm_`'s only job is the rescaling math this section's own manual version also performs, making it a convenience wrapper with no real advantage. In genuine multi-parameter training code, `clip_grad_norm_` accepts a whole collection of parameters at once and computes ONE combined norm across every parameter's gradient jointly (as if all of them were concatenated into a single vector), then rescales each parameter's own gradient by the SAME single scale factor -- a meaningfully different computation from calling the manual single-vector version separately on each parameter, which would compute a different, per-parameter norm and scale factor for each one. This chapter's own test uses a single parameter specifically so the two approaches agree exactly; the real function's genuine value shows up once a model has many parameters, which is the actual, ordinary case it exists to handle.

## 14.4 Statistical Functions: An Honest Divergence in `torch::var`'s Own Default `[FOUNDATIONAL]`

### Intuition

The CUDA book's own two-pass variance computation -- mean first, then the mean of squared deviations from that mean, for numerical stability -- computes POPULATION variance, dividing by `n`. `torch::var` computes variance as a real, already-implemented function, but its own default behavior applies Bessel's correction, the standard statistical convention for an unbiased SAMPLE variance estimator, dividing by `n-1` instead.

### Background

The CUDA book's own Worked Example 14.4.1: `[2,4,4,4,5,5,7,9]` gives mean `5.0`, variance `4.0`, standard deviation `2.0`, using the CUDA book's own population-variance formula (divide by `n=8`).

### Worked Example 14.4.1 -- the CUDA book's own numbers, and torch::var's own default NOT reproducing them

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 14.4 hand-writes a two-pass variance
// computation (mean first, then mean of squared deviations from that
// mean) for numerical stability, using POPULATION variance -- dividing by
// n, not n-1. torch::var is real and already-implemented, but its own
// DEFAULT behavior is an honest divergence from the CUDA book's own
// formula: by default, torch::var applies Bessel's correction, dividing
// by (n-1) rather than n, the standard convention for an unbiased SAMPLE
// variance estimator in statistics. This file verifies the CUDA book's
// own exact numbers, first showing torch::var's own DEFAULT does NOT
// reproduce them, then showing that passing correction=0 (population
// variance, matching the CUDA book's own formula exactly) does.
int main() {
    // Worked Example 14.4.1: [2,4,4,4,5,5,7,9]: mean=5.0, variance=4.0,
    // standard deviation=2.0 (population variance, divide by n=8).
    torch::Tensor x = torch::tensor({2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f});
    torch::Tensor mean_x = x.mean();
    std::cout << "mean([2,4,4,4,5,5,7,9]) = " << mean_x.item<float>()
              << ", CUDA book's own expected = 5.0, match = " << (mean_x.item<float>() == 5.0f) << std::endl;

    // Honest divergence: torch::var's own DEFAULT applies Bessel's
    // correction (divides by n-1=7), the standard unbiased SAMPLE variance
    // estimator -- this does NOT match the CUDA book's own population
    // variance formula (divide by n=8), confirmed here directly rather
    // than assumed.
    torch::Tensor var_default = x.var();
    std::cout << "\ntorch::var(x) DEFAULT (Bessel's correction, divides by n-1=7) = " << var_default.item<float>()
              << ", CUDA book's own expected population variance = 4.0, match = "
              << (std::abs(var_default.item<float>() - 4.0f) < 1e-5) << " (expected to be FALSE -- "
              << "this is the honest divergence)" << std::endl;

    // Passing correction=0 reproduces the CUDA book's own exact population
    // variance formula (divide by n), matching its own worked number
    // precisely.
    torch::Tensor var_population = x.var(/*dim=*/torch::nullopt, /*correction=*/0);
    std::cout << "torch::var(x, correction=0) (population variance, divides by n=8) = "
              << var_population.item<float>() << ", CUDA book's own expected = 4.0, match = "
              << (var_population.item<float>() == 4.0f) << std::endl;

    torch::Tensor std_population = x.std(/*dim=*/torch::nullopt, /*correction=*/0);
    std::cout << "torch::std(x, correction=0) = " << std_population.item<float>()
              << ", CUDA book's own expected = 2.0, match = " << (std_population.item<float>() == 2.0f) << std::endl;

    // Cross-check via the CUDA book's own two-pass method, computed here
    // independently of torch::var entirely: mean first, then mean of
    // squared deviations from that mean, using a hand-written loop.
    float mean_val = mean_x.item<float>();
    float sum_sq_dev = 0.0f;
    auto accessor = x.accessor<float, 1>();
    for (int64_t i = 0; i < x.size(0); i++) {
        float dev = accessor[i] - mean_val;
        sum_sq_dev += dev * dev;
    }
    float two_pass_variance = sum_sq_dev / (float)x.size(0);
    std::cout << "\ntwo-pass hand-written population variance (CUDA book's own method, computed "
              << "independently of torch::var) = " << two_pass_variance
              << ", matches torch::var(correction=0)'s own 4.0? " << (two_pass_variance == 4.0f) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_statistics.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_statistics
./04_statistics
```

```text
mean([2,4,4,4,5,5,7,9]) = 5, CUDA book's own expected = 5.0, match = 1

torch::var(x) DEFAULT (Bessel's correction, divides by n-1=7) = 4.57143, CUDA book's own expected population variance = 4.0, match = 0 (expected to be FALSE -- this is the honest divergence)
torch::var(x, correction=0) (population variance, divides by n=8) = 4, CUDA book's own expected = 4.0, match = 1
torch::std(x, correction=0) = 2, CUDA book's own expected = 2.0, match = 1

two-pass hand-written population variance (CUDA book's own method, computed independently of torch::var) = 4, matches torch::var(correction=0)'s own 4.0? 1
```

Independently cross-checked via NumPy, whose own `var`/`std` functions carry the identical `ddof` (delta degrees of freedom) distinction, computed with no dependence on `torch::Tensor` at all:

```text
numpy mean 5.0 population var (ddof=0) 4.0 sample var (ddof=1) 4.571428571428571 population std 2.0
```

### Discussion

This section's own central finding is the deliberate mismatch: `torch::var(x)`'s own default returns `4.57143`, genuinely NOT matching the CUDA book's own `4.0` -- confirmed here as an explicit, expected `FALSE` rather than silently glossed over. The reason is a real, well-established statistical convention, not a bug in either book: dividing a sum of squared deviations by `n-1` rather than `n` (Bessel's correction) produces an unbiased ESTIMATOR of a population's true variance when working from a finite SAMPLE, the standard default in essentially every general-purpose statistics library, `torch::var` included. The CUDA book's own formula computes population variance directly (the sum of squared deviations divided by `n`), appropriate when the array in hand already IS the entire population of interest rather than a sample drawn from a larger one. `torch::var(x, correction=0)` recovers the CUDA book's own exact number, and NumPy's own independent `ddof=0`/`ddof=1` distinction reproduces the identical two values (`4.0` and `4.571428...`), confirming this is a genuine, well-known statistical distinction shared across tools rather than an implementation quirk specific to `torch::Tensor`.

> `[COMMON TRAP]` A reader porting code from the CUDA book's own hand-written variance function to `torch::var` without reading its own documentation could silently get a different number than intended -- `torch::var(x)`'s own default is NOT a drop-in replacement for a population-variance formula, even though both are called "variance" and operate on the identical input. This is exactly the kind of divergence that is easy to miss precisely because both functions return a plausible-looking numeric answer; nothing crashes, and `4.57143` is not an obviously wrong-looking number the way `nan` would be. The fix is one explicit keyword argument (`correction=0`), but only once a reader knows to look for the distinction -- which is exactly why this section states the mismatch explicitly (`match = 0 (expected to be FALSE)`) rather than only showing the corrected version that happens to agree.

## Complete Runnable Code

### File: `01_sum_mean.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 14.1 hand-writes sum_reduce_kernel: a tree
// reduction that pairs consecutive elements and combines them via addition
// each round, explicitly zero-filling any thread with no work so an
// over-launched grid still produces a correct, deterministic result. Its
// own Worked Example 14.1.3 is a critical bug finding: tensor_sum_incomplete
// has its kernel LAUNCH left as a comment -- the reduction never actually
// runs -- so the "result" is just whatever garbage happened to be sitting
// in freshly malloc'd, never-initialized memory, genuinely observed across
// three real runs as -0.000000, 0.000000, and nan, never the correct 30.
// torch::sum and torch::mean are real, already-implemented, and this file
// verifies the CUDA book's own exact worked numbers, then directly probes
// the CUDA book's own bug's failure mode: repeating the identical
// computation many times and confirming it is genuinely deterministic and
// correct every single time, the opposite of what tensor_sum_incomplete
// produces.
int main() {
    // Worked Example 14.1.1: [1,4,9,16] reduces to 30 over two rounds
    // ((1+4=5, 9+16=25), then (5+25=30)); mean = 7.5.
    torch::Tensor a = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
    torch::Tensor sum_a = a.sum();
    torch::Tensor mean_a = a.mean();
    std::cout << "sum([1,4,9,16]) = " << sum_a.item<float>()
              << ", CUDA book's own expected = 30, match = " << (sum_a.item<float>() == 30.0f) << std::endl;
    std::cout << "mean([1,4,9,16]) = " << mean_a.item<float>()
              << ", CUDA book's own expected = 7.5, match = " << (mean_a.item<float>() == 7.5f) << std::endl;

    // Worked Example 14.1.2: [2,6,3,8,5] (odd length) reduces to 24.
    torch::Tensor b = torch::tensor({2.0f, 6.0f, 3.0f, 8.0f, 5.0f});
    torch::Tensor sum_b = b.sum();
    std::cout << "\nsum([2,6,3,8,5]) = " << sum_b.item<float>()
              << ", CUDA book's own expected = 24, match = " << (sum_b.item<float>() == 24.0f) << std::endl;

    // Direct probe of the CUDA book's own bug's failure mode: its own
    // tensor_sum_incomplete produced THREE DIFFERENT wrong answers across
    // three real runs (-0.000000, 0.000000, nan), because it read
    // uninitialized memory that happened to differ run to run.
    // torch::sum has no missing-launch equivalent -- it is a single real
    // function call that always actually runs the reduction. Repeating the
    // identical computation many times here confirms it is genuinely
    // deterministic, unlike the CUDA book's own broken version.
    bool all_identical_and_correct = true;
    for (int i = 0; i < 5; i++) {
        torch::Tensor a_fresh = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
        float result = a_fresh.sum().item<float>();
        if (result != 30.0f) all_identical_and_correct = false;
    }
    std::cout << "\nsum([1,4,9,16]) repeated 5 fresh times: every result exactly 30.0 "
              << "(no run producing -0.0, 0.0, or nan, unlike the CUDA book's own "
              << "tensor_sum_incomplete)? " << all_identical_and_correct << std::endl;

    return 0;
}
```

### File: `02_max_argmax.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 14.2 hand-writes max_reduce_kernel: a tree
// reduction that pairs elements and compares them, carrying a parallel
// index buffer so the position of the maximum survives the reduction
// alongside its value -- using a left >= right comparison (not strict >)
// to break ties. Its own [COMMON TRAP] notes max_reduce_kernel lacks
// sum_reduce_kernel's explicit zero-fill for over-launched threads: an
// output position an over-launched thread never writes is silently left
// unset. torch::max(x, dim) is real, already-implemented, and returns
// both the maximum value and its index in one call. This file verifies
// the CUDA book's own exact worked numbers, extends to a genuine tie case
// to compare tie-breaking behavior directly, then directly probes the
// over-launch failure mode with an array far larger than any single
// reduction round the CUDA book's own examples use.
int main() {
    // Worked Example 14.2.1: [3,7,2,9] with indices [0,1,2,3]: argmax
    // returns 9.0 at index 3.
    torch::Tensor a = torch::tensor({3.0f, 7.0f, 2.0f, 9.0f});
    auto [max_val_a, max_idx_a] = torch::max(a, 0);
    std::cout << "max([3,7,2,9]) = " << max_val_a.item<float>() << " at index " << max_idx_a.item<int64_t>()
              << ", CUDA book's own expected = 9.0 at index 3, match = "
              << (max_val_a.item<float>() == 9.0f && max_idx_a.item<int64_t>() == 3) << std::endl;

    // Worked Example 14.2.2: [5,2,9,1,7]: argmax returns 9.0 at index 2.
    torch::Tensor b = torch::tensor({5.0f, 2.0f, 9.0f, 1.0f, 7.0f});
    auto [max_val_b, max_idx_b] = torch::max(b, 0);
    std::cout << "\nmax([5,2,9,1,7]) = " << max_val_b.item<float>() << " at index " << max_idx_b.item<int64_t>()
              << ", CUDA book's own expected = 9.0 at index 2, match = "
              << (max_val_b.item<float>() == 9.0f && max_idx_b.item<int64_t>() == 2) << std::endl;

    // Extension: a genuine tie case, [4,12,7,12,3] (value 12 appears at
    // both index 1 and index 3). Hand-tracing the CUDA book's own
    // left >= right tie-break through its own tree structure -- round 1
    // pairs (4,12)->12@1 and (7,12)->12@3, leftover 3@4 passes through;
    // round 2 pairs (12@1, 12@3), a genuine tie, and left >= right keeps
    // the LEFT operand, so 12@1 survives, leftover 3@4 passes through;
    // round 3 pairs (12@1, 3@4), 12 wins -- final answer index 1. This is
    // an extension beyond the CUDA book's own two worked examples, neither
    // of which contains a tie.
    torch::Tensor c = torch::tensor({4.0f, 12.0f, 7.0f, 12.0f, 3.0f});
    auto [max_val_c, max_idx_c] = torch::max(c, 0);
    std::cout << "\nmax([4,12,7,12,3]) (a genuine tie between index 1 and index 3) = "
              << max_val_c.item<float>() << " at index " << max_idx_c.item<int64_t>()
              << ", hand-traced CUDA book's own left>=right tree tie-break expected = 12.0 at index 1, "
              << "match = " << (max_val_c.item<float>() == 12.0f && max_idx_c.item<int64_t>() == 1) << std::endl;

    // Direct probe of the CUDA book's own missing-else-branch bug's
    // failure mode: an array far larger than any single reduction round
    // the CUDA book's own worked examples use, confirming torch::max
    // correctly finds the true maximum and its true index at every scale,
    // with no unwritten output position possible.
    int64_t n = 100000;
    torch::Tensor big = torch::arange(0, n, torch::kFloat32);
    big[73421] = 999999.0f;  // plant a known maximum at a specific position
    auto [big_max_val, big_max_idx] = torch::max(big, 0);
    std::cout << "\nmax over n=" << n << " elements (far beyond any single CUDA reduction round), "
              << "with a planted maximum at index 73421: found value = " << big_max_val.item<float>()
              << " at index " << big_max_idx.item<int64_t>()
              << ", correct? " << (big_max_val.item<float>() == 999999.0f && big_max_idx.item<int64_t>() == 73421)
              << std::endl;

    return 0;
}
```

### File: `03_norm_clipping.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 14.3 hand-writes l2_norm (sqrt of a sum of
// squares, itself built on the reduction machinery from 14.1) and gradient
// clipping: rescale a vector by max_norm/actual_norm whenever its norm
// exceeds max_norm, so the result's own norm becomes exactly max_norm.
// torch::norm computes the identical L2 norm as a real, already-implemented
// function, and torch::nn::utils::clip_grad_norm_ is a real, production
// gradient-clipping utility used throughout actual neural network training
// code -- not a hand-built stand-in but the genuine mechanism deep learning
// frameworks use to prevent exploding gradients. This file verifies the
// CUDA book's own exact numbers via both a manual rescale (mirroring the
// CUDA book's own hand-built version) and the real clip_grad_norm_ utility,
// confirming both agree.
int main() {
    // Worked Example 14.3.1: l2_norm([3,4]) = 5.0 (a 3-4-5 right triangle).
    torch::Tensor v = torch::tensor({3.0f, 4.0f});
    torch::Tensor norm_v = torch::norm(v);
    std::cout << "norm([3,4]) = " << norm_v.item<float>()
              << ", CUDA book's own expected = 5.0, match = " << (norm_v.item<float>() == 5.0f) << std::endl;

    // Worked Example 14.3.2: gradient [3,4] clipped with max_norm=2.0:
    // scale factor 0.4, result [1.2, 1.6], verified norm = 2.0 exactly.
    // First, the manual rescale, mirroring the CUDA book's own hand-built
    // clip_gradient function directly.
    float max_norm = 2.0f;
    float actual_norm = norm_v.item<float>();
    float scale = max_norm / actual_norm;
    torch::Tensor clipped_manual = v * scale;
    std::cout << "\nmanual clip: scale = " << scale << ", CUDA book's own expected = 0.4, match = "
              << (scale == 0.4f) << std::endl;
    std::cout << "clipped [3,4] (manual) = " << clipped_manual << std::endl;
    torch::Tensor expected_clip = torch::tensor({1.2f, 1.6f});
    std::cout << "matches CUDA book's own [1.2, 1.6] (allclose)? "
              << torch::allclose(clipped_manual, expected_clip, 1e-5) << std::endl;
    std::cout << "norm of clipped result = " << torch::norm(clipped_manual).item<float>()
              << ", CUDA book's own expected = exactly 2.0, match = "
              << (std::abs(torch::norm(clipped_manual).item<float>() - 2.0f) < 1e-5) << std::endl;

    // Now the real, production gradient-clipping utility:
    // torch::nn::utils::clip_grad_norm_ -- not hand-built, the identical
    // mechanism real training loops use to prevent exploding gradients.
    // It operates in-place on a parameter's own .grad() field, mirroring
    // how it is genuinely used during backpropagation.
    torch::Tensor param = torch::zeros({2}, torch::requires_grad(true));
    param.mutable_grad() = torch::tensor({3.0f, 4.0f});
    double total_norm_before = torch::nn::utils::clip_grad_norm_(param, max_norm);
    std::cout << "\ntorch::nn::utils::clip_grad_norm_: reported pre-clip norm = " << total_norm_before
              << ", CUDA book's own expected = 5.0, match = " << (std::abs(total_norm_before - 5.0) < 1e-4) << std::endl;
    std::cout << "param.grad() after clip_grad_norm_ = " << param.grad() << std::endl;
    std::cout << "matches the manual rescale's own [1.2, 1.6] (allclose)? "
              << torch::allclose(param.grad(), expected_clip, 1e-4) << std::endl;

    return 0;
}
```

### File: `04_statistics.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 14.4 hand-writes a two-pass variance
// computation (mean first, then mean of squared deviations from that
// mean) for numerical stability, using POPULATION variance -- dividing by
// n, not n-1. torch::var is real and already-implemented, but its own
// DEFAULT behavior is an honest divergence from the CUDA book's own
// formula: by default, torch::var applies Bessel's correction, dividing
// by (n-1) rather than n, the standard convention for an unbiased SAMPLE
// variance estimator in statistics. This file verifies the CUDA book's
// own exact numbers, first showing torch::var's own DEFAULT does NOT
// reproduce them, then showing that passing correction=0 (population
// variance, matching the CUDA book's own formula exactly) does.
int main() {
    // Worked Example 14.4.1: [2,4,4,4,5,5,7,9]: mean=5.0, variance=4.0,
    // standard deviation=2.0 (population variance, divide by n=8).
    torch::Tensor x = torch::tensor({2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f});
    torch::Tensor mean_x = x.mean();
    std::cout << "mean([2,4,4,4,5,5,7,9]) = " << mean_x.item<float>()
              << ", CUDA book's own expected = 5.0, match = " << (mean_x.item<float>() == 5.0f) << std::endl;

    // Honest divergence: torch::var's own DEFAULT applies Bessel's
    // correction (divides by n-1=7), the standard unbiased SAMPLE variance
    // estimator -- this does NOT match the CUDA book's own population
    // variance formula (divide by n=8), confirmed here directly rather
    // than assumed.
    torch::Tensor var_default = x.var();
    std::cout << "\ntorch::var(x) DEFAULT (Bessel's correction, divides by n-1=7) = " << var_default.item<float>()
              << ", CUDA book's own expected population variance = 4.0, match = "
              << (std::abs(var_default.item<float>() - 4.0f) < 1e-5) << " (expected to be FALSE -- "
              << "this is the honest divergence)" << std::endl;

    // Passing correction=0 reproduces the CUDA book's own exact population
    // variance formula (divide by n), matching its own worked number
    // precisely.
    torch::Tensor var_population = x.var(/*dim=*/torch::nullopt, /*correction=*/0);
    std::cout << "torch::var(x, correction=0) (population variance, divides by n=8) = "
              << var_population.item<float>() << ", CUDA book's own expected = 4.0, match = "
              << (var_population.item<float>() == 4.0f) << std::endl;

    torch::Tensor std_population = x.std(/*dim=*/torch::nullopt, /*correction=*/0);
    std::cout << "torch::std(x, correction=0) = " << std_population.item<float>()
              << ", CUDA book's own expected = 2.0, match = " << (std_population.item<float>() == 2.0f) << std::endl;

    // Cross-check via the CUDA book's own two-pass method, computed here
    // independently of torch::var entirely: mean first, then mean of
    // squared deviations from that mean, using a hand-written loop.
    float mean_val = mean_x.item<float>();
    float sum_sq_dev = 0.0f;
    auto accessor = x.accessor<float, 1>();
    for (int64_t i = 0; i < x.size(0); i++) {
        float dev = accessor[i] - mean_val;
        sum_sq_dev += dev * dev;
    }
    float two_pass_variance = sum_sq_dev / (float)x.size(0);
    std::cout << "\ntwo-pass hand-written population variance (CUDA book's own method, computed "
              << "independently of torch::var) = " << two_pass_variance
              << ", matches torch::var(correction=0)'s own 4.0? " << (two_pass_variance == 4.0f) << std::endl;

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

## Chapter Summary

`torch::sum` and `torch::mean` reproduced the CUDA book's own `30`, `7.5`, and `24` exactly, and five fresh, independent repetitions of the identical computation each returned exactly `30.0` -- the direct opposite of the CUDA book's own `tensor_sum_incomplete`, whose missing kernel launch produced three different wrong values across three real runs. `torch::max` reproduced the CUDA book's own argmax results exactly, and a hand-traced tie-break case confirmed agreement with the CUDA book's own `left >= right` tree-reduction convention on one concrete tied array, while remaining correct on a 100,000-element array with a planted maximum, directly probing the CUDA book's own missing-else-branch bug's failure mode. `torch::norm` and the real, production `torch::nn::utils::clip_grad_norm_` reproduced the CUDA book's own gradient-clipping numbers exactly, using LibTorch's own genuine training-loop utility rather than a hand-built stand-in. And this chapter's own most consequential finding was an honest divergence: `torch::var`'s own default does NOT reproduce the CUDA book's own variance number at all, applying Bessel's correction for an unbiased sample-variance estimate rather than the CUDA book's own population-variance formula -- resolved by an explicit `correction=0`, confirmed against both the CUDA book's own two-pass method and NumPy's own identical `ddof` distinction.

## Self-Check Questions

1. Section 14.1's own five-repetition test confirms `torch::sum` is deterministic, but determinism alone does not prove correctness -- a function could deterministically return the SAME wrong answer every time. What in this section's own test specifically rules that out, beyond just checking that all five results are equal to each other?
2. Section 14.2's own tie-case test finds `torch::max`'s tie-breaking rule agrees with the CUDA book's own `left >= right` convention on `[4,12,7,12,3]`. Construct (in words) a different tied array where the two rules would disagree, using the reasoning from this section's own `[COMMON TRAP]`.
3. Section 14.3 tests `clip_grad_norm_` on a single parameter, where it behaves identically to the manual per-vector rescale. Using the section's own `[COMMON TRAP]`, explain what would change if `clip_grad_norm_` were instead called on TWO separate parameters at once.
4. Section 14.4 finds `torch::var(x)`'s default does NOT match the CUDA book's own `4.0`. As `n` (the array's element count) grows very large, what happens to the gap between `torch::var`'s default (dividing by `n-1`) and the CUDA book's own population variance (dividing by `n`), and why?
5. Section 14.4's two-pass hand-written loop recomputes variance independently of `torch::var` entirely, using `x.accessor<float,1>()`. Why is computing the mean in a first, separate pass (rather than accumulating a running mean and variance in a single pass) relevant to the "numerical stability" the CUDA book's own text specifically credits this method with?

## Where We Go Next

This chapter verified `torch::sum`, `torch::max`, `torch::norm`, `clip_grad_norm_`, and `torch::var` against the CUDA book's own exact worked numbers, directly probing two of its own genuine bugs' failure modes (a missing kernel launch, a missing else branch) and finding this chapter's own most consequential honest divergence in `torch::var`'s own default statistical convention. This closes Part 2: Basic Tensor Operations, which has verified `torch::Tensor`'s real element-wise, matrix, and reduction operations against the CUDA book's own hand-built kernels chapter by chapter. Part 3 turns from operations on tensors to the computational graph that remembers how they were produced -- Chapter 15 opens with graph node architecture, the structure `torch::Tensor`'s own `.backward()` (used throughout this book since Chapter 6) has been built on all along.

## Worked Solutions

**1.** All five results being equal to each other would also be true if `torch::sum` deterministically returned some OTHER fixed wrong value every time (for instance, if it had a genuine bug that consistently computed `29.0` instead of `30.0`) -- equality among the five results alone only rules out the CUDA book's own specific failure mode (different garbage each run), not incorrectness in general. What actually establishes correctness here is that each of the five results is independently compared against the known correct answer, `30.0` (via `result != 30.0f`), not merely against each other -- the test checks BOTH that the results agree with each other AND that the shared value they agree on is the mathematically correct one.

**2.** The CUDA book's own `left >= right` tie-break depends on the SPECIFIC pairing structure a tree reduction produces, which is a function of the array's length and how many rounds occur -- not simply "leftmost occurrence wins" in the original array's own reading order. Consider a genuine tie where the two equal maximum values are NOT adjacent after the tree's own first-round pairing puts them in a position where the tie is resolved by comparing against a value that arrived from a different branch of the tree than a naive "first occurrence" rule would predict -- for instance, a longer array where the two tied maxima end up as the LEFT and RIGHT operands of the FINAL round's comparison, having each separately survived several earlier rounds; the CUDA book's own rule would keep whichever one is on the "left" of that final pairing, which depends on the tree's own branching structure and need not correspond to which one appeared earlier in the original array, while `torch::max`'s own rule always simply returns whichever tied maximum appears earliest in the original array's own reading order regardless of any tree structure.

**3.** With two separate parameters, `clip_grad_norm_` would first compute a SINGLE combined norm treating both parameters' gradients as if they were concatenated into one larger vector (the square root of the sum of squares across BOTH gradients together, not two separate per-parameter norms), then rescale EACH parameter's own gradient by that one shared scale factor. This could produce a materially different result than clipping each parameter independently with the manual approach: a parameter whose own individual gradient norm is well under `max_norm` could still get rescaled downward, because the OTHER parameter's much larger gradient pushed the combined norm over the threshold -- a genuinely different, and for real training, more correct, behavior than treating each parameter's gradient as an independent quantity to clip in isolation.

**4.** As `n` grows very large, `n-1` and `n` become proportionally closer to each other (their RATIO `n/(n-1)` approaches `1`), so the two variance formulas' RESULTS converge -- the relative gap between dividing by `n-1` versus `n` shrinks toward zero as `n` increases, even though the two formulas remain technically different at every finite `n`. This is exactly why Bessel's correction matters most for SMALL samples (where `n-1` is a meaningfully smaller divisor than `n`, as in this section's own `n=8` example, producing a visibly different `4.57143` versus `4.0`) and matters progressively less as a genuine "large sample" approximation, which is part of why the distinction is sometimes overlooked in casual use with large datasets even though it remains formally present.

**5.** A single-pass running computation of variance (accumulating a running sum, a running sum of squares, and deriving variance from `E[X^2] - (E[X])^2` without ever computing the mean as a separate first step) is well known to suffer from CATASTROPHIC CANCELLATION in floating-point arithmetic when the data's own mean is large relative to its own spread: `E[X^2]` and `(E[X])^2` can each be large, nearly-equal floating-point numbers whose difference (the true, much smaller variance) loses significant precision to rounding error, or in extreme cases even comes out falsely negative. The two-pass method sidesteps this entirely: by computing the exact mean FIRST, each subsequent deviation (`x[i] - mean`) is a genuinely small number close to zero, and squaring and summing small numbers does not suffer the same cancellation problem -- which is precisely the numerical-stability property the CUDA book's own text credits the two-pass method with, and exactly why this section's own hand-written cross-check deliberately reproduces that same two-pass structure rather than a mathematically equivalent but numerically riskier single-pass shortcut.
