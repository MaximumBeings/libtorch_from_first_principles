# Chapter 8: Tensor Creation Functions

> "The CUDA C++ edition opens Chapter 8 by calling an allocated, uninitialized buffer 'a promise, not a value' -- factory functions exist to fill that promise predictably, whether through arithmetic sequences, reproducible randomness, or disk I/O. `torch::Tensor` has been making good on that promise since this book's own Chapter 1, through `torch::arange`, `torch::linspace`, `torch::eye`, and `torch::rand` -- functions this book has used without ever asking what the CUDA book's own hand-built versions of them would have had to get right. This chapter asks that question directly: for each of the CUDA book's three factory categories, does the real, already-implemented `torch::Tensor` function produce the CUDA book's own published numbers, and where CUDA's device-only `curand` forced the CUDA book to hand-build its own random generator from nothing, does `torch::Tensor`'s real generator already do -- correctly -- what that hand-built one had to discover through two genuine bugs?"

**What you will understand by the end of this chapter:**

- That `torch::arange` and `torch::linspace` reproduce the CUDA book's own worked numbers exactly for shape-generating parameters `(0, 10, 2)` and `(0, 1, 5)` -- and that the CUDA book's own stated distinction, `arange` never reaching its stop value while `linspace` always includes both endpoints, holds on the real, already-implemented functions, checked by direct comparison rather than assumed from their names
- That `torch::eye`'s ordinary identity matches the CUDA book's `eye(4, k=0)` exactly, and that `torch::diag(values, offset)` -- a genuinely different, more general function than `eye`, rather than a `k`-argument overload of it -- reproduces the CUDA book's own `eye(4, k=1)` superdiagonal example bit for bit
- Why `torch::manual_seed()` and `torch::Tensor`'s real random generator make the CUDA book's own reproducibility claim -- identical seed, identical stream -- true by direct, repeated test, without ever needing the CUDA book's hand-built `SimpleRNG` or its own discovered seed-mixing fix
- That `torch::randperm()` and `torch::multinomial()` are the real, already-implemented LibTorch functions performing the exact two algorithms the CUDA book hand-builds from its own xorshift generator -- Fisher-Yates shuffling and inverse-CDF categorical sampling -- verified against the identical invariants the CUDA book itself checks: element-sum preservation, and empirical frequency converging toward target probability within a statistically justified bound
- That the CUDA book's own genuine off-by-one bug in `count_rows_naive` -- undercounting a CSV file whose last line lacks a trailing newline -- is not a CUDA-specific bug at all, and reproduces identically in this book's own plain C++ file I/O, alongside a real round-trip write/read of an actual `torch::Tensor`'s own values

**What you need to know first:**

- Chapter 1's Section 1.4 and Chapter 6, which have both used `torch::arange`, `torch::zeros`, and other factory functions without examining their guarantees directly -- this chapter is the first to test those guarantees rather than merely rely on them
- Chapter 6's Section 6.3, which distinguished a tensor's own values from a separately-allocated gradient buffer -- this chapter's Section 8.2 leans on the same "two things that could plausibly share memory but genuinely don't" style of test, applied to two calls of the same seeded generator instead
- If you've read the CUDA C++ edition's Chapter 8: its Section 8.2 exists specifically because CUDA's `curand` is device-only and unusable in the CUDA book's own host-side testing code, forcing a hand-built `SimpleRNG` from first principles. `torch::Tensor`'s random functions were never under that restriction -- a real, production-quality generator is already present on the CPU this book has been testing against since Chapter 1 -- so this chapter's job is to confirm that generator already gets right what the CUDA book's own `SimpleRNG` had to discover through two genuine, hard-won bugs

## 8.1 Factory Functions: `arange`, `linspace`, and `eye` `[FOUNDATIONAL]`

### Intuition

`torch::arange(start, stop, step)` and `torch::linspace(start, stop, n)` both fill a buffer with an arithmetic sequence, but answer a subtly different question: `arange` asks "what values do I get if I step by a fixed amount, stopping before I reach the boundary?" while `linspace` asks "what values do I get if I need exactly `n` evenly spaced points, including both ends?" `torch::eye(n)` fills a square buffer with `1`s exactly where the CUDA book's own `eye_kernel` places them: positions where `col - row` equals a chosen offset.

### Background

The CUDA book's own Worked Example 8.1.1 gives `arange(start=0, step=2, n=5) = [0.0, 2.0, 4.0, 6.0, 8.0]` -- five values, never reaching a would-be stop of `10` -- and `linspace(0, 1, n=5) = [0.00, 0.25, 0.50, 0.75, 1.00]`, both ends genuinely present. Its Worked Example 8.1.2 gives `eye(4, k=0)` as the ordinary identity matrix, and `eye(4, k=1)` with `1`s at exactly `(0,1)`, `(1,2)`, `(2,3)`.

### Worked Example 8.1.1 -- `arange`, `linspace`, `eye`, and `diag` against the CUDA book's own numbers

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 8.1 hand-writes three device kernels:
// arange_kernel (data[i] = start + i*step, stop excluded), linspace_kernel
// (data[i] = start + i*(stop-start)/(n-1), stop included), and eye_kernel
// (1 where col-row == k, else 0). torch::arange, torch::linspace, and
// torch::eye are real, already-implemented factory functions -- this file
// verifies each against the CUDA book's own two worked examples, plus its
// own diagonal-offset eye(4,k=1) case (built via torch::diag, since
// torch::eye itself takes no offset parameter).
int main() {
    // Worked Example 8.1.1: arange(start=0, step=2, n=5) and linspace(0,1,5).
    torch::Tensor a = torch::arange(0, 10, 2, torch::kFloat32);  // 0,2,4,6,8 -- stop=10 excluded
    std::cout << "arange(0,10,2) = " << a << std::endl;
    std::cout << "arange numel = " << a.numel() << ", CUDA book's own expected [0,2,4,6,8], match = "
              << torch::equal(a, torch::tensor({0.0f, 2.0f, 4.0f, 6.0f, 8.0f})) << std::endl;

    torch::Tensor l = torch::linspace(0, 1, 5);  // 0.00,0.25,0.50,0.75,1.00 -- both ends included
    std::cout << "linspace(0,1,5) = " << l << std::endl;
    std::cout << "linspace matches CUDA book's own [0,0.25,0.5,0.75,1.0]? "
              << torch::allclose(l, torch::tensor({0.0f, 0.25f, 0.5f, 0.75f, 1.0f})) << std::endl;

    // Worked Example 8.1.2: eye(4, k=0), the ordinary identity matrix.
    torch::Tensor e0 = torch::eye(4);
    std::cout << "eye(4):\n" << e0 << std::endl;
    torch::Tensor expected_e0 = torch::tensor({{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f},{0.0f,0.0f,0.0f,1.0f}});
    std::cout << "eye(4) matches ordinary identity? " << torch::equal(e0, expected_e0) << std::endl;

    // eye(4, k=1): the superdiagonal case. torch::eye has no k parameter --
    // the real LibTorch-native way to build an offset diagonal is
    // torch::diag(values, offset), which places `values` along the k-th
    // diagonal of a square matrix, genuinely a different (more general)
    // function than eye rather than a k-argument overload of it.
    torch::Tensor e1 = torch::diag(torch::ones({3}), 1);
    std::cout << "diag(ones(3), 1) [eye(4,k=1) equivalent]:\n" << e1 << std::endl;
    torch::Tensor expected_e1 = torch::tensor({{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f},{0.0f,0.0f,0.0f,1.0f},{0.0f,0.0f,0.0f,0.0f}});
    std::cout << "matches CUDA book's own eye(4,k=1) (1s at (0,1),(1,2),(2,3) only)? "
              << torch::equal(e1, expected_e1) << std::endl;

    // arange never reaches its stop value; linspace always includes both
    // endpoints -- the CUDA book's own stated distinction, checked directly.
    std::cout << "arange(0,10,2) ever equals 10 (its own stop)? "
              << (a.eq(10.0f).any().item<bool>()) << std::endl;
    std::cout << "linspace(0,1,5) includes its own stop 1.0? "
              << (l.eq(1.0f).any().item<bool>()) << std::endl;

    // linspace's n=1 edge case: the CUDA book's kernel divides by (n-1),
    // undefined at n=1. torch::linspace(start, stop, 1) is real, tested
    // code -- does it crash, or handle this some other real way?
    torch::Tensor l1 = torch::linspace(0, 1, 1);
    std::cout << "linspace(0,1,1) = " << l1 << " (n=1: torch::linspace's own real, tested behavior)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_factory_functions.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_factory_functions
./01_factory_functions
```

```text
arange(0,10,2) =  0
 2
 4
 6
 8
[ CPUFloatType{5} ]
arange numel = 5, CUDA book's own expected [0,2,4,6,8], match = 1
linspace(0,1,5) =  0.0000
 0.2500
 0.5000
 0.7500
 1.0000
[ CPUFloatType{5} ]
linspace matches CUDA book's own [0,0.25,0.5,0.75,1.0]? 1
eye(4):
 1  0  0  0
 0  1  0  0
 0  0  1  0
 0  0  0  1
[ CPUFloatType{4,4} ]
eye(4) matches ordinary identity? 1
diag(ones(3), 1) [eye(4,k=1) equivalent]:
 0  1  0  0
 0  0  1  0
 0  0  0  1
 0  0  0  0
[ CPUFloatType{4,4} ]
matches CUDA book's own eye(4,k=1) (1s at (0,1),(1,2),(2,3) only)? 1
arange(0,10,2) ever equals 10 (its own stop)? 0
linspace(0,1,5) includes its own stop 1.0? 1
linspace(0,1,1) =  0
[ CPUFloatType{1} ] (n=1: torch::linspace's own real, tested behavior)
```

Independently cross-checked via NumPy's own `arange`/`linspace`/`eye`, and via a separate call into `torch.linspace(0,1,1)` through Python's own bindings (a genuinely different binding layer from the C++ this book compiled directly):

```text
numpy arange(0,10,2): [0.0, 2.0, 4.0, 6.0, 8.0]
numpy linspace(0,1,5): [0.0, 0.25, 0.5, 0.75, 1.0]
numpy eye(4): [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
numpy eye(4,k=1): [[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
torch.linspace(0,1,1): tensor([0.])
```

### Discussion

Every figure matches the CUDA book's own published numbers exactly, and NumPy's own independent implementations of `arange`, `linspace`, and `eye` -- sharing no code with `torch::Tensor`'s internals -- reproduce all four results without exception. `arange(0,10,2).eq(10.0f).any()` reporting `0` (false) and `linspace(0,1,5).eq(1.0f).any()` reporting `1` (true) turns the CUDA book's own prose distinction into a directly tested fact rather than something taken on faith from the two functions' names. `torch::eye` genuinely has no `k` parameter the way the CUDA book's `eye_kernel` does -- `torch::diag(values, offset)` is the real function that generalizes to an offset diagonal, and it is a conceptually broader function than `eye` (it places arbitrary values along any diagonal, not just `1`s along the main one) rather than a same-named overload.

The `linspace(0,1,1)` test resolves a question the CUDA book's own text raises but a hand-built kernel would crash on: its `linspace_kernel` divides by `(n-1)`, undefined arithmetic at `n=1`. `torch::linspace(0, 1, 1)` does not crash -- it returns a single-element tensor holding `0` (the start value), confirmed identically through Python's own separate binding layer. This is a real, positive finding about `torch::Tensor`'s own implementation, not an assumption: the production function was built to handle the edge case the CUDA book's own from-scratch kernel would need a special-cased guard to avoid.

## 8.2 Random Tensors: A Generator That Already Passes the CUDA Book's Own Tests `[FOUNDATIONAL]`

### Intuition

The CUDA book builds `SimpleRNG` from nothing because CUDA's own `curand` library is device-only, unusable in the CUDA book's own host-side test code -- and along the way, testing that hand-built generator surfaces two genuine bugs: an identical seed always producing an identical stream (a reproducibility trap when reused carelessly) and a small seed producing a measurably weak first draw, fixed by pre-mixing the seed against a fixed 64-bit constant. `torch::Tensor`'s random functions were never built under CUDA's device-only restriction -- `torch::manual_seed()` and a real generator are already present, and this section asks whether that real generator already gets right what the CUDA book's own generator had to discover through those two bugs.

### Background

The CUDA book's own two structural checks: Worked Example 8.2.3's Fisher-Yates shuffle of `[10, 20, 30, 40, 50]` must preserve the array's element sum, `150`, and Worked Example 8.2.4's inverse-CDF sampling from probabilities `[0.2, 0.3, 0.5]` across 10,000 draws must converge its empirical frequencies toward those targets -- the CUDA book's own run reports category differences of `0.0026`, `0.0006`, and `0.0032`.

### Worked Example 8.2.1 -- reproducibility, Fisher-Yates, and categorical sampling, on the real generator

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 8.2 hand-writes a xorshift generator,
// SimpleRNG, because CUDA's own curand is device-only and unavailable in
// the book's host-side testing code -- then uses it to build Fisher-Yates
// shuffling and inverse-CDF categorical sampling from scratch, discovering
// two real bugs along the way (a reproducibility trap from identical
// seeds, and weak first draws from small seeds). torch::Tensor's random
// functions never face that curand restriction: torch::manual_seed() and
// a real, production-quality generator (Mersenne-Twister-derived on CPU)
// are already present and already seedable. This file verifies the CUDA
// book's own two invariant-based tests -- Fisher-Yates preserves the
// array's element sum, and categorical sampling's empirical frequencies
// converge toward their target probabilities -- using torch::randperm()
// and torch::multinomial(), the real, already-implemented LibTorch
// functions that perform exactly those two algorithms.
int main() {
    // Reproducibility: the CUDA book's own SimpleRNG(42) gives an
    // identical stream every time it's reconstructed with the same seed.
    // torch::manual_seed() makes the identical claim about the real
    // generator, tested directly here rather than assumed.
    torch::manual_seed(42);
    torch::Tensor draw1 = torch::rand({4});
    torch::manual_seed(42);
    torch::Tensor draw2 = torch::rand({4});
    std::cout << "draw1 = " << draw1 << std::endl;
    std::cout << "draw2 (same seed, reconstructed) = " << draw2 << std::endl;
    std::cout << "identical seed produces identical stream? " << torch::equal(draw1, draw2) << std::endl;

    // Fisher-Yates: the CUDA book hand-writes the shuffle loop and checks
    // that element sum is preserved as its correctness invariant.
    // torch::randperm(n) IS a Fisher-Yates-family shuffle, real and
    // already implemented -- used here to permute indices into the CUDA
    // book's own example array [10,20,30,40,50].
    torch::manual_seed(7);
    torch::Tensor arr = torch::tensor({10, 20, 30, 40, 50});
    torch::Tensor perm = torch::randperm(5);
    torch::Tensor shuffled = arr.index_select(0, perm);
    std::cout << "original array = " << arr << std::endl;
    std::cout << "randperm(5) indices = " << perm << std::endl;
    std::cout << "shuffled array = " << shuffled << std::endl;
    int64_t sum_before = arr.sum().item<int64_t>();
    int64_t sum_after = shuffled.sum().item<int64_t>();
    std::cout << "sum before = " << sum_before << ", sum after = " << sum_after
              << ", invariant holds? " << (sum_before == sum_after) << std::endl;
    // A permutation must also be a genuine rearrangement: every original
    // value present exactly once, same as the CUDA book's own check.
    torch::Tensor sorted_shuffled = std::get<0>(torch::sort(shuffled));
    torch::Tensor sorted_original = std::get<0>(torch::sort(arr));
    std::cout << "shuffled array is a genuine permutation of the original? "
              << torch::equal(sorted_shuffled, sorted_original) << std::endl;

    // Inverse-CDF categorical sampling: the CUDA book hand-writes
    // sample_category() walking a cumulative-probability array, then
    // draws 10,000 samples from probabilities [0.2, 0.3, 0.5] and checks
    // empirical frequency against target. torch::multinomial() performs
    // exactly this algorithm, already implemented, given the probabilities
    // directly rather than a hand-built cumulative array.
    torch::manual_seed(2024);
    torch::Tensor probs = torch::tensor({0.2, 0.3, 0.5});
    int64_t n_draws = 10000;
    torch::Tensor samples = torch::multinomial(probs, n_draws, /*replacement=*/true);

    torch::Tensor counts = torch::zeros({3}, torch::kInt64);
    for (int64_t c = 0; c < 3; c++) {
        counts[c] = (samples == c).sum();
    }
    std::cout << "10000 draws from multinomial(probs=[0.2,0.3,0.5]):" << std::endl;
    for (int64_t c = 0; c < 3; c++) {
        double freq = counts[c].item<int64_t>() / (double)n_draws;
        double target = probs[c].item<double>();
        std::cout << "  category " << c << ": count=" << counts[c].item<int64_t>()
                  << ", frequency=" << freq << ", target=" << target
                  << ", |diff|=" << std::abs(freq - target) << std::endl;
    }
    bool all_close = true;
    for (int64_t c = 0; c < 3; c++) {
        double freq = counts[c].item<int64_t>() / (double)n_draws;
        double target = probs[c].item<double>();
        if (std::abs(freq - target) > 0.02) all_close = false;
    }
    std::cout << "all three empirical frequencies within 0.02 of target (matching the CUDA book's own convergence check)? "
              << all_close << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_random_generation.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_random_generation
./02_random_generation
```

```text
draw1 =  0.8823
 0.9150
 0.3829
 0.9593
[ CPUFloatType{4} ]
draw2 (same seed, reconstructed) =  0.8823
 0.9150
 0.3829
 0.9593
[ CPUFloatType{4} ]
identical seed produces identical stream? 1
original array =  10
 20
 30
 40
 50
[ CPULongType{5} ]
randperm(5) indices =  0
 1
 3
 2
 4
[ CPULongType{5} ]
shuffled array =  10
 20
 40
 30
 50
[ CPULongType{5} ]
sum before = 150, sum after = 150, invariant holds? 1
shuffled array is a genuine permutation of the original? 1
10000 draws from multinomial(probs=[0.2,0.3,0.5]):
  category 0: count=2071, frequency=0.2071, target=0.2, |diff|=0.0071
  category 1: count=3004, frequency=0.3004, target=0.3, |diff|=0.000399988
  category 2: count=4925, frequency=0.4925, target=0.5, |diff|=0.0075
all three empirical frequencies within 0.02 of target (matching the CUDA book's own convergence check)? 1
```

Cross-checked two independent ways: reproducibility confirmed through Python's own separate `torch` binding, and the three empirical frequencies checked against a hand-derived statistical bound (the binomial proportion standard error, `sqrt(p*(1-p)/n)`) rather than only the fixed `0.02` threshold in the code above:

```text
python torch.rand reproducibility: True
cat0: target=0.2, standard error=0.0040, 3-sigma bound=[0.1880, 0.2120]
cat1: target=0.3, standard error=0.0046, 3-sigma bound=[0.2863, 0.3137]
cat2: target=0.5, standard error=0.0050, 3-sigma bound=[0.4850, 0.5150]
cat0: observed=0.2071, within 3-sigma bound [0.1880,0.2120]? True
cat1: observed=0.3004, within 3-sigma bound [0.2863,0.3137]? True
cat2: observed=0.4925, within 3-sigma bound [0.4850,0.5150]? True
```

### Discussion

`torch::manual_seed(42)` followed by two separate `torch::rand({4})` calls produces bit-identical tensors -- the exact reproducibility guarantee the CUDA book's own `SimpleRNG(42)` makes, confirmed here on the real generator and independently reconfirmed through Python's own separate binding layer. `torch::randperm(5)` -- a real, already-implemented permutation generator from the Fisher-Yates family -- shuffles the CUDA book's own example array `[10, 20, 30, 40, 50]` while preserving its element sum at `150` exactly, and a second check (sorting both arrays and comparing) confirms the result is a genuine rearrangement of the original values rather than merely a sum-preserving coincidence. `torch::multinomial(probs, 10000, true)` -- inverse-CDF sampling, already implemented, given the probability vector directly rather than a hand-built cumulative array -- produces empirical frequencies `0.2071`, `0.3004`, and `0.4925` for target probabilities `0.2`, `0.3`, and `0.5`, all three landing inside a 3-sigma statistical bound computed independently of any RNG at all, from the binomial proportion standard error formula alone. Nowhere in this section did `torch::Tensor`'s random functions need the CUDA book's own two discovered fixes -- a real generator was never at risk of the reproducibility trap being a *bug* (the CUDA book calls it a trap only when a caller mistakenly relies on non-reuse of a seed; used correctly, as here, identical seeds producing identical streams is the entire point) and never needed the small-seed pre-mixing fix, because a production-quality generator's statistical properties don't depend on the numeric size of the seed the way a raw, unmixed xorshift state does.

> `[COMMON TRAP]` "Reproducibility" and "the small-seed bug" are two different properties, easy to conflate. The CUDA book's own `SimpleRNG(42)` producing an identical stream every time is *correct*, desired behavior -- the "trap" the CUDA book warns about is a caller accidentally reusing the same seed when they wanted independent randomness, not a flaw in reproducibility itself. The small-seed weakness is a separate, genuine defect: `NaiveRNG(42)`'s very first draw comes out `0.000000`, because `42`'s bit pattern occupies only its lowest six bits and the xorshift recurrence hasn't had time to spread that entropy into the high bits the CUDA book's own generator reads its output from. `torch::Tensor`'s real generator is checked in this section only for the first property (reproducibility) and the two algorithm-level properties (shuffle and categorical sampling); a full audit of a production Mersenne-Twister-family generator's bit-mixing quality is outside a single worked example's scope, but this book has no reason to expect a production generator to share a from-scratch xorshift's specific mixing weaknesses.

## 8.3 CSV Import/Export: An Off-By-One Bug With Nothing GPU-Specific About It `[FOUNDATIONAL]`

### Intuition

`torch::Tensor` has no built-in CSV reader or writer -- serialization in LibTorch's own native format is a separate mechanism entirely, and CSV I/O for a tensor's own values is ordinary C++ file I/O regardless of whether the tensor's data originated on a CPU or a GPU. The CUDA book's own Section 8.3 code is, itself, already plain host-side C, with nothing CUDA-specific in it; this section reuses that same style of code directly, against a real `torch::Tensor`'s own values, including the CUDA book's own genuine off-by-one bug.

### Background

The CUDA book's own Worked Example 8.3.1: a row counter that counts `'\n'` characters gives the correct answer, `3`, for a 3-row file that ends with a trailing newline, but undercounts to `2` for an otherwise-identical 3-row file whose last line has no trailing newline. Its Worked Example 8.3.2 confirms a plain round trip -- writing `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]` to disk and reading it back reproduces the original values exactly.

### Worked Example 8.3.1 -- round trip and the off-by-one bug, on a real tensor and real files

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstdio>

// The CUDA C++ edition's Section 8.3 writes and reads CSV files holding
// tensor data using plain C file I/O (fopen/fprintf/fscanf) -- code with
// nothing GPU-specific about it, since it runs entirely on the host either
// way. This file reuses that same plain-file-I/O approach, but writes and
// reads a real torch::Tensor's own data via its accessor, and reproduces
// the CUDA book's own genuine off-by-one bug: a naive row counter that
// counts newline characters undercounts by one on a file whose last line
// has no trailing newline.
void write_csv(const char* filename, const torch::Tensor& t) {
    auto acc = t.accessor<float, 2>();
    FILE* f = fopen(filename, "w");
    for (int64_t r = 0; r < acc.size(0); r++) {
        for (int64_t c = 0; c < acc.size(1); c++) {
            fprintf(f, "%.1f", acc[r][c]);
            if (c < acc.size(1) - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// The CUDA book's own buggy row counter: counts '\n' characters only.
int count_rows_naive(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
    }
    fclose(f);
    return count;
}

// The CUDA book's own fix: also count a final, non-newline-terminated line.
int count_rows_fixed(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch, last_ch = '\n';
    bool saw_any_byte = false;
    while ((ch = fgetc(f)) != EOF) {
        saw_any_byte = true;
        if (ch == '\n') count++;
        last_ch = ch;
    }
    if (saw_any_byte && last_ch != '\n') count++;
    fclose(f);
    return count;
}

// Reads a CSV back into a flat buffer, up to max_values floats.
int read_csv(const char* filename, float* out, int max_values) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    while (count < max_values && fscanf(f, "%f", &out[count]) == 1) {
        count++;
        fgetc(f);  // consume ',' or '\n'
    }
    fclose(f);
    return count;
}

int main() {
    // Worked Example 8.3.2: round trip. Build a real torch::Tensor, write
    // it, read it back, compare against the original tensor's own values.
    torch::Tensor original = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    write_csv("/tmp/ch8_roundtrip.csv", original);

    float readback_buf[6];
    int n_read = read_csv("/tmp/ch8_roundtrip.csv", readback_buf, 6);
    torch::Tensor readback = torch::from_blob(readback_buf, {2, 3}, torch::kFloat32).clone();

    std::cout << "original =\n" << original << std::endl;
    std::cout << "n_read = " << n_read << std::endl;
    std::cout << "readback =\n" << readback << std::endl;
    std::cout << "round trip exact match? " << torch::equal(original, readback) << std::endl;

    // Worked Example 8.3.1: the off-by-one bug, triggered on a file
    // deliberately written WITHOUT a trailing newline.
    FILE* f = fopen("/tmp/ch8_with_newline.csv", "w");
    fprintf(f, "1,2\n3,4\n5,6\n");  // 3 rows, trailing newline present
    fclose(f);

    FILE* f2 = fopen("/tmp/ch8_no_newline.csv", "w");
    fprintf(f2, "1,2\n3,4\n5,6");  // 3 rows, NO trailing newline on the last one
    fclose(f2);

    int naive_with = count_rows_naive("/tmp/ch8_with_newline.csv");
    int fixed_with = count_rows_fixed("/tmp/ch8_with_newline.csv");
    int naive_without = count_rows_naive("/tmp/ch8_no_newline.csv");
    int fixed_without = count_rows_fixed("/tmp/ch8_no_newline.csv");

    std::cout << "\nfile WITH trailing newline (3 real rows):" << std::endl;
    std::cout << "  count_rows_naive = " << naive_with << " (expected 3, correct? " << (naive_with == 3) << ")" << std::endl;
    std::cout << "  count_rows_fixed = " << fixed_with << " (expected 3, correct? " << (fixed_with == 3) << ")" << std::endl;

    std::cout << "file WITHOUT trailing newline (still 3 real rows):" << std::endl;
    std::cout << "  count_rows_naive = " << naive_without << " (expected 3, BUG: undercounts? " << (naive_without != 3) << ")" << std::endl;
    std::cout << "  count_rows_fixed = " << fixed_without << " (expected 3, correct? " << (fixed_without == 3) << ")" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_csv_io.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_csv_io
./03_csv_io
```

```text
original =
 1  2  3
 4  5  6
[ CPUFloatType{2,3} ]
n_read = 6
readback =
 1  2  3
 4  5  6
[ CPUFloatType{2,3} ]
round trip exact match? 1

file WITH trailing newline (3 real rows):
  count_rows_naive = 3 (expected 3, correct? 1)
  count_rows_fixed = 3 (expected 3, correct? 1)
file WITHOUT trailing newline (still 3 real rows):
  count_rows_naive = 2 (expected 3, BUG: undercounts? 1)
  count_rows_fixed = 3 (expected 3, correct? 1)
```

Independently cross-checked via NumPy's own `loadtxt()` for the round trip, and via raw Python byte-counting for the off-by-one bug -- both reading the exact same files this book's own C++ binary wrote, through completely separate code:

```text
numpy readback: [[1,2,3],[4,5,6]], round trip exact match? True
file WITH trailing newline: raw newline count = 3 (3 real rows)
file WITHOUT trailing newline: raw newline count = 2 (3 real rows, naive counter undercounts)
```

### Discussion

The round trip reproduces `original`'s values exactly through `readback`, confirmed both by `torch::equal()` inside the C++ binary and by NumPy's own, completely independent `loadtxt()` reading the identical file afterward. The off-by-one bug reproduces the CUDA book's own exact finding: `count_rows_naive` correctly reports `3` for a file with a trailing newline, then silently drops to `2` -- an undercount, not a crash, which is precisely why the CUDA book calls this bug dangerous -- for an otherwise-identical file missing only its very last newline character. `count_rows_fixed` reports `3` in both cases, because it separately tracks whether the file's last character was itself a newline and, if not, counts the unterminated final line explicitly. This bug has nothing to do with CUDA, kernels, or device memory at all -- it is a plain host-side C file-parsing mistake, and it reproduces identically in this book's own environment because the underlying C standard library function, `fgetc`, behaves identically regardless of whether the tensor whose values are being written started life on a CPU or a GPU.

> `[COMMON TRAP]` A round-trip test that only ever writes files with a trailing newline -- like Worked Example 8.3.2's own `write_csv`, which always emits one -- can never expose the off-by-one bug on its own. The bug in Worked Example 8.3.1 only becomes visible because a *second*, deliberately different file was constructed by hand, without the trailing newline `write_csv` would normally add. Testing a round trip and testing an edge case are two different disciplines; passing the first proves nothing about the second.

## Complete Runnable Code

### File: `01_factory_functions.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 8.1 hand-writes three device kernels:
// arange_kernel (data[i] = start + i*step, stop excluded), linspace_kernel
// (data[i] = start + i*(stop-start)/(n-1), stop included), and eye_kernel
// (1 where col-row == k, else 0). torch::arange, torch::linspace, and
// torch::eye are real, already-implemented factory functions -- this file
// verifies each against the CUDA book's own two worked examples, plus its
// own diagonal-offset eye(4,k=1) case (built via torch::diag, since
// torch::eye itself takes no offset parameter).
int main() {
    // Worked Example 8.1.1: arange(start=0, step=2, n=5) and linspace(0,1,5).
    torch::Tensor a = torch::arange(0, 10, 2, torch::kFloat32);  // 0,2,4,6,8 -- stop=10 excluded
    std::cout << "arange(0,10,2) = " << a << std::endl;
    std::cout << "arange numel = " << a.numel() << ", CUDA book's own expected [0,2,4,6,8], match = "
              << torch::equal(a, torch::tensor({0.0f, 2.0f, 4.0f, 6.0f, 8.0f})) << std::endl;

    torch::Tensor l = torch::linspace(0, 1, 5);  // 0.00,0.25,0.50,0.75,1.00 -- both ends included
    std::cout << "linspace(0,1,5) = " << l << std::endl;
    std::cout << "linspace matches CUDA book's own [0,0.25,0.5,0.75,1.0]? "
              << torch::allclose(l, torch::tensor({0.0f, 0.25f, 0.5f, 0.75f, 1.0f})) << std::endl;

    // Worked Example 8.1.2: eye(4, k=0), the ordinary identity matrix.
    torch::Tensor e0 = torch::eye(4);
    std::cout << "eye(4):\n" << e0 << std::endl;
    torch::Tensor expected_e0 = torch::tensor({{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f},{0.0f,0.0f,0.0f,1.0f}});
    std::cout << "eye(4) matches ordinary identity? " << torch::equal(e0, expected_e0) << std::endl;

    // eye(4, k=1): the superdiagonal case. torch::eye has no k parameter --
    // the real LibTorch-native way to build an offset diagonal is
    // torch::diag(values, offset), which places `values` along the k-th
    // diagonal of a square matrix, genuinely a different (more general)
    // function than eye rather than a k-argument overload of it.
    torch::Tensor e1 = torch::diag(torch::ones({3}), 1);
    std::cout << "diag(ones(3), 1) [eye(4,k=1) equivalent]:\n" << e1 << std::endl;
    torch::Tensor expected_e1 = torch::tensor({{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f},{0.0f,0.0f,0.0f,1.0f},{0.0f,0.0f,0.0f,0.0f}});
    std::cout << "matches CUDA book's own eye(4,k=1) (1s at (0,1),(1,2),(2,3) only)? "
              << torch::equal(e1, expected_e1) << std::endl;

    // arange never reaches its stop value; linspace always includes both
    // endpoints -- the CUDA book's own stated distinction, checked directly.
    std::cout << "arange(0,10,2) ever equals 10 (its own stop)? "
              << (a.eq(10.0f).any().item<bool>()) << std::endl;
    std::cout << "linspace(0,1,5) includes its own stop 1.0? "
              << (l.eq(1.0f).any().item<bool>()) << std::endl;

    // linspace's n=1 edge case: the CUDA book's kernel divides by (n-1),
    // undefined at n=1. torch::linspace(start, stop, 1) is real, tested
    // code -- does it crash, or handle this some other real way?
    torch::Tensor l1 = torch::linspace(0, 1, 1);
    std::cout << "linspace(0,1,1) = " << l1 << " (n=1: torch::linspace's own real, tested behavior)" << std::endl;

    return 0;
}
```

### File: `02_random_generation.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 8.2 hand-writes a xorshift generator,
// SimpleRNG, because CUDA's own curand is device-only and unavailable in
// the book's host-side testing code -- then uses it to build Fisher-Yates
// shuffling and inverse-CDF categorical sampling from scratch, discovering
// two real bugs along the way (a reproducibility trap from identical
// seeds, and weak first draws from small seeds). torch::Tensor's random
// functions never face that curand restriction: torch::manual_seed() and
// a real, production-quality generator (Mersenne-Twister-derived on CPU)
// are already present and already seedable. This file verifies the CUDA
// book's own two invariant-based tests -- Fisher-Yates preserves the
// array's element sum, and categorical sampling's empirical frequencies
// converge toward their target probabilities -- using torch::randperm()
// and torch::multinomial(), the real, already-implemented LibTorch
// functions that perform exactly those two algorithms.
int main() {
    // Reproducibility: the CUDA book's own SimpleRNG(42) gives an
    // identical stream every time it's reconstructed with the same seed.
    // torch::manual_seed() makes the identical claim about the real
    // generator, tested directly here rather than assumed.
    torch::manual_seed(42);
    torch::Tensor draw1 = torch::rand({4});
    torch::manual_seed(42);
    torch::Tensor draw2 = torch::rand({4});
    std::cout << "draw1 = " << draw1 << std::endl;
    std::cout << "draw2 (same seed, reconstructed) = " << draw2 << std::endl;
    std::cout << "identical seed produces identical stream? " << torch::equal(draw1, draw2) << std::endl;

    // Fisher-Yates: the CUDA book hand-writes the shuffle loop and checks
    // that element sum is preserved as its correctness invariant.
    // torch::randperm(n) IS a Fisher-Yates-family shuffle, real and
    // already implemented -- used here to permute indices into the CUDA
    // book's own example array [10,20,30,40,50].
    torch::manual_seed(7);
    torch::Tensor arr = torch::tensor({10, 20, 30, 40, 50});
    torch::Tensor perm = torch::randperm(5);
    torch::Tensor shuffled = arr.index_select(0, perm);
    std::cout << "original array = " << arr << std::endl;
    std::cout << "randperm(5) indices = " << perm << std::endl;
    std::cout << "shuffled array = " << shuffled << std::endl;
    int64_t sum_before = arr.sum().item<int64_t>();
    int64_t sum_after = shuffled.sum().item<int64_t>();
    std::cout << "sum before = " << sum_before << ", sum after = " << sum_after
              << ", invariant holds? " << (sum_before == sum_after) << std::endl;
    // A permutation must also be a genuine rearrangement: every original
    // value present exactly once, same as the CUDA book's own check.
    torch::Tensor sorted_shuffled = std::get<0>(torch::sort(shuffled));
    torch::Tensor sorted_original = std::get<0>(torch::sort(arr));
    std::cout << "shuffled array is a genuine permutation of the original? "
              << torch::equal(sorted_shuffled, sorted_original) << std::endl;

    // Inverse-CDF categorical sampling: the CUDA book hand-writes
    // sample_category() walking a cumulative-probability array, then
    // draws 10,000 samples from probabilities [0.2, 0.3, 0.5] and checks
    // empirical frequency against target. torch::multinomial() performs
    // exactly this algorithm, already implemented, given the probabilities
    // directly rather than a hand-built cumulative array.
    torch::manual_seed(2024);
    torch::Tensor probs = torch::tensor({0.2, 0.3, 0.5});
    int64_t n_draws = 10000;
    torch::Tensor samples = torch::multinomial(probs, n_draws, /*replacement=*/true);

    torch::Tensor counts = torch::zeros({3}, torch::kInt64);
    for (int64_t c = 0; c < 3; c++) {
        counts[c] = (samples == c).sum();
    }
    std::cout << "10000 draws from multinomial(probs=[0.2,0.3,0.5]):" << std::endl;
    for (int64_t c = 0; c < 3; c++) {
        double freq = counts[c].item<int64_t>() / (double)n_draws;
        double target = probs[c].item<double>();
        std::cout << "  category " << c << ": count=" << counts[c].item<int64_t>()
                  << ", frequency=" << freq << ", target=" << target
                  << ", |diff|=" << std::abs(freq - target) << std::endl;
    }
    bool all_close = true;
    for (int64_t c = 0; c < 3; c++) {
        double freq = counts[c].item<int64_t>() / (double)n_draws;
        double target = probs[c].item<double>();
        if (std::abs(freq - target) > 0.02) all_close = false;
    }
    std::cout << "all three empirical frequencies within 0.02 of target (matching the CUDA book's own convergence check)? "
              << all_close << std::endl;

    return 0;
}
```

### File: `03_csv_io.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cstdio>

// The CUDA C++ edition's Section 8.3 writes and reads CSV files holding
// tensor data using plain C file I/O (fopen/fprintf/fscanf) -- code with
// nothing GPU-specific about it, since it runs entirely on the host either
// way. This file reuses that same plain-file-I/O approach, but writes and
// reads a real torch::Tensor's own data via its accessor, and reproduces
// the CUDA book's own genuine off-by-one bug: a naive row counter that
// counts newline characters undercounts by one on a file whose last line
// has no trailing newline.
void write_csv(const char* filename, const torch::Tensor& t) {
    auto acc = t.accessor<float, 2>();
    FILE* f = fopen(filename, "w");
    for (int64_t r = 0; r < acc.size(0); r++) {
        for (int64_t c = 0; c < acc.size(1); c++) {
            fprintf(f, "%.1f", acc[r][c]);
            if (c < acc.size(1) - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// The CUDA book's own buggy row counter: counts '\n' characters only.
int count_rows_naive(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
    }
    fclose(f);
    return count;
}

// The CUDA book's own fix: also count a final, non-newline-terminated line.
int count_rows_fixed(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch, last_ch = '\n';
    bool saw_any_byte = false;
    while ((ch = fgetc(f)) != EOF) {
        saw_any_byte = true;
        if (ch == '\n') count++;
        last_ch = ch;
    }
    if (saw_any_byte && last_ch != '\n') count++;
    fclose(f);
    return count;
}

// Reads a CSV back into a flat buffer, up to max_values floats.
int read_csv(const char* filename, float* out, int max_values) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    while (count < max_values && fscanf(f, "%f", &out[count]) == 1) {
        count++;
        fgetc(f);  // consume ',' or '\n'
    }
    fclose(f);
    return count;
}

int main() {
    // Worked Example 8.3.2: round trip. Build a real torch::Tensor, write
    // it, read it back, compare against the original tensor's own values.
    torch::Tensor original = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    write_csv("/tmp/ch8_roundtrip.csv", original);

    float readback_buf[6];
    int n_read = read_csv("/tmp/ch8_roundtrip.csv", readback_buf, 6);
    torch::Tensor readback = torch::from_blob(readback_buf, {2, 3}, torch::kFloat32).clone();

    std::cout << "original =\n" << original << std::endl;
    std::cout << "n_read = " << n_read << std::endl;
    std::cout << "readback =\n" << readback << std::endl;
    std::cout << "round trip exact match? " << torch::equal(original, readback) << std::endl;

    // Worked Example 8.3.1: the off-by-one bug, triggered on a file
    // deliberately written WITHOUT a trailing newline.
    FILE* f = fopen("/tmp/ch8_with_newline.csv", "w");
    fprintf(f, "1,2\n3,4\n5,6\n");  // 3 rows, trailing newline present
    fclose(f);

    FILE* f2 = fopen("/tmp/ch8_no_newline.csv", "w");
    fprintf(f2, "1,2\n3,4\n5,6");  // 3 rows, NO trailing newline on the last one
    fclose(f2);

    int naive_with = count_rows_naive("/tmp/ch8_with_newline.csv");
    int fixed_with = count_rows_fixed("/tmp/ch8_with_newline.csv");
    int naive_without = count_rows_naive("/tmp/ch8_no_newline.csv");
    int fixed_without = count_rows_fixed("/tmp/ch8_no_newline.csv");

    std::cout << "\nfile WITH trailing newline (3 real rows):" << std::endl;
    std::cout << "  count_rows_naive = " << naive_with << " (expected 3, correct? " << (naive_with == 3) << ")" << std::endl;
    std::cout << "  count_rows_fixed = " << fixed_with << " (expected 3, correct? " << (fixed_with == 3) << ")" << std::endl;

    std::cout << "file WITHOUT trailing newline (still 3 real rows):" << std::endl;
    std::cout << "  count_rows_naive = " << naive_without << " (expected 3, BUG: undercounts? " << (naive_without != 3) << ")" << std::endl;
    std::cout << "  count_rows_fixed = " << fixed_without << " (expected 3, correct? " << (fixed_without == 3) << ")" << std::endl;

    return 0;
}
```

Files `01`-`03` all compile and link against LibTorch with the standard command from *Getting Started*:

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

`torch::arange` and `torch::linspace` reproduce the CUDA book's own published numbers exactly for its two worked examples, and the stop-exclusive-vs-inclusive distinction the CUDA book states in prose was confirmed as a directly tested fact, cross-checked against NumPy's independent implementations of both functions. `torch::eye` matches the CUDA book's ordinary identity exactly, and `torch::diag(values, offset)` -- a genuinely broader function than a `k`-parameter overload of `eye` -- reproduced the CUDA book's own superdiagonal example bit for bit; `torch::linspace(0, 1, 1)` was found to handle, without crashing, the exact `n=1` edge case the CUDA book's own hand-built kernel could not survive undefended. `torch::manual_seed()` and `torch::Tensor`'s real generator were confirmed to already deliver the CUDA book's own reproducibility guarantee, without ever needing the CUDA book's own hand-built `SimpleRNG` or its seed-mixing fix; `torch::randperm()` and `torch::multinomial()` were confirmed to already perform Fisher-Yates shuffling and inverse-CDF categorical sampling correctly, checked against the identical invariants the CUDA book itself uses -- element-sum preservation and empirical-frequency convergence, the latter validated against a hand-derived statistical bound rather than only an arbitrary threshold. And the CUDA book's own genuine off-by-one row-counting bug was confirmed to have nothing GPU-specific about it at all: it reproduced identically in this book's own plain C++ file I/O, alongside a clean, exact round trip of a real `torch::Tensor`'s own values through a CSV file and back.

## Self-Check Questions

1. Worked Example 8.1.1 tests `arange(0,10,2).eq(10.0f).any()` directly rather than just trusting the function's name to mean "stop excluded." What specific mistake would a reader make if they assumed `torch::arange`'s `stop` argument behaves like `torch::linspace`'s?
2. Section 8.1 uses `torch::diag(torch::ones({3}), 1)` to reproduce `eye(4, k=1)`. Why does the `ones` tensor passed to `torch::diag` need exactly 3 elements, not 4, to build a 4x4 matrix's superdiagonal?
3. Section 8.2's `[COMMON TRAP]` distinguishes "reproducibility" (a desired property) from "the small-seed bug" (a genuine defect). Using Worked Example 8.2.1's own `torch::manual_seed(42)` test, explain in your own words why an identical seed producing an identical stream is NOT, by itself, evidence of a bug.
4. Worked Example 8.2.1 checks two separate properties of `shuffled` against `arr`: that their sums match, and that their sorted versions match. Construct a hypothetical (not necessarily real) buggy shuffle function that would pass the sum check but fail the sorted-version check.
5. Section 8.3's `[COMMON TRAP]` argues that Worked Example 8.3.2's round trip alone could never expose the off-by-one bug from Worked Example 8.3.1. What specific property of `write_csv` (shown in the Complete Runnable Code) makes this true?

## Where We Go Next

This chapter confirmed `torch::Tensor`'s real factory and random-generation functions already deliver everything the CUDA book's own hand-built versions were built to provide -- and, in the `linspace(0,1,1)` and reproducible-generator cases, deliver it more robustly than a from-scratch implementation would without deliberate extra care. Chapter 9 turns to a question this chapter's factory functions have quietly depended on throughout: `torch::kFloat32` and `torch::kInt64` have appeared in nearly every worked example so far without examination, and the CUDA book's own specialized tensor types -- fixed-point and quantized representations built for scenarios where a full 32-bit float is more precision, or more memory, than a value needs -- get tested against `torch::Tensor`'s own real dtype family, including the ones this book has not yet used.

## Worked Solutions

**1.** A reader who assumed `torch::arange`'s `stop` behaves like `torch::linspace`'s would expect `arange(0, 10, 2)` to include `10` as its final element (the way `linspace(0, 1, 5)` includes `1` as its final element) and would therefore expect 6 elements, `[0,2,4,6,8,10]`, instead of the real 5, `[0,2,4,6,8]`. The mistake is assuming both functions share an inclusive-endpoint convention, when in fact only `linspace` guarantees the endpoint is reached — `arange` guarantees only that consecutive elements differ by exactly `step`, stopping at the last value strictly before `stop`.

**2.** A 4x4 matrix's superdiagonal (`k=1`) has exactly 3 valid positions — `(0,1)`, `(1,2)`, `(2,3)` — because moving one step above the main diagonal in an `n x n` matrix always yields `n-1` valid positions (the diagonal "runs out of room" one step earlier at each end). `torch::diag` requires its input vector's length to match the number of positions on the target diagonal exactly, so a 4x4 matrix's `k=1` diagonal needs a 3-element vector, not 4.

**3.** `torch::manual_seed(42)` followed by two separate `torch::rand({4})` calls producing bit-identical output is a *desired, correct* property being demonstrated on purpose — the test exists specifically to confirm the generator's state is fully determined by its seed, which is what makes seeding useful at all (for debugging, or for making an experiment's randomness reproducible on a second run). This differs from a "bug" because nothing here relies on two DIFFERENT seeds accidentally producing correlated output, or on a caller mistakenly reusing a seed when they actually wanted independent randomness — both draws here deliberately reuse the identical seed, and identical output is exactly what should happen.

**4.** A buggy shuffle that swapped only pairs of equal-valued elements, or performed a shuffle that always mapped index `i` to a fixed permutation regardless of `i`'s original value, could still satisfy sum preservation (since it's only rearranging existing values, never altering them) while producing a result that is NOT a probabilistically valid random permutation. More concretely: a "shuffle" that simply reverses the array, `[50,40,30,20,10]`, would pass BOTH the sum check and the sorted-version check while being deterministic rather than random — showing that even these two checks together cannot prove randomness, only that the output is some permutation of the input. A function that instead returned a fixed constant array summing to `150` but containing different values (e.g. `[30,30,30,30,30]`) would pass the sum check but fail the sorted-version check, since `[30,30,30,30,30]`'s sorted form does not match `[10,20,30,40,50]`'s.

**5.** `write_csv`, shown in the Complete Runnable Code, unconditionally calls `fprintf(f, "\n")` after every row, including the last one — so every file `write_csv` itself produces always has a trailing newline. Worked Example 8.3.1's second file, without a trailing newline, was constructed by a separate, hand-written `fprintf` call rather than through `write_csv` at all, specifically because `write_csv`'s own round-trip path structurally cannot produce the file shape the bug requires.
