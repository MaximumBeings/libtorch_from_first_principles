# Appendix D: C++ Functional and Lambda Programming

> The CUDA C++ edition's own Appendix D opens by extending an ordinary C++11 lambda across the host/device boundary: annotated `__device__` or `__host__ __device__`, compiled with `nvcc`'s own `--extended-lambda` flag, a lambda becomes callable from inside a `__global__` kernel -- and that extension comes with a specific, real set of restrictions (closure residence, reference-capture rejection, `std::function`'s own failure as a kernel parameter, a deduced-return-type restriction on composition helpers) that a raw CUDA kernel author must navigate explicitly. This appendix asks a different, more honest question: what happens to all of that machinery when the code in question never crosses a host/device boundary at all -- which is exactly the kind of code every chapter of this book has written, from Chapter 1 through Appendix C.

**What you will understand by the end of this appendix:** why CUDA's own extended-lambda mechanism -- the annotations, the compiler flag, the host/device closure-residence distinction -- has no analog anywhere in this book's own code, because no chapter has ever written a hand-rolled `__global__` kernel; why a generic host function templated on a callable type (`template<typename F>`) achieves CUDA's own generic-kernel pattern with no kernel launch at all, and why a lambda and an ordinary functor struct are, to the compiler, the identical kind of thing; why a generic reduction over an arbitrary binary operation is genuinely useful host-side C++, cross-checked here against the real `torch::Tensor` reduction ops this book's own chapters actually reach for; why `std::function` is a real, already-used, already-working pattern in this book (Chapter 19's own `Benchmark::time_function`, Appendix C's own `time_op_ms`), not a restriction to route around; why an `auto`-returning `compose(f, g)` -- which CUDA's own Section D.8 reports genuinely fails to compile as an extended device lambda -- compiles and runs with zero workaround in ordinary host C++; and why CUDA's own real, inconvenient restriction against reference captures in extended lambdas turns out, honestly, to PREVENT a real bug class (a dangling reference to a destroyed stack frame) that ordinary host C++ -- including every lambda this book has ever written -- remains fully exposed to, demonstrated here with a genuine AddressSanitizer catch.

**What you need to know first:** ordinary C++ lambda syntax (capture lists, `mutable`, generic/`auto`-parameter lambdas) at the level Chapter 1 onward has already used throughout this book; Chapter 19's own `Benchmark::time_function` helper; Appendix C's own `time_op_ms` helper and its own AddressSanitizer-adjacent honesty conventions (Section C.5's own first-line-only exception handling); this book's own established non-fabrication rule -- a genuine test result, however small or however different from the CUDA book's own reported figure, is reported as measured, never adjusted to look more dramatic.

## D.1 The Extended Lambda, and Why This Book Has Never Needed One `[FOUNDATIONAL]`

**Intuition.** CUDA's own extended lambda exists to solve one specific problem: an ordinary lambda's closure type is not, by default, usable from device code, so `nvcc` provides an opt-in mechanism -- `__device__`/`__host__ __device__` annotations plus the `--extended-lambda` compiler flag -- to make one usable there. Every chapter of this book, from Chapter 1's own first tensor operation through Appendix C's own thread-pool measurements, has written ordinary HOST-side LibTorch calls, never a hand-written `__global__` kernel -- so no lambda in this book has ever needed to cross that boundary, and the entire mechanism this section introduces has, correctly, zero analog in this book's own code.

**Background.** This is not a limitation of LibTorch as a library -- LibTorch's own internal CUDA kernels almost certainly use extended device lambdas extensively, the same way ATen's own C++ source does throughout. It is a direct, structural consequence of the level at which THIS BOOK's own code operates: a LibTorch programmer calling `torch::matmul`, `torch::relu`, or any other real op is calling into code someone else already wrote, compiled, and shipped -- never writing the kernel itself. An ordinary, unannotated lambda, used the way this book's own Chapter 19 (`Benchmark::time_function`) and Appendix C (`time_op_ms`) already have, is the entire lambda vocabulary this book's own code has ever required.

**Worked Example D.1.1.** An ordinary, unannotated lambda; a lambda capturing a real variable by reference (categorically rejected at compile time for a CUDA extended device lambda, per Section D.4's own restriction, covered later in this appendix); a lambda capturing a real `torch::Tensor` by reference alongside an ordinary float; and `std::function` wrapping and calling that same lambda -- all ordinary host C++, all compiled with the identical `g++` invocation every source file in this book has used.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <functional>

// The CUDA C++ edition's own Appendix D opens with CUDA's extended-
// lambda mechanism: an ordinary C++11 lambda, annotated __device__ or
// __host__ __device__ and compiled with nvcc's own --extended-lambda
// flag, becomes callable from inside a __global__ kernel -- along with
// a specific, real restriction its own Section D.2 demonstrates: a
// lambda's TYPE may be device-callable, but the closure OBJECT itself
// still lives in ordinary host memory unless it crosses the host/
// device boundary by being passed BY VALUE as a kernel argument, and
// nvcc genuinely rejects code that tries to reference a host-resident
// closure object directly from device code ("address of a host
// variable cannot be directly taken in a device function"). This
// entire mechanism -- the annotations, the flag, the host/device
// closure-residence distinction, the specific compile errors it
// produces -- exists to solve a problem that never arises anywhere in
// this book: every chapter has written ordinary host-side LibTorch
// calls, never a hand-written __global__ kernel, so no lambda in this
// book has ever needed to cross a host/device boundary at all. This
// file demonstrates the honest baseline every later section in this
// appendix builds on: an ordinary C++ lambda, used exactly the way
// every earlier chapter of this book already has (Chapter 19's own
// Benchmark::time_function, Appendix C's own time_op_ms), with zero
// annotations, zero compiler flags, and zero of CUDA's own
// restrictions to navigate.
int main() {
    // An ordinary, unannotated, plain C++ lambda -- no __device__, no
    // __host__ __device__, no --extended-lambda flag anywhere in this
    // file's own compile line (confirmed by this file's own successful
    // compile with an ordinary g++ invocation, the identical compile
    // line every chapter of this book has used).
    auto square = [](float x) { return x * x; };
    std::cout << "an ordinary, unannotated C++ lambda: square(5.0) = " << square(5.0f) << std::endl;

    // The closure object itself is just an ordinary stack object --
    // there is no separate "device residence" question to ask about
    // it at all, since nothing in this file ever crosses a host/device
    // boundary.
    float bias = 10.0f;
    auto add_bias = [&bias](float x) { return x + bias; };
    std::cout << "a lambda capturing a real variable BY REFERENCE (this alone is a genuine, "
              << "categorical compile-time REJECTION for a CUDA extended __device__/__host__ __device__ "
              << "lambda, per Section D.4's own reported restriction): add_bias(5.0) = " << add_bias(5.0f)
              << std::endl;
    bias = 100.0f;
    std::cout << "  bias changed to 100.0 after add_bias was created: add_bias(5.0) = " << add_bias(5.0f)
              << " (the reference capture genuinely sees the live value -- ordinary, unremarkable C++, "
              << "not something that needs to compile against any device-code restriction here at all)"
              << std::endl;

    // A real torch::Tensor is an entirely ordinary object as far as
    // lambda capture is concerned -- no different from `bias` above.
    torch::Tensor t = torch::tensor({1.0, 2.0, 3.0});
    auto sum_plus_bias = [&t, &bias]() { return t.sum().item<float>() + bias; };
    std::cout << "\na lambda capturing a real torch::Tensor by reference alongside an ordinary float, "
              << "used exactly the way ordinary host LibTorch code already does throughout this book: "
              << "sum_plus_bias() = " << sum_plus_bias() << std::endl;

    // std::function works completely normally here -- Section D.7's
    // own reported failure is specific to CROSSING the host/device
    // boundary (a __global__ kernel parameter), which this file never
    // attempts.
    std::function<float(float)> as_std_function = square;
    std::cout << "\nsquare wrapped in std::function<float(float)> and called through that wrapper: "
              << as_std_function(6.0f) << " (std::function works completely normally in ordinary host "
              << "code -- Section D.4 of this appendix returns to exactly why it does NOT work as a "
              << "kernel parameter, and why that distinction matters not at all to LibTorch code)"
              << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
an ordinary, unannotated C++ lambda: square(5.0) = 25
a lambda capturing a real variable BY REFERENCE (this alone is a genuine, categorical compile-time REJECTION for a CUDA extended __device__/__host__ __device__ lambda, per Section D.4's own reported restriction): add_bias(5.0) = 15
  bias changed to 100.0 after add_bias was created: add_bias(5.0) = 105 (the reference capture genuinely sees the live value -- ordinary, unremarkable C++, not something that needs to compile against any device-code restriction here at all)

a lambda capturing a real torch::Tensor by reference alongside an ordinary float, used exactly the way ordinary host LibTorch code already does throughout this book: sum_plus_bias() = 106

square wrapped in std::function<float(float)> and called through that wrapper: 36 (std::function works completely normally in ordinary host code -- Section D.4 of this appendix returns to exactly why it does NOT work as a kernel parameter, and why that distinction matters not at all to LibTorch code)
```

**Discussion.** Every line of this file's own output is, individually, unremarkable C++ behavior -- a reference capture sees the current value of what it captured, `std::function` wraps and calls a lambda normally. What makes it worth stating explicitly, as this appendix's own opening section, is the CONTRAST: every one of these constructs is either restricted, rejected outright, or requires special handling under CUDA's own extended-lambda rules the moment it crosses into device code, and none of that machinery -- the annotations, the flag, the specific compile errors -- has ever appeared, or ever needed to appear, anywhere in this book's own thirty-plus source files across twenty-two chapters and four appendices so far.

## D.2 Generic Functions Parameterized by a Callable, and Functor/Lambda Equivalence `[FOUNDATIONAL]`

**Intuition.** CUDA's own Section D.3 writes a generic `transform_kernel<F>`, templated on a callable type, launched three times with three different device lambdas (square, cube, a capturing scale-by-k lambda) -- one kernel body, specialized at compile time by whatever callable is handed to it. Its own Section D.5 then shows the identical technique works with an ordinary functor struct in place of a lambda, indistinguishably, because a lambda IS a compiler-synthesized functor. Ordinary host C++ has this exact same generic-programming technique available with no kernel-launch machinery at all: a plain `template<typename F>` function.

**Background.** This section merges CUDA's own Sections D.3 and D.5 into one host-side demonstration, since the generic-programming TECHNIQUE they describe is identical C++ template machinery in both cases -- only the launch mechanism (a `__global__` kernel versus an ordinary function call) differs, and that difference has already been addressed directly by Section D.1.

**Worked Example D.2.1.** A generic `transform_all<F>` host function applied to a real `std::vector<float>` with three different lambdas (square, cube, a capturing scale-by-10 lambda) and then with an ordinary `ScaleFunctor` struct, confirming the lambda-based and functor-based results agree exactly; then a cross-check of the squared values against a real `torch::Tensor::pow(2)` call.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.3 writes a generic __global__
// kernel, transform_kernel<F>, templated on a callable type F, and
// launches it three times with three different device lambdas
// (square, cube, a capturing "scale by k" lambda) -- demonstrating
// that a single generic kernel body can be specialized, at compile
// time, by whatever callable is handed to it. Its own Section D.5
// then shows the identical technique works with an ordinary functor
// struct (one with operator()) in place of a lambda, and that from
// the compiler's point of view the two are indistinguishable: a
// lambda IS a compiler-synthesized functor, nothing more.
//
// Ordinary host-side C++ has exactly this same generic-programming
// technique available, with no device/kernel machinery required at
// all -- a plain template<typename F> function, applied here to a
// real std::vector<float> the way Chapter 1 through Appendix C's own
// helper functions have used ordinary host containers throughout.

// A generic host-side "transform," templated on the callable type F,
// exactly mirroring the CUDA book's own transform_kernel<F> in spirit
// -- one function body, specialized at compile time by whatever F is
// handed to it.
template <typename F>
std::vector<float> transform_all(const std::vector<float>& in, F f) {
    std::vector<float> out;
    out.reserve(in.size());
    for (float x : in) {
        out.push_back(f(x));
    }
    return out;
}

// An ordinary functor struct -- the CUDA book's own Section D.5
// equivalent. From transform_all<F>'s own point of view, below, this
// is indistinguishable from a lambda: both are just callable types
// with an operator()(float) const.
struct ScaleFunctor {
    float k;
    float operator()(float x) const { return x * k; }
};

static void print_vec(const char* label, const std::vector<float>& v) {
    std::cout << label << ": ";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    print_vec("input", data);

    // Same three specializations the CUDA book's own Section D.3
    // demonstrates -- square, cube, and a capturing "scale by k"
    // lambda -- but as one ordinary template<typename F> host
    // function, called three times with three different callables,
    // never a __global__ kernel launch anywhere in this file.
    auto squared = transform_all(data, [](float x) { return x * x; });
    print_vec("squared (lambda [](float x){ return x*x; })", squared);

    auto cubed = transform_all(data, [](float x) { return x * x * x; });
    print_vec("cubed (lambda [](float x){ return x*x*x; })", cubed);

    float k = 10.0f;
    auto scaled_lambda = transform_all(data, [k](float x) { return x * k; });
    print_vec("scaled by 10 (capturing lambda [k](float x){ return x*k; })", scaled_lambda);

    // The functor version: the CUDA book's own Section D.5 point,
    // reproduced here verbatim -- transform_all<ScaleFunctor> and
    // transform_all<lambda-closure-type> are two different template
    // INSTANTIATIONS of the identical function body, because
    // ScaleFunctor and the lambda's own compiler-synthesized closure
    // type both satisfy exactly the same requirement: an
    // operator()(float) const.
    ScaleFunctor scale_functor{10.0f};
    auto scaled_functor = transform_all(data, scale_functor);
    print_vec("scaled by 10 (ScaleFunctor{10.0f}, an ordinary functor struct)", scaled_functor);

    std::cout << "\nscaled_lambda == scaled_functor element-wise? "
              << (scaled_lambda == scaled_functor ? "true" : "false")
              << " -- transform_all<F> instantiated once for the lambda's own compiler-synthesized "
              << "closure type and once for ScaleFunctor, producing identical results from what is, "
              << "to the compiler, the same generic-programming pattern the CUDA book's own "
              << "transform_kernel<F> uses -- with no __global__, no --extended-lambda flag, and no "
              << "host/device closure-residence question anywhere in this file." << std::endl;

    // Applying the identical technique to a real torch::Tensor, tying
    // it back to this book's own domain: torch::Tensor itself already
    // exposes plenty of built-in elementwise ops, but nothing stops an
    // ordinary generic host function from being handed one lambda per
    // call site exactly as transform_all<F> is above.
    torch::Tensor t = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0});
    torch::Tensor t_squared = t.pow(2);
    std::cout << "\ncross-check against a real torch::Tensor: t = " << t
              << ", t.pow(2) = " << t_squared
              << " -- matches the lambda-based 'squared' vector above element-wise? "
              << torch::allclose(t_squared, torch::tensor(squared)) << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
input: 1, 2, 3, 4, 5
squared (lambda [](float x){ return x*x; }): 1, 4, 9, 16, 25
cubed (lambda [](float x){ return x*x*x; }): 1, 8, 27, 64, 125
scaled by 10 (capturing lambda [k](float x){ return x*k; }): 10, 20, 30, 40, 50
scaled by 10 (ScaleFunctor{10.0f}, an ordinary functor struct): 10, 20, 30, 40, 50

scaled_lambda == scaled_functor element-wise? true -- transform_all<F> instantiated once for the lambda's own compiler-synthesized closure type and once for ScaleFunctor, producing identical results from what is, to the compiler, the same generic-programming pattern the CUDA book's own transform_kernel<F> uses -- with no __global__, no --extended-lambda flag, and no host/device closure-residence question anywhere in this file.

cross-check against a real torch::Tensor: t =  1
 2
 3
 4
 5
[ CPUFloatType{5} ], t.pow(2) =   1
  4
  9
 16
 25
[ CPUFloatType{5} ] -- matches the lambda-based 'squared' vector above element-wise? 1
```

**Discussion.** The genuinely compiled output confirms the exact structural point Sections D.3 and D.5 each make in the CUDA C++ edition: `transform_all<F>` is instantiated once for the lambda's own compiler-synthesized closure type and once for `ScaleFunctor`, and the two instantiations produce identical results, because both satisfy the same requirement (`operator()(float) const`) that the template body actually depends on. The `torch::Tensor::pow(2)` cross-check makes the same point from a different direction: LibTorch's own built-in tensor ops already cover the specific "square every element" case this file's own `transform_all<F>` demonstrates generically -- the generic technique remains genuinely useful for an operation with no built-in tensor equivalent, but this book's own chapters have, correctly, reached for the built-in op wherever one already existed.

## D.3 A Generic Reduction, Cross-Checked Against Real `torch::Tensor` Ops `[FOUNDATIONAL]`

**Intuition.** CUDA's own Section D.6 writes a generic `reduce_kernel<T, Op>`, run over the fixed input `{3, 7, 1, 9, 4, 2, 8, 5}` with four different binary-operation lambdas (sum, max, min, product), reporting Sum=39.0, Max=9.0, Min=1.0, Product=60480.0. The generic-reduction TECHNIQUE is the identical kind of `template<typename T, typename Op>` device this appendix's own Section D.2 already demonstrated -- so rather than re-deriving that point a second time, this section reproduces the CUDA book's own exact reduction values as a correctness check, then cross-checks them against the real `torch::Tensor` reduction ops this book's own chapters have actually used throughout (Chapter 22's own portfolio `.sum()`, among others).

**Background.** A host-side `reduce_all<T, Op>`, structurally identical to Section D.2's own `transform_all<F>`, folds a `std::vector<T>` down to one value via whatever binary operation `Op` is handed to it -- genuinely useful for an arbitrary reduction no built-in tensor method covers, but redundant with real `torch::Tensor` methods for the four common cases (sum, max, min, product) this section actually exercises.

**Worked Example D.3.1.** `reduce_all<double>` over the CUDA book's own exact fixed input, with sum/max/min/product lambdas, cross-checked element-for-element against `torch::tensor(data).sum()/.max()/.min()/.prod()`.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.6 writes a generic
// reduce_kernel<T, Op>, templated on both the element type and a
// binary-operation callable, and runs it over the fixed input
// {3, 7, 1, 9, 4, 2, 8, 5} with four different operations -- a sum
// lambda, a max lambda, a min lambda, and a product lambda -- getting
// Sum=39.0, Max=9.0, Min=1.0, Product=60480.0. The generic-reduction
// TECHNIQUE (a function templated on a binary-operation callable) is
// ordinary C++, identical in spirit to Section D.2's own
// transform_all<F> in this appendix -- so rather than re-deriving
// that same point a second time, this file reproduces the CUDA
// book's own exact reduction values as a correctness check, and then
// cross-checks them against real torch::Tensor reduction ops
// (.sum(), .max(), .min(), .prod()), the ones this book's own
// chapters have actually used throughout (Chapter 3's own reductions,
// Chapter 22's own portfolio sums) instead of a hand-written
// reduce_kernel at all.
template <typename T, typename Op>
T reduce_all(const std::vector<T>& data, T init, Op op) {
    T acc = init;
    for (T x : data) {
        acc = op(acc, x);
    }
    return acc;
}

int main() {
    // The CUDA book's own exact fixed input.
    std::vector<double> data = {3, 7, 1, 9, 4, 2, 8, 5};

    double sum_result = reduce_all<double>(data, 0.0, [](double a, double b) { return a + b; });
    double max_result = reduce_all<double>(data, data[0], [](double a, double b) { return a > b ? a : b; });
    double min_result = reduce_all<double>(data, data[0], [](double a, double b) { return a < b ? a : b; });
    double product_result = reduce_all<double>(data, 1.0, [](double a, double b) { return a * b; });

    std::cout << "reduce_all<double>(data, op) over {3, 7, 1, 9, 4, 2, 8, 5}, four different binary-op "
              << "lambdas handed to the identical generic reduce_all<T, Op> body (the CUDA book's own "
              << "Section D.6 pattern, here as an ordinary host template<T, Op> function, no "
              << "reduce_kernel launch anywhere):" << std::endl;
    std::cout << "  Sum      = " << sum_result << " (CUDA book's own expected 39.0)" << std::endl;
    std::cout << "  Max      = " << max_result << " (expected 9.0)" << std::endl;
    std::cout << "  Min      = " << min_result << " (expected 1.0)" << std::endl;
    std::cout << "  Product  = " << product_result << " (expected 60480.0)" << std::endl;

    // The cross-check this file adds beyond the CUDA book's own
    // section: the SAME data, reduced via real torch::Tensor
    // reduction ops -- the ones this book's own chapters have
    // actually called throughout -- rather than any hand-written
    // reduce_all<T, Op> at all.
    torch::Tensor t = torch::tensor(data, torch::kFloat64);
    double t_sum = t.sum().item<double>();
    double t_max = t.max().item<double>();
    double t_min = t.min().item<double>();
    double t_prod = t.prod().item<double>();

    std::cout << "\ncross-check via real torch::Tensor reduction ops on the identical data (torch::tensor"
              << "(data).sum()/.max()/.min()/.prod() -- the actual API this book's own chapters have used, "
              << "e.g. Chapter 22's own portfolio-total .sum()):" << std::endl;
    std::cout << "  t.sum()  = " << t_sum << (t_sum == sum_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.max()  = " << t_max << (t_max == max_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.min()  = " << t_min << (t_min == min_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.prod() = " << t_prod << (t_prod == product_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;

    std::cout << "\nthe generic reduce_all<T, Op> template above is genuinely useful for an ARBITRARY "
              << "binary operation no built-in torch::Tensor method covers -- but for the four common "
              << "reductions this book's own chapters actually reach for, real torch::Tensor methods "
              << "already exist, are already tensor-shape- and dtype-aware, and (unlike this file's own "
              << "serial reduce_all loop) already dispatch to ATen's own parallel reduction kernels "
              << "underneath, with no hand-written reduce_kernel, no block/grid launch configuration, "
              << "and no shared-memory reduction tree for a programmer to get wrong." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
reduce_all<double>(data, op) over {3, 7, 1, 9, 4, 2, 8, 5}, four different binary-op lambdas handed to the identical generic reduce_all<T, Op> body (the CUDA book's own Section D.6 pattern, here as an ordinary host template<T, Op> function, no reduce_kernel launch anywhere):
  Sum      = 39 (CUDA book's own expected 39.0)
  Max      = 9 (expected 9.0)
  Min      = 1 (expected 1.0)
  Product  = 60480 (expected 60480.0)

cross-check via real torch::Tensor reduction ops on the identical data (torch::tensor(data).sum()/.max()/.min()/.prod() -- the actual API this book's own chapters have used, e.g. Chapter 22's own portfolio-total .sum()):
  t.sum()  = 39  (exact match)
  t.max()  = 9  (exact match)
  t.min()  = 1  (exact match)
  t.prod() = 60480  (exact match)

the generic reduce_all<T, Op> template above is genuinely useful for an ARBITRARY binary operation no built-in torch::Tensor method covers -- but for the four common reductions this book's own chapters actually reach for, real torch::Tensor methods already exist, are already tensor-shape- and dtype-aware, and (unlike this file's own serial reduce_all loop) already dispatch to ATen's own parallel reduction kernels underneath, with no hand-written reduce_kernel, no block/grid launch configuration, and no shared-memory reduction tree for a programmer to get wrong.
```

**Discussion.** All four values match the CUDA book's own reported figures exactly (Sum=39, Max=9, Min=1, Product=60480), as expected for deterministic, RNG-free integer arithmetic reduced two structurally different ways. The real `torch::Tensor` cross-check is this section's own addition beyond the CUDA book's own text, and it makes a point in the same spirit as Section D.2's own `torch::Tensor::pow(2)` comparison: a hand-written generic reduction is a genuinely useful TECHNIQUE, but for the specific reductions this book's own chapters actually reach for, a real `torch::Tensor` method already exists, is already shape- and dtype-aware, and already dispatches to ATen's own parallel reduction kernels underneath -- with no hand-written reduction loop, and no shared-memory reduction tree, for a programmer to get wrong.

## D.4 `std::function`: A Real, Documented Non-Failure `[FOUNDATIONAL]`

**Intuition.** CUDA's own Section D.7 reports a genuine, real `nvcc` compile failure: passing a `std::function<...>` as a `__global__` kernel PARAMETER fails, because `std::function`'s own heap-allocated, host-resident, vtable-driven dispatch has no meaning inside device code. That restriction is specific to crossing the host/device boundary as a kernel argument -- it says nothing about `std::function` used in ordinary host code, which is exactly how this book's own Chapter 19 (`Benchmark::time_function`) and Appendix C (`time_op_ms`) have already used it, more than once, without incident.

**Background.** This section treats that existing precedent as the substance of the point, rather than inventing a new example from scratch: `std::function<void()>` wrapping a real `torch::matmul` call, passed by reference into an ordinary host `time_op_ms`-style helper, is literally the same shape this book has already compiled and run successfully in two earlier chapters.

**Worked Example D.4.1.** `std::function<void()>` wrapping a real `torch::matmul(a, b)` call, timed via an ordinary host `time_op_ms` helper; then a `std::vector<std::function<int(int)>>` holding three different callables, called uniformly.

```cpp
#include <torch/torch.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.7 reports a genuine, real
// nvcc compile failure: passing a std::function<...> as a __global__
// kernel PARAMETER fails, because std::function's own virtual-dispatch
// machinery (a heap-allocated, host-resident vtable-driven call) has
// no meaning inside device code, which cannot perform host-side
// virtual dispatch at all. That restriction is specific to crossing
// the host/device boundary as a kernel argument -- it says nothing
// about std::function used in ordinary host code, which is exactly
// how this book's own chapters have already used it, more than once,
// without incident:
//
//   - Chapter 19's own Benchmark::time_function(std::function<void()>)
//     helper wraps an arbitrary block of code to be timed.
//   - Appendix C's own time_op_ms(std::function<void()>) helper does
//     the identical thing for that appendix's own six timing
//     measurements (C.3, C.5, C.6).
//
// Both are real, already-compiled, already-verified precedent
// elsewhere in this book: std::function has never once failed here,
// because neither helper is ever called from device code or passed
// as a kernel argument. This file reproduces that same
// std::function<void()>-wrapping-arbitrary-work pattern directly, as
// a documented non-failure in its own right, rather than treating
// D.7's own restriction as something LibTorch code must somehow work
// around.

// The same shape as Chapter 19's Benchmark::time_function and
// Appendix C's time_op_ms: an ordinary host function taking a
// std::function<void()> parameter, timing whatever it wraps.
double time_op_ms(const std::function<void()>& op, int reps) {
    // warm-up, exactly as Appendix C's own time_op_ms does
    op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) {
        op();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    // A std::function<void()> wrapping a real torch::Tensor
    // computation -- passed by ordinary reference into an ordinary
    // host function, precisely the way Chapter 19 and Appendix C
    // already do it.
    torch::Tensor a = torch::randn({500, 500});
    torch::Tensor b = torch::randn({500, 500});
    std::function<void()> matmul_op = [&a, &b]() { torch::matmul(a, b); };

    double ms = time_op_ms(matmul_op, 20);
    std::cout << "std::function<void()> wrapping torch::matmul(a, b) (500x500), passed by reference into "
              << "an ordinary host time_op_ms(const std::function<void()>&, int) helper -- the same "
              << "pattern Chapter 19's Benchmark::time_function and Appendix C's time_op_ms already use "
              << "throughout this book: compiled and ran with zero issues." << std::endl;
    std::cout << "[TIMING] mean time per call over 20 reps: " << ms << " ms" << std::endl;

    // A vector of std::function<int(int)>, each wrapping a different
    // callable (a plain lambda, a capturing lambda, and a named
    // ordinary function) -- ordinary host-side polymorphism over
    // callables, ordinary heap-allocated type erasure, nothing here
    // needs to be callable from device code at all.
    auto double_it = [](int x) { return x * 2; };
    int offset = 100;
    auto add_offset = [offset](int x) { return x + offset; };
    std::function<int(int)> square_fn = [](int x) { return x * x; };

    std::vector<std::function<int(int)>> ops = {double_it, add_offset, square_fn};
    std::cout << "\na std::vector<std::function<int(int)>> holding three different callables (a plain "
              << "lambda, a capturing lambda, and a lambda already stored in a named std::function) -- "
              << "ordinary host-side runtime polymorphism over callables, applied to x = 7:" << std::endl;
    for (size_t i = 0; i < ops.size(); i++) {
        std::cout << "  ops[" << i << "](7) = " << ops[i](7) << std::endl;
    }

    std::cout << "\nnone of this required anything special: std::function's own vtable-driven, "
              << "heap-allocated dispatch mechanism is exactly what ordinary host C++ is built to run. "
              << "Section D.7 of the CUDA C++ edition's own restriction is specific to a __global__ "
              << "kernel PARAMETER -- host code calling std::function::operator() is the ordinary case "
              << "this mechanism was designed for, and this book's own Chapter 19 and Appendix C have "
              << "already relied on exactly that, more than once, before this appendix ever raised the "
              << "question." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
std::function<void()> wrapping torch::matmul(a, b) (500x500), passed by reference into an ordinary host time_op_ms(const std::function<void()>&, int) helper -- the same pattern Chapter 19's Benchmark::time_function and Appendix C's time_op_ms already use throughout this book: compiled and ran with zero issues.
[TIMING] mean time per call over 20 reps: 1.97501 ms

a std::vector<std::function<int(int)>> holding three different callables (a plain lambda, a capturing lambda, and a lambda already stored in a named std::function) -- ordinary host-side runtime polymorphism over callables, applied to x = 7:
  ops[0](7) = 14
  ops[1](7) = 107
  ops[2](7) = 49

none of this required anything special: std::function's own vtable-driven, heap-allocated dispatch mechanism is exactly what ordinary host C++ is built to run. Section D.7 of the CUDA C++ edition's own restriction is specific to a __global__ kernel PARAMETER -- host code calling std::function::operator() is the ordinary case this mechanism was designed for, and this book's own Chapter 19 and Appendix C have already relied on exactly that, more than once, before this appendix ever raised the question.
```

**Discussion.** Nothing in this file's own output required any special handling: `std::function`'s own vtable-driven dispatch is exactly the mechanism ordinary host C++ is built to run, and this book's own Chapter 19 and Appendix C already relied on it before this appendix ever raised the question. Section D.7's own restriction, in the CUDA C++ edition, is specific to a `__global__` kernel parameter -- an argument that must be usable from DEVICE code, where `std::function`'s own host-resident vtable cannot be dereferenced at all. Ordinary host code calling `std::function::operator()` is the case the mechanism was designed for in the first place, and it is the only case this book's own code has ever needed.

## D.5 Composing Lambdas, With No Functor Workaround Needed `[FOUNDATIONAL]`

**Intuition.** CUDA's own Section D.8 tries the obvious thing first -- an `auto`-returning `compose(f, g)` template -- and reports it genuinely fails to compile as an extended device lambda, with a real, specific error: the enclosing function has a deduced return type, which its own device-lambda compilation model rejects. Its own fix abandons `auto` entirely for a hand-written `Composed<F, G>` functor struct. Ordinary host C++ has no such restriction: the literal `auto`-returning `compose(f, g)` CUDA's own first attempt needed simply compiles and runs.

**Background.** This restriction is specific to how CUDA's own device-lambda compilation model interacts with return-type deduction across a device-lambda boundary -- a structural consequence of the SAME host/device-crossing machinery Section D.1 already introduced, not a general limitation of C++ templates or lambdas.

**Worked Example D.5.1.** An `auto`-returning `template<typename F, typename G> auto compose(F f, G g)`, applied to the CUDA book's own exact pipeline (`compose(square, compose(add_one, scale2))`, computing `(2x+1)^2` for x=1..5), plus a second, independent composition chain to confirm the technique genuinely generalizes.

```cpp
#include <iostream>

// The CUDA C++ edition's own Section D.8 tries the obvious thing
// first -- an auto-returning compose(f, g) template -- and reports it
// genuinely fails to compile, with a real, specific error: the
// enclosing function (compose itself) has a deduced return type
// (auto), and a nested lambda defined INSIDE compose is not allowed
// to be part of that deduction in the way its own first attempt
// needed. Its own fix is to abandon auto entirely and hand-write an
// explicit Composed<F, G> functor struct instead, storing f and g as
// member data and implementing operator() by hand.
//
// This restriction is specific to how CUDA's own device-lambda
// compilation model interacts with return-type deduction across a
// device-lambda boundary -- ordinary host C++ has no such
// restriction at all. This file writes the exact auto-returning
// compose(f, g) the CUDA book's own first attempt needed, and it
// simply works, with no Composed<F, G> functor workaround required.
template <typename F, typename G>
auto compose(F f, G g) {
    // An ordinary lambda, defined inside an auto-returning function,
    // capturing f and g by value -- exactly the construct CUDA's own
    // Section D.8 reports failing to compile as an extended device
    // lambda. Ordinary host C++ imposes no such restriction: this
    // compiles and works exactly as an unremarkable programmer would
    // expect it to.
    return [f, g](auto x) { return f(g(x)); };
}

int main() {
    // The CUDA book's own exact worked pipeline: square(add_one(scale2(x))),
    // i.e. compose(square, compose(add_one, scale2)), which for input x
    // computes ((2x) + 1)^2.
    auto square = [](auto x) { return x * x; };
    auto add_one = [](auto x) { return x + 1; };
    auto scale2 = [](auto x) { return x * 2; };

    auto pipeline = compose(square, compose(add_one, scale2));

    std::cout << "compose(f, g) := an ordinary auto-returning template<F,G> function returning a lambda "
              << "that computes f(g(x)) -- the CUDA book's own Section D.8 reports this exact construct "
              << "genuinely FAILS to compile as an extended __device__ lambda ('the enclosing function ... "
              << "must not have a deduced return type'), forcing a hand-written Composed<F,G> functor "
              << "workaround instead. Ordinary host C++, as used throughout this book, has no such "
              << "restriction -- this file's own compose(f, g) is the literal auto-returning version, "
              << "compiled and run with zero workaround." << std::endl;

    std::cout << "\npipeline := compose(square, compose(add_one, scale2)), i.e. pipeline(x) = "
              << "((2*x) + 1)^2, applied to x = 1..5:" << std::endl;
    for (int x = 1; x <= 5; x++) {
        int computed = pipeline(x);
        int expected = (2 * x + 1) * (2 * x + 1);
        std::cout << "  pipeline(" << x << ") = " << computed
                   << ", hand-expanded (2*" << x << "+1)^2 = " << expected
                   << (computed == expected ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    }

    // A second, independent composition chain to further exercise the
    // same compose(f, g), confirming it genuinely generalizes rather
    // than only working for the one CUDA-book pipeline above: a
    // three-stage chain computing sqrt-free integer work, cube(x-2).
    auto cube = [](auto x) { return x * x * x; };
    auto sub_two = [](auto x) { return x - 2; };
    auto pipeline2 = compose(cube, sub_two);
    std::cout << "\na second, independent chain, pipeline2 := compose(cube, sub_two), i.e. "
              << "pipeline2(x) = (x-2)^3, applied to x = 0..4:" << std::endl;
    for (int x = 0; x <= 4; x++) {
        int computed = pipeline2(x);
        int expected = (x - 2) * (x - 2) * (x - 2);
        std::cout << "  pipeline2(" << x << ") = " << computed
                   << ", hand-expanded (" << x << "-2)^3 = " << expected
                   << (computed == expected ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run:

```text
compose(f, g) := an ordinary auto-returning template<F,G> function returning a lambda that computes f(g(x)) -- the CUDA book's own Section D.8 reports this exact construct genuinely FAILS to compile as an extended __device__ lambda ('the enclosing function ... must not have a deduced return type'), forcing a hand-written Composed<F,G> functor workaround instead. Ordinary host C++, as used throughout this book, has no such restriction -- this file's own compose(f, g) is the literal auto-returning version, compiled and run with zero workaround.

pipeline := compose(square, compose(add_one, scale2)), i.e. pipeline(x) = ((2*x) + 1)^2, applied to x = 1..5:
  pipeline(1) = 9, hand-expanded (2*1+1)^2 = 9  (exact match)
  pipeline(2) = 25, hand-expanded (2*2+1)^2 = 25  (exact match)
  pipeline(3) = 49, hand-expanded (2*3+1)^2 = 49  (exact match)
  pipeline(4) = 81, hand-expanded (2*4+1)^2 = 81  (exact match)
  pipeline(5) = 121, hand-expanded (2*5+1)^2 = 121  (exact match)

a second, independent chain, pipeline2 := compose(cube, sub_two), i.e. pipeline2(x) = (x-2)^3, applied to x = 0..4:
  pipeline2(0) = -8, hand-expanded (0-2)^3 = -8  (exact match)
  pipeline2(1) = -1, hand-expanded (1-2)^3 = -1  (exact match)
  pipeline2(2) = 0, hand-expanded (2-2)^3 = 0  (exact match)
  pipeline2(3) = 1, hand-expanded (3-2)^3 = 1  (exact match)
  pipeline2(4) = 8, hand-expanded (4-2)^3 = 8  (exact match)
```

**Discussion.** Every value matches its own independently hand-expanded formula exactly -- 9, 25, 49, 81, 121 for the CUDA book's own `(2x+1)^2` pipeline, and a second, independently verified `(x-2)^3` chain confirming the result is not specific to that one example. The point this section is making is not that `compose(f, g)` is a clever trick -- it is an entirely ordinary generic-programming pattern -- but that CUDA's own Section D.8 reports it FAILING for a reason (deduced-return-type interaction with device-lambda compilation) that simply does not exist in ordinary host C++, so the workaround its own text needs (a hand-written `Composed<F, G>` functor, with its own explicit member storage and hand-written `operator()`) is, for this book's own code, unnecessary machinery solving a problem this book's own code never has.

## D.6 Capture by Value vs. Capture by Reference, and the Bug CUDA's Own Restriction Prevents `[FOUNDATIONAL]`

**Intuition.** CUDA's own Section D.9 (plain, CUDA-free C++, with no `nvcc`-specific restriction anywhere in it) covers capture-by-value snapshot semantics, capture-by-value mutation via `mutable`, capture-by-reference live-value semantics, default capture modes, and a dangling-reference-capture bug caught by AddressSanitizer. This section reproduces all of that directly, then connects it back to Section D.4's own CUDA restriction (rejecting reference captures in extended lambdas) from a different angle than the earlier sections of this appendix: not as an inconvenience with no analog here, but as a real compile-time PROTECTION against a bug class ordinary host C++ -- including every lambda this book has ever written -- remains fully exposed to.

**Background.** `make_dangling_closure()` returns a `std::function<int()>` that captured a LOCAL variable by reference; the instant the function returns, that reference dangles, and calling the returned closure is undefined behavior. This file's own default run (no arguments, the ordinary compile line every source file in this book has used) skips triggering that call, so its own output stays fully deterministic; a second, separate build of the SAME source, compiled with `-fsanitize=address` and invoked with `--trigger-dangling`, genuinely triggers it, and AddressSanitizer genuinely catches a real `stack-use-after-return` violation, aborting the process -- confirmed directly via Bash before being written into this appendix, following the identical dual-verification approach Chapter 21.4's own dual-build source file established.

**Worked Example D.6.1.** Snapshot-vs-live capture, mutable-by-value-copy vs. reference mutation, default capture modes `[=]`/`[&]`, and a real `torch::Tensor` captured and mutated by reference across calls -- all from one ordinary run of the compiled binary.

```cpp
#include <torch/torch.h>
#include <cstring>
#include <functional>
#include <iostream>

// The CUDA C++ edition's own Section D.9 (plain, CUDA-free C++, no
// nvcc-specific restriction anywhere in it) covers capture-by-value
// snapshot semantics, capture-by-value mutation via `mutable`,
// capture-by-reference live-value semantics, and default capture
// modes [=]/[&] -- ordinary C++ lambda behavior that applies to this
// book's own code exactly as it does to the CUDA book's own host-side
// examples. This file reproduces all of that directly (Sections 1-4
// below), then goes one step further than the CUDA book's own D.9.4:
// where D.9.4 shows a dangling-reference-capture bug and its
// AddressSanitizer report as a cautionary example, this file connects
// it back to Section D.4 of THIS appendix (the reference-capture
// restriction on CUDA's own extended __device__/__host__ __device__
// lambdas) -- reframing that restriction as something that, however
// inconvenient for a raw CUDA kernel author, categorically PREVENTS
// this exact bug class at compile time, a protection ordinary host
// C++ -- including every lambda in every earlier chapter of this
// book -- does not have.

// --- Section 1: snapshot (by value) vs live (by reference) capture ---
static void section1_snapshot_vs_live() {
    int counter = 10;
    auto snapshot = [counter]() { return counter; };  // captures a COPY, frozen at creation
    auto live = [&counter]() { return counter; };      // captures a REFERENCE, always current

    std::cout << "Section 1 -- snapshot (by value) vs live (by reference) capture:" << std::endl;
    std::cout << "  counter = 10 initially. snapshot() = " << snapshot() << ", live() = " << live() << std::endl;
    counter = 99;
    std::cout << "  counter set to 99. snapshot() = " << snapshot()
              << " (frozen at capture time, unaffected by the later mutation), live() = " << live()
              << " (sees the current value, because it holds a reference to counter itself)" << std::endl;
}

// --- Section 2: mutable-by-value copy vs reference mutation ---
static void section2_mutable_and_reference_mutation() {
    int x = 5;

    // Capturing by value makes an internal COPY that is const by
    // default inside the lambda body; `mutable` allows that internal
    // copy itself to be modified, but it is still only a copy -- the
    // outer x is never touched.
    auto by_value_mutable = [x]() mutable {
        x += 1;
        return x;
    };
    std::cout << "\nSection 2 -- mutable-by-value copy vs reference mutation:" << std::endl;
    std::cout << "  x = 5 initially. by_value_mutable() called three times: "
              << by_value_mutable() << ", " << by_value_mutable() << ", " << by_value_mutable()
              << " (the lambda's OWN internal copy increments across calls)" << std::endl;
    std::cout << "  outer x after those three calls = " << x
              << " (unchanged -- the mutable copy inside the lambda is entirely separate from outer x)"
              << std::endl;

    // Capturing by reference lets the lambda body mutate the ORIGINAL
    // variable directly -- no `mutable` needed, since nothing about
    // the captured reference itself is const.
    auto by_reference = [&x]() {
        x += 1;
        return x;
    };
    std::cout << "  by_reference() called three times: "
              << by_reference() << ", " << by_reference() << ", " << by_reference()
              << " (mutates the real outer x directly)" << std::endl;
    std::cout << "  outer x after those three calls = " << x << " (genuinely changed, from 5 to 8)" << std::endl;
}

// --- Section 3: default capture modes [=] and [&] ---
static void section3_default_capture_modes() {
    int a = 1, b = 2, c = 3;

    auto by_value_default = [=]() { return a + b + c; };   // [=]: copies everything it uses
    auto by_ref_default = [&]() { return a + b + c; };       // [&]: references everything it uses

    std::cout << "\nSection 3 -- default capture modes [=] and [&]:" << std::endl;
    std::cout << "  a=1, b=2, c=3. by_value_default() = " << by_value_default()
              << ", by_ref_default() = " << by_ref_default() << std::endl;
    a = 10; b = 20; c = 30;
    std::cout << "  a,b,c changed to 10,20,30. by_value_default() = " << by_value_default()
              << " (still 6 -- [=] copied a, b, c at creation time), by_ref_default() = " << by_ref_default()
              << " (now 60 -- [&] always reads the current a, b, c)" << std::endl;
}

// --- Section 4: a torch::Tensor-flavored variant, tying this
// directly to this book's own domain rather than leaving it as a
// plain-int abstraction. ---
static void section4_tensor_capture() {
    torch::Tensor running_total = torch::zeros({3});
    auto accumulate = [&running_total](const torch::Tensor& x) { running_total += x; };

    std::cout << "\nSection 4 -- capturing a real torch::Tensor by reference and mutating it across calls:"
              << std::endl;
    accumulate(torch::tensor({1.0, 1.0, 1.0}));
    accumulate(torch::tensor({2.0, 2.0, 2.0}));
    accumulate(torch::tensor({3.0, 3.0, 3.0}));
    std::cout << "  running_total after three accumulate() calls = " << running_total
              << " -- an ordinary reference-capturing lambda mutating a real torch::Tensor exactly the "
              << "way it would mutate an ordinary int, no different capture rule required." << std::endl;
}

// --- Section 5: the bug CUDA's own D.4 restriction prevents. ---
// A function that returns a std::function<int()> capturing a LOCAL
// variable BY REFERENCE -- the local goes out of scope the moment
// this function returns, so the returned closure holds a dangling
// reference. Calling it is undefined behavior: a genuine
// stack-use-after-return bug, the exact bug class Section D.4's own
// CUDA restriction ("An extended __host__ __device__ lambda cannot
// capture variables by reference") categorically prevents at compile
// time for device lambdas -- but which ordinary host C++, including
// this function, remains fully exposed to.
static std::function<int()> make_dangling_closure() {
    int local_value = 42;
    return [&local_value]() { return local_value * 2; };  // dangling the instant this function returns
}

int main(int argc, char** argv) {
    section1_snapshot_vs_live();
    section2_mutable_and_reference_mutation();
    section3_default_capture_modes();
    section4_tensor_capture();

    bool trigger_dangling = (argc > 1 && std::strcmp(argv[1], "--trigger-dangling") == 0);

    if (!trigger_dangling) {
        std::cout << "\nSection 5 (skipped in this run -- pass --trigger-dangling to a build compiled with "
                  << "-fsanitize=address to observe the real caught bug): make_dangling_closure() returns a "
                  << "std::function<int()> that captured a local variable BY REFERENCE -- the instant the "
                  << "function returns, that reference dangles. Section D.4's own CUDA restriction rejects "
                  << "this exact pattern at COMPILE TIME for an extended __host__ __device__ lambda; "
                  << "ordinary host C++ compiles it without complaint and only fails, if at all, at "
                  << "runtime -- which is precisely what a real AddressSanitizer build below demonstrates."
                  << std::endl;
        return 0;
    }

    // Only reached in the ASan-instrumented build, invoked with
    // --trigger-dangling: this genuinely triggers a real
    // stack-use-after-return bug, caught and reported by
    // AddressSanitizer, aborting the process.
    std::cout << "\nSection 5 -- triggering the real dangling-reference bug under AddressSanitizer:" << std::endl;
    auto dangling = make_dangling_closure();
    int result = dangling();  // undefined behavior: reads a destroyed stack frame
    std::cout << "  (unreachable under a working ASan build) dangling() = " << result << std::endl;
    return 0;
}
```

Genuinely compiled (the ordinary compile line used throughout this book) and run with no arguments:

```text
Section 1 -- snapshot (by value) vs live (by reference) capture:
  counter = 10 initially. snapshot() = 10, live() = 10
  counter set to 99. snapshot() = 10 (frozen at capture time, unaffected by the later mutation), live() = 99 (sees the current value, because it holds a reference to counter itself)

Section 2 -- mutable-by-value copy vs reference mutation:
  x = 5 initially. by_value_mutable() called three times: 6, 7, 8 (the lambda's OWN internal copy increments across calls)
  outer x after those three calls = 5 (unchanged -- the mutable copy inside the lambda is entirely separate from outer x)
  by_reference() called three times: 6, 7, 8 (mutates the real outer x directly)
  outer x after those three calls = 8 (genuinely changed, from 5 to 8)

Section 3 -- default capture modes [=] and [&]:
  a=1, b=2, c=3. by_value_default() = 6, by_ref_default() = 6
  a,b,c changed to 10,20,30. by_value_default() = 6 (still 6 -- [=] copied a, b, c at creation time), by_ref_default() = 60 (now 60 -- [&] always reads the current a, b, c)

Section 4 -- capturing a real torch::Tensor by reference and mutating it across calls:
  running_total after three accumulate() calls =  6
 6
 6
[ CPUFloatType{3} ] -- an ordinary reference-capturing lambda mutating a real torch::Tensor exactly the way it would mutate an ordinary int, no different capture rule required.

Section 5 (skipped in this run -- pass --trigger-dangling to a build compiled with -fsanitize=address to observe the real caught bug): make_dangling_closure() returns a std::function<int()> that captured a local variable BY REFERENCE -- the instant the function returns, that reference dangles. Section D.4's own CUDA restriction rejects this exact pattern at COMPILE TIME for an extended __host__ __device__ lambda; ordinary host C++ compiles it without complaint and only fails, if at all, at runtime -- which is precisely what a real AddressSanitizer build below demonstrates.
```

**Worked Example D.6.2.** The identical, unmodified source file, compiled a second time with `-fsanitize=address -g`, run with `--trigger-dangling`: `make_dangling_closure()`'s own returned closure is called, reading a destroyed stack frame -- a genuine `stack-use-after-return`, caught and reported by AddressSanitizer, which aborts the process with exit code 1. Only the report's own stable, address-free summary line is locked below; the full report (genuinely produced on every run) additionally includes real memory addresses and a real process ID that, correctly, differ on every single run, for the identical reason Appendix C.5's own `pin_memory()` exception message locks only its first line.

```text
SUMMARY: AddressSanitizer: stack-use-after-return /home/claude/appendixD/06_capture_value_vs_reference_and_asan_bug.cpp:116 in operator()
```

**Discussion.** Sections 1 through 4 of this file's own output are ordinary, unremarkable lambda-capture behavior -- the same behavior this book's own earlier chapters have relied on implicitly every time they captured a variable by reference (Chapter 19's own `Benchmark::time_function` call sites, Appendix C's own `time_op_ms` call sites) without ever naming the distinction explicitly. Section 5 is where this appendix's own honest reframing lands: Section D.4 of THIS appendix already showed that CUDA's own rejection of reference captures in extended lambdas has no direct analog in this book's own code, because this book's own code never crosses a host/device boundary at all -- but that CUDA restriction was never arbitrary. It exists because a device lambda's own closure might genuinely outlive the host stack frame it captured a reference into, and CUDA's own compiler simply refuses to compile that possibility away rather than risk exactly the bug this section's own AddressSanitizer run demonstrates happening for real. Ordinary host C++ offers no such protection: `make_dangling_closure()` compiles without a single warning under an ordinary build, and only a real runtime tool -- AddressSanitizer, genuinely instrumenting the binary, genuinely catching the violation as it happens -- catches what CUDA's own compiler would have rejected before the program ever ran at all. Section D.4's own restriction, revisited here, is inconvenient for a raw CUDA kernel author in exactly the way this appendix's own earlier sections describe -- and it is also, honestly, a real protection ordinary host C++ programmers, including every reader of this book, do not get for free.

## Complete Runnable Code

### `01_ordinary_lambdas_no_restrictions.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <functional>

// The CUDA C++ edition's own Appendix D opens with CUDA's extended-
// lambda mechanism: an ordinary C++11 lambda, annotated __device__ or
// __host__ __device__ and compiled with nvcc's own --extended-lambda
// flag, becomes callable from inside a __global__ kernel -- along with
// a specific, real restriction its own Section D.2 demonstrates: a
// lambda's TYPE may be device-callable, but the closure OBJECT itself
// still lives in ordinary host memory unless it crosses the host/
// device boundary by being passed BY VALUE as a kernel argument, and
// nvcc genuinely rejects code that tries to reference a host-resident
// closure object directly from device code ("address of a host
// variable cannot be directly taken in a device function"). This
// entire mechanism -- the annotations, the flag, the host/device
// closure-residence distinction, the specific compile errors it
// produces -- exists to solve a problem that never arises anywhere in
// this book: every chapter has written ordinary host-side LibTorch
// calls, never a hand-written __global__ kernel, so no lambda in this
// book has ever needed to cross a host/device boundary at all. This
// file demonstrates the honest baseline every later section in this
// appendix builds on: an ordinary C++ lambda, used exactly the way
// every earlier chapter of this book already has (Chapter 19's own
// Benchmark::time_function, Appendix C's own time_op_ms), with zero
// annotations, zero compiler flags, and zero of CUDA's own
// restrictions to navigate.
int main() {
    // An ordinary, unannotated, plain C++ lambda -- no __device__, no
    // __host__ __device__, no --extended-lambda flag anywhere in this
    // file's own compile line (confirmed by this file's own successful
    // compile with an ordinary g++ invocation, the identical compile
    // line every chapter of this book has used).
    auto square = [](float x) { return x * x; };
    std::cout << "an ordinary, unannotated C++ lambda: square(5.0) = " << square(5.0f) << std::endl;

    // The closure object itself is just an ordinary stack object --
    // there is no separate "device residence" question to ask about
    // it at all, since nothing in this file ever crosses a host/device
    // boundary.
    float bias = 10.0f;
    auto add_bias = [&bias](float x) { return x + bias; };
    std::cout << "a lambda capturing a real variable BY REFERENCE (this alone is a genuine, "
              << "categorical compile-time REJECTION for a CUDA extended __device__/__host__ __device__ "
              << "lambda, per Section D.4's own reported restriction): add_bias(5.0) = " << add_bias(5.0f)
              << std::endl;
    bias = 100.0f;
    std::cout << "  bias changed to 100.0 after add_bias was created: add_bias(5.0) = " << add_bias(5.0f)
              << " (the reference capture genuinely sees the live value -- ordinary, unremarkable C++, "
              << "not something that needs to compile against any device-code restriction here at all)"
              << std::endl;

    // A real torch::Tensor is an entirely ordinary object as far as
    // lambda capture is concerned -- no different from `bias` above.
    torch::Tensor t = torch::tensor({1.0, 2.0, 3.0});
    auto sum_plus_bias = [&t, &bias]() { return t.sum().item<float>() + bias; };
    std::cout << "\na lambda capturing a real torch::Tensor by reference alongside an ordinary float, "
              << "used exactly the way ordinary host LibTorch code already does throughout this book: "
              << "sum_plus_bias() = " << sum_plus_bias() << std::endl;

    // std::function works completely normally here -- Section D.7's
    // own reported failure is specific to CROSSING the host/device
    // boundary (a __global__ kernel parameter), which this file never
    // attempts.
    std::function<float(float)> as_std_function = square;
    std::cout << "\nsquare wrapped in std::function<float(float)> and called through that wrapper: "
              << as_std_function(6.0f) << " (std::function works completely normally in ordinary host "
              << "code -- Section D.4 of this appendix returns to exactly why it does NOT work as a "
              << "kernel parameter, and why that distinction matters not at all to LibTorch code)"
              << std::endl;

    return 0;
}
```

### `02_generic_functions_and_functors.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.3 writes a generic __global__
// kernel, transform_kernel<F>, templated on a callable type F, and
// launches it three times with three different device lambdas
// (square, cube, a capturing "scale by k" lambda) -- demonstrating
// that a single generic kernel body can be specialized, at compile
// time, by whatever callable is handed to it. Its own Section D.5
// then shows the identical technique works with an ordinary functor
// struct (one with operator()) in place of a lambda, and that from
// the compiler's point of view the two are indistinguishable: a
// lambda IS a compiler-synthesized functor, nothing more.
//
// Ordinary host-side C++ has exactly this same generic-programming
// technique available, with no device/kernel machinery required at
// all -- a plain template<typename F> function, applied here to a
// real std::vector<float> the way Chapter 1 through Appendix C's own
// helper functions have used ordinary host containers throughout.

// A generic host-side "transform," templated on the callable type F,
// exactly mirroring the CUDA book's own transform_kernel<F> in spirit
// -- one function body, specialized at compile time by whatever F is
// handed to it.
template <typename F>
std::vector<float> transform_all(const std::vector<float>& in, F f) {
    std::vector<float> out;
    out.reserve(in.size());
    for (float x : in) {
        out.push_back(f(x));
    }
    return out;
}

// An ordinary functor struct -- the CUDA book's own Section D.5
// equivalent. From transform_all<F>'s own point of view, below, this
// is indistinguishable from a lambda: both are just callable types
// with an operator()(float) const.
struct ScaleFunctor {
    float k;
    float operator()(float x) const { return x * k; }
};

static void print_vec(const char* label, const std::vector<float>& v) {
    std::cout << label << ": ";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    print_vec("input", data);

    // Same three specializations the CUDA book's own Section D.3
    // demonstrates -- square, cube, and a capturing "scale by k"
    // lambda -- but as one ordinary template<typename F> host
    // function, called three times with three different callables,
    // never a __global__ kernel launch anywhere in this file.
    auto squared = transform_all(data, [](float x) { return x * x; });
    print_vec("squared (lambda [](float x){ return x*x; })", squared);

    auto cubed = transform_all(data, [](float x) { return x * x * x; });
    print_vec("cubed (lambda [](float x){ return x*x*x; })", cubed);

    float k = 10.0f;
    auto scaled_lambda = transform_all(data, [k](float x) { return x * k; });
    print_vec("scaled by 10 (capturing lambda [k](float x){ return x*k; })", scaled_lambda);

    // The functor version: the CUDA book's own Section D.5 point,
    // reproduced here verbatim -- transform_all<ScaleFunctor> and
    // transform_all<lambda-closure-type> are two different template
    // INSTANTIATIONS of the identical function body, because
    // ScaleFunctor and the lambda's own compiler-synthesized closure
    // type both satisfy exactly the same requirement: an
    // operator()(float) const.
    ScaleFunctor scale_functor{10.0f};
    auto scaled_functor = transform_all(data, scale_functor);
    print_vec("scaled by 10 (ScaleFunctor{10.0f}, an ordinary functor struct)", scaled_functor);

    std::cout << "\nscaled_lambda == scaled_functor element-wise? "
              << (scaled_lambda == scaled_functor ? "true" : "false")
              << " -- transform_all<F> instantiated once for the lambda's own compiler-synthesized "
              << "closure type and once for ScaleFunctor, producing identical results from what is, "
              << "to the compiler, the same generic-programming pattern the CUDA book's own "
              << "transform_kernel<F> uses -- with no __global__, no --extended-lambda flag, and no "
              << "host/device closure-residence question anywhere in this file." << std::endl;

    // Applying the identical technique to a real torch::Tensor, tying
    // it back to this book's own domain: torch::Tensor itself already
    // exposes plenty of built-in elementwise ops, but nothing stops an
    // ordinary generic host function from being handed one lambda per
    // call site exactly as transform_all<F> is above.
    torch::Tensor t = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0});
    torch::Tensor t_squared = t.pow(2);
    std::cout << "\ncross-check against a real torch::Tensor: t = " << t
              << ", t.pow(2) = " << t_squared
              << " -- matches the lambda-based 'squared' vector above element-wise? "
              << torch::allclose(t_squared, torch::tensor(squared)) << std::endl;

    return 0;
}
```

### `03_generic_reduction_vs_tensor_ops.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.6 writes a generic
// reduce_kernel<T, Op>, templated on both the element type and a
// binary-operation callable, and runs it over the fixed input
// {3, 7, 1, 9, 4, 2, 8, 5} with four different operations -- a sum
// lambda, a max lambda, a min lambda, and a product lambda -- getting
// Sum=39.0, Max=9.0, Min=1.0, Product=60480.0. The generic-reduction
// TECHNIQUE (a function templated on a binary-operation callable) is
// ordinary C++, identical in spirit to Section D.2's own
// transform_all<F> in this appendix -- so rather than re-deriving
// that same point a second time, this file reproduces the CUDA
// book's own exact reduction values as a correctness check, and then
// cross-checks them against real torch::Tensor reduction ops
// (.sum(), .max(), .min(), .prod()), the ones this book's own
// chapters have actually used throughout (Chapter 3's own reductions,
// Chapter 22's own portfolio sums) instead of a hand-written
// reduce_kernel at all.
template <typename T, typename Op>
T reduce_all(const std::vector<T>& data, T init, Op op) {
    T acc = init;
    for (T x : data) {
        acc = op(acc, x);
    }
    return acc;
}

int main() {
    // The CUDA book's own exact fixed input.
    std::vector<double> data = {3, 7, 1, 9, 4, 2, 8, 5};

    double sum_result = reduce_all<double>(data, 0.0, [](double a, double b) { return a + b; });
    double max_result = reduce_all<double>(data, data[0], [](double a, double b) { return a > b ? a : b; });
    double min_result = reduce_all<double>(data, data[0], [](double a, double b) { return a < b ? a : b; });
    double product_result = reduce_all<double>(data, 1.0, [](double a, double b) { return a * b; });

    std::cout << "reduce_all<double>(data, op) over {3, 7, 1, 9, 4, 2, 8, 5}, four different binary-op "
              << "lambdas handed to the identical generic reduce_all<T, Op> body (the CUDA book's own "
              << "Section D.6 pattern, here as an ordinary host template<T, Op> function, no "
              << "reduce_kernel launch anywhere):" << std::endl;
    std::cout << "  Sum      = " << sum_result << " (CUDA book's own expected 39.0)" << std::endl;
    std::cout << "  Max      = " << max_result << " (expected 9.0)" << std::endl;
    std::cout << "  Min      = " << min_result << " (expected 1.0)" << std::endl;
    std::cout << "  Product  = " << product_result << " (expected 60480.0)" << std::endl;

    // The cross-check this file adds beyond the CUDA book's own
    // section: the SAME data, reduced via real torch::Tensor
    // reduction ops -- the ones this book's own chapters have
    // actually called throughout -- rather than any hand-written
    // reduce_all<T, Op> at all.
    torch::Tensor t = torch::tensor(data, torch::kFloat64);
    double t_sum = t.sum().item<double>();
    double t_max = t.max().item<double>();
    double t_min = t.min().item<double>();
    double t_prod = t.prod().item<double>();

    std::cout << "\ncross-check via real torch::Tensor reduction ops on the identical data (torch::tensor"
              << "(data).sum()/.max()/.min()/.prod() -- the actual API this book's own chapters have used, "
              << "e.g. Chapter 22's own portfolio-total .sum()):" << std::endl;
    std::cout << "  t.sum()  = " << t_sum << (t_sum == sum_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.max()  = " << t_max << (t_max == max_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.min()  = " << t_min << (t_min == min_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    std::cout << "  t.prod() = " << t_prod << (t_prod == product_result ? "  (exact match)" : "  (MISMATCH)") << std::endl;

    std::cout << "\nthe generic reduce_all<T, Op> template above is genuinely useful for an ARBITRARY "
              << "binary operation no built-in torch::Tensor method covers -- but for the four common "
              << "reductions this book's own chapters actually reach for, real torch::Tensor methods "
              << "already exist, are already tensor-shape- and dtype-aware, and (unlike this file's own "
              << "serial reduce_all loop) already dispatch to ATen's own parallel reduction kernels "
              << "underneath, with no hand-written reduce_kernel, no block/grid launch configuration, "
              << "and no shared-memory reduction tree for a programmer to get wrong." << std::endl;

    return 0;
}
```

### `04_std_function_real_non_failure.cpp`

```cpp
#include <torch/torch.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section D.7 reports a genuine, real
// nvcc compile failure: passing a std::function<...> as a __global__
// kernel PARAMETER fails, because std::function's own virtual-dispatch
// machinery (a heap-allocated, host-resident vtable-driven call) has
// no meaning inside device code, which cannot perform host-side
// virtual dispatch at all. That restriction is specific to crossing
// the host/device boundary as a kernel argument -- it says nothing
// about std::function used in ordinary host code, which is exactly
// how this book's own chapters have already used it, more than once,
// without incident:
//
//   - Chapter 19's own Benchmark::time_function(std::function<void()>)
//     helper wraps an arbitrary block of code to be timed.
//   - Appendix C's own time_op_ms(std::function<void()>) helper does
//     the identical thing for that appendix's own six timing
//     measurements (C.3, C.5, C.6).
//
// Both are real, already-compiled, already-verified precedent
// elsewhere in this book: std::function has never once failed here,
// because neither helper is ever called from device code or passed
// as a kernel argument. This file reproduces that same
// std::function<void()>-wrapping-arbitrary-work pattern directly, as
// a documented non-failure in its own right, rather than treating
// D.7's own restriction as something LibTorch code must somehow work
// around.

// The same shape as Chapter 19's Benchmark::time_function and
// Appendix C's time_op_ms: an ordinary host function taking a
// std::function<void()> parameter, timing whatever it wraps.
double time_op_ms(const std::function<void()>& op, int reps) {
    // warm-up, exactly as Appendix C's own time_op_ms does
    op();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) {
        op();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    // A std::function<void()> wrapping a real torch::Tensor
    // computation -- passed by ordinary reference into an ordinary
    // host function, precisely the way Chapter 19 and Appendix C
    // already do it.
    torch::Tensor a = torch::randn({500, 500});
    torch::Tensor b = torch::randn({500, 500});
    std::function<void()> matmul_op = [&a, &b]() { torch::matmul(a, b); };

    double ms = time_op_ms(matmul_op, 20);
    std::cout << "std::function<void()> wrapping torch::matmul(a, b) (500x500), passed by reference into "
              << "an ordinary host time_op_ms(const std::function<void()>&, int) helper -- the same "
              << "pattern Chapter 19's Benchmark::time_function and Appendix C's time_op_ms already use "
              << "throughout this book: compiled and ran with zero issues." << std::endl;
    std::cout << "[TIMING] mean time per call over 20 reps: " << ms << " ms" << std::endl;

    // A vector of std::function<int(int)>, each wrapping a different
    // callable (a plain lambda, a capturing lambda, and a named
    // ordinary function) -- ordinary host-side polymorphism over
    // callables, ordinary heap-allocated type erasure, nothing here
    // needs to be callable from device code at all.
    auto double_it = [](int x) { return x * 2; };
    int offset = 100;
    auto add_offset = [offset](int x) { return x + offset; };
    std::function<int(int)> square_fn = [](int x) { return x * x; };

    std::vector<std::function<int(int)>> ops = {double_it, add_offset, square_fn};
    std::cout << "\na std::vector<std::function<int(int)>> holding three different callables (a plain "
              << "lambda, a capturing lambda, and a lambda already stored in a named std::function) -- "
              << "ordinary host-side runtime polymorphism over callables, applied to x = 7:" << std::endl;
    for (size_t i = 0; i < ops.size(); i++) {
        std::cout << "  ops[" << i << "](7) = " << ops[i](7) << std::endl;
    }

    std::cout << "\nnone of this required anything special: std::function's own vtable-driven, "
              << "heap-allocated dispatch mechanism is exactly what ordinary host C++ is built to run. "
              << "Section D.7 of the CUDA C++ edition's own restriction is specific to a __global__ "
              << "kernel PARAMETER -- host code calling std::function::operator() is the ordinary case "
              << "this mechanism was designed for, and this book's own Chapter 19 and Appendix C have "
              << "already relied on exactly that, more than once, before this appendix ever raised the "
              << "question." << std::endl;

    return 0;
}
```

### `05_composing_lambdas.cpp`

```cpp
#include <iostream>

// The CUDA C++ edition's own Section D.8 tries the obvious thing
// first -- an auto-returning compose(f, g) template -- and reports it
// genuinely fails to compile, with a real, specific error: the
// enclosing function (compose itself) has a deduced return type
// (auto), and a nested lambda defined INSIDE compose is not allowed
// to be part of that deduction in the way its own first attempt
// needed. Its own fix is to abandon auto entirely and hand-write an
// explicit Composed<F, G> functor struct instead, storing f and g as
// member data and implementing operator() by hand.
//
// This restriction is specific to how CUDA's own device-lambda
// compilation model interacts with return-type deduction across a
// device-lambda boundary -- ordinary host C++ has no such
// restriction at all. This file writes the exact auto-returning
// compose(f, g) the CUDA book's own first attempt needed, and it
// simply works, with no Composed<F, G> functor workaround required.
template <typename F, typename G>
auto compose(F f, G g) {
    // An ordinary lambda, defined inside an auto-returning function,
    // capturing f and g by value -- exactly the construct CUDA's own
    // Section D.8 reports failing to compile as an extended device
    // lambda. Ordinary host C++ imposes no such restriction: this
    // compiles and works exactly as an unremarkable programmer would
    // expect it to.
    return [f, g](auto x) { return f(g(x)); };
}

int main() {
    // The CUDA book's own exact worked pipeline: square(add_one(scale2(x))),
    // i.e. compose(square, compose(add_one, scale2)), which for input x
    // computes ((2x) + 1)^2.
    auto square = [](auto x) { return x * x; };
    auto add_one = [](auto x) { return x + 1; };
    auto scale2 = [](auto x) { return x * 2; };

    auto pipeline = compose(square, compose(add_one, scale2));

    std::cout << "compose(f, g) := an ordinary auto-returning template<F,G> function returning a lambda "
              << "that computes f(g(x)) -- the CUDA book's own Section D.8 reports this exact construct "
              << "genuinely FAILS to compile as an extended __device__ lambda ('the enclosing function ... "
              << "must not have a deduced return type'), forcing a hand-written Composed<F,G> functor "
              << "workaround instead. Ordinary host C++, as used throughout this book, has no such "
              << "restriction -- this file's own compose(f, g) is the literal auto-returning version, "
              << "compiled and run with zero workaround." << std::endl;

    std::cout << "\npipeline := compose(square, compose(add_one, scale2)), i.e. pipeline(x) = "
              << "((2*x) + 1)^2, applied to x = 1..5:" << std::endl;
    for (int x = 1; x <= 5; x++) {
        int computed = pipeline(x);
        int expected = (2 * x + 1) * (2 * x + 1);
        std::cout << "  pipeline(" << x << ") = " << computed
                   << ", hand-expanded (2*" << x << "+1)^2 = " << expected
                   << (computed == expected ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    }

    // A second, independent composition chain to further exercise the
    // same compose(f, g), confirming it genuinely generalizes rather
    // than only working for the one CUDA-book pipeline above: a
    // three-stage chain computing sqrt-free integer work, cube(x-2).
    auto cube = [](auto x) { return x * x * x; };
    auto sub_two = [](auto x) { return x - 2; };
    auto pipeline2 = compose(cube, sub_two);
    std::cout << "\na second, independent chain, pipeline2 := compose(cube, sub_two), i.e. "
              << "pipeline2(x) = (x-2)^3, applied to x = 0..4:" << std::endl;
    for (int x = 0; x <= 4; x++) {
        int computed = pipeline2(x);
        int expected = (x - 2) * (x - 2) * (x - 2);
        std::cout << "  pipeline2(" << x << ") = " << computed
                   << ", hand-expanded (" << x << "-2)^3 = " << expected
                   << (computed == expected ? "  (exact match)" : "  (MISMATCH)") << std::endl;
    }

    return 0;
}
```

### `06_capture_value_vs_reference_and_asan_bug.cpp`

```cpp
#include <torch/torch.h>
#include <cstring>
#include <functional>
#include <iostream>

// The CUDA C++ edition's own Section D.9 (plain, CUDA-free C++, no
// nvcc-specific restriction anywhere in it) covers capture-by-value
// snapshot semantics, capture-by-value mutation via `mutable`,
// capture-by-reference live-value semantics, and default capture
// modes [=]/[&] -- ordinary C++ lambda behavior that applies to this
// book's own code exactly as it does to the CUDA book's own host-side
// examples. This file reproduces all of that directly (Sections 1-4
// below), then goes one step further than the CUDA book's own D.9.4:
// where D.9.4 shows a dangling-reference-capture bug and its
// AddressSanitizer report as a cautionary example, this file connects
// it back to Section D.4 of THIS appendix (the reference-capture
// restriction on CUDA's own extended __device__/__host__ __device__
// lambdas) -- reframing that restriction as something that, however
// inconvenient for a raw CUDA kernel author, categorically PREVENTS
// this exact bug class at compile time, a protection ordinary host
// C++ -- including every lambda in every earlier chapter of this
// book -- does not have.

// --- Section 1: snapshot (by value) vs live (by reference) capture ---
static void section1_snapshot_vs_live() {
    int counter = 10;
    auto snapshot = [counter]() { return counter; };  // captures a COPY, frozen at creation
    auto live = [&counter]() { return counter; };      // captures a REFERENCE, always current

    std::cout << "Section 1 -- snapshot (by value) vs live (by reference) capture:" << std::endl;
    std::cout << "  counter = 10 initially. snapshot() = " << snapshot() << ", live() = " << live() << std::endl;
    counter = 99;
    std::cout << "  counter set to 99. snapshot() = " << snapshot()
              << " (frozen at capture time, unaffected by the later mutation), live() = " << live()
              << " (sees the current value, because it holds a reference to counter itself)" << std::endl;
}

// --- Section 2: mutable-by-value copy vs reference mutation ---
static void section2_mutable_and_reference_mutation() {
    int x = 5;

    // Capturing by value makes an internal COPY that is const by
    // default inside the lambda body; `mutable` allows that internal
    // copy itself to be modified, but it is still only a copy -- the
    // outer x is never touched.
    auto by_value_mutable = [x]() mutable {
        x += 1;
        return x;
    };
    std::cout << "\nSection 2 -- mutable-by-value copy vs reference mutation:" << std::endl;
    std::cout << "  x = 5 initially. by_value_mutable() called three times: "
              << by_value_mutable() << ", " << by_value_mutable() << ", " << by_value_mutable()
              << " (the lambda's OWN internal copy increments across calls)" << std::endl;
    std::cout << "  outer x after those three calls = " << x
              << " (unchanged -- the mutable copy inside the lambda is entirely separate from outer x)"
              << std::endl;

    // Capturing by reference lets the lambda body mutate the ORIGINAL
    // variable directly -- no `mutable` needed, since nothing about
    // the captured reference itself is const.
    auto by_reference = [&x]() {
        x += 1;
        return x;
    };
    std::cout << "  by_reference() called three times: "
              << by_reference() << ", " << by_reference() << ", " << by_reference()
              << " (mutates the real outer x directly)" << std::endl;
    std::cout << "  outer x after those three calls = " << x << " (genuinely changed, from 5 to 8)" << std::endl;
}

// --- Section 3: default capture modes [=] and [&] ---
static void section3_default_capture_modes() {
    int a = 1, b = 2, c = 3;

    auto by_value_default = [=]() { return a + b + c; };   // [=]: copies everything it uses
    auto by_ref_default = [&]() { return a + b + c; };       // [&]: references everything it uses

    std::cout << "\nSection 3 -- default capture modes [=] and [&]:" << std::endl;
    std::cout << "  a=1, b=2, c=3. by_value_default() = " << by_value_default()
              << ", by_ref_default() = " << by_ref_default() << std::endl;
    a = 10; b = 20; c = 30;
    std::cout << "  a,b,c changed to 10,20,30. by_value_default() = " << by_value_default()
              << " (still 6 -- [=] copied a, b, c at creation time), by_ref_default() = " << by_ref_default()
              << " (now 60 -- [&] always reads the current a, b, c)" << std::endl;
}

// --- Section 4: a torch::Tensor-flavored variant, tying this
// directly to this book's own domain rather than leaving it as a
// plain-int abstraction. ---
static void section4_tensor_capture() {
    torch::Tensor running_total = torch::zeros({3});
    auto accumulate = [&running_total](const torch::Tensor& x) { running_total += x; };

    std::cout << "\nSection 4 -- capturing a real torch::Tensor by reference and mutating it across calls:"
              << std::endl;
    accumulate(torch::tensor({1.0, 1.0, 1.0}));
    accumulate(torch::tensor({2.0, 2.0, 2.0}));
    accumulate(torch::tensor({3.0, 3.0, 3.0}));
    std::cout << "  running_total after three accumulate() calls = " << running_total
              << " -- an ordinary reference-capturing lambda mutating a real torch::Tensor exactly the "
              << "way it would mutate an ordinary int, no different capture rule required." << std::endl;
}

// --- Section 5: the bug CUDA's own D.4 restriction prevents. ---
// A function that returns a std::function<int()> capturing a LOCAL
// variable BY REFERENCE -- the local goes out of scope the moment
// this function returns, so the returned closure holds a dangling
// reference. Calling it is undefined behavior: a genuine
// stack-use-after-return bug, the exact bug class Section D.4's own
// CUDA restriction ("An extended __host__ __device__ lambda cannot
// capture variables by reference") categorically prevents at compile
// time for device lambdas -- but which ordinary host C++, including
// this function, remains fully exposed to.
static std::function<int()> make_dangling_closure() {
    int local_value = 42;
    return [&local_value]() { return local_value * 2; };  // dangling the instant this function returns
}

int main(int argc, char** argv) {
    section1_snapshot_vs_live();
    section2_mutable_and_reference_mutation();
    section3_default_capture_modes();
    section4_tensor_capture();

    bool trigger_dangling = (argc > 1 && std::strcmp(argv[1], "--trigger-dangling") == 0);

    if (!trigger_dangling) {
        std::cout << "\nSection 5 (skipped in this run -- pass --trigger-dangling to a build compiled with "
                  << "-fsanitize=address to observe the real caught bug): make_dangling_closure() returns a "
                  << "std::function<int()> that captured a local variable BY REFERENCE -- the instant the "
                  << "function returns, that reference dangles. Section D.4's own CUDA restriction rejects "
                  << "this exact pattern at COMPILE TIME for an extended __host__ __device__ lambda; "
                  << "ordinary host C++ compiles it without complaint and only fails, if at all, at "
                  << "runtime -- which is precisely what a real AddressSanitizer build below demonstrates."
                  << std::endl;
        return 0;
    }

    // Only reached in the ASan-instrumented build, invoked with
    // --trigger-dangling: this genuinely triggers a real
    // stack-use-after-return bug, caught and reported by
    // AddressSanitizer, aborting the process.
    std::cout << "\nSection 5 -- triggering the real dangling-reference bug under AddressSanitizer:" << std::endl;
    auto dangling = make_dangling_closure();
    int result = dangling();  // undefined behavior: reads a destroyed stack frame
    std::cout << "  (unreachable under a working ASan build) dangling() = " << result << std::endl;
    return 0;
}
```

## Chapter Summary

This appendix mapped the CUDA C++ edition's own ten-section functional-and-lambda-programming appendix onto ordinary host-side LibTorch C++, and found a genuinely different shape than every earlier chapter and appendix of this book: rather than a rich, section-by-section translation of CUDA-specific machinery into LibTorch-specific machinery, most of what CUDA's own Appendix D covers -- the extended-lambda annotations, the `--extended-lambda` compiler flag, the host/device closure-residence distinction, the reference-capture rejection, `std::function`'s own failure as a kernel parameter, the deduced-return-type restriction on lambda composition -- simply has no analog in this book's own code at all, because no chapter has ever written a hand-rolled `__global__` kernel. Sections D.2 and D.3 showed that the underlying generic-programming TECHNIQUE (a function templated on a callable type) works identically in ordinary host C++, with no kernel-launch machinery required, and cross-checked a generic reduction against the real `torch::Tensor` methods this book's own chapters actually use. Section D.4 showed `std::function` as a real, already-used, already-working pattern, not a restriction to route around. Section D.5 showed an `auto`-returning `compose(f, g)` compiling and running with zero workaround, where CUDA's own device-lambda rules force a hand-written functor. And Section D.6 closed by reframing CUDA's own reference-capture restriction honestly: not merely absent from this book's own code, but a real, compile-time protection against a genuine dangling-reference bug class that ordinary host C++ remains fully exposed to, demonstrated with a real, Bash-and-appendix-verified AddressSanitizer catch.

## Self-Check Questions

1. Section D.1 states that CUDA's own extended-lambda mechanism has "zero analog" in this book's own code. Explain precisely why that is true, in terms of what kind of code every chapter of this book has actually written, rather than in terms of any limitation of LibTorch as a library.
2. Section D.2 merges CUDA's own Sections D.3 and D.5 into a single demonstration. Explain, in your own words, why a lambda and an ordinary functor struct are "the identical kind of thing" from the point of view of a `template<typename F>` function like `transform_all<F>`.
3. Section D.4 reframes CUDA's own Section D.7 restriction as a "real, documented non-failure" for this book's own code. Explain precisely what CUDA's own restriction actually forbids, and why that specific thing is not something Chapter 19's `Benchmark::time_function` or Appendix C's `time_op_ms` has ever needed to do.
4. Section D.5's own `compose(f, g)` is a direct, unmodified version of the construct CUDA's own Section D.8 reports failing to compile. Explain specifically what CUDA's own compile error object to, and why that same objection does not apply to an ordinary host-side `template<typename F, typename G> auto compose(F f, G g)`.
5. Section D.6 compiles its own source file TWICE, with two different compiler flags, and reports two different outcomes. Explain why a single build could not have demonstrated both this section's own ordinary capture-semantics examples and its own AddressSanitizer-caught dangling-reference bug.
6. Section D.6 closes by calling CUDA's own reference-capture restriction (from Section D.4) "a real protection ordinary host C++ programmers... do not get for free." Explain what, specifically, ordinary host C++ is missing that CUDA's own extended-lambda compiler check provides, and what tool this section uses instead to catch the same class of bug.

## Where We Go Next

This appendix, like Appendix C before it, is reference material -- Part 7 already closed this book's own main arc in Chapter 22. Its own honest finding is narrower and more structural than Appendix C's own section-by-section hardware mapping: most of what CUDA's own Appendix D covers simply does not apply to code written the way this book has written every chapter, because this book's own code has never crossed a host/device boundary with a hand-rolled kernel. The appendices that follow continue the same reference spirit -- the streaming-multiprocessor-as-pipeline model, the Rule of Five, and tensor contractions on CPU and GPU -- material this book's own main chapters have used along the way without dwelling on every implementation detail.

## Worked Solutions

**1.** Every chapter of this book, from Chapter 1's own first tensor operation through Appendix C's own thread-pool measurements, has written ordinary HOST-side code that calls real, already-compiled LibTorch/ATen operations (`torch::matmul`, `torch::relu`, and so on) -- never a hand-written `__global__` kernel of its own. CUDA's own extended-lambda mechanism exists specifically to let a lambda's closure be called FROM device code, inside a `__global__` kernel body; since this book's own code has never written a `__global__` kernel body at all, there has never been a device-code call site for any lambda in this book to be extended into. This is not a gap or a limitation of LibTorch itself -- LibTorch's own internal CUDA kernels almost certainly use extended device lambdas extensively in ATen's own source -- it is a direct, structural consequence of the level at which a LibTorch APPLICATION programmer, the audience this book is written for, actually operates.

**2.** A `template<typename F>` function like `transform_all<F>` places exactly one requirement on `F`: that an expression of the form `f(x)` (or, more precisely, `f` applied via `operator()`) is well-formed for the argument type it's called with. An ordinary lambda's own type is a compiler-synthesized, unnamed class with exactly one member function, `operator()`, matching the lambda's own parameter list and body. An ordinary functor struct like `ScaleFunctor`, with a hand-written `float operator()(float x) const`, satisfies the identical requirement by hand. `transform_all<F>`'s own template body never asks HOW `F` came to have an `operator()` -- only that it does -- so instantiating it with a lambda's closure type and instantiating it with `ScaleFunctor` are, to the compiler, two ordinary template instantiations of the identical function body against two different types that both happen to satisfy the same implicit interface.

**3.** CUDA's own Section D.7 restriction specifically forbids passing a `std::function<...>` as a PARAMETER TO A `__global__` KERNEL -- that is, using it as an argument type in code that must be callable from device code, where `std::function`'s own heap-allocated, vtable-driven, host-resident dispatch mechanism has no meaning at all. Chapter 19's own `Benchmark::time_function` and Appendix C's own `time_op_ms` both accept a `std::function<void()>` parameter, but both are ORDINARY HOST FUNCTIONS, called from other host code, never launched as a kernel and never called from device code -- so neither has ever needed `std::function` to cross the one specific boundary CUDA's own restriction is actually about.

**4.** CUDA's own compile error specifically objects to the ENCLOSING FUNCTION -- `compose` itself -- having a deduced return type (`auto`) while also containing a lambda that CUDA's own device-lambda compilation model must analyze as part of an extended-lambda context; its own error message states plainly that the enclosing function "must not have a deduced return type." This restriction is a direct consequence of CUDA's own device-lambda compilation machinery specifically, which must determine, at compile time, various properties of any lambda used as an extended device lambda, in a way ordinary host-side lambda compilation never has to. An ordinary host-side `template<typename F, typename G> auto compose(F f, G g)` never invokes any of that device-lambda machinery at all -- the lambda it returns is an entirely ordinary host closure type, and ordinary C++ return-type deduction (in use throughout modern C++ well before CUDA's own extended-lambda feature existed at all) handles deducing that closure's own type without any special restriction.

**5.** A single build compiled the ordinary way (no `-fsanitize=address`) can run `make_dangling_closure()`'s own returned closure without AddressSanitizer ever detecting anything, since the undefined behavior involved (reading a destroyed stack frame) is not guaranteed to crash, corrupt visible output, or behave any differently from correct code on every single run -- it might simply happen to read stack memory that has not yet been overwritten, and produce a plausible-looking wrong answer instead of a visible crash, which would make for a dishonest, unreliable demonstration. A single build compiled WITH `-fsanitize=address`, calling that same dangling closure unconditionally, would abort the ENTIRE program the moment it reached that call -- meaning Sections 1 through 4's own ordinary, deterministic capture-semantics examples could never be observed to complete normally in that same run, since the program would abort partway through. Compiling the identical source twice, with the trigger gated behind a command-line flag, lets the same file demonstrate both a fully ordinary, deterministic run (Sections 1-4, unconditionally) and a genuinely caught bug (Section 5, only in the ASan build, only when explicitly triggered) without either demonstration interfering with the other.

**6.** CUDA's own extended-lambda compiler check refuses, AT COMPILE TIME, to accept a reference capture in an extended `__host__ __device__` lambda at all -- the program simply does not compile if a raw CUDA kernel author writes that pattern, so the specific bug this section demonstrates (a closure outliving the stack frame its captured reference pointed into) can never reach a running CUDA kernel through that specific mechanism, full stop. Ordinary host C++ imposes no such compile-time check: `make_dangling_closure()`'s own reference-capturing lambda, returned from a function whose local variable is about to go out of scope, compiles cleanly under an ordinary build, and the resulting bug only becomes visible at RUNTIME, and only because this section deliberately compiled a second build with AddressSanitizer, a runtime instrumentation tool that watches every memory access and catches the specific violation (`stack-use-after-return`) as it happens. Without a tool like AddressSanitizer actively watching, the identical bug could silently read stray stack memory and produce a plausible-looking wrong number instead of any visible failure at all -- which is exactly the gap CUDA's own compile-time restriction closes for device lambdas, and exactly the gap ordinary host C++ leaves open unless a programmer reaches for a tool like ASan deliberately.
