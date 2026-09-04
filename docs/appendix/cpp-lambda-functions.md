# Appendix F: C++ Lambda Functions, From First Principles

> The CUDA C++ edition's own Appendix F steps back from Appendix D's own CUDA-specific vocabulary -- extended lambdas, the `--extended-lambda` flag, device-closure residence -- to the plain C++ lambda feature those extensions are built on top of. This book's own Appendix D made exactly the same assumption Appendix D of the CUDA C++ edition did: that a reader already knows ordinary lambda syntax. This appendix is that missing foundation, written the way every chapter since Chapter 1 has quietly relied on it: a capture-and-call syntax this book has used constantly, named and explained explicitly for the first time.

**What you will understand by the end of this appendix:** the four fixed parts of every lambda (capture, parameters, optional return type, body), and when a trailing return type is actually needed rather than merely allowed; why a lambda, despite having no name, is a real compiler-synthesized class with a real `sizeof` and a real, distinct type -- confirmed directly, not asserted; the full capture-clause vocabulary, including the real difference between `[this]` and C++17's `[*this]` on a class holding a real `torch::Tensor` member; why a non-`mutable` lambda's own `operator()` is implicitly `const`, confirmed via a real, genuinely captured compiler error; C++14's generalized (init) capture, `[name = expr]`, applied to a real momentum-style tensor smoother; how an ordinary lambda serves as a comparator or predicate for standard-library algorithms; and how `std::function` solves, by type erasure, the exact problem this appendix's own Section F.2 first demonstrates.

**What you need to know first:** ordinary C++ functions, `auto`, and templates, at the level used throughout this book since Chapter 1; this book's own Appendix D (optional but recommended -- D.6's own capture-by-reference and dangling-reference material and this appendix's Section F.4 are close relatives, one host-only and one CUDA-motivated) and Appendix D.4's own `std::function` demonstration, which Section F.7 below builds on rather than repeats; `std::sort`, `std::count_if`, and `std::find_if`'s own signatures, for Section F.6.

## F.1 Anatomy of a Lambda: Capture, Parameters, Body, Return Type `[FOUNDATIONAL]`

**Intuition.** Every lambda in C++ has four fixed parts, in order: a capture clause `[]`, a parameter list `()`, an optional trailing return type, and a body `{}`. This book has used exactly this shape, unnamed, since its own Chapter 1 -- every comparator, every `std::function<void()>` timing wrapper in Chapter 19 and Appendix C has been one.

**Background.** An explicit trailing return type (`-> int`) and a deduced one are behaviorally identical for a single-return-statement body; a trailing return type earns its keep specifically when plain deduction would infer something narrower than intended -- for instance, a lambda whose body returns string literals, which deduce as `const char*` rather than the `std::string` an explicit `-> std::string` requests.

**Worked Example F.1.1.** Explicit versus deduced return types on an identical addition; an empty-capture lambda, the shape this book's own Chapter 19 and Appendix C helpers have always used; and a real `torch::Tensor`-classifying lambda where an explicit trailing return type genuinely changes the result type.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Appendix F opens by stepping back from
// Appendix D's own CUDA-specific vocabulary (extended lambdas, the
// --extended-lambda flag, device-closure residence) to the plain C++
// lambda feature those extensions are built on top of. This book has
// used ordinary lambdas in nearly every chapter since Chapter 1
// without ever naming their own anatomy explicitly -- this appendix
// is that missing foundation, written the way this book's own
// Appendix D already assumed a reader had it: four fixed parts, in
// order, in every lambda -- a capture clause `[]`, a parameter list
// `()`, an optional trailing return type, and a body `{}`.
int main() {
    // An explicit trailing return type versus a deduced one -- both
    // behaviorally identical, differing only in whether the compiler
    // is told the return type or works it out from the body's own
    // return statement.
    auto add_explicit = [](int a, int b) -> int { return a + b; };
    auto add_deduced = [](int a, int b) { return a + b; };

    std::cout << "add_explicit(3, 4) = " << add_explicit(3, 4)
              << " (explicit trailing return type -> int)" << std::endl;
    std::cout << "add_deduced(3, 4)  = " << add_deduced(3, 4)
              << " (deduced return type, identical result)" << std::endl;

    // An empty-capture lambda -- the smallest, most common shape this
    // book's own chapters have already used dozens of times (every
    // sort comparator, every std::function<void()> timing wrapper in
    // Chapter 19 and Appendix C), here named explicitly for the first
    // time: capture nothing, take one parameter, return one value.
    auto square = [](double x) { return x * x; };
    std::cout << "square(5.0) = " << square(5.0)
              << " (empty capture [] -- this book's own Chapter 19 Benchmark and Appendix C time_op_ms "
              << "lambdas have always been this exact shape)" << std::endl;

    // A trailing return type is genuinely useful, beyond mere style,
    // whenever plain return-type deduction would infer something
    // narrower than what's actually wanted -- here, both branches
    // return a string LITERAL (deducible as `const char*`), but an
    // explicit `-> std::string` tells the compiler to convert each
    // one into a real std::string instead, this book's own domain
    // example: classifying a real torch::Tensor's mean by sign.
    auto classify_mean = [](const torch::Tensor& t) -> std::string {
        double m = t.mean().item<double>();
        if (m > 0.0) return "positive";
        return "non-positive";
    };
    torch::Tensor pos = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor neg = torch::tensor({-1.0, -2.0, -3.0});
    std::cout << "classify_mean(tensor({1,2,3}))    = " << classify_mean(pos) << std::endl;
    std::cout << "classify_mean(tensor({-1,-2,-3})) = " << classify_mean(neg)
              << " (both branches return a string LITERAL, which plain deduction would type as "
              << "const char* -- the explicit -> std::string trailing return type here converts each one "
              << "into a real std::string instead)" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 01_anatomy_of_a_lambda.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_anatomy_of_a_lambda
./01_anatomy_of_a_lambda
```

```text
add_explicit(3, 4) = 7 (explicit trailing return type -> int)
add_deduced(3, 4)  = 7 (deduced return type, identical result)
square(5.0) = 25 (empty capture [] -- this book's own Chapter 19 Benchmark and Appendix C time_op_ms lambdas have always been this exact shape)
classify_mean(tensor({1,2,3}))    = positive
classify_mean(tensor({-1,-2,-3})) = non-positive (both branches return a string LITERAL, which plain deduction would type as const char* -- the explicit -> std::string trailing return type here converts each one into a real std::string instead)
```

**Discussion.** Nothing about this file's own four lambdas is new C++ -- what is new is naming the four parts explicitly, the same way this book's own Appendix C.1 didn't introduce new hardware facts but did introduce a new frame for facts already in use. Every lambda this book has written since Chapter 1, including the ones inside `Benchmark::time_function` calls and every `std::sort` comparator in later appendices, has this identical four-part shape.

## F.2 A Lambda Has No Name, But It Is a Real Object `[FOUNDATIONAL]`

**Intuition.** A lambda expression has no name of its own, but the compiler still synthesizes a real, anonymous class for it -- with a real `operator()`, a real `sizeof`, and a real, distinct type, confirmed directly via `sizeof` and `typeid` rather than asserted.

**Background.** Two lambda expressions with textually identical source -- same capture, same parameter, same body -- are still two DIFFERENT types, because each lambda EXPRESSION synthesizes its own anonymous class at the point it's written in source, regardless of how it looks. This is the exact problem Section F.7 solves via `std::function`.

**Worked Example F.2.1.** `sizeof` a lambda growing with how much closure state it actually captures; `typeid` confirming two textually-identical lambdas are genuinely different types; and a captured real `torch::Tensor`'s own contribution to closure size.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <typeinfo>

// The CUDA C++ edition's own Section F.2 makes a point this book has
// relied on implicitly since its own first chapter without ever
// stating it directly: a lambda has no name, but it is a REAL
// object -- the compiler synthesizes an anonymous class with a real
// operator(), a real size, and a real, distinct type for every single
// lambda expression written in source, even two that look identical.
int main() {
    int a = 1, b = 2, c = 3;

    // Three lambdas capturing zero, one, and three ints by value --
    // each one's own real sizeof grows with how much closure state it
    // actually holds, exactly the way an ordinary struct's own size
    // grows with its own member count.
    auto captures_nothing = []() { return 0; };
    auto captures_one = [a]() { return a; };
    auto captures_three = [a, b, c]() { return a + b + c; };

    std::cout << "sizeof(captures_nothing) = " << sizeof(captures_nothing)
              << " bytes (captures no state -- the smallest a lambda's own closure object can be, still "
              << "nonzero: every C++ object has at least size 1)" << std::endl;
    std::cout << "sizeof(captures_one)     = " << sizeof(captures_one)
              << " bytes (one captured int)" << std::endl;
    std::cout << "sizeof(captures_three)   = " << sizeof(captures_three)
              << " bytes (three captured ints)" << std::endl;

    // Two lambdas with the IDENTICAL source text -- same capture, same
    // parameter, same body -- are still two DIFFERENT types, because
    // each lambda EXPRESSION synthesizes its own anonymous class, at
    // the point it's written, regardless of what it looks like.
    auto square_v1 = [](int x) { return x * x; };
    auto square_v2 = [](int x) { return x * x; };
    bool same_type = (typeid(square_v1) == typeid(square_v2));
    std::cout << "\ntwo lambdas with identical source text, [](int x){ return x*x; }, written twice: "
              << "typeid(square_v1) == typeid(square_v2)? " << (same_type ? "true" : "false")
              << " -- each lambda EXPRESSION is its own distinct, compiler-synthesized type, no matter "
              << "how textually identical two lambda expressions look" << std::endl;
    std::cout << "  square_v1's own mangled type name: " << typeid(square_v1).name() << std::endl;
    std::cout << "  square_v2's own mangled type name: " << typeid(square_v2).name() << std::endl;

    // Tied to this book's own domain: capturing a real torch::Tensor
    // by value adds the sizeof a torch::Tensor handle itself to the
    // closure -- confirming a captured torch::Tensor is ordinary
    // closure state, the same way a captured int is.
    torch::Tensor t = torch::zeros({100});
    auto captures_tensor = [t]() { return t.numel(); };
    std::cout << "\nsizeof(captures_tensor) = " << sizeof(captures_tensor)
              << " bytes (captures one torch::Tensor by value -- adds sizeof(torch::Tensor) = "
              << sizeof(torch::Tensor) << " bytes to the closure, confirming a captured Tensor handle is "
              << "ordinary closure state like any other captured value, regardless of how many elements "
              << "the underlying storage it points to actually holds -- captures_tensor()'s own numel() "
              << "reports " << captures_tensor() << ", the real element count of the storage it refers to)"
              << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 02_lambda_is_a_real_object.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_lambda_is_a_real_object
./02_lambda_is_a_real_object
```

```text
sizeof(captures_nothing) = 1 bytes (captures no state -- the smallest a lambda's own closure object can be, still nonzero: every C++ object has at least size 1)
sizeof(captures_one)     = 4 bytes (one captured int)
sizeof(captures_three)   = 12 bytes (three captured ints)

two lambdas with identical source text, [](int x){ return x*x; }, written twice: typeid(square_v1) == typeid(square_v2)? false -- each lambda EXPRESSION is its own distinct, compiler-synthesized type, no matter how textually identical two lambda expressions look
  square_v1's own mangled type name: Z4mainEUliE_
  square_v2's own mangled type name: Z4mainEUliE0_

sizeof(captures_tensor) = 8 bytes (captures one torch::Tensor by value -- adds sizeof(torch::Tensor) = 8 bytes to the closure, confirming a captured Tensor handle is ordinary closure state like any other captured value, regardless of how many elements the underlying storage it points to actually holds -- captures_tensor()'s own numel() reports 100, the real element count of the storage it refers to)
```

**Discussion.** The `sizeof` progression (1, 4, 12 bytes for zero, one, and three captured `int`s) is exactly what an ordinary struct with the equivalent members would report -- a lambda's own closure object follows the identical layout rules any other class does, because it genuinely is one. The `typeid` comparison makes the same point a different way: `square_v1` and `square_v2`, despite being copy-pasted identically, are provably different types, which is precisely why a `std::vector` cannot hold "a lambda" generically -- it can only hold one SPECIFIC lambda's own type, or, as Section F.7 shows, a type-erased wrapper around any of them.

## F.3 The Capture Clause in Full `[FOUNDATIONAL]`

**Intuition.** C++'s full capture-clause vocabulary: `[]` (nothing), `[x]` (x by value), `[&x]` (x by reference), `[x,&y]` (a mix, per name), `[=]` (everything used, by value), `[&]` (everything used, by reference), `[this]` (the enclosing object, by pointer), and `[*this]` (C++17: a full copy of the enclosing object).

**Background.** `[this]` captures a POINTER to the enclosing object -- mutations through it are visible to anyone else still holding that object. `[*this]` captures a full COPY of the enclosing object at the moment the lambda is created -- mutations through it touch only that private copy, never the original, confirmed here on a class holding a real `torch::Tensor` member rather than a plain `double`.

**Worked Example F.3.1.** A mixed by-value/by-reference capture; then a `GradientAccumulator` class demonstrating `[this]` (mutation visible on the original) against `[*this]` (mutation isolated to the closure's own private copy, including its own copy of a real `torch::Tensor`).

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Section F.3 gives the full table of
// capture-clause forms: [] (nothing), [x] (x by value), [&x] (x by
// reference), [x,&y] (a mix, per-name), [=] (everything used, by
// value), [&] (everything used, by reference), [this] (the enclosing
// object, by pointer), and [*this] (C++17: a full copy of the
// enclosing object). This file demonstrates the by-value/by-reference
// distinction directly, then the [this]/[*this] distinction on a
// real class holding a torch::Tensor member -- tying the CUDA book's
// own Accumulator example directly to this book's own domain.
int main() {
    // [x,&y]: a mixed, per-name capture -- p by value (frozen at
    // capture time), q by reference (mutated for real).
    int p = 10, q = 20;
    auto mixed = [p, &q](int add) {
        q += add;
        return p + q;
    };
    std::cout << "p=10, q=20 initially. mixed(5) = " << mixed(5)
              << " (p captured BY VALUE, q captured BY REFERENCE)" << std::endl;
    std::cout << "  after the call: p = " << p << " (unchanged, it was only a copy), q = " << q
              << " (genuinely mutated, since mixed held a real reference to it)" << std::endl;

    // A small struct holding a real torch::Tensor member, to
    // demonstrate [this] (capture the enclosing object BY POINTER,
    // mutations visible through the original) against [*this]
    // (C++17: capture a full COPY of the enclosing object, mutations
    // visible only in the lambda's own private copy).
    struct GradientAccumulator {
        torch::Tensor total = torch::zeros({3});

        auto make_adder_by_this() {
            // [this]: captures a pointer to THIS object. Every call
            // reassigns the REAL total member -- the mutation is
            // visible to anyone still holding a reference to this
            // same GradientAccumulator.
            return [this](const torch::Tensor& grad) { total = total + grad; return total; };
        }

        auto make_adder_by_copy() {
            // [*this] (C++17): captures a full COPY of this object,
            // including its own copy of `total`. torch::Tensor's own
            // copy constructor gives that copy a new Tensor HANDLE --
            // reassigning the copy's own `total` (via `total = total
            // + grad`, which builds a brand-new tensor rather than
            // mutating storage in place) never touches the original
            // object's own `total` at all.
            return [*this](const torch::Tensor& grad) mutable { total = total + grad; return total; };
        }
    };

    GradientAccumulator acc;
    auto add_via_this = acc.make_adder_by_this();
    add_via_this(torch::tensor({1.0, 1.0, 1.0}));
    add_via_this(torch::tensor({2.0, 2.0, 2.0}));
    std::cout << "\n[this]-capturing lambda: acc.total after two add_via_this() calls = " << acc.total
              << " -- mutations through a [this] capture are genuinely visible on the ORIGINAL object"
              << std::endl;

    GradientAccumulator acc2;
    auto add_via_copy = acc2.make_adder_by_copy();
    torch::Tensor copy_result = add_via_copy(torch::tensor({5.0, 5.0, 5.0}));
    std::cout << "\n[*this]-capturing lambda: the closure's OWN private copy's total, returned by "
              << "add_via_copy() = " << copy_result << ", but the ORIGINAL acc2.total = " << acc2.total
              << " (untouched -- [*this] copied the whole GradientAccumulator, including its own "
              << "torch::Tensor member, at the moment the lambda was created, and the mutation above only "
              << "ever touched that private copy's own state)" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 03_capture_clause_in_full.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_capture_clause_in_full
./03_capture_clause_in_full
```

```text
p=10, q=20 initially. mixed(5) = 35 (p captured BY VALUE, q captured BY REFERENCE)
  after the call: p = 10 (unchanged, it was only a copy), q = 25 (genuinely mutated, since mixed held a real reference to it)

[this]-capturing lambda: acc.total after two add_via_this() calls =  3
 3
 3
[ CPUFloatType{3} ] -- mutations through a [this] capture are genuinely visible on the ORIGINAL object

[*this]-capturing lambda: the closure's OWN private copy's total, returned by add_via_copy() =  5
 5
 5
[ CPUFloatType{3} ], but the ORIGINAL acc2.total =  0
 0
 0
[ CPUFloatType{3} ] (untouched -- [*this] copied the whole GradientAccumulator, including its own torch::Tensor member, at the moment the lambda was created, and the mutation above only ever touched that private copy's own state)
```

**Discussion.** `p` stays 10 while `q` becomes 25 -- the ordinary by-value/by-reference distinction, restated here as the foundation Section F.4 and this appendix's own Appendix D.6 (capture-by-value vs. capture-by-reference, in a CUDA-adjacent context) both build on. The `GradientAccumulator` result is the more interesting confirmation: `acc.total` genuinely changes to reflect two real tensor additions when accessed through a `[this]`-capturing lambda, while `acc2.total` stays exactly `torch::zeros({3})` when accessed through a `[*this]`-capturing lambda, because `[*this]` copied the entire `GradientAccumulator` -- including a fresh copy of its own `torch::Tensor` member -- at the moment the lambda was created, and every mutation afterward reassigns only that private copy's own handle.

## F.4 Mutable Lambdas and Why Lambdas Are `const` by Default `[FOUNDATIONAL]`

**Intuition.** A non-`mutable` lambda's own `operator()` is implicitly `const`, following directly from Section F.2's own finding that a lambda is a real class -- a captured-by-value variable cannot be modified inside the body at all, by default, the identical rule that would apply to any other class's `const` member function.

**Background.** This section demonstrates the resulting compile error directly rather than asserting it: a small, broken snippet was compiled SEPARATELY, on purpose, using the identical `g++` invocation every file in this book uses, specifically to capture its own real, genuine error text -- not assumed, not paraphrased from the CUDA book's own reported message.

**Worked Example F.4.1.** The real, genuinely captured compiler error from attempting to mutate a by-value capture without `mutable`; then the working fix, called three times, confirming the outer variable stays untouched.

```cpp
#include <iostream>

// The CUDA C++ edition's own Section F.4 makes a point that follows
// directly from Section F.2's own finding (a lambda is a real,
// compiler-synthesized class): a non-mutable lambda's own operator()
// is implicitly const, exactly the way any other class's member
// function would be if declared const -- so a captured-by-value
// variable cannot be modified inside the lambda body at all, by
// default.
//
// The broken snippet below is NOT part of this file's own compiled
// source (this file must compile cleanly, the same convention every
// other genuinely-run source file in this book follows) -- it was
// compiled SEPARATELY, on purpose, specifically to capture ITS OWN
// real, genuine g++ error text rather than assume what the compiler
// would say:
//
//   int counter = 0;
//   auto broken_counter = [counter]() { counter++; return counter; };
//
// Compiled with the identical g++ invocation every file in this book
// uses, this genuinely fails with:
//
//   error: increment of read-only variable 'counter'
//
// -- confirmed directly, by actually compiling that exact snippet in
// this sandbox, not assumed from the CUDA book's own reported text.
int main() {
    int counter = 0;

    // The fix: `mutable` removes operator()'s own implicit const,
    // letting the lambda's own PRIVATE COPY of `counter` be modified
    // across calls -- the outer `counter` is still never touched,
    // exactly as Section F.3's own [*this] example already
    // demonstrated for a whole captured object rather than one int.
    auto working_counter = [counter]() mutable {
        counter++;
        return counter;
    };

    std::cout << "the broken version above (captured by value, no mutable) fails to compile with a real, "
              << "genuine g++ error: \"increment of read-only variable 'counter'\" -- confirmed by actually "
              << "compiling that exact snippet separately in this sandbox." << std::endl;

    std::cout << "\nthe fixed version, [counter]() mutable { counter++; return counter; }, called three "
              << "times: " << working_counter() << ", " << working_counter() << ", " << working_counter()
              << " (the lambda's own private copy increments across calls)" << std::endl;
    std::cout << "outer counter after those three calls = " << counter
              << " (still 0 -- mutable only lifts constness on the CLOSURE'S OWN COPY, it never reaches "
              << "back to the variable that was captured)" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 04_mutable_and_const_by_default.cpp -o 04_mutable_and_const_by_default
./04_mutable_and_const_by_default
```

```text
the broken version above (captured by value, no mutable) fails to compile with a real, genuine g++ error: "increment of read-only variable 'counter'" -- confirmed by actually compiling that exact snippet separately in this sandbox.

the fixed version, [counter]() mutable { counter++; return counter; }, called three times: 1, 2, 3 (the lambda's own private copy increments across calls)
outer counter after those three calls = 0 (still 0 -- mutable only lifts constness on the CLOSURE'S OWN COPY, it never reaches back to the variable that was captured)
```

**Discussion.** `mutable` lifts `operator()`'s own implicit `const`, letting the lambda's own PRIVATE COPY of `counter` change across calls (1, 2, 3) -- but the outer `counter` stays 0 throughout, because `mutable` only ever reaches the closure's own copy, never back to the variable that was originally captured. This is the identical structural distinction Section F.3's own `[this]`/`[*this]` comparison already made for a whole object rather than one `int`, and it is the exact same idea Appendix D.6's own `by_value_mutable` example demonstrates in this book's own CUDA-adjacent appendix -- one host-only, one written to motivate a CUDA-specific dangling-reference discussion, but the same C++ rule underneath both.

## F.5 Generalized (Init) Capture: `[name = expr]` `[FOUNDATIONAL]`

**Intuition.** C++14's generalized (init) capture constructs brand-new, closure-PRIVATE state from an arbitrary expression, with no external variable required at all -- `[velocity = torch::Tensor()]` creates a real `torch::Tensor` that exists nowhere outside the closure that owns it.

**Background.** This section's own worked example is a momentum-style optimizer smoother, tying directly into the momentum-based optimizers this book's own neural-network chapters (Part 6) already assume readers understand conceptually: `velocity = beta * velocity + gradient`, applied repeatedly, exactly the recurrence a momentum optimizer's own update rule follows.

**Worked Example F.5.1.** A momentum smoother built entirely around a generalized-captured `torch::Tensor`, applied three times to the same gradient, cross-checked against an independent hand-unrolled scalar computation; then a second, independent smoother instance confirming per-closure state isolation.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Section F.5 covers C++14's generalized
// (init) capture, `[name = expr]` -- constructing brand-new,
// closure-PRIVATE state from an arbitrary expression, with no
// external variable required at all. Its own worked example is a
// momentum-style optimizer smoother, built around a std::vector<double>;
// this book's own version reproduces the identical idea around a real
// torch::Tensor instead, tying it directly to the momentum-based
// optimizers this book's own Chapter 20/21 material assumes readers
// already understand conceptually.
auto make_momentum_smoother(double beta) {
    // [beta, velocity = torch::Tensor()]: beta is an ordinary
    // by-value capture; velocity is a GENERALIZED capture -- a
    // brand-new torch::Tensor, owned entirely by this closure, that
    // exists nowhere outside it. `mutable` is required because the
    // lambda body reassigns its own captured `velocity` on every call.
    return [beta, velocity = torch::Tensor()](const torch::Tensor& gradient) mutable {
        if (!velocity.defined()) {
            velocity = torch::zeros_like(gradient);
        }
        velocity = beta * velocity + gradient;
        return velocity;
    };
}

int main() {
    double beta = 0.9;
    torch::Tensor gradient = torch::tensor({1.0, 2.0, 3.0});

    auto smoother = make_momentum_smoother(beta);

    torch::Tensor v1 = smoother(gradient);
    torch::Tensor v2 = smoother(gradient);
    torch::Tensor v3 = smoother(gradient);

    std::cout << "make_momentum_smoother(0.9), the same gradient tensor {1,2,3} applied 3 times in a row:"
              << std::endl;
    std::cout << "  step 1: " << v1.flatten() << std::endl;
    std::cout << "  step 2: " << v2.flatten() << std::endl;
    std::cout << "  step 3: " << v3.flatten() << std::endl;

    // Independent hand-unrolled check for element 0: v0=0,
    // v1=0.9*0+1=1.0, v2=0.9*1.0+1=1.9, v3=0.9*1.9+1=2.71.
    double v0 = 0.0;
    double hand_v1 = beta * v0 + 1.0;
    double hand_v2 = beta * hand_v1 + 1.0;
    double hand_v3 = beta * hand_v2 + 1.0;
    // The tensors above are ordinary torch::Tensor default dtype
    // (float32), while the hand-unrolled check above is double
    // arithmetic -- so an exact bitwise match isn't expected. A
    // tolerance of 1e-6 (loose enough to absorb float32's own real
    // rounding at each of the three chained multiply-adds, tight
    // enough to catch any genuine logic error) is used instead.
    bool matches = std::abs(v1[0].item<double>() - hand_v1) < 1e-6
                   && std::abs(v2[0].item<double>() - hand_v2) < 1e-6
                   && std::abs(v3[0].item<double>() - hand_v3) < 1e-6;
    std::cout << "\nindependent hand-unrolled check for element 0: v1=" << hand_v1 << ", v2=" << hand_v2
              << ", v3=" << hand_v3 << " -- matches the real tensor's own element 0 at each step (within "
              << "1e-6, since the tensor is float32 and this hand check is double)? " << matches << std::endl;

    // A second, independently created smoother starts fresh -- proof
    // that `velocity` is genuinely PER-CLOSURE state, not some shared
    // global the first smoother happened to also touch.
    auto smoother2 = make_momentum_smoother(beta);
    torch::Tensor v1_again = smoother2(gradient);
    std::cout << "\na second, independent make_momentum_smoother(0.9) instance, same gradient, step 1: "
              << v1_again.flatten() << " -- matches the FIRST smoother's own step 1 exactly, confirming "
              << "each closure owns its own private velocity, entirely isolated from the other's" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 05_generalized_init_capture.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 05_generalized_init_capture
./05_generalized_init_capture
```

```text
make_momentum_smoother(0.9), the same gradient tensor {1,2,3} applied 3 times in a row:
  step 1:  1
 2
 3
[ CPUFloatType{3} ]
  step 2:  1.9000
 3.8000
 5.7000
[ CPUFloatType{3} ]
  step 3:  2.7100
 5.4200
 8.1300
[ CPUFloatType{3} ]

independent hand-unrolled check for element 0: v1=1, v2=1.9, v3=2.71 -- matches the real tensor's own element 0 at each step (within 1e-6, since the tensor is float32 and this hand check is double)? 1

a second, independent make_momentum_smoother(0.9) instance, same gradient, step 1:  1
 2
 3
[ CPUFloatType{3} ] -- matches the FIRST smoother's own step 1 exactly, confirming each closure owns its own private velocity, entirely isolated from the other's
```

**Discussion.** The three steps (1.0/2.0/3.0, then 1.9/3.8/5.7, then 2.71/5.42/8.13) match the independent hand-unrolled check for element 0 to within float32 precision, and the second, independently created smoother's own first step matches the first smoother's own first step exactly -- confirming `velocity` is genuinely PER-CLOSURE state, isolated the same way a class's own private member would be, not some accidental shared value the two closures happened to both touch. Where Section F.3's `[this]`/`[*this]` distinction is about SHARING versus ISOLATING an already-existing object, generalized capture is about CREATING new, isolated state that never existed outside the closure in the first place.

## F.6 Lambdas as Comparators and Predicates `[FOUNDATIONAL]`

**Intuition.** `std::sort`, `std::count_if`, and `std::find_if` are ordinary standard-library algorithms parameterized by a callable -- the identical "generic function parameterized by a callable" shape this book's own Appendix D.2 and D.3 already used for `torch::Tensor`-flavored data, applied here to the standard library's own algorithms instead of a hand-written `template<typename F>` function.

**Background.** A comparator or predicate can be written inline or held in a named variable first -- the algorithm itself doesn't care, the same "a callable is a callable" point Appendix D.2 already made about lambdas and functors both satisfying an identical `operator()` interface.

**Worked Example F.6.1.** Ascending and descending sorts of a small vector of bond z-spreads (basis points, in the spirit of Chapter 22.1's own bond-portfolio data); `count_if`/`find_if` with a hard-coded threshold; then a capturing predicate confirmed equivalent to the hard-coded version.

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section F.6 uses std::sort, std::count_if,
// and std::find_if -- ordinary standard-library algorithms that take
// a callable (a comparator or a predicate) as one of their own
// parameters, the exact "generic function parameterized by a
// callable" shape this book's own Appendix D.2/D.3 already used for
// torch::Tensor-flavored data. This section reproduces the identical
// three algorithms, over a small vector of z-spreads (in basis
// points) reminiscent of Chapter 22.1's own bond-portfolio data,
// rather than the CUDA book's own plain integers.
int main() {
    std::vector<double> spreads_bp = {52.0, 21.0, 88.0, 13.0, 95.0, 34.0};

    std::cout << "spreads (bp): ";
    for (double s : spreads_bp) std::cout << s << " ";
    std::cout << std::endl;

    // An inline lambda comparator, ascending.
    std::vector<double> ascending = spreads_bp;
    std::sort(ascending.begin(), ascending.end(), [](double a, double b) { return a < b; });
    std::cout << "\nascending  (inline lambda a < b): ";
    for (double s : ascending) std::cout << s << " ";
    std::cout << std::endl;

    // A comparator stored in a NAMED variable first -- std::sort
    // doesn't care whether the comparator is written inline or held
    // in a variable first, the same "a callable is a callable"
    // structural point Appendix D.2 already made about lambdas and
    // functors both satisfying the identical operator() interface.
    auto compare_descending = [](double a, double b) { return a > b; };
    std::vector<double> descending = spreads_bp;
    std::sort(descending.begin(), descending.end(), compare_descending);
    std::cout << "descending (named lambda a > b) : ";
    for (double s : descending) std::cout << s << " ";
    std::cout << std::endl;

    // std::count_if / std::find_if with a hard-coded-threshold predicate.
    long count_over_40 = std::count_if(spreads_bp.begin(), spreads_bp.end(), [](double s) { return s > 40.0; });
    auto first_even_ish = std::find_if(spreads_bp.begin(), spreads_bp.end(),
                                        [](double s) { return static_cast<long>(s) % 2 == 0; });
    std::cout << "\ncount_if(s > 40.0)                  = " << count_over_40 << std::endl;
    std::cout << "find_if(static_cast<long>(s) % 2 == 0) found = " << *first_even_ish
              << " (the first spread whose truncated integer part is even)" << std::endl;

    // A capturing predicate, parameterized by an ordinary variable
    // instead of a hard-coded literal -- avoiding hard-coding the
    // threshold is the entire reason a capture list exists at all.
    double threshold = 40.0;
    long count_over_threshold = std::count_if(spreads_bp.begin(), spreads_bp.end(),
                                               [threshold](double s) { return s > threshold; });
    std::cout << "\na capturing predicate, [threshold](double s){ return s > threshold; }, threshold=40.0: "
              << "count_over_threshold = " << count_over_threshold
              << " -- matches the hard-coded count_if(s > 40.0) above exactly (" << count_over_40
              << "), confirming the capture-based version is equivalent, not merely similar" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 06_lambdas_as_comparators_and_predicates.cpp -o 06_lambdas_as_comparators_and_predicates
./06_lambdas_as_comparators_and_predicates
```

```text
spreads (bp): 52 21 88 13 95 34 

ascending  (inline lambda a < b): 13 21 34 52 88 95 
descending (named lambda a > b) : 95 88 52 34 21 13 

count_if(s > 40.0)                  = 3
find_if(static_cast<long>(s) % 2 == 0) found = 52 (the first spread whose truncated integer part is even)

a capturing predicate, [threshold](double s){ return s > threshold; }, threshold=40.0: count_over_threshold = 3 -- matches the hard-coded count_if(s > 40.0) above exactly (3), confirming the capture-based version is equivalent, not merely similar
```

**Discussion.** The capturing predicate (`[threshold](double s){ return s > threshold; }`) produces exactly the same count (3) as the hard-coded `s > 40.0` version -- confirming capture doesn't change WHAT the predicate computes, only where the comparison value comes from, which is the entire point of using a capture at all: the identical predicate SHAPE can be reused against any threshold a caller supplies, rather than one baked-in literal.

## F.7 `std::function`, Type Erasure, and Closing the Loop on Section F.2 `[FOUNDATIONAL]`

**Intuition.** Section F.2 already showed the problem this section solves: two lambdas with identical signatures are still two DIFFERENT, incompatible types, so a plain container can't hold several different lambdas at once. `std::function<Signature>` is the standard library's own answer -- type erasure, wrapping ANY callable matching a given signature behind one uniform, storable, copyable type.

**Background.** This book's own Appendix D.4 already demonstrated `std::function` working normally in ordinary host code at length, contrasting it with the CUDA C++ edition's own Section D.7 finding that `std::function`'s vtable-driven dispatch cannot cross onto a CUDA device as a kernel parameter. This section doesn't repeat that demonstration -- it restates the general-purpose C++ feature plainly, with one fresh, short example, and connects it directly back to Section F.2's own finding rather than to Appendix D's CUDA-specific angle.

**Worked Example F.7.1.** Three lambdas of three provably different types (Section F.2), all stored in one `std::vector<std::function<int(int)>>`, applied uniformly.

```cpp
#include <functional>
#include <iostream>
#include <vector>

// Section F.2 of this appendix already showed the problem this
// section solves: two lambdas with identical signatures are still
// two DIFFERENT, incompatible compiler-synthesized types -- so a
// plain std::vector<auto> holding several different lambdas isn't
// even expressible, let alone useful. std::function<Signature> is
// the standard library's own answer: TYPE ERASURE, wrapping ANY
// callable matching a given signature (a lambda, a functor, a
// function pointer) behind one uniform, storable, copyable type.
//
// This book's own Appendix D.4 already demonstrated std::function
// working normally in ordinary host code at length (a std::function
// wrapping a real torch::matmul call, timed via a host helper
// modeled on Chapter 19's Benchmark::time_function), and explicitly
// contrasted that with the CUDA C++ edition's own Section D.7 finding
// that std::function's own vtable-driven dispatch cannot cross onto a
// CUDA device as a kernel PARAMETER. This section doesn't repeat that
// demonstration -- it states the general-purpose C++ feature plainly,
// with one fresh, short example, and connects it back to exactly the
// problem Section F.2 raised.
int main() {
    int offset = 100;

    // Three lambdas, three DIFFERENT compiler-synthesized types
    // (Section F.2's own finding) -- but std::function<int(int)>
    // erases all three down to one uniform, storable type, so a
    // single std::vector CAN hold all three side by side.
    std::vector<std::function<int(int)>> operations;
    operations.push_back([](int x) { return x * 2; });
    operations.push_back([offset](int x) { return x + offset; });
    operations.push_back([](int x) { return x * x; });

    std::cout << "three lambdas of three genuinely different compiler-synthesized types (Section F.2's "
              << "own finding), all stored in one std::vector<std::function<int(int)>>, each applied to "
              << "x = 7:" << std::endl;
    for (size_t i = 0; i < operations.size(); i++) {
        std::cout << "  operations[" << i << "](7) = " << operations[i](7) << std::endl;
    }

    std::cout << "\nthis compiles and runs with plain g++, no CUDA toolchain, no --extended-lambda flag "
              << "anywhere -- Appendix D.4's own point restated in one sentence: Section D.7's own CUDA "
              << "restriction is specific to a std::function used as a __global__ kernel PARAMETER; "
              << "ordinary host code calling std::function::operator(), exactly as this file does, is the "
              << "case std::function's own type-erasure mechanism was built for in the first place."
              << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 07_std_function_and_type_erasure.cpp -o 07_std_function_and_type_erasure
./07_std_function_and_type_erasure
```

```text
three lambdas of three genuinely different compiler-synthesized types (Section F.2's own finding), all stored in one std::vector<std::function<int(int)>>, each applied to x = 7:
  operations[0](7) = 14
  operations[1](7) = 107
  operations[2](7) = 49

this compiles and runs with plain g++, no CUDA toolchain, no --extended-lambda flag anywhere -- Appendix D.4's own point restated in one sentence: Section D.7's own CUDA restriction is specific to a std::function used as a __global__ kernel PARAMETER; ordinary host code calling std::function::operator(), exactly as this file does, is the case std::function's own type-erasure mechanism was built for in the first place.
```

**Discussion.** This file compiles and runs with plain `g++`, no CUDA toolchain, no `--extended-lambda` flag anywhere -- the same honest point Appendix D.4 already made at greater length: the CUDA C++ edition's own Section D.7 restriction is specific to a `std::function` used as a `__global__` kernel PARAMETER, not to `std::function` itself. Ordinary host code calling `std::function::operator()`, exactly as every file in this appendix and Appendix D.4 already does, is the case type erasure was built for in the first place -- and it is, directly, the fix for the exact problem Section F.2 first demonstrated three sections ago.

## F.8 Complete Runnable Code

### `01_anatomy_of_a_lambda.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Appendix F opens by stepping back from
// Appendix D's own CUDA-specific vocabulary (extended lambdas, the
// --extended-lambda flag, device-closure residence) to the plain C++
// lambda feature those extensions are built on top of. This book has
// used ordinary lambdas in nearly every chapter since Chapter 1
// without ever naming their own anatomy explicitly -- this appendix
// is that missing foundation, written the way this book's own
// Appendix D already assumed a reader had it: four fixed parts, in
// order, in every lambda -- a capture clause `[]`, a parameter list
// `()`, an optional trailing return type, and a body `{}`.
int main() {
    // An explicit trailing return type versus a deduced one -- both
    // behaviorally identical, differing only in whether the compiler
    // is told the return type or works it out from the body's own
    // return statement.
    auto add_explicit = [](int a, int b) -> int { return a + b; };
    auto add_deduced = [](int a, int b) { return a + b; };

    std::cout << "add_explicit(3, 4) = " << add_explicit(3, 4)
              << " (explicit trailing return type -> int)" << std::endl;
    std::cout << "add_deduced(3, 4)  = " << add_deduced(3, 4)
              << " (deduced return type, identical result)" << std::endl;

    // An empty-capture lambda -- the smallest, most common shape this
    // book's own chapters have already used dozens of times (every
    // sort comparator, every std::function<void()> timing wrapper in
    // Chapter 19 and Appendix C), here named explicitly for the first
    // time: capture nothing, take one parameter, return one value.
    auto square = [](double x) { return x * x; };
    std::cout << "square(5.0) = " << square(5.0)
              << " (empty capture [] -- this book's own Chapter 19 Benchmark and Appendix C time_op_ms "
              << "lambdas have always been this exact shape)" << std::endl;

    // A trailing return type is genuinely useful, beyond mere style,
    // whenever plain return-type deduction would infer something
    // narrower than what's actually wanted -- here, both branches
    // return a string LITERAL (deducible as `const char*`), but an
    // explicit `-> std::string` tells the compiler to convert each
    // one into a real std::string instead, this book's own domain
    // example: classifying a real torch::Tensor's mean by sign.
    auto classify_mean = [](const torch::Tensor& t) -> std::string {
        double m = t.mean().item<double>();
        if (m > 0.0) return "positive";
        return "non-positive";
    };
    torch::Tensor pos = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor neg = torch::tensor({-1.0, -2.0, -3.0});
    std::cout << "classify_mean(tensor({1,2,3}))    = " << classify_mean(pos) << std::endl;
    std::cout << "classify_mean(tensor({-1,-2,-3})) = " << classify_mean(neg)
              << " (both branches return a string LITERAL, which plain deduction would type as "
              << "const char* -- the explicit -> std::string trailing return type here converts each one "
              << "into a real std::string instead)" << std::endl;

    return 0;
}
```

### `02_lambda_is_a_real_object.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <typeinfo>

// The CUDA C++ edition's own Section F.2 makes a point this book has
// relied on implicitly since its own first chapter without ever
// stating it directly: a lambda has no name, but it is a REAL
// object -- the compiler synthesizes an anonymous class with a real
// operator(), a real size, and a real, distinct type for every single
// lambda expression written in source, even two that look identical.
int main() {
    int a = 1, b = 2, c = 3;

    // Three lambdas capturing zero, one, and three ints by value --
    // each one's own real sizeof grows with how much closure state it
    // actually holds, exactly the way an ordinary struct's own size
    // grows with its own member count.
    auto captures_nothing = []() { return 0; };
    auto captures_one = [a]() { return a; };
    auto captures_three = [a, b, c]() { return a + b + c; };

    std::cout << "sizeof(captures_nothing) = " << sizeof(captures_nothing)
              << " bytes (captures no state -- the smallest a lambda's own closure object can be, still "
              << "nonzero: every C++ object has at least size 1)" << std::endl;
    std::cout << "sizeof(captures_one)     = " << sizeof(captures_one)
              << " bytes (one captured int)" << std::endl;
    std::cout << "sizeof(captures_three)   = " << sizeof(captures_three)
              << " bytes (three captured ints)" << std::endl;

    // Two lambdas with the IDENTICAL source text -- same capture, same
    // parameter, same body -- are still two DIFFERENT types, because
    // each lambda EXPRESSION synthesizes its own anonymous class, at
    // the point it's written, regardless of what it looks like.
    auto square_v1 = [](int x) { return x * x; };
    auto square_v2 = [](int x) { return x * x; };
    bool same_type = (typeid(square_v1) == typeid(square_v2));
    std::cout << "\ntwo lambdas with identical source text, [](int x){ return x*x; }, written twice: "
              << "typeid(square_v1) == typeid(square_v2)? " << (same_type ? "true" : "false")
              << " -- each lambda EXPRESSION is its own distinct, compiler-synthesized type, no matter "
              << "how textually identical two lambda expressions look" << std::endl;
    std::cout << "  square_v1's own mangled type name: " << typeid(square_v1).name() << std::endl;
    std::cout << "  square_v2's own mangled type name: " << typeid(square_v2).name() << std::endl;

    // Tied to this book's own domain: capturing a real torch::Tensor
    // by value adds the sizeof a torch::Tensor handle itself to the
    // closure -- confirming a captured torch::Tensor is ordinary
    // closure state, the same way a captured int is.
    torch::Tensor t = torch::zeros({100});
    auto captures_tensor = [t]() { return t.numel(); };
    std::cout << "\nsizeof(captures_tensor) = " << sizeof(captures_tensor)
              << " bytes (captures one torch::Tensor by value -- adds sizeof(torch::Tensor) = "
              << sizeof(torch::Tensor) << " bytes to the closure, confirming a captured Tensor handle is "
              << "ordinary closure state like any other captured value, regardless of how many elements "
              << "the underlying storage it points to actually holds -- captures_tensor()'s own numel() "
              << "reports " << captures_tensor() << ", the real element count of the storage it refers to)"
              << std::endl;

    return 0;
}
```

### `03_capture_clause_in_full.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Section F.3 gives the full table of
// capture-clause forms: [] (nothing), [x] (x by value), [&x] (x by
// reference), [x,&y] (a mix, per-name), [=] (everything used, by
// value), [&] (everything used, by reference), [this] (the enclosing
// object, by pointer), and [*this] (C++17: a full copy of the
// enclosing object). This file demonstrates the by-value/by-reference
// distinction directly, then the [this]/[*this] distinction on a
// real class holding a torch::Tensor member -- tying the CUDA book's
// own Accumulator example directly to this book's own domain.
int main() {
    // [x,&y]: a mixed, per-name capture -- p by value (frozen at
    // capture time), q by reference (mutated for real).
    int p = 10, q = 20;
    auto mixed = [p, &q](int add) {
        q += add;
        return p + q;
    };
    std::cout << "p=10, q=20 initially. mixed(5) = " << mixed(5)
              << " (p captured BY VALUE, q captured BY REFERENCE)" << std::endl;
    std::cout << "  after the call: p = " << p << " (unchanged, it was only a copy), q = " << q
              << " (genuinely mutated, since mixed held a real reference to it)" << std::endl;

    // A small struct holding a real torch::Tensor member, to
    // demonstrate [this] (capture the enclosing object BY POINTER,
    // mutations visible through the original) against [*this]
    // (C++17: capture a full COPY of the enclosing object, mutations
    // visible only in the lambda's own private copy).
    struct GradientAccumulator {
        torch::Tensor total = torch::zeros({3});

        auto make_adder_by_this() {
            // [this]: captures a pointer to THIS object. Every call
            // reassigns the REAL total member -- the mutation is
            // visible to anyone still holding a reference to this
            // same GradientAccumulator.
            return [this](const torch::Tensor& grad) { total = total + grad; return total; };
        }

        auto make_adder_by_copy() {
            // [*this] (C++17): captures a full COPY of this object,
            // including its own copy of `total`. torch::Tensor's own
            // copy constructor gives that copy a new Tensor HANDLE --
            // reassigning the copy's own `total` (via `total = total
            // + grad`, which builds a brand-new tensor rather than
            // mutating storage in place) never touches the original
            // object's own `total` at all.
            return [*this](const torch::Tensor& grad) mutable { total = total + grad; return total; };
        }
    };

    GradientAccumulator acc;
    auto add_via_this = acc.make_adder_by_this();
    add_via_this(torch::tensor({1.0, 1.0, 1.0}));
    add_via_this(torch::tensor({2.0, 2.0, 2.0}));
    std::cout << "\n[this]-capturing lambda: acc.total after two add_via_this() calls = " << acc.total
              << " -- mutations through a [this] capture are genuinely visible on the ORIGINAL object"
              << std::endl;

    GradientAccumulator acc2;
    auto add_via_copy = acc2.make_adder_by_copy();
    torch::Tensor copy_result = add_via_copy(torch::tensor({5.0, 5.0, 5.0}));
    std::cout << "\n[*this]-capturing lambda: the closure's OWN private copy's total, returned by "
              << "add_via_copy() = " << copy_result << ", but the ORIGINAL acc2.total = " << acc2.total
              << " (untouched -- [*this] copied the whole GradientAccumulator, including its own "
              << "torch::Tensor member, at the moment the lambda was created, and the mutation above only "
              << "ever touched that private copy's own state)" << std::endl;

    return 0;
}
```

### `04_mutable_and_const_by_default.cpp`

```cpp
#include <iostream>

// The CUDA C++ edition's own Section F.4 makes a point that follows
// directly from Section F.2's own finding (a lambda is a real,
// compiler-synthesized class): a non-mutable lambda's own operator()
// is implicitly const, exactly the way any other class's member
// function would be if declared const -- so a captured-by-value
// variable cannot be modified inside the lambda body at all, by
// default.
//
// The broken snippet below is NOT part of this file's own compiled
// source (this file must compile cleanly, the same convention every
// other genuinely-run source file in this book follows) -- it was
// compiled SEPARATELY, on purpose, specifically to capture ITS OWN
// real, genuine g++ error text rather than assume what the compiler
// would say:
//
//   int counter = 0;
//   auto broken_counter = [counter]() { counter++; return counter; };
//
// Compiled with the identical g++ invocation every file in this book
// uses, this genuinely fails with:
//
//   error: increment of read-only variable 'counter'
//
// -- confirmed directly, by actually compiling that exact snippet in
// this sandbox, not assumed from the CUDA book's own reported text.
int main() {
    int counter = 0;

    // The fix: `mutable` removes operator()'s own implicit const,
    // letting the lambda's own PRIVATE COPY of `counter` be modified
    // across calls -- the outer `counter` is still never touched,
    // exactly as Section F.3's own [*this] example already
    // demonstrated for a whole captured object rather than one int.
    auto working_counter = [counter]() mutable {
        counter++;
        return counter;
    };

    std::cout << "the broken version above (captured by value, no mutable) fails to compile with a real, "
              << "genuine g++ error: \"increment of read-only variable 'counter'\" -- confirmed by actually "
              << "compiling that exact snippet separately in this sandbox." << std::endl;

    std::cout << "\nthe fixed version, [counter]() mutable { counter++; return counter; }, called three "
              << "times: " << working_counter() << ", " << working_counter() << ", " << working_counter()
              << " (the lambda's own private copy increments across calls)" << std::endl;
    std::cout << "outer counter after those three calls = " << counter
              << " (still 0 -- mutable only lifts constness on the CLOSURE'S OWN COPY, it never reaches "
              << "back to the variable that was captured)" << std::endl;

    return 0;
}
```

### `05_generalized_init_capture.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's own Section F.5 covers C++14's generalized
// (init) capture, `[name = expr]` -- constructing brand-new,
// closure-PRIVATE state from an arbitrary expression, with no
// external variable required at all. Its own worked example is a
// momentum-style optimizer smoother, built around a std::vector<double>;
// this book's own version reproduces the identical idea around a real
// torch::Tensor instead, tying it directly to the momentum-based
// optimizers this book's own Chapter 20/21 material assumes readers
// already understand conceptually.
auto make_momentum_smoother(double beta) {
    // [beta, velocity = torch::Tensor()]: beta is an ordinary
    // by-value capture; velocity is a GENERALIZED capture -- a
    // brand-new torch::Tensor, owned entirely by this closure, that
    // exists nowhere outside it. `mutable` is required because the
    // lambda body reassigns its own captured `velocity` on every call.
    return [beta, velocity = torch::Tensor()](const torch::Tensor& gradient) mutable {
        if (!velocity.defined()) {
            velocity = torch::zeros_like(gradient);
        }
        velocity = beta * velocity + gradient;
        return velocity;
    };
}

int main() {
    double beta = 0.9;
    torch::Tensor gradient = torch::tensor({1.0, 2.0, 3.0});

    auto smoother = make_momentum_smoother(beta);

    torch::Tensor v1 = smoother(gradient);
    torch::Tensor v2 = smoother(gradient);
    torch::Tensor v3 = smoother(gradient);

    std::cout << "make_momentum_smoother(0.9), the same gradient tensor {1,2,3} applied 3 times in a row:"
              << std::endl;
    std::cout << "  step 1: " << v1.flatten() << std::endl;
    std::cout << "  step 2: " << v2.flatten() << std::endl;
    std::cout << "  step 3: " << v3.flatten() << std::endl;

    // Independent hand-unrolled check for element 0: v0=0,
    // v1=0.9*0+1=1.0, v2=0.9*1.0+1=1.9, v3=0.9*1.9+1=2.71.
    double v0 = 0.0;
    double hand_v1 = beta * v0 + 1.0;
    double hand_v2 = beta * hand_v1 + 1.0;
    double hand_v3 = beta * hand_v2 + 1.0;
    // The tensors above are ordinary torch::Tensor default dtype
    // (float32), while the hand-unrolled check above is double
    // arithmetic -- so an exact bitwise match isn't expected. A
    // tolerance of 1e-6 (loose enough to absorb float32's own real
    // rounding at each of the three chained multiply-adds, tight
    // enough to catch any genuine logic error) is used instead.
    bool matches = std::abs(v1[0].item<double>() - hand_v1) < 1e-6
                   && std::abs(v2[0].item<double>() - hand_v2) < 1e-6
                   && std::abs(v3[0].item<double>() - hand_v3) < 1e-6;
    std::cout << "\nindependent hand-unrolled check for element 0: v1=" << hand_v1 << ", v2=" << hand_v2
              << ", v3=" << hand_v3 << " -- matches the real tensor's own element 0 at each step (within "
              << "1e-6, since the tensor is float32 and this hand check is double)? " << matches << std::endl;

    // A second, independently created smoother starts fresh -- proof
    // that `velocity` is genuinely PER-CLOSURE state, not some shared
    // global the first smoother happened to also touch.
    auto smoother2 = make_momentum_smoother(beta);
    torch::Tensor v1_again = smoother2(gradient);
    std::cout << "\na second, independent make_momentum_smoother(0.9) instance, same gradient, step 1: "
              << v1_again.flatten() << " -- matches the FIRST smoother's own step 1 exactly, confirming "
              << "each closure owns its own private velocity, entirely isolated from the other's" << std::endl;

    return 0;
}
```

### `06_lambdas_as_comparators_and_predicates.cpp`

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section F.6 uses std::sort, std::count_if,
// and std::find_if -- ordinary standard-library algorithms that take
// a callable (a comparator or a predicate) as one of their own
// parameters, the exact "generic function parameterized by a
// callable" shape this book's own Appendix D.2/D.3 already used for
// torch::Tensor-flavored data. This section reproduces the identical
// three algorithms, over a small vector of z-spreads (in basis
// points) reminiscent of Chapter 22.1's own bond-portfolio data,
// rather than the CUDA book's own plain integers.
int main() {
    std::vector<double> spreads_bp = {52.0, 21.0, 88.0, 13.0, 95.0, 34.0};

    std::cout << "spreads (bp): ";
    for (double s : spreads_bp) std::cout << s << " ";
    std::cout << std::endl;

    // An inline lambda comparator, ascending.
    std::vector<double> ascending = spreads_bp;
    std::sort(ascending.begin(), ascending.end(), [](double a, double b) { return a < b; });
    std::cout << "\nascending  (inline lambda a < b): ";
    for (double s : ascending) std::cout << s << " ";
    std::cout << std::endl;

    // A comparator stored in a NAMED variable first -- std::sort
    // doesn't care whether the comparator is written inline or held
    // in a variable first, the same "a callable is a callable"
    // structural point Appendix D.2 already made about lambdas and
    // functors both satisfying the identical operator() interface.
    auto compare_descending = [](double a, double b) { return a > b; };
    std::vector<double> descending = spreads_bp;
    std::sort(descending.begin(), descending.end(), compare_descending);
    std::cout << "descending (named lambda a > b) : ";
    for (double s : descending) std::cout << s << " ";
    std::cout << std::endl;

    // std::count_if / std::find_if with a hard-coded-threshold predicate.
    long count_over_40 = std::count_if(spreads_bp.begin(), spreads_bp.end(), [](double s) { return s > 40.0; });
    auto first_even_ish = std::find_if(spreads_bp.begin(), spreads_bp.end(),
                                        [](double s) { return static_cast<long>(s) % 2 == 0; });
    std::cout << "\ncount_if(s > 40.0)                  = " << count_over_40 << std::endl;
    std::cout << "find_if(static_cast<long>(s) % 2 == 0) found = " << *first_even_ish
              << " (the first spread whose truncated integer part is even)" << std::endl;

    // A capturing predicate, parameterized by an ordinary variable
    // instead of a hard-coded literal -- avoiding hard-coding the
    // threshold is the entire reason a capture list exists at all.
    double threshold = 40.0;
    long count_over_threshold = std::count_if(spreads_bp.begin(), spreads_bp.end(),
                                               [threshold](double s) { return s > threshold; });
    std::cout << "\na capturing predicate, [threshold](double s){ return s > threshold; }, threshold=40.0: "
              << "count_over_threshold = " << count_over_threshold
              << " -- matches the hard-coded count_if(s > 40.0) above exactly (" << count_over_40
              << "), confirming the capture-based version is equivalent, not merely similar" << std::endl;

    return 0;
}
```

### `07_std_function_and_type_erasure.cpp`

```cpp
#include <functional>
#include <iostream>
#include <vector>

// Section F.2 of this appendix already showed the problem this
// section solves: two lambdas with identical signatures are still
// two DIFFERENT, incompatible compiler-synthesized types -- so a
// plain std::vector<auto> holding several different lambdas isn't
// even expressible, let alone useful. std::function<Signature> is
// the standard library's own answer: TYPE ERASURE, wrapping ANY
// callable matching a given signature (a lambda, a functor, a
// function pointer) behind one uniform, storable, copyable type.
//
// This book's own Appendix D.4 already demonstrated std::function
// working normally in ordinary host code at length (a std::function
// wrapping a real torch::matmul call, timed via a host helper
// modeled on Chapter 19's Benchmark::time_function), and explicitly
// contrasted that with the CUDA C++ edition's own Section D.7 finding
// that std::function's own vtable-driven dispatch cannot cross onto a
// CUDA device as a kernel PARAMETER. This section doesn't repeat that
// demonstration -- it states the general-purpose C++ feature plainly,
// with one fresh, short example, and connects it back to exactly the
// problem Section F.2 raised.
int main() {
    int offset = 100;

    // Three lambdas, three DIFFERENT compiler-synthesized types
    // (Section F.2's own finding) -- but std::function<int(int)>
    // erases all three down to one uniform, storable type, so a
    // single std::vector CAN hold all three side by side.
    std::vector<std::function<int(int)>> operations;
    operations.push_back([](int x) { return x * 2; });
    operations.push_back([offset](int x) { return x + offset; });
    operations.push_back([](int x) { return x * x; });

    std::cout << "three lambdas of three genuinely different compiler-synthesized types (Section F.2's "
              << "own finding), all stored in one std::vector<std::function<int(int)>>, each applied to "
              << "x = 7:" << std::endl;
    for (size_t i = 0; i < operations.size(); i++) {
        std::cout << "  operations[" << i << "](7) = " << operations[i](7) << std::endl;
    }

    std::cout << "\nthis compiles and runs with plain g++, no CUDA toolchain, no --extended-lambda flag "
              << "anywhere -- Appendix D.4's own point restated in one sentence: Section D.7's own CUDA "
              << "restriction is specific to a std::function used as a __global__ kernel PARAMETER; "
              << "ordinary host code calling std::function::operator(), exactly as this file does, is the "
              << "case std::function's own type-erasure mechanism was built for in the first place."
              << std::endl;

    return 0;
}
```

## Chapter Summary

This appendix laid out the plain C++ lambda feature this book has relied on, unnamed, since its own first chapter -- the foundation this book's own Appendix D already assumed a reader had, the way the CUDA C++ edition's own Appendix D does too. A lambda's four fixed parts (capture, parameters, optional return type, body) were named explicitly for the first time. Section F.2 confirmed, via real `sizeof` and `typeid` checks, that a lambda is a genuine compiler-synthesized object with a real, distinct type, even when two lambdas look textually identical -- the problem Section F.7 closes the loop on via `std::function`'s type erasure. Section F.3 walked the full capture-clause vocabulary, including a real `torch::Tensor`-holding class demonstrating `[this]` versus C++17's `[*this]`. Section F.4 confirmed a non-`mutable` lambda's implicit `const` via a real, genuinely captured compiler error, not an assumed one. Section F.5 applied C++14's generalized capture to a real momentum-style tensor smoother, tying directly into this book's own neural-network optimizer material. And Section F.6 showed lambdas serving standard-library algorithms as comparators and predicates, on data in this book's own financial-computing spirit.

## Self-Check Questions

1. Section F.1 uses an explicit `-> std::string` trailing return type on `classify_mean`. Explain precisely what would happen to the return type if that trailing return type were removed and plain `auto` deduction were used instead, given that both of the lambda's own branches return string literals.
2. Section F.2 reports that `square_v1` and `square_v2` -- two lambdas with textually identical source -- have different `typeid` results. Explain why this is true even though the two lambdas would produce identical results for any given input.
3. Section F.3's `GradientAccumulator` example reports that `acc2.total` (the original object) stays exactly `torch::zeros({3})` after `add_via_copy()` is called through a `[*this]`-capturing lambda. Explain precisely what `[*this]` copied at the moment the lambda was created, and why the mutation inside `add_via_copy` never reaches the original object.
4. Section F.4 compiles a broken snippet SEPARATELY from the file's own main, genuinely-run source. Explain why embedding the broken snippet directly inside `04_mutable_and_const_by_default.cpp`'s own `main()` would have broken this appendix's own established verification convention.
5. Section F.7 states that it is "closing the loop on Section F.2" rather than simply repeating Appendix D.4's own `std::function` demonstration. Explain specifically what problem, first raised in Section F.2, `std::function`'s type erasure actually solves.

## Where We Go Next

This appendix is the host-only foundation this book's own Appendix D already assumed a reader had -- nothing in this appendix depends on Appendix D, and nothing in Appendix D needs re-reading in light of it, the identical relationship the CUDA C++ edition's own Appendix F states about its own Appendix D. The appendices that follow return to LibTorch's own domain: the Rule of Five and a risk engine to exercise it, and tensor contractions on CPU and GPU.

## Worked Solutions

**1.** With plain `auto` deduction instead of an explicit trailing return type, `classify_mean`'s own return type would deduce as `const char*` -- both branches (`"positive"` and `"non-positive"`) are string literals, and `const char*` is precisely what a string literal's own type is before any conversion. The function would still compile and still return the correct text, but callers would receive a raw C-style string pointer rather than a real `std::string` object, losing `std::string`'s own value-semantics conveniences (concatenation via `+`, `.length()`, comparison via `==` against another `std::string`, and so on) unless they separately converted it themselves.

**2.** `typeid` reports different results because C++ lambda expressions synthesize a NEW, anonymous class at the exact point in source where the lambda expression appears -- `square_v1`'s own lambda expression and `square_v2`'s own lambda expression are two SEPARATE pieces of source text, even though they read identically, so the compiler generates two separate anonymous classes, one for each. The two lambdas' own BEHAVIOR is identical (both compute `x*x`), but behavior and type are different things in C++: two classes can implement the identical logic and still be two, entirely distinct types, exactly the way two hand-written structs named `Square1` and `Square2`, each with an identical `operator()`, would also be two distinct types despite computing the same thing.

**3.** `[*this]` copies the ENTIRE enclosing `GradientAccumulator` object, including its own `torch::Tensor total` member, at the exact moment the lambda expression is evaluated (inside `make_adder_by_copy()`, when `acc2.make_adder_by_copy()` is called). From that moment on, the returned lambda holds its own, entirely separate `GradientAccumulator` copy -- when `add_via_copy` later executes `total = total + grad`, it reassigns the COPY's own `total` member to a freshly computed tensor, never touching `acc2`'s own `total` at all, because `acc2` and the lambda's own private copy have been two separate objects, with two separate `torch::Tensor` handles, since the instant `[*this]` captured them.

**4.** This appendix's own established convention -- shared with every other genuinely-run source file in this book -- is that the file listed and locked as a chapter's own "Worked Example" or "Complete Runnable Code" source is a file that GENUINELY, VERIFIABLY compiles and runs, confirmed by this appendix's own verification pipeline actually recompiling and rerunning it. A file whose own `main()` contained code that fails to compile would fail that exact check -- there would be no successful binary to run, and no genuine output to lock and compare against a fresh rerun -- so the broken snippet had to be compiled as a SEPARATE, standalone test, specifically to capture its own real error text as a genuine artifact in its own right, while the file that is actually locked into this appendix's own Worked Example and Complete Runnable Code sections remains one that compiles and runs cleanly, exactly like every other file in this book.

**5.** Section F.2 demonstrated that two lambdas -- even two with textually identical source -- are two different, incompatible C++ types, which means an ordinary container like `std::vector<SomeLambdaType>` can hold only ONE specific lambda's own type, never a mix of different lambdas, even ones sharing an identical call signature like `int(int)`. `std::function<int(int)>` solves exactly that problem: it type-erases any callable matching that signature -- regardless of which of the many different, incompatible closure types actually implements it -- behind one single, uniform, storable type, which is why `std::vector<std::function<int(int)>>` can hold three genuinely different lambda types side by side in Section F.7's own worked example, where a `std::vector` of the lambdas' own native types could not.
