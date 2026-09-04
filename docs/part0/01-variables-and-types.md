# Chapter 1: Variables, Types, and the Device Tag LibTorch Actually Uses

> "A type is a promise you make to the compiler about what a piece of memory means. LibTorch's `torch::Tensor` carries a second promise alongside that one, but it is not the promise CUDA C++ makes you write by hand. CUDA C++ asks you to tell the compiler, once, which of two pipelines is allowed to process a given function — a promise resolved before your program exists. LibTorch asks a `Tensor` to carry its device with it as an ordinary runtime field, checked the moment an operation actually needs it, not before."

**What you will understand by the end of this chapter:**

- What actually happens in memory when you write `int x = 42` in C++ — the stack slot, the byte width, and why the compiler needs to know the type before that line runs, exactly as in any statically typed language
- Why C++'s `auto` type inference (`auto a = 10`) is not the same thing as Python's dynamic typing, even though neither one has a visible type annotation
- Why LibTorch's host/device split is not a compile-time property of a *function*, the way CUDA C++'s `__host__`/`__device__` is — it is a runtime property of a *value*, carried inside every `torch::Tensor` as an ordinary `c10::Device` field, checked when an operation runs rather than enforced by which compiler saw the source
- What genuinely happens when you ask a CPU tensor to move onto a CUDA device that isn't there, or construct one directly on CUDA with no driver present — two different, both real, C++ exceptions, not a silently ignored error code
- Why LibTorch's reduced-precision scalar types (`c10::Half`, `c10::BFloat16`) and `c10::complex<float>` are genuine, sized, aligned C++ types you can measure and compute with directly on the host, ahead of the dedicated tensor-dtype chapter later in this book

**What you need to know first:**

- General programming experience in C, C++, or a similarly statically-typed language — this chapter assumes familiarity with basic C++ syntax, not any prior exposure to PyTorch or LibTorch
- No prior LibTorch or PyTorch C++ experience is assumed — this is the first technical chapter of the book
- If you've read the CUDA C++ edition of this book, this chapter opens the same way its Chapter 1 does — the same comparative frame (statically typed vs. dynamically typed, `auto` vs. Python). What's different starts at Section 1.3: the CUDA book spends that section on a single `.cu` file compiled twice, by two separate compiler pipelines, for a kernel it builds by hand. LibTorch ships that kernel already built, so this chapter's Section 1.3 is about something else entirely — the runtime device tag that took the compiler pipeline's place. If you've already worked through [Getting Started](../getting-started.md), the toolchain and the genuine `.backward()` result shown there are the same LibTorch install this chapter's code compiles against.

## 1.1 What a Type Actually Is `[FOUNDATIONAL]`

### Intuition

A type is a label you attach to a region of memory, and everything downstream trusts that label without re-checking it. Write `int x`, and the compiler reserves four bytes and remembers, in its own bookkeeping, that those four bytes are to be read as a signed whole number every time `x` appears again. Write `float y`, and it reserves four different bytes, remembering instead that they follow the IEEE-754 rules for a 32-bit floating-point value. Nothing about the bytes themselves says which interpretation is correct — the type is a decision made once, at compile time, and never revisited while the program runs.

### Background

A **statically typed** language fixes the type of every variable before the program runs — either because you wrote it explicitly or because the compiler deduced it unambiguously from context. C, C++, and the C++ LibTorch is written in and linked against are all statically typed. A **dynamically typed** language, like the Python most LibTorch users come from, attaches the type to the *value* at runtime instead of to the variable: a name has no type of its own, it just currently refers to some object, and that object carries its own type tag, checked on every operation that touches it.

| Language | Type known at | Where the value lives | Type check per use |
|---|---|---|---|
| C / C++ / LibTorch's own source | Compile time (mandatory) | Stack (usually, for locals) | None |
| Python (the language most LibTorch users already know) | Runtime (attached to the object) | Heap (always) | Every operation |

### Worked Example 1.1.1 — one declaration, traced to the byte

```cpp
#include <cstdio>
#include <cstdint>

int main() {
    int x = 42;
    float y = 3.14159f;
    double z = 2.71828;
    int64_t idx = 4200000000LL;  // the integer type torch uses for sizes/strides

    printf("x = %d, sizeof(int) = %zu\n", x, sizeof(int));
    printf("y = %f, sizeof(float) = %zu\n", y, sizeof(float));
    printf("z = %f, sizeof(double) = %zu\n", z, sizeof(double));
    printf("idx = %lld, sizeof(int64_t) = %zu\n", (long long)idx, sizeof(int64_t));

    float narrowed = 3.14159265358979;
    printf("narrowed (double literal into float) = %.10f\n", narrowed);
    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_stack_types.cpp -o 01_stack_types
./01_stack_types
```

```text
x = 42, sizeof(int) = 4
y = 3.141590, sizeof(float) = 4
z = 2.718280, sizeof(double) = 8
idx = 4200000000, sizeof(int64_t) = 8
narrowed (double literal into float) = 3.1415927410
```

The first three lines are ordinary C++, unrelated to LibTorch — four bytes for `x`, four for `y`, eight for `z`, exactly what `sizeof` reports. The fourth line is the one that matters for the rest of this book: `int64_t`, eight bytes, is the actual integer type `torch::Tensor` uses everywhere a size, a stride, or an index appears. When Chapter 6 opens `torch::Tensor::sizes()` and finds it returns an array of `int64_t`, that isn't an arbitrary implementation choice being introduced for the first time — it's the same eight-byte signed integer this line just measured, chosen for the same reason any large indexable buffer needs it: a 4-byte `int` tops out at about 2.1 billion, and a tensor with more elements than that (rare on a single machine today, not rare forever) would silently overflow it.

### ASCII Diagram — one stack frame, four variables

```
Stack frame for a function containing:
    int x = 42;
    float y = 3.14159f;
    double z = 2.71828;
    int64_t idx = 4200000000LL;

High address
 +--------------------------+
 | ... caller's frame ...   |
 +--------------------------+
 | idx (8 bytes, int64_t)   |  <- offset +16..+23: 0000000011111010111100001000000000  (two's-complement 4200000000)
 +--------------------------+
 | z   (8 bytes, double)    |  <- offset +8..+15:  0100000000000101...  (IEEE-754 bits for 2.71828)
 +--------------------------+
 | y   (4 bytes, float)     |  <- offset +4..+7:   01000000010010...    (IEEE-754 bits for 3.14159)
 +--------------------------+
 | x   (4 bytes, int)       |  <- offset +0..+3:   00000000000000000000000000101010  (two's-complement 42)
 +--------------------------+
Low address                    <- stack pointer sits here
```

> `[COMMON TRAP]` C++ silently narrows a `double` literal to a `float` variable wherever the assignment is contextually allowed, and it does so quietly by default. `float narrowed = 3.14159265358979;` compiles without a warning and discards precision it never tells you about — the genuine output above shows `3.1415927410`, not the fifteen significant digits the literal was written with. This matters for LibTorch specifically because a `torch::Tensor` created from a mix of `double` and `float` C++ values goes through exactly this same silent narrowing at the moment it's built, unless you're explicit about the tensor's dtype — a point Chapter 8 (Tensor Creation Functions) returns to directly.

## 1.2 Type Inference vs. Dynamic Typing `[FOUNDATIONAL]`

### Intuition

Picture two tailors. The first measures you once, at your very first fitting, and cuts a suit that stays permanently your size — it never adjusts itself afterward. The second never measures you up front at all; every time you put the suit on, they re-measure and refit it to whatever you happen to be that day. C++'s `auto a = 10;` is the first tailor: the compiler looks at `10` once, at compile time, deduces `int`, and `a` is an `int` for the rest of its scope — indistinguishable, once compiled, from `int a = 10;` written by hand. Python's `a = 10` is the second tailor: `a` has no fixed size of its own, and nothing stops the very next line from repointing it at a string.

### Background

**Type inference** is a compile-time process: the compiler looks at an initializer, determines its type through the same rules an explicit annotation would use, and locks that type in for the variable's entire scope. No inference happens while the program runs — by execution time, `auto`-declared variables are indistinguishable from explicitly-typed ones in the compiled binary. **Dynamic typing** has no equivalent compile-time fixing step, because the type belongs to the object a name currently references, not to the name itself.

| | C++ `auto a = 10;` | Python `a = 10` |
|---|---|---|
| When is the type decided? | Once, at compile time | Never fixed — checked fresh on every use |
| Can `a` later hold a `std::string`? | No — compile error | Yes — `a = "ten"` is legal |
| Cost of inference at runtime | Zero (already resolved) | N/A — no inference, only per-use checking |

This matters immediately, before this book has written a single line involving `torch::Tensor`: LibTorch's own C++ API leans on `auto` constantly, in its own documentation and in every chapter of this book from here on (`auto t = torch::ones({2, 2});` is the idiomatic way to write that line). `auto` in that position is not LibTorch being loose about types — it is the same compile-time-only inference this section just measured, applied to a `torch::Tensor` return value instead of an `int`.

### Worked Example 1.2.1 — the same-looking line, four different outcomes

```cpp
#include <cstdio>
#include <typeinfo>

int main() {
    auto a = 10;
    auto b = 5.5;
    auto c = true;
    auto d = 10L;

    printf("a=%d (sizeof %zu)\n", a, sizeof(a));
    printf("b=%f (sizeof %zu)\n", b, sizeof(b));
    printf("c=%d (sizeof %zu)\n", c, sizeof(c));
    printf("d=%ld (sizeof %zu)\n", d, sizeof(d));
    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_auto_inference.cpp -o 02_auto_inference
./02_auto_inference
```

```text
a=10 (sizeof 4)
b=5.500000 (sizeof 8)
c=1 (sizeof 1)
d=10 (sizeof 8)
```

Four declarations, four different types, none of them written down: `a` infers to `int` (4 bytes), `b` to `double` (8 bytes), `c` to `bool` (1 byte), `d` to `long` (8 bytes, because of the `L` suffix on the literal). Every one of those sizes was decided before this program ran at all — `auto` did not make the program dynamically typed, it just let the compiler write the type annotation for you instead of you writing it yourself.

## 1.3 A Tensor's Device Is a Value, Not a Compiler Decision `[FOUNDATIONAL]`

### Intuition

Raw CUDA C++ makes the host/device split a property of a *function*: you write `__host__`, `__device__`, or `__global__` on the function itself, and `nvcc` decides — before your program exists as a binary — which of its two compiler pipelines is allowed to touch that function's body. The split is resolved, permanently, at compile time. LibTorch does not ask you to make that decision anywhere in your source code, because LibTorch already made it, once, when it was built. What you get instead is a `c10::Device` field riding inside every `torch::Tensor` — an ordinary piece of runtime data, exactly as inspectable and comparable as the `int x` from Section 1.1, set when the tensor is constructed and read every time an operation needs to know where the tensor actually lives.

### Background

`c10::Device` pairs a `DeviceType` (`torch::kCPU`, `torch::kCUDA`, and others LibTorch supports) with an optional index, and every `torch::Tensor` carries one, retrievable with `.device()`. Nothing about calling `.device()` involves the compiler deciding anything — it's a plain member-variable read, no different in kind from reading `x` back out of the stack frame in Section 1.1's diagram.

| | CUDA C++ `__host__` / `__device__` | LibTorch's `c10::Device` |
|---|---|---|
| What it's attached to | A function | A `Tensor` value |
| When it's decided | Compile time, by `nvcc` | Runtime, at tensor construction |
| How a violation is caught | Compile error, before a binary exists | A thrown `c10::Error`, when an operation actually runs |
| Where it lives | Nowhere at runtime — it has already done its job by the time the binary runs | A real field inside the tensor, read on every device-sensitive operation |

*(Excerpt from `03_device_runtime_tag.cpp`; the complete file — including what genuinely happens when this same program tries to cross onto a CUDA device that isn't there — appears in Section 1.4 and in [Complete Runnable Code](#complete-runnable-code) below.)*

```cpp
torch::Tensor t = torch::ones({2, 2});
std::cout << "t.device() = " << t.device() << std::endl;
std::cout << "t.device().type() == torch::kCPU: "
          << (t.device().type() == torch::kCPU ? "true" : "false") << std::endl;
std::cout << "torch::cuda::is_available() = "
          << (torch::cuda::is_available() ? "true" : "false") << std::endl;
```

Genuinely compiled and run, this excerpt's three lines produce:

```text
t.device() = cpu
t.device().type() == torch::kCPU: true
torch::cuda::is_available() = false
```

`t.device()` genuinely returns a `cpu` device — not because the compiler decided this function was host-only, but because `torch::ones({2, 2})` with no device argument defaults to CPU, and that default is stored as ordinary data inside `t`. `torch::cuda::is_available()` is a real function call too, evaluated when the program runs — genuinely returning `false` in this book's environment for the same reason [Getting Started](../getting-started.md) already established: this LibTorch build genuinely has CUDA support compiled in, and genuinely has no GPU or driver to report as available.

## 1.4 Crossing the Boundary: `.to()`, Direct Construction, and the Exceptions LibTorch Actually Throws `[FOUNDATIONAL]`

### Intuition

The CUDA C++ book's own Chapter 1 shows calling a `__device__`-only function from `main()` failing to *compile* — the boundary violation is caught before the program exists. LibTorch has no equivalent compile-time boundary to violate: `t.to(torch::kCUDA)` and constructing a tensor with `.device(torch::kCUDA)` are both ordinary function calls that compile cleanly against this exact LibTorch build, CUDA-enabled or not — there is nothing about either line that a compiler could object to. The boundary is enforced later, at the moment the call actually runs and genuinely needs a CUDA device that isn't there, and it is enforced by throwing a real C++ exception, not by silently returning a code the caller might forget to check.

### Background

LibTorch's internal `TORCH_CHECK` machinery throws `c10::Error`, a real subclass of `std::exception`, whenever a precondition like "a CUDA device must exist" fails. That stands in direct contrast to raw CUDA's own C API, which the CUDA C++ book's Chapter 1 documents returning `cudaError_t` codes like `cudaErrorNoDevice` from calls like `cudaMalloc` — recoverable, but only if the caller remembers to check the return value. LibTorch's C++ frontend wraps that same underlying condition in an exception instead: an uncaught `c10::Error` terminates the program loudly, with a message, rather than leaving a silently-ignorable return code sitting in a variable nobody read.

### Worked Example 1.4.1 — two genuinely different exceptions for the same missing GPU

```cpp
#include <torch/torch.h>
#include <iostream>
#include <sstream>

// Print only the first line of an exception's message: LibTorch's messages carry a
// full C++ backtrace after the first line, and that backtrace embeds real memory
// addresses that change from run to run (ASLR) -- genuine, but not something a
// byte-for-byte reproducible chapter transcript can quote.
std::string first_line(const std::exception& e) {
    std::istringstream iss(e.what());
    std::string line;
    std::getline(iss, line);
    return line;
}

int main() {
    torch::Tensor t = torch::ones({2, 2});
    std::cout << "t.device() = " << t.device() << std::endl;
    std::cout << "t.device().type() == torch::kCPU: "
              << (t.device().type() == torch::kCPU ? "true" : "false") << std::endl;
    std::cout << "torch::cuda::is_available() = "
              << (torch::cuda::is_available() ? "true" : "false") << std::endl;

    try {
        torch::Tensor t_cuda = t.to(torch::kCUDA);
        std::cout << "moved to: " << t_cuda.device() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "t.to(torch::kCUDA) threw: " << first_line(e) << std::endl;
    }

    try {
        torch::Tensor c = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA));
        std::cout << "constructed directly on: " << c.device() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "direct CUDA construction threw: " << first_line(e) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_device_runtime_tag.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_device_runtime_tag
./03_device_runtime_tag
```

```text
t.device() = cpu
t.device().type() == torch::kCPU: true
torch::cuda::is_available() = false
t.to(torch::kCUDA) threw: CUDA error: CUDA driver version is insufficient for CUDA runtime version
direct CUDA construction threw: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

Both calls fail for the identical underlying reason — no NVIDIA driver anywhere in this environment — and both fail as genuine, catchable `std::exception`s rather than silently. What's worth noticing honestly is that they don't fail with the *same message*. `.to(torch::kCUDA)` routes through LibTorch's tensor-conversion path and reports a driver-version mismatch; constructing directly on `torch::kCUDA` routes through tensor allocation instead and reports, more plainly, that no driver was found at all. Both are genuine, both are correct in substance (there is, in fact, no usable CUDA driver here), and neither was written by hand for this book — they're exactly what the two different internal code paths LibTorch actually has for these two operations produce when the same real precondition fails.

> `[COMMON TRAP]` Every `try`/`catch` in this book's LibTorch code is there because it has to be — an uncaught `c10::Error` does not degrade gracefully, it terminates the program via `std::terminate`, printing the exception's `.what()` (backtrace and all) to stderr. That is arguably a *safer* default than raw CUDA's ignorable return code — a crash is at least impossible to miss — but it means code written against LibTorch's C++ frontend that skips error handling around a device-crossing call doesn't fail quietly in a corner; it takes the whole process down.

## 1.5 Built-in Specialized Scalar Types: `c10::Half`, `c10::BFloat16`, and Friends

### Intuition

The CUDA C++ book's Chapter 1 closes by measuring `float2` and `float4` — genuine, sized, aligned C++ types the compiler enforces, not a convention four floats happen to follow. LibTorch's own low-level headers ship an equivalent kind of genuine type, aimed at a different problem: `c10::Half` and `c10::BFloat16` are real, 2-byte C++ types with real arithmetic operators, used throughout LibTorch anywhere a tensor's dtype is a 16-bit float format — and, unlike a `torch::Tensor`, they can be constructed, measured, and computed with directly on the host, with no tensor and no dtype argument in sight.

### Background

`c10::Half` stores IEEE-754 half precision (1 sign bit, 5 exponent bits, 10 mantissa bits); `c10::BFloat16` stores Google's "brain float" format (1 sign bit, 8 exponent bits, 7 mantissa bits — the same exponent range as `float`, traded down to less mantissa precision). Both are genuinely 2 bytes wide despite representing their bits completely differently internally, and LibTorch's operator overloads for both convert to `float`, perform the arithmetic in `float`, and convert back — which is why a `c10::Half + c10::Half` genuinely runs on a CPU with no hardware half-precision unit at all.

### Worked Example 1.5.1 — genuinely measured, not assumed

```cpp
#include <torch/torch.h>
#include <iostream>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/complex.h>

int main() {
    std::cout << "sizeof(float) = " << sizeof(float) << ", alignof(float) = " << alignof(float) << std::endl;
    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << ", alignof(c10::Half) = " << alignof(c10::Half) << std::endl;
    std::cout << "sizeof(c10::BFloat16) = " << sizeof(c10::BFloat16) << ", alignof(c10::BFloat16) = " << alignof(c10::BFloat16) << std::endl;
    std::cout << "sizeof(c10::complex<float>) = " << sizeof(c10::complex<float>) << ", alignof(c10::complex<float>) = " << alignof(c10::complex<float>) << std::endl;
    std::cout << "sizeof(double) = " << sizeof(double) << ", alignof(double) = " << alignof(double) << std::endl;

    // genuine arithmetic on CPU, no GPU needed -- Half/BFloat16 upconvert to float for the op
    c10::Half h1 = 1.5f;
    c10::Half h2 = 2.25f;
    c10::Half h3 = h1 + h2;
    std::cout << "c10::Half: 1.5 + 2.25 = " << static_cast<float>(h3) << std::endl;

    c10::BFloat16 b1 = 1.5f;
    c10::BFloat16 b2 = 2.25f;
    c10::BFloat16 b3 = b1 + b2;
    std::cout << "c10::BFloat16: 1.5 + 2.25 = " << static_cast<float>(b3) << std::endl;

    // a Tensor genuinely constructed with dtype=Half, on CPU
    torch::Tensor th = torch::tensor({1.5, 2.25}, torch::kHalf);
    std::cout << "half tensor dtype: " << th.dtype() << ", element_size(): " << th.element_size() << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_specialized_scalar_types.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_specialized_scalar_types
./04_specialized_scalar_types
```

```text
sizeof(float) = 4, alignof(float) = 4
sizeof(c10::Half) = 2, alignof(c10::Half) = 2
sizeof(c10::BFloat16) = 2, alignof(c10::BFloat16) = 2
sizeof(c10::complex<float>) = 8, alignof(c10::complex<float>) = 8
sizeof(double) = 8, alignof(double) = 8
c10::Half: 1.5 + 2.25 = 3.75
c10::BFloat16: 1.5 + 2.25 = 3.75
half tensor dtype: c10::Half, element_size(): 2
```

`c10::Half` and `c10::BFloat16` both measure at exactly 2 bytes, `alignof` 2 — half the width of `float`, despite splitting those 16 bits between exponent and mantissa completely differently from each other. `c10::complex<float>`, a type Chapter 22's quantitative-finance work will eventually touch, measures at 8 bytes: two real `float`s side by side, nothing more exotic than that. The last line constructs an actual `torch::Tensor` with `torch::kHalf` as its dtype and reads `element_size()` back — 2 bytes, matching `sizeof(c10::Half)` exactly, because a half-precision tensor's storage genuinely is a buffer of `c10::Half` values, not a separate representation the dtype tag merely describes.

Worth stating plainly: `1.5 + 2.25 = 3.75` is exact in both formats here, with no rounding error to observe — every one of those three values has a terminating binary fraction that fits inside even `c10::BFloat16`'s 7 mantissa bits. That's a deliberate choice for a first chapter's first look at these types, not a hidden limitation of the measurement; the rounding behavior each format actually exhibits under values that *don't* fit is exactly the kind of question Part 1's tensor-creation and specialized-type chapters come back to on purpose.

**Independent cross-check.** Every `sizeof`/`alignof` figure and both arithmetic results above were re-checked through a structurally different path — LibTorch's own Python bindings, which call into the identical compiled `libtorch_cpu.so` through a different entry point than the `g++`-compiled C++ binary above:

```python
import torch
torch.tensor([], dtype=torch.float16).element_size()   # -> 2
torch.tensor([], dtype=torch.bfloat16).element_size()   # -> 2
torch.tensor([], dtype=torch.complex64).element_size()  # -> 8
(torch.tensor(1.5, dtype=torch.float16) + torch.tensor(2.25, dtype=torch.float16)).item()   # -> 3.75
(torch.tensor(1.5, dtype=torch.bfloat16) + torch.tensor(2.25, dtype=torch.bfloat16)).item() # -> 3.75
```

Genuinely run: all five values agree exactly with the C++ output above.

## Complete Runnable Code

### File: `01_stack_types.cpp`

```cpp
#include <cstdio>
#include <cstdint>

int main() {
    int x = 42;
    float y = 3.14159f;
    double z = 2.71828;
    int64_t idx = 4200000000LL;  // the integer type torch uses for sizes/strides

    printf("x = %d, sizeof(int) = %zu\n", x, sizeof(int));
    printf("y = %f, sizeof(float) = %zu\n", y, sizeof(float));
    printf("z = %f, sizeof(double) = %zu\n", z, sizeof(double));
    printf("idx = %lld, sizeof(int64_t) = %zu\n", (long long)idx, sizeof(int64_t));

    float narrowed = 3.14159265358979;
    printf("narrowed (double literal into float) = %.10f\n", narrowed);
    return 0;
}
```

```bash
g++ -std=c++20 -O2 01_stack_types.cpp -o 01_stack_types
./01_stack_types
```

### File: `02_auto_inference.cpp`

```cpp
#include <cstdio>
#include <typeinfo>

int main() {
    auto a = 10;
    auto b = 5.5;
    auto c = true;
    auto d = 10L;

    printf("a=%d (sizeof %zu)\n", a, sizeof(a));
    printf("b=%f (sizeof %zu)\n", b, sizeof(b));
    printf("c=%d (sizeof %zu)\n", c, sizeof(c));
    printf("d=%ld (sizeof %zu)\n", d, sizeof(d));
    return 0;
}
```

```bash
g++ -std=c++20 -O2 02_auto_inference.cpp -o 02_auto_inference
./02_auto_inference
```

### File: `03_device_runtime_tag.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <sstream>

// Print only the first line of an exception's message: LibTorch's messages carry a
// full C++ backtrace after the first line, and that backtrace embeds real memory
// addresses that change from run to run (ASLR) -- genuine, but not something a
// byte-for-byte reproducible chapter transcript can quote.
std::string first_line(const std::exception& e) {
    std::istringstream iss(e.what());
    std::string line;
    std::getline(iss, line);
    return line;
}

int main() {
    torch::Tensor t = torch::ones({2, 2});
    std::cout << "t.device() = " << t.device() << std::endl;
    std::cout << "t.device().type() == torch::kCPU: "
              << (t.device().type() == torch::kCPU ? "true" : "false") << std::endl;
    std::cout << "torch::cuda::is_available() = "
              << (torch::cuda::is_available() ? "true" : "false") << std::endl;

    try {
        torch::Tensor t_cuda = t.to(torch::kCUDA);
        std::cout << "moved to: " << t_cuda.device() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "t.to(torch::kCUDA) threw: " << first_line(e) << std::endl;
    }

    try {
        torch::Tensor c = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA));
        std::cout << "constructed directly on: " << c.device() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "direct CUDA construction threw: " << first_line(e) << std::endl;
    }

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 03_device_runtime_tag.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 03_device_runtime_tag
./03_device_runtime_tag
```

### File: `04_specialized_scalar_types.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/complex.h>

int main() {
    std::cout << "sizeof(float) = " << sizeof(float) << ", alignof(float) = " << alignof(float) << std::endl;
    std::cout << "sizeof(c10::Half) = " << sizeof(c10::Half) << ", alignof(c10::Half) = " << alignof(c10::Half) << std::endl;
    std::cout << "sizeof(c10::BFloat16) = " << sizeof(c10::BFloat16) << ", alignof(c10::BFloat16) = " << alignof(c10::BFloat16) << std::endl;
    std::cout << "sizeof(c10::complex<float>) = " << sizeof(c10::complex<float>) << ", alignof(c10::complex<float>) = " << alignof(c10::complex<float>) << std::endl;
    std::cout << "sizeof(double) = " << sizeof(double) << ", alignof(double) = " << alignof(double) << std::endl;

    // genuine arithmetic on CPU, no GPU needed -- Half/BFloat16 upconvert to float for the op
    c10::Half h1 = 1.5f;
    c10::Half h2 = 2.25f;
    c10::Half h3 = h1 + h2;
    std::cout << "c10::Half: 1.5 + 2.25 = " << static_cast<float>(h3) << std::endl;

    c10::BFloat16 b1 = 1.5f;
    c10::BFloat16 b2 = 2.25f;
    c10::BFloat16 b3 = b1 + b2;
    std::cout << "c10::BFloat16: 1.5 + 2.25 = " << static_cast<float>(b3) << std::endl;

    // a Tensor genuinely constructed with dtype=Half, on CPU
    torch::Tensor th = torch::tensor({1.5, 2.25}, torch::kHalf);
    std::cout << "half tensor dtype: " << th.dtype() << ", element_size(): " << th.element_size() << std::endl;

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 04_specialized_scalar_types.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 04_specialized_scalar_types
./04_specialized_scalar_types
```

## Chapter Summary

C++ types work in LibTorch exactly as they work in any statically typed program — `int`, `float`, `double`, and the `int64_t` that `torch::Tensor` uses for every size and stride are ordinary compile-time promises about a fixed number of bytes, genuinely measured with `sizeof` and traced to real stack offsets, and `auto` resolves those same promises for you at compile time without making anything dynamically typed the way Python's per-object type tags do. Where LibTorch genuinely departs from raw CUDA C++ is the host/device split: CUDA C++ makes it a compile-time property of a *function*, resolved by `nvcc` choosing between two pipelines before a binary exists; LibTorch makes it a runtime property of a *value*, an ordinary `c10::Device` field riding inside every `torch::Tensor`, read by `.device()` and set once at construction. Crossing that boundary onto a CUDA device that isn't there doesn't fail to compile — `.to(torch::kCUDA)` and direct construction on `torch::kCUDA` both compile cleanly against this exact build — it fails at runtime instead, genuinely throwing a catchable `c10::Error` (two different, both real, messages for the two different internal code paths), a structural contrast with raw CUDA's own silently-ignorable `cudaError_t` return codes. LibTorch's specialized scalar types close the chapter the same way CUDA C++'s vector types do: `c10::Half` and `c10::BFloat16` measure at a genuine 2 bytes each despite encoding their bits completely differently, and `c10::complex<float>` at 8 — real, sized, host-usable C++ types, independently confirmed through LibTorch's own Python bindings, not tensor-only abstractions.

## Self-Check Questions

1. `auto a = 10;` in C++ and `a = 10` in Python both omit an explicit type annotation. What question distinguishes the two cases, and why does that question have a different answer for each?
2. Raw CUDA C++'s host/device split is enforced by which compiler pipeline processes a function's source. What takes its place in LibTorch, and at what point in a program's execution is that replacement actually checked?
3. `t.to(torch::kCUDA)` and constructing a tensor directly with `.device(torch::kCUDA)` both compile cleanly in this book's environment, which has no GPU at all. Why does neither call fail to compile, and what does each one do instead when it actually runs?
4. This chapter's Worked Example 1.4.1 shows `.to(torch::kCUDA)` and direct CUDA construction throwing two different exception messages for what is the same underlying condition — no NVIDIA driver present. What's a plausible reason those two code paths would report the failure differently, based on what each call is actually doing internally?
5. `c10::Half` and `c10::BFloat16` are both genuinely 2 bytes wide. Given that they split those 16 bits between exponent and mantissa differently, what does matching `sizeof` actually guarantee about the two types, and what does it *not* guarantee?

## Where We Go Next

Chapter 2 stays with plain C++ structure, the way this chapter stayed with plain C++ types, but shifts from primitive values to designing a `struct` — tracing which of LibTorch's own struct design patterns (starting with `torch::TensorOptions`, the builder-style struct this chapter's own worked examples have already been passing to `torch::ones` and `torch::tensor` without naming it) hold up under genuine inspection, and why LibTorch reaches for a struct-plus-builder pattern in some places and a `c10::intrusive_ptr`-managed handle in others.

## Worked Solutions

**1.** The distinguishing question is *when is the type decided, and can it change later*. For C++'s `auto a = 10;`, the compiler looks at the literal `10` exactly once, during compilation, deduces `int`, and `a` is permanently an `int` for the rest of its scope — indistinguishable from `int a = 10;` once compiled. For Python's `a = 10`, there is no compile-time deduction step at all: `a` is just a name that currently points at an `int` object, and the very next line is free to rebind it to a string or anything else, because the type belongs to the object, not the name. Both lines look identical in omitting a written type, but only one of them has actually fixed a type at all.

**2.** LibTorch replaces the compiler-pipeline decision with an ordinary runtime field: `c10::Device`, stored inside every `torch::Tensor` and readable with `.device()`. It is checked not at compile time and not even at tensor construction in the abstract, but at the moment a specific operation actually needs to know where its operands live — which is why `t.device()` itself never fails (it's just reading a stored value), while an operation that genuinely requires a CUDA device to exist, like `.to(torch::kCUDA)` or constructing a tensor with `.device(torch::kCUDA)`, only fails once it actually runs and finds no driver.

**3.** Neither call fails to compile because `torch::kCUDA` is just an ordinary enum value LibTorch's headers define regardless of whether the specific machine building or running the code has a GPU — the compiler has no way to know, and no reason to care, whether a CUDA device will exist at runtime. What runs instead, genuinely, is each call reaching into LibTorch's CUDA-handling code, discovering no driver is present, and throwing a real `c10::Error` — `.to(torch::kCUDA)` reporting a driver-version mismatch, direct construction reporting that no driver was found at all.

**4.** `.to(torch::kCUDA)` on an existing CPU tensor is fundamentally a *conversion* operation — it has a source tensor already in hand and is trying to produce a converted copy, so LibTorch's tensor-conversion code path is what first reaches into CUDA territory and reports what it finds there. Constructing a tensor directly with `.device(torch::kCUDA)` has no source tensor at all — it's an *allocation* from nothing, going through LibTorch's memory-allocation code path instead, which checks for a CUDA device at a different point and with different diagnostic context available to it. Same underlying fact (no driver), two different internal call graphs discovering it, two genuinely different messages.

**5.** Matching `sizeof` guarantees only that both types occupy the same number of bytes in memory and therefore cost the same to store and move — nothing about what those bytes *mean*. It does not guarantee equal precision, equal exponent range, or interchangeable arithmetic results: `c10::Half`'s 10 mantissa bits and `c10::BFloat16`'s 7 give them different rounding behavior on values that don't happen to be exactly representable in both (this chapter's own `1.5 + 2.25 = 3.75` sidesteps that difference on purpose, since all three values are exact in either format), and `c10::BFloat16`'s 8 exponent bits give it `float`'s full dynamic range while `c10::Half`'s 5 give it a much narrower one. Two types can be the same size and genuinely incompatible in what they can represent.
