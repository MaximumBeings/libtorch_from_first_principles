# Chapter 2: Struct Design Patterns, Read From LibTorch's Own Headers

> "Chapter 1 measured a handful of types nobody in this book had to design: `int`, `float`, `c10::Half`. A `struct` is where that changes — a promise about a memory layout that someone had to write, field by field, choosing exactly what to pack in and how tightly. LibTorch's own headers hand-write dozens of them, and this chapter reads four: a struct engineered down to eight bytes on purpose, a template that owns nothing but a pointer, an RAII wrapper genuinely borrowed from LibTorch's real allocator, and the one real, documented place in `torch::nn` where LibTorch reaches for the same compile-time trick CUDA C++'s hand-built classes use to dodge a vtable — used, honestly, for a different reason than dodging one."

**What you will understand by the end of this chapter:**

- Why `torch::TensorOptions` — the struct this book's own code has already been passing to `torch::ones` and `torch::tensor` since Chapter 1 — is engineered down to a measured 8 bytes on purpose, using C++ bitfields instead of the more obvious `std::optional` per field, and what LibTorch's own header comments say that choice costs
- How `TensorOptions`'s builder-style methods (`.dtype(...)`, `.device(...)`, `.requires_grad(...)`) resolve through overloaded and implicit conversion constructors entirely at compile time, and why chaining them never mutates the object you started from
- Why `torch::TensorAccessor<T, N>`, a real template LibTorch ships and this chapter genuinely instantiates twice, proves template specialization the same way Chapter 2 of the CUDA C++ edition does — two distinct compiled symbols, confirmed with `nm` — while measuring the same size regardless of `T`, for a reason a value-owning template like the CUDA book's own `Vector<T, N>` never runs into
- RAII, genuinely observed through LibTorch's own CPU allocator (`c10::GetCPUAllocator()`) and through a real `torch::Tensor`'s own deleter firing at scope exit — not a hand-rolled stand-in for either
- The one place in `torch::nn` where LibTorch's own source code documents using CRTP (`torch::nn::Cloneable<Derived>`) on purpose, genuinely exercised through a real `.clone()` call — and an honest correction to the textbook "CRTP avoids a vtable" story once you actually measure what it costs here

**What you need to know first:**

- Chapter 1 in full — this chapter leans on the runtime-vs-compile-time distinction from Sections 1.3 and 1.4 directly
- No prior LibTorch or PyTorch C++ experience assumed
- If you've read the CUDA C++ edition's Chapter 2: that chapter hand-builds a `Point2D`, a `DeviceBuffer`, and a `Shape`/`Circle` hierarchy from nothing, to teach struct layout, RAII, and CRTP in the abstract. This chapter has nothing to build from nothing — every struct it examines (`TensorOptions`, `TensorAccessor`, `Cloneable`) already exists in the LibTorch headers this book's code has been including since Chapter 1, so the worked examples here read and exercise real, already-shipped source rather than a teaching toy.

## 2.1 What a Struct Actually Is, Engineered Down to Eight Bytes `[FOUNDATIONAL]`

### Intuition

A struct is a blueprint for memory: every field gets a fixed offset, known at compile time, identical for every instance of that struct. Most structs don't think hard about that layout — they just declare the fields they need and let the compiler pad them out. `torch::TensorOptions` is a struct that *did* think hard about it. It needs to represent five independent, optional settings (dtype, device, layout, memory format, requires-grad, pinned-memory) for a tensor under construction, and the obvious design — five `std::optional<T>` fields — is exactly the design its authors rejected, in favor of packing every "is this set?" flag into single bits.

### Background

LibTorch's own `TensorOptions.h` lays the fields out like this (read directly from the installed header, not reconstructed from documentation):

```cpp
Device device_ = at::kCPU;                              // 16-bit
caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 16-bit
Layout layout_ = at::kStrided;                            // 8-bit
MemoryFormat memory_format_ = MemoryFormat::Contiguous;   // 8-bit

bool requires_grad_ : 1 = false;
bool pinned_memory_ : 1 = false;
bool has_device_ : 1 = false;
bool has_dtype_ : 1 = false;
bool has_layout_ : 1 = false;
bool has_requires_grad_ : 1 = false;
bool has_pinned_memory_ : 1 = false;
bool has_memory_format_ : 1 = false;
```

The header's own comment beside these fields explains the reason directly: `std::optional` was deliberately avoided *because* it would have prevented packing the seven `has_*_` presence flags down into a single byte. Every one of those `: 1` suffixes is a bitfield — one bit, not one byte — and the header backs the whole design with a compile-time check: `static_assert(sizeof(TensorOptions) <= sizeof(int64_t) * 2)`, a promise that this struct will never grow past 16 bytes no matter how it's laid out internally.

| | A naive "five settings" struct | `torch::TensorOptions` |
|---|---|---|
| How presence is tracked | `std::optional<T>` per field | A single packed byte of 1-bit flags |
| Compile-time size guarantee | None | `static_assert(sizeof(TensorOptions) <= 16)` |
| Genuinely measured size | Would depend on `optional`'s own overhead per field | 8 bytes |

### Worked Example 2.1.1 — measuring the promise, and exercising the builder

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "sizeof(torch::TensorOptions) = " << sizeof(torch::TensorOptions) << std::endl;

    torch::TensorOptions opts1;
    std::cout << "default opts1.has_dtype() = " << opts1.has_dtype() << std::endl;

    // builder-style chaining: each call returns a NEW TensorOptions by value
    torch::TensorOptions opts2 = opts1.dtype(torch::kFloat64).device(torch::kCPU).requires_grad(true);
    std::cout << "opts2.dtype() = " << opts2.dtype() << ", requires_grad = " << opts2.requires_grad() << std::endl;

    // opts1 itself: unchanged, since dtype()/device()/requires_grad() are all const methods
    std::cout << "opts1.has_dtype() after building opts2 = " << opts1.has_dtype() << std::endl;

    // implicit conversion constructors: TensorOptions(ScalarType) and TensorOptions(Device)
    torch::TensorOptions opts3 = torch::kFloat32;   // resolves via TensorOptions(ScalarType)
    torch::TensorOptions opts4 = torch::kCPU;        // resolves via TensorOptions(Device)
    std::cout << "opts3.dtype() = " << opts3.dtype() << ", opts3.has_device() = " << opts3.has_device() << std::endl;
    std::cout << "opts4.has_dtype() = " << opts4.has_dtype() << ", opts4.device() = " << opts4.device() << std::endl;

    // use it to build a real tensor
    torch::Tensor t = torch::ones({2, 2}, opts2);
    std::cout << "t.dtype() = " << t.dtype() << ", t.requires_grad() = " << t.requires_grad() << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
sizeof(torch::TensorOptions) = 8
default opts1.has_dtype() = 0
opts2.dtype() = double, requires_grad = 1
opts1.has_dtype() after building opts2 = 0
opts3.dtype() = float, opts3.has_device() = 0
opts4.has_dtype() = 0, opts4.device() = cpu
t.dtype() = double, t.requires_grad() = 1
```

`sizeof(torch::TensorOptions)` genuinely measures 8 bytes — half the 16-byte ceiling the header's own `static_assert` allows, and far smaller than five independent `std::optional<T>` fields would have cost. `opts2`'s construction is the builder pattern in action: `opts1.dtype(...)` doesn't modify `opts1`, it returns a brand-new `TensorOptions` with one more field set, and `.device(...)` and `.requires_grad(...)` each do the same to the result of the call before them — three independent, chained, value-returning calls, not three mutations of one object. The last line makes this concrete: `t.dtype()` and `t.requires_grad()` come out as `double` and `true` on the real tensor built from `opts2`, exactly the settings that chain accumulated.

> `[COMMON TRAP]` `has_dtype()` being `false` does not mean `.dtype()` is unsafe to call or will report an error — it genuinely returns `float`, the field's default value, silently. `opts3.has_device() = 0` above shows the same thing from the other side: `opts3` was built from a bare `ScalarType`, never touched `.device(...)` at all, and `has_device()` correctly reports that — but nothing stops code downstream from calling `opts3.device()` and getting back `cpu` with no indication it was never actually set. LibTorch's own construction functions (`torch::ones`, `torch::empty`, and the rest) are the code that actually checks `has_*_` internally to decide what to inherit from a tensor being constructed from another tensor; application code reading a `TensorOptions` back out generally shouldn't assume an unset field will look any different from an explicitly-set one holding the same default.

## 2.2 Multiple Constructors and Value Semantics `[FOUNDATIONAL]`

### Intuition

`torch::TensorOptions` supports more than one way to get one: build it from nothing, build it implicitly from a bare `torch::ScalarType`, or build it implicitly from a bare `torch::Device`. Which constructor actually runs is decided the same way any C++ overload resolution is decided — entirely at compile time, by matching the argument's type against the available constructor signatures — and the result of every builder-method call is a genuinely new, independent object, never a mutation of the one it was called on.

### Background

Worked Example 2.1.1 already exercised this without naming it: `torch::TensorOptions opts3 = torch::kFloat32;` compiles because `TensorOptions` has a non-explicit, implicit constructor taking a `ScalarType`; `torch::TensorOptions opts4 = torch::kCPU;` compiles because it also has one taking a `Device`. The compiler picks between them purely from the argument's type — `torch::kFloat32` is a `ScalarType`, `torch::kCPU` is (convertible to) a `Device`, and there's no ambiguity to resolve at runtime because there is no runtime step here at all.

| Constructor form | What triggers it |
|---|---|
| `TensorOptions()` | No arguments |
| `TensorOptions(ScalarType)` (implicit) | A bare dtype value, e.g. `torch::kFloat32` |
| `TensorOptions(Device)` (implicit) | A bare device value, e.g. `torch::kCPU` |

Value semantics follow directly from how the builder methods are declared: `dtype()`, `device()`, and `requires_grad()` are all `const` member functions marked `[[nodiscard]]`, each returning a *new* `TensorOptions` by value. A `const` method cannot modify the object it's called on even if it wanted to — `opts1.dtype(torch::kFloat64)` is guaranteed, by the method's own signature, to leave `opts1` exactly as it was, which is exactly what Worked Example 2.1.1's genuine output already confirmed: `opts1.has_dtype()` still reports `0` after `opts2` was built from it.

## 2.3 Templates and Compile-Time Specialization, and a View That Doesn't Own Its Data `[FOUNDATIONAL]`

### Intuition

`torch::TensorAccessor<T, N>` is a real LibTorch template — a lightweight, typed, sized window onto a tensor's existing data, letting code index into it with ordinary `[]` syntax and a compile-time-known element type instead of going through `Tensor`'s general (and slower) `.index()` machinery. Instantiating it for two different `T`s produces two entirely separate pieces of compiled machine code, the same proof the CUDA C++ book's own `Vector<T, N>` demonstration relies on — but unlike `Vector<T, N>`, which stores its `N` elements *inside itself*, `TensorAccessor` stores nothing but pointers, and that difference shows up directly in what `sizeof` reports.

### Background

A template is a pattern, not compiled code — the compiler only emits real machine code once it sees a concrete combination of template arguments, and it emits one, fully independent copy per distinct combination. `torch::TensorAccessor<T, N>` (aliased from a shared implementation in `torch::headeronly::detail::TensorAccessor`) holds a `T*` data pointer plus two `index_t*` pointers for the tensor's sizes and strides — three pointers, nothing else, regardless of what `T` or `N` are. That's the structural difference from the CUDA book's `Vector<T, N>`, which embeds a real `T data[N]` array as a member and therefore genuinely changes size with both `T` and `N`. `TensorAccessor` is a *view*: it never owns the memory it lets you index into, so its own footprint doesn't depend on what it's a view of.

### Worked Example 2.3.1 — two real instantiations, proven with `nm`

```cpp
#include <torch/torch.h>
#include <iostream>

// A small template function around Tensor::accessor<T,N>(), mirroring the way
// this book's CUDA sibling proves template instantiation with its own Vector<T,N>::sum().
// Kept noinline so nm can show it as a distinct symbol per instantiation, not folded away.
template <typename T, int N>
__attribute__((noinline))
T accessor_sum(const torch::Tensor& t) {
    auto acc = t.accessor<T, N>();
    T total = T(0);
    for (int i = 0; i < acc.size(0); i++) {
        total += acc[i];
    }
    return total;
}

int main() {
    torch::Tensor float_t = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    torch::Tensor int_t = torch::tensor({10, 20, 30}, torch::kInt64);

    float fsum = accessor_sum<float, 1>(float_t);
    int64_t isum = accessor_sum<int64_t, 1>(int_t);

    std::cout << "accessor_sum<float,1>(float_t) = " << fsum << std::endl;
    std::cout << "accessor_sum<int64_t,1>(int_t) = " << isum << std::endl;
    std::cout << "sizeof(torch::TensorAccessor<float,1>) = " << sizeof(torch::TensorAccessor<float,1>) << std::endl;
    std::cout << "sizeof(torch::TensorAccessor<int64_t,1>) = " << sizeof(torch::TensorAccessor<int64_t,1>) << std::endl;

    return 0;
}
```

Genuinely compiled (with `-O0`, so the compiler doesn't inline `accessor_sum` away before `nm` can see it) and run in this book's environment:

```text
accessor_sum<float,1>(float_t) = 10
accessor_sum<int64_t,1>(int_t) = 60
sizeof(torch::TensorAccessor<float,1>) = 24
sizeof(torch::TensorAccessor<int64_t,1>) = 24
```

And the compiled binary's own symbol table, read with `nm -C`:

```text
000000000001754c W float accessor_sum<float, 1>(at::Tensor const&)
00000000000175fc W long accessor_sum<long, 1>(at::Tensor const&)
```

(`long` is this platform's real underlying type for `int64_t` — `nm`'s demangler reports it that way, not a discrepancy.) Two distinct symbols at two distinct addresses in one compiled binary, exactly the CUDA book's own evidence for "this is genuinely separate machine code, not one generic function branching on a type tag" — reused here on a real LibTorch template instead of a purpose-built teaching one. `sizeof` tells the other half of the story: both instantiations measure exactly 24 bytes (three 8-byte pointers on this platform), identical for `float` and `int64_t` alike, because neither instantiation stores a single element of the tensor it's viewing — only where to find them.

**Independent cross-check.** Both sums were re-verified through LibTorch's Python bindings, calling into the same underlying `libtorch_cpu.so` through a different entry point than the `accessor<T,N>()` call above:

```python
import torch
torch.tensor([1.0, 2.0, 3.0, 4.0]).sum().item()          # -> 10.0
torch.tensor([10, 20, 30], dtype=torch.int64).sum().item()  # -> 60
```

Genuinely run: both values agree exactly with the C++ output above.

## 2.4 RAII: Constructors, Destructors, and LibTorch's Own Allocator `[FOUNDATIONAL]`

### Intuition

RAII — Resource Acquisition Is Initialization — ties a resource's lifetime to an object's scope: a constructor acquires it, a destructor releases it automatically, no matter how the scope is left. The CUDA C++ book has to build this pattern from nothing, wrapping hand-called `cudaMalloc`/`cudaFree` in a struct it writes itself. LibTorch never asks a user of this book to do that — every `torch::Tensor`'s storage is already RAII-managed, by `c10::DataPtr` and the allocator behind it, and this chapter's worked example genuinely borrows that same real mechanism rather than reimplementing `new`/`delete` tracing by hand.

### Background

`c10::GetCPUAllocator()` returns LibTorch's real, singleton CPU allocator — the exact one a CPU `torch::Tensor`'s storage allocates from — and calling `.allocate(n)` on it returns a `c10::DataPtr`: itself an RAII type, releasing its memory in its own destructor. Wrapping one in a struct with trace prints makes construction and destruction order directly observable, the same technique the CUDA book's `DeviceBuffer` example uses, but built on LibTorch's genuine allocator instead of a raw `cudaMalloc`/`cudaFree` pair.

| | Manual `new`/`delete` (or raw `cudaMalloc`/`cudaFree`) | This chapter's `HostBuffer` |
|---|---|---|
| Who releases the memory | You, explicitly, at every exit path | `c10::DataPtr`'s own destructor, automatically |
| What allocator is used | Whatever you call directly | `c10::GetCPUAllocator()` — the same one `torch::Tensor` itself uses |
| Forgetting to release | Silent leak | Not possible without forgetting the variable declaration itself |

### Worked Example 2.4.1 — construction and destruction order, then a real `Tensor`'s own

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>

// RAII around LibTorch's own real CPU allocator (the same allocator a CPU
// torch::Tensor's storage genuinely uses) -- not a hand-rolled new/delete.
// Addresses are deliberately not printed: they're genuine but change every run
// (heap layout), so only "was memory actually acquired" (a non-null DataPtr)
// and construction/destruction *order* are reported.
struct HostBuffer {
    c10::DataPtr ptr;
    size_t n;

    HostBuffer(size_t n_) : ptr(c10::GetCPUAllocator()->allocate(n_ * sizeof(float))), n(n_) {
        std::cout << "  HostBuffer(" << n << ") constructed -> allocated via c10::GetCPUAllocator(), non-null: "
                  << (ptr.get() != nullptr) << std::endl;
    }

    ~HostBuffer() {
        std::cout << "  ~HostBuffer(size=" << n << ") destructor firing" << std::endl;
        // ptr's own destructor (c10::DataPtr is itself RAII) releases the memory here.
    }
};

void scoped_demo() {
    std::cout << "entering scoped_demo" << std::endl;
    HostBuffer a(100);
    HostBuffer b(200);
    std::cout << "both buffers constructed, about to leave scope" << std::endl;
}

int main() {
    scoped_demo();
    std::cout << "back in main, both destructors already ran" << std::endl;

    // Now: torch::Tensor's OWN storage, genuinely observed firing a real deleter at scope exit,
    // via torch::from_blob's deleter callback -- not a separate hand-built mechanism.
    std::cout << "--- torch::Tensor's own RAII, observed directly ---" << std::endl;
    {
        float* raw = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
        torch::Tensor t = torch::from_blob(raw, {4}, [](void* p) {
            std::cout << "  Tensor-owned deleter firing, freeing the same buffer from_blob was given" << std::endl;
            delete[] static_cast<float*>(p);
        });
        std::cout << "  tensor constructed, sum = " << t.sum().item<float>() << std::endl;
    }
    std::cout << "back in main, the from_blob tensor's deleter already ran" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
entering scoped_demo
  HostBuffer(100) constructed -> allocated via c10::GetCPUAllocator(), non-null: 1
  HostBuffer(200) constructed -> allocated via c10::GetCPUAllocator(), non-null: 1
both buffers constructed, about to leave scope
  ~HostBuffer(size=200) destructor firing
  ~HostBuffer(size=100) destructor firing
back in main, both destructors already ran
--- torch::Tensor's own RAII, observed directly ---
  tensor constructed, sum = 10
  Tensor-owned deleter firing, freeing the same buffer from_blob was given
back in main, the from_blob tensor's deleter already ran
```

`a` and `b` are declared in that order and destroyed in the exact reverse — `~HostBuffer(size=200)` fires before `~HostBuffer(size=100)` — the same LIFO guarantee C++ gives any scope's local objects, genuinely captured here on LibTorch's real allocator instead of a toy one. The second half is the more interesting result: `torch::from_blob` lets a tensor wrap externally-owned memory and take a deleter callback to run when that tensor's storage is finally released, and the trace print shows that deleter genuinely firing — after the tensor was used (`sum = 10`, matching `1+2+3+4`), and before the enclosing block's own closing brace finishes executing. This is not a separate demonstration mechanism standing in for how `torch::Tensor` manages memory — it *is* how `torch::Tensor` manages memory, observed directly rather than asserted.

## 2.5 Compile-Time Interfaces: What `torch::nn::Cloneable`'s Real CRTP Actually Buys `[FOUNDATIONAL]`

### Intuition

The CUDA C++ book's Chapter 2 closes with a clean textbook contrast: virtual dispatch costs a vtable pointer and a runtime indirect call, CRTP costs neither, so a hand-built `CircleCRTP` measures smaller than a hand-built `Circle`. LibTorch's own `torch::nn` module has a real, documented CRTP use — `torch::nn::Cloneable<Derived>` — and genuinely measuring it tells a different, more honest story: `Cloneable<Derived>` inherits from `torch::nn::Module`, which already needs a vtable so that `forward()` and other methods can be called polymorphically through a base pointer, so there's no vtable left to dodge. CRTP earns its place here for a completely different reason than the textbook one.

### Background

`torch::nn::Module`'s own `clone()` cannot know, from inside the base class, what concrete type a given module actually is — it would need that type to construct a copy of it. LibTorch's own header comment on `Cloneable` states the design plainly: it uses CRTP so that each derived module tells the compiler its own concrete type, letting `Cloneable<Derived>::clone()` be written once and reused by every subclass, *without* forcing `Module` itself to become a template — which would have made storing modules polymorphically (as `std::shared_ptr<Module>`, the way `torch::nn` containers actually do it) impossible. That is a different problem than the one CRTP solves in the CUDA book's toy example, and it's worth being precise about the difference rather than assuming the two must match:

| | CUDA book's `CircleCRTP` example | `torch::nn::Cloneable<Derived>` |
|---|---|---|
| What CRTP is being used to avoid | A vtable pointer and its runtime dispatch cost | Templatizing the *base* class (`Module`) just to give `clone()` a default body |
| Does the resulting type still have a vtable? | No — no virtual functions at all | Yes — it still inherits `Module`'s virtual `forward()`, `clone()`, and others |
| Expected `sizeof` difference vs. a plain virtual sibling | Large — a whole vtable pointer's worth | None beyond whatever data members the derived type itself adds |

### Worked Example 2.5.1 — measuring the difference honestly, then genuinely cloning

```cpp
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>

// Plain virtual dispatch: DoubleIt overrides Module's virtual forward-like method
// through an ordinary hand-written override, no CRTP involved.
struct Doubler : torch::nn::Module {
    torch::Tensor apply(const torch::Tensor& x) { return x * 2; }
};

// LibTorch's real CRTP mechanism: Cloneable<Derived> gives clone() a default
// implementation that needs to know the concrete derived type, without
// templatizing Module itself.
struct Adder : torch::nn::Cloneable<Adder> {
    torch::Tensor bias;
    void reset() override {
        bias = register_parameter("bias", torch::ones({3}));
    }
    torch::Tensor forward(const torch::Tensor& x) { return x + bias; }
};

int main() {
    std::cout << "sizeof(torch::nn::Module) = " << sizeof(torch::nn::Module) << std::endl;
    std::cout << "sizeof(Doubler) (plain virtual override, no CRTP) = " << sizeof(Doubler) << std::endl;
    std::cout << "sizeof(Adder) (torch::nn::Cloneable<Adder>, real CRTP) = " << sizeof(Adder) << std::endl;

    // Virtual dispatch: stored and called through a base Module pointer.
    std::vector<std::shared_ptr<torch::nn::Module>> modules;
    auto d = std::make_shared<Doubler>();
    modules.push_back(d);
    std::cout << "virtual-dispatch storage: modules.size() = " << modules.size() << std::endl;

    // CRTP: Adder's reset() must run once to materialize its parameter.
    auto original = std::make_shared<Adder>();
    original->reset();
    torch::Tensor x = torch::zeros({3});
    torch::Tensor y = original->forward(x);
    std::cout << "original->forward(zeros(3)) = " << y << std::endl;

    // clone() is Cloneable's real CRTP payoff: a working deep copy with no
    // hand-written clone() on Adder itself.
    auto cloned = std::dynamic_pointer_cast<Adder>(original->clone());
    { torch::NoGradGuard no_grad; cloned->bias.add_(10.0); }  // mutate the CLONE's parameter, out-of-place w.r.t. autograd
    std::cout << "original->bias = " << original->bias << std::endl;
    std::cout << "cloned->bias   = " << cloned->bias << std::endl;
    std::cout << "same underlying storage? " << (original->bias.data_ptr() == cloned->bias.data_ptr()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
sizeof(torch::nn::Module) = 408
sizeof(Doubler) (plain virtual override, no CRTP) = 408
sizeof(Adder) (torch::nn::Cloneable<Adder>, real CRTP) = 416
virtual-dispatch storage: modules.size() = 1
original->forward(zeros(3)) =  1
 1
 1
[ CPUFloatType{3} ]
original->bias =  1
 1
 1
[ CPUFloatType{3} ]
cloned->bias   =  11
 11
 11
[ CPUFloatType{3} ]
same underlying storage? 0
```

`Doubler`, which adds no CRTP and no new members, measures exactly the same 408 bytes as bare `torch::nn::Module` — there was never a vtable to save by using CRTP here in the first place, because ordinary virtual dispatch was already fully in play the moment `Doubler` inherited from `Module`. `Adder`, built on `Cloneable<Adder>`, measures 408 + 8 = 416 — and that extra 8 bytes is exactly `sizeof(torch::Tensor)` (a `Tensor` is a thin handle around a reference-counted pointer, not a data-owning struct itself), i.e. precisely the cost of the one `bias` member `Adder` actually declared. No vtable pointer was avoided, because none could have been — `Cloneable<Derived>` still inherits every one of `Module`'s virtual functions. What CRTP genuinely bought here is visible in the next four lines instead: `Adder` never wrote its own `clone()`, and `original->clone()` still produced a real, independent copy — `cloned->bias` diverges from `original->bias` after mutation, and `data_ptr()` confirms they're genuinely backed by different storage, not two handles to the same tensor.

> `[COMMON TRAP]` It's tempting to read "CRTP" and assume "no vtable" by reflex, because that's the lesson the classic textbook example teaches. `torch::nn::Cloneable<Derived>` is real, shipped, documented CRTP, and it demonstrates directly that the two ideas are separable: CRTP is a compile-time mechanism for giving a class knowledge of its own concrete subtype, and *sometimes* that's used to eliminate a vtable (the CUDA book's `CircleCRTP`), but here it's used for something else entirely — sharing one `clone()` implementation across every `Cloneable` subclass without templatizing `Module` itself — on a type that was always going to carry a vtable regardless.

## Complete Runnable Code

### File: `01_tensor_options.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "sizeof(torch::TensorOptions) = " << sizeof(torch::TensorOptions) << std::endl;

    torch::TensorOptions opts1;
    std::cout << "default opts1.has_dtype() = " << opts1.has_dtype() << std::endl;

    // builder-style chaining: each call returns a NEW TensorOptions by value
    torch::TensorOptions opts2 = opts1.dtype(torch::kFloat64).device(torch::kCPU).requires_grad(true);
    std::cout << "opts2.dtype() = " << opts2.dtype() << ", requires_grad = " << opts2.requires_grad() << std::endl;

    // opts1 itself: unchanged, since dtype()/device()/requires_grad() are all const methods
    std::cout << "opts1.has_dtype() after building opts2 = " << opts1.has_dtype() << std::endl;

    // implicit conversion constructors: TensorOptions(ScalarType) and TensorOptions(Device)
    torch::TensorOptions opts3 = torch::kFloat32;   // resolves via TensorOptions(ScalarType)
    torch::TensorOptions opts4 = torch::kCPU;        // resolves via TensorOptions(Device)
    std::cout << "opts3.dtype() = " << opts3.dtype() << ", opts3.has_device() = " << opts3.has_device() << std::endl;
    std::cout << "opts4.has_dtype() = " << opts4.has_dtype() << ", opts4.device() = " << opts4.device() << std::endl;

    // use it to build a real tensor
    torch::Tensor t = torch::ones({2, 2}, opts2);
    std::cout << "t.dtype() = " << t.dtype() << ", t.requires_grad() = " << t.requires_grad() << std::endl;

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 01_tensor_options.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 01_tensor_options
./01_tensor_options
```

### File: `02_accessor_templates.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// A small template function around Tensor::accessor<T,N>(), mirroring the way
// this book's CUDA sibling proves template instantiation with its own Vector<T,N>::sum().
// Kept noinline so nm can show it as a distinct symbol per instantiation, not folded away.
template <typename T, int N>
__attribute__((noinline))
T accessor_sum(const torch::Tensor& t) {
    auto acc = t.accessor<T, N>();
    T total = T(0);
    for (int i = 0; i < acc.size(0); i++) {
        total += acc[i];
    }
    return total;
}

int main() {
    torch::Tensor float_t = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    torch::Tensor int_t = torch::tensor({10, 20, 30}, torch::kInt64);

    float fsum = accessor_sum<float, 1>(float_t);
    int64_t isum = accessor_sum<int64_t, 1>(int_t);

    std::cout << "accessor_sum<float,1>(float_t) = " << fsum << std::endl;
    std::cout << "accessor_sum<int64_t,1>(int_t) = " << isum << std::endl;
    std::cout << "sizeof(torch::TensorAccessor<float,1>) = " << sizeof(torch::TensorAccessor<float,1>) << std::endl;
    std::cout << "sizeof(torch::TensorAccessor<int64_t,1>) = " << sizeof(torch::TensorAccessor<int64_t,1>) << std::endl;

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O0 02_accessor_templates.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 02_accessor_templates
./02_accessor_templates
nm -C 02_accessor_templates | grep accessor_sum
```

### File: `03_raii_allocator.cpp`

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>

// RAII around LibTorch's own real CPU allocator (the same allocator a CPU
// torch::Tensor's storage genuinely uses) -- not a hand-rolled new/delete.
// Addresses are deliberately not printed: they're genuine but change every run
// (heap layout), so only "was memory actually acquired" (a non-null DataPtr)
// and construction/destruction *order* are reported.
struct HostBuffer {
    c10::DataPtr ptr;
    size_t n;

    HostBuffer(size_t n_) : ptr(c10::GetCPUAllocator()->allocate(n_ * sizeof(float))), n(n_) {
        std::cout << "  HostBuffer(" << n << ") constructed -> allocated via c10::GetCPUAllocator(), non-null: "
                  << (ptr.get() != nullptr) << std::endl;
    }

    ~HostBuffer() {
        std::cout << "  ~HostBuffer(size=" << n << ") destructor firing" << std::endl;
        // ptr's own destructor (c10::DataPtr is itself RAII) releases the memory here.
    }
};

void scoped_demo() {
    std::cout << "entering scoped_demo" << std::endl;
    HostBuffer a(100);
    HostBuffer b(200);
    std::cout << "both buffers constructed, about to leave scope" << std::endl;
}

int main() {
    scoped_demo();
    std::cout << "back in main, both destructors already ran" << std::endl;

    // Now: torch::Tensor's OWN storage, genuinely observed firing a real deleter at scope exit,
    // via torch::from_blob's deleter callback -- not a separate hand-built mechanism.
    std::cout << "--- torch::Tensor's own RAII, observed directly ---" << std::endl;
    {
        float* raw = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
        torch::Tensor t = torch::from_blob(raw, {4}, [](void* p) {
            std::cout << "  Tensor-owned deleter firing, freeing the same buffer from_blob was given" << std::endl;
            delete[] static_cast<float*>(p);
        });
        std::cout << "  tensor constructed, sum = " << t.sum().item<float>() << std::endl;
    }
    std::cout << "back in main, the from_blob tensor's deleter already ran" << std::endl;

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 03_raii_allocator.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 03_raii_allocator
./03_raii_allocator
```

### File: `04_crtp_vs_virtual.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>

// Plain virtual dispatch: DoubleIt overrides Module's virtual forward-like method
// through an ordinary hand-written override, no CRTP involved.
struct Doubler : torch::nn::Module {
    torch::Tensor apply(const torch::Tensor& x) { return x * 2; }
};

// LibTorch's real CRTP mechanism: Cloneable<Derived> gives clone() a default
// implementation that needs to know the concrete derived type, without
// templatizing Module itself.
struct Adder : torch::nn::Cloneable<Adder> {
    torch::Tensor bias;
    void reset() override {
        bias = register_parameter("bias", torch::ones({3}));
    }
    torch::Tensor forward(const torch::Tensor& x) { return x + bias; }
};

int main() {
    std::cout << "sizeof(torch::nn::Module) = " << sizeof(torch::nn::Module) << std::endl;
    std::cout << "sizeof(Doubler) (plain virtual override, no CRTP) = " << sizeof(Doubler) << std::endl;
    std::cout << "sizeof(Adder) (torch::nn::Cloneable<Adder>, real CRTP) = " << sizeof(Adder) << std::endl;

    // Virtual dispatch: stored and called through a base Module pointer.
    std::vector<std::shared_ptr<torch::nn::Module>> modules;
    auto d = std::make_shared<Doubler>();
    modules.push_back(d);
    std::cout << "virtual-dispatch storage: modules.size() = " << modules.size() << std::endl;

    // CRTP: Adder's reset() must run once to materialize its parameter.
    auto original = std::make_shared<Adder>();
    original->reset();
    torch::Tensor x = torch::zeros({3});
    torch::Tensor y = original->forward(x);
    std::cout << "original->forward(zeros(3)) = " << y << std::endl;

    // clone() is Cloneable's real CRTP payoff: a working deep copy with no
    // hand-written clone() on Adder itself.
    auto cloned = std::dynamic_pointer_cast<Adder>(original->clone());
    { torch::NoGradGuard no_grad; cloned->bias.add_(10.0); }  // mutate the CLONE's parameter, out-of-place w.r.t. autograd
    std::cout << "original->bias = " << original->bias << std::endl;
    std::cout << "cloned->bias   = " << cloned->bias << std::endl;
    std::cout << "same underlying storage? " << (original->bias.data_ptr() == cloned->bias.data_ptr()) << std::endl;

    return 0;
}
```

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 04_crtp_vs_virtual.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o 04_crtp_vs_virtual
./04_crtp_vs_virtual
```

## Chapter Summary

`torch::TensorOptions` is a struct genuinely engineered down to a measured 8 bytes, using bitfields instead of `std::optional` per field specifically to keep its seven presence flags packed into a single byte, backed by its own `static_assert` promising it will never exceed 16 — and its builder methods (`.dtype()`, `.device()`, `.requires_grad()`) are `const`, `[[nodiscard]]` functions that return a new object rather than mutating the one they're called on, giving chained construction genuine, verifiable value semantics. `torch::TensorAccessor<T, N>`, a real template this book's own code instantiated twice, proves template specialization the same way the CUDA C++ book's `Vector<T, N>` does — two separate compiled symbols, confirmed with `nm` — while measuring identically at 24 bytes for both `float` and `int64_t`, because it's a three-pointer view onto existing tensor data rather than a container that embeds its own elements. RAII needed no reconstruction from scratch: `c10::GetCPUAllocator()` and `c10::DataPtr` are LibTorch's own real allocation machinery, and this chapter observed a real `torch::Tensor`'s deleter genuinely fire at scope exit through `torch::from_blob`, not a hand-built stand-in for how `Tensor` manages memory. `torch::nn::Cloneable<Derived>` closed the chapter as LibTorch's own real, documented CRTP use — genuinely exercised through a working `.clone()` that produces independently-storage-backed copies — but measuring it honestly overturns the textbook "CRTP means no vtable" assumption: `Cloneable<Derived>` still inherits `Module`'s full vtable, and the pattern is being used here to avoid templatizing the *base* class, not to avoid virtual dispatch at all.

## Self-Check Questions

1. `torch::TensorOptions` could have used `std::optional<T>` for each of its five settable fields instead of a separate `has_*_` bitfield per field. What does the header's own `static_assert` tell you about why its authors didn't, and what would you expect to happen to `sizeof(TensorOptions)` if it had?
2. `torch::TensorOptions opts3 = torch::kFloat32;` and `torch::TensorOptions opts4 = torch::kCPU;` compile against two different constructors, chosen with no runtime branching at all. What C++ mechanism selects between them, and when does that selection happen?
3. `torch::TensorAccessor<float, 1>` and `torch::TensorAccessor<int64_t, 1>` both measure 24 bytes with `sizeof`, despite wrapping tensors of two different dtypes. Why doesn't the element type change the accessor's own size, and how would that answer change for a template that stored its elements directly (like the CUDA book's `Vector<T, N>`) instead of just pointing at them?
4. This chapter's `HostBuffer` struct never calls `c10::GetCPUAllocator()`'s deallocation function directly anywhere in its own code. What actually frees the memory it allocated, and at what point in the program does that happen?
5. `sizeof(Doubler)` and `sizeof(torch::nn::Module)` come out identical in this chapter's genuine measurement, while `sizeof(Adder)` is 8 bytes larger. Given that `Adder` is the one built on `torch::nn::Cloneable<Adder>` (real CRTP) and `Doubler` isn't, what does that 8-byte gap actually represent, and why doesn't it represent "the cost of a vtable"?

## Where We Go Next

Every struct this chapter examined was self-contained — `TensorOptions` describes settings, `TensorAccessor` views one tensor's data, `HostBuffer` owns one allocation. Chapter 3 moves to the piece of memory-layout reasoning this book's own `Tensor` type is actually built on: how shape and stride together turn one flat buffer into something that can be indexed as multi-dimensional data at all, and what LibTorch's own `c10::IntArrayRef`-based sizes and strides genuinely do differently from a `TensorAccessor`'s fixed compile-time `N`.

## Worked Solutions

**1.** The `static_assert(sizeof(TensorOptions) <= sizeof(int64_t) * 2)` tells you the authors made a hard, compile-time-enforced promise that this struct will never exceed 16 bytes, no matter how many settings get added to it later — and the header's own comment states directly that packing the seven `has_*_` presence flags into single bits was necessary specifically because `std::optional<T>` couldn't be packed that tightly. Had they used `std::optional<T>` per field instead, `sizeof(TensorOptions)` would very likely have grown well past that 16-byte ceiling — each `std::optional<T>` typically costs at least `sizeof(T)` plus its own presence flag, with no sharing between fields the way the bitfield design achieves, and the `static_assert` itself would probably have started failing to compile.

**2.** Overload resolution — specifically, resolution among `TensorOptions`'s implicit (non-`explicit`) single-argument constructors — is what selects between them, matching the initializer's static type (`ScalarType` for `torch::kFloat32`, `Device` for `torch::kCPU`) against the available constructor signatures. That selection happens entirely at compile time: by the time the program runs, `opts3`'s and `opts4`'s constructors have already been fully resolved and compiled into ordinary function calls, with no type inspection or branching happening while the program executes.

**3.** `TensorAccessor<T, N>` doesn't change size with `T` because it never stores a single element of the tensor it's viewing — it holds three pointers (to the data, to the sizes array, and to the strides array), and a pointer is the same size regardless of what it points at. A template that stores its elements directly, the way the CUDA book's `Vector<T, N>` embeds `T data[N]` as an actual member, would genuinely change size with both `T` (each element's own width) and `N` (how many of them) — exactly why the CUDA book's own `Vector<float, 4>` and `Vector<int, 3>` measure different sizes from each other, while this chapter's two `TensorAccessor` instantiations measure identically.

**4.** `c10::DataPtr`'s own destructor is what actually frees the memory — `HostBuffer` only ever calls `c10::GetCPUAllocator()->allocate(...)`, once, in its constructor, storing the resulting `DataPtr` as a member; it never calls anything to release it explicitly. That release genuinely happens when `HostBuffer`'s own destructor runs (at the end of the scope that declared it, in this chapter's case `scoped_demo()`), because a struct's destructor automatically destroys every member that has its own destructor — `ptr`'s destructor runs as part of `~HostBuffer()` running, without `~HostBuffer()` needing to say so explicitly anywhere in its body.

**5.** The 8-byte gap is exactly `sizeof(torch::Tensor)` — the one `bias` member `Adder` declares that neither `Module` nor `Doubler` has. It doesn't represent the cost of a vtable because there was no vtable difference to begin with: `Adder` inherits from `Cloneable<Adder>`, which itself inherits from `Module`, so `Adder` already carries exactly the same vtable `Doubler` carries, by virtue of both ultimately being `Module`s — CRTP added a compile-time capability (a default, reusable `clone()` that knows its own concrete type) without adding or removing a single virtual function from the hierarchy, which is exactly why the size difference between `Adder` and `Doubler` is fully accounted for by ordinary data-member size, not vtable presence.
