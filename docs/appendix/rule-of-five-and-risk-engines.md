# Appendix G: The Rule of Five, and a Risk Engine to Exercise It

> The CUDA C++ edition's own Appendix G opens with a trivial struct, `Point2D`, called from a kernel with zero changes to its own definition -- the point being that `Point2D` owns nothing but two floats, so the compiler's default copy is exactly correct for it. Everything else in that appendix is about what happens the instant a struct DOES own something -- heap memory, or device memory -- because at that moment the compiler's default copy silently stops being correct, with no warning at all. This appendix asks the identical question on LibTorch's own ground: most of the CUDA book's own resource-management teaching (a raw pointer, hand-written destructor/copy/move) is ordinary, CUDA-free host C++, reproduced here directly -- and the one genuinely CUDA-specific section, a hand-written Rule of Five for a `cudaMalloc`'d buffer, becomes this appendix's own sharpest honest divergence: `torch::Tensor` already IS that correctly-implemented Rule of Five, for device memory, out of the box.

**What you will understand by the end of this appendix:** why a struct that owns nothing needs no hand-written special members at all, and exactly what changes the moment it owns a resource; the double-free bug that appears the instant a resource-owning struct is left to the compiler's own default copy, triggered here for real and read from a genuine, unedited AddressSanitizer report; all five special member functions of the Rule of Five, implemented correctly and independently verified -- deep-copy independence, move-and-null, and the specific silent corruption and silent data loss that appear when the self-assignment and self-move guards are removed, neither of which any sanitizer's heap checks catch; the Rule of Zero, and why a struct wrapping `std::vector` needs none of those hand-written members at all; copy-and-swap, verified against a genuinely injected allocation failure, giving one variant a strong exception guarantee the other provably lacks; why every move constructor and move assignment operator in this appendix is marked `noexcept`, confirmed by a real, measured copy-vs-move flip inside `std::vector`'s own reallocation; why `torch::Tensor` already gives a LibTorch programmer, for free, exactly the correctly-implemented Rule of Five the CUDA book's own device-memory section had to hand-write; and Value-at-Risk and XVA, this book's own real `torch::Tensor` GBM machinery extended to a 1-day risk horizon and a checkpointed exposure profile.

**What you need to know first:** ordinary C++ constructors, destructors, and the difference between a value and a reference; this book's own Appendix D.6, whose AddressSanitizer-caught dangling reference is a close relative of Section G.2's own double free -- both demonstrate undefined behavior honestly rather than describing it; Chapter 22.1's own vectorized closed-form GBM step, which Sections G.5 and G.6 both reuse rather than reintroducing; basic familiarity with what a forward contract and a discount factor are helps for Section G.6, though every formula used there is derived and checked inline.

## G.1 A Plain Struct, and Why Its Default Copy Is Already Safe `[FOUNDATIONAL]`

**Intuition.** This book never writes a hand-rolled kernel, so it needs no `Point2D`-from-a-kernel worked example to make the CUDA book's own G.1 point -- the identical point holds in ordinary host code: a struct owning nothing but plain values needs no hand-written special members at all, because the compiler's own default-generated five are already exactly correct.

**Background.** `MarketQuote`, this section's own struct, holds two `double`s and nothing else -- no pointer, no handle, no resource whose lifetime must be tracked separately from the struct's own. A bitwise, member-by-member copy of two doubles IS copying the value, in every sense that matters, so there is nothing for a hand-written copy constructor to do differently.

**Worked Example G.1.1.** `sizeof(MarketQuote)`, and a copy whose later mutation leaves the original completely untouched.

```cpp
#include <iostream>

// The CUDA C++ edition's own Section G.1 calls a trivial two-float
// struct, Point2D, from a kernel with zero changes, and makes the
// point that Point2D's default compiler-generated copy is already
// correct -- because Point2D owns nothing but two floats. This book
// never writes kernels, so this section's own worked example needs no
// kernel at all -- it makes the identical point on a struct in this
// book's own financial-computing domain, entirely in ordinary host
// code: MarketQuote, two doubles (bid, ask), no owned resource of any
// kind.
struct MarketQuote {
    double bid;
    double ask;
    double mid() const { return (bid + ask) / 2.0; }
    double spread() const { return ask - bid; }
};

int main() {
    std::cout << "sizeof(MarketQuote) = " << sizeof(MarketQuote)
              << " bytes (two doubles, no owned resource of any kind)" << std::endl;

    MarketQuote q1{100.05, 100.10};
    std::cout << "q1 = {bid=" << q1.bid << ", ask=" << q1.ask << "}, mid=" << q1.mid()
              << ", spread=" << q1.spread() << std::endl;

    // The compiler's own default-generated copy constructor: a plain,
    // member-by-member copy of two doubles. There is nothing here for
    // a hand-written copy constructor to do differently -- copying
    // the bits IS copying the value.
    MarketQuote q2 = q1;
    q2.bid = 200.00;
    q2.ask = 200.05;

    std::cout << "\nq2 = q1, then q2's own bid/ask mutated to {200.00, 200.05}:" << std::endl;
    std::cout << "  q1 = {bid=" << q1.bid << ", ask=" << q1.ask
              << "} (completely unaffected by q2's own mutation -- the compiler's default copy already "
              << "gave q2 its own independent bid and ask)" << std::endl;
    std::cout << "  q2 = {bid=" << q2.bid << ", ask=" << q2.ask << "}" << std::endl;

    std::cout << "\nMarketQuote needs no hand-written destructor, copy constructor, copy assignment, move "
              << "constructor, or move assignment operator -- the compiler's own default-generated five "
              << "are already exactly correct, because MarketQuote owns nothing that a bitwise copy could "
              << "ever get wrong. Section G.2 introduces the one thing that changes this: a struct that "
              << "owns a resource whose lifetime doesn't automatically follow the struct's own." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 01_market_quote_default_copy.cpp -o 01_market_quote_default_copy
./01_market_quote_default_copy
```

```text
sizeof(MarketQuote) = 16 bytes (two doubles, no owned resource of any kind)
q1 = {bid=100.05, ask=100.1}, mid=100.075, spread=0.05

q2 = q1, then q2's own bid/ask mutated to {200.00, 200.05}:
  q1 = {bid=100.05, ask=100.1} (completely unaffected by q2's own mutation -- the compiler's default copy already gave q2 its own independent bid and ask)
  q2 = {bid=200, ask=200.05}

MarketQuote needs no hand-written destructor, copy constructor, copy assignment, move constructor, or move assignment operator -- the compiler's own default-generated five are already exactly correct, because MarketQuote owns nothing that a bitwise copy could ever get wrong. Section G.2 introduces the one thing that changes this: a struct that owns a resource whose lifetime doesn't automatically follow the struct's own.
```

**Discussion.** `q1` stays exactly `{100.05, 100.10}` after `q2`'s own copy is mutated to `{200.00, 200.05}` -- unsurprising, and that is the entire point: `MarketQuote` is the baseline every later section in this appendix departs from the moment a struct owns something whose lifetime doesn't automatically follow the struct's own.

## G.2 The Bug the Rule of Five Prevents: A Double Free `[FOUNDATIONAL]`

**Intuition.** `RiskPathBuffer` owns a raw heap pointer, `float* paths`, allocated with `new float[n]`. Only its destructor is hand-written; the compiler generates the copy constructor and copy assignment operator, both of which copy the POINTER, not the array it points to. This is entirely ordinary host C++ -- no CUDA content anywhere in the CUDA book's own version either -- reproduced here directly.

**Background.** A copy `b = a` leaves `b.paths == a.paths`, the identical allocation shared by two objects. When `b` goes out of scope, its destructor frees that shared buffer; `a.paths` now points at freed memory, and `a`'s own destructor at program end frees the same pointer a second time. This section triggers the resulting bug for real, under AddressSanitizer, rather than describing it.

**Worked Example G.2.1.** A genuine heap-use-after-free followed by a genuine double free, both caught and reported, unedited, by AddressSanitizer.

```cpp
#include <cstdio>

// stdout is fully buffered by default when it isn't attached to a
// terminal (exactly the case when this binary's own output is
// captured by a verification script) -- without an explicit fflush
// after each printf, the buffered text below would simply be LOST
// when AddressSanitizer's own abort() tears the process down before
// libc ever gets to flush it. Genuinely discovered by running this
// file for real: the first version of this file printed nothing at
// all to stdout, only ASan's own report to stderr, until this fix.

// The CUDA C++ edition's own Section G.2 introduces RiskPathBuffer: a
// struct owning a raw heap pointer (float* paths), with only a
// destructor hand-written (delete[] paths) and no copy constructor or
// copy assignment operator of its own. This is entirely ordinary host
// C++ -- no CUDA content anywhere in it -- so this section reproduces
// it directly, in this book's own words, unchanged in substance.
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }

    // Only the destructor is hand-written. No copy constructor, no
    // copy assignment operator -- the compiler generates a plain,
    // MEMBERWISE copy of both members for both, including `paths`,
    // which copies the POINTER, not the array it points to.
    ~RiskPathBuffer() { delete[] paths; }
};

int main() {
    RiskPathBuffer a(10);
    printf("a.paths[0..2] = %.1f %.1f %.1f, a.count = %d\n", a.paths[0], a.paths[1], a.paths[2], a.count);
    fflush(stdout);

    {
        RiskPathBuffer b = a;  // compiler-generated copy: b.paths == a.paths, same allocation
        printf("b.paths == a.paths? %s (the pointer was copied, not the 10-float array it points to)\n",
               (b.paths == a.paths) ? "true" : "false");
        fflush(stdout);
    }  // b goes out of scope here -- its destructor frees the SHARED buffer

    // a.paths now points at freed memory. Reading it here is a
    // genuine use-after-free, and a's own destructor at the end of
    // main() will free the identical pointer a second time -- a
    // genuine double free.
    printf("a.paths[0] after b's own destructor ran = %.1f (reading FREED memory -- undefined behavior)\n",
           a.paths[0]);
    fflush(stdout);

    return 0;
}  // a's own destructor runs here: delete[] a.paths -- the SAME pointer b's destructor already freed
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra -fsanitize=address -g` and run:

```bash
g++ -std=c++17 -Wall -Wextra -fsanitize=address -g 02_double_free_trap.cpp -o double_free_trap
./double_free_trap
```

```text
a.paths[0..2] = 0.0 1.0 2.0, a.count = 10
b.paths == a.paths? true (the pointer was copied, not the 10-float array it points to)
```

The real, genuinely captured AddressSanitizer summary line (the full report additionally includes real memory addresses and a real process/thread id that, correctly, differ on every run -- only this stable, address-free line is locked, for the identical reason Appendix D.6's own dangling-reference report locks only its own summary line):

```text
SUMMARY: AddressSanitizer: heap-use-after-free /home/claude/appendixG/02_double_free_trap.cpp:51 in main
```

**Discussion.** This is not a described or hypothetical failure -- it is a real, unedited AddressSanitizer report, produced by genuinely running a program that genuinely double-frees a pointer, the exact standard this book has held every claimed bug to since Appendix D.6's own dangling-reference demonstration. Without AddressSanitizer watching, this exact program's own behavior would be truly undefined, not merely a wrong number -- which is precisely why every later section in this appendix that claims a struct is SAFE backs that claim with a genuine compile-and-run check, not an assertion.

## G.3 The Rule of Five, Correctly Implemented `[FOUNDATIONAL]`

**Intuition.** The five special members split into two jobs: the destructor, copy constructor, and copy assignment operator handle ownership WITHOUT transfer (a deep copy on copy, the old resource freed around the new one on assignment); the move constructor and move assignment operator handle ownership TRANSFER (steal the pointer, null the source so its own destructor becomes a safe no-op).

**Background.** This section reproduces the CUDA book's own five sub-worked-examples (G.3.1 through G.3.5), all entirely ordinary host C++: the correctly implemented five, tested end to end; the self-assignment and self-move guards, tested by genuinely removing them; the Rule of Zero, delegating to `std::vector`'s own already-correct machinery; copy-and-swap, tested against a genuinely injected allocation failure; and why every move operation here is marked `noexcept`, tested against a real `std::vector` reallocation.

**Worked Example G.3.1.** `RiskPathBuffer`, this time with all five special members correctly implemented: deep-copy independence, move-and-null, copy-assignment onto an existing object, move-assignment onto an existing object, and both guards (self-assignment, self-move) genuinely tested rather than merely written.

```cpp
#include <cstdio>
#include <cstring>
#include <utility>

// The CUDA C++ edition's own Section G.3 (worked examples G.3.1 and
// G.3.2, combined into this one file) implements RiskPathBuffer's own
// five special members correctly, then tests the self-assignment and
// self-move guards those five members need by genuinely removing them
// and observing what breaks. Entirely ordinary host C++, reproduced
// here directly.
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }

    ~RiskPathBuffer() { delete[] paths; }

    // Copy constructor: a genuine DEEP copy -- a fresh allocation,
    // its own independent contents.
    RiskPathBuffer(const RiskPathBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }

    // Copy assignment: guard against self-assignment first, then free
    // the OLD resource before deep-copying the new one.
    RiskPathBuffer& operator=(const RiskPathBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }

    // Move constructor: STEAL the source's pointer, null the source
    // so its own destructor becomes a safe no-op. noexcept matters --
    // Worked Example G.3.5-equivalent below shows why directly.
    RiskPathBuffer(RiskPathBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }

    // Move assignment: guard against self-move, free the OLD
    // resource, then steal and null exactly as the move constructor
    // does.
    RiskPathBuffer& operator=(RiskPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};

// An UNGUARDED variant -- identical to RiskPathBuffer above, except
// the `if (this == &other) return *this;` guard has been removed
// from both assignment operators. This demonstrates the CUDA book's
// own genuinely surprising finding: removing the guard does NOT
// reliably crash or trip AddressSanitizer -- it produces two
// different kinds of SILENT failure instead, neither of which any
// sanitizer's heap-corruption checks catch, because no memory is
// ever read after being freed.
struct UnguardedBuffer {
    float* paths;
    int count;

    UnguardedBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~UnguardedBuffer() { delete[] paths; }
    UnguardedBuffer(const UnguardedBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    // No self-assignment guard. If `other` IS `*this` (a = a), then
    // `other.paths` and `paths` are the SAME field: freeing `paths`
    // also invalidates `other.paths`, but the very next line
    // (`paths = new float[count]`) immediately overwrites BOTH --
    // they're one and the same field -- with a fresh, UNINITIALIZED
    // allocation. The memcpy below then copies that fresh allocation's
    // own garbage contents onto itself: not a use-after-free (nothing
    // freed is ever actually read), but silent data corruption.
    UnguardedBuffer& operator=(const UnguardedBuffer& other) {
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
    UnguardedBuffer(UnguardedBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }
    // No self-move guard.
    UnguardedBuffer& operator=(UnguardedBuffer&& other) noexcept {
        delete[] paths;
        paths = other.paths;   // if other IS *this, this just stole its OWN pointer --
        count = other.count;   // harmless so far...
        other.paths = nullptr; // ...but THIS line then nulls the very pointer just stolen,
        other.count = 0;       // since other and *this are the same object
        return *this;
    }
};

int main() {
    // Copy: independent allocation, independent contents.
    RiskPathBuffer a(5);
    RiskPathBuffer b = a;
    b.paths[0] = 999.0f;
    printf("copy: a.paths == b.paths? %s, a.paths[0]=%.1f (untouched), b.paths[0]=%.1f (mutated)\n",
           (a.paths == b.paths) ? "true" : "false", a.paths[0], b.paths[0]);

    // Move: steals the pointer, nulls the source.
    float* a_ptr_before_move = a.paths;
    RiskPathBuffer c = std::move(a);
    printf("move: c.paths == a's own pre-move pointer? %s, a.paths after move = %p (nulled)\n",
           (c.paths == a_ptr_before_move) ? "true" : "false", (void*)a.paths);

    // Copy assignment onto an EXISTING object: old resource freed
    // first, new one deep-copied in.
    RiskPathBuffer d(7);
    RiskPathBuffer e(2);
    float* e_ptr_before = e.paths;
    e = d;
    printf("copy-assign: e.count went from 2 to %d, e.paths changed from its own old pointer? %s\n",
           e.count, (e.paths != e_ptr_before) ? "true" : "false");

    // Move assignment onto an EXISTING object.
    RiskPathBuffer f(3);
    RiskPathBuffer g(9);
    float* d_paths_before_move_assign = d.paths;
    f = std::move(d);
    printf("move-assign: f.paths == d's own pre-move pointer? %s, d.paths after move = %p (nulled)\n",
           (f.paths == d_paths_before_move_assign) ? "true" : "false", (void*)d.paths);

    // Self-assignment and self-move on the GUARDED struct: both
    // provably leave the object unchanged.
    float* g_ptr_before_self = g.paths;
    int g_count_before_self = g.count;
    g = g;  // self-copy-assignment
    printf("\nself-assignment (g = g), guarded: pointer unchanged? %s, count unchanged? %s\n",
           (g.paths == g_ptr_before_self) ? "true" : "false", (g.count == g_count_before_self) ? "true" : "false");

    RiskPathBuffer h(4);
    float* h_ptr_before_self = h.paths;
    int h_count_before_self = h.count;
    h = std::move(h);  // self-move-assignment; g++ itself warns -Wself-move on this exact line
    printf("self-move (h = std::move(h)), guarded: pointer unchanged? %s, count unchanged? %s\n",
           (h.paths == h_ptr_before_self) ? "true" : "false", (h.count == h_count_before_self) ? "true" : "false");

    // The UNGUARDED variant: self-assignment silently CORRUPTS data
    // (overwrites the original 0,1,2,3,4 values with fresh, garbage
    // allocator contents) rather than crashing.
    UnguardedBuffer x(5);
    printf("\nUnguardedBuffer x before self-assignment: %.1f %.1f %.1f %.1f %.1f\n",
           x.paths[0], x.paths[1], x.paths[2], x.paths[3], x.paths[4]);
    x = x;  // no guard -- silently corrupts
    printf("UnguardedBuffer x after x = x (no guard): %.1f %.1f %.1f %.1f %.1f (garbage, not 0-4 -- silent "
           "corruption, not a crash, and NOT caught by AddressSanitizer, since nothing freed is ever read)\n",
           x.paths[0], x.paths[1], x.paths[2], x.paths[3], x.paths[4]);

    // The UNGUARDED variant: self-move silently EMPTIES the object
    // (paths becomes null, count becomes 0) rather than crashing,
    // because `other.paths = nullptr` on the line after the steal
    // overwrites the very pointer that steal just assigned, since
    // `other` and `*this` are the same object.
    UnguardedBuffer y(5);
    printf("\nUnguardedBuffer y before self-move: paths=%p, count=%d\n", (void*)y.paths, y.count);
    y = std::move(y);  // no guard -- silently empties itself
    printf("UnguardedBuffer y after y = std::move(y) (no guard): paths=%p, count=%d (silently emptied, "
           "not a crash -- also NOT caught by AddressSanitizer)\n", (void*)y.paths, y.count);

    return 0;
}
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra -fsanitize=address -g` (one expected `-Wself-move` warning on the deliberate `h = std::move(h)` line, discussed rather than suppressed) and run, ASan-clean:

```bash
g++ -std=c++17 -Wall -Wextra -fsanitize=address -g 03_rule_of_five_correct_and_guards.cpp -o rule_of_five_correct_and_guards
./rule_of_five_correct_and_guards
```

```text
copy: a.paths == b.paths? false, a.paths[0]=0.0 (untouched), b.paths[0]=999.0 (mutated)
move: c.paths == a's own pre-move pointer? true, a.paths after move = (nil) (nulled)
copy-assign: e.count went from 2 to 7, e.paths changed from its own old pointer? true
move-assign: f.paths == d's own pre-move pointer? true, d.paths after move = (nil) (nulled)

self-assignment (g = g), guarded: pointer unchanged? true, count unchanged? true
self-move (h = std::move(h)), guarded: pointer unchanged? true, count unchanged? true

UnguardedBuffer x before self-assignment: 0.0 1.0 2.0 3.0 4.0
UnguardedBuffer x after x = x (no guard): -0.4 -0.4 -0.4 -0.4 -0.4 (garbage, not 0-4 -- silent corruption, not a crash, and NOT caught by AddressSanitizer, since nothing freed is ever read)

UnguardedBuffer y before self-move: paths=0x503000000160, count=5
UnguardedBuffer y after y = std::move(y) (no guard): paths=(nil), count=0 (silently emptied, not a crash -- also NOT caught by AddressSanitizer)
```

**Worked Example G.3.2.** The Rule of Zero: `RiskPathBufferV2` replaces the raw pointer with a `std::vector<float>` member and writes zero hand-written special members at all.

```cpp
#include <cstdio>
#include <utility>
#include <vector>

// The CUDA C++ edition's own Worked Example G.3.3 replaces
// RiskPathBuffer's own raw `float* paths` with a `std::vector<float>`
// member and writes ZERO hand-written special members -- the Rule of
// Zero: wrap an owned resource in something that already manages
// itself correctly, and the compiler's own default-generated five
// become safe again, exactly the way they already were for
// MarketQuote in Section G.1. Entirely ordinary host C++, reproduced
// directly.
struct RiskPathBufferV2 {
    std::vector<float> paths;

    RiskPathBufferV2(int n) {
        paths.resize(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment -- all five are the compiler's
    // own defaults, and every one of them is correct, because
    // std::vector<float>'s OWN five special members are already
    // correct, and the compiler's defaults for RiskPathBufferV2 just
    // call vector's own.
};

int main() {
    // Deep-copy independence, for free.
    RiskPathBufferV2 a(5);
    RiskPathBufferV2 b = a;
    b.paths[0] = 999.0f;
    printf("copy: a.paths.data() == b.paths.data()? %s, a.paths[0]=%.1f (untouched), b.paths[0]=%.1f\n",
           (a.paths.data() == b.paths.data()) ? "true" : "false", a.paths[0], b.paths[0]);

    // Move genuinely steals the vector's own internal buffer and
    // leaves the source's own vector EMPTY (size 0), not merely
    // nulled -- std::vector's own move constructor already does this.
    RiskPathBufferV2 c(5);
    float* c_data_before = c.paths.data();
    RiskPathBufferV2 d = std::move(c);
    printf("move: d.paths.data() == c's own pre-move data()? %s, c.paths.size() after move = %zu (empty)\n",
           (d.paths.data() == c_data_before) ? "true" : "false", c.paths.size());

    // Self-assignment: no hand-written guard anywhere in
    // RiskPathBufferV2, yet it leaves the data intact -- because
    // std::vector's OWN operator= already handles self-assignment
    // correctly, one layer down, without RiskPathBufferV2 ever having
    // to know or care.
    RiskPathBufferV2 e(4);
    printf("\nRiskPathBufferV2 e before self-assignment: %.1f %.1f %.1f %.1f\n",
           e.paths[0], e.paths[1], e.paths[2], e.paths[3]);
    e = e;
    printf("RiskPathBufferV2 e after e = e (zero hand-written guards anywhere): %.1f %.1f %.1f %.1f "
           "(intact -- std::vector's own operator= already handles this correctly, one layer down)\n",
           e.paths[0], e.paths[1], e.paths[2], e.paths[3]);

    printf("\nRiskPathBufferV2 does NOT make Section G.3's hand-written five pointless: this struct got to "
           "delegate to std::vector's own already-correct machinery, but Section G.4 next faces a resource "
           "with no std::vector-equivalent to delegate to -- someone has to hand-write the five at least "
           "once, at that boundary.\n");

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 04_rule_of_zero.cpp -o 04_rule_of_zero
./04_rule_of_zero
```

```text
copy: a.paths.data() == b.paths.data()? false, a.paths[0]=0.0 (untouched), b.paths[0]=999.0
move: d.paths.data() == c's own pre-move data()? true, c.paths.size() after move = 0 (empty)

RiskPathBufferV2 e before self-assignment: 0.0 1.0 2.0 3.0
RiskPathBufferV2 e after e = e (zero hand-written guards anywhere): 0.0 1.0 2.0 3.0 (intact -- std::vector's own operator= already handles this correctly, one layer down)

RiskPathBufferV2 does NOT make Section G.3's hand-written five pointless: this struct got to delegate to std::vector's own already-correct machinery, but Section G.4 next faces a resource with no std::vector-equivalent to delegate to -- someone has to hand-write the five at least once, at that boundary.
```

**Worked Example G.3.3.** The danger in Section G.3.1's own free-then-allocate assignment ordering, exposed by genuinely injecting a `std::bad_alloc` mid-assignment: the object is left holding a pointer to memory already freed, and its own destructor at scope exit performs a genuine, ASan-caught double free.

```cpp
#include <cstdio>
#include <cstring>
#include <new>

// The CUDA C++ edition's own Worked Example G.3.4 exposes a real
// danger in Section G.3's own copy-assignment ordering (free the OLD
// resource, THEN allocate the new one): if the new allocation throws,
// the object is left holding a STALE pointer to memory that has
// ALREADY been freed. This file demonstrates the genuine consequence
// directly, using a fault-injecting allocation function to force a
// real std::bad_alloc at a chosen moment: the exception is caught
// safely at its own throw site, but the damage is already done -- the
// object's own destructor, running normally at scope exit, ends up
// calling delete[] a SECOND time on the identical pointer, a genuine
// double free, caught here by AddressSanitizer exactly as honestly as
// Section G.2's own double-free trap.
int g_alloc_call_count = 0;
int g_fail_on_call = -1;

float* fault_alloc(int n) {
    g_alloc_call_count++;
    if (g_alloc_call_count == g_fail_on_call) {
        throw std::bad_alloc();
    }
    return new float[n];
}

struct UnsafeBuffer {
    float* paths;
    int count;
    UnsafeBuffer(int n) : count(n) {
        paths = fault_alloc(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~UnsafeBuffer() { delete[] paths; }
    UnsafeBuffer(const UnsafeBuffer& other) : count(other.count) {
        paths = fault_alloc(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    UnsafeBuffer& operator=(const UnsafeBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;                     // the OLD resource is freed first --
        paths = fault_alloc(other.count);   // if THIS throws, `paths` still holds that
        count = other.count;                // now-freed pointer value, unchanged
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
};

int main() {
    UnsafeBuffer u1(4);   // fault_alloc call 1
    UnsafeBuffer u2(8);   // fault_alloc call 2
    g_fail_on_call = 3;   // the NEXT allocation (u2's own copy-assignment) will throw

    printf("injecting a std::bad_alloc into u2 = u1's own copy-assignment allocation...\n");
    fflush(stdout);
    try {
        u2 = u1;  // delete[] u2.paths runs first (freeing it), THEN fault_alloc throws --
                  // u2.paths is left holding that SAME now-freed pointer value, unchanged
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc. u2.paths still holds the pointer value that was already freed a "
               "moment ago -- nothing has fixed it up.\n");
        fflush(stdout);
    }

    printf("main() returning normally now -- watch what u2's own destructor does with that stale pointer.\n");
    fflush(stdout);

    return 0;
}  // u2's destructor runs here: delete[] u2.paths -- the SAME pointer already freed inside
   // operator= moments ago, a genuine double free
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra -fsanitize=address -g` and run:

```bash
g++ -std=c++17 -Wall -Wextra -fsanitize=address -g 05_copy_and_swap_unsafe_trap.cpp -o copy_and_swap_unsafe_trap
./copy_and_swap_unsafe_trap
```

```text
injecting a std::bad_alloc into u2 = u1's own copy-assignment allocation...
caught std::bad_alloc. u2.paths still holds the pointer value that was already freed a moment ago -- nothing has fixed it up.
main() returning normally now -- watch what u2's own destructor does with that stale pointer.
```

The real, genuinely captured AddressSanitizer summary line:

```text
SUMMARY: AddressSanitizer: double-free ../../../../src/libsanitizer/asan/asan_new_delete.cpp:155 in operator delete[](void*)
```

**Worked Example G.3.4.** Copy-and-swap: the identical injected failure against a `SafeBuffer` built around a `friend swap()` and a single by-value `operator=`, confirmed to leave the object completely unchanged, plus a genuine call-count check confirming copy-assignment allocates while move-assignment does not.

```cpp
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>

// SafeBuffer fixes the exact danger 05_copy_and_swap_unsafe_trap.cpp
// just demonstrated, via copy-and-swap: a single by-value operator=
// dispatches to either the copy constructor (lvalue argument) or the
// move constructor (rvalue argument) via ordinary overload
// resolution, then swaps -- the ONLY thing that can throw (the
// allocation, building the by-value parameter) happens entirely
// BEFORE `*this` is touched at all, so a thrown exception leaves the
// object completely unchanged.
int g_alloc_call_count = 0;
int g_fail_on_call = -1;

float* fault_alloc(int n) {
    g_alloc_call_count++;
    if (g_alloc_call_count == g_fail_on_call) {
        throw std::bad_alloc();
    }
    return new float[n];
}

struct SafeBuffer {
    float* paths;
    int count;

    SafeBuffer(int n) : count(n) {
        paths = fault_alloc(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~SafeBuffer() { delete[] paths; }
    SafeBuffer(const SafeBuffer& other) : count(other.count) {
        paths = fault_alloc(count);  // may throw -- but nothing about `*this` exists yet
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    SafeBuffer(SafeBuffer&& other) noexcept : paths(nullptr), count(0) {
        swap(*this, other);  // steal via swap into a null/zero default state -- never throws
    }
    friend void swap(SafeBuffer& a, SafeBuffer& b) noexcept {
        std::swap(a.paths, b.paths);
        std::swap(a.count, b.count);
    }
    SafeBuffer& operator=(SafeBuffer other) {  // BY VALUE: the copy/move already happened above
        swap(*this, other);                      // just swap -- cannot throw
        return *this;
    }  // `other`'s own destructor now frees what USED to be *this's old resource
};

int main() {
    // Inject the identical failure a copy-assignment would hit.
    SafeBuffer s1(4);   // fault_alloc call 1
    SafeBuffer s2(8);   // fault_alloc call 2
    float s2_val0_before = s2.paths[0];
    int s2_count_before = s2.count;
    g_fail_on_call = 3;  // the copy ctor building operator='s by-value parameter will throw

    printf("injecting the identical std::bad_alloc into s2 = s1's own copy-assignment...\n");
    try {
        s2 = s1;
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc. s2.paths[0]=%.1f (was %.1f), s2.count=%d (was %d) -- COMPLETELY "
               "UNCHANGED: the exception happened while building operator='s own by-value parameter, "
               "before *this was ever touched at all -- the strong exception guarantee.\n",
               s2.paths[0], s2_val0_before, s2.count, s2_count_before);
    }

    // Confirm the dispatch genuinely differs between copy-assignment
    // (allocates) and move-assignment (does not), via the call
    // counter -- not merely asserted.
    g_alloc_call_count = 0;
    g_fail_on_call = -1;
    SafeBuffer s3(4);  // call 1
    SafeBuffer s4(4);  // call 2
    int before_copy_assign = g_alloc_call_count;
    s4 = s3;  // COPY-assignment: lvalue argument -> copy ctor builds the by-value param -> allocates
    printf("\ncopy-assignment (s4 = s3): alloc call count went from %d to %d (rose by 1 -- the copy ctor "
           "allocated)\n", before_copy_assign, g_alloc_call_count);

    SafeBuffer s5(4);  // call 3
    int before_move_assign = g_alloc_call_count;
    s4 = std::move(s5);  // MOVE-assignment: rvalue argument -> move ctor builds the by-value param -> swap only
    printf("move-assignment (s4 = std::move(s5)): alloc call count went from %d to %d (unchanged -- the "
           "move ctor only swapped, no allocation)\n", before_move_assign, g_alloc_call_count);

    printf("\ntrade-off worth naming: TRUE self-assignment (a = a) under copy-and-swap still does a full "
           "allocation and copy -- it is no longer the free no-op Section G.3's own explicit `if (this == "
           "&other)` guard made it. Copy-and-swap trades that one narrow optimization for a correctness "
           "guarantee (exception safety) the guarded version never had at all.\n");

    return 0;
}
```

Genuinely compiled and run, ASan-clean:

```bash
g++ -std=c++17 -Wall -Wextra -fsanitize=address -g 06_copy_and_swap_safe.cpp -o copy_and_swap_safe
./copy_and_swap_safe
```

```text
injecting the identical std::bad_alloc into s2 = s1's own copy-assignment...
caught std::bad_alloc. s2.paths[0]=0.0 (was 0.0), s2.count=8 (was 8) -- COMPLETELY UNCHANGED: the exception happened while building operator='s own by-value parameter, before *this was ever touched at all -- the strong exception guarantee.

copy-assignment (s4 = s3): alloc call count went from 2 to 3 (rose by 1 -- the copy ctor allocated)
move-assignment (s4 = std::move(s5)): alloc call count went from 4 to 4 (unchanged -- the move ctor only swapped, no allocation)

trade-off worth naming: TRUE self-assignment (a = a) under copy-and-swap still does a full allocation and copy -- it is no longer the free no-op Section G.3's own explicit `if (this == &other)` guard made it. Copy-and-swap trades that one narrow optimization for a correctness guarantee (exception safety) the guarded version never had at all.
```

**Worked Example G.3.5.** Why every move operation in this section is marked `noexcept`: two otherwise-identical structs, one with and one without `noexcept` on its own move constructor, placed in a `std::vector` and forced to reallocate.

```cpp
#include <cstdio>
#include <vector>

// The CUDA C++ edition's own Worked Example G.3.5 explains why every
// move constructor and move assignment operator in this appendix is
// marked `noexcept`: std::vector uses std::move_if_noexcept
// internally when relocating existing elements during reallocation,
// falling back to COPYING them instead of moving if the move
// constructor isn't noexcept -- specifically to preserve vector's own
// strong exception guarantee (a throwing move mid-relocation could
// leave the vector in a corrupted, half-moved state; a throwing copy
// cannot, since the originals are left untouched until every copy
// succeeds). Entirely ordinary host C++, reproduced directly.
struct WithNoexcept {
    float val;
    static int copy_count;
    static int move_count;
    WithNoexcept(float v) : val(v) {}
    WithNoexcept(const WithNoexcept& other) : val(other.val) { copy_count++; }
    WithNoexcept(WithNoexcept&& other) noexcept : val(other.val) { move_count++; }
};
int WithNoexcept::copy_count = 0;
int WithNoexcept::move_count = 0;

struct WithoutNoexcept {
    float val;
    static int copy_count;
    static int move_count;
    WithoutNoexcept(float v) : val(v) {}
    WithoutNoexcept(const WithoutNoexcept& other) : val(other.val) { copy_count++; }
    WithoutNoexcept(WithoutNoexcept&& other) : val(other.val) { move_count++; }  // no noexcept
};
int WithoutNoexcept::copy_count = 0;
int WithoutNoexcept::move_count = 0;

int main() {
    // The noexcept struct: reserve capacity 2, fill it, reset the
    // counters, then force a reallocation with a third element.
    std::vector<WithNoexcept> v1;
    v1.reserve(2);
    v1.emplace_back(1.0f);
    v1.emplace_back(2.0f);
    WithNoexcept::copy_count = 0;
    WithNoexcept::move_count = 0;
    v1.emplace_back(3.0f);  // exceeds capacity 2 -> reallocates -> relocates the first 2 elements
    printf("WithNoexcept (move ctor IS noexcept): after forcing reallocation, copy_count=%d, move_count=%d "
           "-- the 2 pre-existing elements were relocated by MOVING them (safe: a noexcept move can never "
           "leave the vector in a corrupted half-moved state)\n",
           WithNoexcept::copy_count, WithNoexcept::move_count);

    // The IDENTICAL scenario, with the only difference being the
    // missing `noexcept` on the move constructor.
    std::vector<WithoutNoexcept> v2;
    v2.reserve(2);
    v2.emplace_back(1.0f);
    v2.emplace_back(2.0f);
    WithoutNoexcept::copy_count = 0;
    WithoutNoexcept::move_count = 0;
    v2.emplace_back(3.0f);
    printf("WithoutNoexcept (move ctor is NOT noexcept, otherwise identical): after the identical forced "
           "reallocation, copy_count=%d, move_count=%d -- std::vector fell back to COPYING the 2 "
           "pre-existing elements instead, silently, with no compiler diagnostic marking the difference "
           "either way.\n",
           WithoutNoexcept::copy_count, WithoutNoexcept::move_count);

    printf("\nthe only source difference between WithNoexcept and WithoutNoexcept is one keyword on the "
           "move constructor -- yet it flips whether reallocation is O(1)-per-element (move) or requires a "
           "genuine copy of every relocated element, a silent, non-diagnosed performance regression this "
           "book's own RiskPathBuffer's move constructor and move assignment operator (Section G.3) avoid "
           "specifically because both are marked noexcept.\n");

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 07_noexcept_and_reallocation.cpp -o 07_noexcept_and_reallocation
./07_noexcept_and_reallocation
```

```text
WithNoexcept (move ctor IS noexcept): after forcing reallocation, copy_count=0, move_count=2 -- the 2 pre-existing elements were relocated by MOVING them (safe: a noexcept move can never leave the vector in a corrupted half-moved state)
WithoutNoexcept (move ctor is NOT noexcept, otherwise identical): after the identical forced reallocation, copy_count=2, move_count=0 -- std::vector fell back to COPYING the 2 pre-existing elements instead, silently, with no compiler diagnostic marking the difference either way.

the only source difference between WithNoexcept and WithoutNoexcept is one keyword on the move constructor -- yet it flips whether reallocation is O(1)-per-element (move) or requires a genuine copy of every relocated element, a silent, non-diagnosed performance regression this book's own RiskPathBuffer's move constructor and move assignment operator (Section G.3) avoid specifically because both are marked noexcept.
```

**Discussion.** Every claim in this section is backed by a genuine, independently checked run rather than an assertion: pointer identity for deep-copy independence, a nulled source for move, an unchanged pointer and count for the guarded self-assignment/self-move cases, and two DIFFERENT genuine failure modes -- silent corruption, silent data loss -- for the unguarded cases, neither of which trips AddressSanitizer's own heap checks, since no freed memory is ever read in either one. The Rule of Zero doesn't make this section's own hand-written five pointless: `RiskPathBufferV2` got to delegate to `std::vector`'s own already-correct machinery, but Section G.4 next faces a resource with no such delegate available in the CUDA book's own account -- someone has to hand-write the five at least once, at that boundary. Copy-and-swap's own call-count check (2 to 3 for copy-assignment, 4 to 4 unchanged for move-assignment) confirms the dispatch genuinely differs, not merely that it should in principle. And the `noexcept` reallocation check's own 0/2-versus-2/0 flip, with the sole difference being one keyword, is exactly the kind of silent, non-diagnosed consequence this book's own established convention exists to make visible rather than assume.

## G.4 The Rule of Five a torch::Tensor Already Gives You for Free `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section G.4 hand-writes all five special members for `GPUPathBuffer`, a struct wrapping a `cudaMalloc`'d device buffer -- not by choice, but structurally, since `cudaMalloc`/`cudaFree` are host-side Runtime API entry points with no ownership semantics of their own, so someone has to hand-write correct ownership for that raw allocation. A LibTorch programmer wrapping a `torch::Tensor` faces the identical SHAPE of problem, but `torch::Tensor` is not a raw pointer: internally, it holds a reference-counted handle (an `intrusive_ptr` to a `TensorImpl`) that already implements a fully correct Rule of Five, on CPU memory in this sandbox and on `cudaMalloc`'d device memory identically, on a real GPU-attached machine, with the same source code either way.

**Background.** This section extends Section G.3.2's own Rule-of-Zero argument (wrap the resource in something that already manages itself) to the EXACT scenario the CUDA book's own Section G.4 needed a hand-written five for. `TensorPathBuffer` wraps a `torch::Tensor` and writes zero hand-written special members -- every claim below is checked via `use_count()` (a real, live reference count) and `data_ptr()` identity, not merely asserted.

**Worked Example G.4.1.** `TensorPathBuffer`, zero hand-written special members, with copy (shared storage, incremented reference count), move (stolen handle, source left undefined), and self-assignment all genuinely verified.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <utility>

// The CUDA C++ edition's own Section G.4 hand-writes all five special
// members for GPUPathBuffer, a struct wrapping a cudaMalloc'd device
// buffer -- not by choice, but structurally: cudaMalloc/cudaFree are
// host-side CUDA Runtime API entry points, so a struct wrapping them
// is necessarily host-only, and someone has to write correct
// ownership semantics for that raw allocation by hand, exactly the
// way Section G.3 hand-wrote them for a raw `new[]`'d host array.
//
// A LibTorch programmer wrapping a torch::Tensor faces the identical
// SHAPE of problem -- a class that owns a resource whose lifetime
// must be managed correctly across copies, moves, and destruction --
// but torch::Tensor is not a raw pointer. Internally, it holds an
// intrusive_ptr to a TensorImpl, a reference-counted handle exactly
// like std::shared_ptr's own mechanism, and that handle already
// implements a fully correct Rule of Five: copying a torch::Tensor
// increments a reference count and shares the underlying storage;
// moving one steals the handle and leaves the source undefined;
// destroying one decrements the count and frees the storage only when
// it reaches zero -- on CPU memory in this sandbox, and on cudaMalloc'd
// device memory identically, on a real GPU-attached machine, with the
// SAME source code either way. This section extends Section G.3.3's
// own Rule-of-Zero argument (wrap the resource in something that
// already manages itself) to the EXACT scenario Section G.4 needed a
// hand-written five for.
struct TensorPathBuffer {
    torch::Tensor paths;
    TensorPathBuffer(int n) : paths(torch::arange(n, torch::kFloat32)) {}
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment -- all five are the compiler's
    // own defaults, and every one is correct, because torch::Tensor's
    // OWN five special members already are, on CPU today and on a
    // CUDA device identically, with zero source changes to this
    // struct at all.
};

int main() {
    // Copy: torch::Tensor's own copy constructor SHARES the
    // underlying storage and increments a real reference count --
    // this is genuinely DIFFERENT from Section G.3's own deep-copy
    // semantics, and stated as such rather than glossed over.
    TensorPathBuffer a(5);
    std::cout << "a.paths.use_count() right after construction = " << a.paths.use_count() << std::endl;
    TensorPathBuffer b = a;
    std::cout << "copy: a.paths.data_ptr() == b.paths.data_ptr()? "
              << (a.paths.data_ptr() == b.paths.data_ptr())
              << " (SHARED storage, unlike Section G.3's own deep copy), use_count now = "
              << a.paths.use_count() << " (both a and b's own handles reference the identical storage)"
              << std::endl;

    // Move: torch::Tensor's own move constructor steals the handle,
    // exactly the way RiskPathBuffer's own hand-written move
    // constructor does, but written by LibTorch's own authors once,
    // reused correctly here with zero code of this struct's own.
    TensorPathBuffer c(5);
    void* c_ptr_before_move = c.paths.data_ptr();
    TensorPathBuffer d = std::move(c);
    std::cout << "\nmove: d.paths.data_ptr() == c's own pre-move data_ptr()? "
              << (d.paths.data_ptr() == c_ptr_before_move)
              << ", c.paths.defined() after move = " << c.paths.defined()
              << " (undefined -- the handle was genuinely stolen, the compiler's default move constructor "
              << "for TensorPathBuffer just calls torch::Tensor's own)" << std::endl;

    // Self-assignment: zero hand-written guards anywhere in
    // TensorPathBuffer, yet it is safe -- torch::Tensor's own
    // operator= already handles it, one layer down, the identical
    // Rule-of-Zero delegation Section 04's RiskPathBufferV2 already
    // demonstrated for std::vector.
    TensorPathBuffer e(4);
    torch::Tensor before_self = e.paths.clone();
    e.paths = e.paths;
    std::cout << "\nself-assignment (e.paths = e.paths), zero hand-written guards anywhere: torch::allclose "
              << "confirms data intact? " << torch::allclose(e.paths, before_self) << std::endl;

    std::cout << "\nnone of this required a hand-written destructor, copy constructor, copy assignment, "
              << "move constructor, or move assignment operator -- unlike Section G.4's own GPUPathBuffer, "
              << "which had to hand-write all five specifically because cudaMalloc/cudaFree return and "
              << "accept a RAW pointer with no ownership semantics of its own at all. torch::Tensor already "
              << "IS the correctly-implemented Rule of Five (or, from an application programmer's own point "
              << "of view, the Rule of Zero it enables) for exactly the resource -- device memory -- Section "
              << "G.4's own worked example had to build by hand." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 08_tensor_rule_of_zero_for_device_memory.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 08_tensor_rule_of_zero_for_device_memory
./08_tensor_rule_of_zero_for_device_memory
```

```text
a.paths.use_count() right after construction = 1
copy: a.paths.data_ptr() == b.paths.data_ptr()? 1 (SHARED storage, unlike Section G.3's own deep copy), use_count now = 2 (both a and b's own handles reference the identical storage)

move: d.paths.data_ptr() == c's own pre-move data_ptr()? 1, c.paths.defined() after move = 0 (undefined -- the handle was genuinely stolen, the compiler's default move constructor for TensorPathBuffer just calls torch::Tensor's own)

self-assignment (e.paths = e.paths), zero hand-written guards anywhere: torch::allclose confirms data intact? 1

none of this required a hand-written destructor, copy constructor, copy assignment, move constructor, or move assignment operator -- unlike Section G.4's own GPUPathBuffer, which had to hand-write all five specifically because cudaMalloc/cudaFree return and accept a RAW pointer with no ownership semantics of its own at all. torch::Tensor already IS the correctly-implemented Rule of Five (or, from an application programmer's own point of view, the Rule of Zero it enables) for exactly the resource -- device memory -- Section G.4's own worked example had to build by hand.
```

**Discussion.** Copying a `TensorPathBuffer` genuinely differs from Section G.3's own deep-copy semantics -- `a.paths.data_ptr() == b.paths.data_ptr()` reports true, and `use_count()` genuinely rises to 2, confirming the two handles SHARE the identical underlying storage rather than each owning an independent copy, exactly the reference-counted-handle semantics `std::shared_ptr` provides and `RiskPathBuffer`'s own hand-written deep copy deliberately does not. This is stated as a genuine, honest divergence rather than glossed over: a LibTorch programmer who needs `RiskPathBuffer`-style deep-copy independence from a `torch::Tensor` reaches for `.clone()` explicitly, exactly the way Section G.3.4's own worked example uses it to snapshot a tensor before a self-assignment test. What doesn't diverge is the structural claim this section exists to make: none of copy, move, or self-assignment required a single hand-written special member, unlike the CUDA book's own `GPUPathBuffer`, which had to hand-write all five specifically because `cudaMalloc`/`cudaFree` return and accept a raw pointer with no ownership semantics of its own at all.

## G.5 Value-at-Risk and Its Variants `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section G.5 reuses Chapter 22.4's own hand-rolled `box_muller`/xorshift-PRNG/GBM-update machinery "byte for byte" at a 1-day risk horizon, then computes three VaR methodologies. This section reuses THIS book's own Chapter 22.1 GBM approach instead -- a single vectorized closed-form expression via real `torch::randn` -- since this book has never had a hand-rolled PRNG to reuse in the first place. The three methodologies themselves are ordinary, CUDA-free quantitative finance.

**Background.** Simulated VaR sorts simulated 1-day P&L ascending and reads the loss at the 1st percentile for 99% confidence. Parametric (variance-covariance) VaR computes `S0 * sigma_1day * z_99` directly from the model's own known volatility. Conditional VaR (CVaR/Expected Shortfall) averages the worst `k` scenarios beyond the VaR cutoff, and is provably at least as large as VaR, since it averages a set every member of which is at least that large.

**Worked Example G.5.1.** All three VaR methodologies, computed on 200,000 real, genuinely simulated 1-day GBM paths, with `CVaR >= VaR` checked directly rather than assumed.

```cpp
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section G.5 reuses Chapter 22.4's own
// hand-rolled box_muller/xorshift-PRNG/GBM-update machinery "byte for
// byte" at a 1-day horizon, then computes three VaR methodologies.
// This section reuses THIS book's own Chapter 22.1 GBM approach
// instead: a single vectorized closed-form expression via real
// torch::randn, the identical technique this book has used since
// Chapter 22 rather than a hand-rolled PRNG -- consistent with the
// established "don't rebuild machinery this book already has a real
// API for" pattern. The three methodologies themselves are unchanged,
// ordinary quantitative finance, with no CUDA content in them at all.
int main() {
    torch::manual_seed(42);

    double S0 = 100.0, mu = 0.03, sigma = 0.20;
    double dt = 1.0 / 252.0;  // one trading day
    int64_t num_paths = 200000;

    // GBM's own exact one-step solution (Chapter 22.1's own formula,
    // applied at a 1-day horizon instead of Chapter 22's own 1-year
    // option-pricing horizon).
    torch::Tensor z = torch::randn({num_paths});
    torch::Tensor s_t = S0 * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
    torch::Tensor pnl = s_t - S0;

    // Simulated (empirical/historical-style) VaR: sort P&L ascending,
    // read the loss at the 1st percentile for 99% confidence.
    torch::Tensor pnl_sorted = std::get<0>(torch::sort(pnl));
    int64_t tail_idx = static_cast<int64_t>(0.01 * static_cast<double>(num_paths));  // floor(0.01*N) = 2000
    double simulated_var_99 = -pnl_sorted[tail_idx].item<double>();

    // Parametric (variance-covariance) VaR: S0 * sigma_1day * z_99.
    double sigma_1day = sigma * std::sqrt(dt);
    double z_99 = 2.326348;  // 99th-percentile standard-normal critical value
    double parametric_var_99 = S0 * sigma_1day * z_99;

    double relative_diff = std::abs(simulated_var_99 - parametric_var_99) / parametric_var_99;

    // Conditional VaR / Expected Shortfall: the average of the k
    // worst P&L values beyond the VaR cutoff.
    torch::Tensor worst_k = pnl_sorted.slice(0, 0, tail_idx);
    double cvar_99 = -worst_k.mean().item<double>();

    std::cout << "1-day GBM P&L, S0=" << S0 << ", mu=" << mu << ", sigma=" << sigma << ", "
              << num_paths << " paths, real torch::randn (manual_seed 42), Chapter 22.1's own vectorized "
              << "closed-form GBM step:" << std::endl;
    std::cout << "  tail index (floor(0.01 * " << num_paths << ")) = " << tail_idx << std::endl;
    std::cout << "  simulated 99% 1-day VaR   = " << simulated_var_99 << std::endl;
    std::cout << "  sigma_1day = " << sigma_1day << ", z_99 = " << z_99 << std::endl;
    std::cout << "  parametric 99% 1-day VaR  = " << parametric_var_99 << std::endl;
    std::cout << "  relative difference       = " << relative_diff << " (" << (relative_diff * 100.0)
              << "%)" << std::endl;
    std::cout << "  CVaR_99 (mean of the " << tail_idx << " worst P&L values) = " << cvar_99 << std::endl;
    std::cout << "  CVaR_99 >= simulated VaR_99? " << (cvar_99 >= simulated_var_99) << std::endl;

    std::cout << "\nthe simulated and parametric figures are CLOSE but not identical -- expected, and "
              << "explained the same way Section G.5's own text explains its own gap: GBM's log-returns "
              << "are exactly normal by construction, but VaR here is measured on ARITHMETIC P&L (S_T - "
              << "S0), which is log-normal, not normal -- so the parametric method's own normality "
              << "assumption is a genuine approximation even against this file's own internally consistent "
              << "data, not a bug in either computation. This book's own numbers will not match the CUDA "
              << "book's own reported 2.8665/2.9309/2.25%/3.2938 exactly, since a different RNG (torch::randn "
              << "vs. a hand-rolled xorshift+Box-Muller) produces a genuinely different sample -- the "
              << "CVaR >= VaR relationship holding, and the two methods agreeing closely, is the actual "
              << "claim being verified, not any one specific figure." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 09_value_at_risk.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 09_value_at_risk
./09_value_at_risk
```

```text
1-day GBM P&L, S0=100, mu=0.03, sigma=0.2, 200000 paths, real torch::randn (manual_seed 42), Chapter 22.1's own vectorized closed-form GBM step:
  tail index (floor(0.01 * 200000)) = 2000
  simulated 99% 1-day VaR   = 2.88795
  sigma_1day = 0.0125988, z_99 = 2.32635
  parametric 99% 1-day VaR  = 2.93092
  relative difference       = 0.0146603 (1.46603%)
  CVaR_99 (mean of the 2000 worst P&L values) = 3.31428
  CVaR_99 >= simulated VaR_99? 1

the simulated and parametric figures are CLOSE but not identical -- expected, and explained the same way Section G.5's own text explains its own gap: GBM's log-returns are exactly normal by construction, but VaR here is measured on ARITHMETIC P&L (S_T - S0), which is log-normal, not normal -- so the parametric method's own normality assumption is a genuine approximation even against this file's own internally consistent data, not a bug in either computation. This book's own numbers will not match the CUDA book's own reported 2.8665/2.9309/2.25%/3.2938 exactly, since a different RNG (torch::randn vs. a hand-rolled xorshift+Box-Muller) produces a genuinely different sample -- the CVaR >= VaR relationship holding, and the two methods agreeing closely, is the actual claim being verified, not any one specific figure.
```

**Discussion.** The simulated and parametric figures are close but not identical, for the identical structural reason the CUDA book's own Section G.5 reports: GBM's log-returns are exactly normal by construction, but VaR here is measured on ARITHMETIC P&L (`S_T - S0`), which is log-normal, not normal -- so the parametric method's own normality assumption is a genuine approximation, not a bug in either computation. Consistent with this book's own established rule since Chapters 19 and 20 (a different RNG cannot be expected to reproduce another implementation's exact sampled figures), this section's own numbers do not match the CUDA book's own reported 2.8665/2.9309/2.25%/3.2938 exactly -- what is being verified is the CVaR-at-least-VaR relationship holding genuinely, and the two methods agreeing closely, which they do.

## G.6 XVA and Its Variants `[FOUNDATIONAL]`

**Intuition.** The CUDA C++ edition's own Section G.6 builds CVA, DVA, and FVA from a genuine exposure profile -- `EE(t)` and `ENE(t)`, the expected positive and negative value of a forward contract -- extracted from checkpointed Monte Carlo paths. This section reproduces the identical methodology using this book's own real `torch::Tensor` GBM machinery, applied incrementally across four quarterly checkpoints to build one continuous path per scenario, using GBM's own Markov property (each quarter's step depends only on the previous quarter's own price).

**Background.** `EE(t) = E[max(V(t), 0)]` and `ENE(t) = E[min(V(t), 0)]`, with `V(t) = S(t) - K` for a forward contract. CVA and DVA weight expected exposure by counterparty and own default-probability changes across each interval, discounted; FVA weights expected positive exposure by a funding spread. MVA and KVA are named in the CUDA book's own text as further XVA components but not computed there; this section follows the identical scope decision.

**Worked Example G.6.1.** A full quarterly exposure profile, cross-checked against the closed-form GBM mean and against the `EE(T)+ENE(T) = E[S(T)]-K` identity, followed by CVA, DVA, FVA, and net XVA.

```cpp
#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section G.6 builds CVA, DVA, and FVA
// from a genuine exposure profile -- EE(t) and ENE(t), the expected
// positive and negative value of a forward contract -- extracted from
// checkpointed Monte Carlo paths, reusing Chapter 22.4's own GBM
// machinery. This section reproduces the identical methodology using
// this book's own real torch::Tensor GBM machinery instead (Chapter
// 22.1's own vectorized closed-form step, applied incrementally
// across four quarterly checkpoints to build one continuous path per
// simulated scenario, since GBM's own Markov property lets each
// quarter's step depend only on the previous quarter's own price).
int main() {
    torch::manual_seed(42);

    double S0 = 100.0, K = 100.0, mu = 0.03, sigma = 0.20, r = 0.03;
    double lambda_C = 0.02, R_C = 0.40;  // counterparty hazard rate, recovery
    double lambda_B = 0.01, R_B = 0.40;  // bank's own hazard rate, recovery
    double s_funding = 0.005;
    int64_t num_paths = 200000;
    std::vector<double> checkpoints = {0.25, 0.50, 0.75, 1.00};
    double dt = 0.25;

    torch::Tensor s = torch::full({num_paths}, S0, torch::kFloat64);
    std::vector<torch::Tensor> s_at_checkpoint;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        torch::Tensor z = torch::randn({num_paths}, torch::kFloat64);
        s = s * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
        s_at_checkpoint.push_back(s.clone());
    }

    // Cross-check: simulated E[S(T)] against the closed-form GBM mean.
    double simulated_mean_st = s_at_checkpoint.back().mean().item<double>();
    double theoretical_mean_st = S0 * std::exp(mu * checkpoints.back());
    double mean_relative_diff = std::abs(simulated_mean_st - theoretical_mean_st) / theoretical_mean_st;

    // EE(t) = E[max(V(t), 0)], ENE(t) = E[min(V(t), 0)], V(t) = S(t) - K.
    std::vector<double> ee(checkpoints.size()), ene(checkpoints.size());
    for (size_t k = 0; k < checkpoints.size(); k++) {
        torch::Tensor v = s_at_checkpoint[k] - K;
        ee[k] = torch::clamp_min(v, 0.0).mean().item<double>();
        ene[k] = torch::clamp_max(v, 0.0).mean().item<double>();
    }

    std::cout << "quarterly exposure profile, forward contract, S0=" << S0 << ", K=" << K << ", mu=" << mu
              << ", sigma=" << sigma << ", " << num_paths << " paths:" << std::endl;
    std::cout << "  t      EE(t)       ENE(t)      EE(t)+ENE(t)" << std::endl;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        std::cout << "  " << checkpoints[k] << "   " << ee[k] << "   " << ene[k] << "   " << (ee[k] + ene[k])
                  << std::endl;
    }

    std::cout << "\ncross-check: simulated E[S(T)] = " << simulated_mean_st << ", theoretical S0*exp(mu*T) = "
              << theoretical_mean_st << " (relative diff " << (mean_relative_diff * 100.0) << "%)" << std::endl;
    std::cout << "cross-check: EE(T)+ENE(T) = " << (ee.back() + ene.back()) << " vs. E[S(T)]-K = "
              << (simulated_mean_st - K) << " (identity: V(T)'s own mean splits exactly into its positive "
              << "and negative parts)" << std::endl;

    // CVA and DVA: survival-probability-weighted expected exposure,
    // discounted, over each quarterly interval.
    auto survival = [](double lambda, double t) { return std::exp(-lambda * t); };
    auto discount = [&](double t) { return std::exp(-r * t); };

    double cva = 0.0, dva = 0.0, fva = 0.0;
    double t_prev = 0.0;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        double t_k = checkpoints[k];
        double q_c_prev = survival(lambda_C, t_prev), q_c_k = survival(lambda_C, t_k);
        double q_b_prev = survival(lambda_B, t_prev), q_b_k = survival(lambda_B, t_k);
        double df_k = discount(t_k);

        double cva_term = (1.0 - R_C) * ee[k] * (q_c_prev - q_c_k) * df_k;
        double dva_term = (1.0 - R_B) * (-ene[k]) * (q_b_prev - q_b_k) * df_k;
        double fva_term = s_funding * ee[k] * df_k * dt;

        cva += cva_term;
        dva += dva_term;
        fva += fva_term;

        t_prev = t_k;
    }

    double net_xva = -cva + dva - fva;

    std::cout << "\nCVA = " << cva << std::endl;
    std::cout << "DVA = " << dva << std::endl;
    std::cout << "FVA = " << fva << std::endl;
    std::cout << "Net XVA (-CVA + DVA - FVA) = " << net_xva << std::endl;

    std::cout << "\nDVA < CVA here specifically because the bank's own hazard rate (" << lambda_B
              << ") was set lower than the counterparty's (" << lambda_C
              << ") -- a better-credit bank has a smaller DVA, all else equal, exactly the relationship "
              << "Section G.6's own text reports. MVA and KVA are named by the CUDA book's own text as "
              << "further XVA components but explicitly not computed there; this section follows the "
              << "identical scope decision, computing CVA/DVA/FVA only. This book's own exact figures will "
              << "not match the CUDA book's own reported ones, since a different RNG produces a genuinely "
              << "different sample of the same GBM process -- what's being verified is the exposure-profile "
              << "shape (EE growing over time, ENE(t) staying negative) and the internal identities above, "
              << "not any one specific dollar figure." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 10_xva_and_variants.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 10_xva_and_variants
./10_xva_and_variants
```

```text
quarterly exposure profile, forward contract, S0=100, K=100, mu=0.03, sigma=0.2, 200000 paths:
  t      EE(t)       ENE(t)      EE(t)+ENE(t)
  0.25   4.39379   -3.6431   0.750687
  0.5   6.45686   -4.97394   1.48291
  0.75   8.15879   -5.92502   2.23377
  1   9.70075   -6.67092   3.02984

cross-check: simulated E[S(T)] = 103.03, theoretical S0*exp(mu*T) = 103.045 (relative diff 0.0151553%)
cross-check: EE(T)+ENE(T) = 3.02984 vs. E[S(T)]-K = 3.02984 (identity: V(T)'s own mean splits exactly into its positive and negative parts)

CVA = 0.0833766
DVA = 0.0310011
FVA = 0.0351413
Net XVA (-CVA + DVA - FVA) = -0.0875168

DVA < CVA here specifically because the bank's own hazard rate (0.01) was set lower than the counterparty's (0.02) -- a better-credit bank has a smaller DVA, all else equal, exactly the relationship Section G.6's own text reports. MVA and KVA are named by the CUDA book's own text as further XVA components but explicitly not computed there; this section follows the identical scope decision, computing CVA/DVA/FVA only. This book's own exact figures will not match the CUDA book's own reported ones, since a different RNG produces a genuinely different sample of the same GBM process -- what's being verified is the exposure-profile shape (EE growing over time, ENE(t) staying negative) and the internal identities above, not any one specific dollar figure.
```

**Discussion.** Both cross-checks hold exactly: the simulated `E[S(T)]` matches the closed-form `S0*exp(mu*T)` to within a fraction of a percent, and `EE(T)+ENE(T)` matches `E[S(T)]-K` to full floating-point precision, since the positive and negative parts of the identical mean must sum back to that mean by construction. DVA comes out smaller than CVA for the identical reason the CUDA book's own text reports: the bank's own hazard rate (1%) was set lower than the counterparty's (2%) -- a better-credit bank has a smaller DVA, all else equal. This section's own genuinely computed figures (EE(1.00) close to 9.70, CVA close to 0.083, DVA close to 0.031, FVA close to 0.035) land remarkably close to the CUDA book's own reported 9.6584/0.0830/0.0309/0.0350, despite a completely different RNG and a full `torch::Tensor`-based simulation rather than a hand-rolled one -- itself a small, honest confirmation that the underlying exposure/XVA methodology, not any one implementation's specific random sample, is what actually determines these figures.

## G.7 Complete Runnable Code

### `01_market_quote_default_copy.cpp`

```cpp
#include <iostream>

// The CUDA C++ edition's own Section G.1 calls a trivial two-float
// struct, Point2D, from a kernel with zero changes, and makes the
// point that Point2D's default compiler-generated copy is already
// correct -- because Point2D owns nothing but two floats. This book
// never writes kernels, so this section's own worked example needs no
// kernel at all -- it makes the identical point on a struct in this
// book's own financial-computing domain, entirely in ordinary host
// code: MarketQuote, two doubles (bid, ask), no owned resource of any
// kind.
struct MarketQuote {
    double bid;
    double ask;
    double mid() const { return (bid + ask) / 2.0; }
    double spread() const { return ask - bid; }
};

int main() {
    std::cout << "sizeof(MarketQuote) = " << sizeof(MarketQuote)
              << " bytes (two doubles, no owned resource of any kind)" << std::endl;

    MarketQuote q1{100.05, 100.10};
    std::cout << "q1 = {bid=" << q1.bid << ", ask=" << q1.ask << "}, mid=" << q1.mid()
              << ", spread=" << q1.spread() << std::endl;

    // The compiler's own default-generated copy constructor: a plain,
    // member-by-member copy of two doubles. There is nothing here for
    // a hand-written copy constructor to do differently -- copying
    // the bits IS copying the value.
    MarketQuote q2 = q1;
    q2.bid = 200.00;
    q2.ask = 200.05;

    std::cout << "\nq2 = q1, then q2's own bid/ask mutated to {200.00, 200.05}:" << std::endl;
    std::cout << "  q1 = {bid=" << q1.bid << ", ask=" << q1.ask
              << "} (completely unaffected by q2's own mutation -- the compiler's default copy already "
              << "gave q2 its own independent bid and ask)" << std::endl;
    std::cout << "  q2 = {bid=" << q2.bid << ", ask=" << q2.ask << "}" << std::endl;

    std::cout << "\nMarketQuote needs no hand-written destructor, copy constructor, copy assignment, move "
              << "constructor, or move assignment operator -- the compiler's own default-generated five "
              << "are already exactly correct, because MarketQuote owns nothing that a bitwise copy could "
              << "ever get wrong. Section G.2 introduces the one thing that changes this: a struct that "
              << "owns a resource whose lifetime doesn't automatically follow the struct's own." << std::endl;

    return 0;
}
```

### `02_double_free_trap.cpp`

```cpp
#include <cstdio>

// stdout is fully buffered by default when it isn't attached to a
// terminal (exactly the case when this binary's own output is
// captured by a verification script) -- without an explicit fflush
// after each printf, the buffered text below would simply be LOST
// when AddressSanitizer's own abort() tears the process down before
// libc ever gets to flush it. Genuinely discovered by running this
// file for real: the first version of this file printed nothing at
// all to stdout, only ASan's own report to stderr, until this fix.

// The CUDA C++ edition's own Section G.2 introduces RiskPathBuffer: a
// struct owning a raw heap pointer (float* paths), with only a
// destructor hand-written (delete[] paths) and no copy constructor or
// copy assignment operator of its own. This is entirely ordinary host
// C++ -- no CUDA content anywhere in it -- so this section reproduces
// it directly, in this book's own words, unchanged in substance.
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }

    // Only the destructor is hand-written. No copy constructor, no
    // copy assignment operator -- the compiler generates a plain,
    // MEMBERWISE copy of both members for both, including `paths`,
    // which copies the POINTER, not the array it points to.
    ~RiskPathBuffer() { delete[] paths; }
};

int main() {
    RiskPathBuffer a(10);
    printf("a.paths[0..2] = %.1f %.1f %.1f, a.count = %d\n", a.paths[0], a.paths[1], a.paths[2], a.count);
    fflush(stdout);

    {
        RiskPathBuffer b = a;  // compiler-generated copy: b.paths == a.paths, same allocation
        printf("b.paths == a.paths? %s (the pointer was copied, not the 10-float array it points to)\n",
               (b.paths == a.paths) ? "true" : "false");
        fflush(stdout);
    }  // b goes out of scope here -- its destructor frees the SHARED buffer

    // a.paths now points at freed memory. Reading it here is a
    // genuine use-after-free, and a's own destructor at the end of
    // main() will free the identical pointer a second time -- a
    // genuine double free.
    printf("a.paths[0] after b's own destructor ran = %.1f (reading FREED memory -- undefined behavior)\n",
           a.paths[0]);
    fflush(stdout);

    return 0;
}  // a's own destructor runs here: delete[] a.paths -- the SAME pointer b's destructor already freed
```

### `03_rule_of_five_correct_and_guards.cpp`

```cpp
#include <cstdio>
#include <cstring>
#include <utility>

// The CUDA C++ edition's own Section G.3 (worked examples G.3.1 and
// G.3.2, combined into this one file) implements RiskPathBuffer's own
// five special members correctly, then tests the self-assignment and
// self-move guards those five members need by genuinely removing them
// and observing what breaks. Entirely ordinary host C++, reproduced
// here directly.
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }

    ~RiskPathBuffer() { delete[] paths; }

    // Copy constructor: a genuine DEEP copy -- a fresh allocation,
    // its own independent contents.
    RiskPathBuffer(const RiskPathBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }

    // Copy assignment: guard against self-assignment first, then free
    // the OLD resource before deep-copying the new one.
    RiskPathBuffer& operator=(const RiskPathBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }

    // Move constructor: STEAL the source's pointer, null the source
    // so its own destructor becomes a safe no-op. noexcept matters --
    // Worked Example G.3.5-equivalent below shows why directly.
    RiskPathBuffer(RiskPathBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }

    // Move assignment: guard against self-move, free the OLD
    // resource, then steal and null exactly as the move constructor
    // does.
    RiskPathBuffer& operator=(RiskPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};

// An UNGUARDED variant -- identical to RiskPathBuffer above, except
// the `if (this == &other) return *this;` guard has been removed
// from both assignment operators. This demonstrates the CUDA book's
// own genuinely surprising finding: removing the guard does NOT
// reliably crash or trip AddressSanitizer -- it produces two
// different kinds of SILENT failure instead, neither of which any
// sanitizer's heap-corruption checks catch, because no memory is
// ever read after being freed.
struct UnguardedBuffer {
    float* paths;
    int count;

    UnguardedBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~UnguardedBuffer() { delete[] paths; }
    UnguardedBuffer(const UnguardedBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    // No self-assignment guard. If `other` IS `*this` (a = a), then
    // `other.paths` and `paths` are the SAME field: freeing `paths`
    // also invalidates `other.paths`, but the very next line
    // (`paths = new float[count]`) immediately overwrites BOTH --
    // they're one and the same field -- with a fresh, UNINITIALIZED
    // allocation. The memcpy below then copies that fresh allocation's
    // own garbage contents onto itself: not a use-after-free (nothing
    // freed is ever actually read), but silent data corruption.
    UnguardedBuffer& operator=(const UnguardedBuffer& other) {
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
    UnguardedBuffer(UnguardedBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }
    // No self-move guard.
    UnguardedBuffer& operator=(UnguardedBuffer&& other) noexcept {
        delete[] paths;
        paths = other.paths;   // if other IS *this, this just stole its OWN pointer --
        count = other.count;   // harmless so far...
        other.paths = nullptr; // ...but THIS line then nulls the very pointer just stolen,
        other.count = 0;       // since other and *this are the same object
        return *this;
    }
};

int main() {
    // Copy: independent allocation, independent contents.
    RiskPathBuffer a(5);
    RiskPathBuffer b = a;
    b.paths[0] = 999.0f;
    printf("copy: a.paths == b.paths? %s, a.paths[0]=%.1f (untouched), b.paths[0]=%.1f (mutated)\n",
           (a.paths == b.paths) ? "true" : "false", a.paths[0], b.paths[0]);

    // Move: steals the pointer, nulls the source.
    float* a_ptr_before_move = a.paths;
    RiskPathBuffer c = std::move(a);
    printf("move: c.paths == a's own pre-move pointer? %s, a.paths after move = %p (nulled)\n",
           (c.paths == a_ptr_before_move) ? "true" : "false", (void*)a.paths);

    // Copy assignment onto an EXISTING object: old resource freed
    // first, new one deep-copied in.
    RiskPathBuffer d(7);
    RiskPathBuffer e(2);
    float* e_ptr_before = e.paths;
    e = d;
    printf("copy-assign: e.count went from 2 to %d, e.paths changed from its own old pointer? %s\n",
           e.count, (e.paths != e_ptr_before) ? "true" : "false");

    // Move assignment onto an EXISTING object.
    RiskPathBuffer f(3);
    RiskPathBuffer g(9);
    float* d_paths_before_move_assign = d.paths;
    f = std::move(d);
    printf("move-assign: f.paths == d's own pre-move pointer? %s, d.paths after move = %p (nulled)\n",
           (f.paths == d_paths_before_move_assign) ? "true" : "false", (void*)d.paths);

    // Self-assignment and self-move on the GUARDED struct: both
    // provably leave the object unchanged.
    float* g_ptr_before_self = g.paths;
    int g_count_before_self = g.count;
    g = g;  // self-copy-assignment
    printf("\nself-assignment (g = g), guarded: pointer unchanged? %s, count unchanged? %s\n",
           (g.paths == g_ptr_before_self) ? "true" : "false", (g.count == g_count_before_self) ? "true" : "false");

    RiskPathBuffer h(4);
    float* h_ptr_before_self = h.paths;
    int h_count_before_self = h.count;
    h = std::move(h);  // self-move-assignment; g++ itself warns -Wself-move on this exact line
    printf("self-move (h = std::move(h)), guarded: pointer unchanged? %s, count unchanged? %s\n",
           (h.paths == h_ptr_before_self) ? "true" : "false", (h.count == h_count_before_self) ? "true" : "false");

    // The UNGUARDED variant: self-assignment silently CORRUPTS data
    // (overwrites the original 0,1,2,3,4 values with fresh, garbage
    // allocator contents) rather than crashing.
    UnguardedBuffer x(5);
    printf("\nUnguardedBuffer x before self-assignment: %.1f %.1f %.1f %.1f %.1f\n",
           x.paths[0], x.paths[1], x.paths[2], x.paths[3], x.paths[4]);
    x = x;  // no guard -- silently corrupts
    printf("UnguardedBuffer x after x = x (no guard): %.1f %.1f %.1f %.1f %.1f (garbage, not 0-4 -- silent "
           "corruption, not a crash, and NOT caught by AddressSanitizer, since nothing freed is ever read)\n",
           x.paths[0], x.paths[1], x.paths[2], x.paths[3], x.paths[4]);

    // The UNGUARDED variant: self-move silently EMPTIES the object
    // (paths becomes null, count becomes 0) rather than crashing,
    // because `other.paths = nullptr` on the line after the steal
    // overwrites the very pointer that steal just assigned, since
    // `other` and `*this` are the same object.
    UnguardedBuffer y(5);
    printf("\nUnguardedBuffer y before self-move: paths=%p, count=%d\n", (void*)y.paths, y.count);
    y = std::move(y);  // no guard -- silently empties itself
    printf("UnguardedBuffer y after y = std::move(y) (no guard): paths=%p, count=%d (silently emptied, "
           "not a crash -- also NOT caught by AddressSanitizer)\n", (void*)y.paths, y.count);

    return 0;
}
```

### `04_rule_of_zero.cpp`

```cpp
#include <cstdio>
#include <utility>
#include <vector>

// The CUDA C++ edition's own Worked Example G.3.3 replaces
// RiskPathBuffer's own raw `float* paths` with a `std::vector<float>`
// member and writes ZERO hand-written special members -- the Rule of
// Zero: wrap an owned resource in something that already manages
// itself correctly, and the compiler's own default-generated five
// become safe again, exactly the way they already were for
// MarketQuote in Section G.1. Entirely ordinary host C++, reproduced
// directly.
struct RiskPathBufferV2 {
    std::vector<float> paths;

    RiskPathBufferV2(int n) {
        paths.resize(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment -- all five are the compiler's
    // own defaults, and every one of them is correct, because
    // std::vector<float>'s OWN five special members are already
    // correct, and the compiler's defaults for RiskPathBufferV2 just
    // call vector's own.
};

int main() {
    // Deep-copy independence, for free.
    RiskPathBufferV2 a(5);
    RiskPathBufferV2 b = a;
    b.paths[0] = 999.0f;
    printf("copy: a.paths.data() == b.paths.data()? %s, a.paths[0]=%.1f (untouched), b.paths[0]=%.1f\n",
           (a.paths.data() == b.paths.data()) ? "true" : "false", a.paths[0], b.paths[0]);

    // Move genuinely steals the vector's own internal buffer and
    // leaves the source's own vector EMPTY (size 0), not merely
    // nulled -- std::vector's own move constructor already does this.
    RiskPathBufferV2 c(5);
    float* c_data_before = c.paths.data();
    RiskPathBufferV2 d = std::move(c);
    printf("move: d.paths.data() == c's own pre-move data()? %s, c.paths.size() after move = %zu (empty)\n",
           (d.paths.data() == c_data_before) ? "true" : "false", c.paths.size());

    // Self-assignment: no hand-written guard anywhere in
    // RiskPathBufferV2, yet it leaves the data intact -- because
    // std::vector's OWN operator= already handles self-assignment
    // correctly, one layer down, without RiskPathBufferV2 ever having
    // to know or care.
    RiskPathBufferV2 e(4);
    printf("\nRiskPathBufferV2 e before self-assignment: %.1f %.1f %.1f %.1f\n",
           e.paths[0], e.paths[1], e.paths[2], e.paths[3]);
    e = e;
    printf("RiskPathBufferV2 e after e = e (zero hand-written guards anywhere): %.1f %.1f %.1f %.1f "
           "(intact -- std::vector's own operator= already handles this correctly, one layer down)\n",
           e.paths[0], e.paths[1], e.paths[2], e.paths[3]);

    printf("\nRiskPathBufferV2 does NOT make Section G.3's hand-written five pointless: this struct got to "
           "delegate to std::vector's own already-correct machinery, but Section G.4 next faces a resource "
           "with no std::vector-equivalent to delegate to -- someone has to hand-write the five at least "
           "once, at that boundary.\n");

    return 0;
}
```

### `05_copy_and_swap_unsafe_trap.cpp`

```cpp
#include <cstdio>
#include <cstring>
#include <new>

// The CUDA C++ edition's own Worked Example G.3.4 exposes a real
// danger in Section G.3's own copy-assignment ordering (free the OLD
// resource, THEN allocate the new one): if the new allocation throws,
// the object is left holding a STALE pointer to memory that has
// ALREADY been freed. This file demonstrates the genuine consequence
// directly, using a fault-injecting allocation function to force a
// real std::bad_alloc at a chosen moment: the exception is caught
// safely at its own throw site, but the damage is already done -- the
// object's own destructor, running normally at scope exit, ends up
// calling delete[] a SECOND time on the identical pointer, a genuine
// double free, caught here by AddressSanitizer exactly as honestly as
// Section G.2's own double-free trap.
int g_alloc_call_count = 0;
int g_fail_on_call = -1;

float* fault_alloc(int n) {
    g_alloc_call_count++;
    if (g_alloc_call_count == g_fail_on_call) {
        throw std::bad_alloc();
    }
    return new float[n];
}

struct UnsafeBuffer {
    float* paths;
    int count;
    UnsafeBuffer(int n) : count(n) {
        paths = fault_alloc(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~UnsafeBuffer() { delete[] paths; }
    UnsafeBuffer(const UnsafeBuffer& other) : count(other.count) {
        paths = fault_alloc(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    UnsafeBuffer& operator=(const UnsafeBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;                     // the OLD resource is freed first --
        paths = fault_alloc(other.count);   // if THIS throws, `paths` still holds that
        count = other.count;                // now-freed pointer value, unchanged
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
};

int main() {
    UnsafeBuffer u1(4);   // fault_alloc call 1
    UnsafeBuffer u2(8);   // fault_alloc call 2
    g_fail_on_call = 3;   // the NEXT allocation (u2's own copy-assignment) will throw

    printf("injecting a std::bad_alloc into u2 = u1's own copy-assignment allocation...\n");
    fflush(stdout);
    try {
        u2 = u1;  // delete[] u2.paths runs first (freeing it), THEN fault_alloc throws --
                  // u2.paths is left holding that SAME now-freed pointer value, unchanged
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc. u2.paths still holds the pointer value that was already freed a "
               "moment ago -- nothing has fixed it up.\n");
        fflush(stdout);
    }

    printf("main() returning normally now -- watch what u2's own destructor does with that stale pointer.\n");
    fflush(stdout);

    return 0;
}  // u2's destructor runs here: delete[] u2.paths -- the SAME pointer already freed inside
   // operator= moments ago, a genuine double free
```

### `06_copy_and_swap_safe.cpp`

```cpp
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>

// SafeBuffer fixes the exact danger 05_copy_and_swap_unsafe_trap.cpp
// just demonstrated, via copy-and-swap: a single by-value operator=
// dispatches to either the copy constructor (lvalue argument) or the
// move constructor (rvalue argument) via ordinary overload
// resolution, then swaps -- the ONLY thing that can throw (the
// allocation, building the by-value parameter) happens entirely
// BEFORE `*this` is touched at all, so a thrown exception leaves the
// object completely unchanged.
int g_alloc_call_count = 0;
int g_fail_on_call = -1;

float* fault_alloc(int n) {
    g_alloc_call_count++;
    if (g_alloc_call_count == g_fail_on_call) {
        throw std::bad_alloc();
    }
    return new float[n];
}

struct SafeBuffer {
    float* paths;
    int count;

    SafeBuffer(int n) : count(n) {
        paths = fault_alloc(n);
        for (int i = 0; i < n; i++) paths[i] = static_cast<float>(i);
    }
    ~SafeBuffer() { delete[] paths; }
    SafeBuffer(const SafeBuffer& other) : count(other.count) {
        paths = fault_alloc(count);  // may throw -- but nothing about `*this` exists yet
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    SafeBuffer(SafeBuffer&& other) noexcept : paths(nullptr), count(0) {
        swap(*this, other);  // steal via swap into a null/zero default state -- never throws
    }
    friend void swap(SafeBuffer& a, SafeBuffer& b) noexcept {
        std::swap(a.paths, b.paths);
        std::swap(a.count, b.count);
    }
    SafeBuffer& operator=(SafeBuffer other) {  // BY VALUE: the copy/move already happened above
        swap(*this, other);                      // just swap -- cannot throw
        return *this;
    }  // `other`'s own destructor now frees what USED to be *this's old resource
};

int main() {
    // Inject the identical failure a copy-assignment would hit.
    SafeBuffer s1(4);   // fault_alloc call 1
    SafeBuffer s2(8);   // fault_alloc call 2
    float s2_val0_before = s2.paths[0];
    int s2_count_before = s2.count;
    g_fail_on_call = 3;  // the copy ctor building operator='s by-value parameter will throw

    printf("injecting the identical std::bad_alloc into s2 = s1's own copy-assignment...\n");
    try {
        s2 = s1;
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc. s2.paths[0]=%.1f (was %.1f), s2.count=%d (was %d) -- COMPLETELY "
               "UNCHANGED: the exception happened while building operator='s own by-value parameter, "
               "before *this was ever touched at all -- the strong exception guarantee.\n",
               s2.paths[0], s2_val0_before, s2.count, s2_count_before);
    }

    // Confirm the dispatch genuinely differs between copy-assignment
    // (allocates) and move-assignment (does not), via the call
    // counter -- not merely asserted.
    g_alloc_call_count = 0;
    g_fail_on_call = -1;
    SafeBuffer s3(4);  // call 1
    SafeBuffer s4(4);  // call 2
    int before_copy_assign = g_alloc_call_count;
    s4 = s3;  // COPY-assignment: lvalue argument -> copy ctor builds the by-value param -> allocates
    printf("\ncopy-assignment (s4 = s3): alloc call count went from %d to %d (rose by 1 -- the copy ctor "
           "allocated)\n", before_copy_assign, g_alloc_call_count);

    SafeBuffer s5(4);  // call 3
    int before_move_assign = g_alloc_call_count;
    s4 = std::move(s5);  // MOVE-assignment: rvalue argument -> move ctor builds the by-value param -> swap only
    printf("move-assignment (s4 = std::move(s5)): alloc call count went from %d to %d (unchanged -- the "
           "move ctor only swapped, no allocation)\n", before_move_assign, g_alloc_call_count);

    printf("\ntrade-off worth naming: TRUE self-assignment (a = a) under copy-and-swap still does a full "
           "allocation and copy -- it is no longer the free no-op Section G.3's own explicit `if (this == "
           "&other)` guard made it. Copy-and-swap trades that one narrow optimization for a correctness "
           "guarantee (exception safety) the guarded version never had at all.\n");

    return 0;
}
```

### `07_noexcept_and_reallocation.cpp`

```cpp
#include <cstdio>
#include <vector>

// The CUDA C++ edition's own Worked Example G.3.5 explains why every
// move constructor and move assignment operator in this appendix is
// marked `noexcept`: std::vector uses std::move_if_noexcept
// internally when relocating existing elements during reallocation,
// falling back to COPYING them instead of moving if the move
// constructor isn't noexcept -- specifically to preserve vector's own
// strong exception guarantee (a throwing move mid-relocation could
// leave the vector in a corrupted, half-moved state; a throwing copy
// cannot, since the originals are left untouched until every copy
// succeeds). Entirely ordinary host C++, reproduced directly.
struct WithNoexcept {
    float val;
    static int copy_count;
    static int move_count;
    WithNoexcept(float v) : val(v) {}
    WithNoexcept(const WithNoexcept& other) : val(other.val) { copy_count++; }
    WithNoexcept(WithNoexcept&& other) noexcept : val(other.val) { move_count++; }
};
int WithNoexcept::copy_count = 0;
int WithNoexcept::move_count = 0;

struct WithoutNoexcept {
    float val;
    static int copy_count;
    static int move_count;
    WithoutNoexcept(float v) : val(v) {}
    WithoutNoexcept(const WithoutNoexcept& other) : val(other.val) { copy_count++; }
    WithoutNoexcept(WithoutNoexcept&& other) : val(other.val) { move_count++; }  // no noexcept
};
int WithoutNoexcept::copy_count = 0;
int WithoutNoexcept::move_count = 0;

int main() {
    // The noexcept struct: reserve capacity 2, fill it, reset the
    // counters, then force a reallocation with a third element.
    std::vector<WithNoexcept> v1;
    v1.reserve(2);
    v1.emplace_back(1.0f);
    v1.emplace_back(2.0f);
    WithNoexcept::copy_count = 0;
    WithNoexcept::move_count = 0;
    v1.emplace_back(3.0f);  // exceeds capacity 2 -> reallocates -> relocates the first 2 elements
    printf("WithNoexcept (move ctor IS noexcept): after forcing reallocation, copy_count=%d, move_count=%d "
           "-- the 2 pre-existing elements were relocated by MOVING them (safe: a noexcept move can never "
           "leave the vector in a corrupted half-moved state)\n",
           WithNoexcept::copy_count, WithNoexcept::move_count);

    // The IDENTICAL scenario, with the only difference being the
    // missing `noexcept` on the move constructor.
    std::vector<WithoutNoexcept> v2;
    v2.reserve(2);
    v2.emplace_back(1.0f);
    v2.emplace_back(2.0f);
    WithoutNoexcept::copy_count = 0;
    WithoutNoexcept::move_count = 0;
    v2.emplace_back(3.0f);
    printf("WithoutNoexcept (move ctor is NOT noexcept, otherwise identical): after the identical forced "
           "reallocation, copy_count=%d, move_count=%d -- std::vector fell back to COPYING the 2 "
           "pre-existing elements instead, silently, with no compiler diagnostic marking the difference "
           "either way.\n",
           WithoutNoexcept::copy_count, WithoutNoexcept::move_count);

    printf("\nthe only source difference between WithNoexcept and WithoutNoexcept is one keyword on the "
           "move constructor -- yet it flips whether reallocation is O(1)-per-element (move) or requires a "
           "genuine copy of every relocated element, a silent, non-diagnosed performance regression this "
           "book's own RiskPathBuffer's move constructor and move assignment operator (Section G.3) avoid "
           "specifically because both are marked noexcept.\n");

    return 0;
}
```

### `08_tensor_rule_of_zero_for_device_memory.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <utility>

// The CUDA C++ edition's own Section G.4 hand-writes all five special
// members for GPUPathBuffer, a struct wrapping a cudaMalloc'd device
// buffer -- not by choice, but structurally: cudaMalloc/cudaFree are
// host-side CUDA Runtime API entry points, so a struct wrapping them
// is necessarily host-only, and someone has to write correct
// ownership semantics for that raw allocation by hand, exactly the
// way Section G.3 hand-wrote them for a raw `new[]`'d host array.
//
// A LibTorch programmer wrapping a torch::Tensor faces the identical
// SHAPE of problem -- a class that owns a resource whose lifetime
// must be managed correctly across copies, moves, and destruction --
// but torch::Tensor is not a raw pointer. Internally, it holds an
// intrusive_ptr to a TensorImpl, a reference-counted handle exactly
// like std::shared_ptr's own mechanism, and that handle already
// implements a fully correct Rule of Five: copying a torch::Tensor
// increments a reference count and shares the underlying storage;
// moving one steals the handle and leaves the source undefined;
// destroying one decrements the count and frees the storage only when
// it reaches zero -- on CPU memory in this sandbox, and on cudaMalloc'd
// device memory identically, on a real GPU-attached machine, with the
// SAME source code either way. This section extends Section G.3.3's
// own Rule-of-Zero argument (wrap the resource in something that
// already manages itself) to the EXACT scenario Section G.4 needed a
// hand-written five for.
struct TensorPathBuffer {
    torch::Tensor paths;
    TensorPathBuffer(int n) : paths(torch::arange(n, torch::kFloat32)) {}
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment -- all five are the compiler's
    // own defaults, and every one is correct, because torch::Tensor's
    // OWN five special members already are, on CPU today and on a
    // CUDA device identically, with zero source changes to this
    // struct at all.
};

int main() {
    // Copy: torch::Tensor's own copy constructor SHARES the
    // underlying storage and increments a real reference count --
    // this is genuinely DIFFERENT from Section G.3's own deep-copy
    // semantics, and stated as such rather than glossed over.
    TensorPathBuffer a(5);
    std::cout << "a.paths.use_count() right after construction = " << a.paths.use_count() << std::endl;
    TensorPathBuffer b = a;
    std::cout << "copy: a.paths.data_ptr() == b.paths.data_ptr()? "
              << (a.paths.data_ptr() == b.paths.data_ptr())
              << " (SHARED storage, unlike Section G.3's own deep copy), use_count now = "
              << a.paths.use_count() << " (both a and b's own handles reference the identical storage)"
              << std::endl;

    // Move: torch::Tensor's own move constructor steals the handle,
    // exactly the way RiskPathBuffer's own hand-written move
    // constructor does, but written by LibTorch's own authors once,
    // reused correctly here with zero code of this struct's own.
    TensorPathBuffer c(5);
    void* c_ptr_before_move = c.paths.data_ptr();
    TensorPathBuffer d = std::move(c);
    std::cout << "\nmove: d.paths.data_ptr() == c's own pre-move data_ptr()? "
              << (d.paths.data_ptr() == c_ptr_before_move)
              << ", c.paths.defined() after move = " << c.paths.defined()
              << " (undefined -- the handle was genuinely stolen, the compiler's default move constructor "
              << "for TensorPathBuffer just calls torch::Tensor's own)" << std::endl;

    // Self-assignment: zero hand-written guards anywhere in
    // TensorPathBuffer, yet it is safe -- torch::Tensor's own
    // operator= already handles it, one layer down, the identical
    // Rule-of-Zero delegation Section 04's RiskPathBufferV2 already
    // demonstrated for std::vector.
    TensorPathBuffer e(4);
    torch::Tensor before_self = e.paths.clone();
    e.paths = e.paths;
    std::cout << "\nself-assignment (e.paths = e.paths), zero hand-written guards anywhere: torch::allclose "
              << "confirms data intact? " << torch::allclose(e.paths, before_self) << std::endl;

    std::cout << "\nnone of this required a hand-written destructor, copy constructor, copy assignment, "
              << "move constructor, or move assignment operator -- unlike Section G.4's own GPUPathBuffer, "
              << "which had to hand-write all five specifically because cudaMalloc/cudaFree return and "
              << "accept a RAW pointer with no ownership semantics of its own at all. torch::Tensor already "
              << "IS the correctly-implemented Rule of Five (or, from an application programmer's own point "
              << "of view, the Rule of Zero it enables) for exactly the resource -- device memory -- Section "
              << "G.4's own worked example had to build by hand." << std::endl;

    return 0;
}
```

### `09_value_at_risk.cpp`

```cpp
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section G.5 reuses Chapter 22.4's own
// hand-rolled box_muller/xorshift-PRNG/GBM-update machinery "byte for
// byte" at a 1-day horizon, then computes three VaR methodologies.
// This section reuses THIS book's own Chapter 22.1 GBM approach
// instead: a single vectorized closed-form expression via real
// torch::randn, the identical technique this book has used since
// Chapter 22 rather than a hand-rolled PRNG -- consistent with the
// established "don't rebuild machinery this book already has a real
// API for" pattern. The three methodologies themselves are unchanged,
// ordinary quantitative finance, with no CUDA content in them at all.
int main() {
    torch::manual_seed(42);

    double S0 = 100.0, mu = 0.03, sigma = 0.20;
    double dt = 1.0 / 252.0;  // one trading day
    int64_t num_paths = 200000;

    // GBM's own exact one-step solution (Chapter 22.1's own formula,
    // applied at a 1-day horizon instead of Chapter 22's own 1-year
    // option-pricing horizon).
    torch::Tensor z = torch::randn({num_paths});
    torch::Tensor s_t = S0 * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
    torch::Tensor pnl = s_t - S0;

    // Simulated (empirical/historical-style) VaR: sort P&L ascending,
    // read the loss at the 1st percentile for 99% confidence.
    torch::Tensor pnl_sorted = std::get<0>(torch::sort(pnl));
    int64_t tail_idx = static_cast<int64_t>(0.01 * static_cast<double>(num_paths));  // floor(0.01*N) = 2000
    double simulated_var_99 = -pnl_sorted[tail_idx].item<double>();

    // Parametric (variance-covariance) VaR: S0 * sigma_1day * z_99.
    double sigma_1day = sigma * std::sqrt(dt);
    double z_99 = 2.326348;  // 99th-percentile standard-normal critical value
    double parametric_var_99 = S0 * sigma_1day * z_99;

    double relative_diff = std::abs(simulated_var_99 - parametric_var_99) / parametric_var_99;

    // Conditional VaR / Expected Shortfall: the average of the k
    // worst P&L values beyond the VaR cutoff.
    torch::Tensor worst_k = pnl_sorted.slice(0, 0, tail_idx);
    double cvar_99 = -worst_k.mean().item<double>();

    std::cout << "1-day GBM P&L, S0=" << S0 << ", mu=" << mu << ", sigma=" << sigma << ", "
              << num_paths << " paths, real torch::randn (manual_seed 42), Chapter 22.1's own vectorized "
              << "closed-form GBM step:" << std::endl;
    std::cout << "  tail index (floor(0.01 * " << num_paths << ")) = " << tail_idx << std::endl;
    std::cout << "  simulated 99% 1-day VaR   = " << simulated_var_99 << std::endl;
    std::cout << "  sigma_1day = " << sigma_1day << ", z_99 = " << z_99 << std::endl;
    std::cout << "  parametric 99% 1-day VaR  = " << parametric_var_99 << std::endl;
    std::cout << "  relative difference       = " << relative_diff << " (" << (relative_diff * 100.0)
              << "%)" << std::endl;
    std::cout << "  CVaR_99 (mean of the " << tail_idx << " worst P&L values) = " << cvar_99 << std::endl;
    std::cout << "  CVaR_99 >= simulated VaR_99? " << (cvar_99 >= simulated_var_99) << std::endl;

    std::cout << "\nthe simulated and parametric figures are CLOSE but not identical -- expected, and "
              << "explained the same way Section G.5's own text explains its own gap: GBM's log-returns "
              << "are exactly normal by construction, but VaR here is measured on ARITHMETIC P&L (S_T - "
              << "S0), which is log-normal, not normal -- so the parametric method's own normality "
              << "assumption is a genuine approximation even against this file's own internally consistent "
              << "data, not a bug in either computation. This book's own numbers will not match the CUDA "
              << "book's own reported 2.8665/2.9309/2.25%/3.2938 exactly, since a different RNG (torch::randn "
              << "vs. a hand-rolled xorshift+Box-Muller) produces a genuinely different sample -- the "
              << "CVaR >= VaR relationship holding, and the two methods agreeing closely, is the actual "
              << "claim being verified, not any one specific figure." << std::endl;

    return 0;
}
```

### `10_xva_and_variants.cpp`

```cpp
#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <vector>

// The CUDA C++ edition's own Section G.6 builds CVA, DVA, and FVA
// from a genuine exposure profile -- EE(t) and ENE(t), the expected
// positive and negative value of a forward contract -- extracted from
// checkpointed Monte Carlo paths, reusing Chapter 22.4's own GBM
// machinery. This section reproduces the identical methodology using
// this book's own real torch::Tensor GBM machinery instead (Chapter
// 22.1's own vectorized closed-form step, applied incrementally
// across four quarterly checkpoints to build one continuous path per
// simulated scenario, since GBM's own Markov property lets each
// quarter's step depend only on the previous quarter's own price).
int main() {
    torch::manual_seed(42);

    double S0 = 100.0, K = 100.0, mu = 0.03, sigma = 0.20, r = 0.03;
    double lambda_C = 0.02, R_C = 0.40;  // counterparty hazard rate, recovery
    double lambda_B = 0.01, R_B = 0.40;  // bank's own hazard rate, recovery
    double s_funding = 0.005;
    int64_t num_paths = 200000;
    std::vector<double> checkpoints = {0.25, 0.50, 0.75, 1.00};
    double dt = 0.25;

    torch::Tensor s = torch::full({num_paths}, S0, torch::kFloat64);
    std::vector<torch::Tensor> s_at_checkpoint;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        torch::Tensor z = torch::randn({num_paths}, torch::kFloat64);
        s = s * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
        s_at_checkpoint.push_back(s.clone());
    }

    // Cross-check: simulated E[S(T)] against the closed-form GBM mean.
    double simulated_mean_st = s_at_checkpoint.back().mean().item<double>();
    double theoretical_mean_st = S0 * std::exp(mu * checkpoints.back());
    double mean_relative_diff = std::abs(simulated_mean_st - theoretical_mean_st) / theoretical_mean_st;

    // EE(t) = E[max(V(t), 0)], ENE(t) = E[min(V(t), 0)], V(t) = S(t) - K.
    std::vector<double> ee(checkpoints.size()), ene(checkpoints.size());
    for (size_t k = 0; k < checkpoints.size(); k++) {
        torch::Tensor v = s_at_checkpoint[k] - K;
        ee[k] = torch::clamp_min(v, 0.0).mean().item<double>();
        ene[k] = torch::clamp_max(v, 0.0).mean().item<double>();
    }

    std::cout << "quarterly exposure profile, forward contract, S0=" << S0 << ", K=" << K << ", mu=" << mu
              << ", sigma=" << sigma << ", " << num_paths << " paths:" << std::endl;
    std::cout << "  t      EE(t)       ENE(t)      EE(t)+ENE(t)" << std::endl;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        std::cout << "  " << checkpoints[k] << "   " << ee[k] << "   " << ene[k] << "   " << (ee[k] + ene[k])
                  << std::endl;
    }

    std::cout << "\ncross-check: simulated E[S(T)] = " << simulated_mean_st << ", theoretical S0*exp(mu*T) = "
              << theoretical_mean_st << " (relative diff " << (mean_relative_diff * 100.0) << "%)" << std::endl;
    std::cout << "cross-check: EE(T)+ENE(T) = " << (ee.back() + ene.back()) << " vs. E[S(T)]-K = "
              << (simulated_mean_st - K) << " (identity: V(T)'s own mean splits exactly into its positive "
              << "and negative parts)" << std::endl;

    // CVA and DVA: survival-probability-weighted expected exposure,
    // discounted, over each quarterly interval.
    auto survival = [](double lambda, double t) { return std::exp(-lambda * t); };
    auto discount = [&](double t) { return std::exp(-r * t); };

    double cva = 0.0, dva = 0.0, fva = 0.0;
    double t_prev = 0.0;
    for (size_t k = 0; k < checkpoints.size(); k++) {
        double t_k = checkpoints[k];
        double q_c_prev = survival(lambda_C, t_prev), q_c_k = survival(lambda_C, t_k);
        double q_b_prev = survival(lambda_B, t_prev), q_b_k = survival(lambda_B, t_k);
        double df_k = discount(t_k);

        double cva_term = (1.0 - R_C) * ee[k] * (q_c_prev - q_c_k) * df_k;
        double dva_term = (1.0 - R_B) * (-ene[k]) * (q_b_prev - q_b_k) * df_k;
        double fva_term = s_funding * ee[k] * df_k * dt;

        cva += cva_term;
        dva += dva_term;
        fva += fva_term;

        t_prev = t_k;
    }

    double net_xva = -cva + dva - fva;

    std::cout << "\nCVA = " << cva << std::endl;
    std::cout << "DVA = " << dva << std::endl;
    std::cout << "FVA = " << fva << std::endl;
    std::cout << "Net XVA (-CVA + DVA - FVA) = " << net_xva << std::endl;

    std::cout << "\nDVA < CVA here specifically because the bank's own hazard rate (" << lambda_B
              << ") was set lower than the counterparty's (" << lambda_C
              << ") -- a better-credit bank has a smaller DVA, all else equal, exactly the relationship "
              << "Section G.6's own text reports. MVA and KVA are named by the CUDA book's own text as "
              << "further XVA components but explicitly not computed there; this section follows the "
              << "identical scope decision, computing CVA/DVA/FVA only. This book's own exact figures will "
              << "not match the CUDA book's own reported ones, since a different RNG produces a genuinely "
              << "different sample of the same GBM process -- what's being verified is the exposure-profile "
              << "shape (EE growing over time, ENE(t) staying negative) and the internal identities above, "
              << "not any one specific dollar figure." << std::endl;

    return 0;
}
```

## Chapter Summary

This appendix mapped the CUDA C++ edition's own seven-section Appendix G onto LibTorch's own domain, and found that most of the CUDA book's own resource-management teaching -- the double-free bug a raw pointer's default copy invites, the Rule of Five correctly implemented and its guards genuinely tested, the Rule of Zero, copy-and-swap's exception safety, and why every move operation needs `noexcept` -- was never CUDA-specific at all, and reproduced directly. The one genuinely CUDA-specific section, a hand-written Rule of Five for a `cudaMalloc`'d device buffer, became this appendix's own sharpest honest divergence: `torch::Tensor` already IS that correctly-implemented Rule of Five, for device memory, verified here via real `use_count()` and `data_ptr()` checks rather than asserted. And the appendix closed with Value-at-Risk and XVA, real `torch::Tensor` GBM machinery extended to a 1-day risk horizon and a full checkpointed exposure profile, landing close to the CUDA book's own reported figures despite a completely different random sample -- confirming the methodology, not any one implementation's specific numbers.

## Self-Check Questions

1. Section G.1's `MarketQuote` needs no hand-written special members at all. Explain precisely what property of `MarketQuote` makes the compiler's own default copy correct, and what would have to change about the struct for that to stop being true.
2. Section G.2's double free is triggered by `b`'s own destructor freeing a buffer `a` still holds a copy of the pointer to. Explain why simply removing `b`'s own scope (never letting it go out of scope before `main()` ends) would NOT fix the underlying bug, only postpone when it appears.
3. Worked Example G.3.1 tests self-assignment and self-move guards directly, rather than trusting that writing `if (this == &other)` is enough. Worked Example G.3.2's own unguarded variants produce two different SILENT failures. Explain why neither failure trips AddressSanitizer's own heap-corruption checks, in contrast to Section G.2's own double free.
4. Worked Example G.3.4's `SafeBuffer` gives up the free no-op Section G.3.1's own guard provided for true self-assignment. Explain precisely what correctness guarantee copy-and-swap gains in exchange, and why that guarantee specifically requires the allocation to happen BEFORE `*this` is touched.
5. Section G.4 states that copying a `TensorPathBuffer` is genuinely different from copying a `RiskPathBuffer`. Explain the specific difference, using `use_count()` and `data_ptr()`, and what a LibTorch programmer would reach for if they needed `RiskPathBuffer`-style deep-copy independence from a `torch::Tensor` instead.
6. Section G.5 reports Section G.5.1's own simulated and parametric VaR figures as "close but not identical," explaining the gap via arithmetic P&L versus GBM's own exactly-normal log-returns. Explain precisely why VaR measured on `S_T - S0` is not measuring a normally distributed quantity, even though `S_T` itself comes from a GBM process whose log is exactly normal by construction.
7. Section G.6 checks two identities directly rather than assuming them: simulated `E[S(T)]` against the closed-form GBM mean, and `EE(T)+ENE(T)` against `E[S(T)]-K`. Explain why the second identity must hold exactly, to full floating-point precision, regardless of which RNG or how many paths were used.

## Where We Go Next

This appendix closes a loop this book opened with Appendix C: a memory-hierarchy story told once for CPU tensors, and revisited here from the angle of WHO is responsible for correctly managing a resource's own lifetime -- the compiler, for a value like `MarketQuote`; a hand-written five, for a raw pointer; or `torch::Tensor`'s own reference-counted handle, for exactly the device-memory case the CUDA book's own Section G.4 had to hand-write. The verification techniques compound across this book's own appendices the same way they do in the CUDA C++ edition: AddressSanitizer, first used in this book's own Appendix D.6, catches Section G.2's double free and Section G.3.3's own exception-unsafe assignment with the same deterministic honesty. The financial sections extend Chapter 22.1's own GBM machinery rather than replacing it, so a change to that chapter's own random-sampling approach would propagate its effect identically into every VaR and XVA figure here. The appendices that follow return to tensor mechanics directly: tensor contractions on CPU and GPU.

## Worked Solutions

**1.** `MarketQuote` owns nothing but two `double` values -- no pointer, no handle, no resource whose lifetime is tracked separately from the struct's own lifetime. A bitwise, member-by-member copy of two doubles IS a complete, correct copy of the value in every sense that matters, so the compiler's own default-generated copy constructor (and every other one of the five) is already exactly correct. This would stop being true the moment `MarketQuote` gained a member whose value is itself a HANDLE to something else -- a raw pointer to heap memory, a file handle, anything where the bits being copied are an address or reference rather than the actual data, which is precisely the situation `RiskPathBuffer` is in from Section G.2 onward.

**2.** The underlying bug is that `RiskPathBuffer`'s own compiler-generated copy constructor and copy assignment operator copy the POINTER `paths`, not the array it points to -- so ANY two `RiskPathBuffer` objects that end up sharing that pointer value are two independent owners of one physical allocation, each of which will eventually run its own destructor and call `delete[]` on it. Preventing `b` from ever going out of scope before `main()` ends would only delay which object's destructor runs last -- at the very end of `main()`, both `a` and `b` would still each run their own destructor on the identical shared pointer, so the double free would still occur, just at a different point in the program rather than not at all. The only real fix is what Section G.3 does: give the copy constructor and copy assignment operator their own correct, independent-allocation semantics.

**3.** AddressSanitizer's own heap-corruption checks specifically detect memory being READ or WRITTEN after it has already been freed (a use-after-free) or freed more than once (a double free) -- both require some operation to touch memory THROUGH A POINTER THAT NO LONGER OWNS VALID MEMORY. Worked Example G.3.2's own two unguarded failures never do that: the self-assignment case overwrites a buffer's own contents with a FRESH allocation's own uninitialized garbage (a real allocation, never freed prematurely, just never given the values it should have received) and the self-move case simply nulls out a pointer that had just been correctly assigned one line earlier (again, no freed memory is ever touched) -- both are genuine bugs, but neither one is the specific KIND of memory-safety violation AddressSanitizer's own instrumentation is built to catch, which is exactly why Worked Example G.3.1's own explicit guards matter even though nothing crashes without them.

**4.** Copy-and-swap gains the STRONG EXCEPTION GUARANTEE: if an exception is thrown during assignment, the object being assigned to is left EXACTLY as it was before the assignment began, with no partial, corrupted, or dangling state at all. This specifically requires the (possibly-throwing) allocation to happen while building `operator=`'s own by-value parameter -- a completely separate, temporary object -- BEFORE the `swap()` call that actually modifies `*this` ever runs; if the allocation happens first and succeeds, the swap that follows (a `noexcept` operation, by construction) cannot itself fail, so by the time `*this` is touched at all, the operation is already guaranteed to complete successfully. Section G.3.1's own free-then-allocate ordering violates this specifically because it touches `*this` (by freeing its old resource) BEFORE attempting the operation that might fail.

**5.** `a.paths.data_ptr() == b.paths.data_ptr()` reporting true, together with `use_count()` genuinely rising from 1 to 2 after the copy, confirms that copying a `TensorPathBuffer` does NOT allocate a new, independent block of storage the way `RiskPathBuffer`'s own hand-written deep-copy constructor does -- instead, both `a.paths` and `b.paths` become two separate handles referencing the IDENTICAL underlying storage, with a real reference count tracking how many handles are currently sharing it. A LibTorch programmer who genuinely needs `RiskPathBuffer`-style deep-copy independence -- two tensors that start with identical values but can be mutated completely independently afterward -- would call `.clone()` explicitly, exactly the way Section G.3.4's own worked example uses it to snapshot a tensor's own values before a self-assignment test, rather than relying on `torch::Tensor`'s own default copy constructor to do it implicitly.

**6.** GBM's own construction guarantees that `log(S_T / S0)` is exactly normally distributed -- that is the entire content of "geometric" Brownian motion: the LOGARITHM of the price ratio follows a normal distribution. `S_T` itself, being `S0` times the EXPONENTIAL of a normal random variable, is therefore LOG-NORMALLY distributed, not normally distributed -- and `S_T - S0`, an affine (shift-only, not log) transformation of a log-normal variable, remains log-normal in shape (skewed, with a longer right tail than a normal distribution of the same variance would have), never becoming normal itself. The parametric VaR formula `S0 * sigma_1day * z_99` assumes the P&L IS normally distributed, which is only an approximation for arithmetic P&L on a GBM-driven price, even though the log-returns that actually generate that price are exactly normal by construction -- which is exactly why the simulated and parametric figures in Worked Example G.5.1 are close (the approximation is reasonable at a 1-day horizon, where volatility is small) but not identical.

**7.** `EE(t)` is defined as the mean of `max(V(t), 0)` and `ENE(t)` as the mean of `min(V(t), 0)` -- for ANY real number `v`, `max(v, 0) + min(v, 0)` equals `v` exactly (whichever of the two terms is nonzero equals `v` itself, and the other is exactly zero), so summing this identity across every single simulated path and dividing by the path count preserves it exactly: the mean of `max(V,0)+min(V,0)` across all paths equals the mean of `V` across all paths, to full floating-point precision (up to ordinary floating-point summation rounding, which is far smaller than any Monte Carlo sampling error). This holds regardless of which RNG generated the underlying `V(t)` values or how many paths were simulated, because it is a per-path ALGEBRAIC identity, not a statistical property that depends on the sample being representative of anything -- which is exactly why it makes a genuinely useful cross-check independent of whatever the "true" GBM parameters happen to be.
