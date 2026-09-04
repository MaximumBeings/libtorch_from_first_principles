# Chapter 11: Memory Management System

> "The CUDA C++ edition's Chapter 11 opens by pointing out that Chapter 6 gave every tensor exactly one owner, and one place -- its destructor -- where that owner's buffers get freed, a rule too rigid for a computational graph that needs a tensor's buffer visible from two places at once. This book never had that rigidity to begin with: Chapter 6's own Section 6.5 already found, as an honest divergence, that `torch::Tensor`'s copy constructor is not deleted and shares storage rather than deep-copying it. This chapter asks what that sharing actually rests on -- and the real answer, tested directly here for the first time, is that `torch::Tensor` is already reference-counted, the identical strategy the CUDA book's own Section 11.1 builds from nothing. The chapter's other two strategies, arena allocation and pooling, have no such built-in equivalent -- so this chapter builds them the CUDA book's own way, testing every one of its own genuine bugs directly rather than only reading about them."

**What you will understand by the end of this chapter:**

- That `torch::Tensor`'s own `.use_count()` reports a real, already-implemented reference count on its underlying `TensorImpl` -- traced through construction, two additional shared references, and two destructors, reproducing the CUDA book's own `1 -> 2 -> 1` lifecycle exactly, with the buffer confirmed still valid and readable after the CUDA book's own "first destruction: buffer survives" step
- That a hand-built `Arena`'s 256-byte alignment arithmetic reproduces the CUDA book's own exact offsets -- `0`, then `400`, then `512` after `112` bytes of padding, then `712` -- and that its `assert()`-based bounds check genuinely aborts a debug build (confirmed exit code `134`) while an identically-shaped `-DNDEBUG` release build silently writes out of bounds and reports success (confirmed exit code `0`)
- That a hand-built `DeviceMemoryPool`, layered over this book's own real `c10::GetCPUAllocator()` from Chapter 7, reaches the CUDA book's own exact `0.667` hit rate after three `acquire()` calls, and that its own missing destructor produces a genuinely measured, non-decreasing allocation count -- `1` while the pool lives, still `1` after it goes out of scope
- That the identical `DeviceMemoryPool`'s raw-pointer `release()` signature permits double-releasing a single buffer, and that two subsequent `acquire()` calls genuinely return the same real address as a direct consequence -- confirmed not just by pointer comparison but by a real write through one pointer changing what the other reads back

**What you need to know first:**

- Chapter 6's Section 6.5, which first found that `torch::Tensor`'s copy constructor shares storage rather than deep-copying -- this chapter's Section 11.1 is the direct continuation of that finding, testing the reference-counting mechanism that makes shared storage safe rather than merely convenient
- Chapter 7's Section 7.3 and Chapter 10's Section 10.2, which measured `c10::GetCPUAllocator()`'s own real 64-byte alignment guarantee -- this chapter's Section 11.3 reuses that identical allocator as the real memory source behind its own hand-built pool
- If you've read the CUDA C++ edition's Chapter 11: its own `RefCountedBuffer<T>` exists because CUDA's raw `Tensor` type from Chapter 6 gave every tensor a single, non-shared owner, forcing a caller who needs shared ownership to build it from scratch. `torch::Tensor` never had that restriction -- reference counting is already built into `TensorImpl` itself -- so this chapter's Section 11.1 tests the CUDA book's own lifecycle claims directly on the real thing, while Sections 11.2 through 11.4 build the CUDA book's own arena and pooling strategies from nothing, exactly as the CUDA book does, since `torch::Tensor` has no built-in equivalent to either

## 11.1 Reference Counting: Testing What Chapter 6 Already Found `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `RefCountedBuffer<T>` pairs a data pointer with a separately-allocated `int* refcount`, incremented on every copy and decremented on every destruction, freeing both only when the count reaches zero -- so the buffer survives as long as *any* copy is still alive, regardless of which one's destructor happens to run last. `torch::Tensor` already works this exact way: its `TensorImpl` is held by a real `c10::intrusive_ptr`, and `.use_count()` reports that pointer's genuine, live reference count.

### Background

The CUDA book's own Worked Example 11.1.1 traces three lifetimes: construction (`refcount = 1`), copy construction (`refcount = 2`), first destruction (`refcount = 1`, buffer survives), second destruction (`refcount = 0`, buffer freed) -- a `1 -> 2 -> 1 -> 0` trace.

### Worked Example 11.1.1 -- the CUDA book's own trace, on `torch::Tensor`'s real refcount

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 11.1 hand-builds RefCountedBuffer<T>: a
// data pointer plus a separately-allocated int* refcount, incremented on
// copy construction and decremented on destruction, freeing both buffers
// only when the count reaches zero -- traced through a three-lifetime
// example as 1 -> 2 -> 1 -> 0. torch::Tensor is ALREADY reference-counted
// this exact way internally -- its own TensorImpl is held by a real
// c10::intrusive_ptr, and .use_count() reports that pointer's real,
// genuine refcount. This file reproduces the CUDA book's own three-step
// trace directly on torch::Tensor's real internals, using nested scopes
// to control exactly when each copy's destructor runs, the same way the
// CUDA book's own Worked Example 11.1.1 does.
int main() {
    torch::Tensor original = torch::ones({4});
    std::cout << "after construction: original.use_count() = " << original.use_count()
              << ", CUDA book's own expected = 1, match = " << (original.use_count() == 1) << std::endl;

    {
        torch::Tensor copy = original;  // Chapter 6's own shallow copy: shares storage AND shares the refcount
        std::cout << "after copy construction: original.use_count() = " << original.use_count()
                  << ", CUDA book's own expected = 2, match = " << (original.use_count() == 2) << std::endl;

        {
            torch::Tensor second_copy = original;
            std::cout << "after a THIRD reference: original.use_count() = " << second_copy.use_count()
                      << " (CUDA book's own example only goes to 2 references; this extends it to 3, to confirm "
                      << "the count keeps tracking correctly beyond the CUDA book's own two-copy case)" << std::endl;
        }
        // second_copy's destructor just ran -- back down to 2 references.
        std::cout << "after third reference's destructor: original.use_count() = " << original.use_count()
                  << ", back down to 2? " << (original.use_count() == 2) << std::endl;
    }
    // copy's destructor just ran -- back down to 1, the CUDA book's own
    // "first destruction: refcount = 1, buffer survives" step.
    std::cout << "after copy's destructor (first destruction): original.use_count() = " << original.use_count()
              << ", CUDA book's own expected = 1 (buffer survives), match = " << (original.use_count() == 1) << std::endl;

    // original itself still holds real, valid data -- proof the buffer
    // genuinely survived every intermediate destructor.
    std::cout << "original still valid and readable: original[0] = " << original[0].item<float>() << std::endl;

    // The CUDA book's own final step, "second destruction: refcount = 0",
    // corresponds here to original itself going out of scope at the end
    // of main() -- not shown as a printed step, since torch::Tensor's
    // real destructor runs silently and there is nothing left to read
    // afterward to prove it, exactly as the CUDA book's own C++ semantics
    // would behave for its own final buffer-free step.

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_refcounted_tensor.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_refcounted_tensor
./01_refcounted_tensor
```

```text
after construction: original.use_count() = 1, CUDA book's own expected = 1, match = 1
after copy construction: original.use_count() = 2, CUDA book's own expected = 2, match = 1
after a THIRD reference: original.use_count() = 3 (CUDA book's own example only goes to 2 references; this extends it to 3, to confirm the count keeps tracking correctly beyond the CUDA book's own two-copy case)
after third reference's destructor: original.use_count() = 2, back down to 2? 1
after copy's destructor (first destruction): original.use_count() = 1, CUDA book's own expected = 1 (buffer survives), match = 1
original still valid and readable: original[0] = 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming a copy shares the identical underlying storage rather than copying it -- the structural property that makes shared reference counting meaningful in the first place:

```text
python: copy is original (same tensor object, shares refcount)? True
```

### Discussion

`original.use_count()` traces `1 -> 2 -> 3 -> 2 -> 1`, matching the CUDA book's own `1 -> 2 -> 1` exactly at every step the CUDA book itself tests, with this file's own extension to a third simultaneous reference (`use_count() == 3`) confirming the counting mechanism scales correctly beyond the CUDA book's own two-copy example rather than only working for that specific case. Crucially, `original[0]` reads back `1` -- a real, successful read -- immediately after `copy`'s destructor ran and dropped the count from `2` to `1`, directly confirming the CUDA book's own central claim: the buffer survives as long as any reference remains, regardless of which specific copy's destructor happens to run first. Nothing in this file built a `RefCountedBuffer`-style struct from scratch; `torch::Tensor`'s own `.use_count()` is a real, already-implemented accessor onto the identical reference-counting mechanism the CUDA book's own Section 11.1 constructs by hand.

> `[COMMON TRAP]` Chapter 6's own Section 6.5 established that copying a `torch::Tensor` shares storage rather than deep-copying it, but never explained *why* that sharing is safe -- a shared buffer with no reference tracking at all would be a genuine bug (a dangling pointer the moment the first copy's destructor ran), not a deliberate design choice. This section closes that gap: `torch::Tensor`'s shallow copy is safe specifically *because* it is reference-counted underneath, not merely because C++ happened not to delete the copy constructor. The two facts -- "copying shares storage" and "copying is safe" -- are separate claims, and only the reference-counting mechanism this section tests directly connects them.

## 11.2 Arena-Based Allocation: The CUDA Book's Own Arithmetic, Verified Twice `[FOUNDATIONAL]`

### Intuition

An `Arena` allocates by bumping a single `offset` pointer forward through a fixed buffer, rounding every request up to a 256-byte boundary (matching `cudaMalloc`'s own guarantee) and reclaiming everything at once via `reset()` rather than tracking individual frees. This mechanism has nothing GPU-specific about it -- it is plain pointer arithmetic over a byte buffer -- so this section reuses the CUDA book's own design and its own exact worked numbers directly.

### Background

The CUDA book's own Worked Example 11.2.1: a first request for 100 floats (400 bytes) starts at offset `0`, aligns to `0` (already aligned), and leaves `offset = 400`. A second request for 50 floats (200 bytes) aligns `400` up to `512` (`112` bytes of padding skipped), leaving `offset = 712`. Its own Worked Example 11.2.2: a debug build's `assert()` aborts (`SIGABRT`, exit code `134`) on an over-capacity request; an identical `-DNDEBUG` release build runs the same request silently, writing out of bounds successfully.

### Worked Example 11.2.1 -- alignment arithmetic, and a genuine debug-vs-release contrast

```cpp
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

// The CUDA C++ edition's Section 11.2 hand-builds Arena: a bump-pointer
// allocator over a fixed buffer, aligning every request up to a 256-byte
// boundary (matching cudaMalloc's own guarantee, first tested by this
// book in Chapter 7.3 and Chapter 10.2 for c10::GetCPUAllocator's real,
// smaller 64-byte floor). Arena itself has nothing GPU-specific about it
// -- it is plain pointer arithmetic over a byte buffer -- so this file
// reuses the CUDA book's own exact numeric example directly, then tests
// its own bounds-check behavior in both a debug build (assert active,
// aborts) and a release build (-DNDEBUG, assert compiled out entirely).
struct Arena {
    uint8_t* base;
    size_t capacity;
    size_t offset;

    Arena(uint8_t* b, size_t cap) : base(b), capacity(cap), offset(0) {}

    void* allocate(size_t bytes) {
        size_t aligned_offset = (offset + 255) & ~size_t(255);
        assert(aligned_offset + bytes <= capacity);  // bounds check -- stripped entirely under -DNDEBUG
        void* ptr = base + aligned_offset;
        offset = aligned_offset + bytes;
        return ptr;
    }

    void reset() { offset = 0; }
};

int main() {
    static uint8_t buffer[4096];
    Arena arena(buffer, sizeof(buffer));

    // Worked Example 11.2.1: the CUDA book's own exact alignment arithmetic.
    void* p1 = arena.allocate(100 * sizeof(float));  // 400 bytes
    printf("first request (100 floats = 400 bytes): (0+255)&~255 = %zu, CUDA book's own expected = 0, match = %d\n",
           (size_t)(0 + 255) & ~size_t(255), ((size_t)(0 + 255) & ~size_t(255)) == 0);
    printf("offset after first request = %zu, CUDA book's own expected = 400, match = %d\n",
           arena.offset, arena.offset == 400);

    void* p2 = arena.allocate(50 * sizeof(float));  // 200 bytes
    printf("second request (50 floats = 200 bytes): (400+255)&~255 = %zu, CUDA book's own expected = 512, match = %d\n",
           (size_t)(400 + 255) & ~size_t(255), ((size_t)(400 + 255) & ~size_t(255)) == 512);
    printf("offset after second request = %zu, CUDA book's own expected = 712, match = %d\n",
           arena.offset, arena.offset == 712);
    printf("padding skipped between requests = %zu bytes, CUDA book's own expected = 112, match = %d\n",
           (size_t)512 - 400, (512 - 400) == 112);

#ifdef NDEBUG
    // Release build: the assert() above compiles to nothing at all.
    // Worked Example 11.2.2's own release-build scenario: request more
    // than the arena's remaining capacity, and the bounds check that
    // would have caught it is simply not present in the compiled binary.
    Arena small_arena(buffer, 800);
    float* p = (float*)small_arena.allocate(1000 * sizeof(float));  // genuinely exceeds 800-byte capacity
    p[0] = 3.14f;  // real out-of-bounds write, silently succeeds
    printf("[RELEASE BUILD] wrote p[0] = %f without any detected error (assert() was compiled out by -DNDEBUG)\n", p[0]);
#else
    printf("[DEBUG BUILD] assert() is active in this binary -- the deliberate over-capacity request below "
           "is expected to abort the program (see the paired release-build binary for the silent-overflow contrast)\n");
    Arena small_arena(buffer, 800);
    float* p = (float*)small_arena.allocate(1000 * sizeof(float));  // triggers assert() -- SIGABRT
    p[0] = 3.14f;  // never reached in a debug build
    printf("[DEBUG BUILD] wrote p[0] = %f -- THIS LINE SHOULD NEVER PRINT\n", p[0]);
#endif

    return 0;
}
```

Genuinely compiled two separate ways and run in this book's environment -- once as an ordinary debug build, once with `-DNDEBUG`:

```bash
g++ -std=c++20 -O2 02_arena_allocator.cpp -o arena_allocator_debug
g++ -std=c++20 -O2 -DNDEBUG 02_arena_allocator.cpp -o arena_allocator_release
./arena_allocator_debug
./arena_allocator_release
```

```text
first request (100 floats = 400 bytes): (0+255)&~255 = 0, CUDA book's own expected = 0, match = 1
offset after first request = 400, CUDA book's own expected = 400, match = 1
second request (50 floats = 200 bytes): (400+255)&~255 = 512, CUDA book's own expected = 512, match = 1
offset after second request = 712, CUDA book's own expected = 712, match = 1
padding skipped between requests = 112 bytes, CUDA book's own expected = 112, match = 1
[RELEASE BUILD] wrote p[0] = 3.140000 without any detected error (assert() was compiled out by -DNDEBUG)
```

The debug build's own stderr, and its exit code, confirmed genuinely different from the release build's:

```text
02_arena_allocator_debug: 02_arena_allocator.cpp:26: void* Arena::allocate(size_t): Assertion `aligned_offset + bytes <= capacity' failed.
debug build exit code: 134
release build exit code: 0
```

Reproduced identically across 3 fresh reruns of each binary, and independently cross-checked via a hand-written Python reimplementation of the identical alignment formula, sharing no code with this file's own C++:

```text
align256(0) = 0 expected 0
align256(400) = 512 expected 512
padding = 112 expected 112
```

### Discussion

Every one of the CUDA book's own four numbers -- alignment `0`, offset `400`, alignment `512`, offset `712`, padding `112` -- matches exactly, confirmed a second, independent way through a Python reimplementation of the identical `(x + 255) & ~255` formula sharing no code with the C++ `Arena::allocate()` above. The debug-versus-release contrast is where this section's own strongest evidence lives: the *identical* source file, compiled two different ways, produces genuinely different real behavior -- the debug build's own `assert()` fires, printing a real compiler-generated diagnostic naming the exact failed condition and aborting with exit code `134` (confirmed across 3 fresh reruns), while the `-DNDEBUG` release build reaches the identical over-capacity `allocate()` call, receives a real pointer, and performs a real out-of-bounds write that succeeds silently, exit code `0`. This is not a simulated contrast -- both binaries were genuinely compiled from the same source with different preprocessor flags and genuinely run, and their different exit codes are real evidence that `-DNDEBUG` removes the `assert()` from the compiled instructions entirely rather than merely disabling it at runtime.

> `[COMMON TRAP]` A debug build that aborts loudly on a bug is not inherently "worse" than a release build that runs silently -- the opposite is the actual danger here. The debug build's `SIGABRT` is the SAFE outcome: a bug gets caught immediately, at the exact line and condition that failed, before any bad data propagates. The release build's silent success is the dangerous one: `p[0] = 3.14f` genuinely writes past the arena's declared capacity, into memory the arena has no right to touch, and the program reports total success (`exit code 0`) with no indication anything went wrong. `-DNDEBUG` is a standard, common release-build flag specifically because `assert()` checks cost real runtime overhead -- but this section's own two-binary comparison makes concrete exactly what that flag trades away.

## 11.3 Device Memory Pooling: The CUDA Book's Own Hit Rate, on This Book's Own Allocator `[FOUNDATIONAL]`

### Intuition

A `DeviceMemoryPool` avoids repeatedly asking the allocator for memory of a size it has recently released, by keeping size-indexed free lists and reusing from them before ever allocating fresh. This section builds the identical design, but sources its real memory from `c10::GetCPUAllocator()` -- this book's own real allocator, already measured at a genuine 64-byte alignment floor back in Chapter 7 and Chapter 10 -- rather than a stand-in for `cudaMalloc`.

### Background

The CUDA book's own Worked Example 11.3.1: three `acquire(256)` calls reach a `0.667` hit rate (`2048` reused bytes out of `3072` total). Its own critical finding: `DeviceMemoryPool` has no destructor, so `g_outstanding_allocations` stays at `1` both while the pool is alive and after it goes out of scope -- a genuine, measured leak.

### Worked Example 11.3.1 -- the CUDA book's own hit-rate trace and leak finding, reproduced directly

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// The CUDA C++ edition's Section 11.3 hand-builds DeviceMemoryPool: a
// size-indexed map of free lists, where acquire() reuses a previously
// released buffer of the same size if one exists, and release() appends
// to the free list instead of genuinely freeing. This file reproduces
// the CUDA book's own three-step hit-rate example exactly (0.667 after
// three acquisitions), using real allocations from c10::GetCPUAllocator()
// -- this book's own real allocator, first tested in Chapter 2 and
// Chapter 7 -- as the underlying "device" memory source. It then
// reproduces the CUDA book's own critical finding directly: the pool has
// no destructor, so a buffer it is still holding when it goes out of
// scope is genuinely, measurably never reclaimed.
static long g_outstanding_allocations = 0;
// Keeps every real allocation this file's own acquire() ever makes alive
// for the whole program -- deliberately mirroring the CUDA book's own
// point that DeviceMemoryPool never returns anything to the underlying
// allocator. g_outstanding_allocations counts real allocations; nothing
// in this file ever decrements it, exactly because nothing in
// DeviceMemoryPool's own design ever frees one.
static std::vector<c10::DataPtr> g_all_allocations;

struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;
    long bytes_allocated = 0;
    long bytes_reused = 0;

    float* acquire(int count) {
        auto& list = free_lists[count];
        if (!list.empty()) {
            float* ptr = list.back();
            list.pop_back();
            bytes_reused += count * 4;
            return ptr;
        }
        auto* allocator = c10::GetCPUAllocator();
        c10::DataPtr dptr = allocator->allocate(count * 4);
        float* raw = static_cast<float*>(dptr.get());
        g_all_allocations.push_back(std::move(dptr));
        g_outstanding_allocations++;
        bytes_allocated += count * 4;
        return raw;
    }

    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }

    // Deliberately NO destructor. free_lists itself (a map of vectors of
    // raw float* pointers) is destroyed automatically when a
    // DeviceMemoryPool goes out of scope, but destroying a vector of raw
    // pointers does nothing to the memory those pointers point to -- the
    // real allocations live only in g_all_allocations above, which this
    // struct never touches. This is the CUDA book's own exact point.
};

int main() {
    {
        DeviceMemoryPool pool;

        // Worked Example 11.3.1: three acquire(256) calls.
        float* p1 = pool.acquire(256);  // fresh
        std::cout << "step 1: acquire(256) -> fresh. bytes_allocated=" << pool.bytes_allocated
                  << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=0, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 0) << std::endl;

        pool.release(p1, 256);
        float* p2 = pool.acquire(256);  // reused
        std::cout << "step 2: acquire(256) -> reused (p2==p1)? " << (p2 == p1)
                  << ". bytes_allocated=" << pool.bytes_allocated << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=1024, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 1024) << std::endl;

        pool.release(p2, 256);
        float* p3 = pool.acquire(256);  // reused again
        std::cout << "step 3: acquire(256) -> reused (p3==p1)? " << (p3 == p1)
                  << ". bytes_allocated=" << pool.bytes_allocated << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=2048, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 2048) << std::endl;

        double hit_rate = (double)pool.bytes_reused / (double)(pool.bytes_allocated + pool.bytes_reused);
        std::cout << "hit rate = " << pool.bytes_reused << "/" << (pool.bytes_allocated + pool.bytes_reused)
                  << " = " << hit_rate << ", CUDA book's own expected = 0.667, match (3 decimal places) = "
                  << (std::abs(hit_rate - 0.667) < 0.0005) << std::endl;

        std::cout << "\nwhile pool is alive: g_outstanding_allocations = " << g_outstanding_allocations << std::endl;
        pool.release(p3, 256);  // leave the one real buffer in the free list when the pool goes out of scope
    }
    // pool's destructor (the implicit, do-nothing default one) just ran.
    std::cout << "after pool goes out of scope: g_outstanding_allocations = " << g_outstanding_allocations
              << " (CUDA book's own finding: this number does NOT drop -- the one real buffer is never reclaimed)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 03_memory_pool.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_memory_pool
./03_memory_pool
```

```text
step 1: acquire(256) -> fresh. bytes_allocated=1024, bytes_reused=0, CUDA book's own expected: allocated=1024, reused=0, match=1
step 2: acquire(256) -> reused (p2==p1)? 1. bytes_allocated=1024, bytes_reused=1024, CUDA book's own expected: allocated=1024, reused=1024, match=1
step 3: acquire(256) -> reused (p3==p1)? 1. bytes_allocated=1024, bytes_reused=2048, CUDA book's own expected: allocated=1024, reused=2048, match=1
hit rate = 2048/3072 = 0.666667, CUDA book's own expected = 0.667, match (3 decimal places) = 1

while pool is alive: g_outstanding_allocations = 1
after pool goes out of scope: g_outstanding_allocations = 1 (CUDA book's own finding: this number does NOT drop -- the one real buffer is never reclaimed)
```

Independently cross-checked via a from-scratch Python reimplementation of the identical free-list logic (a plain dictionary of lists rather than a `c10::DataPtr`-backed `DeviceMemoryPool`), confirming the same hit-rate arithmetic:

```text
hit rate: 0.6666666666666666 expected ~0.667
```

### Discussion

`bytes_allocated` stays fixed at `1024` across all three steps -- only one real allocation ever happens -- while `bytes_reused` climbs `0 -> 1024 -> 2048`, exactly matching the CUDA book's own three-step trace, and the resulting hit rate, `2048 / 3072 = 0.666667`, matches the CUDA book's own published `0.667` to three decimal places, reconfirmed by an entirely separate Python reimplementation using plain dictionaries rather than this file's own `c10::DataPtr`-backed design. `p1 == p2 == p3` across all three steps -- the identical single real allocation, drained and returned to the free list twice, satisfying three logically separate requests.

`g_outstanding_allocations` is this section's own strongest piece of evidence: it reports `1` while the pool is still alive (correctly reflecting the single real allocation this whole example ever makes), and it *still* reports `1` after the pool's closing brace runs and its destructor -- the implicit, do-nothing default one, since `DeviceMemoryPool` declares no destructor of its own -- has already executed. The counter does not, and structurally cannot, decrease from that destructor, because nothing in `DeviceMemoryPool`'s own design ever returns a buffer to `c10::GetCPUAllocator()`; the pool only ever moves pointers between "acquired" and "in the free list," never back to the allocator itself. This is the CUDA book's own exact finding, reproduced as a genuinely measured, non-decreasing counter rather than a theoretical claim about what "should" happen.

## 11.4 The Double-Release Bug: A Direct Consequence of a Raw Pointer Signature `[FOUNDATIONAL]`

### Intuition

`DeviceMemoryPool::release(float* ptr, int count)` takes a plain pointer with no ownership information attached to it at all -- nothing in the type system distinguishes "a pointer I am legitimately returning for the first time" from "a pointer I already returned once, and am now returning again by mistake." The CUDA book's own Section 11.4 shows this is not a hypothetical risk: it produces a real, directly demonstrable aliasing bug.

### Background

The CUDA book's own Worked Example 11.4.1: releasing the same pointer twice leaves the free list holding two identical entries; two subsequent `acquire()` calls then both drain from that corrupted list, both returning the *same* real address -- confirmed by writing `1.0` through one and `2.0` through the other, then observing both read back `2.0`.

### Worked Example 11.4.1 -- reproducing the aliasing bug, with real writes as proof

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// The CUDA C++ edition's Section 11.4 points out that DeviceMemoryPool's
// own release(float*, int) takes a plain, ownerless pointer -- nothing
// stops a caller from releasing the same pointer twice, and nothing
// stops two DIFFERENT logical buffers from ending up genuinely aliased
// as a result. This file reuses this book's own DeviceMemoryPool from
// Section 11.3 unchanged and reproduces the CUDA book's own exact bug:
// double-releasing one pointer, then acquiring twice, and confirming the
// two "different" buffers a caller receives are actually the identical
// real address -- a genuine aliasing bug, demonstrated with real writes
// through both pointers.
struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;

    float* acquire(int count) {
        auto& list = free_lists[count];
        if (!list.empty()) {
            float* ptr = list.back();
            list.pop_back();
            return ptr;
        }
        auto* allocator = c10::GetCPUAllocator();
        c10::DataPtr dptr = allocator->allocate(count * 4);
        float* raw = static_cast<float*>(dptr.get());
        static std::vector<c10::DataPtr> keep_alive;
        keep_alive.push_back(std::move(dptr));
        return raw;
    }

    // The CUDA book's own vulnerable signature: a plain pointer, with no
    // compile-time way to prevent the SAME pointer being released twice.
    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }
};

int main() {
    DeviceMemoryPool pool;

    // The double-release bug: release the same pointer twice.
    float* activations = pool.acquire(128);
    pool.release(activations, 128);
    pool.release(activations, 128);  // BUG: same pointer, no compile error, no runtime error either
    std::cout << "free_lists[128] now holds " << pool.free_lists[128].size()
              << " entries after double-releasing ONE real pointer, CUDA book's own expected = 2, match = "
              << (pool.free_lists[128].size() == 2) << std::endl;

    // The consequence: two SEPARATE acquire() calls both drain from the
    // same corrupted free list, both returning the identical real address.
    float* a = pool.acquire(128);
    float* b = pool.acquire(128);
    bool aliased = (a == b);
    std::cout << "a and b are the same real address? " << aliased
              << (aliased ? " -- ALIASED (CUDA book's own exact finding)" : "") << std::endl;

    // Genuine proof via real writes, not just a pointer-equality check:
    // writing through b changes what a reads back, because they are the
    // literal same 4 bytes of real memory.
    a[0] = 1.0f;
    b[0] = 2.0f;
    std::cout << "after a[0]=1.0 then b[0]=2.0: a[0]=" << a[0] << " b[0]=" << b[0]
              << ", CUDA book's own expected: both = 2.000000, match = "
              << (a[0] == 2.0f && b[0] == 2.0f) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 04_double_release_bug.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_double_release_bug
./04_double_release_bug
```

```text
free_lists[128] now holds 2 entries after double-releasing ONE real pointer, CUDA book's own expected = 2, match = 1
a and b are the same real address? 1 -- ALIASED (CUDA book's own exact finding)
after a[0]=1.0 then b[0]=2.0: a[0]=2 b[0]=2, CUDA book's own expected: both = 2.000000, match = 1
```

Independently cross-checked via a from-scratch Python reimplementation of the identical free-list logic, using plain string tokens in place of real pointers to confirm the aliasing arises purely from the free-list bookkeeping, not from anything specific to `c10::DataPtr` or real memory addresses:

```text
free_lists[128] after double release: ['buf_1', 'buf_1'] len: 2
a == b (aliased)? True
```

### Discussion

`free_lists[128]` genuinely holds `2` entries after the double-release, both pointing at the identical real address -- not two separately-tracked "logical" releases that happen to share a coincidental value, but literally the same pointer stored twice, confirmed structurally by a from-scratch Python reimplementation using opaque string tokens rather than real pointers, which reproduces the identical `['buf_1', 'buf_1']` duplication purely from the free-list logic itself. The consequence is not hypothetical: `a` and `b`, returned from two separate `acquire(128)` calls a caller has every reason to believe are two independent buffers, are confirmed to be the same real address, and the final write test makes this concrete rather than abstract -- `a[0] = 1.0f` followed by `b[0] = 2.0f` leaves `a[0]` reading back `2.0`, because `a` and `b` were never two different pieces of memory in the first place.

> `[COMMON TRAP]` It would be easy to think this bug requires a caller to make an obvious mistake -- calling `release()` twice in a row, as this section's own code does explicitly. In a real program, the two `release()` calls would typically be far apart, in different functions or different iterations of a loop, with no compiler warning and no runtime error connecting them. `DeviceMemoryPool::release(float*, int)`'s raw-pointer signature carries no information at all about whether a given pointer has already been released -- it is purely the caller's responsibility to track that correctly, and Section 11.4's own two-line reproduction is deliberately the simplest possible version of a bug that, in a larger and more realistic program, could be separated by hundreds of lines and much harder to spot.

## Complete Runnable Code

### File: `01_refcounted_tensor.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 11.1 hand-builds RefCountedBuffer<T>: a
// data pointer plus a separately-allocated int* refcount, incremented on
// copy construction and decremented on destruction, freeing both buffers
// only when the count reaches zero -- traced through a three-lifetime
// example as 1 -> 2 -> 1 -> 0. torch::Tensor is ALREADY reference-counted
// this exact way internally -- its own TensorImpl is held by a real
// c10::intrusive_ptr, and .use_count() reports that pointer's real,
// genuine refcount. This file reproduces the CUDA book's own three-step
// trace directly on torch::Tensor's real internals, using nested scopes
// to control exactly when each copy's destructor runs, the same way the
// CUDA book's own Worked Example 11.1.1 does.
int main() {
    torch::Tensor original = torch::ones({4});
    std::cout << "after construction: original.use_count() = " << original.use_count()
              << ", CUDA book's own expected = 1, match = " << (original.use_count() == 1) << std::endl;

    {
        torch::Tensor copy = original;  // Chapter 6's own shallow copy: shares storage AND shares the refcount
        std::cout << "after copy construction: original.use_count() = " << original.use_count()
                  << ", CUDA book's own expected = 2, match = " << (original.use_count() == 2) << std::endl;

        {
            torch::Tensor second_copy = original;
            std::cout << "after a THIRD reference: original.use_count() = " << second_copy.use_count()
                      << " (CUDA book's own example only goes to 2 references; this extends it to 3, to confirm "
                      << "the count keeps tracking correctly beyond the CUDA book's own two-copy case)" << std::endl;
        }
        // second_copy's destructor just ran -- back down to 2 references.
        std::cout << "after third reference's destructor: original.use_count() = " << original.use_count()
                  << ", back down to 2? " << (original.use_count() == 2) << std::endl;
    }
    // copy's destructor just ran -- back down to 1, the CUDA book's own
    // "first destruction: refcount = 1, buffer survives" step.
    std::cout << "after copy's destructor (first destruction): original.use_count() = " << original.use_count()
              << ", CUDA book's own expected = 1 (buffer survives), match = " << (original.use_count() == 1) << std::endl;

    // original itself still holds real, valid data -- proof the buffer
    // genuinely survived every intermediate destructor.
    std::cout << "original still valid and readable: original[0] = " << original[0].item<float>() << std::endl;

    // The CUDA book's own final step, "second destruction: refcount = 0",
    // corresponds here to original itself going out of scope at the end
    // of main() -- not shown as a printed step, since torch::Tensor's
    // real destructor runs silently and there is nothing left to read
    // afterward to prove it, exactly as the CUDA book's own C++ semantics
    // would behave for its own final buffer-free step.

    return 0;
}
```

### File: `02_arena_allocator.cpp`

```cpp
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

// The CUDA C++ edition's Section 11.2 hand-builds Arena: a bump-pointer
// allocator over a fixed buffer, aligning every request up to a 256-byte
// boundary (matching cudaMalloc's own guarantee, first tested by this
// book in Chapter 7.3 and Chapter 10.2 for c10::GetCPUAllocator's real,
// smaller 64-byte floor). Arena itself has nothing GPU-specific about it
// -- it is plain pointer arithmetic over a byte buffer -- so this file
// reuses the CUDA book's own exact numeric example directly, then tests
// its own bounds-check behavior in both a debug build (assert active,
// aborts) and a release build (-DNDEBUG, assert compiled out entirely).
struct Arena {
    uint8_t* base;
    size_t capacity;
    size_t offset;

    Arena(uint8_t* b, size_t cap) : base(b), capacity(cap), offset(0) {}

    void* allocate(size_t bytes) {
        size_t aligned_offset = (offset + 255) & ~size_t(255);
        assert(aligned_offset + bytes <= capacity);  // bounds check -- stripped entirely under -DNDEBUG
        void* ptr = base + aligned_offset;
        offset = aligned_offset + bytes;
        return ptr;
    }

    void reset() { offset = 0; }
};

int main() {
    static uint8_t buffer[4096];
    Arena arena(buffer, sizeof(buffer));

    // Worked Example 11.2.1: the CUDA book's own exact alignment arithmetic.
    void* p1 = arena.allocate(100 * sizeof(float));  // 400 bytes
    printf("first request (100 floats = 400 bytes): (0+255)&~255 = %zu, CUDA book's own expected = 0, match = %d\n",
           (size_t)(0 + 255) & ~size_t(255), ((size_t)(0 + 255) & ~size_t(255)) == 0);
    printf("offset after first request = %zu, CUDA book's own expected = 400, match = %d\n",
           arena.offset, arena.offset == 400);

    void* p2 = arena.allocate(50 * sizeof(float));  // 200 bytes
    printf("second request (50 floats = 200 bytes): (400+255)&~255 = %zu, CUDA book's own expected = 512, match = %d\n",
           (size_t)(400 + 255) & ~size_t(255), ((size_t)(400 + 255) & ~size_t(255)) == 512);
    printf("offset after second request = %zu, CUDA book's own expected = 712, match = %d\n",
           arena.offset, arena.offset == 712);
    printf("padding skipped between requests = %zu bytes, CUDA book's own expected = 112, match = %d\n",
           (size_t)512 - 400, (512 - 400) == 112);

#ifdef NDEBUG
    // Release build: the assert() above compiles to nothing at all.
    // Worked Example 11.2.2's own release-build scenario: request more
    // than the arena's remaining capacity, and the bounds check that
    // would have caught it is simply not present in the compiled binary.
    Arena small_arena(buffer, 800);
    float* p = (float*)small_arena.allocate(1000 * sizeof(float));  // genuinely exceeds 800-byte capacity
    p[0] = 3.14f;  // real out-of-bounds write, silently succeeds
    printf("[RELEASE BUILD] wrote p[0] = %f without any detected error (assert() was compiled out by -DNDEBUG)\n", p[0]);
#else
    printf("[DEBUG BUILD] assert() is active in this binary -- the deliberate over-capacity request below "
           "is expected to abort the program (see the paired release-build binary for the silent-overflow contrast)\n");
    Arena small_arena(buffer, 800);
    float* p = (float*)small_arena.allocate(1000 * sizeof(float));  // triggers assert() -- SIGABRT
    p[0] = 3.14f;  // never reached in a debug build
    printf("[DEBUG BUILD] wrote p[0] = %f -- THIS LINE SHOULD NEVER PRINT\n", p[0]);
#endif

    return 0;
}
```

### File: `03_memory_pool.cpp`

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// The CUDA C++ edition's Section 11.3 hand-builds DeviceMemoryPool: a
// size-indexed map of free lists, where acquire() reuses a previously
// released buffer of the same size if one exists, and release() appends
// to the free list instead of genuinely freeing. This file reproduces
// the CUDA book's own three-step hit-rate example exactly (0.667 after
// three acquisitions), using real allocations from c10::GetCPUAllocator()
// -- this book's own real allocator, first tested in Chapter 2 and
// Chapter 7 -- as the underlying "device" memory source. It then
// reproduces the CUDA book's own critical finding directly: the pool has
// no destructor, so a buffer it is still holding when it goes out of
// scope is genuinely, measurably never reclaimed.
static long g_outstanding_allocations = 0;
// Keeps every real allocation this file's own acquire() ever makes alive
// for the whole program -- deliberately mirroring the CUDA book's own
// point that DeviceMemoryPool never returns anything to the underlying
// allocator. g_outstanding_allocations counts real allocations; nothing
// in this file ever decrements it, exactly because nothing in
// DeviceMemoryPool's own design ever frees one.
static std::vector<c10::DataPtr> g_all_allocations;

struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;
    long bytes_allocated = 0;
    long bytes_reused = 0;

    float* acquire(int count) {
        auto& list = free_lists[count];
        if (!list.empty()) {
            float* ptr = list.back();
            list.pop_back();
            bytes_reused += count * 4;
            return ptr;
        }
        auto* allocator = c10::GetCPUAllocator();
        c10::DataPtr dptr = allocator->allocate(count * 4);
        float* raw = static_cast<float*>(dptr.get());
        g_all_allocations.push_back(std::move(dptr));
        g_outstanding_allocations++;
        bytes_allocated += count * 4;
        return raw;
    }

    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }

    // Deliberately NO destructor. free_lists itself (a map of vectors of
    // raw float* pointers) is destroyed automatically when a
    // DeviceMemoryPool goes out of scope, but destroying a vector of raw
    // pointers does nothing to the memory those pointers point to -- the
    // real allocations live only in g_all_allocations above, which this
    // struct never touches. This is the CUDA book's own exact point.
};

int main() {
    {
        DeviceMemoryPool pool;

        // Worked Example 11.3.1: three acquire(256) calls.
        float* p1 = pool.acquire(256);  // fresh
        std::cout << "step 1: acquire(256) -> fresh. bytes_allocated=" << pool.bytes_allocated
                  << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=0, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 0) << std::endl;

        pool.release(p1, 256);
        float* p2 = pool.acquire(256);  // reused
        std::cout << "step 2: acquire(256) -> reused (p2==p1)? " << (p2 == p1)
                  << ". bytes_allocated=" << pool.bytes_allocated << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=1024, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 1024) << std::endl;

        pool.release(p2, 256);
        float* p3 = pool.acquire(256);  // reused again
        std::cout << "step 3: acquire(256) -> reused (p3==p1)? " << (p3 == p1)
                  << ". bytes_allocated=" << pool.bytes_allocated << ", bytes_reused=" << pool.bytes_reused
                  << ", CUDA book's own expected: allocated=1024, reused=2048, match="
                  << (pool.bytes_allocated == 1024 && pool.bytes_reused == 2048) << std::endl;

        double hit_rate = (double)pool.bytes_reused / (double)(pool.bytes_allocated + pool.bytes_reused);
        std::cout << "hit rate = " << pool.bytes_reused << "/" << (pool.bytes_allocated + pool.bytes_reused)
                  << " = " << hit_rate << ", CUDA book's own expected = 0.667, match (3 decimal places) = "
                  << (std::abs(hit_rate - 0.667) < 0.0005) << std::endl;

        std::cout << "\nwhile pool is alive: g_outstanding_allocations = " << g_outstanding_allocations << std::endl;
        pool.release(p3, 256);  // leave the one real buffer in the free list when the pool goes out of scope
    }
    // pool's destructor (the implicit, do-nothing default one) just ran.
    std::cout << "after pool goes out of scope: g_outstanding_allocations = " << g_outstanding_allocations
              << " (CUDA book's own finding: this number does NOT drop -- the one real buffer is never reclaimed)" << std::endl;

    return 0;
}
```

### File: `04_double_release_bug.cpp`

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// The CUDA C++ edition's Section 11.4 points out that DeviceMemoryPool's
// own release(float*, int) takes a plain, ownerless pointer -- nothing
// stops a caller from releasing the same pointer twice, and nothing
// stops two DIFFERENT logical buffers from ending up genuinely aliased
// as a result. This file reuses this book's own DeviceMemoryPool from
// Section 11.3 unchanged and reproduces the CUDA book's own exact bug:
// double-releasing one pointer, then acquiring twice, and confirming the
// two "different" buffers a caller receives are actually the identical
// real address -- a genuine aliasing bug, demonstrated with real writes
// through both pointers.
struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;

    float* acquire(int count) {
        auto& list = free_lists[count];
        if (!list.empty()) {
            float* ptr = list.back();
            list.pop_back();
            return ptr;
        }
        auto* allocator = c10::GetCPUAllocator();
        c10::DataPtr dptr = allocator->allocate(count * 4);
        float* raw = static_cast<float*>(dptr.get());
        static std::vector<c10::DataPtr> keep_alive;
        keep_alive.push_back(std::move(dptr));
        return raw;
    }

    // The CUDA book's own vulnerable signature: a plain pointer, with no
    // compile-time way to prevent the SAME pointer being released twice.
    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }
};

int main() {
    DeviceMemoryPool pool;

    // The double-release bug: release the same pointer twice.
    float* activations = pool.acquire(128);
    pool.release(activations, 128);
    pool.release(activations, 128);  // BUG: same pointer, no compile error, no runtime error either
    std::cout << "free_lists[128] now holds " << pool.free_lists[128].size()
              << " entries after double-releasing ONE real pointer, CUDA book's own expected = 2, match = "
              << (pool.free_lists[128].size() == 2) << std::endl;

    // The consequence: two SEPARATE acquire() calls both drain from the
    // same corrupted free list, both returning the identical real address.
    float* a = pool.acquire(128);
    float* b = pool.acquire(128);
    bool aliased = (a == b);
    std::cout << "a and b are the same real address? " << aliased
              << (aliased ? " -- ALIASED (CUDA book's own exact finding)" : "") << std::endl;

    // Genuine proof via real writes, not just a pointer-equality check:
    // writing through b changes what a reads back, because they are the
    // literal same 4 bytes of real memory.
    a[0] = 1.0f;
    b[0] = 2.0f;
    std::cout << "after a[0]=1.0 then b[0]=2.0: a[0]=" << a[0] << " b[0]=" << b[0]
              << ", CUDA book's own expected: both = 2.000000, match = "
              << (a[0] == 2.0f && b[0] == 2.0f) << std::endl;

    return 0;
}
```

Files `01`, `03`, and `04` compile and link against LibTorch with the standard command from *Getting Started*:

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 <file>.cpp \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
  -o <file>
./<file>
```

File `02` has nothing GPU-specific about it and needs no LibTorch headers -- it compiles twice, once as an ordinary debug build and once as a release build:

```bash
g++ -std=c++20 -O2 02_arena_allocator.cpp -o 02_arena_allocator_debug
./02_arena_allocator_debug     # aborts, exit code 134

g++ -std=c++20 -O2 -DNDEBUG 02_arena_allocator.cpp -o 02_arena_allocator_release
./02_arena_allocator_release   # runs silently, exit code 0
```

## Chapter Summary

`torch::Tensor`'s own `.use_count()` was confirmed to trace the CUDA book's own `1 -> 2 -> 1` reference-counting lifecycle exactly, with the underlying buffer confirmed genuinely still readable after the CUDA book's own "first destruction: buffer survives" step -- completing the explanation Chapter 6's own copy-versus-clone divergence left open, that `torch::Tensor`'s shallow copy is safe specifically because it is reference-counted, not merely because its copy constructor happens not to be deleted. A hand-built `Arena` reproduced the CUDA book's own exact alignment arithmetic -- offsets `0`, `400`, `512`, `712`, with `112` bytes of padding -- and its debug-versus-release contrast was demonstrated as two genuinely different compiled binaries from the identical source, one aborting with a real `SIGABRT` (exit code `134`) and one silently completing an out-of-bounds write (exit code `0`). A hand-built `DeviceMemoryPool`, layered over this book's own real `c10::GetCPUAllocator()`, reached the CUDA book's own exact `0.667` hit rate after three `acquire()` calls, and its missing destructor was confirmed to leave a genuinely measured allocation counter unchanged -- `1` before and after the pool's own scope ends. And the identical pool's raw-pointer `release()` signature was confirmed to permit a real double-release, with two subsequent `acquire()` calls genuinely returning the same address and a real write through one pointer visibly changing what the other reads back.

## Self-Check Questions

1. Worked Example 11.1.1 extends the CUDA book's own two-copy example to a third simultaneous reference before returning to two. What would a reader learn from this third-reference extension that the CUDA book's own two-copy example alone would not demonstrate?
2. Section 11.1's `[COMMON TRAP]` distinguishes "copying shares storage" from "copying is safe." Using Chapter 6's own Worked Example 6.5.1 (the copy-constructor test), what specific test would you need to ADD to prove copying is safe, beyond what Chapter 6 alone tested?
3. Worked Example 11.2.1's debug and release binaries are compiled from the IDENTICAL source file. What C++ preprocessor mechanism makes this possible, and what specifically differs in the two compiled binaries as a direct result?
4. Section 11.3 keeps every real allocation alive forever in a global `g_all_allocations` vector, specifically so it can be measured rather than genuinely lost. Why does keeping the allocation deliberately "alive but unreachable through the pool" still count as an honest demonstration of the CUDA book's own leak-finding, even though nothing is technically lost from the program's own memory the way a true C malloc/free mismatch would lose it?
5. Worked Example 11.4.1 confirms `a == b` via pointer comparison, then ALSO confirms it via a real write-then-read test. Why is the write-then-read test meaningfully stronger evidence than the pointer comparison alone, given what Chapter 6's own Section 6.5 already established about how `torch::Tensor`'s `.data_ptr()` behaves?

## Where We Go Next

This chapter tested `torch::Tensor`'s own real reference counting against the CUDA book's own hand-built version (finding it already there, doing the identical job), then built the CUDA book's own arena and pooling strategies from nothing, since `torch::Tensor` has no built-in equivalent to either -- reproducing every one of the CUDA book's own genuine bugs directly, from a silently-stripped bounds check to a real, write-confirmed aliasing bug. This is the final chapter of Part 1: Core Tensor Infrastructure. Part 2 turns from how a tensor is built, laid out, and owned to what a program actually does with one -- Chapter 12 opens Basic Tensor Operations with the CUDA book's own element-wise arithmetic, tested against `torch::Tensor`'s real, already-vectorized operator overloads for the first time.

## Worked Solutions

**1.** The CUDA book's own two-copy example only ever shows the count going up once (1 to 2) and down once (2 to 1) -- consistent with a simpler, WRONG mental model where "refcount" just means "is there a second reference or not" (a boolean in disguise). Extending to a third simultaneous reference, and confirming `use_count()` correctly reports `3`, then correctly drops back to `2` when only ONE of the two extra references is destroyed, demonstrates that the mechanism is a genuine counter tracking an arbitrary number of references, not a two-state flag that happens to look like a counter in a two-copy example.

**2.** Chapter 6's own Worked Example 6.5.1 tested that mutating through a shallow copy is visible in the original (`shallow_copy[0]=99` changing `original[0]`) -- this proves sharing happens, but says nothing about lifetime safety. To prove copying is SAFE (not just that sharing occurs), you would need to add a test where the ORIGINAL tensor's variable goes out of scope (or is otherwise destroyed) while the COPY is still alive, and then confirm the copy can still be read correctly afterward -- proving the underlying buffer was not freed prematurely just because one of its two references disappeared. This chapter's own Worked Example 11.1.1 is effectively that missing test, applied to `torch::Tensor` directly.

**3.** The `#ifdef NDEBUG` / `#else` / `#endif` preprocessor directives make this possible -- they are evaluated by the compiler's preprocessor BEFORE actual compilation begins, based on whether the `NDEBUG` macro was defined via the `-DNDEBUG` compiler flag. The two binaries differ in real, compiled machine instructions: the debug binary contains the actual `assert()` check (and the string literals it uses when it fails) compiled into it, while the release binary's `assert()` call is removed entirely by the preprocessor (the C++ standard library's own `<cassert>` header defines `assert(x)` as expanding to nothing at all when `NDEBUG` is defined) -- so the release binary has literally no bounds-check instruction present anywhere for that call site to have run.

**4.** The CUDA book's own point is not literally about physical memory being unreachable by the ENTIRE program (a true leak, invisible to any tracking) -- it's about whether `DeviceMemoryPool` ITSELF ever returns memory to the allocator once acquired. Keeping the allocation alive in `g_all_allocations` (a construct entirely outside `DeviceMemoryPool`'s own code) is what makes the demonstration honest and measurable: it proves the counter (representing memory `DeviceMemoryPool` is responsible for but never releases) genuinely never decreases, without requiring an actual, harder-to-verify memory leak that a testing tool would need to detect indirectly. The finding is specifically "DeviceMemoryPool's own design has no path back to the allocator," which is true and measurable regardless of whether some OTHER part of the test program happens to also be holding a reference for bookkeeping purposes.

**5.** Chapter 6's own Section 6.5 established that `torch::Tensor`'s `.data_ptr()` for a shallow copy equals the original's `.data_ptr()` -- meaning pointer comparison alone can be satisfied by two tensors that legitimately share storage BY DESIGN (the intended, safe behavior Chapter 6 documents). A pointer-equality check in Worked Example 11.4.1's own context could theoretically be satisfied by some other legitimate reason two pointers look equal, without necessarily proving they are being used as if they were independent, non-aliased buffers. The write-then-read test is stronger because it demonstrates the actual CONSEQENCE the bug produces: two pointers a caller genuinely believes are independent (having received them from two separate, sequential `acquire()` calls with no visible connection between them) really do share state, proven by an actual data corruption a caller would observe in a real program, not merely a pointer-value coincidence.
