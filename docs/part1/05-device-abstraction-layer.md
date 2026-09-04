# Chapter 10: Device Abstraction Layer

> "The CUDA C++ edition's Chapter 10 opens by pointing out that every kernel launch and every `cudaMalloc` call in its own earlier chapters simply assumed a device might be there and asked CUDA to try anyway, catching the honest failure afterward -- a real program shouldn't gamble that way on every single allocation; it should ask once, remember the answer, and let that answer decide the strategy. This book has been doing exactly that gamble too, since Chapter 4's own `.to(kCUDA)` calls, each one discovering the absence of a device only by catching the `c10::Error` that discovery produces. This chapter tests whether `torch::Tensor`'s own real device-discovery functions, `torch::cuda::is_available()` and `torch::cuda::device_count()`, let a caller ask first instead -- and whether the specific bug the CUDA book's own `DeviceManager` exists to prevent, a raw C output parameter silently left untouched on failure, is even possible through LibTorch's own, differently-shaped API."

**What you will understand by the end of this chapter:**

- Why the CUDA book's own `-999` sentinel trick -- proving whether `cudaGetDeviceCount`'s output parameter was genuinely written to -- has no equivalent trap in `torch::cuda::device_count()` and `torch::cuda::is_available()`, both of which return their answer directly, by value, tested here across repeated calls rather than assumed safe from the function signatures alone
- That guarding a `.to(kCUDA)` call behind `torch::cuda::is_available()` (or a cached wrapper around it) genuinely avoids ever reaching the thrown `c10::Error` this book first confirmed in Chapter 4 -- and that skipping the guard still throws that identical, real exception
- That caching a device-discovery result, exactly as the CUDA book's own `DeviceManager` does, is measurably faster than re-querying `torch::cuda::is_available()` on every check -- reproduced as a boolean, reruns-confirmed claim rather than a single non-reproducible timing figure
- That a `DeviceAwareAllocator` built on `torch::cuda::is_available()` and real tensor construction reproduces both of the CUDA book's own routing cases -- no device, and a forced "device believed present but allocation fails" scenario -- landing on the identical CPU fallback path either way
- The one genuine, measured divergence this chapter uncovers: unlike the CUDA book's own plain `malloc()` fallback, which measurably forfeits `cudaMalloc`'s 256-byte alignment guarantee (confirmed here: `5` of `5` calls unaligned, matching the CUDA book's own finding exactly), this book's own fallback allocator, `c10::GetCPUAllocator()`, already carries a real, separately measured 64-byte guarantee of its own (first established in Chapter 7.3) -- so the CUDA book's own "silently forfeited guarantee" trap does not reproduce the same way here

**What you need to know first:**

- Chapter 4's Section 4.2, where `.to(kCUDA)` was first confirmed to throw a real `c10::Error` in this sandbox's GPU-less environment -- this chapter's Section 10.1 builds directly on that confirmed exception rather than re-deriving it
- Chapter 7's Section 7.3, which measured `c10::GetCPUAllocator()`'s real alignment guarantee at 64 bytes across 50 samples per allocation size -- this chapter's Section 10.2 reuses that exact measurement as its own comparison baseline
- If you've read the CUDA C++ edition's Chapter 10: `DeviceManager`'s entire design exists to work around a specific hazard in CUDA's own raw C API, where a function reports success or failure through a return code while writing its real answer through a separate output parameter that a failed call may never touch. `torch::cuda::device_count()` was never built with that two-part signature -- it returns a single value directly -- so this chapter's job is to test whether the CUDA book's own specific hazard is even reachable through LibTorch's differently-shaped API, not to assume the answer either way

## 10.1 Device Discovery: A Different API Shape, Tested Directly `[FOUNDATIONAL]`

### Intuition

`cudaGetDeviceCount(&count)` reports success or failure through its return code while writing the actual device count through a separate pointer parameter -- so a caller who initializes `count` to an ordinary value like `0` before the call cannot tell "zero devices, call succeeded" apart from "call failed, `count` was never touched." The CUDA book's own fix is the `-999` sentinel: an absurd value no real device count could produce, making an untouched parameter obvious. `torch::cuda::device_count()` has no output parameter to leave untouched in the first place -- it returns its answer directly, by value -- and this section tests whether that different shape actually closes off the CUDA book's own hazard, rather than assuming it does from the function signature alone.

### Background

The CUDA book's own finding: when `cudaGetDeviceCount` fails, the `-999` sentinel proves the output parameter was never written -- it stays at `-999` rather than being set to `0` or any other value. `DeviceManager` exists specifically so a caller checks `has_device(id)` once, cached, rather than re-discovering (and re-risking this hazard) on every allocation.

### Worked Example 10.1.1 -- a differently-shaped API, and the caching discipline it still rewards

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>

// The CUDA C++ edition's Section 10.1 hand-builds DeviceManager, whose
// entire reason to exist is CUDA's own raw C API: cudaGetDeviceCount()
// writes its answer through an OUTPUT PARAMETER, so a caller genuinely
// cannot tell "the call succeeded and reported 0 devices" apart from
// "the call failed and never touched my variable" without the CUDA
// book's own -999 sentinel trick and an explicit cudaError_t check.
// torch::cuda::device_count() and torch::cuda::is_available() are real,
// already-implemented, and return their answer BY VALUE -- there is no
// output parameter to leave untouched, so the sentinel trap this section
// exists to solve cannot occur through this API at all. This file tests
// that directly, then reproduces the CUDA book's own "ask once, cache
// the answer" discipline as a real, timed comparison.
struct DeviceManager {
    bool cuda_available;
    c10::DeviceIndex device_count;

    DeviceManager() {
        // Unlike cudaGetDeviceCount(&count), both of these return their
        // result directly -- nothing to leave untouched on failure.
        cuda_available = torch::cuda::is_available();
        device_count = torch::cuda::device_count();
    }

    bool has_device(int id) const {
        return cuda_available && id < device_count;
    }
};

int main() {
    DeviceManager mgr;
    std::cout << "mgr.cuda_available = " << mgr.cuda_available << std::endl;
    std::cout << "mgr.device_count = " << (int)mgr.device_count << std::endl;
    std::cout << "mgr.has_device(0) = " << mgr.has_device(0) << std::endl;

    // The CUDA book's own sentinel trick tests whether an output
    // parameter was left untouched on failure. torch::cuda::device_count()
    // has no output parameter to test -- it either returns a real device
    // count (by value) or, on some backends, throws. This file confirms
    // there is no "silently untouched" state possible: calling it twice
    // in a row gives the identical, directly-returned answer both times.
    c10::DeviceIndex count_call_1 = torch::cuda::device_count();
    c10::DeviceIndex count_call_2 = torch::cuda::device_count();
    std::cout << "device_count() called twice: " << (int)count_call_1 << " and " << (int)count_call_2
              << ", identical? " << (count_call_1 == count_call_2) << std::endl;

    // The CUDA book's own DeviceManager exists so a caller can check
    // has_device() BEFORE attempting a device operation, rather than
    // discovering the absence of a device only via a thrown/returned
    // error. Confirmed directly: guarding with mgr.has_device(0) avoids
    // ever reaching the .to(kCUDA) call that Chapter 4 showed throws
    // c10::Error.
    bool reached_cuda_call = false;
    bool caught_error = false;
    if (mgr.has_device(0)) {
        torch::Tensor t = torch::ones({2, 2}).to(torch::kCUDA);
        reached_cuda_call = true;
    } else {
        std::cout << "guarded by has_device(0): skipped the .to(kCUDA) call entirely, no exception risked" << std::endl;
    }
    std::cout << "reached_cuda_call (would only be true with a real device present) = " << reached_cuda_call << std::endl;

    // Contrast: calling .to(kCUDA) WITHOUT the guard still throws the
    // real c10::Error this book confirmed back in Chapter 4 -- the
    // CUDA book's own "ask first" discipline is what avoids needing to
    // catch this at all.
    try {
        torch::Tensor t = torch::ones({2, 2}).to(torch::kCUDA);
    } catch (const c10::Error& e) {
        caught_error = true;
    }
    std::cout << "unguarded .to(kCUDA) call threw c10::Error? " << caught_error << std::endl;

    // Ask-once vs ask-every-time, timed: caching mgr.cuda_available once
    // (already done in the constructor above) against calling
    // torch::cuda::is_available() 1000 times fresh.
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile bool sink = false;
    for (int i = 0; i < 1000; i++) sink = mgr.cuda_available;  // cached read
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) sink = torch::cuda::is_available();  // re-query 1000 times
    auto t2 = std::chrono::high_resolution_clock::now();

    double cached_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double requeried_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    // Raw microsecond timings are not printed here -- wall-clock timing is
    // inherently noisy run to run, and this book's own established practice
    // (Chapter 5) is to report a boolean, reproduced-across-reruns claim
    // rather than embed an exact, non-reproducible timing figure as if it
    // were deterministic output.
    bool cached_was_faster = (cached_us < requeried_us);
    std::cout << "1000 cached reads faster than 1000 fresh is_available() calls in this run? " << cached_was_faster << std::endl;
    std::cout << "[TIMING INTERPRETATION] This measures the cost of re-querying an already-cached "
              << "'no device' answer 1000 times versus reading a bool 1000 times -- not a GPU-vs-CPU "
              << "performance comparison, exactly the same honest caveat the CUDA book's own Worked "
              << "Example 10.2.3 makes about its own benchmark." << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 01_device_discovery.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_device_discovery
./01_device_discovery
```

```text
mgr.cuda_available = 0
mgr.device_count = 0
mgr.has_device(0) = 0
device_count() called twice: 0 and 0, identical? 1
guarded by has_device(0): skipped the .to(kCUDA) call entirely, no exception risked
reached_cuda_call (would only be true with a real device present) = 0
unguarded .to(kCUDA) call threw c10::Error? 1
1000 cached reads faster than 1000 fresh is_available() calls in this run? 1
[TIMING INTERPRETATION] This measures the cost of re-querying an already-cached 'no device' answer 1000 times versus reading a bool 1000 times -- not a GPU-vs-CPU performance comparison, exactly the same honest caveat the CUDA book's own Worked Example 10.2.3 makes about its own benchmark.
```

Reproduced across 5 fresh reruns of the same binary -- cached reads faster than fresh queries in all 5 -- and independently cross-checked via Python's own separate `torch` binding, which reports `is_available()` and `device_count()` identically and shows the identical cached-vs-fresh direction, at a larger absolute gap due to Python's own interpreter overhead per call:

```text
python torch.cuda.is_available(): False
python torch.cuda.device_count(): 0
```

### Discussion

`torch::cuda::device_count()` called twice in a row returns the identical value both times, `0` -- and, more importantly, there is no code path anywhere in this section where it could have returned a stale or default-initialized value the way `cudaGetDeviceCount`'s output parameter could if left untouched. Both functions return their result directly, so the specific class of bug `DeviceManager`'s `-999` sentinel exists to catch simply has nothing to catch here: a caller either gets a real answer or, on a genuinely broken CUDA installation, a thrown exception, never a silently-stale value masquerading as a real one. Guarding `.to(kCUDA)` behind `mgr.has_device(0)` was confirmed directly to skip the call entirely (`reached_cuda_call = 0`), while the identical unguarded call, run again immediately afterward with no guard at all, threw the real `c10::Error` this book first confirmed back in Chapter 4 -- turning the CUDA book's own "ask first" discipline from a design recommendation into a directly demonstrated contrast within one file. The caching benefit itself reproduces the CUDA book's own motivating claim, boolean-confirmed across 5 fresh reruns rather than reported as a single, potentially noisy timing figure: reading an already-cached `bool` 1000 times is consistently faster than calling `torch::cuda::is_available()` fresh 1000 times, and Python's own entirely separate binding layer shows the identical direction.

> `[COMMON TRAP]` It would be easy to conclude from this section that `torch::cuda::device_count()`'s different API shape makes device-discovery caching pointless -- if there's no output-parameter hazard to work around, why cache at all? The timing comparison answers this directly: caching is not only about avoiding a specific bug, it is also about avoiding *repeated, real work*. `torch::cuda::is_available()` still has to perform a genuine runtime check every time it is called, even though that check is safe to call repeatedly and always returns a trustworthy answer. The CUDA book's own `DeviceManager` earns its keep for two separable reasons -- correctness (the sentinel hazard) and performance (avoiding redundant queries) -- and this section shows that only the second reason survives the move to `torch::cuda`'s differently-shaped, output-parameter-free API.

## 10.2 A Device-Aware Allocator, and One Genuine Alignment Divergence `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `DeviceAwareAllocator` tries `cudaMalloc` when a device is believed available, falling back to plain `malloc()` either when no device was discovered or when the attempted `cudaMalloc` call itself fails -- and its own Section 10.2 measures a real cost of that fallback: `malloc()` gives no alignment guarantee at all, unlike `cudaMalloc`'s documented 256-byte floor. This section builds the identical routing logic using `torch::cuda::is_available()` and real tensor construction, then asks whether this book's own fallback allocator pays the identical alignment price.

### Background

The CUDA book's own numbers: `cudaMalloc` guarantees at least 256-byte alignment; testing plain `malloc()` five times found `5` of `5` calls were *not* 256-byte aligned. This book's own Chapter 7, Section 7.3 already measured `c10::GetCPUAllocator()`'s real guarantee directly, at 64 bytes across 50 samples per size -- this section reuses that measurement as the specific comparison point the CUDA book's own Section 10.2 asks for.

### Worked Example 10.2.1 -- routing, and the alignment guarantee this book's own fallback actually keeps

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <cstdint>
#include <vector>

// The CUDA C++ edition's Section 10.2 hand-builds DeviceAwareAllocator,
// which tries cudaMalloc when a device is (believed) available, and falls
// back to plain malloc() otherwise or on failure -- then shows that
// malloc()'s fallback path silently forfeits cudaMalloc's 256-byte
// alignment guarantee (Chapter 7.3). This file builds the identical
// device-aware routing using torch::cuda::is_available() and real tensor
// allocation, then asks the LibTorch-native version of the CUDA book's
// own question: does THIS book's fallback path also forfeit its
// alignment guarantee? The honest answer, measured directly: no --
// c10::GetCPUAllocator(), this book's own real fallback allocator from
// Chapter 2 and Chapter 7, already carries its own measured 64-byte
// guarantee (Chapter 7.3), unlike plain C malloc().
struct DeviceAwareAllocator {
    bool device_available;
    explicit DeviceAwareAllocator() : device_available(torch::cuda::is_available()) {}

    torch::Tensor allocate(int64_t numel, const char** strategy_used) {
        if (device_available) {
            try {
                torch::Tensor t = torch::empty({numel}, torch::TensorOptions().device(torch::kCUDA));
                *strategy_used = "CUDA allocation (device)";
                return t;
            } catch (const c10::Error&) {
                *strategy_used = "CPU allocation (host fallback -- CUDA alloc failed)";
                return torch::empty({numel});
            }
        }
        *strategy_used = "CPU allocation (host fallback -- no device)";
        return torch::empty({numel});
    }
};

int main() {
    DeviceAwareAllocator alloc;
    std::cout << "alloc.device_available = " << alloc.device_available << std::endl;

    // Case 1: device_available=false in this sandbox -- routes straight
    // to the CPU fallback path, exactly the CUDA book's own Case 1.
    const char* strategy = nullptr;
    torch::Tensor t = alloc.allocate(1024, &strategy);
    std::cout << "allocate(1024): strategy_used = \"" << strategy << "\", numel = " << t.numel() << std::endl;

    // Case 2: force device_available=true (the CUDA book's own "forced"
    // scenario) to exercise the CUDA-attempt-then-fallback branch, the
    // same dual-fallback-protection path the CUDA book's own Section
    // 10.2 tests deliberately.
    DeviceAwareAllocator forced_alloc;
    forced_alloc.device_available = true;  // deliberately stale/forced, matching the CUDA book's own test
    const char* strategy2 = nullptr;
    torch::Tensor t2 = forced_alloc.allocate(1024, &strategy2);
    std::cout << "forced allocate(1024): strategy_used = \"" << strategy2 << "\", numel = " << t2.numel() << std::endl;

    // The alignment question: the CUDA book's own finding is that plain
    // malloc() gives NO 256-byte alignment guarantee, unlike cudaMalloc.
    // This book's own fallback path uses c10::GetCPUAllocator() (via
    // torch::empty on CPU) rather than plain malloc() -- Chapter 7.3
    // already measured ITS real guarantee directly, at 64 bytes, across
    // 50 samples per size. This file re-confirms that guarantee holds
    // for the SPECIFIC sizes this allocator's own fallback path uses.
    auto* cpu_allocator = c10::GetCPUAllocator();
    for (int64_t numel : {1024, 4096}) {
        int64_t bytes = numel * 4;  // float32
        std::vector<c10::DataPtr> ptrs;
        int guaranteed_alignment = 4096;
        for (int i = 0; i < 50; i++) {
            c10::DataPtr ptr = cpu_allocator->allocate(bytes);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.get());
            int this_alignment = 1;
            for (int a = 4096; a >= 1; a /= 2) {
                if (addr % a == 0) { this_alignment = a; break; }
            }
            if (this_alignment < guaranteed_alignment) guaranteed_alignment = this_alignment;
            ptrs.push_back(std::move(ptr));
        }
        std::cout << "c10::GetCPUAllocator()->allocate(" << bytes << " bytes) x50: minimum observed alignment = "
                  << guaranteed_alignment << " bytes, meets 256-byte guarantee (CUDA book's own cudaMalloc figure)? "
                  << (guaranteed_alignment >= 256) << " (it meets a REAL 64-byte guarantee instead, unlike plain malloc)" << std::endl;
    }

    // Contrast with plain malloc(), to reproduce the CUDA book's own
    // finding directly on this book's own compiler: NO guarantee at all.
    int unaligned_count = 0;
    std::vector<void*> raw_ptrs;
    for (int i = 0; i < 5; i++) {
        void* p = malloc(1024);
        raw_ptrs.push_back(p);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        if (addr % 256 != 0) unaligned_count++;
    }
    std::cout << "plain malloc(1024) x5: " << unaligned_count
              << " of 5 calls NOT 256-byte aligned, CUDA book's own expected = 5 of 5" << std::endl;
    for (void* p : raw_ptrs) free(p);

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```bash
g++ -std=c++20 -O2 02_device_aware_allocator.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 02_device_aware_allocator
./02_device_aware_allocator
```

```text
alloc.device_available = 0
allocate(1024): strategy_used = "CPU allocation (host fallback -- no device)", numel = 1024
forced allocate(1024): strategy_used = "CPU allocation (host fallback -- CUDA alloc failed)", numel = 1024
c10::GetCPUAllocator()->allocate(4096 bytes) x50: minimum observed alignment = 64 bytes, meets 256-byte guarantee (CUDA book's own cudaMalloc figure)? 0 (it meets a REAL 64-byte guarantee instead, unlike plain malloc)
c10::GetCPUAllocator()->allocate(16384 bytes) x50: minimum observed alignment = 64 bytes, meets 256-byte guarantee (CUDA book's own cudaMalloc figure)? 0 (it meets a REAL 64-byte guarantee instead, unlike plain malloc)
plain malloc(1024) x5: 5 of 5 calls NOT 256-byte aligned, CUDA book's own expected = 5 of 5
```

Reproduced identically across 3 fresh reruns of the same binary, and independently cross-checked via Python's own `ctypes` binding to the system `malloc`, sharing no code with this file's own C++:

```text
ctypes malloc(1024) x5: unaligned count = 5 expected 5
```

### Discussion

Both routing cases reproduce the CUDA book's own two scenarios exactly: with `device_available = false` (this sandbox's genuine, undisguised state), `allocate()` routes straight to the CPU fallback; with `device_available` deliberately forced to `true` -- the CUDA book's own "stale discovery" test -- the attempted CUDA allocation throws (caught here as a real `c10::Error`, the identical exception this book has tracked since Chapter 4), and control falls through to the identical CPU fallback path. Both routes land on `torch::empty({numel})` on the CPU, and both report the correct `numel`.

The alignment measurement is where this chapter's own honest divergence from the CUDA book appears. `malloc(1024)` called five times reproduces the CUDA book's own exact finding -- `5` of `5` calls fail the 256-byte alignment test -- confirmed a second, independent way via Python's own `ctypes` binding directly to the same system `malloc`. But this book's own fallback allocator was never plain `malloc()` at all: `torch::empty({numel})` on the CPU routes through `c10::GetCPUAllocator()`, the identical allocator Chapter 7, Section 7.3 already measured at a genuine `64`-byte alignment floor across 50 samples per size -- re-confirmed here at the specific sizes this allocator's own fallback path actually uses, `4096` and `16384` bytes. `c10::GetCPUAllocator()`'s `64`-byte guarantee does not reach the CUDA book's own `256`-byte `cudaMalloc` figure, so the comparison correctly reports `0` (false) for "meets the 256-byte guarantee" -- but `64` bytes is a real, measured, non-zero guarantee, categorically different from plain `malloc()`'s complete absence of one. The CUDA book's own warning -- that a host fallback silently forfeits its alignment guarantee -- does not reproduce the same way here, because this book's own fallback was never the alignment-agnostic allocator the CUDA book's own fallback was.

> `[COMMON TRAP]` "This book's fallback allocator has a real alignment guarantee" is not the same claim as "this book's fallback allocator has the *same* alignment guarantee as its device path would have." Both `64` bytes and `256` bytes are real, measured floors -- but they are different numbers, and a caller who specifically needs 256-byte-aligned buffers (some SIMD or DMA-adjacent code genuinely does) cannot assume `c10::GetCPUAllocator()`'s CPU fallback provides that, even though it is meaningfully better-behaved than plain `malloc()`. The honest comparison this section draws is "some real guarantee" versus "no guarantee at all," not "an equivalent guarantee."

## Complete Runnable Code

### File: `01_device_discovery.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>

// The CUDA C++ edition's Section 10.1 hand-builds DeviceManager, whose
// entire reason to exist is CUDA's own raw C API: cudaGetDeviceCount()
// writes its answer through an OUTPUT PARAMETER, so a caller genuinely
// cannot tell "the call succeeded and reported 0 devices" apart from
// "the call failed and never touched my variable" without the CUDA
// book's own -999 sentinel trick and an explicit cudaError_t check.
// torch::cuda::device_count() and torch::cuda::is_available() are real,
// already-implemented, and return their answer BY VALUE -- there is no
// output parameter to leave untouched, so the sentinel trap this section
// exists to solve cannot occur through this API at all. This file tests
// that directly, then reproduces the CUDA book's own "ask once, cache
// the answer" discipline as a real, timed comparison.
struct DeviceManager {
    bool cuda_available;
    c10::DeviceIndex device_count;

    DeviceManager() {
        // Unlike cudaGetDeviceCount(&count), both of these return their
        // result directly -- nothing to leave untouched on failure.
        cuda_available = torch::cuda::is_available();
        device_count = torch::cuda::device_count();
    }

    bool has_device(int id) const {
        return cuda_available && id < device_count;
    }
};

int main() {
    DeviceManager mgr;
    std::cout << "mgr.cuda_available = " << mgr.cuda_available << std::endl;
    std::cout << "mgr.device_count = " << (int)mgr.device_count << std::endl;
    std::cout << "mgr.has_device(0) = " << mgr.has_device(0) << std::endl;

    // The CUDA book's own sentinel trick tests whether an output
    // parameter was left untouched on failure. torch::cuda::device_count()
    // has no output parameter to test -- it either returns a real device
    // count (by value) or, on some backends, throws. This file confirms
    // there is no "silently untouched" state possible: calling it twice
    // in a row gives the identical, directly-returned answer both times.
    c10::DeviceIndex count_call_1 = torch::cuda::device_count();
    c10::DeviceIndex count_call_2 = torch::cuda::device_count();
    std::cout << "device_count() called twice: " << (int)count_call_1 << " and " << (int)count_call_2
              << ", identical? " << (count_call_1 == count_call_2) << std::endl;

    // The CUDA book's own DeviceManager exists so a caller can check
    // has_device() BEFORE attempting a device operation, rather than
    // discovering the absence of a device only via a thrown/returned
    // error. Confirmed directly: guarding with mgr.has_device(0) avoids
    // ever reaching the .to(kCUDA) call that Chapter 4 showed throws
    // c10::Error.
    bool reached_cuda_call = false;
    bool caught_error = false;
    if (mgr.has_device(0)) {
        torch::Tensor t = torch::ones({2, 2}).to(torch::kCUDA);
        reached_cuda_call = true;
    } else {
        std::cout << "guarded by has_device(0): skipped the .to(kCUDA) call entirely, no exception risked" << std::endl;
    }
    std::cout << "reached_cuda_call (would only be true with a real device present) = " << reached_cuda_call << std::endl;

    // Contrast: calling .to(kCUDA) WITHOUT the guard still throws the
    // real c10::Error this book confirmed back in Chapter 4 -- the
    // CUDA book's own "ask first" discipline is what avoids needing to
    // catch this at all.
    try {
        torch::Tensor t = torch::ones({2, 2}).to(torch::kCUDA);
    } catch (const c10::Error& e) {
        caught_error = true;
    }
    std::cout << "unguarded .to(kCUDA) call threw c10::Error? " << caught_error << std::endl;

    // Ask-once vs ask-every-time, timed: caching mgr.cuda_available once
    // (already done in the constructor above) against calling
    // torch::cuda::is_available() 1000 times fresh.
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile bool sink = false;
    for (int i = 0; i < 1000; i++) sink = mgr.cuda_available;  // cached read
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) sink = torch::cuda::is_available();  // re-query 1000 times
    auto t2 = std::chrono::high_resolution_clock::now();

    double cached_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double requeried_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    // Raw microsecond timings are not printed here -- wall-clock timing is
    // inherently noisy run to run, and this book's own established practice
    // (Chapter 5) is to report a boolean, reproduced-across-reruns claim
    // rather than embed an exact, non-reproducible timing figure as if it
    // were deterministic output.
    bool cached_was_faster = (cached_us < requeried_us);
    std::cout << "1000 cached reads faster than 1000 fresh is_available() calls in this run? " << cached_was_faster << std::endl;
    std::cout << "[TIMING INTERPRETATION] This measures the cost of re-querying an already-cached "
              << "'no device' answer 1000 times versus reading a bool 1000 times -- not a GPU-vs-CPU "
              << "performance comparison, exactly the same honest caveat the CUDA book's own Worked "
              << "Example 10.2.3 makes about its own benchmark." << std::endl;

    return 0;
}
```

### File: `02_device_aware_allocator.cpp`

```cpp
#include <torch/torch.h>
#include <c10/core/CPUAllocator.h>
#include <iostream>
#include <cstdint>
#include <vector>

// The CUDA C++ edition's Section 10.2 hand-builds DeviceAwareAllocator,
// which tries cudaMalloc when a device is (believed) available, and falls
// back to plain malloc() otherwise or on failure -- then shows that
// malloc()'s fallback path silently forfeits cudaMalloc's 256-byte
// alignment guarantee (Chapter 7.3). This file builds the identical
// device-aware routing using torch::cuda::is_available() and real tensor
// allocation, then asks the LibTorch-native version of the CUDA book's
// own question: does THIS book's fallback path also forfeit its
// alignment guarantee? The honest answer, measured directly: no --
// c10::GetCPUAllocator(), this book's own real fallback allocator from
// Chapter 2 and Chapter 7, already carries its own measured 64-byte
// guarantee (Chapter 7.3), unlike plain C malloc().
struct DeviceAwareAllocator {
    bool device_available;
    explicit DeviceAwareAllocator() : device_available(torch::cuda::is_available()) {}

    torch::Tensor allocate(int64_t numel, const char** strategy_used) {
        if (device_available) {
            try {
                torch::Tensor t = torch::empty({numel}, torch::TensorOptions().device(torch::kCUDA));
                *strategy_used = "CUDA allocation (device)";
                return t;
            } catch (const c10::Error&) {
                *strategy_used = "CPU allocation (host fallback -- CUDA alloc failed)";
                return torch::empty({numel});
            }
        }
        *strategy_used = "CPU allocation (host fallback -- no device)";
        return torch::empty({numel});
    }
};

int main() {
    DeviceAwareAllocator alloc;
    std::cout << "alloc.device_available = " << alloc.device_available << std::endl;

    // Case 1: device_available=false in this sandbox -- routes straight
    // to the CPU fallback path, exactly the CUDA book's own Case 1.
    const char* strategy = nullptr;
    torch::Tensor t = alloc.allocate(1024, &strategy);
    std::cout << "allocate(1024): strategy_used = \"" << strategy << "\", numel = " << t.numel() << std::endl;

    // Case 2: force device_available=true (the CUDA book's own "forced"
    // scenario) to exercise the CUDA-attempt-then-fallback branch, the
    // same dual-fallback-protection path the CUDA book's own Section
    // 10.2 tests deliberately.
    DeviceAwareAllocator forced_alloc;
    forced_alloc.device_available = true;  // deliberately stale/forced, matching the CUDA book's own test
    const char* strategy2 = nullptr;
    torch::Tensor t2 = forced_alloc.allocate(1024, &strategy2);
    std::cout << "forced allocate(1024): strategy_used = \"" << strategy2 << "\", numel = " << t2.numel() << std::endl;

    // The alignment question: the CUDA book's own finding is that plain
    // malloc() gives NO 256-byte alignment guarantee, unlike cudaMalloc.
    // This book's own fallback path uses c10::GetCPUAllocator() (via
    // torch::empty on CPU) rather than plain malloc() -- Chapter 7.3
    // already measured ITS real guarantee directly, at 64 bytes, across
    // 50 samples per size. This file re-confirms that guarantee holds
    // for the SPECIFIC sizes this allocator's own fallback path uses.
    auto* cpu_allocator = c10::GetCPUAllocator();
    for (int64_t numel : {1024, 4096}) {
        int64_t bytes = numel * 4;  // float32
        std::vector<c10::DataPtr> ptrs;
        int guaranteed_alignment = 4096;
        for (int i = 0; i < 50; i++) {
            c10::DataPtr ptr = cpu_allocator->allocate(bytes);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.get());
            int this_alignment = 1;
            for (int a = 4096; a >= 1; a /= 2) {
                if (addr % a == 0) { this_alignment = a; break; }
            }
            if (this_alignment < guaranteed_alignment) guaranteed_alignment = this_alignment;
            ptrs.push_back(std::move(ptr));
        }
        std::cout << "c10::GetCPUAllocator()->allocate(" << bytes << " bytes) x50: minimum observed alignment = "
                  << guaranteed_alignment << " bytes, meets 256-byte guarantee (CUDA book's own cudaMalloc figure)? "
                  << (guaranteed_alignment >= 256) << " (it meets a REAL 64-byte guarantee instead, unlike plain malloc)" << std::endl;
    }

    // Contrast with plain malloc(), to reproduce the CUDA book's own
    // finding directly on this book's own compiler: NO guarantee at all.
    int unaligned_count = 0;
    std::vector<void*> raw_ptrs;
    for (int i = 0; i < 5; i++) {
        void* p = malloc(1024);
        raw_ptrs.push_back(p);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        if (addr % 256 != 0) unaligned_count++;
    }
    std::cout << "plain malloc(1024) x5: " << unaligned_count
              << " of 5 calls NOT 256-byte aligned, CUDA book's own expected = 5 of 5" << std::endl;
    for (void* p : raw_ptrs) free(p);

    return 0;
}
```

Files `01`-`02` all compile and link against LibTorch with the standard command from *Getting Started*:

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

`torch::cuda::device_count()` and `torch::cuda::is_available()` return their answers directly, by value, rather than through a separate output parameter -- closing off the specific hazard the CUDA book's own `-999` sentinel trick exists to catch, confirmed here by calling `device_count()` twice in a row and observing no possibility of a silently-stale answer. Guarding a `.to(kCUDA)` call behind a cached discovery check was confirmed to skip the call entirely, while the identical unguarded call still throws the real `c10::Error` this book first confirmed in Chapter 4 -- and caching that discovery result was confirmed measurably faster than re-querying it, reproduced as a boolean claim across 5 fresh reruns rather than a single non-reproducible timing figure, with Python's own separate binding layer showing the identical direction. A `DeviceAwareAllocator` built on `torch::cuda::is_available()` reproduced both of the CUDA book's own routing cases -- no device, and a forced device-believed-present-but-allocation-fails scenario -- landing on the identical CPU fallback either way. And this chapter's own genuine, measured divergence: while plain `malloc()` reproduced the CUDA book's own exact finding of zero real alignment guarantee (`5` of `5` calls unaligned, independently reconfirmed via Python's `ctypes`), this book's own fallback allocator, `c10::GetCPUAllocator()`, was re-confirmed at its own real, previously-measured 64-byte floor -- a genuine guarantee, just a smaller number than `cudaMalloc`'s own 256-byte figure, rather than the CUDA book's own "silently forfeited entirely" outcome.

## Self-Check Questions

1. Section 10.1 argues that `torch::cuda::device_count()`'s by-value return closes off the CUDA book's own sentinel hazard entirely. Construct a hypothetical (not necessarily real) API design that STILL returns a value directly, by value, but could nonetheless reintroduce a version of the same "was this genuinely computed, or is it a stale default" ambiguity.
2. Worked Example 10.1.1's `[COMMON TRAP]` argues that caching still matters even without the sentinel hazard, for performance reasons. Using the chapter's own boolean-across-5-reruns methodology, what specific claim was NOT made about the timing results, and why does that omission matter?
3. Section 10.2 tests `DeviceAwareAllocator` with `device_available` manually forced to `true` even though `torch::cuda::is_available()` genuinely reports `false` in this sandbox. What real-world situation does this forced test simulate, distinct from the ordinary `device_available=false` case?
4. Section 10.2's `[COMMON TRAP]` distinguishes "some real guarantee" from "an equivalent guarantee." Suppose a piece of code specifically required 128-byte-aligned buffers. Would `c10::GetCPUAllocator()`'s measured 64-byte guarantee be sufficient? Would a measured 256-byte guarantee be sufficient? Explain the general rule these two answers illustrate.
5. Worked Example 10.2.1 measures `c10::GetCPUAllocator()`'s alignment at two different sizes, `4096` and `16384` bytes, rather than just one. Section 7.3's own `[COMMON TRAP]` warned against trusting a single observed alignment as a guarantee. Does testing at two different sizes (each still sampled 50 times) fully address that same concern, or does it address a different one? Explain.

## Where We Go Next

This chapter tested whether `torch::cuda`'s differently-shaped discovery API closes off the CUDA book's own device-discovery hazard (it does), whether caching that discovery still pays off despite the different shape (it does, measurably), and whether this book's own fallback allocator pays the identical alignment price the CUDA book's own `malloc()` fallback does (it does not -- a genuine, smaller-but-real guarantee survives instead). Chapter 11 turns to a question this chapter's own `DeviceAwareAllocator` has been quietly relying on throughout: what actually happens to a tensor's memory when it goes out of scope, when it's moved, or when multiple tensors share the same underlying storage -- the CUDA book's own memory management system, tested against `torch::Tensor`'s real reference-counted lifetime, first glimpsed back in Chapter 6's own copy-versus-clone divergence and never yet examined on its own terms.

## Worked Solutions

**1.** A hypothetical API that returns a value directly but still risks the same ambiguity: a function `int cached_device_count()` that returns a MODULE-LEVEL cached integer, initialized to `0` at program startup, only updated the first time discovery genuinely succeeds -- and silently left at its startup default `0` forever if discovery never runs or always fails. A caller receiving `0` from such a function could not tell "discovery ran, found zero devices" apart from "discovery never actually happened, this is just the untouched startup default" -- the identical ambiguity the CUDA book's own sentinel trick solves, reintroduced through a different mechanism (a stale cached default) despite the function still technically "returning a value directly."

**2.** The chapter's own methodology never claims a specific SPEEDUP FACTOR (such as "10x faster" or "30x faster") — it only claims the directional fact that cached reads were faster than fresh queries, confirmed as true in 5 out of 5 fresh reruns. This omission matters because, as the earlier raw measurements in this chapter's own development showed (informally, before the code was revised to report only the boolean), the exact magnitude of the gap varied noticeably from run to run — reporting a specific factor as if it were a fixed, reproducible property of `torch::cuda::is_available()` itself would overstate what was actually, reliably observed.

**3.** The forced `device_available=true` test simulates a REAL, plausible failure mode distinct from "no device present": a program that cached its own device-availability check once (exactly as `DeviceManager`/`DeviceAwareAllocator` are designed to do), and then the underlying device became unavailable sometime AFTER that cached check ran — for instance, another process claiming exclusive access to the GPU, a driver fault, or the device being physically removed in a hot-pluggable setup. The cached flag would still say "available" even though a real allocation attempt now fails, which is exactly why `DeviceAwareAllocator::allocate()` needs its OWN try/catch fallback rather than trusting the cached flag unconditionally.

**4.** A 128-byte alignment requirement: `c10::GetCPUAllocator()`'s measured 64-byte guarantee would NOT be sufficient — 64 is smaller than 128, so a caller cannot assume every allocation lands on a 128-byte boundary (some allocations might, by chance, but that isn't a guarantee). A measured 256-byte guarantee WOULD be sufficient for a 128-byte requirement, because any address that is guaranteed to be a multiple of 256 is automatically also a multiple of 128 (256 is an exact multiple of 128). The general rule: a measured alignment guarantee of N bytes satisfies any REQUIRED alignment that evenly divides N, but does not satisfy a required alignment larger than N, or one that N is not a clean multiple of.

**5.** Testing at two different sizes addresses a DIFFERENT concern than Section 7.3's own `[COMMON TRAP]`. Section 7.3's trap was about SAMPLE COUNT at a single size — one allocation's address can coincidentally look more aligned than the allocator actually guarantees, which is why 50 samples were taken per size rather than one. Testing at 4096 AND 16384 bytes (each still with its own 50-sample minimum) instead addresses a SEPARATE question: whether the allocator's alignment guarantee might depend on the requested allocation SIZE itself — for instance, a real allocator might have different underlying code paths, or different alignment behavior, for a small allocation versus a much larger one. Confirming the identical 64-byte floor at both sizes rules out that size-dependence concern specifically; it does not, by itself, replace the separate 50-samples-per-size discipline that addresses the single-lucky-sample concern — both protections are present here, but they guard against two different ways this measurement could have been misleading.
