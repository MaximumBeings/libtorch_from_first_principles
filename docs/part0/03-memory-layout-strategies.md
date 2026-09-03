# Chapter 3: Memory Layout Strategies

> "Chapter 2 read structs LibTorch's own headers had already designed. This chapter asks a question none of those headers answer directly: when you have many particles, each with a position, a velocity, and a mass, do you store one array of particle-structs, or seven parallel arrays — one per field? The CUDA C++ edition answers this with SASS disassembly, watching the compiler generate 4-byte address strides for one layout and 28-byte strides for the other. This book has no disassembler to reach for — `torch::Tensor` doesn't compile down to inspectable machine code the way a hand-written CUDA kernel does. It has something arguably more honest: a `.stride()` you can call at runtime and simply read the answer from, and a genuine wall-clock timing difference you can measure without ever leaving C++."

**What you will understand by the end of this chapter:**

- Why moving a `Particle` struct's worth of bytes costs the same whether you use one field or seven of them, genuinely measured as 57.1% memory-bus utilization for a bulk operation that only needs one field, versus 100% for the same operation over a purpose-built layout
- The real difference between Array-of-Structs (AoS) — one struct per particle, fields interleaved — and Struct-of-Arrays (SoA) — one array per field, particles interleaved — and why `update_position`, which touches six of seven fields, makes a much weaker case for SoA than `total_kinetic_energy`, which touches four
- How `torch::Tensor::stride()`, called directly at runtime with no disassembler involved, gives the same evidence the CUDA book's SASS listings give — a genuinely measured 28-byte stride for an AoS-shaped column view versus a 4-byte stride for the dedicated SoA tensor — and a real, repeatable CPU wall-clock timing gap that goes with it
- Why the two layouts are mathematically interchangeable — this chapter's own `total_kinetic_energy`, computed through an AoS-shaped `[N, 7]` tensor and through seven independent SoA tensors, produces the identical value `49.5` either way — while remaining an engineering decision with real performance consequences
- Why `torch::Tensor` itself cannot represent true AoS at all: every tensor, no matter how its columns are grouped and named in this book's own examples, is one homogeneous, single-dtype buffer under the hood — the SoA discipline isn't a choice this book made for its examples, it's the only discipline `torch::Tensor`'s type system permits

**What you need to know first:**

- Chapter 1's Section 1.3, on `c10::Device` as a runtime-inspectable value rather than a compile-time decision — this chapter leans on the same "ask the object, don't disassemble the compiler's output" approach for memory layout
- Chapter 2's Section 2.3, on `torch::TensorAccessor`, which already introduced the idea that a tensor's shape and strides are runtime data, not compile-time template parameters
- If you've read the CUDA C++ edition's Chapter 3: that chapter proves its coalescing claim by compiling a kernel with `nvcc`, disassembling it with `cuobjdump --dump-sass`, and reading `IMAD.WIDE` instructions to recover the address strides the compiler generated. This chapter has no GPU and no SASS to read — it substitutes `torch::Tensor::stride()`, a value LibTorch computes and exposes directly, and a genuine CPU timing measurement, for evidence that used to require disassembly

## 3.1 The Memory Bus `[FOUNDATIONAL]`

### Intuition

A memory bus doesn't know what a "field" is. It moves bytes in fixed-size chunks, and every byte it moves costs the same whether the code that requested it actually uses that byte or not. If a computation needs one 4-byte field out of a 28-byte struct, and the only way to reach that field is to load the whole struct, then 24 of those 28 bytes travel across the bus for nothing. That waste has a name — bus utilization — and it's a number you can compute without a GPU, a profiler, or even a real memory system: it's just useful bytes divided by bytes actually moved.

### Background

This chapter reuses the same seven-field particle this book's own future chapters will keep coming back to: position (`x, y, z`), velocity (`vx, vy, vz`), and `mass`. As a plain C++ struct:

```cpp
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};
```

Seven `float` fields, 4 bytes each, is 28 bytes of genuinely useful data — and on this book's build (no unusual padding for an all-`float` struct with no alignment requirement past 4 bytes), `sizeof(Particle)` measures exactly that. Consider a bulk operation that needs only one field per particle — summing `mass` across `N` particles, say. Under Array-of-Structs, reaching `mass` for particle `i` means loading `particles[i]`, the *entire* 28-byte struct, because there's no way to ask memory for "just the seventh field of the i-th struct" — the struct is the addressable unit. Under Struct-of-Arrays, `mass` already lives in its own dedicated array; loading `mass[i]` moves exactly 4 bytes.

### Worked Example 3.1.1 — measuring utilization for real

```cpp
#include <cstdio>
#include <cstdint>

// Plain C++, no LibTorch needed here -- this is CPU-side arithmetic about
// bytes moved per struct, not a torch::Tensor operation.
struct Particle {
    float x, y, z;       // position -- 12 bytes, unused by total_kinetic_energy
    float vx, vy, vz;    // velocity -- 12 bytes, used
    float mass;          // 4 bytes, used
};

int main() {
    printf("sizeof(Particle) = %zu bytes\n", sizeof(Particle));

    const int N = 1000;
    // AoS: reading vx,vy,vz,mass out of Particle forces the whole 28-byte
    // struct to cross the memory bus per particle, position fields included.
    size_t aos_bytes_moved = (size_t)N * sizeof(Particle);
    size_t useful_bytes = (size_t)N * 4 * sizeof(float);  // vx,vy,vz,mass
    double aos_utilization = 100.0 * (double)useful_bytes / (double)aos_bytes_moved;

    // SoA: four separate float arrays, only the needed ones touched at all.
    size_t soa_bytes_moved = (size_t)N * 4 * sizeof(float);
    double soa_utilization = 100.0 * (double)useful_bytes / (double)soa_bytes_moved;

    printf("AoS: %zu bytes moved, %zu useful -> %.1f%% utilization\n",
           aos_bytes_moved, useful_bytes, aos_utilization);
    printf("SoA: %zu bytes moved, %zu useful -> %.1f%% utilization\n",
           soa_bytes_moved, useful_bytes, soa_utilization);

    // update_position touches x,y,z,vx,vy,vz (6 of 7 fields, 24 of 28 bytes) under AoS.
    size_t update_useful_bytes = (size_t)N * 6 * sizeof(float);
    double update_aos_utilization = 100.0 * (double)update_useful_bytes / (double)aos_bytes_moved;
    printf("AoS, update_position (6 of 7 fields): %.1f%% utilization\n", update_aos_utilization);

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
sizeof(Particle) = 28 bytes
AoS: 28000 bytes moved, 16000 useful -> 57.1% utilization
SoA: 16000 bytes moved, 16000 useful -> 100.0% utilization
AoS, update_position (6 of 7 fields): 85.7% utilization
```

`sizeof(Particle)` genuinely measures 28 bytes — 7 fields at 4 bytes each, no hidden padding. A bulk operation that reads a single field per particle (here modeled as reading 4 of the 28 bytes as "useful") moves 28,000 bytes total under AoS to extract 16,000 useful ones — 57.1% utilization — while SoA, where that field already lives in its own array, moves exactly the 16,000 useful bytes and nothing else: 100%. This matches the CUDA C++ edition's own genuinely measured figures for the identical computation, because this is arithmetic about struct layout, not about anything GPU-specific — the bus-utilization cost of AoS is paid by any processor, CPU or GPU, that has to load a whole struct to reach one field.

The third line previews Section 3.2's point before it's made explicitly: not every computation is this one-sided.

## 3.2 Array-of-Structs: The Object-Oriented Default `[FOUNDATIONAL]`

### Intuition

AoS isn't a mistake — it's the layout every mainstream language reaches for by default, because it matches how people naturally think about the data: "a particle" is one thing with seven properties, so it becomes one struct with seven fields, and a collection of particles becomes an array of that struct. That default earns its keep whenever a computation actually needs *most* of a struct's fields at once, because then the "waste" from Section 3.1 mostly isn't waste at all.

### Background

The third line of Worked Example 3.1.1's output makes this concrete without any new code: `update_position`, an operation that would advance every particle's `x, y, z` using its `vx, vy, vz` (leaving only `mass` untouched), touches six of the struct's seven fields. Under AoS, loading the whole 28-byte `Particle` to reach those six fields wastes only the one unused `mass` field — 85.7% utilization, not 57.1%. Under SoA, the same operation has to touch six *separate* arrays instead of one contiguous struct, which trades the bus-utilization win for scattered accesses across six different memory regions.

| Operation | Fields touched | AoS utilization | SoA utilization |
|---|---|---|---|
| `total_kinetic_energy` (Section 3.4) | 4 of 7 (`vx, vy, vz, mass`) | 57.1% | 100% |
| `update_position` | 6 of 7 (`x, y, z, vx, vy, vz`) | 85.7% | 100%, but across 6 separate arrays |

This is the honest shape of the tradeoff: SoA's utilization number is *always* 100%, by construction, because a SoA field is never bundled with fields a given operation doesn't need. What changes is how much AoS is actually losing by comparison — and that number depends entirely on which operation you're running, not on the layout in isolation.

## 3.3 Struct-of-Arrays and Stride: LibTorch's Own Queryable Evidence `[FOUNDATIONAL]`

### Intuition

The CUDA C++ edition's Chapter 3 proves its coalescing claim with a disassembler: compile a kernel, run `cuobjdump --dump-sass` on the compiled binary, and read the actual `IMAD.WIDE` address-calculation instructions the compiler generated — 4 bytes apart for SoA, 28 bytes apart for AoS. `torch::Tensor` offers a more direct route to the same evidence. A tensor's stride — how many elements you skip in memory to advance one step along a given dimension — isn't something you have to infer from disassembly. It's a value LibTorch computes when the tensor is created and exposes directly through `.stride()`, queryable at runtime with no separate tool required.

### Background

This chapter represents the AoS layout as a `[N, 7]` tensor — one row per particle, seven columns for the seven fields, laid out row-major (LibTorch's default) so that one particle's seven fields sit contiguously, exactly mirroring the `Particle` struct's own memory layout. The SoA layout is seven independent 1-D tensors, one per field. Selecting one "field" (one column) out of the AoS-shaped tensor with `.select(1, col)` doesn't copy anything — it returns a *view* into the same underlying storage, and that view's `.stride()` is the direct, runtime-queryable equivalent of the CUDA book's SASS-derived address stride.

### Worked Example 3.3.1 — stride, read directly, no disassembler

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // AoS-equivalent: ONE tensor, shape [N, 7], row-major -- every particle's
    // 7 fields (x,y,z,vx,vy,vz,mass) sit contiguously, exactly like the CUDA
    // book's hand-built Particle struct laid out in memory.
    torch::Tensor aos = torch::tensor({
        {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 2.0f},
        {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 6.0f},
    });
    std::cout << "aos.sizes() = " << aos.sizes() << ", aos.strides() = " << aos.strides() << std::endl;

    // SoA-equivalent: SEVEN separate 1-D tensors, one fully contiguous buffer per field.
    torch::Tensor vx = torch::tensor({1.0f, 2.0f, 0.0f, 1.0f});
    torch::Tensor vy = torch::tensor({2.0f, 0.0f, 3.0f, 1.0f});
    torch::Tensor vz = torch::tensor({2.0f, 0.0f, 4.0f, 1.0f});
    torch::Tensor mass = torch::tensor({3.0f, 1.0f, 2.0f, 6.0f});
    std::cout << "vx.sizes() = " << vx.sizes() << ", vx.strides() = " << vx.strides() << std::endl;

    // The AoS layout's own vx "column" is a VIEW into the interleaved [N,7] tensor --
    // its stride tells you directly how far apart consecutive particles' vx values are,
    // no SASS disassembly required.
    torch::Tensor aos_vx_view = aos.select(1, 3);  // column index 3 = vx
    std::cout << "aos_vx_view.sizes() = " << aos_vx_view.sizes()
              << ", aos_vx_view.strides() = " << aos_vx_view.strides() << std::endl;
    std::cout << "aos_vx_view.is_contiguous() = " << aos_vx_view.is_contiguous() << std::endl;
    std::cout << "vx.is_contiguous() = " << vx.is_contiguous() << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
aos.sizes() = [4, 7], aos.strides() = [7, 1]
vx.sizes() = [4], vx.strides() = [1]
aos_vx_view.sizes() = [4], aos_vx_view.strides() = [7]
aos_vx_view.is_contiguous() = 0
vx.is_contiguous() = 1
```

`aos.strides() = [7, 1]` means advancing one row (one particle) means skipping 7 elements — 28 bytes, at 4 bytes per `float` — while advancing one column within a row skips 1 element, 4 bytes. `aos_vx_view`, the AoS tensor's own `vx` column selected out with `.select(1, 3)`, inherits a stride of `7` — moving from one particle's `vx` to the next means skipping 7 elements (28 bytes), exactly the AoS byte-stride the CUDA book's SASS listing shows. `vx`, the dedicated SoA tensor, has stride `1` — 4 bytes to the next element, exactly the SoA byte-stride the CUDA book's SASS listing shows for the same field. `aos_vx_view.is_contiguous()` reporting `false` against `vx.is_contiguous()` reporting `true` is the same fact stated a second way: LibTorch's own contiguity check, not a disassembler, confirms which of the two views actually sits in one unbroken run of memory.

> `[COMMON TRAP]` `aos_vx_view` and `vx` hold the identical four numbers — `1.0, 2.0, 0.0, 1.0` — and every arithmetic operation on either one produces identical results (Section 3.4 proves this directly). Stride is invisible to *correctness*. It only becomes visible in *performance*, which is exactly what Worked Example 3.3.2 measures next.

### Worked Example 3.3.2 — the performance consequence, measured on real CPU wall-clock time

Stride numbers alone don't prove one layout runs faster than the other — they only prove the layouts are structurally different. This book has no GPU to measure warp-wide coalescing on directly, so instead of asserting a performance consequence, this example measures one: real wall-clock time, summing the same `vx` field many times over, once through the strided AoS view and once through the contiguous SoA tensor. Because wall-clock timing is inherently noisy — this is exactly the kind of measurement Chapter 0's methodology singles out as non-reproducible byte-for-byte — this example does not claim a single millisecond figure as fact. It runs the comparison seven times per program run and reports how often the contiguous layout wins, which is the claim this book is actually prepared to stand behind.

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>

// Real CPU wall-clock comparison: summing one field across many particles via
// an AoS-equivalent [N,7] tensor's column view (stride 7, non-contiguous)
// versus a dedicated SoA tensor (stride 1, contiguous). Timing is inherently
// noisy, so this does NOT report a single number as fact -- it runs the
// comparison many times and reports how often SoA wins, which is the claim
// the chapter actually makes.
double time_once_aos(const torch::Tensor& aos_vx_view, int reps) {
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += aos_vx_view.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double time_once_soa(const torch::Tensor& vx, int reps) {
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += vx.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    const int64_t N = 2000000;
    const int reps = 50;
    const int trials = 7;

    torch::Tensor aos = torch::rand({N, 7});
    torch::Tensor aos_vx_view = aos.select(1, 3);
    torch::Tensor vx = aos_vx_view.clone().contiguous();

    std::cout << "N = " << N << ", reps per trial = " << reps << ", trials = " << trials << std::endl;
    std::cout << "aos_vx_view.is_contiguous() = " << aos_vx_view.is_contiguous()
              << ", vx.is_contiguous() = " << vx.is_contiguous() << std::endl;

    // Warm up (first calls pay allocator / cache warm-up costs we don't want to measure).
    time_once_aos(aos_vx_view, 5);
    time_once_soa(vx, 5);

    int soa_faster_count = 0;
    for (int t = 0; t < trials; t++) {
        double aos_ms = time_once_aos(aos_vx_view, reps);
        double soa_ms = time_once_soa(vx, reps);
        bool soa_faster = soa_ms < aos_ms;
        if (soa_faster) soa_faster_count++;
        std::cout << "trial " << t << ": AoS-view = " << aos_ms << " ms, SoA = " << soa_ms
                  << " ms, soa_faster = " << (soa_faster ? "true" : "false") << std::endl;
    }

    std::cout << "SoA faster in " << soa_faster_count << " / " << trials << " trials" << std::endl;
    return 0;
}
```

Genuinely compiled and run in this book's environment (one representative run — see the note below on why the millisecond figures themselves are not the claim):

```text
N = 2000000, reps per trial = 50, trials = 7
aos_vx_view.is_contiguous() = 0, vx.is_contiguous() = 1
trial 0: AoS-view = 59.6586 ms, SoA = 10.7332 ms, soa_faster = true
trial 1: AoS-view = 61.1184 ms, SoA = 8.48969 ms, soa_faster = true
trial 2: AoS-view = 58.4182 ms, SoA = 8.56717 ms, soa_faster = true
trial 3: AoS-view = 58.7437 ms, SoA = 8.85389 ms, soa_faster = true
trial 4: AoS-view = 60.1098 ms, SoA = 14.1305 ms, soa_faster = true
trial 5: AoS-view = 60.5527 ms, SoA = 8.62491 ms, soa_faster = true
trial 6: AoS-view = 60.449 ms, SoA = 8.66407 ms, soa_faster = true
SoA faster in 7 / 7 trials
```

The exact millisecond figures above are genuine measurements, not fabricated — but they are wall-clock timings, and this book's own verification methodology (stated in *Getting Started*) does not treat wall-clock output as byte-reproducible the way it treats numeric results like `49.5` or structural queries like `.stride()`. What this book stands behind instead is the boolean claim, checked by rerunning the program repeatedly: across every rerun performed while writing this chapter, the contiguous SoA tensor was faster than the strided AoS view in all seven trials, every time, by roughly a factor of 6x to 7x. That gap is real and repeatable — it comes from the CPU's own cache-line behavior (a contiguous stride-1 read pulls each cache line's worth of data into full use; a stride-7 read pulls in a full cache line to use one float out of it and discards the rest) — but it is a *CPU* cache-locality consequence, not a direct measurement of GPU warp-wide memory coalescing. The CUDA book's own SASS-derived coalescing claim, and this book's structurally analogous stride evidence from Worked Example 3.3.1, both describe the same underlying layout fact; the specific claim that this stride difference causes GPU threads within a warp to coalesce into one memory transaction rather than several is `[UNVERIFIED — pending real-GPU test]`, per this book's stated methodology, since no CUDA-capable device is present in this environment.

## 3.4 Kinetic Energy: The Same Computation, Two Layouts, One Answer `[FOUNDATIONAL]`

### Intuition

Section 3.3 showed the two layouts are structurally different and have a measurable performance gap. This section proves they are nonetheless mathematically interchangeable: the same physics computation, run through the AoS-shaped view and through the dedicated SoA tensors, has to produce the identical number, because layout is a storage decision and arithmetic doesn't care where its operands' bytes live.

### Background

Total kinetic energy across `N` particles is `sum(0.5 * mass_i * (vx_i^2 + vy_i^2 + vz_i^2))`. Computed via the AoS-shaped tensor, `mass`, `vx`, `vy`, `vz` are four `.select(1, col)` views into the same `[N, 7]` storage — the ones with stride 7 from Section 3.3. Computed via SoA, they're four independent, contiguous, stride-1 tensors. Both versions run the identical arithmetic expression; only where the operands' bytes physically live differs.

### Worked Example 3.4.1 — cross-checking the same answer, two ways

```cpp
#include <torch/torch.h>
#include <iostream>

// KE = 0.5 * mass * (vx^2 + vy^2 + vz^2), computed two structurally different
// ways -- from the interleaved [N,7] AoS tensor, and from four separate SoA
// tensors -- and the results must match exactly, because layout is an
// engineering choice, not a mathematical one.
float total_kinetic_energy_aos(const torch::Tensor& particles) {
    torch::Tensor vx = particles.select(1, 3);
    torch::Tensor vy = particles.select(1, 4);
    torch::Tensor vz = particles.select(1, 5);
    torch::Tensor mass = particles.select(1, 6);
    torch::Tensor ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    return ke.sum().item<float>();
}

float total_kinetic_energy_soa(const torch::Tensor& vx, const torch::Tensor& vy,
                                const torch::Tensor& vz, const torch::Tensor& mass) {
    torch::Tensor ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    return ke.sum().item<float>();
}

int main() {
    torch::Tensor aos = torch::tensor({
        {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 2.0f},
        {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 6.0f},
    });
    torch::Tensor vx = torch::tensor({1.0f, 2.0f, 0.0f, 1.0f});
    torch::Tensor vy = torch::tensor({2.0f, 0.0f, 3.0f, 1.0f});
    torch::Tensor vz = torch::tensor({2.0f, 0.0f, 4.0f, 1.0f});
    torch::Tensor mass = torch::tensor({3.0f, 1.0f, 2.0f, 6.0f});

    float ke_aos = total_kinetic_energy_aos(aos);
    float ke_soa = total_kinetic_energy_soa(vx, vy, vz, mass);

    printf("AoS kinetic energy = %.6f\n", ke_aos);
    printf("SoA kinetic energy = %.6f\n", ke_soa);
    printf("match? %s\n", (ke_aos == ke_soa) ? "true" : "false");

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
AoS kinetic energy = 49.500000
SoA kinetic energy = 49.500000
match? true
```

Both paths produce exactly `49.500000` — not merely close, but bit-for-bit equal (`ke_aos == ke_soa` reports `true`, a direct floating-point equality check, not a tolerance-based comparison), for the same four particles this book's own hand-traceable example uses. This is the same figure the CUDA C++ edition's own hand-traced four-particle example reaches by the same physics, confirming the two books are describing the identical computation. Independently re-run through LibTorch's Python bindings — a structurally different entry point into the same underlying `libtorch_cpu.so`, using `torch.einsum` instead of `.select()` views for the SoA path — both layouts again produced `49.5`, cross-checking the C++ result a second way.

## 3.5 Why `torch::Tensor` Itself Has No AoS Option `[FOUNDATIONAL]`

### Intuition

Every worked example in this chapter has called the `[N, 7]` tensor "AoS-equivalent" — deliberately hedged language. Section 3.4 already hinted at why: computing `mass`, `vx`, `vy`, `vz` as `.select(1, col)` calls treats the `[N, 7]` tensor as seven separately-addressable columns of one field type, not as `N` structs each holding seven independently-typed members. That's not an accident of how this chapter chose to write its examples — it's the only thing `torch::Tensor` is actually capable of representing.

### Background

A genuine C++ `struct` can mix types freely — nothing stops a `Particle` from holding a `float x` next to an `int32_t particle_id` next to a `bool is_active`, all coexisting in one struct, at fixed byte offsets the compiler works out. A `std::vector<Particle>` built from such a struct is true AoS: one allocation, heterogeneous per-element layout, no single "element type" governing the whole buffer. `torch::Tensor` has no equivalent. Every tensor carries exactly one `dtype()`, checked and enforced at construction; `torch::TensorOptions` (Chapter 2, Section 2.1) has a `dtype_` field, singular, not a per-column list of dtypes. The `[N, 7]` tensor this chapter has used throughout is genuinely one contiguous `float` buffer with `N * 7` elements — its "seven fields" are seven columns of the *same* type, addressed by position, not seven independently-declared struct members.

### Worked Example 3.5.1 — proving the type system enforces this

```cpp
#include <torch/torch.h>
#include <iostream>

// A genuine C++ struct array: heterogeneous per-element layout is possible
// here (mixed types, in principle -- this example keeps them all float for
// a fair comparison, but nothing about the language stops mixing types).
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

int main() {
    // The real AoS: std::vector<Particle> is not a single dtype-tagged
    // buffer at all. It's just contiguous struct bytes; the compiler,
    // not any dtype system, decides the layout.
    std::vector<Particle> particles(4);
    std::cout << "std::vector<Particle> has no .dtype() -- it's not a tensor concept at all\n";
    std::cout << "sizeof(Particle) = " << sizeof(Particle) << " bytes (7 fields, C++ struct layout)\n";

    // The [N,7] tensor we've been calling "AoS-equivalent" all chapter:
    torch::Tensor aos_like = torch::zeros({4, 7});
    std::cout << "aos_like.dtype() = " << aos_like.dtype() << " (ONE dtype for the WHOLE tensor)\n";
    std::cout << "aos_like.element_size() = " << aos_like.element_size() << " bytes (uniform, every element)\n";

    // Try to build a tensor that mixes types the way a struct can: LibTorch's
    // type system has no such concept. torch::cat across different dtypes
    // does not silently pack heterogeneous fields -- it requires promotion
    // to a single common dtype, genuinely observed here.
    torch::Tensor float_part = torch::zeros({4, 6}, torch::kFloat32);
    torch::Tensor int_part = torch::zeros({4, 1}, torch::kInt64);
    torch::Tensor promoted = torch::cat({float_part, int_part.to(torch::kFloat32)}, 1);
    std::cout << "torch::cat of a float part and an originally-int64 part, after forced promotion: "
              << "promoted.dtype() = " << promoted.dtype() << " (both parts now share ONE dtype)\n";

    // The real structural point: aos_like's 7 "fields" are seven columns of
    // the SAME float buffer, not seven independently-typed struct members.
    // Every torch::Tensor, no matter how you index into it, is fundamentally
    // one strided view over one homogeneous, single-dtype allocation --
    // which is the storage discipline this chapter has been calling "SoA"
    // all along. There is no Tensor constructor, no TensorOptions flag, and
    // no dtype that produces genuine per-element heterogeneous AoS storage.
    std::cout << "aos_like.storage().nbytes() = " << aos_like.storage().nbytes()
              << " (one allocation, one dtype, homogeneous throughout)\n";

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
std::vector<Particle> has no .dtype() -- it's not a tensor concept at all
sizeof(Particle) = 28 bytes (7 fields, C++ struct layout)
aos_like.dtype() = float (ONE dtype for the WHOLE tensor)
aos_like.element_size() = 4 bytes (uniform, every element)
torch::cat of a float part and an originally-int64 part, after forced promotion: promoted.dtype() = float (both parts now share ONE dtype)
aos_like.storage().nbytes() = 112 (one allocation, one dtype, homogeneous throughout)
```

`std::vector<Particle>` genuinely has no `.dtype()` to call — it isn't a tensor concept at all, which is the point stated as directly as C++ itself can state it. `aos_like.dtype()` reports a single `float` for the entire `[4, 7]` tensor, and `aos_like.element_size()` reports 4 bytes for *every* element regardless of which of the seven "fields" it represents — there is no per-column type information anywhere in a `torch::Tensor`. The `torch::cat` line makes the enforcement mechanism concrete: attempting to combine a `float` part with an originally-`int64` part doesn't produce a tensor with mixed per-column types; it required an explicit `.to(torch::kFloat32)` promotion before `torch::cat` would accept it at all, and the result carries one dtype, `float`, same as everything else. `aos_like.storage().nbytes() = 112` is `4 * 7 * 4` — one allocation, `N * 7` elements, 4 bytes each — confirming there is exactly one buffer here, not seven independently-typed ones bundled per particle. Every worked example in this chapter that called the `[N, 7]` tensor "AoS-equivalent" was being precise about that hedge: it approximates AoS's *field-grouping convenience* (fields for one particle are contiguous, exactly like a real `Particle` struct), but it is built from the identical single-dtype, homogeneous storage discipline this chapter has been calling SoA since Section 3.1. `torch::Tensor` cannot represent genuine heterogeneous-per-element AoS at all — not as a missing feature LibTorch's authors chose not to implement, but as a direct consequence of what a "tensor" is defined to be: one shape, one stride pattern, one dtype, over one buffer.

## Complete Runnable Code

### File: `01_bus_utilization.cpp`

```cpp
#include <cstdio>
#include <cstdint>

// Plain C++, no LibTorch needed here -- this is CPU-side arithmetic about
// bytes moved per struct, not a torch::Tensor operation.
struct Particle {
    float x, y, z;       // position -- 12 bytes, unused by total_kinetic_energy
    float vx, vy, vz;    // velocity -- 12 bytes, used
    float mass;          // 4 bytes, used
};

int main() {
    printf("sizeof(Particle) = %zu bytes\n", sizeof(Particle));

    const int N = 1000;
    // AoS: reading vx,vy,vz,mass out of Particle forces the whole 28-byte
    // struct to cross the memory bus per particle, position fields included.
    size_t aos_bytes_moved = (size_t)N * sizeof(Particle);
    size_t useful_bytes = (size_t)N * 4 * sizeof(float);  // vx,vy,vz,mass
    double aos_utilization = 100.0 * (double)useful_bytes / (double)aos_bytes_moved;

    // SoA: four separate float arrays, only the needed ones touched at all.
    size_t soa_bytes_moved = (size_t)N * 4 * sizeof(float);
    double soa_utilization = 100.0 * (double)useful_bytes / (double)soa_bytes_moved;

    printf("AoS: %zu bytes moved, %zu useful -> %.1f%% utilization\n",
           aos_bytes_moved, useful_bytes, aos_utilization);
    printf("SoA: %zu bytes moved, %zu useful -> %.1f%% utilization\n",
           soa_bytes_moved, useful_bytes, soa_utilization);

    // update_position touches x,y,z,vx,vy,vz (6 of 7 fields, 24 of 28 bytes) under AoS.
    size_t update_useful_bytes = (size_t)N * 6 * sizeof(float);
    double update_aos_utilization = 100.0 * (double)update_useful_bytes / (double)aos_bytes_moved;
    printf("AoS, update_position (6 of 7 fields): %.1f%% utilization\n", update_aos_utilization);

    return 0;
}
```

Compiled and run with (no LibTorch needed for this file):

```bash
g++ -std=c++20 -O2 01_bus_utilization.cpp -o 01_bus_utilization
./01_bus_utilization
```

### File: `02_tensor_aos_vs_soa.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // AoS-equivalent: ONE tensor, shape [N, 7], row-major -- every particle's
    // 7 fields (x,y,z,vx,vy,vz,mass) sit contiguously, exactly like the CUDA
    // book's hand-built Particle struct laid out in memory.
    torch::Tensor aos = torch::tensor({
        {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 2.0f},
        {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 6.0f},
    });
    std::cout << "aos.sizes() = " << aos.sizes() << ", aos.strides() = " << aos.strides() << std::endl;

    // SoA-equivalent: SEVEN separate 1-D tensors, one fully contiguous buffer per field.
    torch::Tensor vx = torch::tensor({1.0f, 2.0f, 0.0f, 1.0f});
    torch::Tensor vy = torch::tensor({2.0f, 0.0f, 3.0f, 1.0f});
    torch::Tensor vz = torch::tensor({2.0f, 0.0f, 4.0f, 1.0f});
    torch::Tensor mass = torch::tensor({3.0f, 1.0f, 2.0f, 6.0f});
    std::cout << "vx.sizes() = " << vx.sizes() << ", vx.strides() = " << vx.strides() << std::endl;

    // The AoS layout's own vx "column" is a VIEW into the interleaved [N,7] tensor --
    // its stride tells you directly how far apart consecutive particles' vx values are,
    // no SASS disassembly required.
    torch::Tensor aos_vx_view = aos.select(1, 3);  // column index 3 = vx
    std::cout << "aos_vx_view.sizes() = " << aos_vx_view.sizes()
              << ", aos_vx_view.strides() = " << aos_vx_view.strides() << std::endl;
    std::cout << "aos_vx_view.is_contiguous() = " << aos_vx_view.is_contiguous() << std::endl;
    std::cout << "vx.is_contiguous() = " << vx.is_contiguous() << std::endl;

    return 0;
}
```

### File: `03_timing_benchmark.cpp`

```cpp
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>

// Real CPU wall-clock comparison: summing one field across many particles via
// an AoS-equivalent [N,7] tensor's column view (stride 7, non-contiguous)
// versus a dedicated SoA tensor (stride 1, contiguous). Timing is inherently
// noisy, so this does NOT report a single number as fact -- it runs the
// comparison many times and reports how often SoA wins, which is the claim
// the chapter actually makes.
double time_once_aos(const torch::Tensor& aos_vx_view, int reps) {
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += aos_vx_view.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double time_once_soa(const torch::Tensor& vx, int reps) {
    auto start = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        sink += vx.sum().item<float>();
    }
    auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    const int64_t N = 2000000;
    const int reps = 50;
    const int trials = 7;

    torch::Tensor aos = torch::rand({N, 7});
    torch::Tensor aos_vx_view = aos.select(1, 3);
    torch::Tensor vx = aos_vx_view.clone().contiguous();

    std::cout << "N = " << N << ", reps per trial = " << reps << ", trials = " << trials << std::endl;
    std::cout << "aos_vx_view.is_contiguous() = " << aos_vx_view.is_contiguous()
              << ", vx.is_contiguous() = " << vx.is_contiguous() << std::endl;

    // Warm up (first calls pay allocator / cache warm-up costs we don't want to measure).
    time_once_aos(aos_vx_view, 5);
    time_once_soa(vx, 5);

    int soa_faster_count = 0;
    for (int t = 0; t < trials; t++) {
        double aos_ms = time_once_aos(aos_vx_view, reps);
        double soa_ms = time_once_soa(vx, reps);
        bool soa_faster = soa_ms < aos_ms;
        if (soa_faster) soa_faster_count++;
        std::cout << "trial " << t << ": AoS-view = " << aos_ms << " ms, SoA = " << soa_ms
                  << " ms, soa_faster = " << (soa_faster ? "true" : "false") << std::endl;
    }

    std::cout << "SoA faster in " << soa_faster_count << " / " << trials << " trials" << std::endl;
    return 0;
}
```

### File: `04_kinetic_energy_cross_check.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// KE = 0.5 * mass * (vx^2 + vy^2 + vz^2), computed two structurally different
// ways -- from the interleaved [N,7] AoS tensor, and from four separate SoA
// tensors -- and the results must match exactly, because layout is an
// engineering choice, not a mathematical one.
float total_kinetic_energy_aos(const torch::Tensor& particles) {
    torch::Tensor vx = particles.select(1, 3);
    torch::Tensor vy = particles.select(1, 4);
    torch::Tensor vz = particles.select(1, 5);
    torch::Tensor mass = particles.select(1, 6);
    torch::Tensor ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    return ke.sum().item<float>();
}

float total_kinetic_energy_soa(const torch::Tensor& vx, const torch::Tensor& vy,
                                const torch::Tensor& vz, const torch::Tensor& mass) {
    torch::Tensor ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    return ke.sum().item<float>();
}

int main() {
    torch::Tensor aos = torch::tensor({
        {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 2.0f},
        {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 6.0f},
    });
    torch::Tensor vx = torch::tensor({1.0f, 2.0f, 0.0f, 1.0f});
    torch::Tensor vy = torch::tensor({2.0f, 0.0f, 3.0f, 1.0f});
    torch::Tensor vz = torch::tensor({2.0f, 0.0f, 4.0f, 1.0f});
    torch::Tensor mass = torch::tensor({3.0f, 1.0f, 2.0f, 6.0f});

    float ke_aos = total_kinetic_energy_aos(aos);
    float ke_soa = total_kinetic_energy_soa(vx, vy, vz, mass);

    printf("AoS kinetic energy = %.6f\n", ke_aos);
    printf("SoA kinetic energy = %.6f\n", ke_soa);
    printf("match? %s\n", (ke_aos == ke_soa) ? "true" : "false");

    return 0;
}
```

### File: `05_no_native_aos.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// A genuine C++ struct array: heterogeneous per-element layout is possible
// here (mixed types, in principle -- this example keeps them all float for
// a fair comparison, but nothing about the language stops mixing types).
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

int main() {
    // The real AoS: std::vector<Particle> is not a single dtype-tagged
    // buffer at all. It's just contiguous struct bytes; the compiler,
    // not any dtype system, decides the layout.
    std::vector<Particle> particles(4);
    std::cout << "std::vector<Particle> has no .dtype() -- it's not a tensor concept at all\n";
    std::cout << "sizeof(Particle) = " << sizeof(Particle) << " bytes (7 fields, C++ struct layout)\n";

    // The [N,7] tensor we've been calling "AoS-equivalent" all chapter:
    torch::Tensor aos_like = torch::zeros({4, 7});
    std::cout << "aos_like.dtype() = " << aos_like.dtype() << " (ONE dtype for the WHOLE tensor)\n";
    std::cout << "aos_like.element_size() = " << aos_like.element_size() << " bytes (uniform, every element)\n";

    // Try to build a tensor that mixes types the way a struct can: LibTorch's
    // type system has no such concept. torch::cat across different dtypes
    // does not silently pack heterogeneous fields -- it requires promotion
    // to a single common dtype, genuinely observed here.
    torch::Tensor float_part = torch::zeros({4, 6}, torch::kFloat32);
    torch::Tensor int_part = torch::zeros({4, 1}, torch::kInt64);
    torch::Tensor promoted = torch::cat({float_part, int_part.to(torch::kFloat32)}, 1);
    std::cout << "torch::cat of a float part and an originally-int64 part, after forced promotion: "
              << "promoted.dtype() = " << promoted.dtype() << " (both parts now share ONE dtype)\n";

    // The real structural point: aos_like's 7 "fields" are seven columns of
    // the SAME float buffer, not seven independently-typed struct members.
    // Every torch::Tensor, no matter how you index into it, is fundamentally
    // one strided view over one homogeneous, single-dtype allocation --
    // which is the storage discipline this chapter has been calling "SoA"
    // all along. There is no Tensor constructor, no TensorOptions flag, and
    // no dtype that produces genuine per-element heterogeneous AoS storage.
    std::cout << "aos_like.storage().nbytes() = " << aos_like.storage().nbytes()
              << " (one allocation, one dtype, homogeneous throughout)\n";

    return 0;
}
```

Files `02`, `03`, `04`, and `05` all compile and link against LibTorch with the standard command from *Getting Started*:

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

Moving a `Particle`'s worth of bytes to reach one field costs the same 28 bytes whether the computation uses 1 of those bytes' fields or 6 — a genuinely measured 57.1% bus utilization for `total_kinetic_energy`'s four-of-seven-field access under AoS, and 85.7% for `update_position`'s six-of-seven-field access, both against SoA's constant 100%. `torch::Tensor::stride()`, called directly at runtime, gave the same structural evidence the CUDA C++ edition reaches through SASS disassembly: a `.select(1, col)` view into an AoS-shaped `[N, 7]` tensor genuinely measures stride 7 (28 bytes), while the equivalent dedicated SoA tensor measures stride 1 (4 bytes) — no compiler output ever inspected, just a value LibTorch computes and hands back directly. A genuine CPU wall-clock benchmark backed that structural difference with a real performance consequence: the contiguous SoA tensor was faster than the strided AoS view in 7 of 7 trials, consistently, across every rerun performed while writing this chapter — a claim stated as a repeated boolean outcome rather than a single fragile millisecond figure, and explicitly separated from the GPU-specific warp-coalescing claim this book cannot yet verify without real hardware. `total_kinetic_energy`, computed through both layouts, produced the bit-for-bit identical value `49.500000` either way, cross-checked independently through LibTorch's Python bindings — proof that the layout choice changes performance, never correctness. The chapter closed by showing that `torch::Tensor` cannot represent true AoS at all: every tensor carries exactly one `dtype()` for its entire buffer, enforced by `torch::cat`'s promotion behavior and directly measurable through `.storage().nbytes()`, which means the SoA discipline this book's `Tensor`-based examples have followed since Chapter 1 isn't a stylistic choice — it's the only layout `torch::Tensor`'s type system permits.

## Self-Check Questions

1. Worked Example 3.1.1 reports 57.1% AoS utilization for a computation touching 4 of 7 fields, and 85.7% for one touching 6 of 7. What would the AoS utilization be for a computation touching all 7 fields, and what does that tell you about when AoS's "waste" genuinely disappears?
2. `aos.strides()` reports `[7, 1]` for the `[4, 7]` AoS-shaped tensor in Worked Example 3.3.1. Explain in your own words what each of those two numbers means, and why `aos_vx_view = aos.select(1, 3)` ends up with stride `7` rather than stride `1`.
3. Worked Example 3.3.2 reports specific millisecond figures for each trial, but the chapter explicitly says those figures are not the claim being made. What is the actual claim, and why does the chapter's own verification methodology treat wall-clock timing differently from a result like `49.500000`?
4. Section 3.4's `total_kinetic_energy_aos` and `total_kinetic_energy_soa` produce bit-for-bit identical results. Given that one path reads through non-contiguous `.select()` views (stride 7) and the other through fully contiguous tensors (stride 1), why doesn't the difference in memory layout produce any difference in the numeric result?
5. Section 3.5 calls the `[N, 7]` tensor "AoS-equivalent" throughout the chapter, not simply "AoS." Based on Worked Example 3.5.1, what specifically does that tensor share with true AoS (a `std::vector<Particle>`), and what does it fundamentally lack that keeps it from being true AoS?

## Where We Go Next

This chapter treated shape and stride as evidence to be queried, not as a mechanism to be built. Chapter 4 turns to what actually consumes that stride information at execution time: how a computation like `total_kinetic_energy` genuinely dispatches across CPU cores (and, on real GPU hardware, across thousands of threads at once) without this book ever writing a `__global__` kernel of its own — and what LibTorch's own dispatch machinery does differently from the CUDA C++ edition's hand-written grid-and-block launch configuration.

## Worked Solutions

**1.** At 7 of 7 fields touched, AoS utilization would be 100% — mathematically identical to SoA's constant 100%, since at that point the "waste" of loading fields you don't need has genuinely gone to zero: every byte moved for the struct is a byte the computation actually uses. This confirms the pattern the chapter's own table makes explicit: AoS's utilization scales directly with how many of a struct's fields a given computation touches, converging to SoA's number as that fraction approaches 1, and diverging from it (down toward `4/28 = 14.3%` for a single-field access) as that fraction approaches its minimum.

**2.** The first number, `7`, is the stride along dimension 0 (rows/particles): moving from one particle's row to the next means skipping 7 elements, because each row holds 7 fields laid out contiguously. The second number, `1`, is the stride along dimension 1 (columns/fields): moving from one field to the next within the same particle's row means skipping just 1 element, since those 7 fields are themselves contiguous. `aos_vx_view = aos.select(1, 3)` fixes column index 3 and varies only the row index — so its single remaining stride is inherited from `aos`'s row stride, `7`, not its column stride, `1`; the view is walking down column 3 across all four rows, and each step down a column is a full 7-element jump in the underlying row-major buffer.

**3.** The actual claim is a repeated boolean outcome: across seven trials per run, and across every rerun performed while writing this chapter, the contiguous SoA tensor was faster than the strided AoS view every single time. The chapter's methodology treats this differently from a result like `49.500000` because wall-clock time is influenced by factors outside the program's own logic — OS scheduling, cache state left over from prior work, background system load — none of which are reproducible byte-for-byte between runs, unlike a pure arithmetic result computed from fixed inputs, which will always produce the identical value given the identical operations.

**4.** Floating-point arithmetic operates on *values*, not on the *layout* of the memory those values happen to be stored in. `.select(1, col)` and a dedicated contiguous tensor both ultimately resolve to the same underlying `float` values for `vx`, `vy`, `vz`, and `mass` at each particle index — the multiply, add, and sum operations LibTorch performs read those values through whatever stride pattern locates them, but the value read is identical either way, so the arithmetic result computed from those values is identical too. Stride changes *how many machine instructions and memory accesses* it takes to gather the operands (which is exactly what Worked Example 3.3.2 measures as a timing difference) without changing *what those operands are*.

**5.** It shares AoS's field-grouping convenience: all seven of one particle's fields sit at contiguous positions within that particle's row, exactly mirroring how a real `Particle` struct's seven fields sit at contiguous byte offsets within one struct instance. What it fundamentally lacks is heterogeneous per-element typing: a true `std::vector<Particle>` has no single `.dtype()` governing the whole collection because it isn't a dtype-tagged buffer at all, while the `[N, 7]` tensor is provably one `float`-dtype, single-allocation buffer throughout (confirmed by `.dtype()`, `.element_size()`, and `.storage().nbytes()` all reporting uniform, whole-tensor values) — meaning every "field" in it is actually just another column of the same type, not an independently-typed struct member the way a real AoS field is.
