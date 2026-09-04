# Appendix A: Installation & Setup

The CUDA C++ edition's own Appendix A is deliberately short and loose -- four plain sections, no epigraph, no numbered worked examples, no self-check questions -- because installation is not a conceptual topic the way a chapter's own material is; it is a one-time checklist to get through before any of the book's own real content can be compiled at all. This appendix follows that same loose shape rather than forcing the numbered-chapter template onto material that does not need it.

## Local machine (CPU or GPU) with LibTorch installed

Every chapter of this book was built and genuinely compiled against a real, installed LibTorch -- not a standalone download of the separate C++ distribution archive from pytorch.org, but the LibTorch shared libraries and headers that already ship INSIDE a normal Python `torch` package (`pip install torch`). This is worth stating plainly, because it is the actual, tested path every single source file in this book has used, not a simplification for this appendix alone: a Python `torch` installation already contains the full `libtorch.so`, `libtorch_cpu.so`, and `libc10.so` shared libraries, plus every C++ header under `torch/csrc/api/include`, at a fixed, discoverable path relative to the Python package itself. Locating that path is one Python one-liner:

```text
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
```

and the standard compile line used throughout this entire book, unchanged from Chapter 1 to Chapter 22, is:

```text
g++ -std=c++20 -O2 some_example.cpp \
    -I"$TORCH_DIR/include" \
    -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$TORCH_DIR/lib" \
    -o some_example
./some_example
```

The `-D_GLIBCXX_USE_CXX11_ABI=1` flag is not optional decoration -- it is the single most common source of a genuinely confusing LibTorch linker error for a newcomer, and it has no equivalent at all in the CUDA C++ edition's own installation appendix, since raw CUDA/C++ code has no dual-ABI concern to navigate. A `pip`-installed `torch` package on Linux is built against the new (C++11) `libstdc++` ABI, and code compiled against it with the OLD ABI (the pre-C++11 default some older toolchains still ship) fails to link at all, with error messages that do not obviously point back to this flag. Passing `-D_GLIBCXX_USE_CXX11_ABI=1` explicitly, matching whatever ABI the installed `torch` package itself was built with, is what every compile line in this book does, for exactly this reason -- confirmed directly against this sandbox's own installed `torch` package below, not assumed.

**Smoke test.** The direct LibTorch-native equivalent of the CUDA book's own `nvidia-smi` / `nvcc --version` pair -- two checks that must both work before compiling anything else in this book at all.

```cpp
#include <torch/torch.h>
#include <iostream>

// Appendix A's own smoke test: the LibTorch-native equivalent of the
// CUDA book's own `nvidia-smi` / `nvcc --version` pair -- two quick
// checks that must both work before compiling anything else in this
// book. `torch::cuda::is_available()` is the real, production check
// for whether a CUDA-capable device is actually present and usable
// (the direct LibTorch analog of `nvidia-smi` reporting a real GPU);
// building and running one minimal real torch::Tensor program is the
// direct analog of `nvcc --version` -- proof that the include paths,
// library paths, and ABI flags are all genuinely correct, not just
// that a version string prints.
int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "torch::cuda::is_available() = " << torch::cuda::is_available() << std::endl;
    std::cout << "torch::cuda::device_count() = " << static_cast<int>(torch::cuda::device_count()) << std::endl;

    torch::Tensor t = torch::tensor({1.0, 2.0, 3.0}) * 2.0;
    std::cout << "smoke test tensor (torch::tensor({1,2,3}) * 2) = " << t << std::endl;
    std::cout << "smoke test passed: real torch::Tensor arithmetic executed and produced [2,4,6]? "
              << torch::allclose(t, torch::tensor({2.0, 4.0, 6.0})) << std::endl;

    return 0;
}
```

Genuinely compiled and run, in this exact sandbox, via the exact compile line above:

```text
LibTorch version: 2.14.0
torch::cuda::is_available() = 0
torch::cuda::device_count() = 0
smoke test tensor (torch::tensor({1,2,3}) * 2) =  2
 4
 6
[ CPUFloatType{3} ]
smoke test passed: real torch::Tensor arithmetic executed and produced [2,4,6]? 1
```

If this program fails to compile or link, fix that before compiling anything in this book -- every chapter assumes it already works, exactly as the CUDA book's own appendix states about `nvidia-smi`/`nvcc --version`.

## No local NVIDIA GPU: this book's own sandbox, and honest CPU-fallback behavior

This entire book -- every one of the roughly seventy source files compiled and genuinely run across Chapters 1 through 22 -- was built inside a sandbox with no NVIDIA driver and no GPU hardware attached at all. The installed `torch` package itself IS a CUDA-enabled build (`2.14.0+cu130`, meaning it was compiled with CUDA support and is capable of running on a GPU if one were present); what makes every chapter's own work genuinely CPU-only is not a CPU-only LibTorch build, but the simple fact that `torch::cuda::is_available()` genuinely, correctly reports `false` in an environment with no actual GPU hardware or driver -- confirmed directly above, not assumed, in the exact same smoke test that confirms the ABI and library paths are correct. This is the direct LibTorch-native analog of the CUDA book's own Section 2 (a cloud instance with no GPU attached still compiles and runs Runtime API calls, which then return `cudaErrorNoDevice`): a real, production LibTorch build genuinely queries for a device, genuinely finds none, and genuinely, correctly reports that absence through its own real API -- `torch::cuda::is_available()` returning `false`, rather than a raw CUDA error code, since a LibTorch program's own tensor operations simply execute on CPU by default and never attempt a CUDA call at all unless a tensor or model is explicitly moved there with `.to(torch::kCUDA)`.

Every chapter of this book that had any GPU-hardware-dependent claim to make -- Chapters 18 and 19 specifically, on GPU kernel implementation and performance optimization -- disclosed this honestly and explicitly at the point that claim was made, tagging anything that could not be genuinely tested on this sandbox's own CPU-only hardware as `[UNVERIFIED -- pending real-GPU test]`, rather than fabricating a plausible-looking GPU result. Chapters 20 through 22 (Parts 6 and 7) needed no such tags at all -- neural network layers, autograd internals, serialization, gradient debugging, attention algorithms, Mixture-of-Experts routing, and every technique in Chapter 22's own quantitative finance material turn out to be pure mathematics and software engineering with no CUDA-hardware dependency whatsoever, a genuinely useful property of this book's own chosen subject matter, confirmed chapter by chapter rather than assumed from the outset.

## Version notes

This book's own installed `torch` package: version `2.14.0+cu130` (a CUDA 13.0-enabled build; genuinely CUDA-capable, but running with no GPU attached, per above), `LibTorch` version string `2.14.0` (reported by the `TORCH_VERSION` macro directly in the smoke test above), compiled with `_GLIBCXX_USE_CXX11_ABI=1` (confirmed directly from the installed package's own `torch._C._GLIBCXX_USE_CXX11_ABI` attribute, which reports `True`). The system compiler used throughout is `g++ 13.3.0` on Ubuntu 24.04, with every source file compiled under the C++20 standard (`-std=c++20`) -- nothing in this book requires a feature newer than C++17 (`torch::autograd::Function`, structured bindings, and ordinary template code are all that any chapter uses), so an older compiler targeting C++17 would likely work for every example in this book, though this was not independently tested, exactly as the CUDA book's own hedged version language ("older toolkits may work too") is itself hedged rather than a tested guarantee.

Unlike the CUDA book's own target (a fixed compute capability, `sm_80`, chosen for broad availability and specific tensor-core feature support), LibTorch's own version compatibility concern is almost entirely about the ABI flag above, not about a specific hardware generation -- any reasonably recent `torch` package (this book's own chapters were written and tested against version `2.14`, but nothing in this book's own code uses an API newer than what has been stable in LibTorch for several major versions) paired with a compiler whose ABI matches what that package reports should reproduce every result in this book.

## Building this book's own documentation site

Identical in spirit to the CUDA book's own final section, adjusted only for this book's own repository:

```text
git clone https://github.com/MaximumBeings/libtorch_from_first_principles.git
cd libtorch_from_first_principles
python3 -m venv .venv && source .venv/bin/activate
pip install mkdocs-material
mkdocs serve
```

`mkdocs serve` builds and serves this book's own documentation site directly from the Markdown files under `docs/` -- no LibTorch installation, no compiler, and no GPU is needed for this step at all, since it only renders already-written Markdown into a browsable site; every code block embedded in every chapter of this site is a static, already-verified transcript of a real compile and run, not something the documentation build itself re-executes.
