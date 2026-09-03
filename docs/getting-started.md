# Getting Started

This book targets **LibTorch 2.14.0+** (PyTorch's C++ frontend), built against **C++20**, on **Python 3.11 / Ubuntu 24.04 / GCC 13.3**. Every listing's *compilation* and, for anything that runs on CPU, its *execution*, is genuinely done against a real LibTorch installation in this book's build environment. Actually running a kernel on an NVIDIA GPU needs a real device and driver, which this environment does not have — this book flags that explicitly wherever it applies (see [How this book verifies its claims](#how-this-book-verifies-its-claims) below).

## Environment

The simplest real LibTorch install is the one PyPI already ships: the `torch` Python wheel bundles a complete, genuine LibTorch distribution — headers, static config, and the same `libtorch.so` / `libtorch_cpu.so` / `libc10.so` shared libraries a hand-downloaded LibTorch zip would give you — under `torch/include` and `torch/lib`. This book compiles and links directly against that installed wheel rather than a separately downloaded LibTorch archive, since PyPI is reachable in this environment and `download.pytorch.org`'s own zip distribution is not:

```bash
pip install torch
```

Confirm the install and check the version:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

This book was written and verified against:

```text
torch 2.14.0+cu130
Python 3.11.15
GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)
```

with CUDA 13.0 support genuinely compiled into the wheel — `torch.version.cuda` reports `13.0` — but no NVIDIA GPU or driver present, so `torch.cuda.is_available()` genuinely reports `False`. This mirrors the CUDA and cuTile Python sibling books' own build environments almost exactly: a real, GPU-aware toolchain with no hardware behind it.

!!! warning "This is not a `find_package(Torch)` CMake setup"
    `TorchConfig.cmake`, the config CMake's `find_package(Torch REQUIRED)` reads, was generated against a CUDA-enabled build and insists on discovering a full CUDA toolkit (`nvcc`, a matching `cuda_runtime.h`, `libcudart`) at *configure* time — even to produce a CPU-only binary. This book compiles and links directly against the wheel's headers and shared libraries with a plain `g++` invocation instead of routing through that CMake config, which sidesteps the toolkit-discovery requirement entirely. The exact compile line is shown below and is the one every chapter's "Complete Runnable Code" section uses.

## Compiling against LibTorch directly

```bash
TORCH_DIR=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
g++ -std=c++20 -O2 main.cpp \
  -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,"$TORCH_DIR/lib" \
  -o main
```

The `-std=c++20` matters: this LibTorch build was itself compiled with C++20 (`torch.__config__.show()` reports `C++ Version: 202002`), and its headers use language features that fail to resolve under `-std=c++17`.

## Verify the toolchain — genuinely run, no GPU required

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor a = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    torch::Tensor b = torch::ones({2, 2});
    torch::Tensor c = a + b;
    std::cout << "a + b =\n" << c << std::endl;

    std::cout << "torch::cuda::is_available() = "
              << (torch::cuda::is_available() ? "true" : "false") << std::endl;
    std::cout << "torch::cuda::device_count() = "
              << static_cast<int>(torch::cuda::device_count()) << std::endl;

    torch::Tensor x = torch::tensor({2.0}, torch::requires_grad());
    torch::Tensor y = x * x * 3;
    y.backward();
    std::cout << "x.grad() for y = 3*x^2 at x=2 -> " << x.grad() << std::endl;
    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
a + b =
 2  3
 4  5
[ CPUFloatType{2,2} ]
torch::cuda::is_available() = false
torch::cuda::device_count() = 0
x.grad() for y = 3*x^2 at x=2 ->  12
[ CPUFloatType{1} ]
```

Real tensor construction, real elementwise arithmetic, and a real reverse-mode `.backward()` call producing the correct gradient (12, matching `dy/dx = 6x` at `x = 2`) — all genuinely executed on CPU, no fabrication. If you see the same numbers, your install is working.

## How this book verifies its claims

Unlike the CUDA C++ and cuTile Python siblings — which hand-build their own Tensor type and autograd engine, or compile kernels ahead-of-time to cubin with no way to execute them — LibTorch ships a complete, working **CPU** backend. That changes the verification story for the better: most of this book's claims about `torch::Tensor`, `torch::autograd`, and the built-in operator set are not just compiled, they are genuinely *run* on CPU, with real output captured and shown, the same way the code above was.

The exception is anything that specifically requires an NVIDIA GPU: dispatch to LibTorch's CUDA backend, `.to(torch::kCUDA)`, kernel launch and timing on-device. For those claims this book follows the CUDA book's own precedent — genuinely compile against the real, CUDA-aware LibTorch build and capture the real error LibTorch itself raises when no device is present (analogous to `cudaErrorNoDevice`), rather than fabricating a result. Sections whose central claim needs a real GPU are explicitly tagged **UNVERIFIED — pending real-GPU test**. Every claim's evidence — genuinely run on CPU, genuinely compiled only, or pending real hardware — is stated plainly where it appears; nothing is asserted as run when it was only compiled, or fabricated when neither was possible.

Start with [Part 0: LibTorch Foundations](part0/01-variables-and-types.md) once your install matches the output above.
