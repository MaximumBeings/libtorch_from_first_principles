# LibTorch From First Principles

**Learning PyTorch's Real C++ Frontend, One Layer at a Time**

!!! info "This book is under development"
    Chapters are written and posted one at a time, in page order, each one carrying real code and honestly labeled output — the same discipline [*CUDA From First Principles*](https://maximumbeings.github.io/cuda-from-first-principles/) is being built with. It is the sixth sibling project to [*Mojo From First Principles*](https://maximumbeings.github.io/mojo-from-first-principles/), [*Triton From First Principles*](https://maximumbeings.github.io/triton-from-first-principles/), [*CUDA From First Principles*](https://maximumbeings.github.io/cuda-from-first-principles/), [*Rust From First Principles*](https://maximumbeings.github.io/rust-from-first-principles/), and [*CuTile Python From First Principles*](https://maximumbeings.github.io/cutile-python-from-first-principles/).

> "The CUDA book asks what a `Tensor` has to be, from nothing, to survive contact with a GPU. This book asks a different question of the same territory: what did the people who actually had to ship one decide? `torch::Tensor` is not a teaching example — it is production code with a decade of hard-won tradeoffs baked into it, and reading it honestly means running it, not just admiring it from a distance."

This book follows the CUDA C++ book's arc chapter for chapter, part for part, all the way through its appendix — but not its method. The CUDA book builds its own `Tensor`, its own autograd engine, and its own kernels entirely by hand, because raw CUDA C++ gives it nothing else to build on. LibTorch is the opposite case: it *is* the thing the CUDA book is building toward, already built, production-tested, and installable in one line. So where CUDA-book Chapter 6 ("The Tensor") hand-writes a shape/stride/ownership model from scratch, this book's Chapter 6 opens `torch::Tensor` and genuinely inspects what its real implementation does for the same concerns. Where CUDA-book Chapters 15 through 17 hand-build a computational graph and a reverse-mode gradient engine, this book's corresponding chapters read and exercise `torch::autograd::Node`, `torch::autograd::Function`, and `.backward()` against the genuine engine, not a reconstruction of one.

The design principles carried through the whole book:

- **The real library, genuinely exercised.** Every claim about what `torch::Tensor`, `torch::autograd`, or a built-in operator does is checked against actual compiled-and-run LibTorch code in this book's own environment, not against documentation or memory of how PyTorch works.
- **Same map, different territory.** Chapter numbers, part groupings, and chapter titles mirror the CUDA book's exactly, so the two books can be read side by side — one hand-built, one production. A departure from that mapping is only made for a reason discovered while researching the real chapter content, and is stated plainly when it happens.
- **CPU execution is real execution, not a fallback.** LibTorch ships a complete, working CPU backend, so unlike the CUDA and cuTile Python siblings, most of this book's tensor and autograd claims are genuinely *run*, not just compiled. What still can't be verified here — anything requiring an actual NVIDIA GPU — is tagged **UNVERIFIED — pending real-GPU test** rather than guessed at.
- **Financial computing ready.** The closing chapter validates these building blocks against the same kind of quantitative-finance problem the Mojo, Triton, CUDA, Rust, and cuTile Python books end on.

<div class="grid cards" markdown>

- :material-torch:{ .lg .middle } **Chapter-for-chapter with the CUDA book**

    Part 0 through Part 7 plus a closing appendix, mapped one-to-one to [*CUDA From First Principles*](https://maximumbeings.github.io/cuda-from-first-principles/)'s own structure, confirmed against its live site.

- :material-cube-outline:{ .lg .middle } **The real `torch::Tensor`**

    Not a rebuild — the actual production type, its real memory layout, ownership model, and device abstraction, genuinely inspected.

- :material-source-branch:{ .lg .middle } **Genuine autograd**

    `torch::autograd::Function`, `torch::autograd::Node`, and `.backward()` exercised against LibTorch's real graph, not a hand-rolled stand-in.

- :material-finance:{ .lg .middle } **Real financial models**

    Quantitative finance examples cross-checked the same way the sibling books close theirs out.

</div>

## How the book is organized

| Part | Focus |
|---|---|
| **Part 0 — LibTorch Foundations** | The same C++ groundwork the CUDA book opens with — types, structs, memory layout, GPU programming concepts, SIMD — read through LibTorch's own use of them |
| **Part 1 — Core Tensor Infrastructure** | `torch::Tensor`'s real shape/stride representation, creation functions, specialized tensor types, device abstraction, and memory management |
| **Part 2 — Basic Tensor Operations** | Element-wise, matrix, and reduction operations against LibTorch's real operator dispatch |
| **Part 3 — Computational Graph Foundation** | `torch::autograd::Node` and how LibTorch actually records the graph |
| **Part 4 — Automatic Differentiation Engine** | `torch::autograd::Function`'s real backward dispatch and LibTorch's genuine gradient computation engine |
| **Part 5 — GPU Acceleration and Performance** | LibTorch's CUDA backend and performance-optimization surface — compiled and inspected genuinely; execution claims pending real-GPU testing |
| **Part 6 — Neural Network Building Blocks** | `torch::nn` layers and LibTorch's advanced feature surface |
| **Part 7 — Financial Computing Applications** | Quantitative finance examples built on the preceding chapters |
| **Appendix** | Installation, practice materials, memory architecture, and specialized topics, mirroring the CUDA book's own appendix |

Start with [Getting Started](getting-started.md) to stand up a verified LibTorch environment, or jump straight into [Part 0: LibTorch Foundations](part0/01-variables-and-types.md).

!!! note "How this book relates to its siblings"
    Mojo, CUDA C++, and Rust each build their own tensor type and their own autodiff engine because none of them has anything else to lean on for this kind of from-scratch teaching arc. Triton and cuTile Python deliberately build neither — they own only the kernel and lean on PyTorch's existing tensor and autograd graph. LibTorch is a fourth position entirely: it is not a kernel DSL leaning on someone else's tensor, and it is not a from-scratch teaching exercise — it is the real, production tensor-and-autograd library the CUDA, Mojo, and Rust books are each independently reconstructing pieces of. This book's job is to map onto the CUDA book's exact chapter structure while replacing "how would you build this by hand" with "what did the people who actually shipped this decide, and does it hold up when you genuinely run it."
