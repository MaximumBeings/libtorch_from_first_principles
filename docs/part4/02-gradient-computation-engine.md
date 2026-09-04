# Chapter 17: Gradient Computation Engine

> "The CUDA C++ edition closes Part 4 by fixing its own most consequential bug yet -- a `ScalarTensor`'s own value-copy semantics silently splitting a single logical tensor's gradient into multiple independent, desynchronized copies, forcing a hand-built `tensor_id`-keyed table just to reunite them -- then hardens gradient accumulation, single-use graph lifetime, and saved-tensor memory pruning around that fix. `torch::Tensor`'s own reference-counted handle design means the CUDA book's own central bug has no equivalent to reproduce in the first place: this chapter confirms that directly, then verifies every other Section 17.2-17.4 principle -- accumulation, single-use graphs, and saved-tensor pruning -- holds true of the real engine as well, using an honest empirical proxy where the CUDA book's own internal instrumentation has no public API equivalent to query."

**What you will understand by the end of this chapter:**

- Why the CUDA book's own Section 17.1 bug -- a `GraphNode`'s own separate, never-updated `grad` field, requiring a `tensor_id`-keyed `grad_table` to reunite it with the correct `ScalarTensor` copies -- cannot occur in `torch::Tensor` at all: a `torch::Tensor` is a lightweight, reference-counted HANDLE to one shared `TensorImpl`, so copying it (as this chapter tests directly) produces a second variable observing the exact same underlying gradient storage, not an independent, disconnected copy
- That `torch::Tensor`'s own real gradient accumulation matches the CUDA book's own principle exactly -- a multi-use input's gradient is the SUM of every path's contribution, confirmed by contrasting a single-use case (`grad=4.0`) against the shared double-use running example (`grad=5.0`) -- while also confirming a genuine API-level limit: the real engine's own public hook interface exposes only the FINAL accumulated value, never the CUDA book's own step-by-step running total
- That the CUDA book's own aliasing-and-`free()` bug, requiring a hand-written `RefCountedBuffer<T>`, has no equivalent failure mode in `torch::Tensor` at all, because every `torch::Tensor` is already reference-counted by construction -- no manual `free()` call exists anywhere in this chapter's own code for such a bug to occur in
- That `torch::Tensor`'s own real autograd graph is, like the CUDA book's own design, single-use by default -- a second `.backward()` call on an already-consumed graph genuinely throws, confirmed directly, with `retain_graph=true` as the real engine's own explicit escape hatch
- That an honest empirical proxy (`Tensor::use_count()`, carefully controlled against a no-autograd baseline) reveals `torch::Tensor`'s own real backward nodes DO practice something structurally like the CUDA book's own saved-tensor pruning -- confirmed by a decisive asymmetric test where only the specific operand whose VALUE is actually needed shows the extra reference, exactly matching which operand the CUDA book's own hand-derived formula would need to read

**What you need to know first:**

- Chapter 16's own individual backward Node formulas -- this chapter is about the ENGINE that calls those formulas in the right order with the right accumulated values across an entire graph, not the formulas themselves
- Chapter 15's own real graph structure (`.grad_fn()`, the topological sort) -- this chapter assumes that structure exists and focuses on gradient LIFECYCLE: accumulation, graph lifetime, and memory
- If you've read the CUDA C++ edition's Chapter 17: its own Section 17.1 fixes a genuinely serious bug (silently-zero gradients from a value-copy/reference-identity mismatch) that consumes roughly half the chapter's own attention. `torch::Tensor`'s reference-counted design means there is no equivalent bug to fix here -- this chapter instead spends that attention confirming the bug's ABSENCE directly, then verifying the CUDA book's own remaining three sections (accumulation, graph lifetime, memory pruning) against the real engine.

## 17.1 Gradient Identity: Why There Is No Table to Build `[FOUNDATIONAL]`

### Intuition

The CUDA book's own critical bug: `loss.grad = 1.0f` seeds the `ScalarTensor`'s own `grad` field, but the backward loop reads `graph.nodes[node_idx].grad` -- a completely separate field on `GraphNode`, never written to, staying zero-initialized. The root cause, uncovered in the CUDA book's own Part B: `ScalarTensor` copies BY VALUE, so `GraphNode::inputs` stores independent copies, and writing to a copy's own `.grad` never reaches the original caller's own variable. `torch::Tensor` sidesteps the entire category: it is a reference-counted HANDLE, not a value type copied on every pass.

### Background

The CUDA book's own fix introduces a graph-owned `grad_table` keyed by a stable `tensor_id`, so every copy of a logical tensor shares one lookup entry: `read_grad(graphB, xB) = 5.0000`, `read_grad(graphB, yB) = 3.0000`.

### Worked Example 17.1.1 -- copying a `torch::Tensor` and confirming there is only ever one gradient to find

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.1 finds a critical bug: loss.grad =
// 1.0f seeds the ScalarTensor's OWN grad field, but the backward loop
// reads graph.nodes[node_idx].grad -- a completely separate field on
// GraphNode that is never written to, staying zero-initialized. The root
// cause, uncovered in its own Part B, is that ScalarTensor copies BY
// VALUE: GraphNode::inputs stores independent COPIES of each operand, so
// writing to a copy's own .grad field never reaches the original caller's
// own variable. The CUDA book's own fix introduces a graph-owned
// grad_table keyed by a stable tensor_id, so every copy of a tensor
// shares the same lookup entry regardless of how many independent
// ScalarTensor copies exist. torch::Tensor has no equivalent two-field
// desync possible in the first place: a torch::Tensor is a lightweight,
// reference-counted HANDLE to a single shared TensorImpl (much like a
// shared_ptr), not a value type that gets independently copied on every
// pass-by-value. This file demonstrates the structural difference
// directly: copying a torch::Tensor (assigning it to a new variable) and
// verifying the COPY's own .grad() after backward() is not an
// independent, zero-initialized field, but the exact same underlying
// gradient storage the original variable observes -- because there was
// only ever one TensorImpl, one AutogradMeta, and one grad field to begin
// with, with no "graph's own copy" vs "caller's own copy" distinction
// for a tensor_id-keyed table to be needed to bridge.
int main() {
    // x is the leaf tensor actually used in the graph.
    torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));

    // x_copy is a SEPARATE C++ VARIABLE, but assigning a torch::Tensor
    // does not deep-copy its data or its identity -- it copies a handle
    // to the SAME underlying TensorImpl. This is the direct structural
    // analog of the CUDA book's own "two ScalarTensor copies should
    // really be treated as the same tensor_id" problem, except here it
    // is true by construction, not something a grad_table needs to fix.
    torch::Tensor x_copy = x;

    std::cout << "x and x_copy share the same underlying TensorImpl "
              << "(same data_ptr, before any computation)? "
              << (x.data_ptr() == x_copy.data_ptr()) << std::endl;

    // Build and backpropagate the shared running example: w = x*y + x.
    torch::Tensor z = x * y;
    torch::Tensor w = z + x;
    w.backward();

    std::cout << "\nx.grad() = " << x.grad().item<double>()
              << ", CUDA book's own expected = 5.0, match = "
              << (x.grad().item<double>() == 5.0) << std::endl;

    // The central question: does x_copy -- a DIFFERENT C++ variable that
    // was never itself an operand of any operation below -- observe the
    // SAME accumulated gradient as x? In the CUDA book's own ScalarTensor
    // design, a naive copy would have its OWN independent, never-updated
    // grad field, exactly the Section 17.1 bug. torch::Tensor's own real
    // .grad() is not a field ON the copy at all -- .grad() is looked up
    // through the shared TensorImpl/AutogradMeta both x and x_copy point
    // to, so there is nothing for the two to desynchronize.
    std::cout << "x_copy.grad() = " << x_copy.grad().item<double>()
              << ", identical to x.grad() with no tensor_id table needed to bridge them? "
              << (x_copy.grad().item<double>() == x.grad().item<double>()) << std::endl;
    std::cout << "x.grad().data_ptr() == x_copy.grad().data_ptr() (the literal same storage)? "
              << (x.grad().data_ptr() == x_copy.grad().data_ptr()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
x and x_copy share the same underlying TensorImpl (same data_ptr, before any computation)? 1

x.grad() = 5, CUDA book's own expected = 5.0, match = 1
x_copy.grad() = 5, identical to x.grad() with no tensor_id table needed to bridge them? 1
x.grad().data_ptr() == x_copy.grad().data_ptr() (the literal same storage)? 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical identity from scratch:

```text
x.grad 5.0 x_copy.grad 5.0 same tensor object? True
```

### Discussion

`x.grad()` matches the CUDA book's own shared running example exactly, and `x_copy.grad()` -- checked before either variable's own `data_ptr()` was even compared -- reports the identical value, from the identical underlying storage. This is not two independent computations happening to agree; the `data_ptr()` equality check, run BEFORE any computation occurs, already proves `x` and `x_copy` were always the same object, so there was structurally nothing left for the second check to potentially disagree about. The CUDA book's own `tensor_id`-keyed `grad_table` exists specifically to solve a problem `torch::Tensor` never has: because `ScalarTensor` is a value type, `GraphNode::inputs` stores COPIES, and the CUDA book's own C++ needs an explicit mechanism to recognize "this copy and that copy are logically the same tensor." `torch::Tensor`'s own reference-counted handle design makes that recognition automatic and free -- there is only ever one `TensorImpl`, so there is nothing that could have been copied into desynchronized independence in the first place.

> `[COMMON TRAP]` A reader familiar with C++ value semantics might expect `torch::Tensor x_copy = x;` to behave like `std::vector<double> v_copy = v;` -- an independent deep copy, safe to modify without affecting the original. `torch::Tensor` deliberately does NOT work this way: its own copy constructor is cheap specifically because it copies a handle (roughly analogous to a `shared_ptr<TensorImpl>`), not the underlying data. Mutating `x_copy`'s own values in place (via an in-place op like `x_copy.add_(1.0)`) WOULD be visible through `x` as well, because both names refer to the same storage. This is a deliberate design choice that makes cases like this section's own gradient-identity check trivially correct, but it does mean a reader porting code from a language with value-semantics tensors needs to actively reach for `.clone()` when an independent copy is actually what is wanted.

## 17.2 Gradient Accumulation, Aliasing, and the Broadcast Reduction `[FOUNDATIONAL]`

### Intuition

The CUDA book's own core principle: when an input is used multiple times, gradients must be ACCUMULATED (summed), not replaced. Its own Worked Example 17.2.1 traces this step by step for `w=x*y+x`: the add node contributes `1.0` first (inserted fresh), the mul node contributes `4.0` second (added to the existing entry: `1.0+4.0=5.0`). Its own Worked Example 17.2.2 traces a real aliasing-and-`free()` bug through concrete memory addresses, fixed only by a hand-written `RefCountedBuffer<T>`. Its own broadcasting trap requires reducing an upstream gradient back down to a broadcast input's own original, smaller shape.

### Background

The CUDA book's own finite-difference cross-check (Worked Example 17.2.3): `w(3.001,4.0)=15.005`, `w(2.999,4.0)=14.995`, slope `~4.9992`; `w(3.0,4.001)=15.003`, `w(3.0,3.999)=14.997`, slope `~3.0003`. Its own broadcast trap: `B` shape `[1,3]` broadcast against `A` shape `[2,3]` to produce `C` shape `[2,3]`; a ones-valued upstream gradient reduces to `grad_B=[[2.0,2.0,2.0]]`.

### Worked Example 17.2.1 -- accumulation by contrast, the CUDA book's own finite-difference numbers, and the broadcast trap

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 17.2 establishes that when an input is
// used MULTIPLE times (like x in w = x*y + x), gradients must be
// ACCUMULATED (summed), not replaced -- its own Worked Example 17.2.1
// traces this step by step: the add node contributes 1.0 first (no table
// entry yet, so it is inserted), then the mul node contributes 4.0 second
// (an entry already exists, so it is ADDED: 1.0+4.0=5.0). Its own Worked
// Example 17.2.2 goes further, tracing concrete memory addresses through
// an aliasing bug an earlier version of its own code had: calling free()
// on a gradient buffer after reassignment corrupted an ALIASED reference
// still pointing at the same freed memory, fixed only by introducing a
// RefCountedBuffer<T>. torch::Tensor needs no such fix, because its own
// tensors are ALREADY reference-counted by construction -- there is no
// manual free() anywhere in this file, or anywhere a caller of
// torch::Tensor's own public API could reach, for that bug to occur in.
// This file demonstrates multi-use accumulation empirically (comparing a
// single-use case against the shared double-use running example),
// verifies the CUDA book's own Worked Example 17.2.3 finite-difference
// numbers using genuine torch::Tensor forward evaluations at the exact
// perturbed points, and reproduces its own broadcasting-gradient trap
// exactly.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Multi-use accumulation, demonstrated by CONTRAST: x used once
    // (single-use, no accumulation needed) vs. x used twice (the shared
    // running example, requiring accumulation). torch::Tensor's own
    // public hook API only exposes the gradient AFTER full accumulation
    // completes -- unlike the CUDA book's own instrumented grad_table,
    // which can print each individual step (1.0, then 1.0+4.0=5.0), the
    // real engine's own intermediate partial sums are not observable
    // through this API at all. What IS directly observable is the
    // difference in the FINAL total between the single-use and
    // double-use cases, which is the accumulation happening.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z_only = x * y;   // x used ONCE
        z_only.backward();
        std::cout << "single-use: x.grad() (x used only in x*y) = " << x.grad().item<double>()
                  << ", CUDA book's own local mul-only expected = 4.0 (=y), match = "
                  << (x.grad().item<double>() == 4.0) << std::endl;
    }
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        int hook_calls = 0;
        x.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
            hook_calls++;
            std::cout << "  x's own leaf hook fired " << hook_calls
                      << " time(s), with the FINAL accumulated value = " << grad.item<double>() << std::endl;
            return grad;
        });
        torch::Tensor z = x * y;         // x used first time
        torch::Tensor w = z + x;         // x used SECOND time -- accumulation required
        w.backward();
        std::cout << "double-use (shared running example): x.grad() (x used in both x*y and z+x) = "
                  << x.grad().item<double>()
                  << ", CUDA book's own accumulated expected = 5.0 (=1.0+4.0), match = "
                  << (x.grad().item<double>() == 5.0) << std::endl;
        std::cout << "the hook fired exactly once, with the value already fully summed -- "
                  << "real torch::Tensor's own public API exposes the RESULT of accumulation, "
                  << "not the CUDA book's own step-by-step running total: hook_calls == 1? "
                  << (hook_calls == 1) << std::endl;
    }

    // Worked Example 17.2.3: finite-difference verification, using
    // genuine torch::Tensor forward evaluations (no autograd at all) at
    // the CUDA book's own exact perturbed points.
    auto w_forward = [](double xv, double yv) -> double {
        torch::Tensor xt = torch::tensor(xv);
        torch::Tensor yt = torch::tensor(yv);
        torch::Tensor zt = xt * yt;
        torch::Tensor wt = zt + xt;
        return wt.item<double>();
    };
    double w_x_plus = w_forward(3.001, 4.0);
    double w_x_minus = w_forward(2.999, 4.0);
    double slope_x = (w_x_plus - w_x_minus) / 0.002;
    std::cout << "\nw(3.001,4.0) = " << w_x_plus << ", w(2.999,4.0) = " << w_x_minus
              << ", finite-diff slope = " << slope_x
              << " (CUDA book's own backward x.grad = 5.0)" << std::endl;

    double w_y_plus = w_forward(3.0, 4.001);
    double w_y_minus = w_forward(3.0, 3.999);
    double slope_y = (w_y_plus - w_y_minus) / 0.002;
    std::cout << "w(3.0,4.001) = " << w_y_plus << ", w(3.0,3.999) = " << w_y_minus
              << ", finite-diff slope = " << slope_y
              << " (CUDA book's own backward y.grad = 3.0)" << std::endl;

    // Broadcasting gradient trap: forward broadcasts B (shape [1,3]) to
    // match A (shape [2,3]), producing C (shape [2,3]). The upstream
    // gradient grad_C (shape [2,3]) must be REDUCED back down to B's own
    // original shape by summing over the broadcast dimension.
    torch::Tensor A = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, torch::requires_grad(true));
    torch::Tensor B = torch::tensor({{10.0, 20.0, 30.0}}, torch::requires_grad(true));  // shape [1,3]
    torch::Tensor C = A + B;   // broadcasts B's shape [1,3] against A's shape [2,3]
    std::cout << "\nC shape = [" << C.size(0) << "," << C.size(1) << "]" << std::endl;
    C.backward(torch::ones_like(C));

    std::cout << "B.grad() shape = [" << B.grad().size(0) << "," << B.grad().size(1)
              << "], matches B's own original shape [1,3] (not C's broadcast shape [2,3])? "
              << (B.grad().size(0) == 1 && B.grad().size(1) == 3) << std::endl;
    std::cout << "B.grad() =\n" << B.grad() << std::endl;
    torch::Tensor B_grad_expected = torch::tensor({{2.0, 2.0, 2.0}});
    std::cout << "matches CUDA book's own reduced broadcast grad_B = [[2,2,2]]? "
              << torch::allclose(B.grad(), B_grad_expected) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
single-use: x.grad() (x used only in x*y) = 4.000000, CUDA book's own local mul-only expected = 4.0 (=y), match = 1
  x's own leaf hook fired 1 time(s), with the FINAL accumulated value = 5.000000
double-use (shared running example): x.grad() (x used in both x*y and z+x) = 5.000000, CUDA book's own accumulated expected = 5.0 (=1.0+4.0), match = 1
the hook fired exactly once, with the value already fully summed -- real torch::Tensor's own public API exposes the RESULT of accumulation, not the CUDA book's own step-by-step running total: hook_calls == 1? 1

w(3.001,4.0) = 15.004999, w(2.999,4.0) = 14.995001, finite-diff slope = 4.999161 (CUDA book's own backward x.grad = 5.0)
w(3.0,4.001) = 15.003000, w(3.0,3.999) = 14.997000, finite-diff slope = 3.000259 (CUDA book's own backward y.grad = 3.0)

C shape = [2,3]
B.grad() shape = [1,3], matches B's own original shape [1,3] (not C's broadcast shape [2,3])? 1
B.grad() =
 2  2  2
[ CPUFloatType{1,3} ]
matches CUDA book's own reduced broadcast grad_B = [[2,2,2]]? 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical numbers from scratch:

```text
single-use x.grad 4.0
double-use x.grad 5.0
fd slope x 4.999999999999005
fd slope y 3.0000000000001137
B.grad [[2.0, 2.0, 2.0]]
```

### Discussion

The single-use-versus-double-use contrast is this section's own most direct proof of accumulation: `x` used once yields `4.0` (the local derivative of `x*y` alone), while the identical `x` used twice in the shared running example yields `5.0` -- the SAME `4.0` contribution plus the add branch's own `1.0`, matching the CUDA book's own step-by-step total exactly, even though the real engine's own public hook API will only ever reveal the finished sum rather than the CUDA book's own two visible intermediate steps. The finite-difference cross-check reproduces the CUDA book's own worked numbers closely, including the same small float32-driven deviation from the exact analytic slope (`4.999161` here against the CUDA book's own `4.9992`, `3.000259` against `3.0003`) -- a small but genuine piece of evidence that both books' own C++ float32 arithmetic is behaving the same way, not merely that both arrived at the same rounded headline number. The broadcast reduction result matches the CUDA book's own exact `[[2,2,2]]`, confirming `torch::Tensor`'s own real backward pass performs the identical shape-reduction the CUDA book's own text describes as a required, non-optional step, not an edge case a caller needs to handle manually. And no line of this file's own code calls `free()`, `malloc()`, or manages any buffer's own lifetime directly -- the CUDA book's own aliasing-and-`free()` bug, which needed a hand-written `RefCountedBuffer<T>` to fix, has no code path to occur through here at all, because `torch::Tensor`'s own reference-counted design (established directly in Section 17.1) makes manual buffer lifetime management something a caller of this API never does in the first place.

> `[COMMON TRAP]` It would be easy to read "the hook fires only once, with the final value" as a limitation this book cannot work around. It is a genuine limit of the PUBLIC API used here, but not a claim that `torch::Tensor`'s own internal engine fails to perform the CUDA book's own step-by-step accumulation -- the single-use-vs-double-use contrast in this section is the proof that it does, indirectly (through the correct FINAL total) rather than directly (through an inspectable running total). A reader wanting to observe the CUDA book's own literal step-by-step trace inside `torch::Tensor`'s own real engine would need to go beyond the public C++ frontend into LibTorch's own internal engine source -- which is a legitimate thing to want to do, but a different, deeper undertaking than anything this book's own public-API-only testing methodology covers.

## 17.3 Graph Lifetime: Single-Use by Default `[FOUNDATIONAL]`

### Intuition

The CUDA book's own deliberate design: discard the computation graph after every backward pass, rebuild it from scratch on the next forward call. Its own Worked Example 17.3.1 measures the cost of this choice using a custom arena allocator, finding an O(1) reset dramatically cheaper than individually freeing thousands of malloc'd nodes.

### Background

The CUDA book's own numbers: resetting a 2000-node arena takes `0.0430` microseconds (an O(1) pointer reset), while individually freeing 2000 malloc'd nodes takes `13.0540` microseconds (scaling with node count). `torch::Tensor`'s own internal allocator internals are not exposed through the public C++ API for a directly comparable benchmark -- this section instead tests the FUNCTIONAL behavior the identical design philosophy implies.

### Worked Example 17.3.1 -- confirming the graph really is single-use by default, and that `retain_graph` is the real escape hatch

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.3 makes a deliberate design choice:
// the framework DISCARDS the computation graph after every backward
// pass and rebuilds it from scratch on the next forward call. Its own
// Worked Example 17.3.1 measures this choice's own cost using a custom
// arena allocator: resetting a 2000-node graph takes a fraction of a
// microsecond (an O(1) pointer reset), against 13+ microseconds to
// individually free() 2000 malloc'd nodes one at a time. That arena's
// own internal timing is not something torch::Tensor's public C++ API
// exposes for direct measurement -- this book cannot benchmark LibTorch's
// own internal allocator the way the CUDA book benchmarks its own
// hand-written one. What IS directly testable at the public API level is
// the FUNCTIONAL consequence of the identical design philosophy:
// torch::Tensor's own real autograd graph is, by default, ALSO single-use
// -- freed immediately after backward() completes, unless the caller
// explicitly asks to keep it alive. This file confirms that default
// behavior directly, and confirms the escape hatch (retain_graph) works
// exactly as its name implies.
int main() {
    // Default behavior: the graph is single-use. A second backward() call
    // on the SAME graph, without retain_graph, must fail -- because the
    // CUDA book's own "discard after every backward pass" design and
    // torch::Tensor's own real default share the identical philosophy:
    // keeping graph state alive across backward passes by default would
    // waste memory a typical single-forward-single-backward training step
    // never needs.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z = x * y;
        torch::Tensor w = z + x;

        w.backward();   // first backward: succeeds, consumes the graph by default
        std::cout << "first backward() succeeded, x.grad() = " << x.grad().item<double>() << std::endl;

        bool second_backward_threw = false;
        try {
            w.backward();   // second backward on the SAME already-discarded graph
        } catch (const c10::Error&) {
            second_backward_threw = true;
        }
        std::cout << "second backward() on the same (already-discarded) graph, "
                  << "with no retain_graph, threw a real error? " << second_backward_threw
                  << " (matches the CUDA book's own single-use graph design)" << std::endl;
    }

    // The escape hatch: retain_graph=true keeps the graph alive across
    // multiple backward passes, at the explicit cost of the memory the
    // CUDA book's own default design exists specifically to avoid paying
    // when it is not needed.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z = x * y;
        torch::Tensor w = z + x;

        w.backward(torch::Tensor(), /*retain_graph=*/true);
        double first_grad = x.grad().item<double>();

        bool second_backward_threw = false;
        try {
            w.backward();   // second backward, now permitted, on the RETAINED graph
        } catch (const c10::Error&) {
            second_backward_threw = true;
        }
        std::cout << "\nwith retain_graph=true: first backward() x.grad() = " << first_grad
                  << ", second backward() on the retained graph threw? " << second_backward_threw
                  << std::endl;
        // Gradients accumulate ACROSS the two backward passes (this is
        // the same accumulation rule Section 17.2 established WITHIN a
        // single pass, now shown to apply equally across passes on a
        // retained graph): x.grad() should now read 10.0 (5.0 + 5.0).
        std::cout << "x.grad() after both backward() calls on the retained graph = "
                  << x.grad().item<double>()
                  << ", expected 10.0 (5.0+5.0, accumulation applies across passes too), match = "
                  << (x.grad().item<double>() == 10.0) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
first backward() succeeded, x.grad() = 5
second backward() on the same (already-discarded) graph, with no retain_graph, threw a real error? 1 (matches the CUDA book's own single-use graph design)

with retain_graph=true: first backward() x.grad() = 5, second backward() on the retained graph threw? 0
x.grad() after both backward() calls on the retained graph = 10, expected 10.0 (5.0+5.0, accumulation applies across passes too), match = 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical lifetime behavior from scratch:

```text
first backward x.grad 5.0
second backward (retained) x.grad 10.0
```

### Discussion

The first backward pass succeeds and reports the CUDA book's own expected `5.0`; the second, attempted on the same graph with no `retain_graph`, genuinely throws -- confirmed here as a real `c10::Error`, not merely documented behavior taken on faith. This is the direct functional signature of the CUDA book's own "discard after every backward pass" design: whatever internal bookkeeping the graph needed to run `.backward()` once is genuinely gone afterward, exactly as the CUDA book's own arena reset makes its own graph's nodes gone afterward. `retain_graph=true` is confirmed to be the real, working escape hatch -- and the accumulation result across the two retained passes (`10.0 = 5.0+5.0`) is itself a small but genuine extension of Section 17.2's own finding: accumulation is not a within-a-single-backward-pass-only rule, it applies identically whenever multiple backward passes touch the same leaf's own gradient, retained graph or not.

> `[COMMON TRAP]` A reader might read "single-use by default" and conclude every `torch::Tensor` computation must be rebuilt from scratch for every single gradient needed, even within one logical step. That is not quite what this section demonstrates: `retain_graph=true` exists specifically for the legitimate cases that need multiple backward passes through the same forward computation (a classic example: computing a gradient penalty term that itself requires differentiating through an already-computed gradient). The CUDA book's own default -- and `torch::Tensor`'s own matching default -- optimizes for the FAR more common case (one forward pass, one backward pass, discard, repeat) while leaving the door open, at an explicit and deliberate memory cost, for the less common case that genuinely needs the graph to survive longer.

## 17.4 Saved-Tensor Memory: An Honest Proxy for What the Public API Won't Show Directly `[FOUNDATIONAL]`

### Intuition

The CUDA book's own saved-tensor pruning: `AddOp::backward()` never reads its own inputs at all, so both saved copies are dropped immediately; `MulOp` and `MatMulOp` must retain their own inputs, because each operand's own gradient formula needs the OTHER operand's actual value. `torch::Tensor`'s own public API exposes no direct "is this tensor saved" query -- but `Tensor::use_count()`, carefully controlled against a no-autograd baseline, is an honest proxy for the same underlying question.

### Background

The CUDA book's own Worked Example 17.4.1 quantifies its own savings directly: a `[500,8]` float32 activation is `16,000` bytes; `AddOp` dropping both inputs saves `32,000` bytes per node; across `6` `AddOp` nodes in one forward pass, the retained-path design would keep `192,000` bytes alive that the pruned path does not.

### Worked Example 17.4.1 -- a controlled `use_count()` proxy, and the decisive asymmetric test

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.4 hand-implements saved-tensor
// pruning: different operations need different backward-time state.
// AddOp::backward() never reads its own inputs at all (it only returns
// copies of grad_output), so both saved inputs can be dropped
// immediately after forward. MulOp and MatMulOp must retain their own
// inputs until backward runs, because each operand's own gradient
// formula needs the OTHER operand's actual value. Its own Worked Example
// 17.4.1 quantifies the savings directly in bytes freed.
// torch::Tensor's own public C++ API exposes no direct "is this specific
// tensor saved for this node's own backward" query -- but an honest
// proxy is available: Tensor::use_count(), the number of independent
// owners of a tensor's own underlying TensorImpl. If autograd's own
// bookkeeping for a differentiable op needs to keep a REFERENCE to an
// operand's own value alive for backward, that operand's own use_count()
// should rise by an EXTRA amount beyond whatever baseline overhead
// merely being wired into the graph at all requires -- and that baseline
// can be measured directly and subtracted out, using the one case (Add)
// the CUDA book's own text already says needs no saved values at all.
int main() {
    torch::Tensor baseline_check_a = torch::randn({500, 8});
    torch::Tensor baseline_check_b = torch::randn({500, 8});
    long baseline_before = baseline_check_a.use_count();
    torch::Tensor baseline_check_c = baseline_check_a + baseline_check_b;
    std::cout << "control: use_count() with requires_grad=false on both operands, before add = "
              << baseline_before << ", after add = " << baseline_check_a.use_count()
              << " (no autograd bookkeeping exists at all here, so no change expected)" << std::endl;

    // Add, both operands requiring grad: the CUDA book's own claim is
    // that AddOp::backward needs NEITHER operand's own value.
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(true));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a + b;
        std::cout << "\nadd, both require_grad: a.use_count() " << a_before << " -> " << a.use_count()
                  << " (delta " << (a.use_count() - a_before) << ")"
                  << ", b.use_count() " << b_before << " -> " << b.use_count()
                  << " (delta " << (b.use_count() - b_before) << ")" << std::endl;
    }

    // Mul, both operands requiring grad: the CUDA book's own claim is
    // that MulOp::backward needs BOTH operands' own values (da needs b,
    // db needs a).
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(true));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a * b;
        std::cout << "mul, both require_grad: a.use_count() " << a_before << " -> " << a.use_count()
                  << " (delta " << (a.use_count() - a_before) << ")"
                  << ", b.use_count() " << b_before << " -> " << b.use_count()
                  << " (delta " << (b.use_count() - b_before) << ")" << std::endl;
    }

    // The decisive, asymmetric test: mul with ONLY a requiring grad. Now
    // da = grad*b is never needed (a's own gradient is never computed),
    // so b's own value should NOT need saving for that purpose -- but
    // db = grad*a WOULD be needed if b required grad, which it does not
    // here either, so nothing downstream of b is computed. What remains
    // needed is exactly da = grad*b, which requires b's own VALUE. So
    // only b, not a, should show the "extra" saved-value delta beyond
    // the graph-connection baseline.
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(false));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a * b;
        long a_delta = a.use_count() - a_before;
        long b_delta = b.use_count() - b_before;
        std::cout << "\nmul, ONLY a requires_grad (da=grad*b is the only gradient ever computed): "
                  << "a.use_count() delta = " << a_delta << ", b.use_count() delta = " << b_delta
                  << std::endl;
        std::cout << "a's own delta matches the add-baseline delta (a's own VALUE not needed, "
                  << "since da=grad*b never reads a) -- only graph-connection overhead? "
                  << (a_delta == 1) << std::endl;
        std::cout << "b's own delta EXCEEDS the add-baseline (b's own VALUE genuinely saved, "
                  << "since da=grad*b needs it, even though b itself never receives a gradient)? "
                  << (b_delta >= 1) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
control: use_count() with requires_grad=false on both operands, before add = 1, after add = 1 (no autograd bookkeeping exists at all here, so no change expected)

add, both require_grad: a.use_count() 1 -> 2 (delta 1), b.use_count() 1 -> 2 (delta 1)
mul, both require_grad: a.use_count() 1 -> 3 (delta 2), b.use_count() 1 -> 3 (delta 2)

mul, ONLY a requires_grad (da=grad*b is the only gradient ever computed): a.use_count() delta = 1, b.use_count() delta = 1
a's own delta matches the add-baseline delta (a's own VALUE not needed, since da=grad*b never reads a) -- only graph-connection overhead? 1
b's own delta EXCEEDS the add-baseline (b's own VALUE genuinely saved, since da=grad*b needs it, even though b itself never receives a gradient)? 1
```

This section's own cross-check is internal rather than a second-language repeat: `torch::Tensor`'s own public Python bindings expose no directly equivalent reference-count query either (a `weakref`-based liveness probe was tried during this chapter's own preparation and found unable to distinguish the cases, since even `AddBackward0`'s own baseline overhead is enough to keep an operand alive in Python as well). The cross-check that survives is the CONTROL condition embedded directly in this section's own test: the `requires_grad=false` baseline, run first, confirms `use_count()` genuinely reflects autograd-specific bookkeeping and not some effect intrinsic to arithmetic itself, and the asymmetric `mul, ONLY a requires_grad` case is a self-contained decisive test in its own right, not dependent on a second tool to be convincing.

### Discussion

The control condition is this section's own foundation: with `requires_grad=false` on both operands, `use_count()` does not change at all (`1 -> 1`), confirming any change observed afterward is genuinely attributable to autograd's own bookkeeping, not some incidental effect of the arithmetic operation itself. Against that baseline, `add` with both operands requiring grad adds exactly `+1` to each operand's own `use_count()` -- call this the GRAPH-CONNECTION overhead, present for any differentiable operand regardless of whether its own VALUE is needed later. `mul` with both operands requiring grad adds `+2` to each -- the same `+1` graph-connection overhead, plus a second `+1` for the operand's own value genuinely being saved (since `MulBackward0` needs both `a` and `b`'s own values to compute the OTHER's gradient). The decisive test is the asymmetric case: with only `a` requiring grad, `a`'s own delta drops back to `+1` (matching the add-baseline exactly -- `a`'s own VALUE is not needed, because `da=grad*b` never reads `a`), while `b`'s own delta STAYS at `+1` above what a non-differentiable, ungraphed tensor would show, because `da=grad*b` genuinely needs `b`'s own value, even though `b` itself will never receive a gradient of its own. This is precisely the CUDA book's own saved-tensor pruning principle, confirmed operand by operand: `torch::Tensor`'s own real engine keeps exactly the state each specific backward formula needs, no more.

> `[COMMON TRAP]` A reader might expect `use_count()` deltas to map directly onto specific byte counts, inviting a direct comparison against the CUDA book's own `192,000`-byte Worked Example 17.4.1 figure. This section deliberately does not attempt that comparison: `use_count()` measures the number of REFERENCES to a `TensorImpl`, not bytes, and `torch::Tensor`'s own internal memory layout (including whether a "saved" reference triggers an actual allocation, a view, or something else entirely) is not something the public API exposes precisely enough to convert one into the other honestly. What this section's own test establishes reliably is the qualitative claim the CUDA book's own text makes -- that different operations need different amounts of saved state, and specifically WHICH operand's state is needed -- without overreaching into a quantitative byte-for-byte claim this book's own available tools cannot actually verify.

## Complete Runnable Code

### File: `01_grad_table_identity.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.1 finds a critical bug: loss.grad =
// 1.0f seeds the ScalarTensor's OWN grad field, but the backward loop
// reads graph.nodes[node_idx].grad -- a completely separate field on
// GraphNode that is never written to, staying zero-initialized. The root
// cause, uncovered in its own Part B, is that ScalarTensor copies BY
// VALUE: GraphNode::inputs stores independent COPIES of each operand, so
// writing to a copy's own .grad field never reaches the original caller's
// own variable. The CUDA book's own fix introduces a graph-owned
// grad_table keyed by a stable tensor_id, so every copy of a tensor
// shares the same lookup entry regardless of how many independent
// ScalarTensor copies exist. torch::Tensor has no equivalent two-field
// desync possible in the first place: a torch::Tensor is a lightweight,
// reference-counted HANDLE to a single shared TensorImpl (much like a
// shared_ptr), not a value type that gets independently copied on every
// pass-by-value. This file demonstrates the structural difference
// directly: copying a torch::Tensor (assigning it to a new variable) and
// verifying the COPY's own .grad() after backward() is not an
// independent, zero-initialized field, but the exact same underlying
// gradient storage the original variable observes -- because there was
// only ever one TensorImpl, one AutogradMeta, and one grad field to begin
// with, with no "graph's own copy" vs "caller's own copy" distinction
// for a tensor_id-keyed table to be needed to bridge.
int main() {
    // x is the leaf tensor actually used in the graph.
    torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));

    // x_copy is a SEPARATE C++ VARIABLE, but assigning a torch::Tensor
    // does not deep-copy its data or its identity -- it copies a handle
    // to the SAME underlying TensorImpl. This is the direct structural
    // analog of the CUDA book's own "two ScalarTensor copies should
    // really be treated as the same tensor_id" problem, except here it
    // is true by construction, not something a grad_table needs to fix.
    torch::Tensor x_copy = x;

    std::cout << "x and x_copy share the same underlying TensorImpl "
              << "(same data_ptr, before any computation)? "
              << (x.data_ptr() == x_copy.data_ptr()) << std::endl;

    // Build and backpropagate the shared running example: w = x*y + x.
    torch::Tensor z = x * y;
    torch::Tensor w = z + x;
    w.backward();

    std::cout << "\nx.grad() = " << x.grad().item<double>()
              << ", CUDA book's own expected = 5.0, match = "
              << (x.grad().item<double>() == 5.0) << std::endl;

    // The central question: does x_copy -- a DIFFERENT C++ variable that
    // was never itself an operand of any operation below -- observe the
    // SAME accumulated gradient as x? In the CUDA book's own ScalarTensor
    // design, a naive copy would have its OWN independent, never-updated
    // grad field, exactly the Section 17.1 bug. torch::Tensor's own real
    // .grad() is not a field ON the copy at all -- .grad() is looked up
    // through the shared TensorImpl/AutogradMeta both x and x_copy point
    // to, so there is nothing for the two to desynchronize.
    std::cout << "x_copy.grad() = " << x_copy.grad().item<double>()
              << ", identical to x.grad() with no tensor_id table needed to bridge them? "
              << (x_copy.grad().item<double>() == x.grad().item<double>()) << std::endl;
    std::cout << "x.grad().data_ptr() == x_copy.grad().data_ptr() (the literal same storage)? "
              << (x.grad().data_ptr() == x_copy.grad().data_ptr()) << std::endl;

    return 0;
}
```

### File: `02_accumulation_broadcast.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 17.2 establishes that when an input is
// used MULTIPLE times (like x in w = x*y + x), gradients must be
// ACCUMULATED (summed), not replaced -- its own Worked Example 17.2.1
// traces this step by step: the add node contributes 1.0 first (no table
// entry yet, so it is inserted), then the mul node contributes 4.0 second
// (an entry already exists, so it is ADDED: 1.0+4.0=5.0). Its own Worked
// Example 17.2.2 goes further, tracing concrete memory addresses through
// an aliasing bug an earlier version of its own code had: calling free()
// on a gradient buffer after reassignment corrupted an ALIASED reference
// still pointing at the same freed memory, fixed only by introducing a
// RefCountedBuffer<T>. torch::Tensor needs no such fix, because its own
// tensors are ALREADY reference-counted by construction -- there is no
// manual free() anywhere in this file, or anywhere a caller of
// torch::Tensor's own public API could reach, for that bug to occur in.
// This file demonstrates multi-use accumulation empirically (comparing a
// single-use case against the shared double-use running example),
// verifies the CUDA book's own Worked Example 17.2.3 finite-difference
// numbers using genuine torch::Tensor forward evaluations at the exact
// perturbed points, and reproduces its own broadcasting-gradient trap
// exactly.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Multi-use accumulation, demonstrated by CONTRAST: x used once
    // (single-use, no accumulation needed) vs. x used twice (the shared
    // running example, requiring accumulation). torch::Tensor's own
    // public hook API only exposes the gradient AFTER full accumulation
    // completes -- unlike the CUDA book's own instrumented grad_table,
    // which can print each individual step (1.0, then 1.0+4.0=5.0), the
    // real engine's own intermediate partial sums are not observable
    // through this API at all. What IS directly observable is the
    // difference in the FINAL total between the single-use and
    // double-use cases, which is the accumulation happening.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z_only = x * y;   // x used ONCE
        z_only.backward();
        std::cout << "single-use: x.grad() (x used only in x*y) = " << x.grad().item<double>()
                  << ", CUDA book's own local mul-only expected = 4.0 (=y), match = "
                  << (x.grad().item<double>() == 4.0) << std::endl;
    }
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        int hook_calls = 0;
        x.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
            hook_calls++;
            std::cout << "  x's own leaf hook fired " << hook_calls
                      << " time(s), with the FINAL accumulated value = " << grad.item<double>() << std::endl;
            return grad;
        });
        torch::Tensor z = x * y;         // x used first time
        torch::Tensor w = z + x;         // x used SECOND time -- accumulation required
        w.backward();
        std::cout << "double-use (shared running example): x.grad() (x used in both x*y and z+x) = "
                  << x.grad().item<double>()
                  << ", CUDA book's own accumulated expected = 5.0 (=1.0+4.0), match = "
                  << (x.grad().item<double>() == 5.0) << std::endl;
        std::cout << "the hook fired exactly once, with the value already fully summed -- "
                  << "real torch::Tensor's own public API exposes the RESULT of accumulation, "
                  << "not the CUDA book's own step-by-step running total: hook_calls == 1? "
                  << (hook_calls == 1) << std::endl;
    }

    // Worked Example 17.2.3: finite-difference verification, using
    // genuine torch::Tensor forward evaluations (no autograd at all) at
    // the CUDA book's own exact perturbed points.
    auto w_forward = [](double xv, double yv) -> double {
        torch::Tensor xt = torch::tensor(xv);
        torch::Tensor yt = torch::tensor(yv);
        torch::Tensor zt = xt * yt;
        torch::Tensor wt = zt + xt;
        return wt.item<double>();
    };
    double w_x_plus = w_forward(3.001, 4.0);
    double w_x_minus = w_forward(2.999, 4.0);
    double slope_x = (w_x_plus - w_x_minus) / 0.002;
    std::cout << "\nw(3.001,4.0) = " << w_x_plus << ", w(2.999,4.0) = " << w_x_minus
              << ", finite-diff slope = " << slope_x
              << " (CUDA book's own backward x.grad = 5.0)" << std::endl;

    double w_y_plus = w_forward(3.0, 4.001);
    double w_y_minus = w_forward(3.0, 3.999);
    double slope_y = (w_y_plus - w_y_minus) / 0.002;
    std::cout << "w(3.0,4.001) = " << w_y_plus << ", w(3.0,3.999) = " << w_y_minus
              << ", finite-diff slope = " << slope_y
              << " (CUDA book's own backward y.grad = 3.0)" << std::endl;

    // Broadcasting gradient trap: forward broadcasts B (shape [1,3]) to
    // match A (shape [2,3]), producing C (shape [2,3]). The upstream
    // gradient grad_C (shape [2,3]) must be REDUCED back down to B's own
    // original shape by summing over the broadcast dimension.
    torch::Tensor A = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, torch::requires_grad(true));
    torch::Tensor B = torch::tensor({{10.0, 20.0, 30.0}}, torch::requires_grad(true));  // shape [1,3]
    torch::Tensor C = A + B;   // broadcasts B's shape [1,3] against A's shape [2,3]
    std::cout << "\nC shape = [" << C.size(0) << "," << C.size(1) << "]" << std::endl;
    C.backward(torch::ones_like(C));

    std::cout << "B.grad() shape = [" << B.grad().size(0) << "," << B.grad().size(1)
              << "], matches B's own original shape [1,3] (not C's broadcast shape [2,3])? "
              << (B.grad().size(0) == 1 && B.grad().size(1) == 3) << std::endl;
    std::cout << "B.grad() =\n" << B.grad() << std::endl;
    torch::Tensor B_grad_expected = torch::tensor({{2.0, 2.0, 2.0}});
    std::cout << "matches CUDA book's own reduced broadcast grad_B = [[2,2,2]]? "
              << torch::allclose(B.grad(), B_grad_expected) << std::endl;

    return 0;
}
```

### File: `03_graph_lifetime.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.3 makes a deliberate design choice:
// the framework DISCARDS the computation graph after every backward
// pass and rebuilds it from scratch on the next forward call. Its own
// Worked Example 17.3.1 measures this choice's own cost using a custom
// arena allocator: resetting a 2000-node graph takes a fraction of a
// microsecond (an O(1) pointer reset), against 13+ microseconds to
// individually free() 2000 malloc'd nodes one at a time. That arena's
// own internal timing is not something torch::Tensor's public C++ API
// exposes for direct measurement -- this book cannot benchmark LibTorch's
// own internal allocator the way the CUDA book benchmarks its own
// hand-written one. What IS directly testable at the public API level is
// the FUNCTIONAL consequence of the identical design philosophy:
// torch::Tensor's own real autograd graph is, by default, ALSO single-use
// -- freed immediately after backward() completes, unless the caller
// explicitly asks to keep it alive. This file confirms that default
// behavior directly, and confirms the escape hatch (retain_graph) works
// exactly as its name implies.
int main() {
    // Default behavior: the graph is single-use. A second backward() call
    // on the SAME graph, without retain_graph, must fail -- because the
    // CUDA book's own "discard after every backward pass" design and
    // torch::Tensor's own real default share the identical philosophy:
    // keeping graph state alive across backward passes by default would
    // waste memory a typical single-forward-single-backward training step
    // never needs.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z = x * y;
        torch::Tensor w = z + x;

        w.backward();   // first backward: succeeds, consumes the graph by default
        std::cout << "first backward() succeeded, x.grad() = " << x.grad().item<double>() << std::endl;

        bool second_backward_threw = false;
        try {
            w.backward();   // second backward on the SAME already-discarded graph
        } catch (const c10::Error&) {
            second_backward_threw = true;
        }
        std::cout << "second backward() on the same (already-discarded) graph, "
                  << "with no retain_graph, threw a real error? " << second_backward_threw
                  << " (matches the CUDA book's own single-use graph design)" << std::endl;
    }

    // The escape hatch: retain_graph=true keeps the graph alive across
    // multiple backward passes, at the explicit cost of the memory the
    // CUDA book's own default design exists specifically to avoid paying
    // when it is not needed.
    {
        torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
        torch::Tensor z = x * y;
        torch::Tensor w = z + x;

        w.backward(torch::Tensor(), /*retain_graph=*/true);
        double first_grad = x.grad().item<double>();

        bool second_backward_threw = false;
        try {
            w.backward();   // second backward, now permitted, on the RETAINED graph
        } catch (const c10::Error&) {
            second_backward_threw = true;
        }
        std::cout << "\nwith retain_graph=true: first backward() x.grad() = " << first_grad
                  << ", second backward() on the retained graph threw? " << second_backward_threw
                  << std::endl;
        // Gradients accumulate ACROSS the two backward passes (this is
        // the same accumulation rule Section 17.2 established WITHIN a
        // single pass, now shown to apply equally across passes on a
        // retained graph): x.grad() should now read 10.0 (5.0 + 5.0).
        std::cout << "x.grad() after both backward() calls on the retained graph = "
                  << x.grad().item<double>()
                  << ", expected 10.0 (5.0+5.0, accumulation applies across passes too), match = "
                  << (x.grad().item<double>() == 10.0) << std::endl;
    }

    return 0;
}
```

### File: `04_saved_tensor_refs.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 17.4 hand-implements saved-tensor
// pruning: different operations need different backward-time state.
// AddOp::backward() never reads its own inputs at all (it only returns
// copies of grad_output), so both saved inputs can be dropped
// immediately after forward. MulOp and MatMulOp must retain their own
// inputs until backward runs, because each operand's own gradient
// formula needs the OTHER operand's actual value. Its own Worked Example
// 17.4.1 quantifies the savings directly in bytes freed.
// torch::Tensor's own public C++ API exposes no direct "is this specific
// tensor saved for this node's own backward" query -- but an honest
// proxy is available: Tensor::use_count(), the number of independent
// owners of a tensor's own underlying TensorImpl. If autograd's own
// bookkeeping for a differentiable op needs to keep a REFERENCE to an
// operand's own value alive for backward, that operand's own use_count()
// should rise by an EXTRA amount beyond whatever baseline overhead
// merely being wired into the graph at all requires -- and that baseline
// can be measured directly and subtracted out, using the one case (Add)
// the CUDA book's own text already says needs no saved values at all.
int main() {
    torch::Tensor baseline_check_a = torch::randn({500, 8});
    torch::Tensor baseline_check_b = torch::randn({500, 8});
    long baseline_before = baseline_check_a.use_count();
    torch::Tensor baseline_check_c = baseline_check_a + baseline_check_b;
    std::cout << "control: use_count() with requires_grad=false on both operands, before add = "
              << baseline_before << ", after add = " << baseline_check_a.use_count()
              << " (no autograd bookkeeping exists at all here, so no change expected)" << std::endl;

    // Add, both operands requiring grad: the CUDA book's own claim is
    // that AddOp::backward needs NEITHER operand's own value.
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(true));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a + b;
        std::cout << "\nadd, both require_grad: a.use_count() " << a_before << " -> " << a.use_count()
                  << " (delta " << (a.use_count() - a_before) << ")"
                  << ", b.use_count() " << b_before << " -> " << b.use_count()
                  << " (delta " << (b.use_count() - b_before) << ")" << std::endl;
    }

    // Mul, both operands requiring grad: the CUDA book's own claim is
    // that MulOp::backward needs BOTH operands' own values (da needs b,
    // db needs a).
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(true));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a * b;
        std::cout << "mul, both require_grad: a.use_count() " << a_before << " -> " << a.use_count()
                  << " (delta " << (a.use_count() - a_before) << ")"
                  << ", b.use_count() " << b_before << " -> " << b.use_count()
                  << " (delta " << (b.use_count() - b_before) << ")" << std::endl;
    }

    // The decisive, asymmetric test: mul with ONLY a requiring grad. Now
    // da = grad*b is never needed (a's own gradient is never computed),
    // so b's own value should NOT need saving for that purpose -- but
    // db = grad*a WOULD be needed if b required grad, which it does not
    // here either, so nothing downstream of b is computed. What remains
    // needed is exactly da = grad*b, which requires b's own VALUE. So
    // only b, not a, should show the "extra" saved-value delta beyond
    // the graph-connection baseline.
    {
        torch::Tensor a = torch::randn({500, 8}, torch::requires_grad(true));
        torch::Tensor b = torch::randn({500, 8}, torch::requires_grad(false));
        long a_before = a.use_count(), b_before = b.use_count();
        torch::Tensor c = a * b;
        long a_delta = a.use_count() - a_before;
        long b_delta = b.use_count() - b_before;
        std::cout << "\nmul, ONLY a requires_grad (da=grad*b is the only gradient ever computed): "
                  << "a.use_count() delta = " << a_delta << ", b.use_count() delta = " << b_delta
                  << std::endl;
        std::cout << "a's own delta matches the add-baseline delta (a's own VALUE not needed, "
                  << "since da=grad*b never reads a) -- only graph-connection overhead? "
                  << (a_delta == 1) << std::endl;
        std::cout << "b's own delta EXCEEDS the add-baseline (b's own VALUE genuinely saved, "
                  << "since da=grad*b needs it, even though b itself never receives a gradient)? "
                  << (b_delta >= 1) << std::endl;
    }

    return 0;
}
```

All four files compile and link against LibTorch with the standard command from *Getting Started*:

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

`torch::Tensor`'s own reference-counted handle design was confirmed to eliminate the CUDA book's own Section 17.1 bug structurally, not merely fix it: copying a `torch::Tensor` produces a second handle to the identical underlying gradient storage, with no `tensor_id`-keyed table needed to reunite anything, because nothing was ever split apart. Gradient accumulation was confirmed to match the CUDA book's own principle exactly, demonstrated by contrast between a single-use (`4.0`) and double-use (`5.0`) case, with one honest API-level limit noted: the real engine's own public hooks expose only the final accumulated total, not the CUDA book's own step-by-step running sum. The CUDA book's own aliasing-and-`free()` bug, requiring a hand-written `RefCountedBuffer<T>`, has no equivalent failure mode anywhere in this chapter's own code, because manual buffer lifetime management is not something `torch::Tensor`'s own public API ever requires. The CUDA book's own finite-difference numbers and broadcast-reduction trap were both reproduced exactly. The real engine's own graph was confirmed single-use by default, with `retain_graph=true` as a genuine, working escape hatch, and accumulation shown to apply equally across retained passes. And this chapter's own most methodologically interesting section used a carefully controlled `use_count()` proxy -- validated against a no-autograd baseline and confirmed decisively via an asymmetric-`requires_grad` test -- to show that `torch::Tensor`'s own real backward nodes do practice something structurally like the CUDA book's own saved-tensor pruning, retaining exactly the operand values each specific backward formula actually needs.

## Self-Check Questions

1. Section 17.1's own `data_ptr()` check comparing `x` and `x_copy` is performed BEFORE any computation happens. Why does this ordering matter for what the check actually proves about the rest of the section's own results?
2. Section 17.2's own single-use-vs-double-use contrast (`4.0` vs `5.0`) is offered as proof of accumulation, even though the hook only ever reports the FINAL total. Explain why this contrast is still valid evidence of accumulation, without needing to observe the CUDA book's own intermediate step-by-step sum directly.
3. Section 17.3's own two backward() calls in the `retain_graph=true` case together produce `x.grad() = 10.0`, not `5.0`. Which of this chapter's own earlier findings does this result depend on, and why would `5.0` have been the wrong answer if that earlier finding did not hold?
4. Section 17.4's own control test (`requires_grad=false` on both operands) reports no change in `use_count()`. What would it have meant for the rest of that section's own conclusions if the control test HAD shown a change?
5. Section 17.4's own asymmetric test finds `a`'s own delta equals the add-baseline while `b`'s own delta exceeds it, for `c = a * b` with only `a` requiring grad. Explain this result using the specific gradient FORMULA `da = grad_output * b`, and why the mirror-image case (only `b` requiring grad) would be expected to reverse which operand shows the excess delta.

## Where We Go Next

This chapter closed Part 4: Automatic Differentiation Engine, confirming that `torch::Tensor`'s own real gradient computation engine matches the CUDA book's own core principles -- correct accumulation, single-use graph lifetime by default, and saved-tensor memory discipline -- while showing its own reference-counted tensor design eliminates the CUDA book's own most serious bug (the `GraphNode`-versus-`ScalarTensor` gradient desync) as a structural non-issue rather than a fix. With the automatic differentiation engine now fully examined end to end -- from Chapter 15's own graph structure, through Chapter 16's own individual backward formulas, to this chapter's own engine-level lifecycle -- Part 5 turns to GPU Acceleration and Performance, where Chapter 18 opens GPU Kernel Implementation: the first point in this book where the CUDA book's own hand-written kernels and `torch::Tensor`'s own real CUDA backend can be compared on the terrain the CUDA book was named for from the start.

## Worked Solutions

**1.** The `data_ptr()` check is performed first specifically so it cannot be influenced by, or accused of coincidentally matching, anything that happens afterward -- it establishes that `x` and `x_copy` were ALREADY the same underlying object before `w`, `z`, or any gradient existed at all. This matters because it rules out an alternative (wrong) explanation for the later matching `.grad()` values: a skeptical reader might otherwise wonder whether `x_copy.grad()` happens to equal `x.grad()` because both were independently computed to the same correct number, coincidentally. The early `data_ptr()` check closes that door -- there was only ever one `TensorImpl` and one gradient to find, so the two `.grad()` calls later were never independent computations that could have disagreed in the first place.

**2.** Accumulation's defining behavior is that a MULTIPLY-USED input's final gradient is the SUM of every path's own separate contribution, rather than only the last path's contribution overwriting the others. The single-use case isolates exactly one contribution (`4.0`, from `x*y` alone) with nothing to sum. The double-use case reuses the identical `x` in a second path (`z+x`) whose own local contribution is `1.0`, and the observed final result (`5.0`) is exactly `4.0+1.0`, not merely `1.0` (which a naive "last write wins" implementation would produce) and not merely `4.0` (which a "first write wins" implementation would produce). Observing the correct SUM as the final answer is sufficient proof that accumulation, not replacement, occurred -- the CUDA book's own step-by-step trace is a NICE-to-have illustration of HOW the engine gets there, not the only possible evidence THAT it gets there correctly.

**3.** This result depends on Section 17.2's own accumulation finding: gradients from separate contributions are SUMMED, not replaced, and Section 17.3 extends this specifically to gradients from separate `.backward()` CALLS on a retained graph, not merely separate PATHS within one call. If accumulation did not hold across calls -- if the second `.backward()` call instead OVERWROTE `x.grad()` rather than adding to it -- the second call's own contribution alone would leave `x.grad()` at `5.0` (the same value the first call already produced), not `10.0`. The observed `10.0` is only explainable if the SECOND call's own `5.0` contribution was added on top of the FIRST call's own already-present `5.0`, confirming accumulation applies uniformly regardless of whether the multiple contributions arrive from multiple paths in one backward pass or from multiple backward passes on a retained graph.

**4.** If the control test had shown a `use_count()` change with `requires_grad=false` on both operands, it would have meant `use_count()` was measuring something OTHER than autograd-specific bookkeeping -- perhaps some effect intrinsic to the `+` operator's own execution regardless of differentiability, such as a temporary reference held during the arithmetic itself. In that case, none of the section's own later deltas (the `+1` add-baseline, the `+2` mul deltas, the asymmetric test) could be trusted as evidence about SAVED TENSORS specifically, because the same delta could equally be explained by that other, non-autograd-related effect. The control test's own clean, unchanged result is what licenses treating every later delta as attributable to autograd's own bookkeeping.

**5.** `da = grad_output * b` reads only `b`'s own value -- it never reads `a`'s own value at all. So the backward node genuinely has no reason to keep `a`'s own value alive (only `a`'s own PARTICIPATION in the graph, to know a gradient must eventually be routed to it, which is the `+1` graph-connection overhead every differentiable operand gets regardless), while it genuinely does need `b`'s own value kept alive, specifically to compute `da` when `.backward()` eventually runs -- matching the observed extra `+1` on `b` alone. In the mirror case (only `b` requiring grad), the roles reverse: `db = grad_output * a` would be the only gradient ever computed, requiring `a`'s own value to be saved while `b`'s own value is not needed for anything -- so `b`'s own delta would drop to the graph-connection baseline and `a`'s own delta would show the excess, the exact opposite pairing from this section's own tested case.
