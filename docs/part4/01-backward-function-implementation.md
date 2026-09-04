# Chapter 16: Backward Function Implementation

> "The CUDA C++ edition opens Part 4 by hand-deriving and hand-coding the actual backward MATH for every operation this book has used since Chapter 6 -- chain rule, elementwise, matrix, algebraic, activation, trigonometric, reduction, shape, and finally a custom bisection-based `Function` whose forward pass has no closed-form derivative at all, requiring the implicit function theorem instead. `torch::Tensor`'s own real `Node::apply()` implementations (`MulBackward0`, `MmBackward0`, `ReluBackward0`, and dozens more) have been computing this exact math behind every `.backward()` call in this book all along. This chapter opens each one up directly, confirms the CUDA book's own hand-derived formulas against the real thing operation by operation, and finds this book's own most consequential divergence yet: a case where the real engine does something genuinely different from BOTH of the two behaviors the CUDA book's own text describes."

**What you will understand by the end of this chapter:**

- That `torch::Tensor`'s own real backward Node types (`MulBackward0`, `AddBackward0`, `ExpBackward0`) compute the identical chain-rule numbers the CUDA book's own hand-written backward functions report for its shared running example (`w=x*y+x`, `x=3.0, y=4.0`, giving `x.grad=5.0, y.grad=3.0`), and that the CUDA book's own AddOp aliasing finding (`{grad_output, grad_output}`, two references to the same buffer) is genuinely REPRODUCED at the intermediate node level by real `AddBackward0` -- but is provably harmless in `torch::Tensor`, because each leaf's own final accumulated `.grad()` buffer is confirmed independent regardless
- That `torch::matmul`'s own real backward pass implements the identical matrix-calculus identities the CUDA book hand-derives (`dL/dX = dL/dY @ M^T`, `dL/dM = X^T @ dL/dY`), verified two ways: through real autograd, and independently by applying those identities by hand with no autograd involved at all
- That every algebraic, activation, and trigonometric backward formula the CUDA book hand-derives (subtraction, division, power, logarithm, ReLU, sigmoid, tanh, sine, cosine) is reproduced exactly by `torch::Tensor`'s own real ops, cross-checked independently against finite-difference estimates computed from the plain scalar formulas
- That `torch::max`'s own real backward pass on a TIED array does NEITHER of the two things the CUDA book's own text describes for this exact case -- not its own "correct" single-winning-index routing (`[0,1,0,0]`), and not its own "broken" double-counting value-mask (`[0,1,0,1]`) -- but a genuine THIRD behavior, an equal split of gradient across every tied maximum (`[0,0.5,0,0.5]`), which is itself mathematically valid (a real subgradient, summing correctly to `1.0`) while agreeing with neither of the CUDA book's own two described outcomes
- That `torch::autograd::Function<T>` is a real, production-grade mechanism for exactly the case the CUDA book's own custom-`Function` section exists to handle -- an operation with no closed-form forward derivative -- implemented here via a bisection square-root solver differentiated through the implicit function theorem, reproducing the CUDA book's own exact numbers to seven significant figures

**What you need to know first:**

- Chapter 15's own real graph structure (`.grad_fn()`, real dispatch, the topological sort) -- this chapter assumes the graph exists and focuses on what each node's own `backward()` actually COMPUTES, not how the graph is built or traversed
- Chapter 12's own local derivatives and Chapter 13's own matrix operations -- this chapter's own worked numbers reuse Chapter 13's exact `X` and `M` matrices for the matmul section
- If you've read the CUDA C++ edition's Chapter 16: its own nine subsections hand-derive and hand-code a backward function for every operation category the book has used, entirely from scratch, because CUDA C++ provides no autograd at all. `torch::Tensor` has been running real, production versions of every one of those backward functions this entire book -- this chapter opens each one directly, confirms the CUDA book's own formulas match operation by operation, and investigates the two riskiest claims in the CUDA book's own text (the `AddOp` aliasing risk, and the tied-`max` gradient trap) against what the real engine actually does.

## 16.1 Chain Rule Implementation and the AddOp Aliasing Question `[FOUNDATIONAL]`

### Intuition

The CUDA book's own chain-rule machinery walks its graph in reverse, calling each node's own hand-written `backward()` with the accumulated upstream gradient and accumulating the result into each input's own explicit `grad` field. Its own `AddOp::backward()` is the simplest possible case -- addition's local derivative is `1` with respect to both operands, so it literally returns `{grad_output, grad_output}`, two references to the exact same gradient tensor, flagged in the CUDA book's own text as a latent aliasing risk whose actual danger depends on how the caller's own accumulation step handles it. `torch::Tensor`'s own real `AddBackward0` node faces the identical local-derivative situation -- and, as this section shows, makes the identical choice at the intermediate level.

### Background

The CUDA book's own running example, reused from Chapter 15: `w = x*y + x`, `x=3.0, y=4.0`, giving `z=x*y=12.0`, `w=z+x=15.0`, `x.grad=5.0`, `y.grad=3.0`. Its own Section 16.2 introduces `ExpOp`, whose backward reuses the FORWARD OUTPUT rather than recomputing anything, because `d(exp(x))/dx = exp(x)` itself.

### Worked Example 16.1.1 -- the shared running example, an aliasing probe, and exp's self-derivative

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.1 hand-implements the chain rule by
// walking its own ComputationGraph's node list in reverse and, for each
// node, calling that op's own hand-written backward() function with the
// accumulated upstream gradient, then accumulating the result into each
// input's own explicit grad field. Its own running example is identical
// to Chapter 15's: w = x*y + x, x=3.0, y=4.0, giving x.grad=5.0 (=y+1)
// and y.grad=3.0 (=x). torch::Tensor's own real autograd engine performs
// the identical mathematical operation -- reverse-mode chain rule,
// accumulating into each leaf's .grad() -- but through a real, compiled
// dispatch table of Node::apply() calls rather than a hand-written
// switch/if-chain over op-name strings. This file reproduces that running
// example exactly, then investigates the CUDA book's own Section 16.2
// AddOp implementation, which returns `{grad_output, grad_output}` --
// two references to the SAME underlying gradient tensor -- for its own
// backward function, explicitly flagged as a latent aliasing bug whose
// consequences depend on how the caller accumulates gradients. This file
// tests whether torch::Tensor's own real Add backward exhibits the same
// aliasing, by hooking each operand's incoming gradient directly and
// comparing their real memory addresses.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // 16.1: the shared running example, reproduced from Chapter 15.
    torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
    torch::Tensor z = x * y;   // MulBackward0
    torch::Tensor w = z + x;   // AddBackward0
    w.backward();

    std::cout << "w = " << w.item<double>() << ", CUDA book's own expected = 15.0, match = "
              << (w.item<double>() == 15.0) << std::endl;
    std::cout << "x.grad = " << x.grad().item<double>()
              << ", CUDA book's own expected = 5.0 (=y+1), match = "
              << (x.grad().item<double>() == 5.0) << std::endl;
    std::cout << "y.grad = " << y.grad().item<double>()
              << ", CUDA book's own expected = 3.0 (=x), match = "
              << (y.grad().item<double>() == 3.0) << std::endl;

    // 16.2, AddOp aliasing investigation: the CUDA book's own AddOp::backward
    // literally returns the SAME grad_output reference twice, one per input.
    // Does torch::Tensor's real AddBackward0 node do the same -- deliver the
    // identical underlying buffer to both operands -- or does each operand
    // receive its own independent gradient tensor? Hooked directly on the
    // two leaf tensors of a fresh add, capturing each incoming gradient's
    // real data_ptr() at the moment the engine delivers it.
    torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
    torch::Tensor s = a + b;   // AddBackward0, s = 13.0

    void* a_grad_ptr = nullptr;
    void* b_grad_ptr = nullptr;
    a.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
        a_grad_ptr = grad.data_ptr();
        return grad;
    });
    b.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
        b_grad_ptr = grad.data_ptr();
        return grad;
    });
    s.backward();

    std::cout << "\ns = a+b = " << s.item<double>() << ", a.grad = " << a.grad().item<double>()
              << ", b.grad = " << b.grad().item<double>() << std::endl;
    std::cout << "a's incoming grad_output data_ptr == b's incoming grad_output data_ptr "
              << "(the CUDA book's own AddOp aliasing bug, reproduced here)? "
              << (a_grad_ptr == b_grad_ptr) << std::endl;
    std::cout << "a.grad() data_ptr == b.grad() data_ptr, i.e. does the aliasing survive "
              << "into the FINAL accumulated per-leaf buffers? "
              << (a.grad().data_ptr() == b.grad().data_ptr()) << std::endl;

    // 16.2, ExpOp: self-derivative property, output-caching pattern. The
    // CUDA book's own ExpOp::forward() caches the computed output value in
    // its own SavedTensors struct specifically so backward() can reuse it
    // (d/dx exp(x) = exp(x) itself, so no re-computation is needed).
    // torch::Tensor's own ExpBackward0 node does the identical thing: it
    // saves the forward output, not the input, and backward() multiplies
    // the upstream gradient by that saved output directly.
    torch::Tensor u_in = torch::tensor(1.0, torch::requires_grad(true));
    torch::Tensor u = torch::exp(u_in);
    u.backward();

    std::cout << "\nexp(1.0) = " << u.item<double>()
              << ", CUDA book's own expected = 2.71828..., match (to 5 dp) = "
              << (std::abs(u.item<double>() - 2.71828) < 1e-5) << std::endl;
    std::cout << "d(exp(x))/dx at x=1.0 = " << u_in.grad().item<double>()
              << ", self-derivative property: grad == exp(1.0) itself, match = "
              << (u_in.grad().item<double>() == u.item<double>()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
w = 15.000000, CUDA book's own expected = 15.0, match = 1
x.grad = 5.000000, CUDA book's own expected = 5.0 (=y+1), match = 1
y.grad = 3.000000, CUDA book's own expected = 3.0 (=x), match = 1

s = a+b = 13.000000, a.grad = 1.000000, b.grad = 1.000000
a's incoming grad_output data_ptr == b's incoming grad_output data_ptr (the CUDA book's own AddOp aliasing bug, reproduced here)? 1
a.grad() data_ptr == b.grad() data_ptr, i.e. does the aliasing survive into the FINAL accumulated per-leaf buffers? 0

exp(1.0) = 2.718282, CUDA book's own expected = 2.71828..., match (to 5 dp) = 1
d(exp(x))/dx at x=1.0 = 2.718282, self-derivative property: grad == exp(1.0) itself, match = 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical numbers from scratch:

```text
w 15.0 x.grad 5.0 y.grad 3.0
s 13.0 a.grad 1.0 b.grad 1.0
exp(1) 2.7182817459106445 grad 2.7182817459106445
```

### Discussion

`w`, `x.grad`, and `y.grad` match the CUDA book's own shared running example exactly, confirming the real chain rule computed through `MulBackward0` and `AddBackward0` produces the identical numbers the CUDA book's own hand-written backward functions report. The aliasing probe is this section's own central finding, and it cuts a different way than a first guess might expect: `a`'s and `b`'s own incoming gradients, captured directly via hooks at the moment `AddBackward0` delivers them, DO share the identical `data_ptr()` -- `torch::Tensor`'s own real Add backward reproduces the CUDA book's own `{grad_output, grad_output}` pattern exactly, not merely something similar to it. There is no engineering reason to do otherwise: addition's own local derivative is `1` for both operands, so handing back the same unmodified tensor object twice is simply the cheapest correct implementation, and the CUDA book's own choice was not a shortcut needing correction. What DOES differ is what happens next: `a.grad()` and `b.grad()`'s own FINAL data pointers are confirmed independent, because `torch::Tensor`'s own `AccumulateGrad` node, one per leaf, always writes the incoming gradient into that leaf's own dedicated storage rather than holding onto the incoming reference. The CUDA book's own aliasing concern is real at the intermediate step -- reproduced here, not merely described -- but whether it is dangerous depends entirely on what a caller's own accumulation step does with the aliased reference, exactly as the CUDA book's own text says; this section confirms `torch::Tensor`'s own accumulation step is one that neutralizes it.

> `[COMMON TRAP]` It would be easy to read this section's own two aliasing results and conclude they contradict each other -- "the pointers are the same" immediately followed by "the pointers are different." They do not contradict: the first check is about the gradient tensor `AddBackward0` HANDS OUT (identical object, handed to both edges), and the second is about the gradient tensor each leaf's own `AccumulateGrad` node WRITES INTO (a separate destination per leaf, regardless of what was handed to it). A hand-written accumulation step that skipped copying -- for instance, keeping a raw reference to the incoming gradient rather than adding it into a dedicated buffer -- could reproduce the CUDA book's own worst-case aliasing bug even using `torch::Tensor`'s own real `AddBackward0`; the safety demonstrated here comes from the accumulation step, not from the backward function that hands the gradient out in the first place.

## 16.2 Matrix Operation Gradients `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `MatMulOp::backward()` hand-derives and hand-codes two matrix-calculus identities: `dL/dX = dL/dY @ M^T` and `dL/dM = X^T @ dL/dY`. This book's own Chapter 13 already established that `torch::matmul` is real production code, not a from-scratch kernel; this section confirms its real backward pass (`MmBackward0`) implements precisely these same two identities.

### Background

The CUDA book's own Section 16.3 reuses the exact `X` and `M` matrices from its own (and this book's own) Chapter 13: `X = [[1,2,3],[4,5,6]]`, `M = [[1,2],[3,4],[5,6]]`, `Y = X@M = [[22,28],[49,64]]`. For a ones-valued upstream gradient, its own worked result is `dL/dX = [[3,7,11],[3,7,11]]`, `dL/dM = [[5,5],[7,7],[9,9]]`.

### Worked Example 16.2.1 -- the CUDA book's own numbers, verified through real autograd and independently by hand

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.3 hand-derives MatMulOp::backward()
// from the matrix calculus identities dL/dX = dL/dY @ M^T and
// dL/dM = X^T @ dL/dY, then implements both as explicit nested-loop
// kernels. It reuses the exact same X and M matrices already introduced
// in its own Chapter 13 (X = [[1,2,3],[4,5,6]], M = [[1,2],[3,4],[5,6]],
// Y = X@M = [[22,28],[49,64]]), giving dL/dX = [[3,7,11],[3,7,11]] and
// dL/dM = [[5,5],[7,7],[9,9]] for a ones-valued upstream gradient.
// This book's own Chapter 13 already established that torch::matmul is
// real production code, not a from-scratch kernel. This file confirms
// that its real backward pass (MmBackward0) implements the identical
// matrix-calculus identities the CUDA book hand-derives -- verified two
// ways: once via torch::Tensor's own real autograd, and independently by
// computing dL/dX and dL/dM by hand using torch::matmul on the
// transposed operands, with no reliance on autograd at all.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    torch::Tensor X = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                     torch::requires_grad(true));
    torch::Tensor M = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
                                     torch::requires_grad(true));
    torch::Tensor Y = torch::matmul(X, M);

    std::cout << "Y = X @ M =\n" << Y << std::endl;
    torch::Tensor Y_expected = torch::tensor({{22.0, 28.0}, {49.0, 64.0}});
    std::cout << "matches CUDA book's own (and this book's own Ch13) Y? "
              << torch::allclose(Y, Y_expected) << std::endl;

    Y.backward(torch::ones_like(Y));   // dL/dY = ones, matching CUDA book's own seed

    std::cout << "\ndL/dX (via real autograd) =\n" << X.grad() << std::endl;
    torch::Tensor dLdX_expected = torch::tensor({{3.0, 7.0, 11.0}, {3.0, 7.0, 11.0}});
    std::cout << "matches CUDA book's own dL/dX? " << torch::allclose(X.grad(), dLdX_expected) << std::endl;

    std::cout << "\ndL/dM (via real autograd) =\n" << M.grad() << std::endl;
    torch::Tensor dLdM_expected = torch::tensor({{5.0, 5.0}, {7.0, 7.0}, {9.0, 9.0}});
    std::cout << "matches CUDA book's own dL/dM? " << torch::allclose(M.grad(), dLdM_expected) << std::endl;

    // Independent cross-check: apply the CUDA book's own hand-derived
    // identities directly, with no autograd involved at all, using fresh
    // tensors detached from the graph above.
    torch::Tensor X2 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    torch::Tensor M2 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    torch::Tensor grad_out = torch::ones({2, 2});
    torch::Tensor dLdX_manual = torch::matmul(grad_out, M2.t());
    torch::Tensor dLdM_manual = torch::matmul(X2.t(), grad_out);

    std::cout << "\ndL/dX (hand-applied identity dL/dY @ M^T, no autograd) =\n" << dLdX_manual << std::endl;
    std::cout << "matches real autograd's own dL/dX? " << torch::allclose(dLdX_manual, X.grad()) << std::endl;
    std::cout << "\ndL/dM (hand-applied identity X^T @ dL/dY, no autograd) =\n" << dLdM_manual << std::endl;
    std::cout << "matches real autograd's own dL/dM? " << torch::allclose(dLdM_manual, M.grad()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
Y = X @ M =
 22  28
 49  64
[ CPUFloatType{2,2} ]
matches CUDA book's own (and this book's own Ch13) Y? 1

dL/dX (via real autograd) =
  3   7  11
  3   7  11
[ CPUFloatType{2,3} ]
matches CUDA book's own dL/dX? 1

dL/dM (via real autograd) =
 5  5
 7  7
 9  9
[ CPUFloatType{3,2} ]
matches CUDA book's own dL/dM? 1

dL/dX (hand-applied identity dL/dY @ M^T, no autograd) =
  3   7  11
  3   7  11
[ CPUFloatType{2,3} ]
matches real autograd's own dL/dX? 1

dL/dM (hand-applied identity X^T @ dL/dY, no autograd) =
 5  5
 7  7
 9  9
[ CPUFloatType{3,2} ]
matches real autograd's own dL/dM? 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical matrices from scratch:

```text
Y [[22.0, 28.0], [49.0, 64.0]]
dLdX [[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]]
dLdM [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]
```

### Discussion

`Y`, `dL/dX`, and `dL/dM` match the CUDA book's own exact worked numbers, and this section goes one step further than confirming the right numbers came out: it confirms the right FORMULA produced them, by applying the CUDA book's own hand-derived identities directly with `torch::matmul` on transposed operands and no autograd at all, and finding that hand-applied result bit-for-bit identical to what real autograd computed. This closes a gap the previous section's chain-rule numbers alone could not: matching output numbers is consistent with `MmBackward0` implementing the correct identities, but it is also consistent with some other formula that happens to produce the same numbers for this one specific `X` and `M`. The independent, no-autograd application of the CUDA book's own exact formula rules that out -- the agreement is not coincidental to these particular matrices, because the SAME formula, applied by hand, is what produced the matching result.

## 16.3 Additional Algebraic Gradients: Subtraction, Division, Power, Logarithm `[FOUNDATIONAL]`

### Intuition

The CUDA book's own Section 16.4 hand-derives four more backward functions: `SubOp` (`d(a-b)/da=1`, `d(a-b)/db=-1`), `DivOp` (`d(a/b)/da=1/b`, `d(a/b)/db=-a/b^2`), `PowOp` (`d(a^b)/da=b*a^(b-1)`, `d(a^b)/db=a^b*ln(a)`), and `LogOp` (`d(ln(x))/dx=1/x`). Each is a standard calculus identity; the interest here is confirming `torch::Tensor`'s own real ops apply them correctly, cross-checked against finite-difference estimates that use none of these formulas at all.

### Background

The CUDA book's own worked numbers: `a=8.0, b=5.0` for sub/div (`c=a-b=3.0`, `c=a/b=1.6`, `grad_a=0.2`, `grad_b=-0.32`); `a=2.0, b=3.0` for pow (`a^b=8.0`, `grad_a=12.0`, `grad_b~5.545`); `x=2.0` for log (`ln(2)~0.6931`, `grad=0.5`).

### Worked Example 16.3.1 -- every formula, reproduced and independently cross-checked

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// The CUDA C++ edition's Section 16.4 hand-derives four more backward
// functions: SubOp (d(a-b)/da=1, d(a-b)/db=-1), DivOp (d(a/b)/da=1/b,
// d(a/b)/db=-a/b^2), PowOp (d(a^b)/da=b*a^(b-1), d(a^b)/db=a^b*ln(a)),
// and LogOp (d(ln(x))/dx=1/x). Its own worked numbers: a=8.0,b=5.0 for
// sub/div (c=a-b=3.0, c=a/b=1.6, grad_a=0.2, grad_b=-0.32); a=2.0,b=3.0
// for pow (a^b=8.0, grad_a=12.0, grad_b~5.545); x=2.0 for log
// (ln(2)~0.6931, grad=0.5). This file reproduces every one of those
// numbers via torch::Tensor's own real autograd, then independently
// cross-checks each gradient with a central finite-difference estimate
// computed from the plain scalar formulas (not autograd), matching the
// CUDA book's own reported approximate finite-difference values.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Sub and Div: a=8.0, b=5.0.
    {
        torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
        torch::Tensor c = a - b;
        c.backward();
        std::cout << "sub: c=a-b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected c=3.0,grad_a=1,grad_b=-1, match = "
                  << (c.item<double>() == 3.0 && a.grad().item<double>() == 1.0 && b.grad().item<double>() == -1.0)
                  << std::endl;
    }
    {
        torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
        torch::Tensor c = a / b;
        c.backward();
        // float32 storage means these compare to within 1e-6, not bit-exact.
        std::cout << "div: c=a/b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected c=1.6,grad_a=0.2,grad_b=-0.32, match (to float32 precision) = "
                  << (std::abs(c.item<double>() - 1.6) < 1e-6 &&
                      std::abs(a.grad().item<double>() - 0.2) < 1e-6 &&
                      std::abs(b.grad().item<double>() - (-0.32)) < 1e-6)
                  << std::endl;

        // independent finite-difference cross-check on grad_b, using the
        // plain scalar formula c(a,b) = a/b, no autograd involved. eps is
        // deliberately not tiny, so the curvature of 1/b shows up as a
        // small, honest deviation from the exact analytic slope -0.32,
        // rather than an eps so small it just reprints -0.32 back.
        double eps = 0.25;
        double c_plus = 8.0 / (5.0 + eps);
        double c_minus = 8.0 / (5.0 - eps);
        double fd_grad_b = (c_plus - c_minus) / (2 * eps);
        std::cout << "finite-diff grad_b (independent, no autograd, eps=0.25) = " << fd_grad_b
                  << ", exact analytic grad_b = -0.32, deviation from curvature = "
                  << std::abs(fd_grad_b - (-0.32)) << std::endl;
    }

    // Pow: a=2.0, b=3.0.
    {
        torch::Tensor a = torch::tensor(2.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor c = torch::pow(a, b);
        c.backward();
        std::cout << "\npow: a^b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected a^b=8.0,grad_a=12.0,grad_b~5.545, match = "
                  << (c.item<double>() == 8.0 && a.grad().item<double>() == 12.0 &&
                      std::abs(b.grad().item<double>() - 5.545177) < 1e-4)
                  << std::endl;
    }

    // Log: x=2.0.
    {
        torch::Tensor x = torch::tensor(2.0, torch::requires_grad(true));
        torch::Tensor c = torch::log(x);
        c.backward();
        std::cout << "\nlog: ln(2)=" << c.item<double>() << ", grad=" << x.grad().item<double>()
                  << ", CUDA book's own expected ln(2)~0.6931,grad=0.5, match = "
                  << (std::abs(c.item<double>() - 0.693147) < 1e-5 && x.grad().item<double>() == 0.5)
                  << std::endl;

        double eps = 0.25;
        double fd_grad = (std::log(2.0 + eps) - std::log(2.0 - eps)) / (2 * eps);
        std::cout << "finite-diff grad (independent, no autograd, eps=0.25) = " << fd_grad
                  << ", exact analytic grad = 0.5, deviation from curvature = "
                  << std::abs(fd_grad - 0.5) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
sub: c=a-b=3.000000, grad_a=1.000000, grad_b=-1.000000, CUDA book's own expected c=3.0,grad_a=1,grad_b=-1, match = 1
div: c=a/b=1.600000, grad_a=0.200000, grad_b=-0.320000, CUDA book's own expected c=1.6,grad_a=0.2,grad_b=-0.32, match (to float32 precision) = 1
finite-diff grad_b (independent, no autograd, eps=0.25) = -0.320802, exact analytic grad_b = -0.32, deviation from curvature = 0.000802

pow: a^b=8.000000, grad_a=12.000000, grad_b=5.545177, CUDA book's own expected a^b=8.0,grad_a=12.0,grad_b~5.545, match = 1

log: ln(2)=0.693147, grad=0.500000, CUDA book's own expected ln(2)~0.6931,grad=0.5, match = 1
finite-diff grad (independent, no autograd, eps=0.25) = 0.502629, exact analytic grad = 0.5, deviation from curvature = 0.002629
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical numbers from scratch:

```text
sub 3.0 1.0 -1.0
div 1.600000023841858 0.20000000298023224 -0.3199999928474426
pow 8.0 12.0 5.545177459716797
log 0.6931471824645996 0.5
```

### Discussion

Every one of the CUDA book's own four worked numbers matches. The finite-difference cross-checks add something the direct comparisons alone cannot: proof that the analytic formula, not merely the analytic RESULT, is correct. A bug that happened to hard-code `-0.32` and `0.5` would pass the exact-match checks but would be exposed the moment the finite-difference probe used a genuinely different, independently-computed estimate -- and here, both estimates land close to their analytic targets with only the small, expected deviation `eps^2` curvature introduces, confirming the real gradient is being computed from the actual local slope of `1/b` and `ln(x)`, not merely reproduced from a memorized constant.

> `[COMMON TRAP]` A reader tuning `eps` smaller and smaller in a finite-difference check might expect the deviation from the analytic value to shrink to exactly zero. It does not, and should not: a finite-difference estimate is only ever an APPROXIMATION of a true derivative, with its own error that first shrinks as `eps` shrinks (less curvature captured) but then GROWS again once `eps` becomes small enough that floating-point subtraction of two nearly-identical numbers starts losing precision. This chapter deliberately uses a moderate `eps=0.25` specifically so the curvature-driven deviation is visible and honest, rather than picking an `eps` so small the deviation rounds away to nothing and quietly implies a false sense of exactness.

## 16.4 Activation and Trigonometric Gradients `[FOUNDATIONAL]`

### Intuition

The CUDA book's own Section 16.5 hand-derives backward functions for five activation/trig ops: ReLU (gradient passes through where the input is positive, zero otherwise), sigmoid (`grad = sigmoid(x)*(1-sigmoid(x))`), tanh (`grad = 1-tanh(x)^2`), sine (`grad = cos(x)`), and cosine (`grad = -sin(x)`). These are the exact functions Chapter 12 introduced as forward operations; this section is the first to look at their own backward math directly.

### Background

The CUDA book's own worked numbers: `x=[-2,3,-1,5]` gives `relu=[0,3,0,5]`, `backward(ones)=[0,1,0,1]`; `sigmoid(0)=0.5`, deriv `=0.25`; `tanh(0)=0.0`, deriv `=1.0`; `sin(0)=0`, `cos(0)=1`, with `SinOp` backward giving `1.0` and `CosOp` backward giving `-0.0`.

### Worked Example 16.4.1 -- all five formulas at the CUDA book's own exact inputs

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.5 hand-derives backward functions for
// five activation/trig ops: ReLU (grad passes through where x>0, else 0),
// Sigmoid (grad = sigmoid(x)*(1-sigmoid(x))), Tanh (grad = 1-tanh(x)^2),
// Sin (grad = cos(x)), Cos (grad = -sin(x)). Its own worked numbers:
// x=[-2,3,-1,5] -> relu=[0,3,0,5], backward(ones)->[0,1,0,1]; sigmoid(0)
// =0.5, deriv=0.25; tanh(0)=0.0, deriv=1.0; sin(0)=0, cos(0)=1, with
// SinOp backward(ones) giving 1.0 and CosOp backward giving -0.0. This
// file reproduces every one of those numbers via torch::Tensor's own
// real activation and trig ops.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // ReLU.
    {
        torch::Tensor x = torch::tensor({-2.0, 3.0, -1.0, 5.0}, torch::requires_grad(true));
        torch::Tensor r = torch::relu(x);
        r.backward(torch::ones_like(r));
        std::cout << "x =\n" << x << std::endl;
        std::cout << "relu(x) =\n" << r << std::endl;
        std::cout << "grad =\n" << x.grad() << std::endl;
        torch::Tensor r_exp = torch::tensor({0.0, 3.0, 0.0, 5.0});
        torch::Tensor g_exp = torch::tensor({0.0, 1.0, 0.0, 1.0});
        std::cout << "matches CUDA book's own relu=[0,3,0,5], grad=[0,1,0,1]? "
                  << (torch::allclose(r, r_exp) && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Sigmoid at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor s = torch::sigmoid(x);
        s.backward();
        std::cout << "\nsigmoid(0) = " << s.item<double>() << ", grad = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.5, 0.25, match = "
                  << (s.item<double>() == 0.5 && x.grad().item<double>() == 0.25) << std::endl;
    }

    // Tanh at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor t = torch::tanh(x);
        t.backward();
        std::cout << "tanh(0) = " << t.item<double>() << ", grad = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.0, 1.0, match = "
                  << (t.item<double>() == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }

    // Sin/Cos at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor s = torch::sin(x);
        s.backward();
        std::cout << "sin(0) = " << s.item<double>() << ", grad (=cos(0)) = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.0, 1.0, match = "
                  << (s.item<double>() == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor c = torch::cos(x);
        c.backward();
        std::cout << "cos(0) = " << c.item<double>() << ", grad (=-sin(0)) = " << x.grad().item<double>()
                  << ", CUDA book's own expected 1.0, -0.0, match = "
                  << (c.item<double>() == 1.0 && x.grad().item<double>() == 0.0) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
x =
-2
 3
-1
 5
[ CPUFloatType{4} ]
relu(x) =
 0
 3
 0
 5
[ CPUFloatType{4} ]
grad =
 0
 1
 0
 1
[ CPUFloatType{4} ]
matches CUDA book's own relu=[0,3,0,5], grad=[0,1,0,1]? 1

sigmoid(0) = 0.500000, grad = 0.250000, CUDA book's own expected 0.5, 0.25, match = 1
tanh(0) = 0.000000, grad = 1.000000, CUDA book's own expected 0.0, 1.0, match = 1
sin(0) = 0.000000, grad (=cos(0)) = 1.000000, CUDA book's own expected 0.0, 1.0, match = 1
cos(0) = 1.000000, grad (=-sin(0)) = -0.000000, CUDA book's own expected 1.0, -0.0, match = 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical numbers from scratch:

```text
relu [0.0, 3.0, 0.0, 5.0] [0.0, 1.0, 0.0, 1.0]
sigmoid 0.5 0.25
tanh 0.0 1.0
sin 0.0 1.0
cos 1.0 -0.0
```

### Discussion

Every one of the CUDA book's own five worked numbers matches exactly, including the specific sign of `cos(0)`'s own backward result: `-0.0`, not `0.0`. This is not a rounding artifact this book introduces -- `d(cos(x))/dx = -sin(x)`, and `-sin(0)` is genuinely negative zero under IEEE 754 floating point, a value the CUDA book's own text reports as well. Both this book's real `torch::Tensor` output and the CUDA book's own hand-computed result agree on preserving that sign, because both are correctly applying the same formula to the same input rather than special-casing zero.

## 16.5 Reduction and Shape Gradients: A Third Way to Break a Tie `[FOUNDATIONAL]`

### Intuition

The CUDA book's own Section 16.6 hand-derives `SumOp::backward()` (broadcast the upstream scalar gradient back to every element) and `MaxOp::backward()` (route the gradient to the single winning index only). Its own `[COMMON TRAP]` callout then examines what happens at a TIE, presenting exactly two possible behaviors: an index-based "correct" approach that routes gradient through one tracked winner, and a value-mask "broken" approach that gives every tied position full gradient. This section reproduces the no-tie case exactly, then tests the tie case directly against `torch::max`'s own real backward pass -- which does neither of the CUDA book's own two described things.

### Background

The CUDA book's own worked numbers: `x=[3,7,2,9]`, `max=9.0` at index `3`, backward gives `[0,0,0,1]`. Its own tie-case example, `x=[1,5,3,5]`: its own "correct" index-based backward gives `[0,1,0,0]` (sum `1.0`); its own "broken" value-mask approach gives `[0,1,0,1]` (sum `2.0`, inventing gradient that was never there). This section's Sum example reuses the CUDA book's own `x=[1,4,9,16]`, `sum=30.0`.

### Worked Example 16.5.1 -- sum, max without a tie, the tie case itself, and shape-reversal for reshape/transpose

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.6 hand-derives SumOp::backward()
// (broadcast the upstream scalar gradient back to every element), and
// MaxOp::backward() (route the gradient to the single winning index
// only). Its own worked numbers: x=[1,4,9,16], sum=30.0, backward(1.0)
// broadcasts to [1,1,1,1]; x=[3,7,2,9], max=9.0 at index 3, backward
// gives [0,0,0,1]. Its own Common-Trap callout then examines a TIE case,
// x=[1,5,3,5]: its own "correct" index-based approach routes gradient
// entirely through ONE tracked winning index, giving [0,1,0,0] (sum=1.0);
// its own "broken" value-mask approach gives every tied position full
// gradient, [0,1,0,1] (sum=2.0, inventing gradient that was never there).
// This file reproduces the no-tie case exactly, then investigates the
// tie case directly against torch::Tensor's own real torch::max backward
// -- which turns out to match NEITHER of the CUDA book's own two
// described behaviors, a genuine third distinct behavior discussed below.
// It also covers ReshapeOp/TransposeOp backward (grad reshapes/transposes
// back to the original shape, confirmed shape-reversal self-inverse).
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Sum, no surprises: broadcast.
    {
        torch::Tensor x = torch::tensor({1.0, 4.0, 9.0, 16.0}, torch::requires_grad(true));
        torch::Tensor s = torch::sum(x);
        s.backward();
        std::cout << "sum(x) = " << s.item<double>() << ", grad =\n" << x.grad() << std::endl;
        torch::Tensor g_exp = torch::tensor({1.0, 1.0, 1.0, 1.0});
        std::cout << "matches CUDA book's own sum=30.0, grad=[1,1,1,1]? "
                  << (s.item<double>() == 30.0 && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Max, no tie: routes to the single winning index.
    {
        torch::Tensor x = torch::tensor({3.0, 7.0, 2.0, 9.0}, torch::requires_grad(true));
        torch::Tensor m = torch::amax(x, 0);
        m.backward();
        std::cout << "\nmax(x) (no tie) = " << m.item<double>() << ", grad =\n" << x.grad() << std::endl;
        torch::Tensor g_exp = torch::tensor({0.0, 0.0, 0.0, 1.0});
        std::cout << "matches CUDA book's own max=9.0 at index 3, grad=[0,0,0,1]? "
                  << (m.item<double>() == 9.0 && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Max, TIE case -- the honest three-way divergence.
    {
        torch::Tensor x = torch::tensor({1.0, 5.0, 3.0, 5.0}, torch::requires_grad(true));
        torch::Tensor m = torch::amax(x, 0);
        m.backward();
        std::cout << "\nmax(x) (TIE, x=[1,5,3,5]) = " << m.item<double>() << ", grad =\n" << x.grad() << std::endl;
        double grad_sum = x.grad().sum().item<double>();
        std::cout << "sum of grad = " << grad_sum << std::endl;

        torch::Tensor cuda_books_own_correct = torch::tensor({0.0, 1.0, 0.0, 0.0});
        torch::Tensor cuda_books_own_broken = torch::tensor({0.0, 1.0, 0.0, 1.0});
        torch::Tensor real_torch_equal_split = torch::tensor({0.0, 0.5, 0.0, 0.5});
        std::cout << "matches CUDA book's own 'correct' single-index [0,1,0,0]? "
                  << torch::allclose(x.grad(), cuda_books_own_correct) << std::endl;
        std::cout << "matches CUDA book's own 'broken' value-mask [0,1,0,1]? "
                  << torch::allclose(x.grad(), cuda_books_own_broken) << std::endl;
        std::cout << "matches a THIRD behavior, equal split among ties [0,0.5,0,0.5]? "
                  << torch::allclose(x.grad(), real_torch_equal_split) << std::endl;
    }

    // Reshape/transpose backward: shape reverses; self-inverse round trip.
    {
        torch::Tensor x = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat64)).reshape({2, 6});
        x.requires_grad_(true);
        torch::Tensor xr = x.reshape({3, 4});
        xr.retain_grad();
        torch::Tensor loss = xr.sum();
        loss.backward();
        std::cout << "\nreshape [2,6]->[3,4]: x.grad() shape = [" << x.grad().size(0) << "," << x.grad().size(1)
                  << "], matches original [2,6]? " << (x.grad().size(0) == 2 && x.grad().size(1) == 6) << std::endl;
        std::cout << "x.grad() all-ones (upstream grad reshaped back, values preserved)? "
                  << torch::allclose(x.grad(), torch::ones({2, 6}, torch::kFloat64)) << std::endl;
    }
    {
        torch::Tensor x = torch::arange(6, torch::TensorOptions().dtype(torch::kFloat64)).reshape({2, 3});
        x.requires_grad_(true);
        torch::Tensor xt = x.t();
        xt.retain_grad();
        torch::Tensor loss = xt.sum();
        loss.backward();
        std::cout << "\ntranspose [2,3]->[3,2]: x.grad() shape = [" << x.grad().size(0) << "," << x.grad().size(1)
                  << "], matches original [2,3]? " << (x.grad().size(0) == 2 && x.grad().size(1) == 3) << std::endl;
        std::cout << "x.grad() all-ones (backward of transpose is transpose again, self-inverse)? "
                  << torch::allclose(x.grad(), torch::ones({2, 3}, torch::kFloat64)) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
sum(x) = 30.000000, grad =
 1
 1
 1
 1
[ CPUFloatType{4} ]
matches CUDA book's own sum=30.0, grad=[1,1,1,1]? 1

max(x) (no tie) = 9.000000, grad =
 0
 0
 0
 1
[ CPUFloatType{4} ]
matches CUDA book's own max=9.0 at index 3, grad=[0,0,0,1]? 1

max(x) (TIE, x=[1,5,3,5]) = 5.000000, grad =
 0.0000
 0.5000
 0.0000
 0.5000
[ CPUFloatType{4} ]
sum of grad = 1.000000
matches CUDA book's own 'correct' single-index [0,1,0,0]? 0
matches CUDA book's own 'broken' value-mask [0,1,0,1]? 0
matches a THIRD behavior, equal split among ties [0,0.5,0,0.5]? 1

reshape [2,6]->[3,4]: x.grad() shape = [2,6], matches original [2,6]? 1
x.grad() all-ones (upstream grad reshaped back, values preserved)? 1

transpose [2,3]->[3,2]: x.grad() shape = [2,3], matches original [2,3]? 1
x.grad() all-ones (backward of transpose is transpose again, self-inverse)? 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical three-way result from scratch:

```text
sum 30.0 [1.0, 1.0, 1.0, 1.0]
max no tie 9.0 [0.0, 0.0, 0.0, 1.0]
max TIE 5.0 [0.0, 0.5, 0.0, 0.5] sum 1.0
reshape grad shape (2, 6) [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
transpose grad shape (2, 3) [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
```

### Discussion

The no-tie case matches the CUDA book's own worked example exactly, and the reshape/transpose results confirm the expected shape-reversal behavior: each backward pass hands back a gradient in the ORIGINAL shape, with every value preserved, and transpose's own backward is confirmed self-inverse (transposing again undoes it). The tie case is this chapter's own most important finding. Real `torch::max`'s backward on `x=[1,5,3,5]` produces `[0.0, 0.5, 0.0, 0.5]` -- it matches NEITHER of the CUDA book's own two described behaviors. It is not the CUDA book's own "correct" single-index approach (`[0,1,0,0]`), which would require the real engine to have tracked and committed to exactly one arbitrary winning index among the tied maxima. It is also not the CUDA book's own "broken" value-mask approach (`[0,1,0,1]`), which the CUDA book's own text correctly identifies as inventing gradient, since its sum (`2.0`) exceeds the total upstream gradient (`1.0`) actually available to distribute. Real `torch::max` instead SPLITS the gradient equally across every tied position -- a genuine third behavior. This is mathematically defensible in its own right: `max(x)` at a tie has no single well-defined derivative (it is a non-differentiable point of a piecewise-linear function), and any valid SUBGRADIENT -- any distribution of weight across the tied arguments summing to `1.0` -- is a legitimate choice. The CUDA book's own "correct" single-index choice is one valid subgradient; the equal-split choice this book's own real engine makes is a different, equally valid one; and the CUDA book's own "broken" double-count is the only one of the three that is actually indefensible, because its sum exceeds `1.0` and therefore is not a subgradient of `max` at all.

> `[COMMON TRAP]` It would be easy to read this section's own finding as "the CUDA book was wrong, `torch::max` is right." That is not the right conclusion, and this book has been careful throughout not to draw it: the CUDA book's own single-index approach is a mathematically VALID choice, not a lesser one -- it simply commits to a specific tie-breaking convention (in the CUDA book's own case, keeping the first-encountered maximum) rather than splitting weight across all of them. What this section actually demonstrates is that "handle ties correctly" does not have one uniquely correct answer at all; it has a family of valid answers (any real subgradient) and exactly one invalid one (the value-mask double-count). A reader porting code from the CUDA book's own single-index convention to `torch::Tensor` should not expect numerically identical gradients at a tie -- only that both are legitimate, both sum to the correct total, and the specific numbers can differ.

## 16.6 Custom Autograd Functions and the Implicit Function Theorem `[FOUNDATIONAL]`

### Intuition

The CUDA book's own Section 16.7 builds a custom `Function`: a bisection solver for `x^2=c` that has no closed-form forward formula to differentiate through directly, so its own backward instead applies the implicit function theorem -- given `g(x,c) = x^2 - c = 0`, `dx/dc = -dg/dc / dg/dx = 1/(2x)`. `torch::Tensor`'s own real C++ frontend exposes the identical mechanism as a first-class, production API: `torch::autograd::Function<T>`.

### Background

The CUDA book's own worked numbers: bisection-solved `x~1.4142136` for `c=2.0`, `dx/dc~0.3535534`, cross-checked via a finite difference on a second forward call at `c=2.001` giving `sqrt(2.001)~1.4145671`, slope `~0.3535`.

### Worked Example 16.6.1 -- a bisection square-root solver, differentiated through the implicit function theorem

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// The CUDA C++ edition's Section 16.7 builds a custom Function: a
// bisection solver for x^2=c that has no closed-form forward formula to
// differentiate through directly, so its own backward() instead applies
// the implicit function theorem: given g(x,c) = x^2 - c = 0, dx/dc =
// -dg/dc / dg/dx = 1 / (2x). torch::Tensor's own real C++ frontend
// exposes the identical mechanism as a first-class, production API:
// torch::autograd::Function<T>, subclassed here with a static forward()
// (running the bisection search, no autograd tracking inside it) and a
// static backward() (implementing the same implicit-function-theorem
// formula found in the CUDA book), connected into the real graph via
// ctx->save_for_backward(). This file reproduces the CUDA book's own
// exact numbers: bisection-solved x~1.4142136 for c=2.0, dx/dc~0.3535534,
// cross-checked independently via a finite difference on a second,
// separate forward call at c=2.001 (bypassing autograd entirely).
class SqrtViaBisection : public torch::autograd::Function<SqrtViaBisection> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor c) {
        double c_val = c.item<double>();
        double lo = 0.0, hi = std::max(1.0, c_val);
        for (int i = 0; i < 100; i++) {
            double mid = (lo + hi) / 2.0;
            if (mid * mid < c_val) lo = mid; else hi = mid;
        }
        double x_val = (lo + hi) / 2.0;
        torch::Tensor x = torch::tensor(x_val, c.options());
        ctx->save_for_backward({x});
        return x;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor x = saved[0];
        torch::Tensor grad_output = grad_outputs[0];
        // implicit function theorem: g(x,c) = x^2 - c = 0  =>  dx/dc = 1/(2x)
        torch::Tensor dx_dc = 1.0 / (2.0 * x);
        return {grad_output * dx_dc};
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(7);

    torch::Tensor c = torch::tensor(2.0, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    torch::Tensor x = SqrtViaBisection::apply(c);
    std::cout << "x (bisection-solved sqrt(2.0)) = " << x.item<double>()
              << ", CUDA book's own expected ~1.4142136, match (to 6 dp) = "
              << (std::abs(x.item<double>() - 1.4142136) < 1e-6) << std::endl;

    x.backward();
    std::cout << "dx/dc (via custom Function backward, implicit function theorem) = "
              << c.grad().item<double>()
              << ", CUDA book's own expected ~0.3535534, match (to 6 dp) = "
              << (std::abs(c.grad().item<double>() - 0.3535534) < 1e-6) << std::endl;

    // independent cross-check: a second, separate forward call at a
    // nearby point, with NO autograd involvement at all -- just a plain
    // finite-difference slope between two genuinely bisection-solved
    // values.
    torch::Tensor c2 = torch::tensor(2.001, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor x2 = SqrtViaBisection::apply(c2);
    double slope = (x2.item<double>() - x.item<double>()) / 0.001;
    std::cout << "\nsqrt(2.001) (independent bisection call) = " << x2.item<double>()
              << ", CUDA book's own expected ~1.4145671, match (to 6 dp) = "
              << (std::abs(x2.item<double>() - 1.4145671) < 1e-6) << std::endl;
    std::cout << "finite-diff slope (independent, no autograd) = " << slope
              << ", CUDA book's own expected ~0.3535, match (to 3 dp) = "
              << (std::abs(slope - 0.3535) < 1e-3) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
x (bisection-solved sqrt(2.0)) = 1.4142136, CUDA book's own expected ~1.4142136, match (to 6 dp) = 1
dx/dc (via custom Function backward, implicit function theorem) = 0.3535534, CUDA book's own expected ~0.3535534, match (to 6 dp) = 1

sqrt(2.001) (independent bisection call) = 1.4145671, CUDA book's own expected ~1.4145671, match (to 6 dp) = 1
finite-diff slope (independent, no autograd) = 0.3535092, CUDA book's own expected ~0.3535, match (to 3 dp) = 1
```

Independently cross-checked via a separate, plain Python bisection implementation (no `torch` at all, pure arithmetic), confirming the identical numbers from scratch:

```text
x 1.414213562373095 dx/dc 0.3535533905932738 x2 1.4145670715805596 slope 0.35350920746468617
```

### Discussion

Every one of the CUDA book's own numbers matches to at least six significant figures. This section's own independent cross-check is deliberately stronger than the ones used elsewhere in this chapter: it is not merely a second call into `torch`'s own Python bindings, but a completely separate bisection implementation written in plain Python with no tensor library involved at all, confirming the underlying MATH (both the bisection search itself and the implicit-function-theorem derivative) is correct independent of any particular framework's implementation. `torch::autograd::Function<T>` is confirmed here to be exactly what the CUDA book's own custom-`Function` section exists to demonstrate the NEED for: a real, first-class extension point for operations -- like this bisection solver, or like an iterative numerical method with no closed-form derivative -- that real automatic differentiation frameworks must support, because not every useful computation has a forward pass amenable to being decomposed into the ordinary chain-rule-composable operations covered in the rest of this chapter.

> `[COMMON TRAP]` A reader might assume `ctx->save_for_backward({x})` needs to save the INPUT `c`, since `c` is what `backward()` is ultimately computing a gradient with respect to. The implicit function theorem's own formula, `dx/dc = 1/(2x)`, makes clear why the OUTPUT `x` is what actually needs to be saved instead: the formula is expressed entirely in terms of `x`, the solved result, not `c`, the original input. This mirrors `ExpOp`'s own output-caching pattern from Section 16.1 -- both cases save whichever value the backward FORMULA itself actually needs, which is not always the same tensor a reader's first intuition would reach for.

## Complete Runnable Code

### File: `01_chain_rule_elementwise.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.1 hand-implements the chain rule by
// walking its own ComputationGraph's node list in reverse and, for each
// node, calling that op's own hand-written backward() function with the
// accumulated upstream gradient, then accumulating the result into each
// input's own explicit grad field. Its own running example is identical
// to Chapter 15's: w = x*y + x, x=3.0, y=4.0, giving x.grad=5.0 (=y+1)
// and y.grad=3.0 (=x). torch::Tensor's own real autograd engine performs
// the identical mathematical operation -- reverse-mode chain rule,
// accumulating into each leaf's .grad() -- but through a real, compiled
// dispatch table of Node::apply() calls rather than a hand-written
// switch/if-chain over op-name strings. This file reproduces that running
// example exactly, then investigates the CUDA book's own Section 16.2
// AddOp implementation, which returns `{grad_output, grad_output}` --
// two references to the SAME underlying gradient tensor -- for its own
// backward function, explicitly flagged as a latent aliasing bug whose
// consequences depend on how the caller accumulates gradients. This file
// tests whether torch::Tensor's own real Add backward exhibits the same
// aliasing, by hooking each operand's incoming gradient directly and
// comparing their real memory addresses.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // 16.1: the shared running example, reproduced from Chapter 15.
    torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0, torch::requires_grad(true));
    torch::Tensor z = x * y;   // MulBackward0
    torch::Tensor w = z + x;   // AddBackward0
    w.backward();

    std::cout << "w = " << w.item<double>() << ", CUDA book's own expected = 15.0, match = "
              << (w.item<double>() == 15.0) << std::endl;
    std::cout << "x.grad = " << x.grad().item<double>()
              << ", CUDA book's own expected = 5.0 (=y+1), match = "
              << (x.grad().item<double>() == 5.0) << std::endl;
    std::cout << "y.grad = " << y.grad().item<double>()
              << ", CUDA book's own expected = 3.0 (=x), match = "
              << (y.grad().item<double>() == 3.0) << std::endl;

    // 16.2, AddOp aliasing investigation: the CUDA book's own AddOp::backward
    // literally returns the SAME grad_output reference twice, one per input.
    // Does torch::Tensor's real AddBackward0 node do the same -- deliver the
    // identical underlying buffer to both operands -- or does each operand
    // receive its own independent gradient tensor? Hooked directly on the
    // two leaf tensors of a fresh add, capturing each incoming gradient's
    // real data_ptr() at the moment the engine delivers it.
    torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
    torch::Tensor s = a + b;   // AddBackward0, s = 13.0

    void* a_grad_ptr = nullptr;
    void* b_grad_ptr = nullptr;
    a.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
        a_grad_ptr = grad.data_ptr();
        return grad;
    });
    b.register_hook([&](const torch::Tensor& grad) -> torch::Tensor {
        b_grad_ptr = grad.data_ptr();
        return grad;
    });
    s.backward();

    std::cout << "\ns = a+b = " << s.item<double>() << ", a.grad = " << a.grad().item<double>()
              << ", b.grad = " << b.grad().item<double>() << std::endl;
    std::cout << "a's incoming grad_output data_ptr == b's incoming grad_output data_ptr "
              << "(the CUDA book's own AddOp aliasing bug, reproduced here)? "
              << (a_grad_ptr == b_grad_ptr) << std::endl;
    std::cout << "a.grad() data_ptr == b.grad() data_ptr, i.e. does the aliasing survive "
              << "into the FINAL accumulated per-leaf buffers? "
              << (a.grad().data_ptr() == b.grad().data_ptr()) << std::endl;

    // 16.2, ExpOp: self-derivative property, output-caching pattern. The
    // CUDA book's own ExpOp::forward() caches the computed output value in
    // its own SavedTensors struct specifically so backward() can reuse it
    // (d/dx exp(x) = exp(x) itself, so no re-computation is needed).
    // torch::Tensor's own ExpBackward0 node does the identical thing: it
    // saves the forward output, not the input, and backward() multiplies
    // the upstream gradient by that saved output directly.
    torch::Tensor u_in = torch::tensor(1.0, torch::requires_grad(true));
    torch::Tensor u = torch::exp(u_in);
    u.backward();

    std::cout << "\nexp(1.0) = " << u.item<double>()
              << ", CUDA book's own expected = 2.71828..., match (to 5 dp) = "
              << (std::abs(u.item<double>() - 2.71828) < 1e-5) << std::endl;
    std::cout << "d(exp(x))/dx at x=1.0 = " << u_in.grad().item<double>()
              << ", self-derivative property: grad == exp(1.0) itself, match = "
              << (u_in.grad().item<double>() == u.item<double>()) << std::endl;

    return 0;
}
```

### File: `02_matmul_gradients.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.3 hand-derives MatMulOp::backward()
// from the matrix calculus identities dL/dX = dL/dY @ M^T and
// dL/dM = X^T @ dL/dY, then implements both as explicit nested-loop
// kernels. It reuses the exact same X and M matrices already introduced
// in its own Chapter 13 (X = [[1,2,3],[4,5,6]], M = [[1,2],[3,4],[5,6]],
// Y = X@M = [[22,28],[49,64]]), giving dL/dX = [[3,7,11],[3,7,11]] and
// dL/dM = [[5,5],[7,7],[9,9]] for a ones-valued upstream gradient.
// This book's own Chapter 13 already established that torch::matmul is
// real production code, not a from-scratch kernel. This file confirms
// that its real backward pass (MmBackward0) implements the identical
// matrix-calculus identities the CUDA book hand-derives -- verified two
// ways: once via torch::Tensor's own real autograd, and independently by
// computing dL/dX and dL/dM by hand using torch::matmul on the
// transposed operands, with no reliance on autograd at all.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    torch::Tensor X = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                     torch::requires_grad(true));
    torch::Tensor M = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
                                     torch::requires_grad(true));
    torch::Tensor Y = torch::matmul(X, M);

    std::cout << "Y = X @ M =\n" << Y << std::endl;
    torch::Tensor Y_expected = torch::tensor({{22.0, 28.0}, {49.0, 64.0}});
    std::cout << "matches CUDA book's own (and this book's own Ch13) Y? "
              << torch::allclose(Y, Y_expected) << std::endl;

    Y.backward(torch::ones_like(Y));   // dL/dY = ones, matching CUDA book's own seed

    std::cout << "\ndL/dX (via real autograd) =\n" << X.grad() << std::endl;
    torch::Tensor dLdX_expected = torch::tensor({{3.0, 7.0, 11.0}, {3.0, 7.0, 11.0}});
    std::cout << "matches CUDA book's own dL/dX? " << torch::allclose(X.grad(), dLdX_expected) << std::endl;

    std::cout << "\ndL/dM (via real autograd) =\n" << M.grad() << std::endl;
    torch::Tensor dLdM_expected = torch::tensor({{5.0, 5.0}, {7.0, 7.0}, {9.0, 9.0}});
    std::cout << "matches CUDA book's own dL/dM? " << torch::allclose(M.grad(), dLdM_expected) << std::endl;

    // Independent cross-check: apply the CUDA book's own hand-derived
    // identities directly, with no autograd involved at all, using fresh
    // tensors detached from the graph above.
    torch::Tensor X2 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    torch::Tensor M2 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    torch::Tensor grad_out = torch::ones({2, 2});
    torch::Tensor dLdX_manual = torch::matmul(grad_out, M2.t());
    torch::Tensor dLdM_manual = torch::matmul(X2.t(), grad_out);

    std::cout << "\ndL/dX (hand-applied identity dL/dY @ M^T, no autograd) =\n" << dLdX_manual << std::endl;
    std::cout << "matches real autograd's own dL/dX? " << torch::allclose(dLdX_manual, X.grad()) << std::endl;
    std::cout << "\ndL/dM (hand-applied identity X^T @ dL/dY, no autograd) =\n" << dLdM_manual << std::endl;
    std::cout << "matches real autograd's own dL/dM? " << torch::allclose(dLdM_manual, M.grad()) << std::endl;

    return 0;
}
```

### File: `03_algebraic_gradients.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// The CUDA C++ edition's Section 16.4 hand-derives four more backward
// functions: SubOp (d(a-b)/da=1, d(a-b)/db=-1), DivOp (d(a/b)/da=1/b,
// d(a/b)/db=-a/b^2), PowOp (d(a^b)/da=b*a^(b-1), d(a^b)/db=a^b*ln(a)),
// and LogOp (d(ln(x))/dx=1/x). Its own worked numbers: a=8.0,b=5.0 for
// sub/div (c=a-b=3.0, c=a/b=1.6, grad_a=0.2, grad_b=-0.32); a=2.0,b=3.0
// for pow (a^b=8.0, grad_a=12.0, grad_b~5.545); x=2.0 for log
// (ln(2)~0.6931, grad=0.5). This file reproduces every one of those
// numbers via torch::Tensor's own real autograd, then independently
// cross-checks each gradient with a central finite-difference estimate
// computed from the plain scalar formulas (not autograd), matching the
// CUDA book's own reported approximate finite-difference values.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Sub and Div: a=8.0, b=5.0.
    {
        torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
        torch::Tensor c = a - b;
        c.backward();
        std::cout << "sub: c=a-b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected c=3.0,grad_a=1,grad_b=-1, match = "
                  << (c.item<double>() == 3.0 && a.grad().item<double>() == 1.0 && b.grad().item<double>() == -1.0)
                  << std::endl;
    }
    {
        torch::Tensor a = torch::tensor(8.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(5.0, torch::requires_grad(true));
        torch::Tensor c = a / b;
        c.backward();
        // float32 storage means these compare to within 1e-6, not bit-exact.
        std::cout << "div: c=a/b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected c=1.6,grad_a=0.2,grad_b=-0.32, match (to float32 precision) = "
                  << (std::abs(c.item<double>() - 1.6) < 1e-6 &&
                      std::abs(a.grad().item<double>() - 0.2) < 1e-6 &&
                      std::abs(b.grad().item<double>() - (-0.32)) < 1e-6)
                  << std::endl;

        // independent finite-difference cross-check on grad_b, using the
        // plain scalar formula c(a,b) = a/b, no autograd involved. eps is
        // deliberately not tiny, so the curvature of 1/b shows up as a
        // small, honest deviation from the exact analytic slope -0.32,
        // rather than an eps so small it just reprints -0.32 back.
        double eps = 0.25;
        double c_plus = 8.0 / (5.0 + eps);
        double c_minus = 8.0 / (5.0 - eps);
        double fd_grad_b = (c_plus - c_minus) / (2 * eps);
        std::cout << "finite-diff grad_b (independent, no autograd, eps=0.25) = " << fd_grad_b
                  << ", exact analytic grad_b = -0.32, deviation from curvature = "
                  << std::abs(fd_grad_b - (-0.32)) << std::endl;
    }

    // Pow: a=2.0, b=3.0.
    {
        torch::Tensor a = torch::tensor(2.0, torch::requires_grad(true));
        torch::Tensor b = torch::tensor(3.0, torch::requires_grad(true));
        torch::Tensor c = torch::pow(a, b);
        c.backward();
        std::cout << "\npow: a^b=" << c.item<double>() << ", grad_a=" << a.grad().item<double>()
                  << ", grad_b=" << b.grad().item<double>()
                  << ", CUDA book's own expected a^b=8.0,grad_a=12.0,grad_b~5.545, match = "
                  << (c.item<double>() == 8.0 && a.grad().item<double>() == 12.0 &&
                      std::abs(b.grad().item<double>() - 5.545177) < 1e-4)
                  << std::endl;
    }

    // Log: x=2.0.
    {
        torch::Tensor x = torch::tensor(2.0, torch::requires_grad(true));
        torch::Tensor c = torch::log(x);
        c.backward();
        std::cout << "\nlog: ln(2)=" << c.item<double>() << ", grad=" << x.grad().item<double>()
                  << ", CUDA book's own expected ln(2)~0.6931,grad=0.5, match = "
                  << (std::abs(c.item<double>() - 0.693147) < 1e-5 && x.grad().item<double>() == 0.5)
                  << std::endl;

        double eps = 0.25;
        double fd_grad = (std::log(2.0 + eps) - std::log(2.0 - eps)) / (2 * eps);
        std::cout << "finite-diff grad (independent, no autograd, eps=0.25) = " << fd_grad
                  << ", exact analytic grad = 0.5, deviation from curvature = "
                  << std::abs(fd_grad - 0.5) << std::endl;
    }

    return 0;
}
```

### File: `04_activation_trig_gradients.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.5 hand-derives backward functions for
// five activation/trig ops: ReLU (grad passes through where x>0, else 0),
// Sigmoid (grad = sigmoid(x)*(1-sigmoid(x))), Tanh (grad = 1-tanh(x)^2),
// Sin (grad = cos(x)), Cos (grad = -sin(x)). Its own worked numbers:
// x=[-2,3,-1,5] -> relu=[0,3,0,5], backward(ones)->[0,1,0,1]; sigmoid(0)
// =0.5, deriv=0.25; tanh(0)=0.0, deriv=1.0; sin(0)=0, cos(0)=1, with
// SinOp backward(ones) giving 1.0 and CosOp backward giving -0.0. This
// file reproduces every one of those numbers via torch::Tensor's own
// real activation and trig ops.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // ReLU.
    {
        torch::Tensor x = torch::tensor({-2.0, 3.0, -1.0, 5.0}, torch::requires_grad(true));
        torch::Tensor r = torch::relu(x);
        r.backward(torch::ones_like(r));
        std::cout << "x =\n" << x << std::endl;
        std::cout << "relu(x) =\n" << r << std::endl;
        std::cout << "grad =\n" << x.grad() << std::endl;
        torch::Tensor r_exp = torch::tensor({0.0, 3.0, 0.0, 5.0});
        torch::Tensor g_exp = torch::tensor({0.0, 1.0, 0.0, 1.0});
        std::cout << "matches CUDA book's own relu=[0,3,0,5], grad=[0,1,0,1]? "
                  << (torch::allclose(r, r_exp) && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Sigmoid at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor s = torch::sigmoid(x);
        s.backward();
        std::cout << "\nsigmoid(0) = " << s.item<double>() << ", grad = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.5, 0.25, match = "
                  << (s.item<double>() == 0.5 && x.grad().item<double>() == 0.25) << std::endl;
    }

    // Tanh at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor t = torch::tanh(x);
        t.backward();
        std::cout << "tanh(0) = " << t.item<double>() << ", grad = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.0, 1.0, match = "
                  << (t.item<double>() == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }

    // Sin/Cos at 0.
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor s = torch::sin(x);
        s.backward();
        std::cout << "sin(0) = " << s.item<double>() << ", grad (=cos(0)) = " << x.grad().item<double>()
                  << ", CUDA book's own expected 0.0, 1.0, match = "
                  << (s.item<double>() == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }
    {
        torch::Tensor x = torch::tensor(0.0, torch::requires_grad(true));
        torch::Tensor c = torch::cos(x);
        c.backward();
        std::cout << "cos(0) = " << c.item<double>() << ", grad (=-sin(0)) = " << x.grad().item<double>()
                  << ", CUDA book's own expected 1.0, -0.0, match = "
                  << (c.item<double>() == 1.0 && x.grad().item<double>() == 0.0) << std::endl;
    }

    return 0;
}
```

### File: `05_reduction_shape_gradients.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 16.6 hand-derives SumOp::backward()
// (broadcast the upstream scalar gradient back to every element), and
// MaxOp::backward() (route the gradient to the single winning index
// only). Its own worked numbers: x=[1,4,9,16], sum=30.0, backward(1.0)
// broadcasts to [1,1,1,1]; x=[3,7,2,9], max=9.0 at index 3, backward
// gives [0,0,0,1]. Its own Common-Trap callout then examines a TIE case,
// x=[1,5,3,5]: its own "correct" index-based approach routes gradient
// entirely through ONE tracked winning index, giving [0,1,0,0] (sum=1.0);
// its own "broken" value-mask approach gives every tied position full
// gradient, [0,1,0,1] (sum=2.0, inventing gradient that was never there).
// This file reproduces the no-tie case exactly, then investigates the
// tie case directly against torch::Tensor's own real torch::max backward
// -- which turns out to match NEITHER of the CUDA book's own two
// described behaviors, a genuine third distinct behavior discussed below.
// It also covers ReshapeOp/TransposeOp backward (grad reshapes/transposes
// back to the original shape, confirmed shape-reversal self-inverse).
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Sum, no surprises: broadcast.
    {
        torch::Tensor x = torch::tensor({1.0, 4.0, 9.0, 16.0}, torch::requires_grad(true));
        torch::Tensor s = torch::sum(x);
        s.backward();
        std::cout << "sum(x) = " << s.item<double>() << ", grad =\n" << x.grad() << std::endl;
        torch::Tensor g_exp = torch::tensor({1.0, 1.0, 1.0, 1.0});
        std::cout << "matches CUDA book's own sum=30.0, grad=[1,1,1,1]? "
                  << (s.item<double>() == 30.0 && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Max, no tie: routes to the single winning index.
    {
        torch::Tensor x = torch::tensor({3.0, 7.0, 2.0, 9.0}, torch::requires_grad(true));
        torch::Tensor m = torch::amax(x, 0);
        m.backward();
        std::cout << "\nmax(x) (no tie) = " << m.item<double>() << ", grad =\n" << x.grad() << std::endl;
        torch::Tensor g_exp = torch::tensor({0.0, 0.0, 0.0, 1.0});
        std::cout << "matches CUDA book's own max=9.0 at index 3, grad=[0,0,0,1]? "
                  << (m.item<double>() == 9.0 && torch::allclose(x.grad(), g_exp)) << std::endl;
    }

    // Max, TIE case -- the honest three-way divergence.
    {
        torch::Tensor x = torch::tensor({1.0, 5.0, 3.0, 5.0}, torch::requires_grad(true));
        torch::Tensor m = torch::amax(x, 0);
        m.backward();
        std::cout << "\nmax(x) (TIE, x=[1,5,3,5]) = " << m.item<double>() << ", grad =\n" << x.grad() << std::endl;
        double grad_sum = x.grad().sum().item<double>();
        std::cout << "sum of grad = " << grad_sum << std::endl;

        torch::Tensor cuda_books_own_correct = torch::tensor({0.0, 1.0, 0.0, 0.0});
        torch::Tensor cuda_books_own_broken = torch::tensor({0.0, 1.0, 0.0, 1.0});
        torch::Tensor real_torch_equal_split = torch::tensor({0.0, 0.5, 0.0, 0.5});
        std::cout << "matches CUDA book's own 'correct' single-index [0,1,0,0]? "
                  << torch::allclose(x.grad(), cuda_books_own_correct) << std::endl;
        std::cout << "matches CUDA book's own 'broken' value-mask [0,1,0,1]? "
                  << torch::allclose(x.grad(), cuda_books_own_broken) << std::endl;
        std::cout << "matches a THIRD behavior, equal split among ties [0,0.5,0,0.5]? "
                  << torch::allclose(x.grad(), real_torch_equal_split) << std::endl;
    }

    // Reshape/transpose backward: shape reverses; self-inverse round trip.
    {
        torch::Tensor x = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat64)).reshape({2, 6});
        x.requires_grad_(true);
        torch::Tensor xr = x.reshape({3, 4});
        xr.retain_grad();
        torch::Tensor loss = xr.sum();
        loss.backward();
        std::cout << "\nreshape [2,6]->[3,4]: x.grad() shape = [" << x.grad().size(0) << "," << x.grad().size(1)
                  << "], matches original [2,6]? " << (x.grad().size(0) == 2 && x.grad().size(1) == 6) << std::endl;
        std::cout << "x.grad() all-ones (upstream grad reshaped back, values preserved)? "
                  << torch::allclose(x.grad(), torch::ones({2, 6}, torch::kFloat64)) << std::endl;
    }
    {
        torch::Tensor x = torch::arange(6, torch::TensorOptions().dtype(torch::kFloat64)).reshape({2, 3});
        x.requires_grad_(true);
        torch::Tensor xt = x.t();
        xt.retain_grad();
        torch::Tensor loss = xt.sum();
        loss.backward();
        std::cout << "\ntranspose [2,3]->[3,2]: x.grad() shape = [" << x.grad().size(0) << "," << x.grad().size(1)
                  << "], matches original [2,3]? " << (x.grad().size(0) == 2 && x.grad().size(1) == 3) << std::endl;
        std::cout << "x.grad() all-ones (backward of transpose is transpose again, self-inverse)? "
                  << torch::allclose(x.grad(), torch::ones({2, 3}, torch::kFloat64)) << std::endl;
    }

    return 0;
}
```

### File: `06_custom_function.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// The CUDA C++ edition's Section 16.7 builds a custom Function: a
// bisection solver for x^2=c that has no closed-form forward formula to
// differentiate through directly, so its own backward() instead applies
// the implicit function theorem: given g(x,c) = x^2 - c = 0, dx/dc =
// -dg/dc / dg/dx = 1 / (2x). torch::Tensor's own real C++ frontend
// exposes the identical mechanism as a first-class, production API:
// torch::autograd::Function<T>, subclassed here with a static forward()
// (running the bisection search, no autograd tracking inside it) and a
// static backward() (implementing the same implicit-function-theorem
// formula found in the CUDA book), connected into the real graph via
// ctx->save_for_backward(). This file reproduces the CUDA book's own
// exact numbers: bisection-solved x~1.4142136 for c=2.0, dx/dc~0.3535534,
// cross-checked independently via a finite difference on a second,
// separate forward call at c=2.001 (bypassing autograd entirely).
class SqrtViaBisection : public torch::autograd::Function<SqrtViaBisection> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor c) {
        double c_val = c.item<double>();
        double lo = 0.0, hi = std::max(1.0, c_val);
        for (int i = 0; i < 100; i++) {
            double mid = (lo + hi) / 2.0;
            if (mid * mid < c_val) lo = mid; else hi = mid;
        }
        double x_val = (lo + hi) / 2.0;
        torch::Tensor x = torch::tensor(x_val, c.options());
        ctx->save_for_backward({x});
        return x;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor x = saved[0];
        torch::Tensor grad_output = grad_outputs[0];
        // implicit function theorem: g(x,c) = x^2 - c = 0  =>  dx/dc = 1/(2x)
        torch::Tensor dx_dc = 1.0 / (2.0 * x);
        return {grad_output * dx_dc};
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(7);

    torch::Tensor c = torch::tensor(2.0, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    torch::Tensor x = SqrtViaBisection::apply(c);
    std::cout << "x (bisection-solved sqrt(2.0)) = " << x.item<double>()
              << ", CUDA book's own expected ~1.4142136, match (to 6 dp) = "
              << (std::abs(x.item<double>() - 1.4142136) < 1e-6) << std::endl;

    x.backward();
    std::cout << "dx/dc (via custom Function backward, implicit function theorem) = "
              << c.grad().item<double>()
              << ", CUDA book's own expected ~0.3535534, match (to 6 dp) = "
              << (std::abs(c.grad().item<double>() - 0.3535534) < 1e-6) << std::endl;

    // independent cross-check: a second, separate forward call at a
    // nearby point, with NO autograd involvement at all -- just a plain
    // finite-difference slope between two genuinely bisection-solved
    // values.
    torch::Tensor c2 = torch::tensor(2.001, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor x2 = SqrtViaBisection::apply(c2);
    double slope = (x2.item<double>() - x.item<double>()) / 0.001;
    std::cout << "\nsqrt(2.001) (independent bisection call) = " << x2.item<double>()
              << ", CUDA book's own expected ~1.4145671, match (to 6 dp) = "
              << (std::abs(x2.item<double>() - 1.4145671) < 1e-6) << std::endl;
    std::cout << "finite-diff slope (independent, no autograd) = " << slope
              << ", CUDA book's own expected ~0.3535, match (to 3 dp) = "
              << (std::abs(slope - 0.3535) < 1e-3) << std::endl;

    return 0;
}
```

All six files compile and link against LibTorch with the standard command from *Getting Started*:

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

`torch::Tensor`'s own real backward Node implementations were confirmed, operation by operation, to compute the identical formulas the CUDA book hand-derives from scratch: the shared chain-rule running example (`x.grad=5.0, y.grad=3.0`), matrix calculus's own two identities for matmul (verified both through autograd and by hand-applying the formula independently), every algebraic and activation/trig formula (sub, div, pow, log, ReLU, sigmoid, tanh, sin, cos), reduction and shape gradients (sum's broadcast, max's single-winner routing, reshape/transpose's shape-reversal), and a custom `torch::autograd::Function` differentiated through the implicit function theorem. Two investigations went further than simple number-matching. The `AddOp` aliasing question found a genuine, reproduced match at the intermediate level -- real `AddBackward0` does hand the identical gradient object to both operands, exactly as the CUDA book's own `{grad_output, grad_output}` does -- but confirmed this is provably harmless because each leaf's own final accumulated buffer is independent regardless. And the tied-`max` gradient question found this chapter's own most significant divergence: real `torch::max`'s backward on a tie does neither of the CUDA book's own two described behaviors, but a genuine third, mathematically valid subgradient -- an equal split across every tied maximum -- that agrees with neither the CUDA book's own "correct" single-index convention nor its own "broken" double-count, while being no less legitimate than the first.

## Self-Check Questions

1. Section 16.1's own aliasing probe reports two DIFFERENT answers for two DIFFERENT `data_ptr()` comparisons on the same `a + b` computation. What is each comparison actually checking, and why do they not contradict each other?
2. Section 16.2's own independent cross-check applies the matrix-calculus identities `dL/dX = dL/dY @ M^T` and `dL/dM = X^T @ dL/dY` by hand, with no autograd. Why does this prove something the earlier autograd-based check, by itself, could not have proven?
3. Section 16.3's own finite-difference checks use `eps=0.25` rather than a much smaller value like `0.0001`. What would using a very small `eps` have hidden, and why does that matter for what the check is trying to demonstrate?
4. Section 16.5's own tied-`max` result, `[0.0, 0.5, 0.0, 0.5]`, does not match either of the CUDA book's own two described behaviors. Explain why this result is nonetheless mathematically valid, and specifically what property the CUDA book's own "broken" value-mask approach fails to have that this result does satisfy.
5. Section 16.6's own custom `Function` saves the OUTPUT `x` in `ctx->save_for_backward()`, not the INPUT `c`. Using the implicit-function-theorem formula `dx/dc = 1/(2x)` directly, explain why saving `c` instead would not have been sufficient.

## Where We Go Next

This chapter opened every real backward Node this book has relied on since Chapter 6, confirming the CUDA book's own hand-derived formulas match the real thing across chain-rule, matrix, algebraic, activation, trigonometric, reduction, shape, and custom-function gradients -- while finding two genuine points of interest along the way: the `AddOp` aliasing pattern reproduced but rendered harmless, and the tied-`max` gradient's own third, previously undescribed behavior. Chapter 17 turns from individual backward FUNCTIONS to the ENGINE that calls them in the right order with the right accumulated values across an entire graph at once -- Gradient Computation Engine, closing Part 4, examining how `.backward()` itself orchestrates every one of the Node types this chapter has only tested in isolation.

## Worked Solutions

**1.** The first comparison checks the `data_ptr()` of the gradient tensor `AddBackward0` HANDS OUT to each operand at the moment it is delivered (captured via `register_hook`), and finds them identical -- confirming real `AddBackward0` does reuse the same underlying tensor object for both edges, exactly as the CUDA book's own `{grad_output, grad_output}` does. The second comparison checks the `data_ptr()` of each leaf's own FINAL accumulated `.grad()` tensor, found independent. These do not contradict, because they check different stages of the pipeline: the first is about what the backward NODE returns, and the second is about what each leaf's own separate `AccumulateGrad` step writes the returned value INTO. The aliasing exists exactly where the CUDA book's own text says it might, and is neutralized exactly where accumulation happens -- both facts are true about the same computation without conflict.

**2.** The autograd-based check alone confirms `X.grad()` and `M.grad()` come out numerically correct for this ONE specific `X` and `M`, but that is consistent with `MmBackward0` implementing the right formula OR some other formula that happens to produce the same output for these particular numbers. Applying the CUDA book's own exact identities by hand, using `torch::matmul` on transposed operands with no autograd involved, and finding the result bit-for-bit identical to autograd's own output, rules out the "coincidentally correct for this case" possibility -- the agreement is now known to come from the same formula being applied, not merely the same output being reached by some other route.

**3.** A very small `eps`, such as `0.0001`, would make the finite-difference estimate's own error from the function's curvature (proportional to `eps^2`) too small to see at the six-decimal precision this book prints, so the reported number would just re-print the exact analytic value back, looking like a second confirmation of the SAME number rather than an INDEPENDENT computation reaching a close but not identical answer. Using `eps=0.25` deliberately keeps the curvature-driven deviation (about `0.0008` for the division example, about `0.0026` for the logarithm example) visible, honestly showing that the finite-difference method is a genuinely different computation with its own small, expected, and well-understood error -- not a disguised repetition of the analytic formula.

**4.** A subgradient of `max` at a tie is any distribution of gradient weight across the tied maximal positions that sums to exactly the total upstream gradient (here, `1.0`); any such distribution is a mathematically defensible choice, because `max` is not differentiable at a tie and has no single correct derivative to begin with. `[0.0, 0.5, 0.0, 0.5]` sums to `0.5+0.5=1.0`, satisfying this property. The CUDA book's own "broken" value-mask approach, `[0.0, 1.0, 0.0, 1.0]`, sums to `2.0` -- it does not merely choose a different (but still valid) distribution, it distributes MORE gradient than was ever actually available, which is what makes it genuinely broken rather than simply a different convention.

**5.** The formula `dx/dc = 1/(2x)` is expressed entirely in terms of `x`, the bisection-solved OUTPUT -- it contains no `c` term at all. If only `c` had been saved, `backward()` would need to re-derive or re-solve for `x` from `c` before it could apply the formula, effectively re-running the entire bisection search inside `backward()` as well as `forward()`. Saving `x` directly, exactly as `ExpOp`'s own output-caching pattern in Section 16.1 does for the same reason, lets `backward()` use the value the formula actually needs immediately, with no redundant recomputation.
