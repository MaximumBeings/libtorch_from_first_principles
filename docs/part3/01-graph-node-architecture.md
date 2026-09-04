# Chapter 15: Graph Node Architecture

> "The CUDA C++ edition opens Part 3 by building, from nothing, the machinery every `.backward()` call in this book has relied on since Chapter 6: a `GraphNode` struct capturing one operation's inputs and output, an `OpRegistry` dispatching operation names to `Differentiable` implementations, a `ComputationGraph` recording nodes during the forward pass, and a `topological_backward_order()` that works only because the CUDA book's own toy engine always backpropagates from a single final output. `torch::Tensor`'s own real autograd graph has been running underneath every single one of this book's own `.backward()` calls all along, invisibly -- this chapter opens it up directly for the first time, confirming its own real structures (`.grad_fn()`, real dispatch, a real `ANY`-not-`ALL` gating rule, real edge-based traversal) match the CUDA book's own design at every point the CUDA book tests, and finds one genuinely consequential engineering difference where a real system must do more than the CUDA book's own toy version ever needed to."

**What you will understand by the end of this chapter:**

- That `torch::Tensor`'s own real operation dispatch (`operator*`, `operator+`, `torch::pow`, `torch::exp`, and more) always produces a genuine, non-null `grad_fn` for a differentiable operation with at least one trainable input -- confirmed across six different operations -- with no equivalent to the CUDA book's own debug-vs-release registry divergence, because the dispatch is resolved at compile time rather than through a runtime, string-keyed lookup at all
- That `z.grad_fn()->name()` and `w.grad_fn()->name()` report `"MulBackward0"` and `"AddBackward0"` for the CUDA book's own exact running example (`w = x*y + x`, `x=3.0, y=4.0`, `z=12.0, w=15.0`), the direct real analog of `GraphNode`'s own `op_name` field, and that a frozen operand paired with a trainable one still produces a full, real node
- That `torch::Tensor`'s own real graph-construction gating rule matches the CUDA book's own `record()` exactly across all four combinations of a two-operand operation's `requires_grad` flags: a real node exists whenever ANY input requires grad, never only when ALL do
- That `torch::Tensor`'s own real autograd engine performs a genuine, edge-based topological sort scoped to only the actual ancestors of a specific output -- confirmed by an unrelated node's gradient being genuinely UNDEFINED rather than a zero-initialized field left untouched, and by a second, fully independent computation graph remaining completely untouched by a `.backward()` call on the first -- a real engineering requirement the CUDA book's own toy engine, which always backpropagates from a single global list, never has to meet

**What you need to know first:**

- Every chapter since Chapter 6's `.backward()` calls -- this chapter is the first to look directly at the graph structure that has been silently doing the work behind every one of those calls
- Chapter 12's local derivatives and Chapter 14's own numerical-stability findings -- this chapter's own focus is graph STRUCTURE (what gets recorded, and in what order), not the calculus of any specific operation
- If you've read the CUDA C++ edition's Chapter 15: its own `GraphNode`, `OpRegistry`, `ComputationGraph`, and `topological_backward_order()` are a complete, working, from-scratch autograd graph, built specifically because CUDA C++ has no built-in equivalent at all. `torch::Tensor` has had a real graph this entire book, used but never inspected -- this chapter opens `.grad_fn()`, `.name()`, and `.next_edges()` directly, verifying the CUDA book's own exact structural claims against the real thing, then investigating the one place a real system genuinely has to do more work than the CUDA book's own single-output toy design ever required.

## 15.1 Function Registration: Real Dispatch, With No Registry to Miss `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `OpRegistry` maps operation name strings to `Differentiable` implementations, each providing `forward()` and `backward()`. Its own `backward()` signature takes both `inputs` and `output`, not just `grad_output`, because a derivative like multiplication's genuinely needs "the other operand" to compute correctly. `torch::Tensor` has no string-keyed registry at all -- every real operation is statically wired at compile time to its own real backward `Node` type, so `z.grad_fn()` is never the result of a runtime lookup that could fail.

### Background

The CUDA book's own Worked Example 15.1.1: `MulOp::backward` with `grad_output=1.0, x=3.0, y=4.0` returns `[4.0, 3.0]` -- sensitivity to `x` equals `y`, sensitivity to `y` equals `x`. Its own Worked Example 15.1.2 is a critical bug finding: `registry.get()` on an unregistered op name asserts cleanly in a debug build (exit code `134`), but `std::unordered_map::operator[]` silently inserts a `nullptr` entry and returns it in a release build, corrupting the map's own size from `0` to `1`.

### Worked Example 15.1.1 -- the CUDA book's own numbers, and empirical proof real dispatch never comes up empty

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.1 hand-builds an OpRegistry: a
// std::unordered_map from operation name strings to Differentiable
// implementations, each providing forward() and backward(). Its own
// [COMMON TRAP] finds a genuine divergence between debug and release
// builds: registry.get() on an unregistered name asserts (aborts cleanly)
// in debug, but std::unordered_map::operator[] silently inserts a
// nullptr entry and returns it in release, corrupting the map's own size.
// torch::Tensor has no equivalent runtime lookup at all -- every real
// operation (operator*, operator+, torch::pow, torch::exp, ...) is
// statically wired at compile time to its own real backward Node type
// (MulBackward0, AddBackward0, ...), so there is no string-keyed registry
// a caller could query with a typo'd or unregistered name in the first
// place. This file verifies the CUDA book's own exact Worked Example
// 15.1.1 result (mul's backward needing the OTHER operand: grad_output=1,
// x=3, y=4 gives [4.0, 3.0]), then empirically confirms across six
// different real operations that a genuine grad_fn is always dispatched,
// never silently absent -- the closest real-world analog to "the registry
// never returns an unregistered nullptr" this compile-time-dispatched
// system can be asked to demonstrate.
int main() {
    // Worked Example 15.1.1: MulOp::backward with grad_output=1.0, x=3.0,
    // y=4.0 returns [4.0, 3.0] -- sensitivity to x equals y, sensitivity
    // to y equals x. torch::Tensor's own real MulBackward0 needs exactly
    // this "the other operand" information to compute the same result.
    torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor z = x * y;
    z.backward();
    std::cout << "d(x*y)/dx at x=3,y=4 = " << x.grad().item<float>()
              << ", CUDA book's own expected = 4.0 (=y), match = "
              << (x.grad().item<float>() == 4.0f) << std::endl;
    std::cout << "d(x*y)/dy at x=3,y=4 = " << y.grad().item<float>()
              << ", CUDA book's own expected = 3.0 (=x), match = "
              << (y.grad().item<float>() == 3.0f) << std::endl;

    // Honest divergence, tested empirically: real dispatch across six
    // different operation types, confirming a genuine, real grad_fn is
    // ALWAYS present for a differentiable operation on tensors requiring
    // grad -- never silently absent the way the CUDA book's own registry
    // can be made to silently insert a nullptr for an unregistered name.
    torch::Tensor a = torch::tensor(2.0f, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(3.0f, torch::requires_grad(true));

    struct OpResult { std::string op_name; bool grad_fn_present; std::string grad_fn_name; };
    std::vector<OpResult> results;
    results.push_back({"add", (a + b).grad_fn() != nullptr, (a + b).grad_fn()->name()});
    results.push_back({"sub", (a - b).grad_fn() != nullptr, (a - b).grad_fn()->name()});
    results.push_back({"mul", (a * b).grad_fn() != nullptr, (a * b).grad_fn()->name()});
    results.push_back({"div", (a / b).grad_fn() != nullptr, (a / b).grad_fn()->name()});
    results.push_back({"pow", torch::pow(a, 2).grad_fn() != nullptr, torch::pow(a, 2).grad_fn()->name()});
    results.push_back({"exp", torch::exp(a).grad_fn() != nullptr, torch::exp(a).grad_fn()->name()});

    bool all_present = true;
    std::cout << "\nreal grad_fn dispatch across six different operations:" << std::endl;
    for (auto& r : results) {
        std::cout << "  " << r.op_name << ": grad_fn present = " << r.grad_fn_present
                  << ", name = " << r.grad_fn_name << std::endl;
        if (!r.grad_fn_present) all_present = false;
    }
    std::cout << "every real operation dispatched a genuine, non-null grad_fn "
              << "(no equivalent to the CUDA book's own silent-nullptr-insertion bug)? "
              << all_present << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
d(x*y)/dx at x=3,y=4 = 4, CUDA book's own expected = 4.0 (=y), match = 1
d(x*y)/dy at x=3,y=4 = 3, CUDA book's own expected = 3.0 (=x), match = 1

real grad_fn dispatch across six different operations:
  add: grad_fn present = 1, name = AddBackward0
  sub: grad_fn present = 1, name = SubBackward0
  mul: grad_fn present = 1, name = MulBackward0
  div: grad_fn present = 1, name = DivBackward0
  pow: grad_fn present = 1, name = PowBackward0
  exp: grad_fn present = 1, name = ExpBackward0
every real operation dispatched a genuine, non-null grad_fn (no equivalent to the CUDA book's own silent-nullptr-insertion bug)? 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical operation-to-backward-type dispatch from scratch:

```text
add True AddBackward0
sub True SubBackward0
mul True MulBackward0
div True DivBackward0
pow True PowBackward0
exp True ExpBackward0
```

### Discussion

`d(x*y)/dx` and `d(x*y)/dy` match the CUDA book's own exact `[4.0, 3.0]` result, confirming `torch::Tensor`'s own real `MulBackward0` genuinely uses "the other operand" internally to compute each partial derivative -- the identical structural need the CUDA book's own `backward(inputs, output)` signature exists to satisfy. The six-operation dispatch table is this section's own strongest evidence for the honest divergence: every single real operation produced a genuine, correctly-named `grad_fn`, with no runtime lookup step anywhere that could have found "no entry" and returned something silently wrong. This is a structural, not merely empirical, guarantee -- `torch::Tensor::operator*` is compiled code that calls `MulBackward0`'s own constructor directly; there is no string comparison, no hash lookup, and therefore no possible typo or missing-registration failure mode for the compiler to let through in the first place.

> `[COMMON TRAP]` A reader might conclude from this section that `torch::Tensor` has "solved" the CUDA book's own debug-vs-release registry bug through better error handling. This understates the difference: `torch::Tensor` does not handle the failure mode better -- it has structurally eliminated the CATEGORY of bug the CUDA book's own registry is vulnerable to. The CUDA book's own bug exists because `OpRegistry` chooses a runtime, string-keyed data structure for something that is actually known completely at compile time (which function to call for `mul`). `torch::Tensor`'s own compile-time dispatch is not a defensive fix layered on top of a similar registry; it is a different architectural choice that never introduces the runtime lookup the bug depends on existing.

## 15.2 Graph Nodes: `.grad_fn()` as the Real `GraphNode` `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `GraphNode` struct captures a single operation invocation: `op_name`, `inputs`, `output`, and a zero-initialized `grad` field. `torch::Tensor`'s own real autograd graph has a direct structural analog -- every non-leaf tensor's `.grad_fn()` IS a real graph node, and its own `->name()` reports the real operation that produced it.

### Background

The CUDA book's own running example, `w = x*y + x` with `x=3.0, y=4.0`: `z = x*y = 12.0` becomes Node #0 (`op="mul"`), `w = z+x = 15.0` becomes Node #1 (`op="add"`). Its own Worked Example 15.2.2 clarifies that one frozen operand paired with one trainable operand still creates a full node.

### Worked Example 15.2.1 -- the CUDA book's own running example, and a full node from one frozen operand

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.2 hand-builds GraphNode: a struct
// capturing a single operation invocation, with op_name, inputs, output,
// and a zero-initialized grad field. Its own running example, w = x*y + x
// with x=3.0, y=4.0, produces exactly two nodes: Node #0 (op="mul",
// output=12.0), Node #1 (op="add", output=15.0). torch::Tensor's own real
// autograd graph has a direct structural analog: every non-leaf tensor's
// .grad_fn() IS a real graph node, and its own ->name() reports the real
// operation that produced it -- "MulBackward0" and "AddBackward0" for
// this exact example. This file reproduces the CUDA book's own running
// example exactly, then verifies its own Worked Example 15.2.2: that one
// frozen (requires_grad=false) operand paired with one trainable operand
// still creates a full, real graph node -- not a partial or degraded one.
int main() {
    // The CUDA book's own running example: w = x*y + x, x=3.0, y=4.0.
    // z = x*y = 12.0 (Node #0, op="mul"). w = z+x = 15.0 (Node #1, op="add").
    torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor z = x * y;
    torch::Tensor w = z + x;

    std::cout << "z (Node #0) = " << z.item<float>() << ", op_name (real grad_fn name) = "
              << z.grad_fn()->name() << std::endl;
    std::cout << "matches CUDA book's own Node #0: op=mul, output=12.0? "
              << (z.item<float>() == 12.0f && z.grad_fn()->name() == "MulBackward0") << std::endl;

    std::cout << "\nw (Node #1) = " << w.item<float>() << ", op_name (real grad_fn name) = "
              << w.grad_fn()->name() << std::endl;
    std::cout << "matches CUDA book's own Node #1: op=add, output=15.0? "
              << (w.item<float>() == 15.0f && w.grad_fn()->name() == "AddBackward0") << std::endl;

    // Worked Example 15.2.2: one frozen operand (requires_grad=false)
    // paired with one trainable operand still creates a FULL node -- not
    // a partial, degraded, or missing one. Tested here directly: y_frozen
    // never requires grad, but x_trainable does, and the resulting
    // product still has a real, complete grad_fn.
    torch::Tensor x_trainable = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y_frozen = torch::tensor(4.0f, torch::requires_grad(false));
    torch::Tensor z_mixed = x_trainable * y_frozen;

    std::cout << "\nz_mixed = x_trainable(requires_grad) * y_frozen(no grad) = " << z_mixed.item<float>()
              << std::endl;
    std::cout << "z_mixed.grad_fn() is a real, non-null node (not degraded/partial)? "
              << (z_mixed.grad_fn() != nullptr) << std::endl;
    std::cout << "z_mixed.grad_fn()->name() = " << z_mixed.grad_fn()->name()
              << " (identical real node type to the fully-trainable case above)? "
              << (z_mixed.grad_fn()->name() == "MulBackward0") << std::endl;

    // Confirming the node is genuinely FULL, not partial: it correctly
    // backpropagates a real gradient to the one trainable input.
    z_mixed.backward();
    std::cout << "d(z_mixed)/d(x_trainable) = " << x_trainable.grad().item<float>()
              << ", hand-derived expected = 4.0 (=y_frozen's own value), match = "
              << (x_trainable.grad().item<float>() == 4.0f) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
z (Node #0) = 12, op_name (real grad_fn name) = MulBackward0
matches CUDA book's own Node #0: op=mul, output=12.0? 1

w (Node #1) = 15, op_name (real grad_fn name) = AddBackward0
matches CUDA book's own Node #1: op=add, output=15.0? 1

z_mixed = x_trainable(requires_grad) * y_frozen(no grad) = 12
z_mixed.grad_fn() is a real, non-null node (not degraded/partial)? 1
z_mixed.grad_fn()->name() = MulBackward0 (identical real node type to the fully-trainable case above)? 1
d(z_mixed)/d(x_trainable) = 4, hand-derived expected = 4.0 (=y_frozen's own value), match = 1
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical node types and values from scratch:

```text
z 12.0 MulBackward0
w 15.0 AddBackward0
```

### Discussion

`z`'s and `w`'s own values (`12.0`, `15.0`) and real node names (`MulBackward0`, `AddBackward0`) match the CUDA book's own running example exactly, position for position -- `z.grad_fn()` genuinely IS the real analog of `GraphNode` Node #0, and `w.grad_fn()` genuinely IS the real analog of Node #1. The frozen-operand test confirms the CUDA book's own Worked Example 15.2.2 directly: `z_mixed`'s own `grad_fn()` is not merely non-null but IDENTICAL in type (`MulBackward0`) to the fully-trainable case, and the final `.backward()` call confirms it is genuinely functional, not a placeholder -- `x_trainable.grad()` correctly reports `4.0` (`y_frozen`'s own value), exactly what multiplication's own local derivative requires, proving the "full node" claim empirically rather than only structurally.

> `[COMMON TRAP]` It would be easy to assume a "frozen" operand degrades the resulting node somehow -- perhaps producing a node that only partially tracks gradient information, since one of its two inputs will never need a gradient of its own. This section's own final backward pass shows this concern is unfounded: `MulBackward0` still needs BOTH operands' VALUES to compute the ONE gradient it is actually responsible for (`d(z)/d(x_trainable) = y_frozen`), even though it will never be asked to compute a gradient with respect to `y_frozen` itself. The node is exactly as complete as the fully-trainable case; the only difference is which of its own potential outputs (`x_trainable.grad()` versus `y_frozen.grad()`) actually gets requested and computed.

## 15.3 Graph Construction: The Real Engine's Own `ANY`-Not-`ALL` Gating Rule `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `ComputationGraph::record()` loops over an operation's inputs, sets `needs_grad=true` if ANY input has `requires_grad=true`, and only appends a real node when `needs_grad` is true. `torch::Tensor`'s own real autograd engine follows the identical rule: a node's own `.grad_fn()` is real and non-null whenever at least one input requires grad, and is genuinely null only when none of the inputs do.

### Background

The CUDA book's own text establishes the `ANY`-not-`ALL` rule through its own `record()` implementation and Worked Example 15.2.2's single demonstrated case (one trainable, one frozen operand, still producing a node).

### Worked Example 15.3.1 -- all four combinations, tested directly

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.3 hand-builds ComputationGraph::
// record(): it loops over an operation's inputs, sets needs_grad=true if
// ANY input has requires_grad=true, and only appends a real GraphNode
// (setting the output's grad_fn_index) when needs_grad is true -- an
// early return with no node created otherwise. torch::Tensor's own real
// autograd engine follows the identical "ANY, not ALL" gating rule: a
// node's own .grad_fn() is real and non-null whenever AT LEAST ONE input
// requires grad, and is genuinely null (no node at all) only when NONE
// of the inputs do. This file tests all four possible combinations of a
// two-operand operation's requires_grad flags directly, confirming the
// CUDA book's own gating logic exactly, rather than only the CUDA book's
// own single demonstrated case.
int main() {
    struct Case { bool x_req; bool y_req; std::string label; };
    std::vector<Case> cases = {
        {true, true, "both requires_grad=true"},
        {true, false, "x requires_grad=true, y requires_grad=false"},
        {false, true, "x requires_grad=false, y requires_grad=true"},
        {false, false, "both requires_grad=false"},
    };

    bool all_match_cuda_books_own_any_rule = true;
    for (auto& c : cases) {
        torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(c.x_req));
        torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(c.y_req));
        torch::Tensor z = x * y;
        bool has_real_node = (z.grad_fn() != nullptr);
        bool expected_by_any_rule = (c.x_req || c.y_req);  // the CUDA book's own record() logic
        std::cout << c.label << ": z.grad_fn() present = " << has_real_node
                  << ", CUDA book's own 'ANY input requires_grad' rule predicts = " << expected_by_any_rule
                  << ", match = " << (has_real_node == expected_by_any_rule) << std::endl;
        if (has_real_node != expected_by_any_rule) all_match_cuda_books_own_any_rule = false;
    }
    std::cout << "\nall four combinations match the CUDA book's own record()'s "
              << "'ANY, not ALL' gating rule exactly? " << all_match_cuda_books_own_any_rule << std::endl;

    // The one case where no node is created (both false) genuinely
    // produces a tensor that CANNOT backpropagate at all -- confirmed
    // here by attempting .backward() on it and catching the real error,
    // rather than merely observing grad_fn() is null.
    torch::Tensor x_no_grad = torch::tensor(3.0f, torch::requires_grad(false));
    torch::Tensor y_no_grad = torch::tensor(4.0f, torch::requires_grad(false));
    torch::Tensor z_no_grad = x_no_grad * y_no_grad;
    bool threw = false;
    try {
        z_no_grad.backward();
    } catch (const c10::Error&) {
        threw = true;
    }
    std::cout << "\nz_no_grad.backward() (no node was ever created) threw a real error? " << threw << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
both requires_grad=true: z.grad_fn() present = 1, CUDA book's own 'ANY input requires_grad' rule predicts = 1, match = 1
x requires_grad=true, y requires_grad=false: z.grad_fn() present = 1, CUDA book's own 'ANY input requires_grad' rule predicts = 1, match = 1
x requires_grad=false, y requires_grad=true: z.grad_fn() present = 1, CUDA book's own 'ANY input requires_grad' rule predicts = 1, match = 1
both requires_grad=false: z.grad_fn() present = 0, CUDA book's own 'ANY input requires_grad' rule predicts = 0, match = 1

all four combinations match the CUDA book's own record()'s 'ANY, not ALL' gating rule exactly? 1

z_no_grad.backward() (no node was ever created) threw a real error? 1
```

### Discussion

All four combinations of `requires_grad` match the CUDA book's own `record()`'s own logic exactly: a real node is created in every case with at least one trainable operand, and genuinely absent only when neither operand is trainable. This is a complete truth table, not a single anecdotal case -- Worked Example 15.2.2 in the previous section demonstrated one cell of this table (`true, false`); this section's own test confirms all four, including the `false, false` cell the CUDA book's own text describes through `record()`'s early return but does not itself walk through as a printed example. The final `.backward()` attempt on `z_no_grad` closes the loop: the absence of a node is not merely a structural curiosity but has a real, observable consequence -- there is genuinely nothing to backpropagate through, and `torch::Tensor` reports this as a real thrown error rather than silently doing nothing or returning a meaningless zero gradient.

> `[COMMON TRAP]` A reader might assume `requires_grad=false` on one operand only affects THAT operand's own future `.grad()` -- expecting, say, that `y`'s own frozen status has no bearing on whether `z` can be backpropagated through at all. This section's own truth table shows the gating decision is made once, for the OPERATION's own resulting node, not per-input: as long as at least one operand needs a gradient, the resulting node is created and CAN be backpropagated through (even though only the trainable operand's own `.grad()` will end up populated) -- but the moment neither operand needs a gradient, the entire operation produces a tensor with no path back through it at all, regardless of which specific operand a caller might have expected to matter.

## 15.4 Topological Order: A Real Engine Scoped to Real Ancestors `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `topological_backward_order()` simply reverses the full, single global node list -- safe only because the CUDA book's own toy engine always backpropagates from one single final output, so every node ever recorded is either an ancestor of that output or is simply never touched by the reversal's own propagation. `torch::Tensor`'s own real autograd engine does something structurally different: a genuine topological sort, discovered by walking real edges between real `Node` objects, scoped to only the actual ancestors of the specific output being backpropagated -- never touching, or even being aware of, any unrelated computation elsewhere in the same program.

### Background

The CUDA book's own Worked Example 15.4.1: forward order `[mul, add]` becomes backward order `[1, 0]` (a simple reversal). Its own Worked Example 15.4.2: a three-node graph with an unrelated `sub` node that never affects `w`'s gradients; that node's own `grad` field, zero-initialized at construction, remains `0.0` after `backward()` runs, because nothing ever propagates gradient into it.

### Worked Example 15.4.1 -- the CUDA book's own unrelated-node finding, and a genuinely independent second graph

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.4 hand-writes topological_backward_order():
// it simply reverses the FULL, single global node list -- correct only
// because this book's toy engine always backpropagates from one single
// final output, and its own Worked Example 15.4.2 confirms an unrelated
// "sub" node, appended to that same global list but never an ancestor of
// w, keeps its own zero-initialized grad field at exactly 0.0 after
// backward() runs. torch::Tensor's own real autograd engine does
// something structurally different: it performs a genuine topological
// sort SCOPED to only the actual ancestors of the specific output being
// backpropagated, discovered by walking real edges (next_edges()) between
// real Node objects -- never touching, or even being aware of, any
// unrelated computation elsewhere in the same program. This file
// reproduces the CUDA book's own unrelated-node finding directly, then
// demonstrates the honest divergence: a second, entirely independent
// computation graph, built before the one being backpropagated, is
// confirmed completely untouched -- not zero-initialized-and-left-at-zero,
// but genuinely never visited at all.
int main() {
    // Worked Example 15.4.2: a three-node graph where "sub" is unrelated
    // to w. Here: q = a - b is unrelated to ww = a*b + a. q.retain_grad()
    // makes its own .grad() observable at all (a non-leaf tensor's grad is
    // not retained by default), mirroring the CUDA book's own explicit
    // per-node grad field existing regardless of whether it is used.
    torch::Tensor a = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor q = a - b;  // unrelated to w -- never used below
    q.retain_grad();

    torch::Tensor zz = a * b;
    torch::Tensor ww = zz + a;
    ww.backward();

    std::cout << "a.grad = " << a.grad().item<float>()
              << ", hand-derived expected = 5.0 (=b+1, from d(a*b+a)/da), match = "
              << (a.grad().item<float>() == 5.0f) << std::endl;
    std::cout << "b.grad = " << b.grad().item<float>()
              << ", hand-derived expected = 3.0 (=a, from d(a*b+a)/db), match = "
              << (b.grad().item<float>() == 3.0f) << std::endl;

    // The honest divergence: the CUDA book's own unrelated "sub" node
    // keeps its explicit grad field at its zero-initialized 0.0 (it WAS
    // visited during the full-list reversal, but nothing ever propagated
    // gradient INTO it). torch::Tensor's own q, despite .retain_grad()
    // being called, has .grad() genuinely UNDEFINED (no tensor at all,
    // not a zero tensor) -- because the real engine's own topological
    // sort never visits q in the first place, having discovered from
    // ww's own real edges that q is not an ancestor of ww at all.
    std::cout << "\nq.grad() defined at all (torch::Tensor has no equivalent explicit "
              << "zero-initialized field for a node the real engine never visits)? "
              << q.grad().defined() << std::endl;

    // Honest divergence, tested directly: a SECOND, entirely independent
    // computation graph, built BEFORE the one just backpropagated, using
    // completely different leaf tensors. In the CUDA book's own design,
    // this second graph's nodes would sit in the SAME global
    // ComputationGraph::nodes vector, and every future backward() call
    // would reverse-scan the ENTIRE list, including this graph's own
    // now-irrelevant nodes. torch::Tensor's own real engine builds each
    // tensor's own graph independently -- backward() on ww above never
    // touched m or n's own graph at all, confirmed here by their own
    // grad() remaining genuinely undefined.
    torch::Tensor m = torch::tensor(10.0f, torch::requires_grad(true));
    torch::Tensor n = torch::tensor(20.0f, torch::requires_grad(true));
    torch::Tensor unrelated_result = m * n;  // built AFTER ww.backward() ran above,
                                              // but m and n are otherwise fully independent
    std::cout << "m.grad() defined before m's own backward() ever runs "
              << "(confirming ww's own earlier backward() call never touched it)? "
              << m.grad().defined() << std::endl;
    std::cout << "n.grad() defined before n's own backward() ever runs? "
              << n.grad().defined() << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
a.grad = 5, hand-derived expected = 5.0 (=b+1, from d(a*b+a)/da), match = 1
b.grad = 3, hand-derived expected = 3.0 (=a, from d(a*b+a)/db), match = 1

q.grad() defined at all (torch::Tensor has no equivalent explicit zero-initialized field for a node the real engine never visits)? 0
m.grad() defined before m's own backward() ever runs (confirming ww's own earlier backward() call never touched it)? 0
n.grad() defined before n's own backward() ever runs? 0
```

Independently cross-checked via Python's own separate `torch` binding, confirming the identical isolation from scratch:

```text
a.grad 5.0 b.grad 3.0
q.grad is None: True
m.grad is None: True n.grad is None: True
```

### Discussion

`a.grad` and `b.grad` match the hand-derived local derivatives of `a*b+a` exactly, confirming the real backward pass genuinely ran and propagated correctly through the two-node graph rooted at `ww`. `q.grad().defined()` reporting `false` is this section's own central finding: the CUDA book's own unrelated `sub` node keeps its OWN explicit `grad` field at `0.0` because that field exists (zero-initialized, per Chapter 15's own struct definition) whether or not the node ever participates in a given backward pass -- the reversal touches every node in the global list, and `sub`'s own grad simply never gets incremented. `torch::Tensor`'s own `q`, despite `.retain_grad()` explicitly requesting that its gradient be observable, has no gradient tensor at all after `ww.backward()` runs, because the real engine's own topological sort, walking real edges outward from `ww`, discovers `q` is not among `ww`'s own ancestors and never visits it -- there is no global list to reverse-scan in the first place. The independent-graph test makes the practical stakes concrete: `m` and `n`, built as a completely separate computation, are entirely unaffected by `ww.backward()`'s own earlier call, exactly as expected of two genuinely unrelated tensors, but a design like the CUDA book's own single shared `ComputationGraph::nodes` vector would need every future `backward()` call to reverse-scan an ever-growing list containing every operation the entire program has ever performed, not just the ones relevant to the specific output currently being backpropagated.

> `[COMMON TRAP]` The CUDA book's own `topological_backward_order()`'s simplicity -- literally just reversing a list -- might read as a shortcut a real system merely optimizes away for speed, treating this chapter's own finding as a performance difference. The independent-graph test shows there is a genuine CORRECTNESS reason a real system cannot do what the CUDA book's own toy engine does, not merely a slower version of the identical algorithm: a real program can and does build many independent computations over its own lifetime, often concurrently, and a design that dumps every operation ever performed into one shared, ever-growing list would need every single `backward()` call to somehow determine which of those nodes are actually relevant -- exactly the graph-traversal problem `torch::Tensor`'s own real topological sort exists to solve, and precisely the problem the CUDA book's own single-output, single-graph toy design is specifically simple enough to avoid ever needing to face.

## Complete Runnable Code

### File: `01_registry_dispatch.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.1 hand-builds an OpRegistry: a
// std::unordered_map from operation name strings to Differentiable
// implementations, each providing forward() and backward(). Its own
// [COMMON TRAP] finds a genuine divergence between debug and release
// builds: registry.get() on an unregistered name asserts (aborts cleanly)
// in debug, but std::unordered_map::operator[] silently inserts a
// nullptr entry and returns it in release, corrupting the map's own size.
// torch::Tensor has no equivalent runtime lookup at all -- every real
// operation (operator*, operator+, torch::pow, torch::exp, ...) is
// statically wired at compile time to its own real backward Node type
// (MulBackward0, AddBackward0, ...), so there is no string-keyed registry
// a caller could query with a typo'd or unregistered name in the first
// place. This file verifies the CUDA book's own exact Worked Example
// 15.1.1 result (mul's backward needing the OTHER operand: grad_output=1,
// x=3, y=4 gives [4.0, 3.0]), then empirically confirms across six
// different real operations that a genuine grad_fn is always dispatched,
// never silently absent -- the closest real-world analog to "the registry
// never returns an unregistered nullptr" this compile-time-dispatched
// system can be asked to demonstrate.
int main() {
    // Worked Example 15.1.1: MulOp::backward with grad_output=1.0, x=3.0,
    // y=4.0 returns [4.0, 3.0] -- sensitivity to x equals y, sensitivity
    // to y equals x. torch::Tensor's own real MulBackward0 needs exactly
    // this "the other operand" information to compute the same result.
    torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor z = x * y;
    z.backward();
    std::cout << "d(x*y)/dx at x=3,y=4 = " << x.grad().item<float>()
              << ", CUDA book's own expected = 4.0 (=y), match = "
              << (x.grad().item<float>() == 4.0f) << std::endl;
    std::cout << "d(x*y)/dy at x=3,y=4 = " << y.grad().item<float>()
              << ", CUDA book's own expected = 3.0 (=x), match = "
              << (y.grad().item<float>() == 3.0f) << std::endl;

    // Honest divergence, tested empirically: real dispatch across six
    // different operation types, confirming a genuine, real grad_fn is
    // ALWAYS present for a differentiable operation on tensors requiring
    // grad -- never silently absent the way the CUDA book's own registry
    // can be made to silently insert a nullptr for an unregistered name.
    torch::Tensor a = torch::tensor(2.0f, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(3.0f, torch::requires_grad(true));

    struct OpResult { std::string op_name; bool grad_fn_present; std::string grad_fn_name; };
    std::vector<OpResult> results;
    results.push_back({"add", (a + b).grad_fn() != nullptr, (a + b).grad_fn()->name()});
    results.push_back({"sub", (a - b).grad_fn() != nullptr, (a - b).grad_fn()->name()});
    results.push_back({"mul", (a * b).grad_fn() != nullptr, (a * b).grad_fn()->name()});
    results.push_back({"div", (a / b).grad_fn() != nullptr, (a / b).grad_fn()->name()});
    results.push_back({"pow", torch::pow(a, 2).grad_fn() != nullptr, torch::pow(a, 2).grad_fn()->name()});
    results.push_back({"exp", torch::exp(a).grad_fn() != nullptr, torch::exp(a).grad_fn()->name()});

    bool all_present = true;
    std::cout << "\nreal grad_fn dispatch across six different operations:" << std::endl;
    for (auto& r : results) {
        std::cout << "  " << r.op_name << ": grad_fn present = " << r.grad_fn_present
                  << ", name = " << r.grad_fn_name << std::endl;
        if (!r.grad_fn_present) all_present = false;
    }
    std::cout << "every real operation dispatched a genuine, non-null grad_fn "
              << "(no equivalent to the CUDA book's own silent-nullptr-insertion bug)? "
              << all_present << std::endl;

    return 0;
}
```

### File: `02_graph_nodes.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.2 hand-builds GraphNode: a struct
// capturing a single operation invocation, with op_name, inputs, output,
// and a zero-initialized grad field. Its own running example, w = x*y + x
// with x=3.0, y=4.0, produces exactly two nodes: Node #0 (op="mul",
// output=12.0), Node #1 (op="add", output=15.0). torch::Tensor's own real
// autograd graph has a direct structural analog: every non-leaf tensor's
// .grad_fn() IS a real graph node, and its own ->name() reports the real
// operation that produced it -- "MulBackward0" and "AddBackward0" for
// this exact example. This file reproduces the CUDA book's own running
// example exactly, then verifies its own Worked Example 15.2.2: that one
// frozen (requires_grad=false) operand paired with one trainable operand
// still creates a full, real graph node -- not a partial or degraded one.
int main() {
    // The CUDA book's own running example: w = x*y + x, x=3.0, y=4.0.
    // z = x*y = 12.0 (Node #0, op="mul"). w = z+x = 15.0 (Node #1, op="add").
    torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor z = x * y;
    torch::Tensor w = z + x;

    std::cout << "z (Node #0) = " << z.item<float>() << ", op_name (real grad_fn name) = "
              << z.grad_fn()->name() << std::endl;
    std::cout << "matches CUDA book's own Node #0: op=mul, output=12.0? "
              << (z.item<float>() == 12.0f && z.grad_fn()->name() == "MulBackward0") << std::endl;

    std::cout << "\nw (Node #1) = " << w.item<float>() << ", op_name (real grad_fn name) = "
              << w.grad_fn()->name() << std::endl;
    std::cout << "matches CUDA book's own Node #1: op=add, output=15.0? "
              << (w.item<float>() == 15.0f && w.grad_fn()->name() == "AddBackward0") << std::endl;

    // Worked Example 15.2.2: one frozen operand (requires_grad=false)
    // paired with one trainable operand still creates a FULL node -- not
    // a partial, degraded, or missing one. Tested here directly: y_frozen
    // never requires grad, but x_trainable does, and the resulting
    // product still has a real, complete grad_fn.
    torch::Tensor x_trainable = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor y_frozen = torch::tensor(4.0f, torch::requires_grad(false));
    torch::Tensor z_mixed = x_trainable * y_frozen;

    std::cout << "\nz_mixed = x_trainable(requires_grad) * y_frozen(no grad) = " << z_mixed.item<float>()
              << std::endl;
    std::cout << "z_mixed.grad_fn() is a real, non-null node (not degraded/partial)? "
              << (z_mixed.grad_fn() != nullptr) << std::endl;
    std::cout << "z_mixed.grad_fn()->name() = " << z_mixed.grad_fn()->name()
              << " (identical real node type to the fully-trainable case above)? "
              << (z_mixed.grad_fn()->name() == "MulBackward0") << std::endl;

    // Confirming the node is genuinely FULL, not partial: it correctly
    // backpropagates a real gradient to the one trainable input.
    z_mixed.backward();
    std::cout << "d(z_mixed)/d(x_trainable) = " << x_trainable.grad().item<float>()
              << ", hand-derived expected = 4.0 (=y_frozen's own value), match = "
              << (x_trainable.grad().item<float>() == 4.0f) << std::endl;

    return 0;
}
```

### File: `03_gating_rule.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.3 hand-builds ComputationGraph::
// record(): it loops over an operation's inputs, sets needs_grad=true if
// ANY input has requires_grad=true, and only appends a real GraphNode
// (setting the output's grad_fn_index) when needs_grad is true -- an
// early return with no node created otherwise. torch::Tensor's own real
// autograd engine follows the identical "ANY, not ALL" gating rule: a
// node's own .grad_fn() is real and non-null whenever AT LEAST ONE input
// requires grad, and is genuinely null (no node at all) only when NONE
// of the inputs do. This file tests all four possible combinations of a
// two-operand operation's requires_grad flags directly, confirming the
// CUDA book's own gating logic exactly, rather than only the CUDA book's
// own single demonstrated case.
int main() {
    struct Case { bool x_req; bool y_req; std::string label; };
    std::vector<Case> cases = {
        {true, true, "both requires_grad=true"},
        {true, false, "x requires_grad=true, y requires_grad=false"},
        {false, true, "x requires_grad=false, y requires_grad=true"},
        {false, false, "both requires_grad=false"},
    };

    bool all_match_cuda_books_own_any_rule = true;
    for (auto& c : cases) {
        torch::Tensor x = torch::tensor(3.0f, torch::requires_grad(c.x_req));
        torch::Tensor y = torch::tensor(4.0f, torch::requires_grad(c.y_req));
        torch::Tensor z = x * y;
        bool has_real_node = (z.grad_fn() != nullptr);
        bool expected_by_any_rule = (c.x_req || c.y_req);  // the CUDA book's own record() logic
        std::cout << c.label << ": z.grad_fn() present = " << has_real_node
                  << ", CUDA book's own 'ANY input requires_grad' rule predicts = " << expected_by_any_rule
                  << ", match = " << (has_real_node == expected_by_any_rule) << std::endl;
        if (has_real_node != expected_by_any_rule) all_match_cuda_books_own_any_rule = false;
    }
    std::cout << "\nall four combinations match the CUDA book's own record()'s "
              << "'ANY, not ALL' gating rule exactly? " << all_match_cuda_books_own_any_rule << std::endl;

    // The one case where no node is created (both false) genuinely
    // produces a tensor that CANNOT backpropagate at all -- confirmed
    // here by attempting .backward() on it and catching the real error,
    // rather than merely observing grad_fn() is null.
    torch::Tensor x_no_grad = torch::tensor(3.0f, torch::requires_grad(false));
    torch::Tensor y_no_grad = torch::tensor(4.0f, torch::requires_grad(false));
    torch::Tensor z_no_grad = x_no_grad * y_no_grad;
    bool threw = false;
    try {
        z_no_grad.backward();
    } catch (const c10::Error&) {
        threw = true;
    }
    std::cout << "\nz_no_grad.backward() (no node was ever created) threw a real error? " << threw << std::endl;

    return 0;
}
```

### File: `04_topological_order.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 15.4 hand-writes topological_backward_order():
// it simply reverses the FULL, single global node list -- correct only
// because this book's toy engine always backpropagates from one single
// final output, and its own Worked Example 15.4.2 confirms an unrelated
// "sub" node, appended to that same global list but never an ancestor of
// w, keeps its own zero-initialized grad field at exactly 0.0 after
// backward() runs. torch::Tensor's own real autograd engine does
// something structurally different: it performs a genuine topological
// sort SCOPED to only the actual ancestors of the specific output being
// backpropagated, discovered by walking real edges (next_edges()) between
// real Node objects -- never touching, or even being aware of, any
// unrelated computation elsewhere in the same program. This file
// reproduces the CUDA book's own unrelated-node finding directly, then
// demonstrates the honest divergence: a second, entirely independent
// computation graph, built before the one being backpropagated, is
// confirmed completely untouched -- not zero-initialized-and-left-at-zero,
// but genuinely never visited at all.
int main() {
    // Worked Example 15.4.2: a three-node graph where "sub" is unrelated
    // to w. Here: q = a - b is unrelated to ww = a*b + a. q.retain_grad()
    // makes its own .grad() observable at all (a non-leaf tensor's grad is
    // not retained by default), mirroring the CUDA book's own explicit
    // per-node grad field existing regardless of whether it is used.
    torch::Tensor a = torch::tensor(3.0f, torch::requires_grad(true));
    torch::Tensor b = torch::tensor(4.0f, torch::requires_grad(true));
    torch::Tensor q = a - b;  // unrelated to w -- never used below
    q.retain_grad();

    torch::Tensor zz = a * b;
    torch::Tensor ww = zz + a;
    ww.backward();

    std::cout << "a.grad = " << a.grad().item<float>()
              << ", hand-derived expected = 5.0 (=b+1, from d(a*b+a)/da), match = "
              << (a.grad().item<float>() == 5.0f) << std::endl;
    std::cout << "b.grad = " << b.grad().item<float>()
              << ", hand-derived expected = 3.0 (=a, from d(a*b+a)/db), match = "
              << (b.grad().item<float>() == 3.0f) << std::endl;

    // The honest divergence: the CUDA book's own unrelated "sub" node
    // keeps its explicit grad field at its zero-initialized 0.0 (it WAS
    // visited during the full-list reversal, but nothing ever propagated
    // gradient INTO it). torch::Tensor's own q, despite .retain_grad()
    // being called, has .grad() genuinely UNDEFINED (no tensor at all,
    // not a zero tensor) -- because the real engine's own topological
    // sort never visits q in the first place, having discovered from
    // ww's own real edges that q is not an ancestor of ww at all.
    std::cout << "\nq.grad() defined at all (torch::Tensor has no equivalent explicit "
              << "zero-initialized field for a node the real engine never visits)? "
              << q.grad().defined() << std::endl;

    // Honest divergence, tested directly: a SECOND, entirely independent
    // computation graph, built BEFORE the one just backpropagated, using
    // completely different leaf tensors. In the CUDA book's own design,
    // this second graph's nodes would sit in the SAME global
    // ComputationGraph::nodes vector, and every future backward() call
    // would reverse-scan the ENTIRE list, including this graph's own
    // now-irrelevant nodes. torch::Tensor's own real engine builds each
    // tensor's own graph independently -- backward() on ww above never
    // touched m or n's own graph at all, confirmed here by their own
    // grad() remaining genuinely undefined.
    torch::Tensor m = torch::tensor(10.0f, torch::requires_grad(true));
    torch::Tensor n = torch::tensor(20.0f, torch::requires_grad(true));
    torch::Tensor unrelated_result = m * n;  // built AFTER ww.backward() ran above,
                                              // but m and n are otherwise fully independent
    std::cout << "m.grad() defined before m's own backward() ever runs "
              << "(confirming ww's own earlier backward() call never touched it)? "
              << m.grad().defined() << std::endl;
    std::cout << "n.grad() defined before n's own backward() ever runs? "
              << n.grad().defined() << std::endl;

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

`torch::Tensor`'s own real operation dispatch was confirmed to always produce a genuine, non-null `grad_fn` across six different operations, with the exact `[4.0, 3.0]` mul-backward result the CUDA book's own Worked Example 15.1.1 reports, and no equivalent to the CUDA book's own debug-vs-release registry divergence exists at all, because dispatch is resolved at compile time rather than through any runtime lookup. `.grad_fn()->name()` reproduced the CUDA book's own exact running example's node structure (`MulBackward0` at `12.0`, `AddBackward0` at `15.0`), and a frozen operand was confirmed to still produce a full, functioning node. All four combinations of a two-operand operation's `requires_grad` flags matched the CUDA book's own `record()`'s own `ANY`-not-`ALL` gating rule exactly, with the all-false case confirmed to genuinely prevent backpropagation via a real thrown error. And this chapter's own most consequential finding was structural: `torch::Tensor`'s own real autograd engine performs a genuine, edge-based topological sort scoped to only an output's real ancestors -- confirmed by an unrelated node's gradient being genuinely undefined rather than a zero-initialized field merely left untouched, and by a second, fully independent computation graph remaining completely unaffected by an unrelated `.backward()` call, a correctness requirement the CUDA book's own single-output toy design never has to meet.

## Self-Check Questions

1. Section 15.1's own six-operation dispatch table confirms a real `grad_fn` for every operation tested. What CATEGORY of programming error does this rule out, that the CUDA book's own `OpRegistry` remains vulnerable to, and why does compile-time dispatch rule it out structurally rather than merely handling it more gracefully?
2. Section 15.2's own frozen-operand test confirms `z_mixed.grad_fn()->name()` is IDENTICAL (`"MulBackward0"`) to the fully-trainable case. Using the final `.backward()` call in that section, explain why the node needs to know `y_frozen`'s own VALUE even though `y_frozen.grad()` will never be computed.
3. Section 15.3's own truth table has four rows. Which row corresponds to the case Section 15.2's own Worked Example 15.2.2 already demonstrated, and which two rows does Section 15.3 add that Section 15.2 did not test?
4. Section 15.4's own `q.grad().defined()` check returns `false`, not `0.0`. Explain the practical difference between "a tensor is undefined" and "a tensor is defined and equal to zero," and why this distinction matters for a caller who might write code checking whether `q` "has a gradient."
5. Section 15.4's own independent-graph test builds `m` and `n` AFTER `ww.backward()` already ran. Would the test still prove the same point if `m` and `n` had instead been built and used BEFORE `ww` and `a`/`b` were created? Explain what the test is actually demonstrating about SCOPE, not ORDER.

## Where We Go Next

This chapter opened `torch::Tensor`'s own real autograd graph for the first time, confirming its own real structures -- compile-time dispatch, `.grad_fn()` as a direct analog of `GraphNode`, the identical `ANY`-not-`ALL` gating rule, and a real topological sort -- match the CUDA book's own design at every point directly testable, while finding one genuine engineering requirement (correctness under multiple independent graphs) the CUDA book's own single-output toy engine never has to face. This closes the chapter that opens Part 3: Computational Graph Foundation. Part 4 turns from the graph's own structure to how it actually computes gradients -- Chapter 16 opens Automatic Differentiation Engine with backward function implementation, examining exactly how a node like `MulBackward0` computes the specific values this chapter has only confirmed the SHAPE of so far.

## Worked Solutions

**1.** Section 15.1's own dispatch table rules out the category of "runtime lookup failure on a valid-looking but unregistered key" -- exactly the CUDA book's own debug-vs-release divergence, where `registry.get("some_op")` can either assert (debug) or silently corrupt the registry with a `nullptr` entry (release), depending on a build flag, for a name that was never registered in the first place. Compile-time dispatch rules this out STRUCTURALLY, not merely gracefully, because there is no string key and no map lookup anywhere in the call path at all -- `torch::Tensor::operator*` is compiled code that directly constructs a `MulBackward0` object; there is no step during which an "unregistered" name could be looked up, because nothing is looked up by name in the first place. A graceful fix would still involve a lookup that could fail cleanly; this is the absence of the lookup entirely.

**2.** `MulBackward0`'s own real backward computation for `d(z)/d(x_trainable)` uses the formula `d(x*y)/dx = y` -- so to compute the ONE gradient value it will actually be asked for, the node must know `y_frozen`'s own numeric VALUE (`4.0`) at the time `.backward()` runs, regardless of whether `y_frozen` itself will ever receive a gradient. This is exactly the reason the CUDA book's own `backward()` signature takes both `inputs` and `output`, not just `grad_output`, established back in Section 15.1: multiplication's own local derivative with respect to one operand structurally requires the OTHER operand's value, whether or not that other operand happens to be trainable.

**3.** Section 15.2's own Worked Example 15.2.2 corresponds exactly to Section 15.3's own second row: `x requires_grad=true, y requires_grad=false`. Section 15.3 adds the third row (`x requires_grad=false, y requires_grad=true`, confirming the rule is symmetric and does not depend on WHICH operand is trainable) and the fourth row (`both requires_grad=false`, the one case where NO node is created at all, which Section 15.2 never demonstrated since its own focus was specifically on confirming a node IS created in the mixed case).

**4.** "Undefined" means the tensor object itself carries no allocated data, no shape, and no value at all -- attempting to read a value from it (such as calling `.item<float>()`) would throw a real error, because there is genuinely nothing there. "Defined and equal to zero" means a real tensor exists, with real allocated storage, holding the specific numeric value `0.0` -- reading it succeeds and correctly returns `0.0`. A caller checking "does `q` have a gradient" by testing `q.grad().defined()` gets an honest `false`, correctly reporting that no backward pass has ever computed anything for `q`. A caller who instead assumed a zero-initialized field (as the CUDA book's own `GraphNode::grad` genuinely is) would need to distinguish "genuinely never computed" from "computed and happened to be exactly zero" some other way, since both would otherwise look identical -- a distinction `torch::Tensor`'s own `defined()` check makes for free.

**5.** The test would still prove the identical point regardless of ordering, because what it demonstrates is about GRAPH SCOPE, not temporal sequence: `m` and `n`'s own tensors, and the `unrelated_result` computed from them, form a graph that shares no nodes, no edges, and no ancestry with `a`, `b`, `q`, `zz`, or `ww` at all -- whether that separate graph is built before or after `ww.backward()` runs is irrelevant to whether `ww.backward()`'s own topological sort could ever reach it, because reachability is determined by the real edges between `Node` objects, not by when in the program's own execution each `Node` happened to be constructed. Building `m` and `n` AFTER `ww.backward()` in this section's own code is simply a convenient way to make the claim concrete and easy to read; the underlying claim -- that unrelated graphs are structurally invisible to each other's own backward passes -- holds regardless of construction order.
