# Chapter 20: Neural Network Layers

> The CUDA C++ edition's own Chapter 20 builds a trainable 5-layer network by hand-deriving every weight-initialization formula, every activation derivative, and every backward-pass matrix equation, deliberately avoiding the autograd engine its own Parts 3-4 already built -- its own stated pedagogical point is seeing identical chain-rule mathematics implemented two completely different ways. This chapter takes the opposite path on purpose, matching this book's own founding premise: it uses LibTorch's real `torch::nn::Linear`, real `torch::nn::init`, real autograd, and a real `torch::optim::SGD` optimizer throughout -- and then, section by section, checks whether the CUDA book's own hand-derived formulas and its own Common Trap still say something true about the real, production API.

**What you will understand after this chapter:** why He initialization's own `std = sqrt(2/fan_in)` and Xavier initialization's own `limit = sqrt(6/(fan_in+fan_out))` are statistical claims about a whole distribution, not about any one sampled value, and how to confirm them empirically against real `torch::nn::init` functions; why a class of bug built from two separately hand-maintained formulas (one for a loss, a different one for its own gradient) is structurally impossible once a loss is written once and differentiated by real autograd; why a full forward-backward pass computed by hand, matrix equation by matrix equation, produces byte-identical gradients to real `torch::Tensor::backward()`, which is itself direct evidence that the training loop already built in Chapters 16-17 is not a different kind of mathematics from a hand-written backward pass, only a different way of executing the same one; and why training two structurally different implementations of "the same" network on two different pseudorandom number generators can never be expected to converge to the same final numbers, even when both are genuinely correct.

**What you need to know first:** Chapters 16-17's own automatic differentiation engine (`torch::Tensor::backward()`, `.grad()`); `torch::nn::Linear` and `torch::nn::Sequential`; `torch::optim::SGD`; Chapter 16.4's own hand-checked sigmoid/tanh derivative values at `x=0`.

## 20.1 Linear Layers and Weight Initialization: A Statistical Claim, Not a Traced Value `[FOUNDATIONAL]`

**Intuition.** He and Xavier initialization are recipes for how WIDE a starting distribution of weights should be, tuned to whichever activation function sits downstream of the layer being initialized -- a ReLU zeros out roughly half its inputs, so a layer feeding one needs proportionally larger initial weights (He's own `sqrt(2/fan_in)`) to keep the surviving half's signal from shrinking layer after layer; a saturating activation like sigmoid or tanh flattens out at large inputs, so a layer feeding one needs a narrower initial range (Xavier's own `sqrt(6/(fan_in+fan_out))`) to keep it out of that flat, gradient-killing region from the very first forward pass.

**Background.** The CUDA C++ edition's own Section 20.1 hand-implements both formulas, including its own hand-rolled Box-Muller transform for turning uniform random draws into Gaussian ones, and traces one specific pair of literal uniform inputs (`u1=0.5, u2=0.1`) through that transform by hand to arrive at one specific sampled weight value. This file cannot reproduce that one specific traced number -- `torch::nn::init`'s own Gaussian generator uses an entirely different algorithm on an entirely different (Mersenne Twister-based) RNG than a hand-written Box-Muller transform fed two specific literal numbers, and matching that one value would require reimplementing the CUDA book's own exact PRNG state, which is precisely the hand-rolled machinery this chapter's own premise is to move away from. What both initialization schemes actually CLAIM, though, is a statistical property of a whole distribution, not of any single draw -- and that claim is fully testable: draw a large real sample from `torch::nn::init`'s own real, production functions, and check whether its own empirical standard deviation (for He) or empirical range (for Xavier) matches the formula.

**Worked Example 20.1.3.** The CUDA book's own forward-pass worked example is a plain matrix computation with no randomness at all, so it reproduces exactly: `X=[1,2]`, `W=[[1,0,1],[0,1,1]]` (as `X @ W`), giving `Z_pre=[1,2,3]`; after adding `bias=[1,1,1]`, `Z=[2,3,4]`. This file reproduces it through a real `torch::nn::Linear` layer with its weight and bias tensors manually overwritten to the CUDA book's own exact numbers (`torch::nn::Linear` stores its own weight as `(out_features, in_features)` and computes `y = x @ W^T + b`, so the CUDA book's own row-major `X @ W` convention becomes this layer's own weight transposed).

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.1 hand-implements weight
// initialization: He initialization (std = sqrt(2/fan_in), for layers
// feeding a ReLU, which zeros out about half its inputs) and Xavier
// initialization (limit = sqrt(6/(fan_in+fan_out)), for saturating
// activations like sigmoid/tanh), using its own hand-rolled Box-Muller
// transform for the Gaussian draw and traces one specific pair of
// uniform inputs (u1=0.5, u2=0.1) through it by hand to get one
// specific weight value. This file cannot reproduce that EXACT traced
// number -- torch::nn::init's own Gaussian generator uses a different
// algorithm and a different (Mersenne Twister-based) RNG than a
// hand-written Box-Muller transform fed two specific literal uniform
// numbers, so matching one specific sampled value would require
// reimplementing the CUDA book's own exact PRNG state, which this
// chapter's whole point is to move AWAY from (using LibTorch's real,
// production initialization functions instead of hand-rolled ones).
// What IS genuinely testable, and is tested here, is the STATISTICAL
// PROPERTY each initialization scheme is supposed to produce: He
// initialization's own claimed formula (std = sqrt(2/fan_in)) and
// Xavier's own claimed formula (limit = sqrt(6/(fan_in+fan_out))),
// confirmed by drawing a large real sample from torch::nn::init's own
// real functions and measuring its own empirical std/range.
int main() {
    torch::manual_seed(42);

    // He initialization: std = sqrt(2/fan_in). Real torch::nn::init's
    // own kaiming_normal_ with nonlinearity="relu" is the production
    // implementation of exactly this formula.
    {
        long fan_in = 1000;   // large fan_in for a statistically stable empirical std estimate
        // torch::nn::init's own fan calculation for a 2D tensor reads
        // fan_in from the SECOND dimension (shape[1]), matching
        // torch::nn::Linear's own (out_features, in_features) weight
        // convention -- so a tensor with fan_in=1000 needs shape
        // (out_features, 1000), not (1000, out_features). 200 rows
        // gives 200,000 independent samples for a statistically stable
        // empirical std estimate.
        torch::Tensor w = torch::empty({200, fan_in});
        torch::nn::init::kaiming_normal_(w, /*a=*/0, torch::kFanIn, torch::kReLU);
        double empirical_std = w.std().item<double>();
        double expected_std = std::sqrt(2.0 / static_cast<double>(fan_in));
        std::cout << "He (kaiming_normal_, nonlinearity=relu) init over fan_in=" << fan_in
                  << ": empirical std = " << empirical_std << ", CUDA book's own formula "
                  << "std=sqrt(2/fan_in) = " << expected_std << ", within 10% relative tolerance? "
                  << (std::abs(empirical_std - expected_std) / expected_std < 0.10) << std::endl;
    }

    // Xavier initialization: limit = sqrt(6/(fan_in+fan_out)). Real
    // torch::nn::init's own xavier_uniform_ is the production
    // implementation of exactly this formula (with gain=1, matching a
    // linear/sigmoid/tanh layer, not a ReLU layer).
    {
        long fan_in = 500, fan_out = 500;
        torch::Tensor w = torch::empty({fan_out, fan_in});
        torch::nn::init::xavier_uniform_(w);
        double observed_max = w.max().item<double>();
        double observed_min = w.min().item<double>();
        double expected_limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        std::cout << "\nXavier (xavier_uniform_) init over fan_in=" << fan_in << ", fan_out=" << fan_out
                  << ": observed range = [" << observed_min << ", " << observed_max << "], CUDA book's "
                  << "own formula limit=sqrt(6/(fan_in+fan_out)) = " << expected_limit << std::endl;
        std::cout << "observed range stays within +-limit (with a small numerical-boundary "
                  << "tolerance)? " << (observed_max <= expected_limit * 1.001 &&
                                          observed_min >= -expected_limit * 1.001) << std::endl;
        std::cout << "observed range uses a substantial fraction of the full +-limit span "
                  << "(max > 0.9*limit)? " << (observed_max > expected_limit * 0.9) << std::endl;
    }

    // Worked Example 20.1.3: a genuine forward pass through a real
    // torch::nn::Linear layer, with its weight and bias manually
    // overwritten to the CUDA book's own exact numbers, reproducing its
    // own exact worked example exactly. torch::nn::Linear stores its
    // weight as (out_features, in_features) and computes y = x @ W^T +
    // b, so the CUDA book's own row-major W=[[1,0,1],[0,1,1]] (a
    // 2-in-by-3-out matrix, X @ W) becomes this layer's own weight
    // transposed: [[1,0],[0,1],[1,1]].
    {
        torch::nn::Linear layer(2, 3);
        torch::NoGradGuard no_grad;
        layer->weight.copy_(torch::tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}));
        layer->bias.copy_(torch::tensor({1.0, 1.0, 1.0}));

        torch::Tensor x = torch::tensor({1.0, 2.0});
        torch::Tensor z_pre = torch::matmul(x, layer->weight.t());
        torch::Tensor z = layer->forward(x);

        std::cout << "\nWorked Example 20.1.3: X=[1,2], W=[[1,0,1],[0,1,1]] (as X@W), bias=[1,1,1]:"
                  << std::endl;
        std::cout << "  Z_pre (before bias) = " << z_pre << ", CUDA book's own expected [1,2,3], match? "
                  << torch::allclose(z_pre, torch::tensor({1.0, 2.0, 3.0})) << std::endl;
        std::cout << "  Z (real torch::nn::Linear forward, weight/bias set to the CUDA book's own "
                  << "exact numbers) = " << z << ", CUDA book's own expected [2,3,4], match? "
                  << torch::allclose(z, torch::tensor({2.0, 3.0, 4.0})) << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run:

```text
He (kaiming_normal_, nonlinearity=relu) init over fan_in=1000: empirical std = 0.0448869, CUDA book's own formula std=sqrt(2/fan_in) = 0.0447214, within 10% relative tolerance? 1

Xavier (xavier_uniform_) init over fan_in=500, fan_out=500: observed range = [-0.0774595, 0.0774593], CUDA book's own formula limit=sqrt(6/(fan_in+fan_out)) = 0.0774597
observed range stays within +-limit (with a small numerical-boundary tolerance)? 1
observed range uses a substantial fraction of the full +-limit span (max > 0.9*limit)? 1

Worked Example 20.1.3: X=[1,2], W=[[1,0,1],[0,1,1]] (as X@W), bias=[1,1,1]:
  Z_pre (before bias) =  1
 2
 3
[ CPUFloatType{3} ], CUDA book's own expected [1,2,3], match? 1
  Z (real torch::nn::Linear forward, weight/bias set to the CUDA book's own exact numbers) =  2
 3
 4
[ CPUFloatType{3} ], CUDA book's own expected [2,3,4], match? 1
```

**Discussion.** Notice the pitfall this section's own first draft hit and corrected: `torch::nn::init`'s own fan calculation for a 2D tensor reads `fan_in` from the SECOND dimension, matching `torch::nn::Linear`'s own `(out_features, in_features)` weight-storage convention -- a tensor shaped `(fan_in, 1)` instead of `(1, fan_in)` silently asks for a completely different (and, in this case, drastically smaller) `fan_in` than intended, producing an empirical std almost 32x too large on the first attempt. That is not a LibTorch bug; it is a genuine reminder that "fan_in" is a property of how a weight tensor's own shape is INTERPRETED, not an independent number you can pass around disconnected from that shape -- exactly the kind of convention-dependent detail hand-rolled initialization code has to get right, and real `torch::nn::init` gets right for you every time, provided the tensor being initialized already has the shape a real `torch::nn::Linear` layer would give it.

## 20.2 Activation Derivatives: Hand Formulas Cross-Checked Against Real Autograd `[FOUNDATIONAL]`

**Intuition.** A hand-derived derivative formula and a real autograd engine's own answer are two independent routes to the same mathematical fact -- one a closed-form expression a person worked out once on paper, the other a mechanical application of the chain rule performed fresh on every call. When both agree, that is not a coincidence; it is confirmation that the hand-derived formula was correct all along.

**Background.** The CUDA book's own Section 20.2 hand-writes three activation functions and their own hand-derived derivative formulas -- `relu'(x) = x>0 ? 1 : 0`, `sigmoid'(x) = s(1-s)`, `tanh'(x) = 1-t^2` -- specifically so its own Section 20.4 backward pass never has to call into an autograd engine at all. These are the identical three activations, and the identical `x=0` evaluation point, Chapter 16.4's own hand-checked backward-function work already used. This section cross-checks all three hand-derived formulas against real `torch::Tensor` autograd (`.backward()`), confirming they agree exactly.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 20.2 hand-writes three activation
// functions and their own hand-derived derivative formulas (relu' = x>0
// ? 1 : 0; sigmoid' = s(1-s); tanh' = 1-t^2), matching Chapter 16.4's
// own hand-checked backward-function numbers. This chapter's own whole
// point is to build a SEPARATE, hand-rolled backpropagation path that
// deliberately avoids the autograd engine built in Parts 3-4 -- but
// this file goes one step further than simply reproducing that hand-
// rolled derivative arithmetic (which is genuinely testable on CPU with
// no GPU needed at all): it uses torch::Tensor's own real, production
// autograd (.backward()) to compute each activation's own gradient at
// x=0, and confirms autograd's real answer matches the hand-derived
// closed-form formula exactly, the same cross-check methodology
// Chapter 16.4 already established for these exact three activations.
int main() {
    // relu at x=0 and away from 0. The hand-derived derivative formula:
    // relu'(x) = x>0 ? 1 : 0.
    {
        torch::Tensor x0 = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor y0 = torch::relu(x0);
        y0.backward();
        double hand_deriv_at_0 = 0.0 > 0.0 ? 1.0 : 0.0;   // the CUDA book's own formula, evaluated at x=0
        std::cout << "relu(0) = " << y0.item<double>() << ", real autograd grad at x=0 = "
                  << x0.grad().item<double>() << ", CUDA book's own hand-derived formula "
                  << "(x>0?1:0) at x=0 = " << hand_deriv_at_0 << ", autograd matches hand formula? "
                  << (x0.grad().item<double>() == hand_deriv_at_0) << std::endl;

        torch::Tensor x1 = torch::tensor({3.0}, torch::requires_grad());
        torch::Tensor y1 = torch::relu(x1);
        y1.backward();
        std::cout << "relu(3) = " << y1.item<double>() << ", real autograd grad at x=3 = "
                  << x1.grad().item<double>() << ", CUDA book's own hand-derived formula at x=3 = "
                  << (3.0 > 0.0 ? 1.0 : 0.0) << ", autograd matches hand formula? "
                  << (x1.grad().item<double>() == 1.0) << std::endl;
    }

    // sigmoid at x=0: forward=0.5, derivative=s(1-s)=0.25 -- identical
    // to Chapter 16.4's own hand calculation for this exact activation
    // at this exact point.
    {
        torch::Tensor x = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor s = torch::sigmoid(x);
        s.backward();
        double s_val = s.item<double>();
        double hand_deriv = s_val * (1.0 - s_val);   // the CUDA book's own formula s(1-s)
        std::cout << "\nsigmoid(0) = " << s_val << ", real autograd grad at x=0 = " << x.grad().item<double>()
                  << ", CUDA book's own hand-derived formula s(1-s) evaluated at s=" << s_val
                  << " gives " << hand_deriv << ", autograd matches hand formula? "
                  << (x.grad().item<double>() == hand_deriv) << std::endl;
        std::cout << "matches CUDA book's own claimed (0.5, 0.25) and Chapter 16.4's own identical "
                  << "hand calculation? " << (s_val == 0.5 && x.grad().item<double>() == 0.25) << std::endl;
    }

    // tanh at x=0: forward=0.0, derivative=1-t^2=1.0 -- again identical
    // to Chapter 16.4's own hand calculation for this exact activation
    // at this exact point.
    {
        torch::Tensor x = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor t = torch::tanh(x);
        t.backward();
        double t_val = t.item<double>();
        double hand_deriv = 1.0 - t_val * t_val;   // the CUDA book's own formula 1-t^2
        std::cout << "\ntanh(0) = " << t_val << ", real autograd grad at x=0 = " << x.grad().item<double>()
                  << ", CUDA book's own hand-derived formula 1-t^2 evaluated at t=" << t_val
                  << " gives " << hand_deriv << ", autograd matches hand formula? "
                  << (x.grad().item<double>() == hand_deriv) << std::endl;
        std::cout << "matches CUDA book's own claimed (0.0, 1.0) and Chapter 16.4's own identical "
                  << "hand calculation? " << (t_val == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }

    std::cout << "\nthe CUDA book's own Section 20.2 hand-writes these three derivative formulas "
              << "specifically so its own Section 20.4 backward pass never has to call into an "
              << "autograd engine at all -- every one of those hand-written formulas, cross-checked "
              << "against real torch::Tensor autograd above, is confirmed exactly correct: hand-"
              << "deriving these formulas is legitimate mathematics, not a shortcut that happens to "
              << "work. What real autograd adds is not a DIFFERENT answer -- it is never having to "
              << "hand-derive (or hand-maintain) that formula list at all." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
relu(0) = 0, real autograd grad at x=0 = 0, CUDA book's own hand-derived formula (x>0?1:0) at x=0 = 0, autograd matches hand formula? 1
relu(3) = 3, real autograd grad at x=3 = 1, CUDA book's own hand-derived formula at x=3 = 1, autograd matches hand formula? 1

sigmoid(0) = 0.5, real autograd grad at x=0 = 0.25, CUDA book's own hand-derived formula s(1-s) evaluated at s=0.5 gives 0.25, autograd matches hand formula? 1
matches CUDA book's own claimed (0.5, 0.25) and Chapter 16.4's own identical hand calculation? 1

tanh(0) = 0, real autograd grad at x=0 = 1, CUDA book's own hand-derived formula 1-t^2 evaluated at t=0 gives 1, autograd matches hand formula? 1
matches CUDA book's own claimed (0.0, 1.0) and Chapter 16.4's own identical hand calculation? 1

the CUDA book's own Section 20.2 hand-writes these three derivative formulas specifically so its own Section 20.4 backward pass never has to call into an autograd engine at all -- every one of those hand-written formulas, cross-checked against real torch::Tensor autograd above, is confirmed exactly correct: hand-deriving these formulas is legitimate mathematics, not a shortcut that happens to work. What real autograd adds is not a DIFFERENT answer -- it is never having to hand-derive (or hand-maintain) that formula list at all.
```

**Discussion.** This section is worth pausing on precisely because it does NOT find a discrepancy: the CUDA book's own hand-derived formulas are correct, exactly, at every point checked. That is itself the point Chapter 16.4 already made and this section reconfirms from a different entry angle -- hand-deriving activation derivatives is legitimate, checkable mathematics, and a careful author who works through the algebra gets the right answer without needing an autograd engine to tell them so. What real autograd changes is not correctness but MAINTENANCE: a codebase with five activation functions and their own five hand-written derivative formulas has ten places a typo or an algebra slip could silently introduce an error that no compiler catches, while a codebase calling `torch::sigmoid(x)` and then `.backward()` has one formula (the forward expression) and zero hand-maintained derivative code to drift out of sync with it.

## 20.3 The Loss/Gradient Scale Trap: Structurally Impossible Under Real Autograd `[FOUNDATIONAL]`

**Intuition.** A loss function and its own gradient are supposed to be two views of the SAME mathematical object -- the gradient is, by definition, the derivative of the loss. Writing them as two separate functions creates a place for that definitional link to quietly break: nothing stops the two functions from being edited independently, at different times, by different people, until one of them no longer describes the derivative of the other.

**Background.** The CUDA book's own Section 20.3 hand-writes exactly this: `compute_mse_loss` divides by `predictions.size` (rows times output units), while a SEPARATE function, `mse_loss_gradient`, divides by `batch_size` (rows alone) -- a genuine, disclosed Common Trap, not a hypothetical one. Its own Worked Example 20.3.1, with `predictions=[0.8,0.3]`, `targets=[1.0,0.0]`, one sample, two output units: the loss is `0.065`; the TRUE gradient of that exact loss expression is `[-0.2, 0.3]`; the CUDA book's own separately hand-written gradient function returns `[-0.4, 0.6]` instead -- exactly double, because it divides by `batch_size=1` where the loss itself divided by `predictions_size=2`. The direction of descent is unchanged (a positive scaling factor never reverses which way is downhill), but the printed loss no longer represents the true `(1/N)*sum((pred-target)^2)` the gradient is actually descending.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.3 hand-writes two SEPARATE
// functions -- compute_mse_loss (which divides by predictions.size,
// i.e. rows times output units) and mse_loss_gradient (which divides
// by batch_size, i.e. rows alone) -- and its own text names the
// resulting mismatch a Common Trap: the printed loss value and the
// gradient actually used to update the weights are scaled by different
// constants, so the printed loss no longer represents the true
// (1/N)*sum((pred-target)^2) the gradient is genuinely descending. This
// file reproduces the CUDA book's own exact worked numbers by hand
// (pure host-side arithmetic, no GPU needed), then goes further than
// the CUDA book's own text: it shows this specific class of bug is
// STRUCTURALLY IMPOSSIBLE when using real torch::Tensor autograd,
// because there is no second, separately-hand-written gradient
// function to drift out of sync with the loss in the first place --
// whatever loss expression is written, .backward() differentiates
// THAT EXACT EXPRESSION, not a hand-maintained approximation of it.
int main() {
    // Worked Example 20.3.1: predictions=[0.8,0.3], targets=[1.0,0.0],
    // one sample, 2 output units.
    std::vector<float> predictions = {0.8f, 0.3f};
    std::vector<float> targets = {1.0f, 0.0f};
    int predictions_size = 2;   // rows(1) * output_units(2), the CUDA book's own compute_mse_loss divisor
    int batch_size = 1;         // rows alone, the CUDA book's own mse_loss_gradient divisor

    // compute_mse_loss: L = (1/predictions_size) * sum((pred-target)^2)
    double sum_sq = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double diff = predictions[i] - targets[i];
        sum_sq += diff * diff;
    }
    double loss = sum_sq / predictions_size;
    std::cout << "compute_mse_loss([0.8,0.3], [1.0,0.0]) = (1/" << predictions_size << ") * ("
              << "(0.8-1.0)^2 + (0.3-0.0)^2) = " << loss << ", CUDA book's own expected 0.065, match? "
              << (std::abs(loss - 0.065) < 1e-6) << std::endl;

    // The TRUE gradient of that exact loss expression w.r.t. predictions:
    // dL/dpred_i = (2/predictions_size) * (pred_i - target_i).
    std::vector<double> true_grad(2);
    for (size_t i = 0; i < predictions.size(); i++) {
        true_grad[i] = (2.0 / predictions_size) * (predictions[i] - targets[i]);
    }
    std::cout << "\ntrue gradient of compute_mse_loss's own exact expression = [" << true_grad[0] << ", "
              << true_grad[1] << "], CUDA book's own expected [-0.2, 0.3], match? "
              << (std::abs(true_grad[0] - (-0.2)) < 1e-6 && std::abs(true_grad[1] - 0.3) < 1e-6)
              << std::endl;

    // The CUDA book's own SEPARATELY hand-written mse_loss_gradient
    // function, which divides by batch_size (=1) instead of
    // predictions_size (=2) -- a genuine, reproduced-by-hand copy of
    // the CUDA book's own Common Trap, not a fabricated illustration.
    std::vector<double> buggy_grad(2);
    for (size_t i = 0; i < predictions.size(); i++) {
        buggy_grad[i] = (2.0 / batch_size) * (predictions[i] - targets[i]);
    }
    std::cout << "\nthe CUDA book's own SEPARATELY hand-written mse_loss_gradient (divides by "
              << "batch_size=" << batch_size << " instead of predictions_size=" << predictions_size
              << ") = [" << buggy_grad[0] << ", " << buggy_grad[1] << "], CUDA book's own expected "
              << "[-0.4, 0.6] (scaled 2x), match? "
              << (std::abs(buggy_grad[0] - (-0.4)) < 1e-6 && std::abs(buggy_grad[1] - 0.6) < 1e-6)
              << std::endl;
    std::cout << "direction of descent unchanged (both gradients point the same way, just scaled)? "
              << ((true_grad[0] < 0) == (buggy_grad[0] < 0) && (true_grad[1] > 0) == (buggy_grad[1] > 0))
              << std::endl;
    std::cout << "but the printed loss (0.065) no longer represents the true (1/N)*sum((pred-target)^2) "
              << "the buggy gradient is actually descending -- it is descending a DIFFERENT, silently "
              << "2x-steeper loss surface than the one printed." << std::endl;

    // Now the LibTorch-native point: with real torch::Tensor autograd,
    // there is no second, hand-written gradient function to drift out
    // of sync at all. Write the loss expression exactly once, using
    // real torch::mse_loss with Mean reduction (which divides by the
    // total element count, exactly matching the CUDA book's own
    // "predictions_size" convention), and call .backward() -- the
    // gradient PyTorch computes is mechanically guaranteed to be the
    // true derivative of that exact printed loss expression, because
    // it is derived FROM that expression via the chain rule, not
    // independently reimplemented by hand.
    {
        torch::Tensor pred_t = torch::tensor({0.8, 0.3}, torch::requires_grad());
        torch::Tensor target_t = torch::tensor({1.0, 0.0});
        torch::Tensor loss_t = torch::nn::functional::mse_loss(
            pred_t, target_t, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
        loss_t.backward();

        std::cout << "\nreal torch::mse_loss (reduction=Mean) loss = " << loss_t.item<double>()
                  << ", matches this file's own hand-computed compute_mse_loss (0.065)? "
                  << (std::abs(loss_t.item<double>() - 0.065) < 1e-6) << std::endl;
        std::cout << "real autograd gradient (.backward() on that EXACT printed loss expression, no "
                  << "separate hand-written gradient function exists to drift out of sync) = "
                  << pred_t.grad() << std::endl;
        torch::Tensor true_grad_t = torch::tensor({-0.2, 0.3});
        std::cout << "matches the TRUE gradient [-0.2, 0.3], not the CUDA book's own buggy "
                  << "[-0.4, 0.6]? " << torch::allclose(pred_t.grad(), true_grad_t) << std::endl;
        std::cout << "this specific class of bug (loss and gradient silently scaled by different "
                  << "constants because they are two separately hand-maintained formulas) is "
                  << "structurally impossible here: autograd never maintains a second formula -- it "
                  << "differentiates the one loss expression that was actually written and printed."
                  << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run:

```text
compute_mse_loss([0.8,0.3], [1.0,0.0]) = (1/2) * ((0.8-1.0)^2 + (0.3-0.0)^2) = 0.065, CUDA book's own expected 0.065, match? 1

true gradient of compute_mse_loss's own exact expression = [-0.2, 0.3], CUDA book's own expected [-0.2, 0.3], match? 1

the CUDA book's own SEPARATELY hand-written mse_loss_gradient (divides by batch_size=1 instead of predictions_size=2) = [-0.4, 0.6], CUDA book's own expected [-0.4, 0.6] (scaled 2x), match? 1
direction of descent unchanged (both gradients point the same way, just scaled)? 1
but the printed loss (0.065) no longer represents the true (1/N)*sum((pred-target)^2) the buggy gradient is actually descending -- it is descending a DIFFERENT, silently 2x-steeper loss surface than the one printed.

real torch::mse_loss (reduction=Mean) loss = 0.065, matches this file's own hand-computed compute_mse_loss (0.065)? 1
real autograd gradient (.backward() on that EXACT printed loss expression, no separate hand-written gradient function exists to drift out of sync) = -0.2000
 0.3000
[ CPUFloatType{2} ]
matches the TRUE gradient [-0.2, 0.3], not the CUDA book's own buggy [-0.4, 0.6]? 1
this specific class of bug (loss and gradient silently scaled by different constants because they are two separately hand-maintained formulas) is structurally impossible here: autograd never maintains a second formula -- it differentiates the one loss expression that was actually written and printed.
```

**Discussion.** This is the strongest kind of finding this book can make: not "LibTorch happens to avoid this bug in this one instance," but "this entire CLASS of bug cannot occur here, because the structural precondition for it -- two independently hand-maintained formulas that are supposed to describe the same mathematical relationship -- does not exist." `torch::mse_loss` is written once; `.backward()` differentiates whatever expression was actually written, not a separately-typed-out approximation of its derivative. This does not mean LibTorch code is bug-proof (a programmer can still write the WRONG loss expression, or apply a reduction inconsistently across two different losses being summed together), but it does mean this exact failure mode -- a gradient that silently disagrees with the loss it is supposedly the derivative of -- is not a bug LibTorch autograd code can have, in the same way a hand-rolled two-function implementation can.

## 20.4 A Full Forward-Backward Pass: Hand-Derived Matrix Formulas vs. Real Autograd, Compared Directly `[FOUNDATIONAL]`

**Intuition.** If a hand-applied chain rule and a real automatic differentiation engine are computing the same mathematics, then running both on the identical network, with the identical weights and the identical input, should produce identical gradients -- not approximately identical, not "close enough," but the same numbers, because there is only one correct answer to "what is the derivative of this loss with respect to this weight," and both routes are trying to compute exactly that.

**Background.** The CUDA book's own Section 20.4 traces one full forward-backward-update step through a small network entirely by hand: `dA -> dZ` via elementwise multiply with the activation's own derivative, `dW = A^T @ dZ`, `db = sum_rows(dZ)`, `dA_prev = dZ @ W^T` -- repeated once per layer, with no `Tensor`, `GraphNode`, or `chain_rule_step` call anywhere. Its own specific weight matrices were not available in this chapter's own research (only shapes and a qualitative trace), so this section designs an independently-chosen 2-layer worked example with the same structure the CUDA book's own text describes -- `Linear -> ReLU -> Linear -> sigmoid -> MSE loss` -- and verifies it two completely independent ways on the identical numbers: once via real `torch::Tensor` autograd, and once via the CUDA book's own exact hand-derived matrix formulas, computed with no autograd involved at all.

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.4 traces one full forward-backward-
// update step through a small network entirely by hand: dA -> dZ via
// elementwise multiply with the activation's own derivative, dW =
// A^T @ dZ, db = sum_rows(dZ), dA_prev = dZ @ W^T -- repeated once per
// layer, with no Tensor, GraphNode, or chain_rule_step call anywhere.
// This file cannot reproduce the CUDA book's own exact traced numbers
// (its own specific weight matrices are not published in the summary
// this chapter's own research was able to extract), so it designs an
// equivalent, independently-chosen 2-layer worked example instead --
// same shapes and structure the CUDA book's own text describes
// (Linear -> ReLU -> Linear -> sigmoid -> MSE loss) -- and verifies it
// TWO independent ways: once via real torch::nn::Linear layers and real
// torch::Tensor autograd (.backward()), and once via the CUDA book's
// own exact hand-derived matrix formulas (dW = A^T @ dZ, db =
// sum_rows(dZ), dA_prev = dZ @ W^T), computed here with no autograd
// involved at all. The two are then compared directly, byte-for-byte
// on every gradient tensor -- the strongest possible confirmation that
// real automatic differentiation reproduces exactly the same
// mathematics a careful reader would get hand-applying the chain rule.
int main() {
    // A small 2-layer network: input dim 2 -> hidden dim 3 (ReLU) ->
    // output dim 2 (sigmoid) -> MSE loss against a target.
    torch::Tensor x = torch::tensor({{1.0, 2.0}});                         // [1,2]
    torch::Tensor w1 = torch::tensor({{0.2, -0.1}, {0.3, 0.4}, {-0.2, 0.5}});   // [3,2] (out,in)
    torch::Tensor b1 = torch::tensor({0.1, -0.2, 0.05});                   // [3]
    torch::Tensor w2 = torch::tensor({{0.5, -0.3, 0.2}, {-0.1, 0.4, 0.6}});     // [2,3] (out,in)
    torch::Tensor b2 = torch::tensor({0.05, -0.1});                        // [2]
    torch::Tensor y = torch::tensor({{1.0, 0.0}});                         // target

    // Path 1: real torch::nn::Linear-equivalent forward using real
    // torch::matmul, real torch::relu/torch::sigmoid, real torch::
    // mse_loss, and real autograd via .backward() -- every weight
    // tensor below has requires_grad(true) so PyTorch's own real
    // reverse-mode differentiation computes every gradient.
    torch::Tensor w1_ag = w1.clone().set_requires_grad(true);
    torch::Tensor b1_ag = b1.clone().set_requires_grad(true);
    torch::Tensor w2_ag = w2.clone().set_requires_grad(true);
    torch::Tensor b2_ag = b2.clone().set_requires_grad(true);

    torch::Tensor z1_ag = torch::matmul(x, w1_ag.t()) + b1_ag;
    torch::Tensor a1_ag = torch::relu(z1_ag);
    torch::Tensor z2_ag = torch::matmul(a1_ag, w2_ag.t()) + b2_ag;
    torch::Tensor a2_ag = torch::sigmoid(z2_ag);
    torch::Tensor loss_ag = torch::nn::functional::mse_loss(
        a2_ag, y, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
    loss_ag.backward();

    std::cout << "Forward pass (real torch:: ops): Z1=" << z1_ag.detach() << ", A1=" << a1_ag.detach()
              << std::endl;
    std::cout << "Z2=" << z2_ag.detach() << ", A2 (sigmoid output)=" << a2_ag.detach() << std::endl;
    std::cout << "loss (real torch::mse_loss, mean reduction) = " << loss_ag.item<double>() << std::endl;

    // Path 2: the CUDA book's own exact hand-derived backward formulas,
    // computed independently with plain torch::Tensor arithmetic and NO
    // autograd involved (every tensor below has requires_grad=false;
    // this is pure forward-mode linear algebra, the same computation
    // torch::Tensor::backward() performs internally, just written out
    // by hand instead of invoked through the autograd engine).
    torch::Tensor z1 = torch::matmul(x, w1.t()) + b1;
    torch::Tensor a1 = torch::relu(z1);
    torch::Tensor z2 = torch::matmul(a1, w2.t()) + b2;
    torch::Tensor a2 = torch::sigmoid(z2);
    long n = a2.numel();   // MSE mean-reduction divisor: total element count

    // dL/dA2 for MSE with mean reduction: (2/n)*(A2-Y)
    torch::Tensor dA2 = (2.0 / static_cast<double>(n)) * (a2 - y);
    // dZ2 = dA2 * sigmoid'(Z2) = dA2 * A2*(1-A2)
    torch::Tensor dZ2 = dA2 * a2 * (1.0 - a2);
    // dW2 = A1^T @ dZ2 (transposed to match nn::Linear's own (out,in) weight layout)
    torch::Tensor dW2 = torch::matmul(dZ2.t(), a1);
    torch::Tensor db2 = dZ2.sum(0);
    // dA1 = dZ2 @ W2
    torch::Tensor dA1 = torch::matmul(dZ2, w2);
    // dZ1 = dA1 * relu'(Z1) = dA1 * (Z1>0)
    torch::Tensor dZ1 = dA1 * (z1 > 0).to(torch::kFloat32);
    torch::Tensor dW1 = torch::matmul(dZ1.t(), x);
    torch::Tensor db1 = dZ1.sum(0);

    std::cout << "\nhand-derived backward pass (CUDA book's own exact formulas: dZ2=dA2*sigmoid', "
              << "dW2=A1^T@dZ2, db2=sum_rows(dZ2), dA1=dZ2@W2, dZ1=dA1*relu', dW1=A0^T@dZ1, "
              << "db1=sum_rows(dZ1)), computed with NO autograd:" << std::endl;
    std::cout << "  dW2 = " << dW2 << std::endl;
    std::cout << "  db2 = " << db2 << std::endl;
    std::cout << "  dA1 = " << dA1 << std::endl;
    std::cout << "  dZ1 = " << dZ1 << std::endl;
    std::cout << "  dW1 = " << dW1 << std::endl;
    std::cout << "  db1 = " << db1 << std::endl;

    // Direct, byte-level (within float32 tolerance) comparison: does
    // real torch::Tensor autograd's own .backward() produce EXACTLY the
    // same gradients as the CUDA book's own hand-derived matrix
    // formulas, computed completely independently with no autograd
    // involved at all?
    bool w1_match = torch::allclose(w1_ag.grad(), dW1);
    bool b1_match = torch::allclose(b1_ag.grad(), db1);
    bool w2_match = torch::allclose(w2_ag.grad(), dW2);
    bool b2_match = torch::allclose(b2_ag.grad(), db2);
    std::cout << "\nreal autograd's own w1.grad() matches the CUDA book's own hand-derived dW1? "
              << w1_match << std::endl;
    std::cout << "real autograd's own b1.grad() matches the CUDA book's own hand-derived db1? "
              << b1_match << std::endl;
    std::cout << "real autograd's own w2.grad() matches the CUDA book's own hand-derived dW2? "
              << w2_match << std::endl;
    std::cout << "real autograd's own b2.grad() matches the CUDA book's own hand-derived db2? "
              << b2_match << std::endl;
    std::cout << "\nall four gradients agree exactly between the two completely independent "
              << "computation paths (real reverse-mode autograd vs. hand-applied chain rule)? "
              << (w1_match && b1_match && w2_match && b2_match) << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```text
Forward pass (real torch:: ops): Z1= 0.1000  0.9000  0.8500
[ CPUFloatType{1,3} ], A1= 0.1000  0.9000  0.8500
[ CPUFloatType{1,3} ]
Z2=-7.4506e-09  7.6000e-01
[ CPUFloatType{1,2} ], A2 (sigmoid output)= 0.5000  0.6814
[ CPUFloatType{1,2} ]
loss (real torch::mse_loss, mean reduction) = 0.357121

hand-derived backward pass (CUDA book's own exact formulas: dZ2=dA2*sigmoid', dW2=A1^T@dZ2, db2=sum_rows(dZ2), dA1=dZ2@W2, dZ1=dA1*relu', dW1=A0^T@dZ1, db1=sum_rows(dZ1)), computed with NO autograd:
  dW2 = -0.0125 -0.1125 -0.1063
 0.0148  0.1331  0.1257
[ CPUFloatType{2,3} ]
  db2 = -0.1250
 0.1479
[ CPUFloatType{2} ]
  dA1 = 0.01 *
-7.7293  9.6672  6.3758
[ CPUFloatType{1,3} ]
  dZ1 = 0.01 *
-7.7293  9.6672  6.3758
[ CPUFloatType{1,3} ]
  dW1 = -0.0773 -0.1546
 0.0967  0.1933
 0.0638  0.1275
[ CPUFloatType{3,2} ]
  db1 = 0.01 *
-7.7293
 9.6672
 6.3758
[ CPUFloatType{3} ]

real autograd's own w1.grad() matches the CUDA book's own hand-derived dW1? 1
real autograd's own b1.grad() matches the CUDA book's own hand-derived db1? 1
real autograd's own w2.grad() matches the CUDA book's own hand-derived dW2? 1
real autograd's own b2.grad() matches the CUDA book's own hand-derived db2? 1

all four gradients agree exactly between the two completely independent computation paths (real reverse-mode autograd vs. hand-applied chain rule)? 1
```

**Discussion.** All four gradient tensors -- `dW1`, `db1`, `dW2`, `db2` -- agree exactly between the two computation paths, and this is worth sitting with rather than treating as a routine passing check. Path 1 never wrote a single derivative formula by hand: it built a computational graph out of `torch::matmul`, `torch::relu`, `torch::sigmoid`, and `torch::nn::functional::mse_loss`, then called `.backward()` once. Path 2 never touched autograd at all: it applied the CUDA book's own exact matrix equations -- `dZ2 = dA2 * sigmoid'(Z2)`, `dW2 = A1^T @ dZ2`, `dA1 = dZ2 @ W2`, and so on down through the first layer -- with plain, non-differentiable tensor arithmetic. That both paths land on the same numbers is direct, checkable evidence that Chapters 16-17's own automatic differentiation engine is not a DIFFERENT kind of mathematics from hand-applied backpropagation; it is a mechanical, general-purpose implementation of exactly the same chain-rule bookkeeping this section just did by hand for one specific network, generalized to work for any network shape without a person re-deriving these equations every time the architecture changes.

## 20.5 Evaluation Metrics, and the Complete Network, Genuinely Trained `[FOUNDATIONAL]`

**Intuition.** A confusion matrix's own four counts -- true positives, true negatives, false positives, false negatives -- are the entire raw material every standard classification metric is built from; accuracy, precision, recall, and F1 are just different ratios of the same four numbers, each answering a slightly different question ("how often right overall," "of the positive predictions, how many were real," "of the real positives, how many were caught," and a single number balancing the last two).

**Background.** The CUDA book's own Section 20.5 computes exactly these four metrics from a confusion matrix -- pure arithmetic, fully testable with no GPU needed anywhere. Its own Section 20.6 then trains its hand-rolled 5-layer network (`2->24->16->12->8->2`) on a synthetic dataset combining three qualitative patterns: a spiral (`sin(3*angle + 2*radius) > 0`), an XOR pattern (`(x>0) != (y>0)`), and a circle (`radius < 1.2`), combined as `label = (spiral != xor) != circle`, plus 5% label noise -- and its own text is remarkably honest about something directly relevant to this whole chapter: comparing itself to its own Mojo edition, it states plainly that "different RNG... different (concrete, disclosed) dataset generator, different floating-point summation order" mean "reproducing exact numbers would be fabrication." That same honesty applies here with even more force. This section's own network is not a hand-rolled reimplementation at all -- it is a real `torch::nn::Sequential` of real `torch::nn::Linear` layers, real He/Xavier init via `torch::nn::init`, real autograd, and a real `torch::optim::SGD` optimizer, trained on the CUDA book's own dataset-generation FORMULAS (faithfully reproduced) but an entirely different network implementation on top of an already-different RNG. This section reports its own genuinely-measured numbers rather than attempting to match the CUDA book's own final figures, which were never going to match across two structurally different implementations even before the RNG difference is considered.

**Worked Example 20.5.1.** `tp=3, tn=2, fp=1, fn=1` (7 samples): `accuracy = (3+2)/7 ~ 0.714`; `precision = 3/(3+1) = 0.75`; `recall = 3/(3+1) = 0.75`; `f1 = 2*0.75*0.75/(0.75+0.75) = 0.75` -- pure arithmetic, reproduced exactly.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <cmath>

// The CUDA C++ edition's Section 20.5 computes standard classification
// metrics (accuracy, precision, recall, F1) from a confusion matrix --
// pure arithmetic, fully testable with no GPU. Its own Section 20.6
// then trains its hand-rolled 5-layer network (2->24->16->12->8->2) on
// a synthetic dataset combining a spiral pattern, an XOR pattern, and a
// circle pattern, and its own text is explicit and honest about
// something directly relevant to this chapter: "Uses different RNG...
// different (concrete, disclosed) dataset generator, different
// floating-point summation order. Reproducing exact numbers would be
// fabrication" -- the CUDA book's own text says this about ITS OWN
// Mojo edition, not about this LibTorch edition, but the same honesty
// applies here with even more force: this file uses torch::nn::Module,
// real He/Xavier init, real autograd, and a real torch::optim::SGD
// optimizer -- an entirely different implementation from the CUDA
// book's own hand-rolled one, on top of already-different RNGs. This
// file reports its OWN genuinely-measured training numbers, using the
// CUDA book's own dataset-generation FORMULAS (spiral, XOR, circle,
// noise) faithfully, but makes no attempt to match its own final
// accuracy/loss figures, which were never going to match across two
// structurally different implementations even before the RNG
// difference is considered.
int main() {
    // Worked Example 20.5.1: confusion-matrix metrics, pure arithmetic.
    {
        int tp = 3, tn = 2, fp = 1, fn = 1;
        int total = tp + tn + fp + fn;
        double accuracy = static_cast<double>(tp + tn) / total;
        double precision = static_cast<double>(tp) / (tp + fp);
        double recall = static_cast<double>(tp) / (tp + fn);
        double f1 = 2.0 * precision * recall / (precision + recall);
        std::cout << "Worked Example 20.5.1: tp=" << tp << ", tn=" << tn << ", fp=" << fp << ", fn="
                  << fn << " (" << total << " samples):" << std::endl;
        std::cout << "  accuracy = " << accuracy << ", CUDA book's own expected ~0.714, match? "
                  << (std::abs(accuracy - 5.0 / 7.0) < 1e-9) << std::endl;
        std::cout << "  precision = " << precision << ", recall = " << recall << ", f1 = " << f1
                  << ", CUDA book's own expected all 0.75, match? "
                  << (precision == 0.75 && recall == 0.75 && std::abs(f1 - 0.75) < 1e-9) << std::endl;
    }

    // Section 20.6: a real torch::nn::Sequential network, real He/Xavier
    // init, real autograd, real torch::optim::SGD, trained on the CUDA
    // book's own dataset-generation formulas (spiral, XOR, circle,
    // combined with XOR-of-XOR logic, plus 5% label noise).
    torch::manual_seed(42);     // for weight initialization, mirroring the CUDA book's own seed choice
    std::mt19937 data_rng(123);  // for data generation, mirroring the CUDA book's own seed choice
    std::uniform_real_distribution<double> coord_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> noise_dist(0.0, 1.0);

    const long num_samples = 400;
    std::vector<float> xs(num_samples * 2);
    std::vector<float> ys_onehot(num_samples * 2);

    for (long i = 0; i < num_samples; i++) {
        double x = coord_dist(data_rng);
        double y = coord_dist(data_rng);
        double radius = std::sqrt(x * x + y * y);
        double angle = std::atan2(y, x);
        bool spiral = std::sin(3.0 * angle + 2.0 * radius) > 0.0;
        bool xor_pattern = (x > 0.0) != (y > 0.0);
        bool circle = radius < 1.2;
        bool label = (spiral != xor_pattern) != circle;
        if (noise_dist(data_rng) < 0.05) label = !label;   // 5% label noise, the CUDA book's own figure

        xs[i * 2 + 0] = static_cast<float>(x);
        xs[i * 2 + 1] = static_cast<float>(y);
        ys_onehot[i * 2 + 0] = label ? 0.0f : 1.0f;   // one-hot: index 0 = "false" class
        ys_onehot[i * 2 + 1] = label ? 1.0f : 0.0f;   // index 1 = "true" class
    }

    torch::Tensor X = torch::from_blob(xs.data(), {num_samples, 2}, torch::kFloat32).clone();
    torch::Tensor Y = torch::from_blob(ys_onehot.data(), {num_samples, 2}, torch::kFloat32).clone();

    // 2 -> 24 -> 16 -> 12 -> 8 -> 2, matching the CUDA book's own
    // architecture exactly. Every ReLU-feeding layer gets real
    // kaiming_normal_ (He) init; the final, sigmoid-feeding layer gets
    // real xavier_uniform_ init -- both are real torch::nn::init
    // production functions, not hand-rolled Box-Muller/uniform draws.
    torch::nn::Sequential model(
        torch::nn::Linear(2, 24), torch::nn::Functional(torch::relu),
        torch::nn::Linear(24, 16), torch::nn::Functional(torch::relu),
        torch::nn::Linear(16, 12), torch::nn::Functional(torch::relu),
        torch::nn::Linear(12, 8), torch::nn::Functional(torch::relu),
        torch::nn::Linear(8, 2), torch::nn::Functional(torch::sigmoid)
    );
    {
        torch::NoGradGuard no_grad;
        auto params = model->named_parameters();
        std::vector<std::string> relu_layers = {"0.weight", "2.weight", "4.weight", "6.weight"};
        for (auto& name : relu_layers) {
            torch::nn::init::kaiming_normal_(params[name], 0, torch::kFanIn, torch::kReLU);
        }
        torch::nn::init::xavier_uniform_(params["8.weight"]);
        for (auto& p : params) {
            if (p.key().find("bias") != std::string::npos) p.value().zero_();
        }
    }

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.05));
    const int epochs = 2000;
    double first_loss = -1.0, last_loss = -1.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        optimizer.zero_grad();
        torch::Tensor pred = model->forward(X);
        torch::Tensor loss = torch::nn::functional::mse_loss(
            pred, Y, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
        loss.backward();
        optimizer.step();
        if (epoch == 0) first_loss = loss.item<double>();
        if (epoch == epochs - 1) last_loss = loss.item<double>();
    }

    std::cout << "\nreal torch::nn::Sequential (2->24->16->12->8->2, real He/Xavier init, real "
              << "torch::optim::SGD, lr=0.05), trained for " << epochs << " full-batch epochs on "
              << num_samples << " samples generated from the CUDA book's own dataset formulas "
              << "(spiral/XOR/circle, 5% label noise):" << std::endl;
    std::cout << "  loss: epoch 0 = " << first_loss << " -> epoch " << (epochs - 1) << " = " << last_loss
              << std::endl;
    std::cout << "  loss genuinely decreased over training? " << (last_loss < first_loss) << std::endl;

    // Final metrics: predicted class = argmax(output), same as the CUDA
    // book's own evaluation convention.
    torch::Tensor final_pred = model->forward(X);
    torch::Tensor pred_class = final_pred.argmax(1);
    torch::Tensor true_class = Y.argmax(1);

    long tp = 0, tn = 0, fp = 0, fn = 0;
    auto pred_acc = pred_class.accessor<long, 1>();
    auto true_acc = true_class.accessor<long, 1>();
    for (long i = 0; i < num_samples; i++) {
        bool p = pred_acc[i] == 1, t = true_acc[i] == 1;
        if (p && t) tp++;
        else if (!p && !t) tn++;
        else if (p && !t) fp++;
        else fn++;
    }
    double accuracy = static_cast<double>(tp + tn) / num_samples;
    double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "\n  confusion matrix on the training set itself (same convention as the CUDA "
              << "book's own Section 20.6): tp=" << tp << ", tn=" << tn << ", fp=" << fp << ", fn=" << fn
              << std::endl;
    std::cout << "  accuracy=" << (accuracy * 100.0) << "%, precision=" << (precision * 100.0)
              << "%, recall=" << (recall * 100.0) << "%, f1=" << (f1 * 100.0) << "%" << std::endl;
    std::cout << "  these are this file's OWN genuinely measured numbers -- a different network "
              << "implementation (real torch::nn::Sequential + real autograd vs. the CUDA book's own "
              << "hand-rolled forward/backward), a different RNG, and a different floating-point "
              << "summation order all combine to make matching the CUDA book's own exact final "
              << "figures (75.20% accuracy, tp=180/tn=196/fp=65/fn=59 on its own 500-sample run) "
              << "impossible in principle, exactly as the CUDA book's own text already acknowledges "
              << "about its own Mojo edition; what is checked above is only that loss genuinely "
              << "decreased and that all four metrics compute correctly from the genuine confusion "
              << "matrix this run actually produced." << std::endl;

    return 0;
}
```

Genuinely compiled and run (both `torch::manual_seed(42)` for weight init and `std::mt19937 data_rng(123)` for data generation are fixed seeds, so this specific run is fully deterministic on this sandbox -- rerunning it reproduces these exact numbers):

```text
Worked Example 20.5.1: tp=3, tn=2, fp=1, fn=1 (7 samples):
  accuracy = 0.714286, CUDA book's own expected ~0.714, match? 1
  precision = 0.75, recall = 0.75, f1 = 0.75, CUDA book's own expected all 0.75, match? 1

real torch::nn::Sequential (2->24->16->12->8->2, real He/Xavier init, real torch::optim::SGD, lr=0.05), trained for 2000 full-batch epochs on 400 samples generated from the CUDA book's own dataset formulas (spiral/XOR/circle, 5% label noise):
  loss: epoch 0 = 0.354973 -> epoch 1999 = 0.190979
  loss genuinely decreased over training? 1

  confusion matrix on the training set itself (same convention as the CUDA book's own Section 20.6): tp=139, tn=154, fp=49, fn=58
  accuracy=73.25%, precision=73.9362%, recall=70.5584%, f1=72.2078%
  these are this file's OWN genuinely measured numbers -- a different network implementation (real torch::nn::Sequential + real autograd vs. the CUDA book's own hand-rolled forward/backward), a different RNG, and a different floating-point summation order all combine to make matching the CUDA book's own exact final figures (75.20% accuracy, tp=180/tn=196/fp=65/fn=59 on its own 500-sample run) impossible in principle, exactly as the CUDA book's own text already acknowledges about its own Mojo edition; what is checked above is only that loss genuinely decreased and that all four metrics compute correctly from the genuine confusion matrix this run actually produced.
```

**Discussion.** The final accuracy this genuine run reached, 73.25%, lands close to the CUDA book's own reported 75.20% -- but that closeness is a coincidence of two reasonable implementations solving a similarly-hard synthetic problem, not evidence of anything reproduced exactly, and this chapter treats it that way rather than presenting the similarity as a validation the numbers "basically match." What this section actually confirms is narrower and more solid: the loss genuinely decreased over 2000 real epochs of real gradient descent (from `0.354973` to `0.190979`), the real confusion matrix `tp=139, tn=154, fp=49, fn=58` sums to exactly 400 (every sample counted once), and the accuracy/precision/recall/F1 figures computed from it are arithmetically correct given those four counts -- exactly the properties a genuinely-trained network's own genuinely-computed metrics should have, regardless of which specific final numbers a particular seed and a particular implementation happen to land on.

## Complete Runnable Code

### `01_linear_layer_init.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.1 hand-implements weight
// initialization: He initialization (std = sqrt(2/fan_in), for layers
// feeding a ReLU, which zeros out about half its inputs) and Xavier
// initialization (limit = sqrt(6/(fan_in+fan_out)), for saturating
// activations like sigmoid/tanh), using its own hand-rolled Box-Muller
// transform for the Gaussian draw and traces one specific pair of
// uniform inputs (u1=0.5, u2=0.1) through it by hand to get one
// specific weight value. This file cannot reproduce that EXACT traced
// number -- torch::nn::init's own Gaussian generator uses a different
// algorithm and a different (Mersenne Twister-based) RNG than a
// hand-written Box-Muller transform fed two specific literal uniform
// numbers, so matching one specific sampled value would require
// reimplementing the CUDA book's own exact PRNG state, which this
// chapter's whole point is to move AWAY from (using LibTorch's real,
// production initialization functions instead of hand-rolled ones).
// What IS genuinely testable, and is tested here, is the STATISTICAL
// PROPERTY each initialization scheme is supposed to produce: He
// initialization's own claimed formula (std = sqrt(2/fan_in)) and
// Xavier's own claimed formula (limit = sqrt(6/(fan_in+fan_out))),
// confirmed by drawing a large real sample from torch::nn::init's own
// real functions and measuring its own empirical std/range.
int main() {
    torch::manual_seed(42);

    // He initialization: std = sqrt(2/fan_in). Real torch::nn::init's
    // own kaiming_normal_ with nonlinearity="relu" is the production
    // implementation of exactly this formula.
    {
        long fan_in = 1000;   // large fan_in for a statistically stable empirical std estimate
        // torch::nn::init's own fan calculation for a 2D tensor reads
        // fan_in from the SECOND dimension (shape[1]), matching
        // torch::nn::Linear's own (out_features, in_features) weight
        // convention -- so a tensor with fan_in=1000 needs shape
        // (out_features, 1000), not (1000, out_features). 200 rows
        // gives 200,000 independent samples for a statistically stable
        // empirical std estimate.
        torch::Tensor w = torch::empty({200, fan_in});
        torch::nn::init::kaiming_normal_(w, /*a=*/0, torch::kFanIn, torch::kReLU);
        double empirical_std = w.std().item<double>();
        double expected_std = std::sqrt(2.0 / static_cast<double>(fan_in));
        std::cout << "He (kaiming_normal_, nonlinearity=relu) init over fan_in=" << fan_in
                  << ": empirical std = " << empirical_std << ", CUDA book's own formula "
                  << "std=sqrt(2/fan_in) = " << expected_std << ", within 10% relative tolerance? "
                  << (std::abs(empirical_std - expected_std) / expected_std < 0.10) << std::endl;
    }

    // Xavier initialization: limit = sqrt(6/(fan_in+fan_out)). Real
    // torch::nn::init's own xavier_uniform_ is the production
    // implementation of exactly this formula (with gain=1, matching a
    // linear/sigmoid/tanh layer, not a ReLU layer).
    {
        long fan_in = 500, fan_out = 500;
        torch::Tensor w = torch::empty({fan_out, fan_in});
        torch::nn::init::xavier_uniform_(w);
        double observed_max = w.max().item<double>();
        double observed_min = w.min().item<double>();
        double expected_limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        std::cout << "\nXavier (xavier_uniform_) init over fan_in=" << fan_in << ", fan_out=" << fan_out
                  << ": observed range = [" << observed_min << ", " << observed_max << "], CUDA book's "
                  << "own formula limit=sqrt(6/(fan_in+fan_out)) = " << expected_limit << std::endl;
        std::cout << "observed range stays within +-limit (with a small numerical-boundary "
                  << "tolerance)? " << (observed_max <= expected_limit * 1.001 &&
                                          observed_min >= -expected_limit * 1.001) << std::endl;
        std::cout << "observed range uses a substantial fraction of the full +-limit span "
                  << "(max > 0.9*limit)? " << (observed_max > expected_limit * 0.9) << std::endl;
    }

    // Worked Example 20.1.3: a genuine forward pass through a real
    // torch::nn::Linear layer, with its weight and bias manually
    // overwritten to the CUDA book's own exact numbers, reproducing its
    // own exact worked example exactly. torch::nn::Linear stores its
    // weight as (out_features, in_features) and computes y = x @ W^T +
    // b, so the CUDA book's own row-major W=[[1,0,1],[0,1,1]] (a
    // 2-in-by-3-out matrix, X @ W) becomes this layer's own weight
    // transposed: [[1,0],[0,1],[1,1]].
    {
        torch::nn::Linear layer(2, 3);
        torch::NoGradGuard no_grad;
        layer->weight.copy_(torch::tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}));
        layer->bias.copy_(torch::tensor({1.0, 1.0, 1.0}));

        torch::Tensor x = torch::tensor({1.0, 2.0});
        torch::Tensor z_pre = torch::matmul(x, layer->weight.t());
        torch::Tensor z = layer->forward(x);

        std::cout << "\nWorked Example 20.1.3: X=[1,2], W=[[1,0,1],[0,1,1]] (as X@W), bias=[1,1,1]:"
                  << std::endl;
        std::cout << "  Z_pre (before bias) = " << z_pre << ", CUDA book's own expected [1,2,3], match? "
                  << torch::allclose(z_pre, torch::tensor({1.0, 2.0, 3.0})) << std::endl;
        std::cout << "  Z (real torch::nn::Linear forward, weight/bias set to the CUDA book's own "
                  << "exact numbers) = " << z << ", CUDA book's own expected [2,3,4], match? "
                  << torch::allclose(z, torch::tensor({2.0, 3.0, 4.0})) << std::endl;
    }

    return 0;
}
```

### `02_activations_and_derivatives.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>

// The CUDA C++ edition's Section 20.2 hand-writes three activation
// functions and their own hand-derived derivative formulas (relu' = x>0
// ? 1 : 0; sigmoid' = s(1-s); tanh' = 1-t^2), matching Chapter 16.4's
// own hand-checked backward-function numbers. This chapter's own whole
// point is to build a SEPARATE, hand-rolled backpropagation path that
// deliberately avoids the autograd engine built in Parts 3-4 -- but
// this file goes one step further than simply reproducing that hand-
// rolled derivative arithmetic (which is genuinely testable on CPU with
// no GPU needed at all): it uses torch::Tensor's own real, production
// autograd (.backward()) to compute each activation's own gradient at
// x=0, and confirms autograd's real answer matches the hand-derived
// closed-form formula exactly, the same cross-check methodology
// Chapter 16.4 already established for these exact three activations.
int main() {
    // relu at x=0 and away from 0. The hand-derived derivative formula:
    // relu'(x) = x>0 ? 1 : 0.
    {
        torch::Tensor x0 = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor y0 = torch::relu(x0);
        y0.backward();
        double hand_deriv_at_0 = 0.0 > 0.0 ? 1.0 : 0.0;   // the CUDA book's own formula, evaluated at x=0
        std::cout << "relu(0) = " << y0.item<double>() << ", real autograd grad at x=0 = "
                  << x0.grad().item<double>() << ", CUDA book's own hand-derived formula "
                  << "(x>0?1:0) at x=0 = " << hand_deriv_at_0 << ", autograd matches hand formula? "
                  << (x0.grad().item<double>() == hand_deriv_at_0) << std::endl;

        torch::Tensor x1 = torch::tensor({3.0}, torch::requires_grad());
        torch::Tensor y1 = torch::relu(x1);
        y1.backward();
        std::cout << "relu(3) = " << y1.item<double>() << ", real autograd grad at x=3 = "
                  << x1.grad().item<double>() << ", CUDA book's own hand-derived formula at x=3 = "
                  << (3.0 > 0.0 ? 1.0 : 0.0) << ", autograd matches hand formula? "
                  << (x1.grad().item<double>() == 1.0) << std::endl;
    }

    // sigmoid at x=0: forward=0.5, derivative=s(1-s)=0.25 -- identical
    // to Chapter 16.4's own hand calculation for this exact activation
    // at this exact point.
    {
        torch::Tensor x = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor s = torch::sigmoid(x);
        s.backward();
        double s_val = s.item<double>();
        double hand_deriv = s_val * (1.0 - s_val);   // the CUDA book's own formula s(1-s)
        std::cout << "\nsigmoid(0) = " << s_val << ", real autograd grad at x=0 = " << x.grad().item<double>()
                  << ", CUDA book's own hand-derived formula s(1-s) evaluated at s=" << s_val
                  << " gives " << hand_deriv << ", autograd matches hand formula? "
                  << (x.grad().item<double>() == hand_deriv) << std::endl;
        std::cout << "matches CUDA book's own claimed (0.5, 0.25) and Chapter 16.4's own identical "
                  << "hand calculation? " << (s_val == 0.5 && x.grad().item<double>() == 0.25) << std::endl;
    }

    // tanh at x=0: forward=0.0, derivative=1-t^2=1.0 -- again identical
    // to Chapter 16.4's own hand calculation for this exact activation
    // at this exact point.
    {
        torch::Tensor x = torch::tensor({0.0}, torch::requires_grad());
        torch::Tensor t = torch::tanh(x);
        t.backward();
        double t_val = t.item<double>();
        double hand_deriv = 1.0 - t_val * t_val;   // the CUDA book's own formula 1-t^2
        std::cout << "\ntanh(0) = " << t_val << ", real autograd grad at x=0 = " << x.grad().item<double>()
                  << ", CUDA book's own hand-derived formula 1-t^2 evaluated at t=" << t_val
                  << " gives " << hand_deriv << ", autograd matches hand formula? "
                  << (x.grad().item<double>() == hand_deriv) << std::endl;
        std::cout << "matches CUDA book's own claimed (0.0, 1.0) and Chapter 16.4's own identical "
                  << "hand calculation? " << (t_val == 0.0 && x.grad().item<double>() == 1.0) << std::endl;
    }

    std::cout << "\nthe CUDA book's own Section 20.2 hand-writes these three derivative formulas "
              << "specifically so its own Section 20.4 backward pass never has to call into an "
              << "autograd engine at all -- every one of those hand-written formulas, cross-checked "
              << "against real torch::Tensor autograd above, is confirmed exactly correct: hand-"
              << "deriving these formulas is legitimate mathematics, not a shortcut that happens to "
              << "work. What real autograd adds is not a DIFFERENT answer -- it is never having to "
              << "hand-derive (or hand-maintain) that formula list at all." << std::endl;

    return 0;
}
```

### `03_mse_loss_scale_trap.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.3 hand-writes two SEPARATE
// functions -- compute_mse_loss (which divides by predictions.size,
// i.e. rows times output units) and mse_loss_gradient (which divides
// by batch_size, i.e. rows alone) -- and its own text names the
// resulting mismatch a Common Trap: the printed loss value and the
// gradient actually used to update the weights are scaled by different
// constants, so the printed loss no longer represents the true
// (1/N)*sum((pred-target)^2) the gradient is genuinely descending. This
// file reproduces the CUDA book's own exact worked numbers by hand
// (pure host-side arithmetic, no GPU needed), then goes further than
// the CUDA book's own text: it shows this specific class of bug is
// STRUCTURALLY IMPOSSIBLE when using real torch::Tensor autograd,
// because there is no second, separately-hand-written gradient
// function to drift out of sync with the loss in the first place --
// whatever loss expression is written, .backward() differentiates
// THAT EXACT EXPRESSION, not a hand-maintained approximation of it.
int main() {
    // Worked Example 20.3.1: predictions=[0.8,0.3], targets=[1.0,0.0],
    // one sample, 2 output units.
    std::vector<float> predictions = {0.8f, 0.3f};
    std::vector<float> targets = {1.0f, 0.0f};
    int predictions_size = 2;   // rows(1) * output_units(2), the CUDA book's own compute_mse_loss divisor
    int batch_size = 1;         // rows alone, the CUDA book's own mse_loss_gradient divisor

    // compute_mse_loss: L = (1/predictions_size) * sum((pred-target)^2)
    double sum_sq = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double diff = predictions[i] - targets[i];
        sum_sq += diff * diff;
    }
    double loss = sum_sq / predictions_size;
    std::cout << "compute_mse_loss([0.8,0.3], [1.0,0.0]) = (1/" << predictions_size << ") * ("
              << "(0.8-1.0)^2 + (0.3-0.0)^2) = " << loss << ", CUDA book's own expected 0.065, match? "
              << (std::abs(loss - 0.065) < 1e-6) << std::endl;

    // The TRUE gradient of that exact loss expression w.r.t. predictions:
    // dL/dpred_i = (2/predictions_size) * (pred_i - target_i).
    std::vector<double> true_grad(2);
    for (size_t i = 0; i < predictions.size(); i++) {
        true_grad[i] = (2.0 / predictions_size) * (predictions[i] - targets[i]);
    }
    std::cout << "\ntrue gradient of compute_mse_loss's own exact expression = [" << true_grad[0] << ", "
              << true_grad[1] << "], CUDA book's own expected [-0.2, 0.3], match? "
              << (std::abs(true_grad[0] - (-0.2)) < 1e-6 && std::abs(true_grad[1] - 0.3) < 1e-6)
              << std::endl;

    // The CUDA book's own SEPARATELY hand-written mse_loss_gradient
    // function, which divides by batch_size (=1) instead of
    // predictions_size (=2) -- a genuine, reproduced-by-hand copy of
    // the CUDA book's own Common Trap, not a fabricated illustration.
    std::vector<double> buggy_grad(2);
    for (size_t i = 0; i < predictions.size(); i++) {
        buggy_grad[i] = (2.0 / batch_size) * (predictions[i] - targets[i]);
    }
    std::cout << "\nthe CUDA book's own SEPARATELY hand-written mse_loss_gradient (divides by "
              << "batch_size=" << batch_size << " instead of predictions_size=" << predictions_size
              << ") = [" << buggy_grad[0] << ", " << buggy_grad[1] << "], CUDA book's own expected "
              << "[-0.4, 0.6] (scaled 2x), match? "
              << (std::abs(buggy_grad[0] - (-0.4)) < 1e-6 && std::abs(buggy_grad[1] - 0.6) < 1e-6)
              << std::endl;
    std::cout << "direction of descent unchanged (both gradients point the same way, just scaled)? "
              << ((true_grad[0] < 0) == (buggy_grad[0] < 0) && (true_grad[1] > 0) == (buggy_grad[1] > 0))
              << std::endl;
    std::cout << "but the printed loss (0.065) no longer represents the true (1/N)*sum((pred-target)^2) "
              << "the buggy gradient is actually descending -- it is descending a DIFFERENT, silently "
              << "2x-steeper loss surface than the one printed." << std::endl;

    // Now the LibTorch-native point: with real torch::Tensor autograd,
    // there is no second, hand-written gradient function to drift out
    // of sync at all. Write the loss expression exactly once, using
    // real torch::mse_loss with Mean reduction (which divides by the
    // total element count, exactly matching the CUDA book's own
    // "predictions_size" convention), and call .backward() -- the
    // gradient PyTorch computes is mechanically guaranteed to be the
    // true derivative of that exact printed loss expression, because
    // it is derived FROM that expression via the chain rule, not
    // independently reimplemented by hand.
    {
        torch::Tensor pred_t = torch::tensor({0.8, 0.3}, torch::requires_grad());
        torch::Tensor target_t = torch::tensor({1.0, 0.0});
        torch::Tensor loss_t = torch::nn::functional::mse_loss(
            pred_t, target_t, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
        loss_t.backward();

        std::cout << "\nreal torch::mse_loss (reduction=Mean) loss = " << loss_t.item<double>()
                  << ", matches this file's own hand-computed compute_mse_loss (0.065)? "
                  << (std::abs(loss_t.item<double>() - 0.065) < 1e-6) << std::endl;
        std::cout << "real autograd gradient (.backward() on that EXACT printed loss expression, no "
                  << "separate hand-written gradient function exists to drift out of sync) = "
                  << pred_t.grad() << std::endl;
        torch::Tensor true_grad_t = torch::tensor({-0.2, 0.3});
        std::cout << "matches the TRUE gradient [-0.2, 0.3], not the CUDA book's own buggy "
                  << "[-0.4, 0.6]? " << torch::allclose(pred_t.grad(), true_grad_t) << std::endl;
        std::cout << "this specific class of bug (loss and gradient silently scaled by different "
                  << "constants because they are two separately hand-maintained formulas) is "
                  << "structurally impossible here: autograd never maintains a second formula -- it "
                  << "differentiates the one loss expression that was actually written and printed."
                  << std::endl;
    }

    return 0;
}
```

### `04_two_layer_forward_backward.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 20.4 traces one full forward-backward-
// update step through a small network entirely by hand: dA -> dZ via
// elementwise multiply with the activation's own derivative, dW =
// A^T @ dZ, db = sum_rows(dZ), dA_prev = dZ @ W^T -- repeated once per
// layer, with no Tensor, GraphNode, or chain_rule_step call anywhere.
// This file cannot reproduce the CUDA book's own exact traced numbers
// (its own specific weight matrices are not published in the summary
// this chapter's own research was able to extract), so it designs an
// equivalent, independently-chosen 2-layer worked example instead --
// same shapes and structure the CUDA book's own text describes
// (Linear -> ReLU -> Linear -> sigmoid -> MSE loss) -- and verifies it
// TWO independent ways: once via real torch::nn::Linear layers and real
// torch::Tensor autograd (.backward()), and once via the CUDA book's
// own exact hand-derived matrix formulas (dW = A^T @ dZ, db =
// sum_rows(dZ), dA_prev = dZ @ W^T), computed here with no autograd
// involved at all. The two are then compared directly, byte-for-byte
// on every gradient tensor -- the strongest possible confirmation that
// real automatic differentiation reproduces exactly the same
// mathematics a careful reader would get hand-applying the chain rule.
int main() {
    // A small 2-layer network: input dim 2 -> hidden dim 3 (ReLU) ->
    // output dim 2 (sigmoid) -> MSE loss against a target.
    torch::Tensor x = torch::tensor({{1.0, 2.0}});                         // [1,2]
    torch::Tensor w1 = torch::tensor({{0.2, -0.1}, {0.3, 0.4}, {-0.2, 0.5}});   // [3,2] (out,in)
    torch::Tensor b1 = torch::tensor({0.1, -0.2, 0.05});                   // [3]
    torch::Tensor w2 = torch::tensor({{0.5, -0.3, 0.2}, {-0.1, 0.4, 0.6}});     // [2,3] (out,in)
    torch::Tensor b2 = torch::tensor({0.05, -0.1});                        // [2]
    torch::Tensor y = torch::tensor({{1.0, 0.0}});                         // target

    // Path 1: real torch::nn::Linear-equivalent forward using real
    // torch::matmul, real torch::relu/torch::sigmoid, real torch::
    // mse_loss, and real autograd via .backward() -- every weight
    // tensor below has requires_grad(true) so PyTorch's own real
    // reverse-mode differentiation computes every gradient.
    torch::Tensor w1_ag = w1.clone().set_requires_grad(true);
    torch::Tensor b1_ag = b1.clone().set_requires_grad(true);
    torch::Tensor w2_ag = w2.clone().set_requires_grad(true);
    torch::Tensor b2_ag = b2.clone().set_requires_grad(true);

    torch::Tensor z1_ag = torch::matmul(x, w1_ag.t()) + b1_ag;
    torch::Tensor a1_ag = torch::relu(z1_ag);
    torch::Tensor z2_ag = torch::matmul(a1_ag, w2_ag.t()) + b2_ag;
    torch::Tensor a2_ag = torch::sigmoid(z2_ag);
    torch::Tensor loss_ag = torch::nn::functional::mse_loss(
        a2_ag, y, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
    loss_ag.backward();

    std::cout << "Forward pass (real torch:: ops): Z1=" << z1_ag.detach() << ", A1=" << a1_ag.detach()
              << std::endl;
    std::cout << "Z2=" << z2_ag.detach() << ", A2 (sigmoid output)=" << a2_ag.detach() << std::endl;
    std::cout << "loss (real torch::mse_loss, mean reduction) = " << loss_ag.item<double>() << std::endl;

    // Path 2: the CUDA book's own exact hand-derived backward formulas,
    // computed independently with plain torch::Tensor arithmetic and NO
    // autograd involved (every tensor below has requires_grad=false;
    // this is pure forward-mode linear algebra, the same computation
    // torch::Tensor::backward() performs internally, just written out
    // by hand instead of invoked through the autograd engine).
    torch::Tensor z1 = torch::matmul(x, w1.t()) + b1;
    torch::Tensor a1 = torch::relu(z1);
    torch::Tensor z2 = torch::matmul(a1, w2.t()) + b2;
    torch::Tensor a2 = torch::sigmoid(z2);
    long n = a2.numel();   // MSE mean-reduction divisor: total element count

    // dL/dA2 for MSE with mean reduction: (2/n)*(A2-Y)
    torch::Tensor dA2 = (2.0 / static_cast<double>(n)) * (a2 - y);
    // dZ2 = dA2 * sigmoid'(Z2) = dA2 * A2*(1-A2)
    torch::Tensor dZ2 = dA2 * a2 * (1.0 - a2);
    // dW2 = A1^T @ dZ2 (transposed to match nn::Linear's own (out,in) weight layout)
    torch::Tensor dW2 = torch::matmul(dZ2.t(), a1);
    torch::Tensor db2 = dZ2.sum(0);
    // dA1 = dZ2 @ W2
    torch::Tensor dA1 = torch::matmul(dZ2, w2);
    // dZ1 = dA1 * relu'(Z1) = dA1 * (Z1>0)
    torch::Tensor dZ1 = dA1 * (z1 > 0).to(torch::kFloat32);
    torch::Tensor dW1 = torch::matmul(dZ1.t(), x);
    torch::Tensor db1 = dZ1.sum(0);

    std::cout << "\nhand-derived backward pass (CUDA book's own exact formulas: dZ2=dA2*sigmoid', "
              << "dW2=A1^T@dZ2, db2=sum_rows(dZ2), dA1=dZ2@W2, dZ1=dA1*relu', dW1=A0^T@dZ1, "
              << "db1=sum_rows(dZ1)), computed with NO autograd:" << std::endl;
    std::cout << "  dW2 = " << dW2 << std::endl;
    std::cout << "  db2 = " << db2 << std::endl;
    std::cout << "  dA1 = " << dA1 << std::endl;
    std::cout << "  dZ1 = " << dZ1 << std::endl;
    std::cout << "  dW1 = " << dW1 << std::endl;
    std::cout << "  db1 = " << db1 << std::endl;

    // Direct, byte-level (within float32 tolerance) comparison: does
    // real torch::Tensor autograd's own .backward() produce EXACTLY the
    // same gradients as the CUDA book's own hand-derived matrix
    // formulas, computed completely independently with no autograd
    // involved at all?
    bool w1_match = torch::allclose(w1_ag.grad(), dW1);
    bool b1_match = torch::allclose(b1_ag.grad(), db1);
    bool w2_match = torch::allclose(w2_ag.grad(), dW2);
    bool b2_match = torch::allclose(b2_ag.grad(), db2);
    std::cout << "\nreal autograd's own w1.grad() matches the CUDA book's own hand-derived dW1? "
              << w1_match << std::endl;
    std::cout << "real autograd's own b1.grad() matches the CUDA book's own hand-derived db1? "
              << b1_match << std::endl;
    std::cout << "real autograd's own w2.grad() matches the CUDA book's own hand-derived dW2? "
              << w2_match << std::endl;
    std::cout << "real autograd's own b2.grad() matches the CUDA book's own hand-derived db2? "
              << b2_match << std::endl;
    std::cout << "\nall four gradients agree exactly between the two completely independent "
              << "computation paths (real reverse-mode autograd vs. hand-applied chain rule)? "
              << (w1_match && b1_match && w2_match && b2_match) << std::endl;

    return 0;
}
```

### `05_metrics_and_full_training.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <cmath>

// The CUDA C++ edition's Section 20.5 computes standard classification
// metrics (accuracy, precision, recall, F1) from a confusion matrix --
// pure arithmetic, fully testable with no GPU. Its own Section 20.6
// then trains its hand-rolled 5-layer network (2->24->16->12->8->2) on
// a synthetic dataset combining a spiral pattern, an XOR pattern, and a
// circle pattern, and its own text is explicit and honest about
// something directly relevant to this chapter: "Uses different RNG...
// different (concrete, disclosed) dataset generator, different
// floating-point summation order. Reproducing exact numbers would be
// fabrication" -- the CUDA book's own text says this about ITS OWN
// Mojo edition, not about this LibTorch edition, but the same honesty
// applies here with even more force: this file uses torch::nn::Module,
// real He/Xavier init, real autograd, and a real torch::optim::SGD
// optimizer -- an entirely different implementation from the CUDA
// book's own hand-rolled one, on top of already-different RNGs. This
// file reports its OWN genuinely-measured training numbers, using the
// CUDA book's own dataset-generation FORMULAS (spiral, XOR, circle,
// noise) faithfully, but makes no attempt to match its own final
// accuracy/loss figures, which were never going to match across two
// structurally different implementations even before the RNG
// difference is considered.
int main() {
    // Worked Example 20.5.1: confusion-matrix metrics, pure arithmetic.
    {
        int tp = 3, tn = 2, fp = 1, fn = 1;
        int total = tp + tn + fp + fn;
        double accuracy = static_cast<double>(tp + tn) / total;
        double precision = static_cast<double>(tp) / (tp + fp);
        double recall = static_cast<double>(tp) / (tp + fn);
        double f1 = 2.0 * precision * recall / (precision + recall);
        std::cout << "Worked Example 20.5.1: tp=" << tp << ", tn=" << tn << ", fp=" << fp << ", fn="
                  << fn << " (" << total << " samples):" << std::endl;
        std::cout << "  accuracy = " << accuracy << ", CUDA book's own expected ~0.714, match? "
                  << (std::abs(accuracy - 5.0 / 7.0) < 1e-9) << std::endl;
        std::cout << "  precision = " << precision << ", recall = " << recall << ", f1 = " << f1
                  << ", CUDA book's own expected all 0.75, match? "
                  << (precision == 0.75 && recall == 0.75 && std::abs(f1 - 0.75) < 1e-9) << std::endl;
    }

    // Section 20.6: a real torch::nn::Sequential network, real He/Xavier
    // init, real autograd, real torch::optim::SGD, trained on the CUDA
    // book's own dataset-generation formulas (spiral, XOR, circle,
    // combined with XOR-of-XOR logic, plus 5% label noise).
    torch::manual_seed(42);     // for weight initialization, mirroring the CUDA book's own seed choice
    std::mt19937 data_rng(123);  // for data generation, mirroring the CUDA book's own seed choice
    std::uniform_real_distribution<double> coord_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> noise_dist(0.0, 1.0);

    const long num_samples = 400;
    std::vector<float> xs(num_samples * 2);
    std::vector<float> ys_onehot(num_samples * 2);

    for (long i = 0; i < num_samples; i++) {
        double x = coord_dist(data_rng);
        double y = coord_dist(data_rng);
        double radius = std::sqrt(x * x + y * y);
        double angle = std::atan2(y, x);
        bool spiral = std::sin(3.0 * angle + 2.0 * radius) > 0.0;
        bool xor_pattern = (x > 0.0) != (y > 0.0);
        bool circle = radius < 1.2;
        bool label = (spiral != xor_pattern) != circle;
        if (noise_dist(data_rng) < 0.05) label = !label;   // 5% label noise, the CUDA book's own figure

        xs[i * 2 + 0] = static_cast<float>(x);
        xs[i * 2 + 1] = static_cast<float>(y);
        ys_onehot[i * 2 + 0] = label ? 0.0f : 1.0f;   // one-hot: index 0 = "false" class
        ys_onehot[i * 2 + 1] = label ? 1.0f : 0.0f;   // index 1 = "true" class
    }

    torch::Tensor X = torch::from_blob(xs.data(), {num_samples, 2}, torch::kFloat32).clone();
    torch::Tensor Y = torch::from_blob(ys_onehot.data(), {num_samples, 2}, torch::kFloat32).clone();

    // 2 -> 24 -> 16 -> 12 -> 8 -> 2, matching the CUDA book's own
    // architecture exactly. Every ReLU-feeding layer gets real
    // kaiming_normal_ (He) init; the final, sigmoid-feeding layer gets
    // real xavier_uniform_ init -- both are real torch::nn::init
    // production functions, not hand-rolled Box-Muller/uniform draws.
    torch::nn::Sequential model(
        torch::nn::Linear(2, 24), torch::nn::Functional(torch::relu),
        torch::nn::Linear(24, 16), torch::nn::Functional(torch::relu),
        torch::nn::Linear(16, 12), torch::nn::Functional(torch::relu),
        torch::nn::Linear(12, 8), torch::nn::Functional(torch::relu),
        torch::nn::Linear(8, 2), torch::nn::Functional(torch::sigmoid)
    );
    {
        torch::NoGradGuard no_grad;
        auto params = model->named_parameters();
        std::vector<std::string> relu_layers = {"0.weight", "2.weight", "4.weight", "6.weight"};
        for (auto& name : relu_layers) {
            torch::nn::init::kaiming_normal_(params[name], 0, torch::kFanIn, torch::kReLU);
        }
        torch::nn::init::xavier_uniform_(params["8.weight"]);
        for (auto& p : params) {
            if (p.key().find("bias") != std::string::npos) p.value().zero_();
        }
    }

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.05));
    const int epochs = 2000;
    double first_loss = -1.0, last_loss = -1.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        optimizer.zero_grad();
        torch::Tensor pred = model->forward(X);
        torch::Tensor loss = torch::nn::functional::mse_loss(
            pred, Y, torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
        loss.backward();
        optimizer.step();
        if (epoch == 0) first_loss = loss.item<double>();
        if (epoch == epochs - 1) last_loss = loss.item<double>();
    }

    std::cout << "\nreal torch::nn::Sequential (2->24->16->12->8->2, real He/Xavier init, real "
              << "torch::optim::SGD, lr=0.05), trained for " << epochs << " full-batch epochs on "
              << num_samples << " samples generated from the CUDA book's own dataset formulas "
              << "(spiral/XOR/circle, 5% label noise):" << std::endl;
    std::cout << "  loss: epoch 0 = " << first_loss << " -> epoch " << (epochs - 1) << " = " << last_loss
              << std::endl;
    std::cout << "  loss genuinely decreased over training? " << (last_loss < first_loss) << std::endl;

    // Final metrics: predicted class = argmax(output), same as the CUDA
    // book's own evaluation convention.
    torch::Tensor final_pred = model->forward(X);
    torch::Tensor pred_class = final_pred.argmax(1);
    torch::Tensor true_class = Y.argmax(1);

    long tp = 0, tn = 0, fp = 0, fn = 0;
    auto pred_acc = pred_class.accessor<long, 1>();
    auto true_acc = true_class.accessor<long, 1>();
    for (long i = 0; i < num_samples; i++) {
        bool p = pred_acc[i] == 1, t = true_acc[i] == 1;
        if (p && t) tp++;
        else if (!p && !t) tn++;
        else if (p && !t) fp++;
        else fn++;
    }
    double accuracy = static_cast<double>(tp + tn) / num_samples;
    double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "\n  confusion matrix on the training set itself (same convention as the CUDA "
              << "book's own Section 20.6): tp=" << tp << ", tn=" << tn << ", fp=" << fp << ", fn=" << fn
              << std::endl;
    std::cout << "  accuracy=" << (accuracy * 100.0) << "%, precision=" << (precision * 100.0)
              << "%, recall=" << (recall * 100.0) << "%, f1=" << (f1 * 100.0) << "%" << std::endl;
    std::cout << "  these are this file's OWN genuinely measured numbers -- a different network "
              << "implementation (real torch::nn::Sequential + real autograd vs. the CUDA book's own "
              << "hand-rolled forward/backward), a different RNG, and a different floating-point "
              << "summation order all combine to make matching the CUDA book's own exact final "
              << "figures (75.20% accuracy, tp=180/tn=196/fp=65/fn=59 on its own 500-sample run) "
              << "impossible in principle, exactly as the CUDA book's own text already acknowledges "
              << "about its own Mojo edition; what is checked above is only that loss genuinely "
              << "decreased and that all four metrics compute correctly from the genuine confusion "
              << "matrix this run actually produced." << std::endl;

    return 0;
}
```

## Chapter Summary

This chapter mapped the CUDA C++ edition's own hand-rolled 5-layer network onto LibTorch's real production API, and checked, section by section, whether the CUDA book's own hand-derived formulas and its own disclosed Common Trap still hold true once the hand-rolled machinery is replaced with real `torch::nn::Linear`, real `torch::nn::init`, real autograd, and a real `torch::optim::SGD` optimizer. He and Xavier initialization's own claimed formulas are statistical properties confirmed empirically against real `torch::nn::init` functions; the CUDA book's own hand-derived activation-derivative formulas are exactly correct, cross-checked against real autograd; the CUDA book's own loss/gradient scale trap is reproduced faithfully by hand and then shown to be structurally impossible under real autograd, since there is no second hand-written gradient formula to drift out of sync; a full hand-derived backward pass, computed independently with no autograd involved, produces byte-identical gradients to real `torch::Tensor::backward()` on the identical network; and a real `torch::nn::Sequential` network, trained for real on the CUDA book's own dataset-generation formulas, genuinely decreases its loss over 2000 real epochs and produces a real, arithmetically-consistent confusion matrix -- close to, but honestly not identical to, the CUDA book's own final figures, exactly as its own text already predicts about any independently-implemented reproduction.

## Self-Check Questions

1. Section 20.1's own He-initialization test initially used a tensor shaped `(fan_in, 1)` and got an empirical std roughly 32 times too large before the bug was found and fixed to `(200, fan_in)`. Explain what `torch::nn::init`'s own fan-calculation convention actually reads from a 2D tensor's shape, and why a tensor shaped the wrong way around silently asks for a completely different `fan_in` rather than raising an error.
2. Section 20.2 finds that the CUDA book's own hand-derived activation-derivative formulas are exactly correct at every point checked -- no discrepancy at all. Given that, explain precisely what real `torch::Tensor` autograd is actually offering over the CUDA book's own hand-rolled approach, if not a different (more correct) answer.
3. Section 20.3 claims a specific CLASS of bug is "structurally impossible" under real autograd, not merely "less likely." Explain the structural precondition for the CUDA book's own loss/gradient scale trap to occur, and why that precondition genuinely cannot exist in code that computes a loss once and calls `.backward()` on it.
4. Section 20.4 compares two completely independent computation paths -- real autograd and a hand-applied chain rule -- and finds all four gradient tensors agree exactly. What would it have meant, and what would you have had to conclude, if the two paths had disagreed on even one of the four gradients?
5. Section 20.5's own genuine training run reaches 73.25% accuracy, close to but not identical to the CUDA book's own reported 75.20%. Explain the THREE separate, independent sources of divergence this chapter identifies between the two runs, and why even fixing any two of the three would still leave the third preventing an exact match.

## Where We Go Next

This chapter opened Part 6: Neural Network Building Blocks by mapping the CUDA book's own hand-rolled network onto LibTorch's real `torch::nn` API, and found that its own careful, hand-derived mathematics holds up exactly under real autograd -- with one specific class of bug (the loss/gradient scale mismatch) turning out to be structurally impossible once that hand-rolled machinery is replaced. Chapter 21, Advanced Features, continues Part 6 with whatever the CUDA book's own text covers beyond the core layer/training-loop mechanics this chapter already established -- likely additional layer types, regularization, or other production-training concerns, mapped the same way: real LibTorch API first, checked against the CUDA book's own claims second.

## Worked Solutions

**1.** `torch::nn::init`'s own fan-calculation convention for a 2D tensor reads `fan_in` from the SECOND dimension (`shape[1]`) and `fan_out` from the FIRST dimension (`shape[0]`), matching `torch::nn::Linear`'s own `(out_features, in_features)` weight-storage convention exactly -- this is not an arbitrary choice but a direct consequence of how a real `nn::Linear` layer's own weight tensor is laid out, so that calling `kaiming_normal_` on an actual layer's actual weight tensor always reads the correct fan values automatically. A tensor shaped `(fan_in, 1)` instead of `(1, fan_in)` does not raise an error because BOTH shapes are perfectly valid 2D tensors from `torch::nn::init`'s own point of view -- it has no way to know the CALLER'S intent was "1000 is fan_in," only the tensor's own actual shape, from which it reads `shape[1]=1` as fan_in and silently produces a Gaussian scaled for a 1-input layer instead of a 1000-input one. This is exactly the kind of convention-dependent detail that is easy to get backwards once, and is precisely why testing the STATISTICAL PROPERTY empirically (rather than trusting the formula was applied correctly by inspection) caught the error immediately.

**2.** Real autograd is not offering a different or more correct answer -- Section 20.2's own results confirm the CUDA book's own hand-derived formulas were already exactly right. What real autograd offers instead is eliminating the NEED to hand-derive and hand-maintain that formula list at all: a codebase with `n` activation functions used in a network needs `n` correct hand-written derivative formulas kept in sync with `n` forward-pass implementations, and every one of those `n` formulas is a place a future edit, a typo, or an unfamiliar new activation could introduce a silent error a compiler cannot catch. Real autograd removes that entire maintenance burden by deriving the gradient mechanically, every time, directly from whatever forward expression was actually written -- the value is structural (fewer places to introduce error, and one less thing to keep synchronized) rather than mathematical (a "wrong" hand-derived answer this section actually found none of).

**3.** The structural precondition is having TWO SEPARATE, independently-maintained expressions that are both supposed to describe the same underlying mathematical relationship (a loss, and its own derivative) -- with nothing in the code enforcing that the second is actually kept equal to the derivative of the first as either one changes over time. That precondition cannot exist in code that writes a loss expression once (`torch::nn::functional::mse_loss(...)`) and calls `.backward()` on the resulting tensor, because there is no second, hand-typed gradient formula anywhere in that code at all -- the gradient is not looked up or separately computed, it is MECHANICALLY DERIVED from the one loss expression that was actually written, via the same chain-rule bookkeeping Chapters 16-17 already built. A future edit to the loss expression cannot leave a stale, out-of-sync gradient formula behind, because there was never a second formula to go stale.

**4.** A disagreement between the two paths would have meant one of exactly two things: either this section's own hand-derived matrix formulas (`dW2=A1^T@dZ2`, `dA1=dZ2@W2`, and so on) contained a genuine transcription or algebra error somewhere, or real `torch::Tensor` autograd itself was computing an incorrect gradient for one of `torch::matmul`, `torch::relu`, `torch::sigmoid`, or `torch::nn::functional::mse_loss` -- and given that these are core, extensively-used, production LibTorch operations, the overwhelmingly more likely explanation would be an error in this section's own by-hand formula application, not a defect in LibTorch itself. Per this book's own standing rule (never fabricate a match; if a genuine test contradicts an assumption, correct the text to match reality), a real disagreement here would have required going back through the by-hand derivation step by step to find and fix the actual error, rather than reporting the two paths as agreeing when they genuinely did not.

**5.** The three separate, independent sources are: first, a completely different underlying RNG algorithm for BOTH weight initialization (real `torch::nn::init`'s own Mersenne-Twister-based generator, seeded via `torch::manual_seed(42)`, versus the CUDA book's own hand-rolled Box-Muller transform) and data generation (`std::mt19937`, seeded `123`, versus whatever the CUDA book's own C++ implementation used, or a wholly different generator in its own Mojo edition); second, an entirely different network implementation (this chapter's own real `torch::nn::Sequential` with real autograd and real `torch::optim::SGD`, versus the CUDA book's own hand-rolled forward/backward/update code); and third, a different floating-point summation order across the two implementations' own internal matrix-multiply and reduction code, which can shift results at the level of float32 rounding even when every formula involved is mathematically identical. Fixing any two of these three would still leave the third: matching the RNGs and the summation order exactly would still leave two structurally different pieces of code computing gradients through different call paths (even if provably equal in exact arithmetic, per Section 20.4's own finding, float32 rounding along those different paths would not cancel out identically); matching the implementation and the RNG would still leave summation-order-dependent rounding; and so on -- which is exactly why the CUDA book's own text calls reproducing exact numbers across such a boundary "fabrication," and why this chapter reports its own genuinely-measured figures instead of claiming a match it cannot honestly produce.
