# Chapter 22: Quantitative Finance Examples

> "A model that merely prices an instrument is a calculator. A model that differentiates through pricing it is a risk desk -- the difference is whether every sensitivity a trader needs comes free with the price, or has to be derived by hand, once, for every new instrument." The CUDA C++ edition's own Chapter 22 opens Part 7 by applying everything its own book has built so far -- GPU-parallel pricing, numerical root-finding, portfolio-level aggregation, Monte Carlo simulation, and autodiff -- to a genuinely new domain: quantitative finance. This chapter opens this book's own Part 7 the same way, applying everything Parts 1 through 6 have built -- real tensors, real autograd, a real custom `torch::autograd::Function`, real `torch::randn` -- to the identical four topics, mapped through LibTorch's real production API rather than hand-rolled CUDA kernels.

**What you will understand after this chapter:** why pricing a PORTFOLIO of bonds with real `torch::Tensor` arithmetic is not a kernel-launch-configuration decision at all, but ordinary vectorized arithmetic over a 1D tensor that is already laid out exactly the way a hand-rolled struct-of-arrays kernel would want; why a coupon bond's own credit spread has no closed-form solution and must be found by bisection, and why an iteration cap on that bisection is a silent failure mode a caller has to defend against explicitly; why a portfolio's own duration is a PV-weighted average, and why checking that portfolio weights sum to exactly 1.0 is the cheapest correctness check available for an entire pricing pipeline; and why Monte Carlo Greeks obtained via a single `backward()` pass through a real simulation are not merely faster than bump-and-reprice, but structurally free of an entire category of noise that bump-and-reprice has to fight off explicitly with common random numbers.

**What you need to know first:** Chapter 21.1's own custom `torch::autograd::Function` and the implicit function theorem for differentiating through a solver that cannot itself be traced; Chapters 16-17's `.backward()` and `torch::autograd::grad()`; `torch::randn` and `torch::manual_seed` (Chapter 20's own real, production Gaussian sampling); ordinary vectorized `torch::Tensor` arithmetic (`torch::exp`, `torch::relu`, `.sum()`, `.mean()`).

## 22.1 Bond Pricing with Automatic Differentiation: Struct-of-Arrays, For Free `[FOUNDATIONAL]`

**Intuition.** A zero-coupon bond is the simplest possible IOU: pay less today for a promise to receive a fixed, larger amount on a fixed future date, with nothing in between. The gap between what is paid and what is promised is rent, paid partly for the pure inconvenience of waiting (the risk-free rate) and partly as compensation for the chance the promise does not get kept (credit spread). Pricing one bond is one exponential; pricing a portfolio of a thousand of them is a thousand INDEPENDENT exponentials, since every bond's own price depends on nothing but its own four numbers -- exactly the embarrassingly-parallel shape a vectorized computation wants.

**Background.** The CUDA C++ edition's own Section 22.1 hand-lays-out a struct-of-arrays -- eight separate, contiguous arrays, one per field -- specifically so that a GPU kernel's own per-thread reads land on coalesced memory, and launches one thread per bond to compute each bond's own price in parallel. A real `torch::Tensor` is already exactly this: a contiguous buffer, one per field, with no separate "struct-of-arrays decision" to make at all -- pricing an entire portfolio is one ordinary vectorized expression (`face_value * torch::exp(-yield * time_to_maturity)`), with the embarrassingly-parallel execution handled entirely underneath by LibTorch's own tensor kernels, on CPU here and, with a CUDA build, on GPU without changing a line of this code. DV01 (the dollar change in a bond's own value per one-basis-point move in yield) is obtained here three genuinely independent ways: an analytic closed-form derivative, a real `backward()` pass through the bond's own pricing formula, and an independent central finite difference.

**Worked Example 22.1.1 and 22.1.2.** A deterministic, formula-generated 1024-bond portfolio (face cycling `[1000, 5000, 10000]`; maturity, risk-free rate, and credit spread each a pure function of the bond's own index, no randomness anywhere) -- deterministic generation means this file's own numbers are expected to match the CUDA book's own exact figures, unlike a trained network's own final numbers, since there is no RNG divergence possible here at all. Bond 2's own DV01: `t=0.75`, `PV=$9,814.2469`, so `DV01 = t * PV * 0.0001 ~= $0.736069` per basis point (the dollar LOSS in this bond's own value per one-basis-point INCREASE in yield, since a bond's price falls as its own discount rate rises -- this file computes `DV01 = -(dPV/dyield) * 0.0001`, and since `dPV/dyield = -t*PV` for this pricing formula, the two negative signs combine into a positive number, the standard convention for reporting DV01 as a loss magnitude).

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 22.1 prices a portfolio of zero-coupon
// bonds by hand-laying out 8 separate contiguous arrays (a struct-of-
// arrays layout, chosen specifically for coalesced GPU memory access)
// and launching one thread per bond -- an embarrassingly parallel
// kernel, since each bond's price depends on nothing but its own four
// numbers. A real torch::Tensor already IS a contiguous, struct-of-
// arrays-shaped buffer -- pricing "a portfolio of bonds" with real
// LibTorch tensor ops is not a kernel-launch decision at all, it is
// just ordinary vectorized arithmetic over a 1D tensor, with the
// embarrassingly-parallel execution handled entirely underneath by
// LibTorch's own CPU (or, with a CUDA build, GPU) tensor kernels. This
// file reproduces the CUDA book's own exact worked numbers (the bond
// portfolio is generated by a deterministic formula, not randomness,
// so genuinely matching digits is expected here, unlike a trained
// network's own final numbers) and then reproduces its own reduction
// bug with a real, LibTorch-native equivalent: not an out-of-bounds
// GPU memory read (real torch::Tensor indexing is bounds-checked), but
// a syntactically valid, silently-too-narrow SLICE -- a portfolio
// summed over only its first 128 of 1024 bonds, with nothing about
// that slice raising any error at all.
int main() {
    std::cout << std::fixed;

    const int N = 1024;
    const int64_t faces_cycle[3] = {1000, 5000, 10000};

    // Deterministic portfolio generation, matching the CUDA book's own
    // exact formulas: face cycles [1000,5000,10000]; t=0.25+(i%120)*0.25;
    // risk_free=0.02+(i%31)*0.001; spread=0.001+(i%30)*0.001. No
    // randomness anywhere -- every one of these 1024 bonds' own inputs
    // is a pure function of its own index i.
    torch::Tensor face_value = torch::empty({N}, torch::kFloat64);
    torch::Tensor time_to_maturity = torch::empty({N}, torch::kFloat64);
    torch::Tensor risk_free_rate = torch::empty({N}, torch::kFloat64);
    torch::Tensor credit_spread = torch::empty({N}, torch::kFloat64);
    {
        auto fv_acc = face_value.accessor<double, 1>();
        auto t_acc = time_to_maturity.accessor<double, 1>();
        auto rf_acc = risk_free_rate.accessor<double, 1>();
        auto sp_acc = credit_spread.accessor<double, 1>();
        for (int i = 0; i < N; i++) {
            fv_acc[i] = static_cast<double>(faces_cycle[i % 3]);
            t_acc[i] = 0.25 + (i % 120) * 0.25;
            rf_acc[i] = 0.02 + (i % 31) * 0.001;
            sp_acc[i] = 0.001 + (i % 30) * 0.001;
        }
    }

    // The whole portfolio's own prices, computed in ONE vectorized
    // expression -- this single line IS the "one thread per bond"
    // kernel the CUDA book hand-writes, just expressed as real tensor
    // arithmetic instead of an explicit kernel launch.
    torch::Tensor yield = risk_free_rate + credit_spread;
    torch::Tensor present_value = face_value * torch::exp(-yield * time_to_maturity);
    torch::Tensor duration = time_to_maturity;   // exact for a zero-coupon bond

    // Worked Example 22.1.1: the first three bonds' own exact numbers.
    auto pv_acc = present_value.accessor<double, 1>();
    auto y_acc = yield.accessor<double, 1>();
    std::cout << "Worked Example 22.1.1 -- first three bonds of the deterministic 1024-bond portfolio:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  bond " << i << ": face=" << face_value[i].item<double>()
                  << ", t=" << time_to_maturity[i].item<double>()
                  << ", risk_free=" << risk_free_rate[i].item<double>()
                  << ", spread=" << credit_spread[i].item<double>()
                  << ", yield=" << y_acc[i]
                  << ", PV=" << std::setprecision(4) << pv_acc[i] << std::setprecision(6) << std::endl;
    }

    // Worked Example 22.1.2: DV01 for bond index 2, three independent ways.
    {
        double t2 = time_to_maturity[2].item<double>();
        double pv2 = present_value[2].item<double>();
        // DV01 = -(dPV/dyield) * 0.0001. For PV = FV*exp(-yield*t),
        // dPV/dyield = -t*PV (a bond's own price FALLS as yield rises),
        // so DV01 = -(-t*PV)*0.0001 = +t*PV*0.0001 -- a POSITIVE number,
        // by the standard convention that DV01 reports the dollar LOSS
        // in a bond's own value per one-basis-point INCREASE in yield.
        double analytic_dv01 = t2 * pv2 * 0.0001;

        // Real autograd: a fresh scalar computation for bond 2 alone,
        // with its own risk_free_rate as the differentiable leaf --
        // d(PV)/d(risk_free_rate) equals d(PV)/d(yield) exactly, since
        // yield = risk_free_rate + credit_spread and the spread term's
        // own derivative with respect to risk_free_rate is zero.
        torch::Tensor face2 = torch::tensor(face_value[2].item<double>(), torch::kFloat64);
        torch::Tensor t2_t = torch::tensor(t2, torch::kFloat64);
        torch::Tensor sp2_t = torch::tensor(credit_spread[2].item<double>(), torch::kFloat64);
        torch::Tensor rf2_t = torch::tensor(risk_free_rate[2].item<double>(),
                                             torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
        torch::Tensor pv2_t = face2 * torch::exp(-(rf2_t + sp2_t) * t2_t);
        pv2_t.backward();
        double dpv_dyield_autograd = rf2_t.grad().item<double>();
        double autograd_dv01 = -dpv_dyield_autograd * 0.0001;

        // Independent central finite-difference cross-check, double
        // precision, on the SAME bond-2 pricing formula.
        double eps = 1e-6;
        double rf2 = risk_free_rate[2].item<double>();
        double sp2 = credit_spread[2].item<double>();
        double face2d = face_value[2].item<double>();
        auto price_at = [&](double rf) { return face2d * std::exp(-(rf + sp2) * t2); };
        double fd_dpv_dyield = (price_at(rf2 + eps) - price_at(rf2 - eps)) / (2.0 * eps);
        double fd_dv01 = -fd_dpv_dyield * 0.0001;

        std::cout << "\nWorked Example 22.1.2 -- DV01 for bond 2 (t=" << t2 << ", PV=" << pv2 << "):" << std::endl;
        std::cout << "  analytic DV01 (t*PV*0.0001, i.e. -(dPV/dyield)*0.0001 since dPV/dyield=-t*PV) = "
                  << analytic_dv01 << " per basis point" << std::endl;
        std::cout << "  real autograd DV01 (backward() through this bond's own pricing formula) = "
                  << autograd_dv01 << " per basis point, matches analytic? "
                  << (std::abs(autograd_dv01 - analytic_dv01) < 1e-9) << std::endl;
        std::cout << "  central finite-difference DV01 (eps=1e-6, independent third method) = "
                  << fd_dv01 << " per basis point, matches analytic within 1e-6? "
                  << (std::abs(fd_dv01 - analytic_dv01) < 1e-6) << std::endl;
    }

    // Full portfolio: total value, PV-weighted average yield, PV-
    // weighted average duration -- each a single vectorized reduction
    // over the real 1024-element tensor.
    double total_correct = present_value.sum().item<double>();
    double weighted_yield = (present_value * yield).sum().item<double>() / total_correct;
    double weighted_duration = (present_value * duration).sum().item<double>() / total_correct;
    std::cout << "\nfull 1024-bond portfolio (correct, real torch::Tensor::sum() over all 1024 bonds):" << std::endl;
    std::cout << "  total portfolio value = " << std::setprecision(2) << total_correct << std::setprecision(6) << std::endl;
    std::cout << "  PV-weighted average yield = " << (weighted_yield * 100.0) << "%" << std::endl;
    std::cout << "  PV-weighted average duration = " << weighted_duration << " years" << std::endl;

    // [COMMON TRAP], the real LibTorch equivalent of the CUDA book's own
    // reduction bug: not an out-of-bounds memory read (torch::Tensor
    // indexing is bounds-checked on CPU), but a syntactically valid
    // slice that is silently too narrow -- summing only the first 128
    // of 1024 bonds, exactly as if a batch size of 128 from an earlier,
    // smaller test run were never updated when the portfolio grew to
    // its real size of 1024. No exception, no warning: torch::Tensor
    // slicing is a completely ordinary, legal operation, whether the
    // slice happens to be the WHOLE portfolio or only a fraction of it.
    double total_buggy = present_value.narrow(0, 0, 128).sum().item<double>();
    double ratio = total_buggy / total_correct;
    std::cout << "\n[COMMON TRAP] present_value.narrow(0, 0, 128).sum() -- summing only the first 128 of "
              << "1024 bonds (a stale batch-size constant from an earlier, smaller test run, never updated "
              << "when the real portfolio grew to 1024 bonds): buggy total = " << std::setprecision(2)
              << total_buggy << std::setprecision(6) << ", correct total = " << std::setprecision(2)
              << total_correct << std::setprecision(6) << " -- the buggy total is only "
              << (ratio * 100.0) << "% of the real portfolio's own value, with no error, no exception, "
              << "and no warning anywhere: .narrow(0, 0, 128) is a completely ordinary, valid tensor "
              << "operation, whether or not 128 happens to be the WHOLE portfolio." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 01_bond_pricing_soa.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 01_bond_pricing_soa
./01_bond_pricing_soa
```

```text
Worked Example 22.1.1 -- first three bonds of the deterministic 1024-bond portfolio:
  bond 0: face=1000.000000, t=0.250000, risk_free=0.020000, spread=0.001000, yield=0.021000, PV=994.7638
  bond 1: face=5000.000000, t=0.500000, risk_free=0.021000, spread=0.002000, yield=0.023000, PV=4942.8294
  bond 2: face=10000.000000, t=0.750000, risk_free=0.022000, spread=0.003000, yield=0.025000, PV=9814.2469

Worked Example 22.1.2 -- DV01 for bond 2 (t=0.750000, PV=9814.246877):
  analytic DV01 (t*PV*0.0001, i.e. -(dPV/dyield)*0.0001 since dPV/dyield=-t*PV) = 0.736069 per basis point
  real autograd DV01 (backward() through this bond's own pricing formula) = 0.736069 per basis point, matches analytic? 1
  central finite-difference DV01 (eps=1e-6, independent third method) = 0.736069 per basis point, matches analytic within 1e-6? 1

full 1024-bond portfolio (correct, real torch::Tensor::sum() over all 1024 bonds):
  total portfolio value = 2831177.17
  PV-weighted average yield = 4.815494%
  PV-weighted average duration = 11.189073 years

[COMMON TRAP] present_value.narrow(0, 0, 128).sum() -- summing only the first 128 of 1024 bonds (a stale batch-size constant from an earlier, smaller test run, never updated when the real portfolio grew to 1024 bonds): buggy total = 364130.53, correct total = 2831177.17 -- the buggy total is only 12.861453% of the real portfolio's own value, with no error, no exception, and no warning anywhere: .narrow(0, 0, 128) is a completely ordinary, valid tensor operation, whether or not 128 happens to be the WHOLE portfolio.
```

**Discussion.** All three of bond 2's own DV01 estimates -- analytic, real autograd, and an independent central finite difference -- agree exactly, which is expected: this is ordinary, well-behaved calculus with no numerical instability anywhere near it, unlike Section 21.4's own float32 catastrophic-cancellation demonstration. The `[COMMON TRAP]` is this file's own honest LibTorch-native equivalent of the CUDA book's own reduction bug: raw CUDA's own bug reads only 128 of 1024 bonds because a kernel is launched with too few threads, silently reading past-the-intended-range GPU memory with no bounds check to catch it; real `torch::Tensor` indexing IS bounds-checked on CPU, so that EXACT failure mode cannot occur here -- but a syntactically identical, equally silent failure mode can: `.narrow(0, 0, 128)` is a completely ordinary, valid tensor operation whether or not 128 happens to be the whole portfolio, and nothing about calling it on a 1024-bond tensor raises any error, warning, or exception at all. The buggy total this file's own bug produces ($364,130.53) and the correct total ($2,831,177.17) are the SAME two numbers the CUDA book's own bug produces, because this is the identical deterministic portfolio and the identical slice size -- not a coincidence, a direct consequence of both books generating the same formula-defined bonds.

## 22.2 Credit Spread Solving: The Iteration Cap as a Silent Escape Hatch `[FOUNDATIONAL]`

**Intuition.** A coupon-paying bond's own credit spread has no closed-form inverse, because the unknown spread appears inside EVERY discount factor at once, across every coupon payment -- there is no way to isolate it algebraically the way DV01's own closed-form derivative could be isolated in Section 22.1. Bisection solves this by narrowing a bracket that is known to contain the answer, one halving at a time, until the bracket is smaller than some acceptable tolerance -- but "until the bracket is small enough" and "or after some maximum number of tries, whichever comes first" are two different stopping conditions, and only one of them is actually a guarantee of correctness.

**Background.** Section 21.1's own custom `torch::autograd::Function` already showed how to DIFFERENTIATE through a bisection solve like this one, via the implicit function theorem -- this section's own focus is different: not differentiating through the solver, but the solver's own robustness. The CUDA book's own bisection loop runs `while (fabs(right-left) > tolerance && iterations < 100)`, and that `iterations < 100` clause is a silent escape hatch -- if a particular bond, bracket, and tolerance combination genuinely needed MORE than 100 halvings, the loop would still exit, and would still return whatever midpoint it happened to be at, with absolutely nothing in that return value distinguishing "converged normally" from "gave up."

**Worked Example 22.2.1.** A 2-year, quarterly-coupon bond, 3% coupon rate, 3% risk-free rate, trading at a market price of $98.00 against $100 face value -- since the coupon rate equals the risk-free rate, this bond would price to almost exactly par ($100.00) at zero credit spread, so a genuine positive spread is required to explain its own $98.00 market price. Bisection over bracket `[-0.1, 0.1]`, tolerance `1e-8`, converges in 25 iterations to a spread of `0.0104605...` (about 104.6 basis points), repricing the bond to within `4.06e-7` of its own $98.00 target.

```cpp
#include <iostream>
#include <cmath>
#include <iomanip>

// The CUDA C++ edition's Section 22.2 solves for a coupon-paying bond's
// own Z-spread (the flat spread over the risk-free curve that reprices
// the bond to its observed market price) via bisection -- there is no
// closed-form inverse here, because the unknown spread appears inside
// EVERY discount factor at once, across every coupon payment. Chapter
// 21.1's own custom torch::autograd::Function already showed how to
// DIFFERENTIATE through a bisection solve like this one via the
// implicit function theorem; this section's own focus is different --
// not differentiating through the solver, but the solver's own
// robustness: the CUDA book's own bisection loop runs `while
// (fabs(right-left) > tolerance && iterations < 100)`, and its own
// iteration cap is a silent escape hatch that this file demonstrates
// concretely, not just describes.
double bond_price(double spread, double face, double r, double coupon_rate, int years, int m) {
    int periods = years * m;
    double coupon = (coupon_rate / m) * face;
    double total = 0.0;
    for (int k = 1; k <= periods; k++) {
        double df = std::pow(1.0 + (r + spread) / m, static_cast<double>(k));
        total += coupon / df;
    }
    total += face / std::pow(1.0 + (r + spread) / m, static_cast<double>(periods));
    return total;
}

struct BisectionResult { double spread; int iterations; bool hit_cap; };

BisectionResult solve_zspread(double market_price, double face, double r, double coupon_rate,
                               int years, int m, double lo, double hi, double tol, int max_iter) {
    int iterations = 0;
    while (std::fabs(hi - lo) > tol && iterations < max_iter) {
        double mid = (lo + hi) / 2.0;
        double p = bond_price(mid, face, r, coupon_rate, years, m);
        // A bond's own price falls as its own discount spread rises, so
        // if the bisection midpoint prices ABOVE the market price, the
        // true spread must be HIGHER than mid (raise the lower bound);
        // if it prices below, the true spread is LOWER (lower the upper bound).
        if (p > market_price) {
            lo = mid;
        } else {
            hi = mid;
        }
        iterations++;
    }
    bool hit_cap = (iterations >= max_iter) && (std::fabs(hi - lo) > tol);
    return {(lo + hi) / 2.0, iterations, hit_cap};
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Worked Example 22.2.1: the CUDA book's own exact bond -- 2-year,
    // quarterly coupons, 3% coupon rate, 3% risk-free rate, market
    // price $98.00 against $100 face (a genuine market observation that
    // the bond trades BELOW its coupon-rate-equals-risk-free-rate par
    // value, implying a positive credit spread is required to explain it).
    double market_price = 98.00, face = 100.00, r = 0.03, coupon_rate = 0.03;
    int years = 2, m = 4;
    double tol = 1e-8;
    double lo = -0.1, hi = 0.1;

    double price_at_zero = bond_price(0.0, face, r, coupon_rate, years, m);
    double coupon = (coupon_rate / m) * face;
    std::cout << "Worked Example 22.2.1: 2-year, quarterly-coupon bond, coupon_rate=risk_free=3%, "
              << "market price=$98.00, face=$100.00." << std::endl;
    std::cout << "  quarterly coupon = " << coupon << ", " << (years * m) << " total coupons = "
              << (coupon * years * m) << ", + face = " << (coupon * years * m + face)
              << " undiscounted" << std::endl;
    std::cout << "  price at spread=0 (coupon rate equals risk-free rate) = " << std::setprecision(12)
              << price_at_zero << std::setprecision(6) << " (approx. par, as expected)" << std::endl;

    BisectionResult result = solve_zspread(market_price, face, r, coupon_rate, years, m, lo, hi, tol, 100);
    double converged_price = bond_price(result.spread, face, r, coupon_rate, years, m);
    std::cout << "\n  bisection, bracket=[-0.1, 0.1], tolerance=1e-8, max_iterations=100:" << std::endl;
    std::cout << "  iterations to convergence = " << result.iterations << std::endl;
    std::cout << "  converged Z-spread = " << std::setprecision(12) << result.spread << std::setprecision(6)
              << " (" << (result.spread * 10000.0) << " basis points)" << std::endl;
    std::cout << "  implied yield to maturity = " << ((r + result.spread) * 100.0) << "%" << std::endl;
    std::cout << "  verification: price at converged spread = " << std::setprecision(12) << converged_price
              << std::setprecision(6) << " vs. target $98.00, absolute error = "
              << std::scientific << std::abs(converged_price - market_price) << std::fixed << std::endl;

    // [COMMON TRAP]: the iteration cap is a silent escape hatch. This
    // bond converges comfortably in well under 100 iterations against a
    // reasonable tolerance -- but nothing about the function's own
    // return value distinguishes "converged normally" from "hit the cap
    // and gave up." This is demonstrated concretely, not hypothetically:
    // the SAME solver, called with an artificially tiny max_iter=10
    // against the SAME reasonable 1e-8 tolerance (needing 25 iterations
    // to actually converge, as shown above), is forced to stop early --
    // and returns a plausible-looking spread with no indication
    // whatsoever that it never actually converged.
    BisectionResult capped = solve_zspread(market_price, face, r, coupon_rate, years, m, lo, hi, tol, 10);
    double capped_price = bond_price(capped.spread, face, r, coupon_rate, years, m);
    std::cout << "\n[COMMON TRAP] the SAME solver, called with max_iterations artificially capped at 10 "
              << "(the real convergence above needed 25): iterations run = " << capped.iterations
              << ", hit_cap flag (computed by THIS FILE, not returned by the solver itself) = "
              << capped.hit_cap << std::endl;
    std::cout << "  spread returned = " << std::setprecision(12) << capped.spread << std::setprecision(6)
              << ", price at that spread = " << capped_price << " vs. target $98.00, absolute error = "
              << std::scientific << std::abs(capped_price - market_price) << std::fixed << std::endl;
    std::cout << "  the function's own return value -- a plain double -- carries no signal at all that "
              << "this spread is over $" << std::abs(capped_price - market_price)
              << " away from actually repricing the bond to its market price: a caller reading only "
              << "`result.spread` (ignoring `result.iterations`/`result.hit_cap`, which this file adds "
              << "specifically to make the failure visible -- the CUDA book's own bisection loop returns "
              << "only the spread) would silently use an unconverged number as if it were correct." << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 02_zspread_bisection.cpp -o 02_zspread_bisection
./02_zspread_bisection
```

```text
Worked Example 22.2.1: 2-year, quarterly-coupon bond, coupon_rate=risk_free=3%, market price=$98.00, face=$100.00.
  quarterly coupon = 0.750000, 8 total coupons = 6.000000, + face = 106.000000 undiscounted
  price at spread=0 (coupon rate equals risk-free rate) = 100.000000000000 (approx. par, as expected)

  bisection, bracket=[-0.1, 0.1], tolerance=1e-8, max_iterations=100:
  iterations to convergence = 25
  converged Z-spread = 0.010460522771 (104.605228 basis points)
  implied yield to maturity = 4.046052%
  verification: price at converged spread = 98.000000405662 vs. target $98.00, absolute error = 4.056619e-07

[COMMON TRAP] the SAME solver, called with max_iterations artificially capped at 10 (the real convergence above needed 25): iterations run = 10, hit_cap flag (computed by THIS FILE, not returned by the solver itself) = 1
  spread returned = 0.010449218750, price at that spread = 98.002137 vs. target $98.00, absolute error = 2.136821e-03
  the function's own return value -- a plain double -- carries no signal at all that this spread is over $0.002137 away from actually repricing the bond to its market price: a caller reading only `result.spread` (ignoring `result.iterations`/`result.hit_cap`, which this file adds specifically to make the failure visible -- the CUDA book's own bisection loop returns only the spread) would silently use an unconverged number as if it were correct.
```

**Discussion.** The `[COMMON TRAP]` demonstration is not hypothetical -- it is the identical solver, called on the identical bond, with `max_iterations` artificially lowered to 10 instead of 100. Since this specific bond genuinely needs 25 iterations to converge to `1e-8`, the capped run is FORCED to stop early, and its own returned spread is off by over $0.002 in repriced value -- a small-looking number in isolation, but one that would propagate silently into every downstream Greek, every duration calculation, and every risk report built on top of it, with nothing about the return value itself (a plain `double`) able to signal that anything went wrong. This file's own `BisectionResult` struct adds an `iterations` count and a `hit_cap` flag specifically to make this failure mode visible; the CUDA book's own bisection loop, as described, returns only the spread itself -- which is exactly the point this section is making concrete: an iteration cap that exists purely to guarantee TERMINATION says nothing at all about whether termination happened because the answer was found, or because the loop simply ran out of patience.

## 22.3 Portfolio Duration: The Cheapest Correctness Check This Pipeline Has `[FOUNDATIONAL]`

**Intuition.** A portfolio's own duration is not the average of its bonds' own individual durations -- it is a WEIGHTED average, where each bond's own weight is its own share of the portfolio's total present value. A $10,000 bond with a long duration moves a portfolio's own overall sensitivity far more than a $10 bond with the identical duration, and a PV-weighted average is exactly the calculation that reflects this correctly.

**Background.** The CUDA book's own kernel here is a single elementwise multiply per thread (`duration[idx] * weight[idx]`), which real `torch::Tensor` arithmetic performs as one ordinary vectorized expression, with the reduction (`.sum()`) handled by real, production LibTorch code rather than a hand-written multi-round kernel loop. Its own `[COMMON TRAP]` is an explicit callback to Section 22.1's own reduction bug: portfolio weights summing to exactly 1.0 is not a coincidence, it is a mathematical necessity of how weights are DEFINED (`weight_i = PV_i / sum(PV)`), and checking that sum is the single cheapest correctness check this entire pipeline has -- cheaper than checking any individual bond's own price, and cheap enough to run on every single computation of a portfolio's own weights, every time.

**Worked Example 22.3.1 and 22.3.2.** Three bonds, PV `[$400, $350, $250]`, duration `[2, 5, 10]` years: weights `[0.40, 0.35, 0.25]` (summing to exactly 1.0), portfolio duration `5.05` years. Section 22.1's own real bonds 0-2: weights `[0.0632, 0.3138, 0.6231]`, portfolio duration `0.639975` years.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 22.3 computes portfolio duration as a
// PV-weighted average of each bond's own duration -- for a zero-coupon
// bond, duration equals time to maturity exactly, so this reduces to
// weighting each bond's own maturity by its own share of the
// portfolio's total value. Its own kernel is a single elementwise
// multiply per thread (`output[idx] = duration[idx] * weight[idx]`),
// which real torch::Tensor arithmetic performs as one ordinary
// vectorized expression, with the reduction (`sum()`) handled by real,
// production LibTorch code rather than a hand-written multi-round
// kernel loop. Its own [COMMON TRAP] is a callback to Section 22.1's
// own reduction bug: portfolio weights summing to exactly 1.0 is not a
// coincidence, it is the single cheapest correctness check this whole
// pipeline has -- and this file demonstrates concretely what that
// check would have caught.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Worked Example 22.3.1: a small, hand-specified 3-bond portfolio.
    {
        torch::Tensor pv = torch::tensor({400.0, 350.0, 250.0}, torch::kFloat64);
        torch::Tensor duration = torch::tensor({2.0, 5.0, 10.0}, torch::kFloat64);
        torch::Tensor weights = pv / pv.sum();
        torch::Tensor contributions = weights * duration;
        double portfolio_duration = contributions.sum().item<double>();

        std::cout << "Worked Example 22.3.1: bonds A ($400, 2yr), B ($350, 5yr), C ($250, 10yr):" << std::endl;
        std::cout << "  total = " << pv.sum().item<double>() << ", weights = " << weights
                  << ", sum of weights = " << weights.sum().item<double>() << std::endl;
        std::cout << "  weighted contributions = " << contributions << std::endl;
        std::cout << "  portfolio duration = " << portfolio_duration << " years" << std::endl;
    }

    // Worked Example 22.3.2: Section 22.1's own real first-three bonds.
    {
        torch::Tensor face = torch::tensor({1000.0, 5000.0, 10000.0}, torch::kFloat64);
        torch::Tensor t = torch::tensor({0.25, 0.50, 0.75}, torch::kFloat64);
        torch::Tensor rf = torch::tensor({0.020, 0.021, 0.022}, torch::kFloat64);
        torch::Tensor sp = torch::tensor({0.001, 0.002, 0.003}, torch::kFloat64);
        torch::Tensor yield = rf + sp;
        torch::Tensor pv = face * torch::exp(-yield * t);
        torch::Tensor duration = t;

        torch::Tensor weights = pv / pv.sum();
        double portfolio_duration = (weights * duration).sum().item<double>();
        std::cout << "\nWorked Example 22.3.2: Section 22.1's own real bonds 0-2 (PV=" << pv << "):" << std::endl;
        std::cout << "  total = " << pv.sum().item<double>() << ", weights = " << weights
                  << ", sum of weights = " << weights.sum().item<double>() << std::endl;
        std::cout << "  portfolio duration = " << portfolio_duration << " years" << std::endl;
    }

    // Full 1024-bond portfolio, regenerated with the SAME deterministic
    // formula as Section 22.1 (same portfolio, independently rebuilt in
    // this file rather than passed in, confirming the formula alone --
    // not any hidden shared state -- is what reproduces it).
    const int N = 1024;
    const int64_t faces_cycle[3] = {1000, 5000, 10000};
    torch::Tensor face_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor t_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor rf_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor sp_full = torch::empty({N}, torch::kFloat64);
    {
        auto fv_acc = face_full.accessor<double, 1>();
        auto t_acc = t_full.accessor<double, 1>();
        auto rf_acc = rf_full.accessor<double, 1>();
        auto sp_acc = sp_full.accessor<double, 1>();
        for (int i = 0; i < N; i++) {
            fv_acc[i] = static_cast<double>(faces_cycle[i % 3]);
            t_acc[i] = 0.25 + (i % 120) * 0.25;
            rf_acc[i] = 0.02 + (i % 31) * 0.001;
            sp_acc[i] = 0.001 + (i % 30) * 0.001;
        }
    }
    torch::Tensor yield_full = rf_full + sp_full;
    torch::Tensor pv_full = face_full * torch::exp(-yield_full * t_full);
    double total_correct = pv_full.sum().item<double>();
    torch::Tensor weights_full = pv_full / total_correct;
    double weight_sum_correct = weights_full.sum().item<double>();
    double duration_correct = (weights_full * t_full).sum().item<double>();

    std::cout << "\nfull 1024-bond portfolio, correct total (from Section 22.1): weights sum to "
              << std::setprecision(10) << weight_sum_correct << std::setprecision(6)
              << " (within 1e-9 of exactly 1.0? " << (std::abs(weight_sum_correct - 1.0) < 1e-9)
              << "), portfolio duration = " << duration_correct << " years" << std::endl;

    // [COMMON TRAP]: rebuilding Section 22.1's own buggy scenario -- if
    // every bond's own weight is computed against the WRONG total
    // (summed over only the first 128 of 1024 bonds, exactly as in
    // Section 22.1's own [COMMON TRAP]) instead of the real portfolio
    // total, the weights no longer sum to 1.0 at all, and that failure
    // is visible IMMEDIATELY, with a single cheap check -- long before
    // it would need to propagate into a wrong weighted-duration number
    // to be noticed.
    double total_buggy = pv_full.narrow(0, 0, 128).sum().item<double>();
    torch::Tensor weights_buggy = pv_full / total_buggy;   // every bond's own weight, wrong denominator
    double weight_sum_buggy = weights_buggy.sum().item<double>();
    double duration_buggy = (weights_buggy * t_full).sum().item<double>();
    std::cout << "\n[COMMON TRAP] the SAME 1024 bonds, weights computed against Section 22.1's own buggy "
              << "total (summed over only the first 128 bonds): weights sum to " << weight_sum_buggy
              << " instead of 1.0 -- immediately, visibly wrong, before any weighted-average computation "
              << "downstream is even examined. (This also produces a nonsensical 'portfolio duration' of "
              << duration_buggy << " years, since weights that do not sum to 1.0 are not really weights "
              << "at all -- but the weight-sum check alone already catches the problem, which is exactly "
              << "the CUDA book's own point: it is the single cheapest correctness check this whole "
              << "pipeline has.)" << std::endl;

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 03_portfolio_duration.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 03_portfolio_duration
./03_portfolio_duration
```

```text
Worked Example 22.3.1: bonds A ($400, 2yr), B ($350, 5yr), C ($250, 10yr):
  total = 1000.000000, weights =  0.4000
 0.3500
 0.2500
[ CPUDoubleType{3} ], sum of weights = 1.000000
  weighted contributions =  0.8000
 1.7500
 2.5000
[ CPUDoubleType{3} ]
  portfolio duration = 5.050000 years

Worked Example 22.3.2: Section 22.1's own real bonds 0-2 (PV=  994.7638
 4942.8294
 9814.2469
[ CPUDoubleType{3} ]):
  total = 15751.839996, weights =  0.0632
 0.3138
 0.6231
[ CPUDoubleType{3} ], sum of weights = 1.000000
  portfolio duration = 0.639975 years

full 1024-bond portfolio, correct total (from Section 22.1): weights sum to 1.0000000000 (within 1e-9 of exactly 1.0? 1), portfolio duration = 11.189073 years

[COMMON TRAP] the SAME 1024 bonds, weights computed against Section 22.1's own buggy total (summed over only the first 128 bonds): weights sum to 7.775171 instead of 1.0 -- immediately, visibly wrong, before any weighted-average computation downstream is even examined. (This also produces a nonsensical 'portfolio duration' of 86.996957 years, since weights that do not sum to 1.0 are not really weights at all -- but the weight-sum check alone already catches the problem, which is exactly the CUDA book's own point: it is the single cheapest correctness check this whole pipeline has.)
```

**Discussion.** The full 1024-bond portfolio's own real weights sum to `1.0000000000`, confirmed to within `1e-9` of exactly 1.0 -- the expected, unremarkable result of computing weights correctly. The `[COMMON TRAP]` rebuilds Section 22.1's own buggy scenario directly: computing every bond's own weight against the WRONG total (summed over only the first 128 bonds) makes the weights sum to `7.775171` instead of `1.0`, a visibly, immediately wrong number that requires no downstream computation at all to notice -- no weighted-duration calculation, no comparison against an expected range, nothing beyond a single `.sum()` call on the weights themselves. This is precisely why the CUDA book's own text calls it the cheapest correctness check available: Section 22.1's own bug, buried inside a `.narrow(0, 0, 128)` call that raises no error and produces a plausible-looking (if quantitatively wrong) total, becomes trivially, instantly visible the moment anyone downstream checks whether the weights derived from it actually sum to 1.0.

## 22.4 Monte Carlo Greeks: Autodiff vs. Bump-and-Reprice, and Why Random Numbers Should Sometimes Be Shared `[FOUNDATIONAL]`

**Intuition.** A Monte Carlo option price is an average over many simulated random outcomes, and averages of random things carry sampling noise -- more paths reduce that noise, but never eliminate it. A GREEK (a price's own sensitivity to some input, like the underlying's own starting price) computed by bump-and-reprice -- price the option once at the base input, once at a slightly bumped input, and divide the difference by the bump -- inherits sampling noise from BOTH of those two simulations, and if the two simulations use independent random draws, that noise does not cancel in the difference at all; it compounds. A Greek computed instead by differentiating a SINGLE simulation directly, via autodiff, has no second simulation's own noise to compound with in the first place.

**Background.** The CUDA book's own Section 22.4 hand-writes a Box-Muller transform (turning uniform draws into Gaussian ones) and an inline xorshift RNG, seeded per-path, specifically to allow reproducible bump-and-reprice comparisons via common random numbers (CRN): reuse the IDENTICAL underlying random draws for both the base and the bumped scenario, so that whatever sampling noise both scenarios happen to share cancels out when one is subtracted from the other, leaving mostly the genuine sensitivity being measured. Real `torch::randn` is already a real, production Gaussian sampler -- there is no hand-rolled Box-Muller transform to write, and reusing the identical draws for two scenarios is simply a matter of NOT calling `torch::randn` a second time, rather than a bespoke per-path seeding scheme. This file also uses GBM's own exact solution formula (`S_T = S0*exp((mu-sigma^2/2)*T + sigma*sqrt(T)*Z)`) directly in one vectorized expression across all paths at once, instead of compounding 50 discrete per-step updates in a loop -- mathematically identical, since the CUDA book's own per-step update is itself GBM's exact solution applied to one small time increment, with no discretization bias to accumulate over steps.

**Worked Example 22.4.1.** Five hand-specified terminal prices (no simulation): payoffs `[0, 2, 8, 30, 0]`, mean payoff `8.0`, discounted (`e^(-0.03)`) call price `~=$7.76`. The large-scale run (200,000 real `torch::randn`-sampled paths, `S0=100, K=100, r=mu=3%, sigma=20%, T=1`) is cross-checked directly against the closed-form Black-Scholes price and delta, computed in this same file from scratch via a `std::erf`-based normal CDF -- an independent, mathematically exact reference, used here specifically because two independently-implemented, independently-seeded Monte Carlo simulations (this file's own real `torch::randn`, and the CUDA book's own hand-rolled xorshift generator) cannot be expected to land on the identical sampled price, exactly as this book's own standing policy has established since Sections 19.4 and 20.6 -- an exact, closed-form reference sidesteps that problem entirely by not depending on either book's own RNG at all.

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

// The CUDA C++ edition's Section 22.4 simulates geometric Brownian
// motion (GBM) paths for option pricing, one thread per path, with a
// hand-rolled Box-Muller transform turning uniform draws into N(0,1)
// samples and an inline deterministic xorshift RNG seeded per-path.
// Real torch::randn IS a real, production Gaussian sampler -- there is
// no hand-rolled Box-Muller transform to write here at all, and no
// per-thread RNG seeding scheme to design, since a single real
// torch::manual_seed() call makes an entire tensor of samples
// reproducible at once. This file also reproduces GBM's own closed-
// form terminal distribution directly (S_T = S0*exp((mu-sigma^2/2)*T +
// sigma*sqrt(T)*Z)) in ONE vectorized expression over all paths at
// once, rather than compounding 50 discrete per-step updates in a
// loop -- mathematically identical, since the CUDA book's own per-step
// update formula is itself GBM's exact solution applied to one small
// time increment (no discretization bias), so compounding it 50 times
// and jumping straight to the terminal distribution are the identical
// distribution. Every price and Greek this file computes is cross-
// checked against a completely independent, exact method: the closed-
// form Black-Scholes formula, computed here from scratch via
// std::erf-based normal CDF -- not a reproduction of the CUDA book's
// own reported number (which this book's own standing policy treats as
// impossible to match exactly across two independently-implemented,
// independently-seeded RNG paths, exactly as established in Sections
// 19.4 and 20.6), but an honest, mathematically exact reference this
// file's own genuinely-sampled Monte Carlo estimate is measured against.
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double black_scholes_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1);
}

double black_scholes_call_delta(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return norm_cdf(d1);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double S0 = 100.0, K = 100.0, r = 0.03, mu = 0.03, sigma = 0.20, T = 1.0;

    // Worked Example 22.4.1: a small, hand-specified 5-path example, no
    // simulation involved at all -- five literal terminal prices.
    {
        std::vector<double> st = {95.0, 102.0, 108.0, 130.0, 90.0};
        double sum_payoff = 0.0;
        std::cout << "Worked Example 22.4.1 -- 5 hand-specified terminal prices, S0=100, K=100, r=3%, T=1:"
                  << std::endl;
        for (double s : st) {
            double payoff = std::max(s - K, 0.0);
            sum_payoff += payoff;
            std::cout << "  S_T=" << s << " -> payoff=" << payoff << std::endl;
        }
        double mean_payoff = sum_payoff / static_cast<double>(st.size());
        double discount = std::exp(-r * T);
        std::cout << "  mean payoff = " << mean_payoff << ", discount factor = " << discount
                  << ", call price = " << (discount * mean_payoff) << std::endl;
    }

    // Large-scale run: 200,000 real, genuinely sampled GBM paths, real
    // torch::manual_seed for reproducibility, real torch::randn for the
    // N(0,1) draws -- no hand-rolled RNG or Box-Muller transform
    // anywhere in this file.
    torch::manual_seed(42);
    const int64_t N = 200000;
    torch::Tensor Z = torch::randn({N}, torch::kFloat32);
    torch::Tensor S_T = S0 * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);

    double mean_ST = S_T.mean().item<double>();
    double expected_ST = S0 * std::exp(mu * T);
    std::cout << "\nlarge-scale run: N=" << N << " real torch::randn-sampled GBM paths, S0=" << S0
              << ", mu=" << mu << ", sigma=" << sigma << ", T=" << T << ", seed=42:" << std::endl;
    std::cout << "  mean terminal price = " << mean_ST << ", risk-neutral expectation S0*exp(mu*T) = "
              << expected_ST << " (independent closed-form check, no simulation involved)" << std::endl;

    torch::Tensor call_payoff = torch::relu(S_T - K);
    torch::Tensor put_payoff = torch::relu(K - S_T);
    double discount = std::exp(-r * T);
    double mc_call = discount * call_payoff.mean().item<double>();
    double mc_put = discount * put_payoff.mean().item<double>();
    double call_stderr = discount * call_payoff.std().item<double>() / std::sqrt(static_cast<double>(N));
    double put_stderr = discount * put_payoff.std().item<double>() / std::sqrt(static_cast<double>(N));

    double bs_call = black_scholes_call(S0, K, r, sigma, T);
    double bs_put = black_scholes_put(S0, K, r, sigma, T);

    std::cout << "  Monte Carlo call price = " << mc_call << " (stderr=" << call_stderr << "), closed-form "
              << "Black-Scholes call = " << bs_call << ", difference in stderrs = "
              << ((mc_call - bs_call) / call_stderr) << std::endl;
    std::cout << "  Monte Carlo put price = " << mc_put << " (stderr=" << put_stderr << "), closed-form "
              << "Black-Scholes put = " << bs_put << ", difference in stderrs = "
              << ((mc_put - bs_put) / put_stderr) << std::endl;
    std::cout << "  both MC estimates within 3 standard errors of the independent closed-form reference? "
              << (std::abs(mc_call - bs_call) < 3.0 * call_stderr && std::abs(mc_put - bs_put) < 3.0 * put_stderr)
              << std::endl;

    // Greeks via autodiff: a real backward() pass through the SAME
    // simulation, differentiating the discounted mean payoff with
    // respect to S0 directly -- no bump, no second simulation, no
    // finite-difference noise anywhere in this estimate at all.
    {
        torch::Tensor S0_t = torch::tensor(S0, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        torch::Tensor S_T_diff = S0_t * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        torch::Tensor call_price_diff = discount * torch::relu(S_T_diff - K).mean();
        call_price_diff.backward();
        double autograd_delta = S0_t.grad().item<double>();
        double bs_delta = black_scholes_call_delta(S0, K, r, sigma, T);
        std::cout << "\nGreeks via autodiff: real backward() through the SAME " << N << "-path simulation, "
                  << "d(call_price)/d(S0) = " << autograd_delta << " (a single backward pass, zero extra "
                  << "simulations), closed-form Black-Scholes delta = " << bs_delta
                  << ", difference = " << std::abs(autograd_delta - bs_delta) << std::endl;
    }

    // [COMMON TRAP]: bump-and-reprice with FRESH random paths vs.
    // common random numbers (CRN). A smaller path count (2,000, not
    // 200,000) is used here specifically because the point is to make
    // sampling NOISE visible -- 200,000 paths would make Section 22.4's
    // own honest point (that CRN removes noise a smaller, noisier
    // simulation actually has) hard to see at all.
    {
        const int64_t n_small = 2000;
        const int R = 30;
        const double bump = 1.0;

        auto price_at = [&](double s0v, const torch::Tensor& z) {
            torch::Tensor s_t = s0v * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
            return (discount * torch::relu(s_t - K).mean()).item<double>();
        };

        std::vector<double> fresh_deltas, crn_deltas;
        for (int trial = 0; trial < R; trial++) {
            // Fresh paths: an INDEPENDENT torch::randn() call for the
            // bumped scenario, uncorrelated with the base scenario's
            // own sampling noise -- both scenarios' own sampling error
            // is present, and does not cancel in the difference.
            torch::Tensor z_base_fresh = torch::randn({n_small}, torch::kFloat32);
            torch::Tensor z_bump_fresh = torch::randn({n_small}, torch::kFloat32);
            double p_base_fresh = price_at(S0, z_base_fresh);
            double p_bump_fresh = price_at(S0 + bump, z_bump_fresh);
            fresh_deltas.push_back((p_bump_fresh - p_base_fresh) / bump);

            // Common random numbers: the SAME z tensor priced at BOTH
            // S0 and S0+bump -- correct because Z, in this GBM
            // formulation, does not depend on S0 at all, so reusing it
            // is not a shortcut that introduces bias, it is simply not
            // regenerating an input that was never a function of S0 to
            // begin with.
            torch::Tensor z_crn = torch::randn({n_small}, torch::kFloat32);
            double p_base_crn = price_at(S0, z_crn);
            double p_bump_crn = price_at(S0 + bump, z_crn);
            crn_deltas.push_back((p_bump_crn - p_base_crn) / bump);
        }

        auto mean_of = [](const std::vector<double>& v) {
            double s = 0.0;
            for (double x : v) s += x;
            return s / static_cast<double>(v.size());
        };
        auto std_of = [&](const std::vector<double>& v) {
            double m = mean_of(v), ss = 0.0;
            for (double x : v) ss += (x - m) * (x - m);
            return std::sqrt(ss / static_cast<double>(v.size()));
        };

        double fresh_mean = mean_of(fresh_deltas), fresh_std = std_of(fresh_deltas);
        double crn_mean = mean_of(crn_deltas), crn_std = std_of(crn_deltas);
        double bs_delta = black_scholes_call_delta(S0, K, r, sigma, T);

        std::cout << "\n[COMMON TRAP] bump-and-reprice delta estimate, " << R << " independent trials of "
                  << n_small << " paths each, bump=$" << bump << ", closed-form reference delta = "
                  << bs_delta << ":" << std::endl;
        std::cout << "  FRESH random paths for the bumped scenario each trial: mean delta = " << fresh_mean
                  << ", std across " << R << " trials = " << fresh_std << std::endl;
        std::cout << "  COMMON random numbers (same Z reused for both scenarios) each trial: mean delta = "
                  << crn_mean << ", std across " << R << " trials = " << crn_std << std::endl;
        std::cout << "  CRN reduces the standard deviation of the delta estimate by a factor of "
                  << (fresh_std / crn_std) << "x -- both approaches estimate the SAME underlying "
                  << "quantity (both means are genuinely close to the closed-form delta of " << bs_delta
                  << "), but CRN's own shared sampling noise cancels almost entirely in the difference, "
                  << "while fresh paths' own independent sampling noise in each scenario does not cancel "
                  << "at all, and instead compounds." << std::endl;
    }

    return 0;
}
```

Genuinely compiled and run:

```bash
g++ -std=c++20 -O2 04_monte_carlo_gbm.cpp \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=1 -L"$TORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" \
    -o 04_monte_carlo_gbm
./04_monte_carlo_gbm
```

```text
Worked Example 22.4.1 -- 5 hand-specified terminal prices, S0=100, K=100, r=3%, T=1:
  S_T=95.000000 -> payoff=0.000000
  S_T=102.000000 -> payoff=2.000000
  S_T=108.000000 -> payoff=8.000000
  S_T=130.000000 -> payoff=30.000000
  S_T=90.000000 -> payoff=0.000000
  mean payoff = 8.000000, discount factor = 0.970446, call price = 7.763564

large-scale run: N=200000 real torch::randn-sampled GBM paths, S0=100.000000, mu=0.030000, sigma=0.200000, T=1.000000, seed=42:
  mean terminal price = 103.051231, risk-neutral expectation S0*exp(mu*T) = 103.045453 (independent closed-form check, no simulation involved)
  Monte Carlo call price = 9.445841 (stderr=0.031719), closed-form Black-Scholes call = 9.413403, difference in stderrs = 1.022657
  Monte Carlo put price = 6.484788 (stderr=0.020958), closed-form Black-Scholes put = 6.457957, difference in stderrs = 1.280255
  both MC estimates within 3 standard errors of the independent closed-form reference? 1

Greeks via autodiff: real backward() through the SAME 200000-path simulation, d(call_price)/d(S0) = 0.598454 (a single backward pass, zero extra simulations), closed-form Black-Scholes delta = 0.598706, difference = 0.000252

[COMMON TRAP] bump-and-reprice delta estimate, 30 independent trials of 2000 paths each, bump=$1.000000, closed-form reference delta = 0.598706:
  FRESH random paths for the bumped scenario each trial: mean delta = 0.537445, std across 30 trials = 0.436001
  COMMON random numbers (same Z reused for both scenarios) each trial: mean delta = 0.609748, std across 30 trials = 0.013073
  CRN reduces the standard deviation of the delta estimate by a factor of 33.350067x -- both approaches estimate the SAME underlying quantity (both means are genuinely close to the closed-form delta of 0.598706), but CRN's own shared sampling noise cancels almost entirely in the difference, while fresh paths' own independent sampling noise in each scenario does not cancel at all, and instead compounds.
```

**Discussion.** Both the Monte Carlo call price and put price land within roughly one standard error of their own closed-form Black-Scholes reference -- exactly the behavior a correctly-implemented, unbiased Monte Carlo estimator should show, and independent confirmation that neither this file's own simulation nor its own discounting has a bug, without needing to match any other book's own specific sampled number at all. The autodiff-computed delta (`~=0.5985`) differs from the closed-form delta (`~=0.5987`) by only `0.00025` -- small residual Monte Carlo noise from estimating an expectation with a finite number of paths, not error in the differentiation itself, since `backward()` differentiates the SAME finite-sample estimator exactly; a larger path count would shrink this residual further, exactly as it would shrink the price estimate's own sampling error. The `[COMMON TRAP]` comparison is the section's own central point made numerically concrete: bump-and-reprice with FRESH random paths for the bumped scenario each trial produces a delta estimate with a standard deviation of `~=0.436` across 30 independent trials -- more than 70% the size of the true delta itself, essentially unusable as a risk number -- while the IDENTICAL bump-and-reprice calculation, changed only to reuse the SAME random draws for both scenarios (common random numbers), produces a standard deviation of `~=0.013`, over 33 times smaller, with both approaches' own means still genuinely close to the closed-form reference. Reusing those draws is correct here specifically because `Z`, in this GBM formulation, is sampled independently of `S0` -- it is not a shortcut that introduces bias, it is simply choosing not to regenerate an input that was never a function of the bumped parameter to begin with; and the autodiff Greek computed earlier in this section sidesteps the entire fresh-vs-CRN question, since there is no second simulation, and therefore no second source of sampling noise, to reconcile against the first.

## Complete Runnable Code

### `01_bond_pricing_soa.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 22.1 prices a portfolio of zero-coupon
// bonds by hand-laying out 8 separate contiguous arrays (a struct-of-
// arrays layout, chosen specifically for coalesced GPU memory access)
// and launching one thread per bond -- an embarrassingly parallel
// kernel, since each bond's price depends on nothing but its own four
// numbers. A real torch::Tensor already IS a contiguous, struct-of-
// arrays-shaped buffer -- pricing "a portfolio of bonds" with real
// LibTorch tensor ops is not a kernel-launch decision at all, it is
// just ordinary vectorized arithmetic over a 1D tensor, with the
// embarrassingly-parallel execution handled entirely underneath by
// LibTorch's own CPU (or, with a CUDA build, GPU) tensor kernels. This
// file reproduces the CUDA book's own exact worked numbers (the bond
// portfolio is generated by a deterministic formula, not randomness,
// so genuinely matching digits is expected here, unlike a trained
// network's own final numbers) and then reproduces its own reduction
// bug with a real, LibTorch-native equivalent: not an out-of-bounds
// GPU memory read (real torch::Tensor indexing is bounds-checked), but
// a syntactically valid, silently-too-narrow SLICE -- a portfolio
// summed over only its first 128 of 1024 bonds, with nothing about
// that slice raising any error at all.
int main() {
    std::cout << std::fixed;

    const int N = 1024;
    const int64_t faces_cycle[3] = {1000, 5000, 10000};

    // Deterministic portfolio generation, matching the CUDA book's own
    // exact formulas: face cycles [1000,5000,10000]; t=0.25+(i%120)*0.25;
    // risk_free=0.02+(i%31)*0.001; spread=0.001+(i%30)*0.001. No
    // randomness anywhere -- every one of these 1024 bonds' own inputs
    // is a pure function of its own index i.
    torch::Tensor face_value = torch::empty({N}, torch::kFloat64);
    torch::Tensor time_to_maturity = torch::empty({N}, torch::kFloat64);
    torch::Tensor risk_free_rate = torch::empty({N}, torch::kFloat64);
    torch::Tensor credit_spread = torch::empty({N}, torch::kFloat64);
    {
        auto fv_acc = face_value.accessor<double, 1>();
        auto t_acc = time_to_maturity.accessor<double, 1>();
        auto rf_acc = risk_free_rate.accessor<double, 1>();
        auto sp_acc = credit_spread.accessor<double, 1>();
        for (int i = 0; i < N; i++) {
            fv_acc[i] = static_cast<double>(faces_cycle[i % 3]);
            t_acc[i] = 0.25 + (i % 120) * 0.25;
            rf_acc[i] = 0.02 + (i % 31) * 0.001;
            sp_acc[i] = 0.001 + (i % 30) * 0.001;
        }
    }

    // The whole portfolio's own prices, computed in ONE vectorized
    // expression -- this single line IS the "one thread per bond"
    // kernel the CUDA book hand-writes, just expressed as real tensor
    // arithmetic instead of an explicit kernel launch.
    torch::Tensor yield = risk_free_rate + credit_spread;
    torch::Tensor present_value = face_value * torch::exp(-yield * time_to_maturity);
    torch::Tensor duration = time_to_maturity;   // exact for a zero-coupon bond

    // Worked Example 22.1.1: the first three bonds' own exact numbers.
    auto pv_acc = present_value.accessor<double, 1>();
    auto y_acc = yield.accessor<double, 1>();
    std::cout << "Worked Example 22.1.1 -- first three bonds of the deterministic 1024-bond portfolio:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  bond " << i << ": face=" << face_value[i].item<double>()
                  << ", t=" << time_to_maturity[i].item<double>()
                  << ", risk_free=" << risk_free_rate[i].item<double>()
                  << ", spread=" << credit_spread[i].item<double>()
                  << ", yield=" << y_acc[i]
                  << ", PV=" << std::setprecision(4) << pv_acc[i] << std::setprecision(6) << std::endl;
    }

    // Worked Example 22.1.2: DV01 for bond index 2, three independent ways.
    {
        double t2 = time_to_maturity[2].item<double>();
        double pv2 = present_value[2].item<double>();
        // DV01 = -(dPV/dyield) * 0.0001. For PV = FV*exp(-yield*t),
        // dPV/dyield = -t*PV (a bond's own price FALLS as yield rises),
        // so DV01 = -(-t*PV)*0.0001 = +t*PV*0.0001 -- a POSITIVE number,
        // by the standard convention that DV01 reports the dollar LOSS
        // in a bond's own value per one-basis-point INCREASE in yield.
        double analytic_dv01 = t2 * pv2 * 0.0001;

        // Real autograd: a fresh scalar computation for bond 2 alone,
        // with its own risk_free_rate as the differentiable leaf --
        // d(PV)/d(risk_free_rate) equals d(PV)/d(yield) exactly, since
        // yield = risk_free_rate + credit_spread and the spread term's
        // own derivative with respect to risk_free_rate is zero.
        torch::Tensor face2 = torch::tensor(face_value[2].item<double>(), torch::kFloat64);
        torch::Tensor t2_t = torch::tensor(t2, torch::kFloat64);
        torch::Tensor sp2_t = torch::tensor(credit_spread[2].item<double>(), torch::kFloat64);
        torch::Tensor rf2_t = torch::tensor(risk_free_rate[2].item<double>(),
                                             torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
        torch::Tensor pv2_t = face2 * torch::exp(-(rf2_t + sp2_t) * t2_t);
        pv2_t.backward();
        double dpv_dyield_autograd = rf2_t.grad().item<double>();
        double autograd_dv01 = -dpv_dyield_autograd * 0.0001;

        // Independent central finite-difference cross-check, double
        // precision, on the SAME bond-2 pricing formula.
        double eps = 1e-6;
        double rf2 = risk_free_rate[2].item<double>();
        double sp2 = credit_spread[2].item<double>();
        double face2d = face_value[2].item<double>();
        auto price_at = [&](double rf) { return face2d * std::exp(-(rf + sp2) * t2); };
        double fd_dpv_dyield = (price_at(rf2 + eps) - price_at(rf2 - eps)) / (2.0 * eps);
        double fd_dv01 = -fd_dpv_dyield * 0.0001;

        std::cout << "\nWorked Example 22.1.2 -- DV01 for bond 2 (t=" << t2 << ", PV=" << pv2 << "):" << std::endl;
        std::cout << "  analytic DV01 (t*PV*0.0001, i.e. -(dPV/dyield)*0.0001 since dPV/dyield=-t*PV) = "
                  << analytic_dv01 << " per basis point" << std::endl;
        std::cout << "  real autograd DV01 (backward() through this bond's own pricing formula) = "
                  << autograd_dv01 << " per basis point, matches analytic? "
                  << (std::abs(autograd_dv01 - analytic_dv01) < 1e-9) << std::endl;
        std::cout << "  central finite-difference DV01 (eps=1e-6, independent third method) = "
                  << fd_dv01 << " per basis point, matches analytic within 1e-6? "
                  << (std::abs(fd_dv01 - analytic_dv01) < 1e-6) << std::endl;
    }

    // Full portfolio: total value, PV-weighted average yield, PV-
    // weighted average duration -- each a single vectorized reduction
    // over the real 1024-element tensor.
    double total_correct = present_value.sum().item<double>();
    double weighted_yield = (present_value * yield).sum().item<double>() / total_correct;
    double weighted_duration = (present_value * duration).sum().item<double>() / total_correct;
    std::cout << "\nfull 1024-bond portfolio (correct, real torch::Tensor::sum() over all 1024 bonds):" << std::endl;
    std::cout << "  total portfolio value = " << std::setprecision(2) << total_correct << std::setprecision(6) << std::endl;
    std::cout << "  PV-weighted average yield = " << (weighted_yield * 100.0) << "%" << std::endl;
    std::cout << "  PV-weighted average duration = " << weighted_duration << " years" << std::endl;

    // [COMMON TRAP], the real LibTorch equivalent of the CUDA book's own
    // reduction bug: not an out-of-bounds memory read (torch::Tensor
    // indexing is bounds-checked on CPU), but a syntactically valid
    // slice that is silently too narrow -- summing only the first 128
    // of 1024 bonds, exactly as if a batch size of 128 from an earlier,
    // smaller test run were never updated when the portfolio grew to
    // its real size of 1024. No exception, no warning: torch::Tensor
    // slicing is a completely ordinary, legal operation, whether the
    // slice happens to be the WHOLE portfolio or only a fraction of it.
    double total_buggy = present_value.narrow(0, 0, 128).sum().item<double>();
    double ratio = total_buggy / total_correct;
    std::cout << "\n[COMMON TRAP] present_value.narrow(0, 0, 128).sum() -- summing only the first 128 of "
              << "1024 bonds (a stale batch-size constant from an earlier, smaller test run, never updated "
              << "when the real portfolio grew to 1024 bonds): buggy total = " << std::setprecision(2)
              << total_buggy << std::setprecision(6) << ", correct total = " << std::setprecision(2)
              << total_correct << std::setprecision(6) << " -- the buggy total is only "
              << (ratio * 100.0) << "% of the real portfolio's own value, with no error, no exception, "
              << "and no warning anywhere: .narrow(0, 0, 128) is a completely ordinary, valid tensor "
              << "operation, whether or not 128 happens to be the WHOLE portfolio." << std::endl;

    return 0;
}
```

### `02_zspread_bisection.cpp`

```cpp
#include <iostream>
#include <cmath>
#include <iomanip>

// The CUDA C++ edition's Section 22.2 solves for a coupon-paying bond's
// own Z-spread (the flat spread over the risk-free curve that reprices
// the bond to its observed market price) via bisection -- there is no
// closed-form inverse here, because the unknown spread appears inside
// EVERY discount factor at once, across every coupon payment. Chapter
// 21.1's own custom torch::autograd::Function already showed how to
// DIFFERENTIATE through a bisection solve like this one via the
// implicit function theorem; this section's own focus is different --
// not differentiating through the solver, but the solver's own
// robustness: the CUDA book's own bisection loop runs `while
// (fabs(right-left) > tolerance && iterations < 100)`, and its own
// iteration cap is a silent escape hatch that this file demonstrates
// concretely, not just describes.
double bond_price(double spread, double face, double r, double coupon_rate, int years, int m) {
    int periods = years * m;
    double coupon = (coupon_rate / m) * face;
    double total = 0.0;
    for (int k = 1; k <= periods; k++) {
        double df = std::pow(1.0 + (r + spread) / m, static_cast<double>(k));
        total += coupon / df;
    }
    total += face / std::pow(1.0 + (r + spread) / m, static_cast<double>(periods));
    return total;
}

struct BisectionResult { double spread; int iterations; bool hit_cap; };

BisectionResult solve_zspread(double market_price, double face, double r, double coupon_rate,
                               int years, int m, double lo, double hi, double tol, int max_iter) {
    int iterations = 0;
    while (std::fabs(hi - lo) > tol && iterations < max_iter) {
        double mid = (lo + hi) / 2.0;
        double p = bond_price(mid, face, r, coupon_rate, years, m);
        // A bond's own price falls as its own discount spread rises, so
        // if the bisection midpoint prices ABOVE the market price, the
        // true spread must be HIGHER than mid (raise the lower bound);
        // if it prices below, the true spread is LOWER (lower the upper bound).
        if (p > market_price) {
            lo = mid;
        } else {
            hi = mid;
        }
        iterations++;
    }
    bool hit_cap = (iterations >= max_iter) && (std::fabs(hi - lo) > tol);
    return {(lo + hi) / 2.0, iterations, hit_cap};
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Worked Example 22.2.1: the CUDA book's own exact bond -- 2-year,
    // quarterly coupons, 3% coupon rate, 3% risk-free rate, market
    // price $98.00 against $100 face (a genuine market observation that
    // the bond trades BELOW its coupon-rate-equals-risk-free-rate par
    // value, implying a positive credit spread is required to explain it).
    double market_price = 98.00, face = 100.00, r = 0.03, coupon_rate = 0.03;
    int years = 2, m = 4;
    double tol = 1e-8;
    double lo = -0.1, hi = 0.1;

    double price_at_zero = bond_price(0.0, face, r, coupon_rate, years, m);
    double coupon = (coupon_rate / m) * face;
    std::cout << "Worked Example 22.2.1: 2-year, quarterly-coupon bond, coupon_rate=risk_free=3%, "
              << "market price=$98.00, face=$100.00." << std::endl;
    std::cout << "  quarterly coupon = " << coupon << ", " << (years * m) << " total coupons = "
              << (coupon * years * m) << ", + face = " << (coupon * years * m + face)
              << " undiscounted" << std::endl;
    std::cout << "  price at spread=0 (coupon rate equals risk-free rate) = " << std::setprecision(12)
              << price_at_zero << std::setprecision(6) << " (approx. par, as expected)" << std::endl;

    BisectionResult result = solve_zspread(market_price, face, r, coupon_rate, years, m, lo, hi, tol, 100);
    double converged_price = bond_price(result.spread, face, r, coupon_rate, years, m);
    std::cout << "\n  bisection, bracket=[-0.1, 0.1], tolerance=1e-8, max_iterations=100:" << std::endl;
    std::cout << "  iterations to convergence = " << result.iterations << std::endl;
    std::cout << "  converged Z-spread = " << std::setprecision(12) << result.spread << std::setprecision(6)
              << " (" << (result.spread * 10000.0) << " basis points)" << std::endl;
    std::cout << "  implied yield to maturity = " << ((r + result.spread) * 100.0) << "%" << std::endl;
    std::cout << "  verification: price at converged spread = " << std::setprecision(12) << converged_price
              << std::setprecision(6) << " vs. target $98.00, absolute error = "
              << std::scientific << std::abs(converged_price - market_price) << std::fixed << std::endl;

    // [COMMON TRAP]: the iteration cap is a silent escape hatch. This
    // bond converges comfortably in well under 100 iterations against a
    // reasonable tolerance -- but nothing about the function's own
    // return value distinguishes "converged normally" from "hit the cap
    // and gave up." This is demonstrated concretely, not hypothetically:
    // the SAME solver, called with an artificially tiny max_iter=10
    // against the SAME reasonable 1e-8 tolerance (needing 25 iterations
    // to actually converge, as shown above), is forced to stop early --
    // and returns a plausible-looking spread with no indication
    // whatsoever that it never actually converged.
    BisectionResult capped = solve_zspread(market_price, face, r, coupon_rate, years, m, lo, hi, tol, 10);
    double capped_price = bond_price(capped.spread, face, r, coupon_rate, years, m);
    std::cout << "\n[COMMON TRAP] the SAME solver, called with max_iterations artificially capped at 10 "
              << "(the real convergence above needed 25): iterations run = " << capped.iterations
              << ", hit_cap flag (computed by THIS FILE, not returned by the solver itself) = "
              << capped.hit_cap << std::endl;
    std::cout << "  spread returned = " << std::setprecision(12) << capped.spread << std::setprecision(6)
              << ", price at that spread = " << capped_price << " vs. target $98.00, absolute error = "
              << std::scientific << std::abs(capped_price - market_price) << std::fixed << std::endl;
    std::cout << "  the function's own return value -- a plain double -- carries no signal at all that "
              << "this spread is over $" << std::abs(capped_price - market_price)
              << " away from actually repricing the bond to its market price: a caller reading only "
              << "`result.spread` (ignoring `result.iterations`/`result.hit_cap`, which this file adds "
              << "specifically to make the failure visible -- the CUDA book's own bisection loop returns "
              << "only the spread) would silently use an unconverged number as if it were correct." << std::endl;

    return 0;
}
```

### `03_portfolio_duration.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// The CUDA C++ edition's Section 22.3 computes portfolio duration as a
// PV-weighted average of each bond's own duration -- for a zero-coupon
// bond, duration equals time to maturity exactly, so this reduces to
// weighting each bond's own maturity by its own share of the
// portfolio's total value. Its own kernel is a single elementwise
// multiply per thread (`output[idx] = duration[idx] * weight[idx]`),
// which real torch::Tensor arithmetic performs as one ordinary
// vectorized expression, with the reduction (`sum()`) handled by real,
// production LibTorch code rather than a hand-written multi-round
// kernel loop. Its own [COMMON TRAP] is a callback to Section 22.1's
// own reduction bug: portfolio weights summing to exactly 1.0 is not a
// coincidence, it is the single cheapest correctness check this whole
// pipeline has -- and this file demonstrates concretely what that
// check would have caught.
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Worked Example 22.3.1: a small, hand-specified 3-bond portfolio.
    {
        torch::Tensor pv = torch::tensor({400.0, 350.0, 250.0}, torch::kFloat64);
        torch::Tensor duration = torch::tensor({2.0, 5.0, 10.0}, torch::kFloat64);
        torch::Tensor weights = pv / pv.sum();
        torch::Tensor contributions = weights * duration;
        double portfolio_duration = contributions.sum().item<double>();

        std::cout << "Worked Example 22.3.1: bonds A ($400, 2yr), B ($350, 5yr), C ($250, 10yr):" << std::endl;
        std::cout << "  total = " << pv.sum().item<double>() << ", weights = " << weights
                  << ", sum of weights = " << weights.sum().item<double>() << std::endl;
        std::cout << "  weighted contributions = " << contributions << std::endl;
        std::cout << "  portfolio duration = " << portfolio_duration << " years" << std::endl;
    }

    // Worked Example 22.3.2: Section 22.1's own real first-three bonds.
    {
        torch::Tensor face = torch::tensor({1000.0, 5000.0, 10000.0}, torch::kFloat64);
        torch::Tensor t = torch::tensor({0.25, 0.50, 0.75}, torch::kFloat64);
        torch::Tensor rf = torch::tensor({0.020, 0.021, 0.022}, torch::kFloat64);
        torch::Tensor sp = torch::tensor({0.001, 0.002, 0.003}, torch::kFloat64);
        torch::Tensor yield = rf + sp;
        torch::Tensor pv = face * torch::exp(-yield * t);
        torch::Tensor duration = t;

        torch::Tensor weights = pv / pv.sum();
        double portfolio_duration = (weights * duration).sum().item<double>();
        std::cout << "\nWorked Example 22.3.2: Section 22.1's own real bonds 0-2 (PV=" << pv << "):" << std::endl;
        std::cout << "  total = " << pv.sum().item<double>() << ", weights = " << weights
                  << ", sum of weights = " << weights.sum().item<double>() << std::endl;
        std::cout << "  portfolio duration = " << portfolio_duration << " years" << std::endl;
    }

    // Full 1024-bond portfolio, regenerated with the SAME deterministic
    // formula as Section 22.1 (same portfolio, independently rebuilt in
    // this file rather than passed in, confirming the formula alone --
    // not any hidden shared state -- is what reproduces it).
    const int N = 1024;
    const int64_t faces_cycle[3] = {1000, 5000, 10000};
    torch::Tensor face_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor t_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor rf_full = torch::empty({N}, torch::kFloat64);
    torch::Tensor sp_full = torch::empty({N}, torch::kFloat64);
    {
        auto fv_acc = face_full.accessor<double, 1>();
        auto t_acc = t_full.accessor<double, 1>();
        auto rf_acc = rf_full.accessor<double, 1>();
        auto sp_acc = sp_full.accessor<double, 1>();
        for (int i = 0; i < N; i++) {
            fv_acc[i] = static_cast<double>(faces_cycle[i % 3]);
            t_acc[i] = 0.25 + (i % 120) * 0.25;
            rf_acc[i] = 0.02 + (i % 31) * 0.001;
            sp_acc[i] = 0.001 + (i % 30) * 0.001;
        }
    }
    torch::Tensor yield_full = rf_full + sp_full;
    torch::Tensor pv_full = face_full * torch::exp(-yield_full * t_full);
    double total_correct = pv_full.sum().item<double>();
    torch::Tensor weights_full = pv_full / total_correct;
    double weight_sum_correct = weights_full.sum().item<double>();
    double duration_correct = (weights_full * t_full).sum().item<double>();

    std::cout << "\nfull 1024-bond portfolio, correct total (from Section 22.1): weights sum to "
              << std::setprecision(10) << weight_sum_correct << std::setprecision(6)
              << " (within 1e-9 of exactly 1.0? " << (std::abs(weight_sum_correct - 1.0) < 1e-9)
              << "), portfolio duration = " << duration_correct << " years" << std::endl;

    // [COMMON TRAP]: rebuilding Section 22.1's own buggy scenario -- if
    // every bond's own weight is computed against the WRONG total
    // (summed over only the first 128 of 1024 bonds, exactly as in
    // Section 22.1's own [COMMON TRAP]) instead of the real portfolio
    // total, the weights no longer sum to 1.0 at all, and that failure
    // is visible IMMEDIATELY, with a single cheap check -- long before
    // it would need to propagate into a wrong weighted-duration number
    // to be noticed.
    double total_buggy = pv_full.narrow(0, 0, 128).sum().item<double>();
    torch::Tensor weights_buggy = pv_full / total_buggy;   // every bond's own weight, wrong denominator
    double weight_sum_buggy = weights_buggy.sum().item<double>();
    double duration_buggy = (weights_buggy * t_full).sum().item<double>();
    std::cout << "\n[COMMON TRAP] the SAME 1024 bonds, weights computed against Section 22.1's own buggy "
              << "total (summed over only the first 128 bonds): weights sum to " << weight_sum_buggy
              << " instead of 1.0 -- immediately, visibly wrong, before any weighted-average computation "
              << "downstream is even examined. (This also produces a nonsensical 'portfolio duration' of "
              << duration_buggy << " years, since weights that do not sum to 1.0 are not really weights "
              << "at all -- but the weight-sum check alone already catches the problem, which is exactly "
              << "the CUDA book's own point: it is the single cheapest correctness check this whole "
              << "pipeline has.)" << std::endl;

    return 0;
}
```

### `04_monte_carlo_gbm.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

// The CUDA C++ edition's Section 22.4 simulates geometric Brownian
// motion (GBM) paths for option pricing, one thread per path, with a
// hand-rolled Box-Muller transform turning uniform draws into N(0,1)
// samples and an inline deterministic xorshift RNG seeded per-path.
// Real torch::randn IS a real, production Gaussian sampler -- there is
// no hand-rolled Box-Muller transform to write here at all, and no
// per-thread RNG seeding scheme to design, since a single real
// torch::manual_seed() call makes an entire tensor of samples
// reproducible at once. This file also reproduces GBM's own closed-
// form terminal distribution directly (S_T = S0*exp((mu-sigma^2/2)*T +
// sigma*sqrt(T)*Z)) in ONE vectorized expression over all paths at
// once, rather than compounding 50 discrete per-step updates in a
// loop -- mathematically identical, since the CUDA book's own per-step
// update formula is itself GBM's exact solution applied to one small
// time increment (no discretization bias), so compounding it 50 times
// and jumping straight to the terminal distribution are the identical
// distribution. Every price and Greek this file computes is cross-
// checked against a completely independent, exact method: the closed-
// form Black-Scholes formula, computed here from scratch via
// std::erf-based normal CDF -- not a reproduction of the CUDA book's
// own reported number (which this book's own standing policy treats as
// impossible to match exactly across two independently-implemented,
// independently-seeded RNG paths, exactly as established in Sections
// 19.4 and 20.6), but an honest, mathematically exact reference this
// file's own genuinely-sampled Monte Carlo estimate is measured against.
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double black_scholes_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1);
}

double black_scholes_call_delta(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return norm_cdf(d1);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double S0 = 100.0, K = 100.0, r = 0.03, mu = 0.03, sigma = 0.20, T = 1.0;

    // Worked Example 22.4.1: a small, hand-specified 5-path example, no
    // simulation involved at all -- five literal terminal prices.
    {
        std::vector<double> st = {95.0, 102.0, 108.0, 130.0, 90.0};
        double sum_payoff = 0.0;
        std::cout << "Worked Example 22.4.1 -- 5 hand-specified terminal prices, S0=100, K=100, r=3%, T=1:"
                  << std::endl;
        for (double s : st) {
            double payoff = std::max(s - K, 0.0);
            sum_payoff += payoff;
            std::cout << "  S_T=" << s << " -> payoff=" << payoff << std::endl;
        }
        double mean_payoff = sum_payoff / static_cast<double>(st.size());
        double discount = std::exp(-r * T);
        std::cout << "  mean payoff = " << mean_payoff << ", discount factor = " << discount
                  << ", call price = " << (discount * mean_payoff) << std::endl;
    }

    // Large-scale run: 200,000 real, genuinely sampled GBM paths, real
    // torch::manual_seed for reproducibility, real torch::randn for the
    // N(0,1) draws -- no hand-rolled RNG or Box-Muller transform
    // anywhere in this file.
    torch::manual_seed(42);
    const int64_t N = 200000;
    torch::Tensor Z = torch::randn({N}, torch::kFloat32);
    torch::Tensor S_T = S0 * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);

    double mean_ST = S_T.mean().item<double>();
    double expected_ST = S0 * std::exp(mu * T);
    std::cout << "\nlarge-scale run: N=" << N << " real torch::randn-sampled GBM paths, S0=" << S0
              << ", mu=" << mu << ", sigma=" << sigma << ", T=" << T << ", seed=42:" << std::endl;
    std::cout << "  mean terminal price = " << mean_ST << ", risk-neutral expectation S0*exp(mu*T) = "
              << expected_ST << " (independent closed-form check, no simulation involved)" << std::endl;

    torch::Tensor call_payoff = torch::relu(S_T - K);
    torch::Tensor put_payoff = torch::relu(K - S_T);
    double discount = std::exp(-r * T);
    double mc_call = discount * call_payoff.mean().item<double>();
    double mc_put = discount * put_payoff.mean().item<double>();
    double call_stderr = discount * call_payoff.std().item<double>() / std::sqrt(static_cast<double>(N));
    double put_stderr = discount * put_payoff.std().item<double>() / std::sqrt(static_cast<double>(N));

    double bs_call = black_scholes_call(S0, K, r, sigma, T);
    double bs_put = black_scholes_put(S0, K, r, sigma, T);

    std::cout << "  Monte Carlo call price = " << mc_call << " (stderr=" << call_stderr << "), closed-form "
              << "Black-Scholes call = " << bs_call << ", difference in stderrs = "
              << ((mc_call - bs_call) / call_stderr) << std::endl;
    std::cout << "  Monte Carlo put price = " << mc_put << " (stderr=" << put_stderr << "), closed-form "
              << "Black-Scholes put = " << bs_put << ", difference in stderrs = "
              << ((mc_put - bs_put) / put_stderr) << std::endl;
    std::cout << "  both MC estimates within 3 standard errors of the independent closed-form reference? "
              << (std::abs(mc_call - bs_call) < 3.0 * call_stderr && std::abs(mc_put - bs_put) < 3.0 * put_stderr)
              << std::endl;

    // Greeks via autodiff: a real backward() pass through the SAME
    // simulation, differentiating the discounted mean payoff with
    // respect to S0 directly -- no bump, no second simulation, no
    // finite-difference noise anywhere in this estimate at all.
    {
        torch::Tensor S0_t = torch::tensor(S0, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        torch::Tensor S_T_diff = S0_t * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        torch::Tensor call_price_diff = discount * torch::relu(S_T_diff - K).mean();
        call_price_diff.backward();
        double autograd_delta = S0_t.grad().item<double>();
        double bs_delta = black_scholes_call_delta(S0, K, r, sigma, T);
        std::cout << "\nGreeks via autodiff: real backward() through the SAME " << N << "-path simulation, "
                  << "d(call_price)/d(S0) = " << autograd_delta << " (a single backward pass, zero extra "
                  << "simulations), closed-form Black-Scholes delta = " << bs_delta
                  << ", difference = " << std::abs(autograd_delta - bs_delta) << std::endl;
    }

    // [COMMON TRAP]: bump-and-reprice with FRESH random paths vs.
    // common random numbers (CRN). A smaller path count (2,000, not
    // 200,000) is used here specifically because the point is to make
    // sampling NOISE visible -- 200,000 paths would make Section 22.4's
    // own honest point (that CRN removes noise a smaller, noisier
    // simulation actually has) hard to see at all.
    {
        const int64_t n_small = 2000;
        const int R = 30;
        const double bump = 1.0;

        auto price_at = [&](double s0v, const torch::Tensor& z) {
            torch::Tensor s_t = s0v * torch::exp((mu - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
            return (discount * torch::relu(s_t - K).mean()).item<double>();
        };

        std::vector<double> fresh_deltas, crn_deltas;
        for (int trial = 0; trial < R; trial++) {
            // Fresh paths: an INDEPENDENT torch::randn() call for the
            // bumped scenario, uncorrelated with the base scenario's
            // own sampling noise -- both scenarios' own sampling error
            // is present, and does not cancel in the difference.
            torch::Tensor z_base_fresh = torch::randn({n_small}, torch::kFloat32);
            torch::Tensor z_bump_fresh = torch::randn({n_small}, torch::kFloat32);
            double p_base_fresh = price_at(S0, z_base_fresh);
            double p_bump_fresh = price_at(S0 + bump, z_bump_fresh);
            fresh_deltas.push_back((p_bump_fresh - p_base_fresh) / bump);

            // Common random numbers: the SAME z tensor priced at BOTH
            // S0 and S0+bump -- correct because Z, in this GBM
            // formulation, does not depend on S0 at all, so reusing it
            // is not a shortcut that introduces bias, it is simply not
            // regenerating an input that was never a function of S0 to
            // begin with.
            torch::Tensor z_crn = torch::randn({n_small}, torch::kFloat32);
            double p_base_crn = price_at(S0, z_crn);
            double p_bump_crn = price_at(S0 + bump, z_crn);
            crn_deltas.push_back((p_bump_crn - p_base_crn) / bump);
        }

        auto mean_of = [](const std::vector<double>& v) {
            double s = 0.0;
            for (double x : v) s += x;
            return s / static_cast<double>(v.size());
        };
        auto std_of = [&](const std::vector<double>& v) {
            double m = mean_of(v), ss = 0.0;
            for (double x : v) ss += (x - m) * (x - m);
            return std::sqrt(ss / static_cast<double>(v.size()));
        };

        double fresh_mean = mean_of(fresh_deltas), fresh_std = std_of(fresh_deltas);
        double crn_mean = mean_of(crn_deltas), crn_std = std_of(crn_deltas);
        double bs_delta = black_scholes_call_delta(S0, K, r, sigma, T);

        std::cout << "\n[COMMON TRAP] bump-and-reprice delta estimate, " << R << " independent trials of "
                  << n_small << " paths each, bump=$" << bump << ", closed-form reference delta = "
                  << bs_delta << ":" << std::endl;
        std::cout << "  FRESH random paths for the bumped scenario each trial: mean delta = " << fresh_mean
                  << ", std across " << R << " trials = " << fresh_std << std::endl;
        std::cout << "  COMMON random numbers (same Z reused for both scenarios) each trial: mean delta = "
                  << crn_mean << ", std across " << R << " trials = " << crn_std << std::endl;
        std::cout << "  CRN reduces the standard deviation of the delta estimate by a factor of "
                  << (fresh_std / crn_std) << "x -- both approaches estimate the SAME underlying "
                  << "quantity (both means are genuinely close to the closed-form delta of " << bs_delta
                  << "), but CRN's own shared sampling noise cancels almost entirely in the difference, "
                  << "while fresh paths' own independent sampling noise in each scenario does not cancel "
                  << "at all, and instead compounds." << std::endl;
    }

    return 0;
}
```

## Chapter Summary

This chapter opened Part 7 by applying the LibTorch-native machinery built across Parts 1 through 6 to quantitative finance, mapped section by section against the CUDA C++ edition's own four topics. A 1024-bond portfolio is priced as one ordinary vectorized `torch::Tensor` expression, with DV01 confirmed three independent ways (analytic, real autograd, finite difference) and the CUDA book's own reduction bug reproduced honestly as its real LibTorch-native equivalent: not an out-of-bounds GPU memory read, but a syntactically valid, silently-too-narrow tensor slice. A coupon bond's own credit spread is solved by bisection, with a genuine demonstration of the iteration cap's own silent-failure mode: the identical solver, capped at too few iterations, returns a plausible but unconverged answer with no signal that anything went wrong. Portfolio duration, a PV-weighted average, comes with the cheapest correctness check this whole pipeline has -- weights summing to exactly 1.0 -- shown to catch Section 22.1's own bug instantly, before any downstream calculation is even examined. And Monte Carlo option pricing, built from real `torch::randn` and GBM's own exact solution formula, cross-checks its own price and Greeks against an independent closed-form Black-Scholes reference, then demonstrates concretely why a Greek obtained via a single `backward()` pass is structurally free of a noise problem that bump-and-reprice has to fight off explicitly with common random numbers -- a real, measured 33-times reduction in estimator standard deviation from changing nothing but whether two simulations share their own random draws.

## Self-Check Questions

1. Section 22.1's own `[COMMON TRAP]` (`.narrow(0, 0, 128).sum()`) and the CUDA book's own reduction bug (a kernel launched with too few threads) are described as producing the "identical" failure, yet the underlying MECHANISM is different -- one is a real LibTorch tensor operation, the other is unchecked GPU memory access. Explain precisely what is genuinely identical between the two failures, and what is genuinely different.
2. Section 22.1 computes DV01 as `t*PV*0.0001`, a POSITIVE number, even though a bond's own price FALLS as yield rises (`dPV/dyield` is negative). Explain why DV01 is conventionally reported as a positive number despite this, and what that positive number represents financially.
3. Section 22.2 demonstrates a solver returning a plausible-looking but unconverged spread when its own iteration cap is set too low. What SPECIFIC piece of information would a caller need in order to detect this failure, given that the CUDA book's own bisection loop, as described, returns only the spread itself?
4. Section 22.3's own weight-sum check catches Section 22.1's own bug "instantly," before any weighted-duration number is even computed. Explain why this ordering (checking the weights themselves, before using them in a further calculation) is a strictly cheaper and strictly earlier check than comparing the FINAL weighted-duration result against some expected range.
5. Section 22.4 finds that reusing the same `Z` tensor for both the base and bumped Monte Carlo scenarios (common random numbers) is not merely allowed but CORRECT, whereas reusing weights across genuinely different inputs would usually be a bug. Explain what specific property of this section's own GBM formulation makes reusing `Z` safe, and why that property would not hold for every possible bumped-input scenario in general.

## Where We Go Next

This chapter opened Part 7: Financial Computing Applications, applying the LibTorch-native machinery built across every part before it -- real tensors, real autograd, a real custom `torch::autograd::Function`, real `torch::randn` -- to bond pricing, credit spread solving, portfolio risk aggregation, and Monte Carlo option pricing with autodiff Greeks. Each section's own concrete numbers were checked against either the CUDA book's own genuinely reproducible (deterministic, formula-generated) figures, or, where a real random number generator makes exact reproduction across two independent implementations impossible in principle, against an independent, exact reference computed from scratch in this same file. The appendices that follow return to lower-level material -- installation, memory architecture, functional and lambda programming, the SM pipeline, and tensor contractions -- providing reference depth for topics this book's own main chapters have used along the way without dwelling on every implementation detail.

## Worked Solutions

**1.** What is genuinely identical between the two failures is the SHAPE of the mistake and its own consequence: in both cases, a portfolio of 1024 bonds is silently reduced to a computation over only the first 128 of them, with no error, no exception, and no warning raised anywhere, and the resulting total is reported and used as if it were the correct total for the whole portfolio -- the same deterministic bonds, the same slice size, and (as a direct consequence) essentially the same wrong numbers. What is genuinely different is the MECHANISM producing that shape: the CUDA book's own bug is a kernel launched with too few threads, so 896 of the portfolio's own 1024 bonds are simply never READ by any thread at all -- an under-provisioned kernel silently leaving memory untouched, with no bounds violation and technically nothing "illegal" happening, just fewer threads than the problem needed. This file's own bug is a `.narrow(0, 0, 128)` call, a perfectly ordinary and explicitly-requested tensor slicing operation -- real LibTorch tensor indexing IS bounds-checked (reading past a tensor's real allocated size would raise an exception, not silently succeed), so the identical "reading out of bounds" failure mode literally cannot occur here; what CAN occur, and does, is a programmer explicitly asking for a slice that is smaller than the whole portfolio, on purpose or by a stale leftover constant, with the tensor API having no way to know that 128 was not the programmer's own genuine intent.

**2.** DV01 is conventionally defined as `-(dPV/dyield) * 0.0001` specifically so that it reports a POSITIVE number representing a LOSS: since `dPV/dyield` is negative (a bond's own price falls as yield rises), the leading minus sign flips that negative derivative into a positive quantity. Financially, this positive number answers a specific, practically useful question in a specific, conventional direction: "how many dollars does this bond LOSE in value if its own yield rises by one basis point?" -- framed as a magnitude of loss (a risk number a trader wants to see as a plain positive dollar figure, comparable in size across a whole portfolio of similar-magnitude risks) rather than as a signed rate of change (which would require a reader to separately remember and interpret the sign convention every time). The underlying calculus is unambiguous and was confirmed three independent ways in this section (analytic, real autograd, finite difference) — all three agree exactly once the same sign convention is applied consistently; the convention itself is a reporting choice about how to present the RESULT of that calculus, not a second, competing mathematical claim about what the derivative actually is.

**3.** A caller would need, at minimum, the ITERATION COUNT the solver actually used, compared against its own maximum -- exactly the `iterations` field this file's own `BisectionResult` struct adds specifically to make this visible (or, more directly, a boolean `hit_cap` flag, computed here as `iterations >= max_iter && remaining bracket width > tolerance`). Neither piece of information is derivable from the returned spread value alone: a converged spread and an unconverged-but-capped spread are both just ordinary, plausible-looking floating-point numbers, indistinguishable from each other by inspection. The CUDA book's own bisection loop, as described, returns only the spread itself -- meaning any caller of that exact function has no way at all to distinguish a genuinely converged answer from one that silently hit the cap, short of independently re-checking `bond_price(returned_spread)` against the market price and comparing the residual to some acceptable tolerance themselves, which is exactly the verification step this section's own worked example performs explicitly, and which a caller relying only on the bare return value would have no reason to think to do.

**4.** Checking that the weights themselves sum to 1.0 requires exactly one operation -- a single `.sum()` call over an already-computed tensor -- and catches ANY error that corrupts the weights, regardless of what that error's own root cause happens to be (a wrong denominator, a dropped bond, a unit mismatch, or anything else that would make the weights not actually be proper proportions of a whole). Comparing a FINAL weighted-duration result against "some expected range" requires first computing that weighted duration at all (an additional multiply-and-reduce step downstream of the weights), and then requires the checker to already know, independently, what a "reasonable" duration range looks like for this specific portfolio -- a much fuzzier, portfolio-specific judgment call, compared to checking against the single, universal, portfolio-independent fact that proper weights must sum to exactly 1.0. The weight-sum check is therefore cheaper (fewer operations, computed earlier in the pipeline, before any further calculation depends on the corrupted values) and sharper (a precise, unambiguous pass/fail test with no judgment call required) than waiting to notice that a final downstream number merely looks unusual.

**5.** The specific property making it safe is that, in this section's own GBM formulation, `Z` (a standard normal draw) is generated INDEPENDENTLY of `S0` -- nothing about how `Z` is sampled depends in any way on the value of the starting price being priced, so reusing the identical `Z` tensor for `S0=100` and `S0=101` is not smuggling in information that shouldn't be shared between the two scenarios, it is simply declining to re-draw a random input that was never a function of `S0` to begin with. This property would NOT hold in general for every possible bumped-input scenario: if the quantity being bumped were something that genuinely changes the distribution `Z` itself should be drawn from -- for instance, bumping `sigma` (volatility) in a model where the discretization or sampling scheme depends on `sigma`, or bumping a parameter that changes the number of paths or time steps needed for numerical stability -- then reusing the identical `Z` draws across the base and bumped scenario could introduce a genuine bias rather than merely cancelling shared noise, since `Z` would no longer represent "the same underlying randomness, evaluated at a different input" but a stale sample that no longer matches what the bumped scenario's own model actually calls for. Common random numbers are a valid variance-reduction technique specifically when the randomness being reused is provably independent of the input being varied -- a property that has to be checked for each specific bump, not assumed to hold universally.
