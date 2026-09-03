# Chapter 13: Matrix Operations

> "The CUDA C++ edition's Chapter 13 treats matrix multiplication as tensor contraction restricted to two dimensions, hand-writing scalar and SIMD versions with a genuine shape-check gap -- its own Worked Example 13.1.4 shows a mismatched call silently reading a sentinel value and returning a wrong but successfully-returned result. `torch::matmul` is the real, production equivalent, and this chapter's own first honest divergence is that it refuses to do that: a genuine dimension mismatch throws a real error rather than reading past a buffer's end. The rest of the chapter follows the same pattern -- transpose, reshape, and structure-aware diagonal and triangular operations all have real, already-implemented LibTorch equivalents, each tested against the CUDA book's own exact worked numbers, and in two cases (reshape's own trap, and triangular solving) reproducing the CUDA book's own findings almost exactly, for related but genuinely different underlying reasons."

**What you will understand by the end of this chapter:**

- That `torch::matmul` reproduces the CUDA book's own `[[22,28],[49,64]]` result exactly, and that a genuine shape mismatch throws a real `c10::Error` rather than silently reading a sentinel value the way the CUDA book's own unchecked kernel does
- That `torch::Tensor`'s own `.t()` is a zero-copy VIEW -- the identical real memory address before and after, confirmed via `.data_ptr()` -- unlike the CUDA book's own transpose, which genuinely moves every element into a freshly-allocated buffer
- That `.view()` refuses to reinterpret a non-contiguous tensor's strides (throwing a real error), while `.reshape()` falls back to a genuine copy -- and that copy reproduces the CUDA book's own reshape TRAP almost exactly, mismatching the original matrix at the identical 4 of 6 positions, for a related but distinct underlying reason
- That `A * d` (broadcasting a diagonal vector across columns) computes the CUDA book's own exact diagonal-multiplication result using the identical O(n^2) technique, confirmed bit-identical to a full O(n^3) dense matmul against the materialized diagonal matrix, and that `torch::linalg_solve_triangular` reproduces the CUDA book's own forward-substitution solution (`x0=3.0, x1=3.5`) exactly

**What you need to know first:**

- Chapter 12's element-wise operations and broadcasting -- this chapter's own diagonal-multiplication technique in Section 13.4 reuses Chapter 12's own broadcasting rule directly, applied to a matrix-vector pair rather than two matrices
- Chapter 7's memory layout and strides -- Section 13.2's transpose-as-view and Section 13.3's reshape/view distinction both rest directly on the stride mechanics Chapter 7 first measured
- If you've read the CUDA C++ edition's Chapter 13: its own four hand-built pieces -- unchecked matrix multiply, a genuinely data-moving transpose, a `View2D`/`reshape` pair with a real trap, and structure-aware diagonal/triangular operations -- all have real LibTorch equivalents; this chapter verifies the CUDA book's own exact numbers on each, and specifically investigates where the real implementations' behavior differs from the CUDA book's own hand-built version, including two cases where a related trap or technique reappears for a different underlying reason.

## 13.1 Matrix Multiplication: A Real Shape Check Where the CUDA Book Has None `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `scalar_matrix_multiply` and `simd_matrix_multiply<SIMD_WIDTH>` compute matrix multiplication as tensor contraction restricted to two dimensions: every output position is a full dot product between a row of the left operand and a column of the right one. `torch::matmul` computes the identical mathematical operation as a real, already-implemented, production function -- but unlike the CUDA book's own version, it validates that the two operands' shapes are actually compatible before doing any arithmetic at all.

### Background

The CUDA book's own Worked Example 13.1.1: `X` (2x3) times `M` (3x2) gives `Y(0,0)=1*1+2*3+3*5=22`, `Y(0,1)=1*2+2*4+3*6=28`, `Y(1,0)=4*1+5*3+6*5=49`, `Y(1,1)=4*2+5*4+6*6=64`. Its own Worked Example 13.1.3: multiplication counts of `12` (2x3 by 3x2), `27` (3x3 by 3x3), and `262144` (64x64 by 64x64). Its own Worked Example 13.1.4 is a critical bug finding: with `a.cols=3` but `b.rows=2`, `b.get(2,0)` reads a sentinel value `999.0` placed just past the real data, producing a silently wrong result `[[3004.0, 3007.0], [6013.0, 6022.0]]` rather than any error.

### Worked Example 13.1.1 -- the CUDA book's own numbers, and a genuine shape check where the CUDA book has none

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.1 hand-writes scalar_matrix_multiply and
// a SIMD variant, computing each output position as a full dot product
// between a row of X and a column of M. Its own Worked Example 13.1.4 finds
// a genuine bug: neither function validates shape compatibility, so a
// caller passing a.cols=3 against b.rows=2 reads a sentinel value
// (999.0) placed just past the real data, producing a silently WRONG
// result ([[3004,3007],[6013,6022]]) rather than any error at all.
// torch::matmul is real, already-implemented, and -- unlike the CUDA
// book's own version -- performs a genuine shape check, throwing a real
// c10::Error on a dimension mismatch rather than reading past the end of
// a buffer. This file verifies the CUDA book's own exact worked numbers,
// then directly tests this honest divergence: this book's own
// matrix-multiply cannot silently produce a wrong-shape result at all.
int main() {
    // Worked Example 13.1.1: X (2x3) times M (3x2).
    torch::Tensor X = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor M = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});

    torch::Tensor Y = torch::matmul(X, M);
    std::cout << "X (2x3) @ M (3x2) =\n" << Y << std::endl;
    torch::Tensor expected = torch::tensor({{22.0f, 28.0f}, {49.0f, 64.0f}});
    std::cout << "matches CUDA book's own [[22,28],[49,64]]? "
              << torch::equal(Y, expected) << std::endl;

    // Worked Example 13.1.3: multiplication counts. torch::matmul's own
    // real cost is the same a.rows*a.cols*b.cols multiplications as the
    // CUDA book's own naive triple loop -- this is hand-counted here
    // rather than measured, since torch::matmul does not expose an
    // internal multiplication counter.
    int64_t count_2x3_3x2 = 2 * 3 * 2;
    int64_t count_3x3_3x3 = 3 * 3 * 3;
    int64_t count_64x64 = 64 * 64 * 64;
    std::cout << "\nhand-counted multiplications for 2x3 @ 3x2 = " << count_2x3_3x2
              << ", CUDA book's own expected = 12, match = " << (count_2x3_3x2 == 12) << std::endl;
    std::cout << "hand-counted multiplications for 3x3 @ 3x3 = " << count_3x3_3x3
              << ", CUDA book's own expected = 27, match = " << (count_3x3_3x3 == 27) << std::endl;
    std::cout << "hand-counted multiplications for 64x64 @ 64x64 = " << count_64x64
              << ", CUDA book's own expected = 262144, match = " << (count_64x64 == 262144) << std::endl;

    // Honest divergence: the CUDA book's own Worked Example 13.1.4 shows a
    // shape mismatch reading a sentinel value and producing a WRONG but
    // successfully-returned result. torch::matmul instead throws a real
    // c10::Error -- confirmed here by genuinely catching it, not by
    // reading documentation about what it should do.
    bool threw = false;
    std::string error_msg;
    try {
        torch::Tensor bad_b = torch::zeros({2, 2});  // b.rows=2, but X.cols=3 -- genuine mismatch
        torch::Tensor bad_result = torch::matmul(X, bad_b);
    } catch (const c10::Error& e) {
        threw = true;
        error_msg = e.what_without_backtrace();
    }
    std::cout << "\nmatmul(X [2x3], mismatched [2x2]) threw a real c10::Error (rather than silently "
              << "reading a sentinel value, the CUDA book's own bug)? " << threw << std::endl;
    std::cout << "error message contains 'size' or 'shape' or 'mat': "
              << (error_msg.find("size") != std::string::npos ||
                  error_msg.find("shape") != std::string::npos ||
                  error_msg.find("mat") != std::string::npos) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
X (2x3) @ M (3x2) =
 22  28
 49  64
[ CPUFloatType{2,2} ]
matches CUDA book's own [[22,28],[49,64]]? 1

hand-counted multiplications for 2x3 @ 3x2 = 12, CUDA book's own expected = 12, match = 1
hand-counted multiplications for 3x3 @ 3x3 = 27, CUDA book's own expected = 27, match = 1
hand-counted multiplications for 64x64 @ 64x64 = 262144, CUDA book's own expected = 262144, match = 1

matmul(X [2x3], mismatched [2x2]) threw a real c10::Error (rather than silently reading a sentinel value, the CUDA book's own bug)? 1
error message contains 'size' or 'shape' or 'mat': 1
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all:

```text
matmul [[22. 28.]
 [49. 64.]]
```

### Discussion

`X @ M` matches the CUDA book's own numbers exactly, and all three hand-counted multiplication totals confirm the CUDA book's own scaling claim that a general matmul costs `a.rows * a.cols * b.cols` multiplications regardless of which implementation performs them. The shape-mismatch test is this section's own strongest evidence: `torch::matmul` genuinely throws a real `c10::Error`, caught here with an ordinary `try`/`catch` rather than assumed from documentation, and the error message itself references the operands' shapes -- a real, structural difference from the CUDA book's own `b.get(2,0)`, which has no bounds check at all and simply reads whatever sentinel value happens to sit past the buffer's real data. The CUDA book's own bug is not a hypothetical concern; `torch::matmul`'s real shape validation is exactly the fix that bug is missing, tested here as a genuinely thrown exception rather than a claim.

> `[COMMON TRAP]` It's tempting to think a shape-mismatch error is a minor convenience -- surely a caller would notice their matrices don't multiply cleanly before ever running the code. The CUDA book's own Worked Example 13.1.4 shows exactly why that's not reliable: the buggy call does not crash, does not warn, and returns a completely well-formed-looking 2x2 matrix of plausible numbers (`[[3004.0, 3007.0], [6013.0, 6022.0]]`) that a caller has no particular reason to distrust without independently verifying it by hand. A real shape check, like `torch::matmul`'s own, converts a silent wrong-answer bug into an immediate, loud, unmissable failure -- a strictly better outcome even though it interrupts the program.

## 13.2 Transpose: A Real View, Where the CUDA Book Moves Every Element `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `matrix_transpose_simd<SIMD_WIDTH>` reads sequentially from one row and scatters writes into different rows of a freshly-allocated output buffer -- real data movement, one element (or SIMD lane group) at a time. `torch::Tensor`'s own `.t()` does something structurally different: rather than moving any data, it returns a VIEW over the identical underlying storage, with the row and column strides simply swapped.

### Background

The CUDA book's own Worked Example 13.2.1: transposing `A=[[1,2,3],[4,5,6]]` gives `A^T=[[1,4],[2,5],[3,6]]`, with the CUDA book's own freshly-written output buffer stored in the order `[1,4,2,5,3,6]`.

### Worked Example 13.2.1 -- the CUDA book's own numbers, and a genuine zero-copy view

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.2 hand-writes matrix_transpose_simd:
// sequential reads from one row, scattered writes into DIFFERENT rows of a
// freshly-allocated output buffer -- real data movement, one element at a
// time (or one SIMD lane group at a time). torch::Tensor's own .t() (and
// .transpose(0,1)) does something structurally different: rather than
// moving any data, it returns a VIEW over the identical underlying
// storage, with swapped strides -- an honest divergence from the CUDA
// book's own genuine element-by-element scatter. This file verifies the
// CUDA book's own exact worked numbers, then directly demonstrates the
// zero-copy nature of torch::Tensor's transpose: the same real memory
// address before and after, confirmed via .data_ptr(), something the CUDA
// book's own scatter-based transpose could never do since it allocates an
// entirely new output buffer.
int main() {
    // Worked Example 13.2.1: transposing a 2x3 matrix.
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor At = A.t();
    std::cout << "A^T =\n" << At << std::endl;
    torch::Tensor expected = torch::tensor({{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}});
    std::cout << "matches CUDA book's own [[1,4],[2,5],[3,6]]? "
              << torch::equal(At, expected) << std::endl;

    // Honest divergence: the CUDA book's own transpose genuinely moves
    // every element into a new buffer (storage order [1,4,2,5,3,6] on a
    // freshly allocated output). torch::Tensor's .t() instead returns a
    // VIEW: the identical underlying storage, just reinterpreted via
    // swapped strides -- confirmed here by real pointer equality, not by
    // reading documentation about what a view "should" do.
    bool same_storage = (A.data_ptr() == At.data_ptr());
    std::cout << "\nA and A^T share the identical real memory address (data_ptr equal)? "
              << same_storage << std::endl;
    std::cout << "A^T.is_contiguous() = " << At.is_contiguous()
              << " (false is expected -- a transposed VIEW is not contiguous in row-major order, "
              << "unlike the CUDA book's own freshly-allocated, genuinely contiguous output buffer)" << std::endl;

    auto strides = At.strides();
    std::cout << "A^T strides = [" << strides[0] << "," << strides[1]
              << "], hand-derived expected = [1,3] (A's own row-major strides, simply swapped), match = "
              << (strides[0] == 1 && strides[1] == 3) << std::endl;

    // Despite being a view with no data movement, reading A^T still
    // produces the mathematically correct values at every position --
    // confirmed exhaustively here rather than only at the positions the
    // CUDA book's own Worked Example 13.2.1 happens to print.
    bool all_correct = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 2; j++) {
            float via_transpose = At[i][j].item<float>();
            float via_original = A[j][i].item<float>();
            if (via_transpose != via_original) all_correct = false;
        }
    }
    std::cout << "every A^T[i][j] == A[j][i], read through the view with no data movement at all? "
              << all_correct << std::endl;

    // A real consequence of the view: writing through A^T changes A too,
    // since they are literally the same storage -- something impossible
    // for the CUDA book's own transpose, which allocates a separate buffer.
    torch::Tensor A2 = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    torch::Tensor A2t = A2.t();
    A2t[0][1] = 99.0f;
    std::cout << "\nafter writing A2t[0][1]=99: A2[1][0] = " << A2[1][0].item<float>()
              << " (also changed, since A2 and A2t share real storage -- impossible for the CUDA "
              << "book's own copying transpose)" << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
A^T =
 1  4
 2  5
 3  6
[ CPUFloatType{3,2} ]
matches CUDA book's own [[1,4],[2,5],[3,6]]? 1

A and A^T share the identical real memory address (data_ptr equal)? 1
A^T.is_contiguous() = 0 (false is expected -- a transposed VIEW is not contiguous in row-major order, unlike the CUDA book's own freshly-allocated, genuinely contiguous output buffer)
A^T strides = [1,3], hand-derived expected = [1,3] (A's own row-major strides, simply swapped), match = 1
every A^T[i][j] == A[j][i], read through the view with no data movement at all? 1

after writing A2t[0][1]=99: A2[1][0] = 99 (also changed, since A2 and A2t share real storage -- impossible for the CUDA book's own copying transpose)
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all:

```text
transpose [[1. 4.]
 [2. 5.]
 [3. 6.]]
```

### Discussion

`A^T` matches the CUDA book's own numbers exactly, but the mechanism behind it is fundamentally different: `A.data_ptr() == At.data_ptr()` is real pointer equality, confirming `.t()` allocated nothing at all -- it is a view over `A`'s own existing storage, with the strides `[1,3]` (swapped from `A`'s own `[3,1]`) encoding the reinterpretation. `A^T.is_contiguous()` reporting `false` is the direct, structural signature of this: a row-major-contiguous `A` becomes a genuinely non-contiguous `A^T`, because the transpose changed how the same bytes are addressed without changing where any of them live. The final write-through test makes the consequence concrete: writing `A2t[0][1] = 99.0f` changes `A2[1][0]` too, because `A2` and `A2t` are not two related-but-separate objects the way the CUDA book's own original matrix and its freshly-allocated transposed output would be -- they are the same 16 bytes of real memory, addressed two different ways.

> `[COMMON TRAP]` A reader might assume `torch::Tensor`'s zero-copy transpose is simply a "faster" version of the CUDA book's own data-moving transpose, doing the identical logical operation more efficiently. The write-through test above shows this framing understates the difference: the two are not interchangeable in every context, because a zero-copy view shares mutability with its source in a way a genuine copy never does. Code that transposes a tensor and then mutates the result, expecting the original to be unaffected, would behave correctly against the CUDA book's own copying transpose but would silently corrupt the original tensor against `torch::Tensor`'s own view-based one -- a real behavioral difference, not merely a performance one, and precisely why `.contiguous()` (forcing a genuine copy) exists as a separate, explicit operation when independence is actually required.

## 13.3 Reshape and View: The CUDA Book's Own Trap, Reproduced for a Different Reason `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `View2D`/`reshape()` allows a free (zero-copy) reshape for a contiguous layout, but its own Worked Example 13.3.2 shows a genuine trap: naively reshaping a transposed matrix's own storage, without accounting for what the transpose did, produces a result that looks plausible but is mathematically wrong. `torch::Tensor` splits this territory into two real functions: `.view()`, which always stays zero-copy and throws a real error when that is impossible, and `.reshape()`, which falls back to a genuine copy rather than ever silently misinterpreting memory.

### Background

The CUDA book's own Worked Example 13.3.1: 12 contiguous values reshaped from `(2,6)` to `(3,4)`, at the same buffer address (zero-copy). Its own Worked Example 13.3.2: naively reshaping transposed storage `[1,4,2,5,3,6]` as `(2,3)` produces `[[1,4,2],[5,3,6]]`, mismatching the original `A=[[1,2,3],[4,5,6]]` at 4 of 6 positions.

### Worked Example 13.3.1 -- the CUDA book's own free reshape, and its own trap reproduced for a different reason

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.3 hand-builds View2D (a data pointer,
// rows, cols, row_stride) and a reshape() function: free (zero-copy) for a
// contiguous layout, but its own Worked Example 13.3.2 shows a genuine
// trap -- naively reshaping a transposed matrix's own storage, without
// accounting for the transpose, produces a result that looks plausible but
// mismatches the original matrix at 4 of 6 positions. torch::Tensor has
// two real, already-implemented functions covering this exact territory:
// .view() (always zero-copy, but throws a real error when the requested
// shape is not expressible as a view of the current strides) and
// .reshape() (falls back to a real copy when a view is not possible).
// This file verifies the CUDA book's own free-reshape case directly, then
// reproduces the CUDA book's own reshape TRAP almost exactly -- calling
// .reshape() on a transposed tensor produces the identical wrong numbers
// the CUDA book's own naive reshape produces, for a related but distinct
// reason (row-major logical flattening, not raw storage reinterpretation).
int main() {
    // Worked Example 13.3.1: a free (zero-copy) reshape of a contiguous
    // buffer, from (2,6) to (3,4).
    torch::Tensor data = torch::arange(12, torch::kFloat32);
    torch::Tensor r1 = data.view({2, 6});
    std::cout << "data.view({2,6}) =\n" << r1 << std::endl;
    std::cout << "r1 shares data.data_ptr() (zero-copy, CUDA book's own 'free reshape' claim)? "
              << (data.data_ptr() == r1.data_ptr()) << std::endl;

    torch::Tensor r2 = r1.reshape({3, 4});
    std::cout << "\nr1.reshape({3,4}) =\n" << r2 << std::endl;
    std::cout << "r2 ALSO shares the original data_ptr (still zero-copy, since r1 is contiguous)? "
              << (data.data_ptr() == r2.data_ptr()) << std::endl;

    // Honest divergence + parity: .view() is torch::Tensor's own
    // zero-copy-only reshape -- it genuinely throws a real error when the
    // requested shape cannot be expressed via the current tensor's
    // strides, rather than silently reinterpreting raw bytes the way the
    // CUDA book's own naive reshape does.
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor At = A.t();  // a non-contiguous VIEW, per Section 13.2

    bool view_threw = false;
    try {
        torch::Tensor bad = At.view({2, 3});
    } catch (const c10::Error&) {
        view_threw = true;
    }
    std::cout << "\nAt.view({2,3}) on a non-contiguous transposed tensor threw a real error "
              << "(view refuses to silently reinterpret incompatible strides)? " << view_threw << std::endl;

    // .reshape() falls back to a real copy -- and the CUDA book's own
    // Worked Example 13.3.2 trap reproduces almost exactly: reshaping the
    // transposed matrix's own values (in row-major logical order) as (2,3)
    // produces [[1,4,2],[5,3,6]], NOT the original A -- mismatching it at
    // 4 of 6 positions, the CUDA book's own exact finding, even though the
    // underlying mechanism differs (torch::Tensor reshapes the tensor's
    // logical row-major values; the CUDA book's own naive reshape
    // reinterprets raw storage bytes directly -- both wrong for the same
    // underlying reason: a transpose is not undone by reshaping alone).
    torch::Tensor reshaped = At.reshape({2, 3});
    std::cout << "\nAt.reshape({2,3}) =\n" << reshaped << std::endl;
    torch::Tensor trap_expected = torch::tensor({{1.0f, 4.0f, 2.0f}, {5.0f, 3.0f, 6.0f}});
    std::cout << "matches CUDA book's own naive-reshape trap result [[1,4,2],[5,3,6]]? "
              << torch::equal(reshaped, trap_expected) << std::endl;

    int mismatches = 0;
    for (int64_t i = 0; i < 2; i++)
        for (int64_t j = 0; j < 3; j++)
            if (reshaped[i][j].item<float>() != A[i][j].item<float>()) mismatches++;
    std::cout << "mismatches against original A [[1,2,3],[4,5,6]] = " << mismatches
              << ", CUDA book's own expected = 4 (of 6 positions), match = " << (mismatches == 4) << std::endl;

    std::cout << "reshaped shares At's data_ptr (a real copy was made, since At's strides "
              << "made a zero-copy view impossible)? " << (At.data_ptr() == reshaped.data_ptr()) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
data.view({2,6}) =
  0   1   2   3   4   5
  6   7   8   9  10  11
[ CPUFloatType{2,6} ]
r1 shares data.data_ptr() (zero-copy, CUDA book's own 'free reshape' claim)? 1

r1.reshape({3,4}) =
  0   1   2   3
  4   5   6   7
  8   9  10  11
[ CPUFloatType{3,4} ]
r2 ALSO shares the original data_ptr (still zero-copy, since r1 is contiguous)? 1

At.view({2,3}) on a non-contiguous transposed tensor threw a real error (view refuses to silently reinterpret incompatible strides)? 1

At.reshape({2,3}) =
 1  4  2
 5  3  6
[ CPUFloatType{2,3} ]
matches CUDA book's own naive-reshape trap result [[1,4,2],[5,3,6]]? 1
mismatches against original A [[1,2,3],[4,5,6]] = 4, CUDA book's own expected = 4 (of 6 positions), match = 1
reshaped shares At's data_ptr (a real copy was made, since At's strides made a zero-copy view impossible)? 0
```

### Discussion

The contiguous case matches the CUDA book's own "free reshape" claim exactly: both `r1` and `r2` share the identical `data_ptr()` as the original `data` tensor, confirming zero-copy behavior across two successive reshapes as long as every intermediate result stays contiguous. `At.view({2,3})` throwing a real error is `torch::Tensor`'s own honest refusal to do what the CUDA book's own naive reshape does silently -- rather than reinterpreting `At`'s non-contiguous strides as if they were a fresh, contiguous `(2,3)` buffer, `.view()` detects the incompatibility and fails loudly. `.reshape()`, asked to do the same thing, instead performs a real copy -- and the result, remarkably, reproduces the CUDA book's own exact wrong numbers (`[[1,4,2],[5,3,6]]`, mismatching original `A` at exactly 4 of 6 positions), even though the reason is different: `torch::Tensor`'s `.reshape()` copies `At`'s logical values in row-major reading order (`1,4,2,5,3,6`, since that is what `At` logically contains, read top-to-bottom, left-to-right), which happens to be numerically identical to the CUDA book's own raw-storage reinterpretation, because the CUDA book's own transpose had already physically written its output in that exact order.

> `[COMMON TRAP]` A reader might conclude from this section that `.reshape()` is "unsafe" in the same way the CUDA book's own naive reshape is, since both produce the identical wrong numbers here. This is not quite the right lesson: `.reshape()` is not buggy -- every value it produces is a mathematically well-defined function of `At`'s own actual logical contents (a genuine row-major flatten-then-reshape), and it never reads uninitialized or out-of-bounds memory the way a true bug would. The trap is conceptual, not a bug: reshaping a transposed tensor does not undo the transpose, because reshape and transpose answer different questions (how are these values grouped versus which index refers to which value) -- and a caller who calls `.reshape()` on a transposed tensor expecting to recover the pre-transpose matrix will be wrong regardless of which framework they're using, `torch::Tensor` included.

## 13.4 Diagonal and Triangular Structure: Real Techniques for Both `[FOUNDATIONAL]`

### Intuition

The CUDA book's own `DiagonalView` and `IdentityView` exploit structure so that multiplying by a diagonal matrix costs `O(n^2)` rather than the `O(n^3)` a general matmul would need, and its own hand-written forward substitution solves a lower-triangular system `Lx=b` directly, without computing a matrix inverse. `torch::Tensor` has no dedicated diagonal-matrix type, but the identical `O(n^2)` technique is directly expressible via Chapter 12's own broadcasting rule, and `torch::linalg_solve_triangular` is a real, already-implemented forward/backward substitution solver.

### Background

The CUDA book's own Worked Example 13.4.1: multiplication counts of `n=3: general=27, diagonal=9, ratio=3` and `n=5: general=125, diagonal=25, ratio=5`; `A x D` for `D=diag(2,5,10)` gives `[[2,10,30],[8,25,60],[14,40,90]]` using 9 multiplications. Its own Worked Example 13.4.2: forward substitution for `Lx=b` with `L=[[2,0],[3,4]]`, `b=[6,23]`, solving `x0=3.0, x1=3.5`, verified via `L @ x = [6.0, 23.0]`.

### Worked Example 13.4.1 -- the CUDA book's own diagonal multiplication, via broadcasting, and real forward substitution

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.4 hand-builds DiagonalView and
// IdentityView, exploiting structure so that A x D (D diagonal) costs
// O(n^2) multiplications instead of the O(n^3) a general matmul would need,
// and hand-writes forward substitution to solve a lower-triangular system
// Lx=b directly, without ever computing a matrix inverse. torch::Tensor has
// no dedicated DiagonalView type, but the identical O(n^2) technique is
// directly expressible via broadcasting: A * d (elementwise, with d
// broadcast across columns) computes exactly A @ diag(d) without ever
// materializing the full n x n diagonal matrix or performing any of the
// diagonal's own n^2-n zero multiplications. And torch::linalg_
// solve_triangular is a real, already-implemented, production forward/
// backward substitution solver -- this section verifies the CUDA book's
// own exact numbers for both, then confirms the broadcast technique
// produces bit-identical results to a genuine dense matmul against the
// materialized diagonal matrix, proving the shortcut is not an
// approximation.
int main() {
    // Worked Example 13.4.1: A x D where D = diag(2,5,10).
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
    torch::Tensor d = torch::tensor({2.0f, 5.0f, 10.0f});

    // The O(n^2) technique: broadcast-multiply A's columns by d directly,
    // never materializing the full diagonal matrix.
    torch::Tensor R_broadcast = A * d;
    std::cout << "A * d (broadcast, O(n^2) technique) =\n" << R_broadcast << std::endl;
    torch::Tensor expected = torch::tensor({{2.0f, 10.0f, 30.0f}, {8.0f, 25.0f, 60.0f}, {14.0f, 40.0f, 90.0f}});
    std::cout << "matches CUDA book's own [[2,10,30],[8,25,60],[14,40,90]]? "
              << torch::equal(R_broadcast, expected) << std::endl;

    // Cross-check: the broadcast shortcut against a genuine, full O(n^3)
    // dense matmul with D materialized via torch::diag -- confirming the
    // shortcut is not an approximation but bit-identical to doing the
    // full, wasteful computation.
    torch::Tensor D = torch::diag(d);
    torch::Tensor R_full = torch::matmul(A, D);
    std::cout << "\nfull dense matmul A @ diag(d) (genuine O(n^3), computing all 9 - 6 = "
              << "wasted zero-products too) =\n" << R_full << std::endl;
    std::cout << "broadcast result bit-identical to full dense matmul? "
              << torch::equal(R_broadcast, R_full) << std::endl;

    // Multiplication counting, per the CUDA book's own Worked Example
    // 13.4.1: n=3 general=27, diagonal=9, ratio=3; n=5 general=125,
    // diagonal=25, ratio=5.
    for (int64_t n : {3, 5}) {
        int64_t general = n * n * n;
        int64_t diagonal = n * n;
        std::cout << "\nn=" << n << ": general (O(n^3)) = " << general
                  << ", diagonal (O(n^2)) = " << diagonal
                  << ", ratio = " << (general / diagonal) << std::endl;
    }
    std::cout << "n=3 matches CUDA book's own general=27,diagonal=9,ratio=3? "
              << (27 == 3*3*3 && 9 == 3*3 && 27/9 == 3) << std::endl;
    std::cout << "n=5 matches CUDA book's own general=125,diagonal=25,ratio=5? "
              << (125 == 5*5*5 && 25 == 5*5 && 125/25 == 5) << std::endl;

    // Worked Example 13.4.2: forward substitution for Lx=b, L lower
    // triangular. The CUDA book hand-writes this from scratch;
    // torch::linalg_solve_triangular is real, already-implemented,
    // production forward/backward substitution.
    torch::Tensor L = torch::tensor({{2.0f, 0.0f}, {3.0f, 4.0f}});
    torch::Tensor b = torch::tensor({{6.0f}, {23.0f}});
    torch::Tensor x = torch::linalg_solve_triangular(L, b, /*upper=*/false);
    std::cout << "\nsolve_triangular(L,b,upper=false) x =\n" << x << std::endl;
    torch::Tensor x_expected = torch::tensor({{3.0f}, {3.5f}});
    std::cout << "matches CUDA book's own x0=3.0, x1=3.5? "
              << torch::allclose(x, x_expected, 1e-5) << std::endl;

    // Verification, matching the CUDA book's own: L @ x should reproduce b.
    torch::Tensor check = torch::matmul(L, x);
    std::cout << "L @ x =\n" << check << ", matches original b=[6,23]? "
              << torch::allclose(check, b, 1e-4) << std::endl;

    return 0;
}
```

Genuinely compiled and run in this book's environment:

```text
A * d (broadcast, O(n^2) technique) =
  2  10  30
  8  25  60
 14  40  90
[ CPUFloatType{3,3} ]
matches CUDA book's own [[2,10,30],[8,25,60],[14,40,90]]? 1

full dense matmul A @ diag(d) (genuine O(n^3), computing all 9 - 6 = wasted zero-products too) =
  2  10  30
  8  25  60
 14  40  90
[ CPUFloatType{3,3} ]
broadcast result bit-identical to full dense matmul? 1

n=3: general (O(n^3)) = 27, diagonal (O(n^2)) = 9, ratio = 3

n=5: general (O(n^3)) = 125, diagonal (O(n^2)) = 25, ratio = 5
n=3 matches CUDA book's own general=27,diagonal=9,ratio=3? 1
n=5 matches CUDA book's own general=125,diagonal=25,ratio=5? 1

solve_triangular(L,b,upper=false) x =
 3.0000
 3.5000
[ CPUFloatType{2,1} ]
matches CUDA book's own x0=3.0, x1=3.5? 1
L @ x =
  6
 23
[ CPUFloatType{2,1} ], matches original b=[6,23]? 1
```

Independently cross-checked via NumPy, computed with no dependence on `torch::Tensor` at all (using NumPy's own general `linalg.solve` rather than a dedicated triangular solver, as a structurally different solution method):

```text
triangular solve via full solve [3.  3.5]
diag scale [[ 2. 10. 30.]
 [ 8. 25. 60.]
 [14. 40. 90.]]
```

### Discussion

`A * d` matches the CUDA book's own exact diagonal-multiplication result, and the bit-identical comparison against a full dense `matmul(A, diag(d))` confirms the broadcast shortcut is not an approximation -- it computes precisely the same real numbers a full `O(n^3)` computation would, while genuinely never materializing the `n^2 - n` off-diagonal zero entries or performing any of the multiplications by them. This is the identical algorithmic insight the CUDA book's own `DiagonalView` captures, expressed through `torch::Tensor`'s own general-purpose broadcasting mechanism (already tested in Chapter 12) rather than a dedicated diagonal-matrix type. `torch::linalg_solve_triangular` reproduces the CUDA book's own forward-substitution answer exactly, and the `L @ x = b` verification closes the loop the same way the CUDA book's own Worked Example 13.4.2 does -- confirming the solution is correct by substituting it back into the original system, independent of trusting the solver's own internal correctness.

> `[COMMON TRAP]` It would be easy to assume `torch::linalg_solve_triangular` is doing something fundamentally different from the CUDA book's own hand-written forward substitution, since one is a single library call and the other is an explicit loop. Structurally, they solve the identical problem the identical way: forward substitution for a lower-triangular system computes each `x[i]` in order, using only the already-solved `x[0..i-1]` and the corresponding row of `L` -- there is no other correct general method for a triangular system that avoids computing a full matrix inverse. `torch::linalg_solve_triangular`'s real implementation is a production-grade version of exactly this algorithm, not a different one; the L @ x = b verification in this section confirms both produce a genuinely correct answer, independent of which one a reader trusts more by default.

## Complete Runnable Code

### File: `01_matmul.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.1 hand-writes scalar_matrix_multiply and
// a SIMD variant, computing each output position as a full dot product
// between a row of X and a column of M. Its own Worked Example 13.1.4 finds
// a genuine bug: neither function validates shape compatibility, so a
// caller passing a.cols=3 against b.rows=2 reads a sentinel value
// (999.0) placed just past the real data, producing a silently WRONG
// result ([[3004,3007],[6013,6022]]) rather than any error at all.
// torch::matmul is real, already-implemented, and -- unlike the CUDA
// book's own version -- performs a genuine shape check, throwing a real
// c10::Error on a dimension mismatch rather than reading past the end of
// a buffer. This file verifies the CUDA book's own exact worked numbers,
// then directly tests this honest divergence: this book's own
// matrix-multiply cannot silently produce a wrong-shape result at all.
int main() {
    // Worked Example 13.1.1: X (2x3) times M (3x2).
    torch::Tensor X = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor M = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});

    torch::Tensor Y = torch::matmul(X, M);
    std::cout << "X (2x3) @ M (3x2) =\n" << Y << std::endl;
    torch::Tensor expected = torch::tensor({{22.0f, 28.0f}, {49.0f, 64.0f}});
    std::cout << "matches CUDA book's own [[22,28],[49,64]]? "
              << torch::equal(Y, expected) << std::endl;

    // Worked Example 13.1.3: multiplication counts. torch::matmul's own
    // real cost is the same a.rows*a.cols*b.cols multiplications as the
    // CUDA book's own naive triple loop -- this is hand-counted here
    // rather than measured, since torch::matmul does not expose an
    // internal multiplication counter.
    int64_t count_2x3_3x2 = 2 * 3 * 2;
    int64_t count_3x3_3x3 = 3 * 3 * 3;
    int64_t count_64x64 = 64 * 64 * 64;
    std::cout << "\nhand-counted multiplications for 2x3 @ 3x2 = " << count_2x3_3x2
              << ", CUDA book's own expected = 12, match = " << (count_2x3_3x2 == 12) << std::endl;
    std::cout << "hand-counted multiplications for 3x3 @ 3x3 = " << count_3x3_3x3
              << ", CUDA book's own expected = 27, match = " << (count_3x3_3x3 == 27) << std::endl;
    std::cout << "hand-counted multiplications for 64x64 @ 64x64 = " << count_64x64
              << ", CUDA book's own expected = 262144, match = " << (count_64x64 == 262144) << std::endl;

    // Honest divergence: the CUDA book's own Worked Example 13.1.4 shows a
    // shape mismatch reading a sentinel value and producing a WRONG but
    // successfully-returned result. torch::matmul instead throws a real
    // c10::Error -- confirmed here by genuinely catching it, not by
    // reading documentation about what it should do.
    bool threw = false;
    std::string error_msg;
    try {
        torch::Tensor bad_b = torch::zeros({2, 2});  // b.rows=2, but X.cols=3 -- genuine mismatch
        torch::Tensor bad_result = torch::matmul(X, bad_b);
    } catch (const c10::Error& e) {
        threw = true;
        error_msg = e.what_without_backtrace();
    }
    std::cout << "\nmatmul(X [2x3], mismatched [2x2]) threw a real c10::Error (rather than silently "
              << "reading a sentinel value, the CUDA book's own bug)? " << threw << std::endl;
    std::cout << "error message contains 'size' or 'shape' or 'mat': "
              << (error_msg.find("size") != std::string::npos ||
                  error_msg.find("shape") != std::string::npos ||
                  error_msg.find("mat") != std::string::npos) << std::endl;

    return 0;
}
```

### File: `02_transpose.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.2 hand-writes matrix_transpose_simd:
// sequential reads from one row, scattered writes into DIFFERENT rows of a
// freshly-allocated output buffer -- real data movement, one element at a
// time (or one SIMD lane group at a time). torch::Tensor's own .t() (and
// .transpose(0,1)) does something structurally different: rather than
// moving any data, it returns a VIEW over the identical underlying
// storage, with swapped strides -- an honest divergence from the CUDA
// book's own genuine element-by-element scatter. This file verifies the
// CUDA book's own exact worked numbers, then directly demonstrates the
// zero-copy nature of torch::Tensor's transpose: the same real memory
// address before and after, confirmed via .data_ptr(), something the CUDA
// book's own scatter-based transpose could never do since it allocates an
// entirely new output buffer.
int main() {
    // Worked Example 13.2.1: transposing a 2x3 matrix.
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor At = A.t();
    std::cout << "A^T =\n" << At << std::endl;
    torch::Tensor expected = torch::tensor({{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}});
    std::cout << "matches CUDA book's own [[1,4],[2,5],[3,6]]? "
              << torch::equal(At, expected) << std::endl;

    // Honest divergence: the CUDA book's own transpose genuinely moves
    // every element into a new buffer (storage order [1,4,2,5,3,6] on a
    // freshly allocated output). torch::Tensor's .t() instead returns a
    // VIEW: the identical underlying storage, just reinterpreted via
    // swapped strides -- confirmed here by real pointer equality, not by
    // reading documentation about what a view "should" do.
    bool same_storage = (A.data_ptr() == At.data_ptr());
    std::cout << "\nA and A^T share the identical real memory address (data_ptr equal)? "
              << same_storage << std::endl;
    std::cout << "A^T.is_contiguous() = " << At.is_contiguous()
              << " (false is expected -- a transposed VIEW is not contiguous in row-major order, "
              << "unlike the CUDA book's own freshly-allocated, genuinely contiguous output buffer)" << std::endl;

    auto strides = At.strides();
    std::cout << "A^T strides = [" << strides[0] << "," << strides[1]
              << "], hand-derived expected = [1,3] (A's own row-major strides, simply swapped), match = "
              << (strides[0] == 1 && strides[1] == 3) << std::endl;

    // Despite being a view with no data movement, reading A^T still
    // produces the mathematically correct values at every position --
    // confirmed exhaustively here rather than only at the positions the
    // CUDA book's own Worked Example 13.2.1 happens to print.
    bool all_correct = true;
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 2; j++) {
            float via_transpose = At[i][j].item<float>();
            float via_original = A[j][i].item<float>();
            if (via_transpose != via_original) all_correct = false;
        }
    }
    std::cout << "every A^T[i][j] == A[j][i], read through the view with no data movement at all? "
              << all_correct << std::endl;

    // A real consequence of the view: writing through A^T changes A too,
    // since they are literally the same storage -- something impossible
    // for the CUDA book's own transpose, which allocates a separate buffer.
    torch::Tensor A2 = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    torch::Tensor A2t = A2.t();
    A2t[0][1] = 99.0f;
    std::cout << "\nafter writing A2t[0][1]=99: A2[1][0] = " << A2[1][0].item<float>()
              << " (also changed, since A2 and A2t share real storage -- impossible for the CUDA "
              << "book's own copying transpose)" << std::endl;

    return 0;
}
```

### File: `03_reshape_view.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.3 hand-builds View2D (a data pointer,
// rows, cols, row_stride) and a reshape() function: free (zero-copy) for a
// contiguous layout, but its own Worked Example 13.3.2 shows a genuine
// trap -- naively reshaping a transposed matrix's own storage, without
// accounting for the transpose, produces a result that looks plausible but
// mismatches the original matrix at 4 of 6 positions. torch::Tensor has
// two real, already-implemented functions covering this exact territory:
// .view() (always zero-copy, but throws a real error when the requested
// shape is not expressible as a view of the current strides) and
// .reshape() (falls back to a real copy when a view is not possible).
// This file verifies the CUDA book's own free-reshape case directly, then
// reproduces the CUDA book's own reshape TRAP almost exactly -- calling
// .reshape() on a transposed tensor produces the identical wrong numbers
// the CUDA book's own naive reshape produces, for a related but distinct
// reason (row-major logical flattening, not raw storage reinterpretation).
int main() {
    // Worked Example 13.3.1: a free (zero-copy) reshape of a contiguous
    // buffer, from (2,6) to (3,4).
    torch::Tensor data = torch::arange(12, torch::kFloat32);
    torch::Tensor r1 = data.view({2, 6});
    std::cout << "data.view({2,6}) =\n" << r1 << std::endl;
    std::cout << "r1 shares data.data_ptr() (zero-copy, CUDA book's own 'free reshape' claim)? "
              << (data.data_ptr() == r1.data_ptr()) << std::endl;

    torch::Tensor r2 = r1.reshape({3, 4});
    std::cout << "\nr1.reshape({3,4}) =\n" << r2 << std::endl;
    std::cout << "r2 ALSO shares the original data_ptr (still zero-copy, since r1 is contiguous)? "
              << (data.data_ptr() == r2.data_ptr()) << std::endl;

    // Honest divergence + parity: .view() is torch::Tensor's own
    // zero-copy-only reshape -- it genuinely throws a real error when the
    // requested shape cannot be expressed via the current tensor's
    // strides, rather than silently reinterpreting raw bytes the way the
    // CUDA book's own naive reshape does.
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    torch::Tensor At = A.t();  // a non-contiguous VIEW, per Section 13.2

    bool view_threw = false;
    try {
        torch::Tensor bad = At.view({2, 3});
    } catch (const c10::Error&) {
        view_threw = true;
    }
    std::cout << "\nAt.view({2,3}) on a non-contiguous transposed tensor threw a real error "
              << "(view refuses to silently reinterpret incompatible strides)? " << view_threw << std::endl;

    // .reshape() falls back to a real copy -- and the CUDA book's own
    // Worked Example 13.3.2 trap reproduces almost exactly: reshaping the
    // transposed matrix's own values (in row-major logical order) as (2,3)
    // produces [[1,4,2],[5,3,6]], NOT the original A -- mismatching it at
    // 4 of 6 positions, the CUDA book's own exact finding, even though the
    // underlying mechanism differs (torch::Tensor reshapes the tensor's
    // logical row-major values; the CUDA book's own naive reshape
    // reinterprets raw storage bytes directly -- both wrong for the same
    // underlying reason: a transpose is not undone by reshaping alone).
    torch::Tensor reshaped = At.reshape({2, 3});
    std::cout << "\nAt.reshape({2,3}) =\n" << reshaped << std::endl;
    torch::Tensor trap_expected = torch::tensor({{1.0f, 4.0f, 2.0f}, {5.0f, 3.0f, 6.0f}});
    std::cout << "matches CUDA book's own naive-reshape trap result [[1,4,2],[5,3,6]]? "
              << torch::equal(reshaped, trap_expected) << std::endl;

    int mismatches = 0;
    for (int64_t i = 0; i < 2; i++)
        for (int64_t j = 0; j < 3; j++)
            if (reshaped[i][j].item<float>() != A[i][j].item<float>()) mismatches++;
    std::cout << "mismatches against original A [[1,2,3],[4,5,6]] = " << mismatches
              << ", CUDA book's own expected = 4 (of 6 positions), match = " << (mismatches == 4) << std::endl;

    std::cout << "reshaped shares At's data_ptr (a real copy was made, since At's strides "
              << "made a zero-copy view impossible)? " << (At.data_ptr() == reshaped.data_ptr()) << std::endl;

    return 0;
}
```

### File: `04_diagonal_triangular.cpp`

```cpp
#include <torch/torch.h>
#include <iostream>

// The CUDA C++ edition's Section 13.4 hand-builds DiagonalView and
// IdentityView, exploiting structure so that A x D (D diagonal) costs
// O(n^2) multiplications instead of the O(n^3) a general matmul would need,
// and hand-writes forward substitution to solve a lower-triangular system
// Lx=b directly, without ever computing a matrix inverse. torch::Tensor has
// no dedicated DiagonalView type, but the identical O(n^2) technique is
// directly expressible via broadcasting: A * d (elementwise, with d
// broadcast across columns) computes exactly A @ diag(d) without ever
// materializing the full n x n diagonal matrix or performing any of the
// diagonal's own n^2-n zero multiplications. And torch::linalg_
// solve_triangular is a real, already-implemented, production forward/
// backward substitution solver -- this section verifies the CUDA book's
// own exact numbers for both, then confirms the broadcast technique
// produces bit-identical results to a genuine dense matmul against the
// materialized diagonal matrix, proving the shortcut is not an
// approximation.
int main() {
    // Worked Example 13.4.1: A x D where D = diag(2,5,10).
    torch::Tensor A = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
    torch::Tensor d = torch::tensor({2.0f, 5.0f, 10.0f});

    // The O(n^2) technique: broadcast-multiply A's columns by d directly,
    // never materializing the full diagonal matrix.
    torch::Tensor R_broadcast = A * d;
    std::cout << "A * d (broadcast, O(n^2) technique) =\n" << R_broadcast << std::endl;
    torch::Tensor expected = torch::tensor({{2.0f, 10.0f, 30.0f}, {8.0f, 25.0f, 60.0f}, {14.0f, 40.0f, 90.0f}});
    std::cout << "matches CUDA book's own [[2,10,30],[8,25,60],[14,40,90]]? "
              << torch::equal(R_broadcast, expected) << std::endl;

    // Cross-check: the broadcast shortcut against a genuine, full O(n^3)
    // dense matmul with D materialized via torch::diag -- confirming the
    // shortcut is not an approximation but bit-identical to doing the
    // full, wasteful computation.
    torch::Tensor D = torch::diag(d);
    torch::Tensor R_full = torch::matmul(A, D);
    std::cout << "\nfull dense matmul A @ diag(d) (genuine O(n^3), computing all 9 - 6 = "
              << "wasted zero-products too) =\n" << R_full << std::endl;
    std::cout << "broadcast result bit-identical to full dense matmul? "
              << torch::equal(R_broadcast, R_full) << std::endl;

    // Multiplication counting, per the CUDA book's own Worked Example
    // 13.4.1: n=3 general=27, diagonal=9, ratio=3; n=5 general=125,
    // diagonal=25, ratio=5.
    for (int64_t n : {3, 5}) {
        int64_t general = n * n * n;
        int64_t diagonal = n * n;
        std::cout << "\nn=" << n << ": general (O(n^3)) = " << general
                  << ", diagonal (O(n^2)) = " << diagonal
                  << ", ratio = " << (general / diagonal) << std::endl;
    }
    std::cout << "n=3 matches CUDA book's own general=27,diagonal=9,ratio=3? "
              << (27 == 3*3*3 && 9 == 3*3 && 27/9 == 3) << std::endl;
    std::cout << "n=5 matches CUDA book's own general=125,diagonal=25,ratio=5? "
              << (125 == 5*5*5 && 25 == 5*5 && 125/25 == 5) << std::endl;

    // Worked Example 13.4.2: forward substitution for Lx=b, L lower
    // triangular. The CUDA book hand-writes this from scratch;
    // torch::linalg_solve_triangular is real, already-implemented,
    // production forward/backward substitution.
    torch::Tensor L = torch::tensor({{2.0f, 0.0f}, {3.0f, 4.0f}});
    torch::Tensor b = torch::tensor({{6.0f}, {23.0f}});
    torch::Tensor x = torch::linalg_solve_triangular(L, b, /*upper=*/false);
    std::cout << "\nsolve_triangular(L,b,upper=false) x =\n" << x << std::endl;
    torch::Tensor x_expected = torch::tensor({{3.0f}, {3.5f}});
    std::cout << "matches CUDA book's own x0=3.0, x1=3.5? "
              << torch::allclose(x, x_expected, 1e-5) << std::endl;

    // Verification, matching the CUDA book's own: L @ x should reproduce b.
    torch::Tensor check = torch::matmul(L, x);
    std::cout << "L @ x =\n" << check << ", matches original b=[6,23]? "
              << torch::allclose(check, b, 1e-4) << std::endl;

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

`torch::matmul` reproduced the CUDA book's own `[[22,28],[49,64]]` exactly, and its real shape check was confirmed by genuinely catching a `c10::Error` on a mismatched call -- the direct fix for the CUDA book's own Worked Example 13.1.4 sentinel-value bug. `torch::Tensor`'s own `.t()` was confirmed to be a genuine zero-copy view via real pointer equality on `.data_ptr()`, contrasted with the CUDA book's own data-moving transpose through a real write-through test showing a mutation to the view changes the original tensor too. `.view()` and `.reshape()` were confirmed to split the CUDA book's own single `reshape()` function into two real behaviors -- a strict zero-copy-only version that throws on incompatibility, and a copying fallback whose result, on a transposed tensor, reproduced the CUDA book's own exact reshape trap numbers for a related but distinct reason. And the CUDA book's own diagonal-multiplication technique was confirmed expressible via ordinary broadcasting, bit-identical to a full dense matmul, while `torch::linalg_solve_triangular` reproduced the CUDA book's own forward-substitution answer exactly, verified the same way the CUDA book verifies its own: by substituting the solution back into the original system.

## Self-Check Questions

1. Section 13.1's own shape-mismatch test catches a `c10::Error` rather than checking a return value or an error code. What would have to be true about `torch::matmul`'s own implementation for a caller to instead receive a plausible-looking but wrong result, the way the CUDA book's own Worked Example 13.1.4 does?
2. Section 13.2's write-through test (`A2t[0][1]=99` changing `A2[1][0]`) would NOT reproduce if `A2t` had been constructed via `.contiguous()` immediately after `.t()`. Explain why, referring to what `.contiguous()` actually does to a non-contiguous view.
3. Section 13.3 shows `.reshape()` producing the CUDA book's own exact trap numbers, but for a "related but distinct" reason from the CUDA book's own raw-storage reinterpretation. In your own words, what is the actual difference between "reshaping logical row-major values" and "reinterpreting raw storage bytes," even though they agree in this specific example?
4. Section 13.4's broadcast technique (`A * d`) and the full dense matmul (`A @ diag(d)`) are confirmed bit-identical. Given that they are mathematically equivalent, what is the actual advantage of the broadcast version, since both produce the exact same 9 numbers here?
5. Section 13.4's forward-substitution verification computes `L @ x` and compares it to the original `b`. Why is this a meaningful check of `torch::linalg_solve_triangular`'s correctness even though it uses `torch::matmul` (tested independently in Section 13.1) rather than some entirely separate verification method?

## Where We Go Next

This chapter verified `torch::matmul`, `.t()`, `.view()`/`.reshape()`, and structure-aware diagonal and triangular operations against the CUDA book's own exact worked numbers, finding a real shape check where the CUDA book has none, a real zero-copy view where the CUDA book genuinely moves data, and -- in two cases -- the CUDA book's own exact trap or technique reappearing for a related but distinct underlying reason. Chapter 14 turns from operations that produce a tensor of the same or related shape to reduction operations, which collapse a tensor down to fewer values entirely -- testing `torch::sum`, `torch::mean`, and friends against the CUDA book's own hand-written parallel reduction kernels and their own classic numerical-stability findings.

## Worked Solutions

**1.** For `torch::matmul` to return a plausible-but-wrong result instead of throwing, its own internal implementation would have to skip validating that the inner dimensions agree (`X.size(1) == M.size(0)`) and instead proceed directly into the dot-product loop, reading whatever memory happens to sit at the computed offset once the loop runs past the smaller operand's real data -- exactly the CUDA book's own `b.get(2,0)` reading a sentinel value. `torch::matmul`'s real implementation performs this shape check as one of its very first steps, before any arithmetic begins, which is precisely why the mismatched call in this section throws immediately rather than producing any output at all.

**2.** `.contiguous()` forces a genuine copy whenever its receiver is not already contiguous -- and `.t()`'s own output is never contiguous (Section 13.2's own test confirms `is_contiguous()` is `false`), so calling `.contiguous()` on `A2t` immediately after `.t()` would allocate a fresh, separate buffer and copy `A2t`'s logical values into it in proper row-major order. The resulting tensor would share nothing with `A2`'s own storage, so writing to it afterward would have no effect on `A2` at all -- `.contiguous()` is precisely the tool for regaining the independence a zero-copy view like `.t()` gives up, at the cost of a real, explicit data copy.

**3.** "Reshaping logical row-major values" means torch::Tensor first determines what `At`'s values ARE, read in standard row-major order according to `At`'s own shape and strides (which correctly account for the transpose, since indexing `At[i][j]` always returns the mathematically correct value `A[j][i]`), and only then repackages that already-correct sequence of values into the new shape. "Reinterpreting raw storage bytes" means treating the underlying memory buffer's physical byte order as if it were already in the new shape's row-major order, with no reference to what shape or strides originally described that memory. They agree in this specific example only because `At`'s own storage happens to have been laid out, by the CUDA book's own real transpose, in the exact byte order that also equals `At`'s own logical row-major reading order -- a coincidence specific to how the CUDA book's own transpose physically wrote its output, not a general rule that would hold for an arbitrary non-contiguous tensor.

**4.** Even though both approaches produce the identical 9 numbers here, the broadcast version (`A * d`) never allocates or computes the `n x n` diagonal matrix `D` at all -- no `torch::diag(d)` call, no `n^2` entries stored, and no `n^2 - n` wasted multiplications by the zero off-diagonal entries the full matmul path genuinely performs. For a small `3x3` example the difference is invisible in practice, but the advantage scales directly with `n`: at `n=1000`, the broadcast version does `1,000,000` real multiplications while a full dense matmul against a materialized diagonal matrix would perform `1,000,000,000`, the overwhelming majority of them multiplying by a zero that was never going to contribute anything to the result.

**5.** `L @ x = b` is a meaningful check specifically because it tests a different mathematical DIRECTION than `solve_triangular` itself computed: `solve_triangular` solves for the unknown `x` given `L` and `b`, while `matmul` computes a known product given two known operands (`L` and the just-solved `x`) -- reusing `matmul` here doesn't validate `solve_triangular`'s own internal algorithm circularly, because forward substitution and matrix multiplication are structurally unrelated operations that happen to be inverses of each other for this specific relationship. If `solve_triangular` had a genuine bug and returned a wrong `x`, `L @ x` would almost certainly NOT reproduce the original `b` (since an incorrect `x` has no particular reason to satisfy the original equation), making this an honest end-to-end check of correctness rather than a test that could pass merely because both functions share some common underlying mistake.
