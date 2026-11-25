# CudaMath

CudaMath is a Windows math workspace for the NVIDIA N1X. A C++ engine owns the data and calls CUDA for the math. The user types a MATLAB-style language in a REPL. Two apps share that engine: a console app and a WinUI 3 split-pane app.

The engine is a front end. Dense linear algebra goes to cuBLAS and cuSOLVER. Random arrays come from cuRAND. Elementwise and reduction math goes through small kernels that call the CUDA math API. The project does not reimplement those libraries.

## Platform

- Windows on ARM64, NVIDIA N1X only.
- Unified memory only. Every numeric array is allocated with `cudaMallocManaged`.
- Host and device use the same pointers. There is no explicit host/device copy of workspace data.
- One CUDA context per process, owned by the engine.

Before either app opens a REPL, the engine checks that this process can use CUDA unified memory. The CUDA runtime is loaded dynamically, so a missing driver DLL takes the same path as a failed check. The check passes only when all of these are true:

- The CUDA runtime initializes.
- At least one CUDA device is present.
- That device reports `cudaDevAttrManagedMemory`.
- That device reports `cudaDevAttrConcurrentManagedAccess`, so host and device can use one `cudaMallocManaged` pointer without an explicit copy.

If any step fails, the engine returns a short error and does not create a context. The text says that CudaMath requires an NVIDIA GPU with CUDA unified memory, and it includes the CUDA error string when the runtime provided one. There is no CPU fallback.

## Language and build

The implementation language is C++20. CUDA C++ is used for the elementwise kernels and for the cuBLAS, cuSOLVER, and cuRAND calls. The GUI is C++/WinRT (WinUI 3).

Tool chain on this machine:

- Visual Studio 2026, MSVC toolset v145
- Windows SDK 10.0.28000.0
- CUDA Toolkit 13.4
- NVCC
- Platform ARM64 only

`CudaMath.sln` contains four projects. The console app and the GUI both link the same static libraries.

| Project | Output | Role |
|---|---|---|
| CudaMathEngine | static lib | Values, unified memory, CUDA calls, random generator, workspace, saveworkspace/loadworkspace |
| CudaMathRepl | static lib | MATLAB-style parser and interpreter |
| CudaMathCli | console exe | REPL in a terminal |
| CudaMathGui | WinUI 3 exe | REPL pane plus a pane of variables and functions |

## Numeric model

Version 1 stores dense 2-D arrays in column-major order, including scalars as 1-by-1 arrays. Vectors are column vectors unless the user builds a row.

Types:

| Workspace type | MATLAB name | CUDA BLAS/SOLVER prefix |
|---|---|---|
| float64 | `double` | `D` |
| float32 | `single` | `S` |
| int32 | `int32` | none |
| int64 | `int64` | none |

A bare numeric literal is `double`, as in MATLAB. `single`, `int32`, and `int64` are the conversion functions. There is no complex type in version 1, and no logical or bool type.

Integer arrays stay integer for `+`, `-`, `.*`, `./`, `abs`, `sign`, `min`, `max`, `sum`, and `prod`. Comparisons and the operators `&`, `|`, and `~` return `double` `1` or `0`. Transcendental functions promote the input to `double`. cuBLAS and cuSOLVER entry points require floating types; an integer matrix passed to those functions is promoted to `double` and the result is `double`.

## REPL language

The interpreter stores each user function as source text and evaluates it on the host. Array operations inside that evaluation call the engine. User functions are not compiled into CUDA kernels.

Statements:

- Expression, with `;` suppressing the printed result
- Assignment, including multiple outputs: `[L,U,P] = lu(A)`
- A bare expression result is stored in `ans`
- `function` / `end` definitions, with zero or more output names and zero or more arguments
- `if` / `elseif` / `else` / `end`
- `for` / `end` and `while` / `end`
- `test 'name'` / `end` blocks, which group `expect` checks
- `%` comments

`if` and `while` do not use a bool type. A condition is true when the value is nonempty and every element is nonzero. An empty array is false. `&&` and `||` apply that test to scalars and stop at the first decisive operand. A `for` or `while` runs until its normal end. Version 1 has no `break`, `continue`, `return`, or `switch`, and a function returns by assigning its output names.

Expressions:

- `+` `-` `.*` `./` `.^` `*` `/` `\` `'` (transpose) and elementwise comparisons
- Matrix literals `[1 2; 3 4]`, horizontal and vertical concatenation
- Colon `a:b` and `a:step:b`
- Indexing `A(i,j)` and `A(i,:)` with `end`
- Parentheses for grouping and for function calls

Names that are not variables resolve to built-ins, then to user functions.

## Built-in functions

### Workspace

These run on the host. `saveworkspace` and `loadworkspace` are the serialization surface.

| Function | Behavior |
|---|---|
| `who` | List defined names |
| `whos` | Name, type, and size of each variable, plus each user function |
| `clear` | Remove one name, or the whole workspace when called with no arguments |
| `clear functions` | Remove user functions and keep variables |
| `saveworkspace path` | Write the workspace document, including variables, user functions, and the cuRAND seed |
| `loadworkspace path` | Replace the workspace from a document |
| `clc` | Clear the REPL output |
| `expect(expected, actual)` | Compare `actual` with `expected` and record a pass or a fail. See Tests. |
| `expect(expected, actual, tol)` | Same comparison with an explicit absolute tolerance |
| `runtests path` | Run a test file in a scratch workspace and print the summary |
| `timer on` | Start reporting elapsed time after each REPL submission |
| `timer off` | Stop reporting elapsed time |
| `timer` | Print `timer on` or `timer off` |
| `help` | Print commands, types, and built-in functions. See the console app. |
| `cudaversion` | Return the loaded CUDA runtime version from the engine, such as `13.4`. The value is a string in `ans`. |
| `memory` | Print the three byte counts below, each with its explanation. `ans` is a 3-by-1 `double` of workspace, CUDA total, and CUDA free. |

`memory` prints three labeled lines. The sentence after each number is part of the command output:

```
Workspace   12582912 bytes
    Bytes stored in the variables defined right now. This is the sum of the arrays the engine allocated for this workspace.

CUDA total  17179869184 bytes
    Size of the unified memory pool the CUDA driver reports. On this machine that pool is shared with Windows. It is not a separate graphics card.

CUDA free   8589934592 bytes
    Bytes in that pool the driver currently reports as free for new CUDA allocations. A managed array is counted here only once its pages are committed, so free memory can be larger than total minus the workspace.
```

### Shape and construction

These allocate unified memory and fill it on the host. They do not call cuBLAS.

| Function | Behavior |
|---|---|
| `zeros(m,n)` | M-by-n array of zeros. One argument makes a square matrix. Type follows `like` when given, otherwise `double`. |
| `ones(m,n)` | Same shape rules as `zeros`, filled with ones |
| `eye(m,n)` | Identity, same shape rules |
| `diag(v)` | Diagonal matrix from a vector, or the diagonal vector of a matrix |
| `linspace(a,b,n)` | Row vector of n samples from a to b. n defaults to 100 |
| `colon` via `a:step:b` | Row vector |
| `repmat(A,m,n)` | Tiled copy |
| `reshape(A,m,n)` | Same elements, new 2-D shape, column-major |
| `numel(A)` | Element count |
| `size(A)` | 1-by-2 size vector. `size(A,dim)` returns one dimension |
| `length(A)` | Larger dimension |
| `ndims(A)` | Always 2 in version 1 |
| `isempty(A)` | True when either dimension is 0 |
| `class(A)` | `'double'`, `'single'`, `'int32'`, or `'int64'` |
| `double(A)` `single(A)` `int32(A)` `int64(A)` | Convert, truncating toward zero for the integer conversions |
| `horzcat` `vertcat` | Concatenation, also spelled `[A B]` and `[A; B]` |
| `tril(A)` `triu(A)` | Lower or upper triangle. Optional second argument is the diagonal offset |

### cuRAND

The engine keeps one cuRAND Philox generator for the process. `rng` sets its seed. `rand`, `randn`, and `randi` allocate a unified-memory array and fill it on the device. Shape rules match `zeros`: one size argument builds a square matrix, so `rand(n,1)` is a column vector and `rand(1,n)` is a row vector. The default type is `double`. A trailing `'single'` or `'double'` selects the floating type.

These sequences are not bit-identical to MATLAB. MATLAB's default generator is Mersenne Twister; CudaMath uses cuRAND Philox.

| Function | Library call | Behavior |
|---|---|---|
| `rng(seed)` | `curandSetPseudoRandomGeneratorSeed` | Seed the workspace generator. `seed` is a nonnegative integer. `rng` with no arguments returns the current seed. |
| `rand(m,n)` | `curandGenerateUniform` or `curandGenerateUniformDouble` | Uniform on (0, 1) |
| `randn(m,n)` | `curandGenerateNormal` or `curandGenerateNormalDouble` | Standard normal, mean 0, variance 1 |
| `randi(imax,m,n)` | uniform draw, then scale | Integers from 1 through `imax`, MATLAB's default range. `randi([imin imax],m,n)` uses that closed interval. The result is `double` unless the call ends with `'single'`, `'int32'`, or `'int64'`. |

`saveworkspace` includes the generator seed and the count of values already drawn in the workspace document. `loadworkspace` recreates the generator and skips that many values, so later `rand` calls continue the saved stream. There is no separate seed file.

### Elementwise math

Each of these is a small CUDA kernel over unified memory. The kernel calls the matching CUDA math function (`sqrt`, `sin`, and so on) for floating types.

| Function | Notes |
|---|---|
| `abs` | Integer and floating |
| `sign` | Integer and floating |
| `sqrt` | Promotes integers to `double` |
| `exp` `log` `log2` `log10` | Promotes integers to `double` |
| `sin` `cos` `tan` | Radians. Promotes integers to `double` |
| `asin` `acos` `atan` `atan2` | Promotes integers to `double` |
| `sinh` `cosh` `tanh` | Promotes integers to `double` |
| `floor` `ceil` `round` `fix` | `fix` truncates toward zero. Floating inputs stay floating |
| `mod(a,b)` `rem(a,b)` | MATLAB remainder rules |
| `.^` via `power(a,b)` | Promotes integers to `double` |

Operators `+`, `-`, `.*`, and `./` use the same kind of kernel. `./` on integers follows MATLAB and produces a floating result.

### Reductions

Reductions run as CUDA kernels. `norm` on vectors uses cuBLAS `nrm2`.

| Function | Behavior |
|---|---|
| `sum(A)` | Sum down columns. `sum(A,dim)` selects the dimension. `sum(A,'all')` sums every element |
| `prod(A)` | Same shape rules as `sum` |
| `mean(A)` | `sum` divided by the length of that dimension. Result is floating |
| `min(A)` `max(A)` | Column reductions, or `min(A,[],dim)`. Two-argument `min(A,B)` is elementwise |
| `any(A)` `all(A)` | Nonzero tests, same shape rules as `sum` |
| `norm(A)` | Vector 2-norm via cuBLAS. Matrix `norm(A)` is the largest singular value via cuSOLVER. `norm(A,1)`, `norm(A,inf)`, and `norm(A,'fro')` match MATLAB |
| `dot(a,b)` | Dot product of two vectors via cuBLAS |
| `trace(A)` | Sum of the diagonal |

### cuBLAS

| Function | Library call | MATLAB form |
|---|---|---|
| `mtimes` | `cublas<t>gemm` | `A * B` |
| scalar `*` | `cublas<t>scal` or a kernel when both sides are arrays of one scale | `s * A` |
| `cross(a,b)` | Kernel over the 3-vectors, after a size check | `cross(a,b)` |

`<t>` is `S` or `D`.

### cuSOLVER

Inputs are promoted to `single` or `double` as described above. Multiple outputs use the MATLAB assignment form.

| Function | Library call | Returns |
|---|---|---|
| `mldivide` | `getrf` + `getrs`, or `potrf` + `potrs` when the user writes `chol` form is not required; general `\` uses LU | `A \ B` |
| `mrdivide` | Solved as `(B' \ A')'` | `B / A` |
| `inv(A)` | LU, then solve against the identity | One matrix |
| `det(A)` | Product of LU diagonals, with pivot sign | Scalar |
| `rank(A)` | Singular values above a MATLAB-style tolerance | Scalar |
| `lu(A)` | `getrf` | `[L,U,P]`. A single output returns the packed LU |
| `qr(A)` | `geqrf` and `orgqr` | `[Q,R]` |
| `chol(A)` | `potrf` | Upper triangle `R` such that `R' * R = A` |
| `svd(A)` | `gesvd` | `[U,S,V]` with `S` a diagonal matrix. One output returns the singular-value vector |
| `eig(A)` | `syevd` for symmetric A, `geev` otherwise | `[V,D]` with eigenvalues on the diagonal of `D`. One output returns the eigenvalue vector |
| `transpose` | No solver call. A kernel or a metadata swap when the next consumer can read the transpose as an op | `A'` and `A.'` are the same for real arrays |

`chol` reports an error when the matrix is not positive definite. `eig` on a nonsymmetric matrix returns real eigenvalues only when they are real; a complex eigenvalue is an error in version 1 because there is no complex type.

## Tests

The test suite is a set of REPL scripts. A check names the value it expects, evaluates an expression, and prints a pass or a fail. The suite guards parsing, array shape, type rules, dispatch into cuBLAS, cuSOLVER, and cuRAND, and saveworkspace/loadworkspace. It does not try to certify the CUDA libraries themselves. A few reconstruction checks are enough to show that a call was wired to the right library.

```
test 'lu reconstructs'
  A = [1 2 3; 4 5 6; 7 8 10];
  [L,U,P] = lu(A);
  expect(P * A, L * U, 1e-8);
end
```

`expect` takes the expected value first and the expression under test second. The optional third argument is an absolute tolerance.

| Outcome | Rule |
|---|---|
| Pass | Class and size match, and every element matches |
| Fail | Class or size differs, an element differs, or the expression throws |

Integer values match exactly. Floating values match when `max(abs(actual - expected))` is within tolerance. The default tolerance is `1e-8` when either side is `double`, and `1e-5` when the wider floating type is `single`. A thrown error inside `expect` or inside a `test` block is a failure that names the error. The REPL stays open.

Each check prints one line:

```
PASS  lu reconstructs
FAIL  lu reconstructs: max abs diff 1.4e-7, tol 1e-8
```

The name on that line is the enclosing `test` block. An `expect` outside any block is named `expect`.

`runtests 'tests/smoke.cm'` is the suite runner.

- It snapshots the workspace and the cuRAND generator, then runs the file in an empty workspace.
- It restores both after the file finishes, whether the suite passed or failed.
- It prints one line per check, then `N passed, M failed`.
- After the restore, `ans` is the number of failures.

`CudaMathCli --runtests tests/smoke.cm` runs that command and exits. The process exit code is 0 when every check passed and 1 otherwise. The GUI has a Run tests action that sends the same command to the REPL and shows the summary in the REPL pane.

`tests/smoke.cm` is the version 1 suite. It covers:

- `zeros`, `ones`, `eye`, `reshape`, and `size`
- `int32` addition, exactly
- `rng`, then `rand` and `randn` shape, class, and range
- `randi` bounds
- `A \ A` and `inv(eye(n))` near `eye(n)` for a small well-conditioned matrix
- `lu`, `chol`, and `svd` reconstruction within tolerance
- a user `function` called from a test
- `saveworkspace` and `loadworkspace` of one variable and one function

## Timing

`timer on` and `timer off` switch a session flag. It starts off. The flag is not a workspace variable, and `saveworkspace` / `loadworkspace` do not change it.

While the flag is on, the REPL times each submission. The clock starts when evaluation of that submission begins and stops after `cudaDeviceSynchronize`, so the number includes cuBLAS, cuSOLVER, cuRAND, and kernel time rather than only the launch. Printing the result is outside the clock. A submission that ends with `;` still prints the elapsed line. A submission that throws prints the error, then the elapsed line.

The report is one line after the result:

```
Elapsed 12.345 ms
```

`runtests` is one submission, so the suite gets one elapsed line for the whole file. Statements inside a `test` block or a user function do not print their own times. `timer on` and `timer off` themselves are not timed.

## Serialization

`saveworkspace` writes one workspace document. `loadworkspace` replaces the current workspace.

The document records:

- A format version
- Each variable: name, type, row count, column count, and values in column-major order
- Each user function: name and source text
- The cuRAND seed and the number of random values already drawn

Small documents are JSON. Array values larger than 1 MiB are stored in a binary sidecar next to the JSON, named from the document, and the JSON holds the sidecar offset and length. `loadworkspace` allocates new unified memory and copies the values in. Function source is kept as text and parsed again on the next call.

## Console app

`CudaMathCli` runs the unified-memory check before it reads any input. On failure it prints the error to standard error and exits with code 1. It does not prompt. On success it reads a line, sends it to the interpreter, and prints the result when the line does not end with `;`. Errors print the message and the REPL continues. `exit` or `quit` leaves the process. `CudaMathCli --runtests <file>` runs the same startup check, then runs that test file and exits with code 0 when every check passed and code 1 when any check failed. A failed startup check exits with code 1 before any test runs.

`help` prints three sections and returns to the prompt:

- **Commands.** One line each for `help`, `who`, `whos`, `clear`, `clear functions`, `saveworkspace`, `loadworkspace`, `clc`, `expect`, `runtests`, `timer on`, `timer off`, `timer`, `cudaversion`, `memory`, `exit`, and `quit`.
- **Types.** `double`, `single`, `int32`, and `int64`. A bare literal is `double`. Values are dense 2-D column-major arrays, and a scalar is 1-by-1.
- **Functions.** One line each for the built-ins in this spec, in the same groups as the GUI help: workspace, shape, cuRAND, elementwise math, reductions, cuBLAS, and cuSOLVER.

The Types and Functions text is the same text the GUI Help window uses.

## GUI app

`CudaMathGui` runs the unified-memory check before it creates the main window. On failure it shows the error in a dialog, and the process exits when the dialog is dismissed. It does not open an empty workspace. On success it shows a WinUI 3 window with a menu bar and two panes:

- One pane is the REPL: input, history, and printed results
- The other pane lists variables and user functions from the live workspace

The variable list shows name, class, and size. Selecting a function shows its source. The list refreshes after each successful statement. The REPL commands `saveworkspace`, `loadworkspace`, `runtests`, `timer on`, and `timer off` stay available. The menu calls those same engine operations.

### Status bar

The bottom of the window is a status bar with four fields, left to right. It refreshes after each successful statement, after `saveworkspace` and `loadworkspace`, and when the timer flag changes.

| Field | Visible text |
|---|---|
| Workspace | File name, or `Untitled` when the window has no path. A trailing `*` means unsaved changes. |
| Timer | `Timer on` or `Timer off`. When the timer is on, the last submission's elapsed time stays in this field, for example `Timer on  12.345 ms`. Clicking this field sends `timer off` when the timer is on, and `timer on` when it is off. |
| Memory | `Workspace used 12.0 MB    CUDA Free 8192.0 MB` |
| CUDA | The runtime version string that `cudaversion` returns, for example `CUDA 13.4`. |

The memory labels are `Workspace used` and `CUDA Free`, each followed by a number in megabytes of 1024×1024 bytes, with one decimal place. Those are the workspace-byte and CUDA-free counts from `memory`, scaled into MB. CUDA total stays on the `memory` command.

### File

| Item | Action |
|---|---|
| Open Workspace | Pick a workspace document and `loadworkspace` it, replacing the current variables, user functions, and cuRAND stream. The chosen path becomes the current workspace path. |
| Save Workspace | `saveworkspace` to the current workspace path. If the window has no current path yet, ask for one first. |
| Exit | Close the window and leave the process. |

Open Workspace and Exit ask for confirmation when the session has changes that have not been saved. Cancel leaves the session as it is. A successful save clears that dirty flag. Closing the window with the title-bar button takes the same path as Exit. REPL `loadworkspace` and `saveworkspace` update the same current path and dirty flag.

### Run

| Item | Action |
|---|---|
| Run Tests | Execute `runtests 'tests/smoke.cm'` and print the pass and fail lines in the REPL pane. |
| Timer | Checkable item. Checked sends `timer on`. Unchecked sends `timer off`. The check mark follows the session flag, including after a click on the status-bar timer field. |

### Help

| Item | Action |
|---|---|
| Help | Open the help window. |
| About | Open the about dialog. |

The help window is a short document stored in the app, with three sections:

- **App.** What CudaMath is, the REPL, the variable pane, the timer, tests, and workspace files.
- **Types.** `double`, `single`, `int32`, and `int64`. A bare literal is `double`. Values are dense 2-D column-major arrays, and a scalar is 1-by-1.
- **Functions.** One line each for the built-ins in this spec, in the same groups: workspace, shape, cuRAND, elementwise math, reductions, cuBLAS, and cuSOLVER. Include `expect`, `runtests`, `timer`, `help`, `cudaversion`, and `memory`. The Types and Functions text is shared with the console `help` command.

About shows the application name, the version, the line "Windows ARM64 for the NVIDIA N1X", the CUDA toolkit version the app was built with, and a one-line description: a MATLAB-style REPL that calls the CUDA math libraries.

## Version 1 boundaries

In version 1 the engine does not provide:

- N-D arrays beyond matrices
- Complex numbers
- A logical or bool type, and logical indexing such as `A(A > 0)`
- Sparse matrices
- cuFFT
- GPU compilation of user functions
- Toolboxes beyond the built-ins in this spec
