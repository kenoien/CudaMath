# CudaMath

CudaMath is a Windows math workspace for the NVIDIA N1X. A C++ engine owns the data and calls CUDA for the math. The user types a MATLAB-style language in a REPL. Two apps share that engine: a console app with one session, and a WinUI 3 window with one document window per session.

The engine is a front end. Dense linear algebra goes to cuBLAS and cuSOLVER. Random arrays come from cuRAND. Elementwise and reduction math goes through small kernels that call the CUDA math API. The project does not reimplement those libraries.

## Platform

- Windows on ARM64, NVIDIA N1X only.
- Unified memory only. Every numeric array is allocated with `cudaMallocManaged`.
- Host and device use the same pointers. There is no explicit host/device copy of workspace data.
- One CUDA context per process, owned by the engine. A session never creates its own context.
- The GUI holds one session per document window. The console app holds one session. Sessions do not share variables, user functions, or cuRAND state.

Before either app opens a REPL, the engine checks that this process can use CUDA unified memory. The CUDA runtime is loaded dynamically, so a missing driver DLL takes the same path as a failed check. The check passes only when all of these are true:

- The CUDA runtime initializes.
- At least one CUDA device is present.
- That device reports `cudaDevAttrManagedMemory`, so allocations can use `cudaMallocManaged`.
- When a device also reports `cudaDevAttrConcurrentManagedAccess`, the engine selects it. The N1X Windows driver can report managed memory without concurrent access. That device is still accepted. The engine calls `cudaDeviceSynchronize` before the host reads results.

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
| CudaMathGui | WinUI 3 exe | One document window per session, each window a REPL pane plus a pane of variables and functions |

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
- `tests 'name'` / `end tests` suite blocks in a script file
- `test 'name'` / `end` blocks, which group `expect` checks inside a suite
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
| `saveworkspace path` | Write a `.cmw` workspace file, including variables, user functions, and the cuRAND seed |
| `loadworkspace path` | Replace the workspace from a `.cmw` file |
| `save(path, a, b, ...)` | Write the named variables to a `.cma` data file, as in `save('picked.cma', a, b, c, d)`. The file holds those names and values only. |
| `load(path)` | Assign every variable stored in a `.cma` file. Other variables, user functions, and the cuRAND seed stay. `load(path, a, b)` assigns only the names given. |
| `run path` | Execute a `.cm` script in the current workspace |
| `clc` | Clear the REPL output |
| `expect(expected, actual)` | Compare `actual` with `expected` and record a pass or a fail. See Tests. |
| `expect(expected, actual, tol)` | Same comparison with an explicit absolute tolerance |
| `runtests path` | Execute a `.cm` suite file in a scratch workspace. The file's `end tests` prints the summary |
| `timer on` | Start reporting elapsed time after each REPL submission |
| `timer off` | Stop reporting elapsed time |
| `timer` | Print `timer on` or `timer off` |
| `help` | Print commands, types, and built-in functions. See the console app. |
| `?` | Same as `help`. In the console REPL and in a GUI document window's REPL. |
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
| `zeros(m,n)` | M-by-n array of zeros. One argument makes a square matrix. A trailing `'double'`, `'single'`, `'int32'`, or `'int64'` selects the type, as in `zeros(2, 3, 'int32')`. With no type string the result is `double`. |
| `ones(m,n)` | Same shape and type rules as `zeros`, filled with ones |
| `eye(m,n)` | Identity, same shape and type rules as `zeros` |
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
| `print(A)` `prn(A)` | List every element, one matrix row per line. Typing a name summarizes an array with more than 64 elements as its type and size. `print` and `prn` list the elements, and the result is that text. |
| `double(A)` `single(A)` `int32(A)` `int64(A)` | Convert, truncating toward zero for the integer conversions |
| `horzcat` `vertcat` | Concatenation, also spelled `[A B]` and `[A; B]` |
| `tril(A)` `triu(A)` | Lower or upper triangle. Optional second argument is the diagonal offset |

### cuRAND

Each session keeps its own cuRAND Philox generator. `rng` sets the seed for the session that is running the command. `rand`, `randn`, and `randi` allocate a unified-memory array and fill it on the device. Shape rules match `zeros`: one size argument builds a square matrix, so `rand(n,1)` is a column vector and `rand(1,n)` is a row vector. The default type is `double`. A trailing `'single'` or `'double'` selects the floating type.

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

## Files

CudaMath has three file types.

| Extension | Role |
|---|---|
| `.cmw` | Workspace. Variables, user functions, and the cuRAND seed. Opening one replaces the session. |
| `.cma` | Data file. The variables named by `save`. `load` assigns those variables and leaves the rest of the session alone. |
| `.cm` | Script. Source the REPL executes. It is not loaded as workspace data. |

File > Open Workspace and File > Save Workspace accept only `.cmw`. `loadworkspace` and `saveworkspace` do too. File > Open Script accepts only `.cm`. A script that contains `tests` / `end tests` is run with `runtests`. Any other script is run with `run`. Neither file type is a Windows executable.

## Tests

A suite is a `.cm` script. It starts with `tests 'name'` and ends with `end tests`. Each `test 'name'` / `end` block inside it is one named check. The suite guards parsing, array shape, type rules, dispatch into cuBLAS, cuSOLVER, and cuRAND, and saveworkspace/loadworkspace. It does not try to certify the CUDA libraries themselves. A few reconstruction checks are enough to show that a call was wired to the right library.

```
tests 'smoke'
  test 'lu reconstructs'
    A = [1 2 3; 4 5 6; 7 8 10];
    [L,U,P] = lu(A);
    expect(P * A, L * U, 1e-8);
  end
end tests
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

`end tests` stops the suite clock and prints the summary. The clock covers the suite only, from `tests` through the last check, and it prints even when the session timer is off:

```
smoke
Elapsed 12.345 ms
3 passed, 0 failed
```

`ans` is the number of failures. A `.cm` file with no `tests` / `end tests` block is a script. `run 'file.cm'` executes a script in the current workspace. `runtests` rejects a file that is not a suite.

`runtests 'tests/smoke.cm'` is the suite runner.

- It snapshots that session's workspace and cuRAND generator, then runs the file in an empty workspace for the same session. Other sessions are left alone.
- It restores that session after the file finishes, whether the suite passed or failed.
- The pass, fail, and suite lines come from `expect` and `end tests`, not from a second report by the runner.
- After the restore, `ans` is still the number of failures.

`CudaMathCli --runtests tests/smoke.cm` runs that command and exits. The process exit code is 0 when every check passed and 1 otherwise. In the GUI, the same command is typed in the REPL. There is no Run menu.

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

`timer on` and `timer off` switch the flag for the session that received the command. It starts off in each new session. The flag is not a workspace variable, and `saveworkspace` / `loadworkspace` do not change it.

While the flag is on, the REPL times each submission. The clock starts when evaluation of that submission begins and stops after `cudaDeviceSynchronize`, so the number includes cuBLAS, cuSOLVER, cuRAND, and kernel time rather than only the launch. Printing the result is outside the clock. A submission that ends with `;` still prints the elapsed line. A submission that throws prints the error, then the elapsed line.

The report is one line after the result:

```
Elapsed 12.345 ms
```

`runtests` is one submission. When the session timer is on, the REPL prints one more `Elapsed` line for that whole submission after the suite summary. `end tests` has already printed the suite clock. Statements inside a `test` block or a user function do not print their own session-timer lines. `timer on` and `timer off` themselves are not timed.

## Serialization

`saveworkspace` writes one `.cmw` workspace file. `loadworkspace` replaces the current workspace from a `.cmw` file. A `.cm` script is never written by `saveworkspace`.

`save('picked.cma', a, b, c, d)` writes a `.cma` data file containing the variables `a`, `b`, `c`, and `d` under those names. It does not store the rest of the workspace, user functions, or the cuRAND seed, and it does not change the session path. `load('picked.cma')` assigns those variables into the current workspace. `load('picked.cma', a, c)` assigns only `a` and `c`. A path that does not end in `.cma` is an error.

The document records:

- A format version
- Each variable: name, type, row count, column count, and values in column-major order
- Each user function: name and source text
- The cuRAND seed and the number of random values already drawn

Small documents are JSON. Array values larger than 1 MiB are stored in a binary sidecar next to the JSON, named from the document, and the JSON holds the sidecar offset and length. `loadworkspace` allocates new unified memory and copies the values in. Function source is kept as text and parsed again on the next call.

## Console app

`CudaMathCli` runs the unified-memory check before it reads any input. On failure it prints the error to standard error and exits with code 1. It does not prompt. On success it reads a line, sends it to the interpreter, and prints the result when the line does not end with `;`. Errors print the message and the REPL continues. `exit` or `quit` leaves the process. `CudaMathCli --runtests <file>` runs the same startup check, then runs that test file and exits with code 0 when every check passed and code 1 when any check failed. A failed startup check exits with code 1 before any test runs.

A long submission runs on the console thread. The terminal process owns the window, so the console does not show "Not Responding" while CUDA is busy. Keystrokes typed during that submission stay in the terminal input buffer and are not echoed. When the submission returns, the REPL prints the result and then reads the buffered line. The console app does not move a submission onto a worker thread.

`help` and `?` print the same three sections and return to the prompt. `?` is only that command: it is not an operator.

- **Commands.** One line each for `help`, `?`, `who`, `whos`, `clear`, `clear functions`, `saveworkspace`, `loadworkspace`, `save`, `load`, `run`, `clc`, `expect`, `runtests`, `timer on`, `timer off`, `timer`, `cudaversion`, `memory`, `exit`, and `quit`. `?` is the same command as `help`. `save(path, a, b)` writes those variables to a `.cma` data file. `load(path)` reads one back.
- **Types.** `double`, `single`, `int32`, and `int64`. A bare literal is `double`. Values are dense 2-D column-major arrays, and a scalar is 1-by-1.
- **Functions.** One line each for the built-ins in this spec, in the same groups as the GUI help: workspace, shape, cuRAND, elementwise math, reductions, cuBLAS, and cuSOLVER. This section also shows a user function: `function y = add(a, b)` through `end`, called as `add(1, 2)`. The call passes one value for each argument name, in that order, and the function returns the value assigned to `y`. `function y = name()` takes no arguments. `function show(a)` returns no value.

The Types and Functions text is the same text the GUI Help window uses.

## Sessions

A session holds the variables, user functions, cuRAND generator, `.cmw` path, unsaved-changes flag, timer flag, and REPL history. The console app has one session. The GUI has one session per document window. Creating a document window creates a session. Closing a document window frees that session's unified-memory arrays and discards its generator.

The process has one CUDA context and one unified-memory pool. The engine runs one submission at a time. A command typed in a second document window waits until the running submission finishes, then runs. Document windows do not issue overlapping cuBLAS, cuSOLVER, cuRAND, or kernel calls. In the GUI that submission runs on a worker thread, so a large calculation does not freeze the window. See [Large calculations](#large-calculations).

## GUI app

`CudaMathGui` runs the unified-memory check before it creates the main window. On failure it shows the error in a dialog, and the process exits when the dialog is dismissed. On success it shows a WinUI 3 frame: a menu bar, an MDI client, and a process status bar. The window opens with one document window, an empty session named `Workspace 1`.

Each document window is one session. It has a border, a title bar, two panes, and its own status bar. The active window's border uses the accent color. The title bar shows that session's name. Dragging the title bar moves the window inside the frame. Dragging an edge or a corner resizes it. Double-clicking the title bar fills the client area, and double-clicking again restores the previous size and position. Clicking a window brings it to the front and makes it the active document. File commands apply to that active document. New document windows cascade down and to the right.

The two panes are:

- That session's REPL: input, history, and printed results. While the input is empty it shows `Enter expression here` in grey italic type. Enter runs the submission. Shift+Enter inserts a line.
- A list of that session's variables and user functions

The variable list shows name, class, and size. Selecting a function shows its source. Selecting a variable shows its elements; a matrix larger than 24 by 12 shows the first 24 rows and 12 columns. The list refreshes when a submission in that session finishes. The menu bar is File and Help. `runtests`, `timer on`, and `timer off` are typed in that document's REPL. `help` and `?` typed there print the same command text the console prints, into that document. Clicking the timer field on that document's status bar toggles that session's timer.

### Large calculations

The GUI is for calculations that are large enough to be worth running on the GPU. A submission such as `rand` or `inv` of a big matrix can run for a long time. The GUI window belongs to this process. If the submission ran on the UI thread, Windows would mark that window "Not Responding". The console avoids that only because the terminal process owns its window. The UI thread does not call the engine. It echoes the command, then a worker thread bound to the process CUDA device runs that submission. While it runs, the window keeps moving, resizing, and repainting. The frame title is `CudaMath — Running`. The document that owns the submission shows `Running` on its status bar, and its empty input shows `Running…`. Another command can be typed. It waits, and that document shows `Waiting`, until the engine is free. One submission still runs at a time.

The variable list, the timer, workspace memory, and CUDA free update when the submission finishes, success or failure. Selecting a variable in the document that is running does not read its arrays until the submission finishes. Closing that document, or exiting, waits until the running submission finishes and then asks about unsaved changes. A command that was only waiting in a document being closed is discarded. A CUDA call that has already started runs to completion.

### Status bars

A document window's status bar belongs to that session. It shows the timer, whether that session is running or waiting, and that session's workspace memory. The frame's status bar belongs to the process. It shows CUDA free memory and the CUDA version, and it does not change when a different document becomes active. The frame title shows `Running` while a submission is executing.

Workspace memory and CUDA free refresh once when a REPL submission finishes, success or failure. That is the same moment the timer stops. A `for` loop, a `while` loop, a user function, and `runtests` are one submission, so the bars do not repaint on every inner statement. There is no background poll. `cudaMemGetInfo` also runs once when the window opens, so the frame shows the real free amount before any command. The timer label also refreshes when its field is clicked.

| Field | Where | Visible text |
|---|---|---|
| Title | Document title bar | That session's file name, or `Workspace N` when it has no path. `N` is the smallest positive integer that no other open unsaved workspace is using. A trailing `*` means that session has unsaved changes. |
| Timer | Document status bar | That session's `Timer on` or `Timer off`. When the timer is on, the last submission's elapsed time stays in this field, for example `Timer on  12.345 ms`. Clicking this field toggles that session. |
| Activity | Document status bar | `Running` while that session's submission is executing on the GPU. `Waiting` while a command in that session is queued behind another submission. The field is absent when the session is idle. |
| Memory | Document status bar | `Memory` and that session's array bytes, for example `0.0 MB`. |
| CUDA free | Frame status bar | `CUDA free` and the shared unified pool, for example `8192.0 MB`. |
| CUDA | Frame status bar | The runtime version string that `cudaversion` returns, for example `CUDA 13.4`. |

Memory numbers are megabytes of 1024×1024 bytes, with one decimal place. Document memory is that session's arrays. CUDA free is the shared unified pool. Both are the counts from `memory`, scaled into MB. CUDA total stays on the `memory` command. In a document, `memory` reports that session's workspace bytes and the process CUDA total and free.

### File

| Item | Action |
|---|---|
| New Workspace | Open a document window on a new empty session. |
| Open Workspace | Pick a `.cmw` file and open it in a new document window. If the active document is an unsaved session with no variables, no user functions, and no unsaved changes, load the file into that document instead. |
| Save Workspace | `saveworkspace` for the active session. If that session has no path yet, ask for one and require the `.cmw` extension. |
| Open Script | Pick a `.cm` file and execute it in the active document. Output appears in that document's REPL. A file that contains `tests` / `end tests` is executed with `runtests`, so that session is restored afterward. Any other script is executed with `run` in that session. |
| Exit | Close every document window and leave the process. |

Closing a document window asks for confirmation when that session has unsaved changes. Cancel leaves the window open. Exit, and the frame's close button, ask once for each dirty session. A successful save clears that session's dirty flag. REPL `loadworkspace` in a document replaces that session and sets its path. The document title is the file name, or `Workspace N` while the session has no path. Saving or loading a path frees that `N` for a later unsaved workspace. A `*` on the document title means unsaved changes.

### Help

| Item | Action |
|---|---|
| Help | Open the help window. |
| About | Open the about dialog. |

The help window is a short document stored in the app, with three sections:

- **App.** What CudaMath is, the REPL, document windows and sessions, the variable pane, the timer, tests, and workspace files.
- **Types.** `double`, `single`, `int32`, and `int64`. A bare literal is `double`. Values are dense 2-D column-major arrays, and a scalar is 1-by-1.
- **Functions.** One line each for the built-ins in this spec, in the same groups: workspace, shape, cuRAND, elementwise math, reductions, cuBLAS, and cuSOLVER. Include `expect`, `run`, `runtests`, `timer`, `help`, `?`, `cudaversion`, and `memory`. `?` is the same command as `help`. The same section shows a user function: `function y = add(a, b)` through `end`, called as `add(1, 2)`. The call passes one value for each argument name, in that order, and the function returns the value assigned to `y`. `function y = name()` takes no arguments. `function show(a)` returns no value. The Types and Functions text is shared with the console `help` command.

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
