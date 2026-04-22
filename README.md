# TensorJIT

TensorJIT is a from-scratch C++17 and LLVM 17+ JIT compiler for a tiny tensor expression DSL. It lexes and parses `.tir` source, lowers tensor operations to LLVM IR, applies optimization passes, and executes kernels in-process with LLVM ORC JIT. The first supported kernels cover matrix multiplication, ReLU, softmax, and `@tile(N)` loop tiling hints for matmul.

## Pipeline

```text
.tir source -> Lexer -> Parser -> AST -> Codegen -> LLVM IR -> OrcJIT -> Native Exec
```

## Requirements

Recommended Windows setup:

- MSYS2 UCRT64 shell
- GCC from MSYS2 UCRT64
- CMake from MSYS2 UCRT64 or `C:\Program Files\CMake\bin\cmake.exe`
- Ninja
- LLVM 17 or newer with CMake package files

Install the required MSYS2 packages from the **MSYS2 UCRT64** terminal:

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-llvm mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-gcc
```

If `pacman -Syu` asks you to close the terminal, close MSYS2 UCRT64, reopen it, and then run the second command.

You can verify LLVM's CMake package exists with:

```bash
ls /ucrt64/lib/cmake/llvm/LLVMConfig.cmake
```

This project has been tested with MSYS2 LLVM 22.1.3, and the CMake file accepts LLVM 17 or newer.

## Build And Test

From the **MSYS2 UCRT64** terminal:

```bash
cd "/c/Users/sirap/OneDrive/Documents/New project/tensoriit"
rm -rf build
cmake -S . -B build -G Ninja -DLLVM_DIR=/ucrt64/lib/cmake/llvm
cmake --build build
ctest --test-dir build --output-on-failure
```

If all tests pass, run the compiler demo:

```bash
./build/tensoriit examples/basic.tir
./build/tensoriit examples/tiled.tir
```

Run the benchmark:

```bash
./build/bench_matmul
```

## PowerShell Notes

If you run commands from Windows PowerShell and CMake is not on `PATH`, use the full path:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --version
```

PowerShell requires `&` before a quoted executable path. Command Prompt (`cmd`) does not use `&`.

For this project, the most reliable setup is to build inside **MSYS2 UCRT64** because the installed LLVM package is also from MSYS2. Avoid mixing MSYS2 LLVM with the Visual Studio generator unless you intentionally install an MSVC-compatible LLVM development package.

## Example

`examples/basic.tir`:

```tir
let C = matmul(A, B)
let D = relu(C + bias)
let out = softmax(D)
```

Run:

```bash
./build/tensoriit examples/basic.tir
```

Output includes an AST dump and generated LLVM IR. A matmul statement emits a callable kernel with this ABI:

```cpp
void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
```

Tiled input:

```tir
@tile(64) let C = matmul(A, B)
```

Run:

```bash
./build/tensoriit examples/tiled.tir
```

Expected high-level output:

```text
Program
  Let(C, tile=64)
    FuncCall(matmul)
      Var(A)
      Var(B)

Generated LLVM IR:
define void @matmul_kernel(...)
```

The generated IR contains the tiled loop nest:

```text
ii -> jj -> kk -> ti -> tj -> tk
```

## Benchmarks

Run:

```bash
./build/bench_matmul
```

Measured on the current machine:

```text
Size       |   Naive (ms) |      JIT no-tile |      JIT tile=64 |  Speedup
----------------------------------------------------------------------------
64x64      |        0.463 |            0.145 |            0.438 |    1.06x
256x256    |       31.237 |           10.549 |           24.080 |    1.30x
512x512    |      267.979 |           95.292 |          190.326 |    1.41x
1024x1024  |     2688.845 |         3368.728 |         2774.480 |    0.97x
```

The speedup column compares naive C++ against the tiled JIT kernel. These numbers are useful as a smoke test, not as a final optimized GEMM result. The current tiled kernel is a simple generated loop nest and does not yet include vectorization, packing, cache-aware micro-kernels, or CPU-specific tuning.

## Troubleshooting

If CMake cannot find LLVM:

```bash
cmake -S . -B build -G Ninja -DLLVM_DIR=/ucrt64/lib/cmake/llvm
```

If CTest says it cannot find `test_lexer.exe`, `test_parser.exe`, or `test_codegen.exe`, build first:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

If PowerShell says `cmake` is not recognized, either add CMake to `PATH` or call it directly:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --version
```

If Command Prompt says `Remove-Item` is not recognized, that is expected: `Remove-Item` is a PowerShell command. In `cmd`, use:

```bat
rmdir /s /q build
```

## Project Layout

```text
tensoriit/
|-- src/
|   |-- lexer.h / lexer.cpp
|   |-- parser.h / parser.cpp
|   |-- ast.h
|   |-- codegen.h / codegen.cpp
|   |-- optimizer.h / optimizer.cpp
|   `-- jit.h / jit.cpp
|-- tests/
|   |-- test_lexer.cpp
|   |-- test_parser.cpp
|   `-- test_codegen.cpp
|-- benchmarks/
|   `-- bench_matmul.cpp
|-- examples/
|   |-- basic.tir
|   `-- tiled.tir
|-- SPEC.md
|-- CMakeLists.txt
`-- README.md
```

## Deliverable Checklist

- [x] DSL grammar defined in `SPEC.md`
- [x] Hand-rolled lexer with line/column tracking
- [x] Recursive-descent parser producing typed AST nodes
- [x] LLVM IR codegen for matmul, ReLU, and softmax
- [x] Tiled matmul lowering for `@tile(N)`
- [x] ORC JIT integration for in-process kernel execution
- [x] LLVM optimization pass pipeline before JIT
- [x] Benchmark harness comparing naive, JIT no-tile, and JIT tile=64
- [x] Assert-based unit tests for lexer, parser, codegen, and JIT smoke test
- [x] CMake build with LLVM 17+ linkage
- [x] Pipeline diagram and benchmark results
- [x] Example `.tir` programs
