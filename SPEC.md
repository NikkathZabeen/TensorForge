# TensorJIT Tensor IR (.tir) Specification

TensorJIT accepts a small tensor expression language stored in `.tir` files. The language is intentionally compact: each statement binds the result of an expression to a name with `let`, and optional annotations provide optimization hints to the compiler.

## Overview

A `.tir` program is a sequence of statements:

```tir
let C = matmul(A, B)
let D = relu(C + bias)
let out = softmax(D)
@tile(64) let C = matmul(A, B)
```

Names such as `A`, `B`, `C`, and `bias` refer to tensor values supplied by the host program or produced by earlier statements. Function calls represent tensor operations. Binary operators represent elementwise tensor arithmetic.

## Lexical Rules

Identifiers begin with an ASCII letter or underscore and may contain ASCII letters, digits, and underscores. Numbers are decimal numeric literals accepted by the lexer as integer or floating-point values. Whitespace separates tokens and is otherwise ignored.

Reserved words are:

```text
let
tile
```

The language uses these punctuation and operator tokens:

```text
@ ( ) , + * - =
```

## Formal Grammar

The grammar below is written in EBNF. `IDENT` and `NUMBER` are lexical tokens.

```ebnf
program     := stmt* ;

stmt        := tile_annotation? "let" IDENT "=" expr ;

tile_annotation
            := "@tile(" NUMBER ")" ;

expr        := add_expr ;

add_expr    := mul_expr (("+" | "-") mul_expr)* ;

mul_expr    := primary ("*" primary)* ;

primary     := func_call | IDENT | NUMBER | "(" expr ")" ;

func_call   := IDENT "(" expr ("," expr)* ")" ;
```

This grammar gives `*` higher precedence than `+` and `-`, and all binary operators are left-associative.

The minimal grammar requested for Phase 1 is:

```ebnf
program     := stmt* ;
stmt        := "@tile(" NUMBER ")" "let" IDENT "=" expr | "let" IDENT "=" expr ;
expr        := func_call | binary_expr | IDENT ;
func_call   := IDENT "(" expr ("," expr)* ")" ;
binary_expr := expr ("+" | "*" | "-") expr ;
```

TensorJIT implements the precedence-aware version above because it removes ambiguity while preserving the same language.

## Statements

A `let` statement binds an expression to a tensor name:

```tir
let C = matmul(A, B)
```

An optional tile annotation may appear before `let`:

```tir
@tile(64) let C = matmul(A, B)
```

The annotation is a compiler hint. For `matmul`, a positive tile size requests tiled loop generation. If no tile annotation is present, TensorJIT emits the default untiled triple-loop kernel.

## Expressions

TensorJIT supports variables, numbers, function calls, parenthesized expressions, and binary arithmetic.

```tir
A
1.0
matmul(A, B)
relu(C + bias)
(A + B) * C
```

Binary arithmetic is elementwise for tensor values. Scalars may be used by future lowering stages as constants.

## Built-in Tensor Operations

`matmul(A, B)` computes matrix multiplication. The generated kernel has the ABI:

```cpp
void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
```

`relu(X)` computes an elementwise rectified linear unit:

```text
max(X[i], 0.0)
```

`softmax(X)` computes:

```text
exp(X[i]) / sum(exp(X[j]))
```

## Errors

The lexer reports unknown characters with source location:

```text
Unknown character 'X' at line Y:Z
```

The parser reports missing or unexpected tokens with source location, for example:

```text
Expected ')' at line Y:Z
```

## Pipeline

```text
.tir source -> Lexer -> Parser -> AST -> Codegen -> LLVM IR -> OrcJIT -> Native Exec
```
