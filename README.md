# Piste.jl

**Forward-mode automatic differentiation that survives `juliac --trim`.**

A *piste* is a groomed ski run: a slope, trimmed and prepared. That is the whole
package — gradients, from code slim enough to compile into a standalone binary.

## Why

Julia 1.12's `--trim` compiles a program to a small native binary needing no
Julia install at the target. Every established AD package fails under it, and
they all fail for the same underlying reason: **the derivative code is built at
runtime**, in some form the trimmer cannot see.

Measured directly, same objective, `--trim=safe`:

| package | result |
|---|---|
| ForwardDiff | builds with **0 verifier errors**, then `MethodError`s at runtime |
| FastDifferentiation | 6 errors — `RuntimeGeneratedFunctions` (an `Expr` in a type parameter) |
| Enzyme | 1276 errors — `Enzyme.Compiler.thunk(...)::Any`, calls the compiler |
| Mooncake | 48 errors — reaches `Compiler.InferenceState`, needs the compiler |
| Differ.jl | **0 verifier errors**, then dies: `ContextualInterpreter` |
| **Piste** | **0 errors, and the binary reproduces the JIT exactly** |

The rule those measurements produced, and the one this package obeys:

> The derivative must exist as ordinary compiled Julia, needing nothing from the
> compiler or an interpreter at any point the trimmed binary runs.

ForwardDiff is the instructive case, because **duals are not what breaks it**.
The machinery around them is: a `GradientConfig` built at runtime, per-call-site
`Tag` types that multiply the method table, and dynamically chosen chunk widths.
So Piste has no tag, no config, and a chunk width that is a *type parameter*.

⚠️ Note the two rows that build cleanly and then die. **A clean verifier pass
proves nothing.** Always run the binary with `JULIA_LOAD_CODEGEN_LIB=0`, which
makes a silent JIT fallback fail loudly, and diff against the JIT.

## Usage

```julia
using Piste

f(x) = sum(abs2, x) + exp(x[1]) * log(x[2])
x = [0.7, 1.3, -0.4]
g = similar(x)

gradient!(g, f, x, Val(4))                     # 4 directional derivatives per pass
v, g = value_and_gradient!(g, f, x, Val(4))    # primal comes free
```

`Val(N)` is the chunk width: a `K`-parameter gradient costs `ceil(K/N)`
evaluations of `f`, with `N` lanes computed at once in SIMD-friendly tuples.
8 is a reasonable default; small problems can prefer 4.

Float32 works — the dual is generic in its element type, so a `Float32` input
stays `Float32` all the way through.

## Scope, honestly

This is **forward mode**. Cost grows with the parameter count, so at large `K` a
reverse-mode package beats it by a wide margin (around `K=200`, Mooncake is
roughly 20× faster). At matched chunk width Piste is within ~15% of ForwardDiff
and agrees with it bit for bit.

**Use Piste when you want a trimmed standalone binary** — currently the only
option that works — **or when `K` is small.** Otherwise use reverse mode.

## Working with Distributions.jl

`Dual <: Real`, which is the door `Distributions.logpdf` and similar generic
code walks through. `Distributions` is only a *test* dependency; the engine
needs nothing from it.

Three methods beyond plain arithmetic turned out to be needed to make a whole
distribution suite work, each found by measurement rather than reasoning:

- **`SpecialFunctions._logabsgamma`** — the single missing method behind Gamma,
  Beta, Poisson, Binomial, NegativeBinomial, TDist *and* Chisq. All seven failed
  on it and nothing else. Implemented here as a Lanczos series written in terms
  of `+ - * / log exp`, so it differentiates through the existing rules with no
  extra work (verified against `SpecialFunctions.loggamma` to ~1e-15, and its
  derivative against `digamma`).
- **Two-argument `atan(y, x)`** — its absence was a `StackOverflowError`
  (infinite promotion recursion), not a `MethodError`. Needed by Cauchy's `cdf`,
  hence by any `truncated(Cauchy(...), 0, Inf)` prior. The rule also has to
  special-case an infinite argument, or a truncation bound at `Inf` yields a
  `NaN` derivative alongside a perfectly correct value.
- **`isinteger`** — discrete distributions call it in their support check.

If you hit a `MethodError` on some other function, that is the pattern: add the
rule, and check it against finite differences.

## Performance note

One method is worth a factor of six. Generic matrix-vector multiply's inner loop
is `s = muladd(A[i,j], x[j], s)`; without a three-argument `muladd` on the dual
that promotes and falls back to a separate `*` and `+`, building a temporary per
iteration. On a 200×50 matvec at chunk 8 that was 138.5 µs versus ForwardDiff's
22.9 µs — on *identical* memory layout. With the method: 23.0 µs.

If you extend Piste and something is unexpectedly slow, profile before
theorising; the cause is usually a missing method, not the approach.

## Status

Early. The rule set covers what the test suite exercises (162 tests, all
comparing against ForwardDiff or central differences — never a hand-computed
number, because a wrong gradient does not crash, it silently gives a wrong
answer). Not yet tested: nested differentiation, GPU arrays, Hessians.
