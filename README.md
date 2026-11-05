# Piste.jl

[![CI](https://github.com/EvoArt/Piste.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoArt/Piste.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/EvoArt/Piste.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EvoArt/Piste.jl)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/Piste.jl/dev/)

Forward- and reverse-mode automatic differentiation that survives
`juliac --trim`.

Gradients, from code slim enough to compile into a standalone binary. Not just
any slope. A nicely groomed piste.

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
`pickchunk(K)` gives a reasonable width if you have no reason to prefer one --
it matters, at K=200 the wrong choice is a 5x difference.

In a hot loop, pass a workspace to reuse the dual buffer and allocate nothing:

```julia
ws = GradientWorkspace(x, Val(32))
for _ in 1:nsteps
    gradient!(g, f, x, Val(32), ws)            # 0 bytes per call
end
```

At K=200 on a Rosenbrock objective this is within ~10% of ForwardDiff at matched
chunk width, and faster than it at widths above 48. See `PERFORMANCE_NOTES.md`.

## Reverse mode

Forward mode costs `ceil(K/N)` passes, so it loses badly as `K` grows. Reverse
mode costs one sweep regardless:

```julia
ws = ReverseWorkspace(length(x))
rev_gradient!(g, f, x, ws)
v, g = rev_value_and_gradient!(g, f, x, ws)
```

Use reverse for parameter-heavy models, forward for small ones.

## Benchmarks

Median time for one gradient of a Bayesian regression log-density with N
observations and P coefficients (the normal model also has a noise scale σ).
The models are linear (`normal`), Poisson (`poisson`) and logistic
(`bernoulli_logit`) regressions written in
[PracticalBayes](https://github.com/EvoArt/PracticalBayes.jl), and every backend
differentiates the same function. ForwardDiff, Mooncake and Enzyme (reverse
mode) are called through DifferentiationInterface with a prepared gradient.
Piste forward uses `pickchunk(K)` with a `GradientWorkspace`. Piste reverse uses a
`ReverseWorkspace`. The fastest time in each row is in bold.

Forward mode, from Piste or ForwardDiff, is competitive only while P is small.
At P=200 the reverse-mode backends are one to two orders of magnitude faster.
With few parameters and many observations, Piste forward is level with or ahead
of everything else on the Poisson and logistic models. With many of both, Piste
reverse is the fastest backend on those two models. Enzyme is the fastest
backend on the normal model, where Piste reverse is well behind when P is small.
Piste is the only backend in these tables that survives `juliac --trim`.

The timings come from PracticalBayes' benchmark sweep. The corner cells of its
5x5 grid are shown here, and
[`benchmark/make_readme_table.jl`](benchmark/make_readme_table.jl) regenerates
these tables from its output.

<!-- BENCH:START -->
Measured with Julia 1.12.7 on one Julia thread and 1 BLAS thread of an exclusive Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz node: Piste 0.3.0 (003fd5d), ForwardDiff 1.4.7, Mooncake 0.5.67, Enzyme 0.13.216. Each figure is the median of up to 200 samples or 3 s.

### Float64

| Model | N | P | Piste fwd | Piste rev | ForwardDiff | Mooncake | Enzyme |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 50 | 2 | 1.49 µs | 3.85 µs | **968 ns** | 2.55 µs | 1.04 µs |
| `normal` | 50 | 200 | 532 µs | 17.5 µs | 1.71 ms | 32.3 µs | **9.75 µs** |
| `normal` | 20000 | 2 | 448 µs | 1.08 ms | 320 µs | 262 µs | **144 µs** |
| `normal` | 20000 | 200 | 370 ms | 5.58 ms | 614 ms | 14.3 ms | **3.92 ms** |
| `poisson` | 50 | 2 | 4.23 µs | 5.53 µs | 4.26 µs | 13.3 µs | **3.89 µs** |
| `poisson` | 50 | 200 | 577 µs | **19.5 µs** | 1.77 ms | 43.1 µs | 20.5 µs |
| `poisson` | 20000 | 2 | 1.15 ms | 1.88 ms | **1.13 ms** | 2.87 ms | 1.24 ms |
| `poisson` | 20000 | 200 | 386 ms | **8.55 ms** | 630 ms | 19.6 ms | 12.8 ms |
| `bernoulli_logit` | 50 | 2 | 4.38 µs | 6.50 µs | **3.61 µs** | 14.8 µs | 5.74 µs |
| `bernoulli_logit` | 50 | 200 | 577 µs | 20.3 µs | 1.82 ms | 42.8 µs | **16.8 µs** |
| `bernoulli_logit` | 20000 | 2 | **1.20 ms** | 2.29 ms | 1.31 ms | 3.68 ms | 1.25 ms |
| `bernoulli_logit` | 20000 | 200 | 370 ms | **7.84 ms** | 626 ms | 17.8 ms | 11.0 ms |

### Float32

| Model | N | P | Piste fwd | Piste rev | ForwardDiff | Mooncake | Enzyme |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 50 | 2 | 2.53 µs | 3.92 µs | 1.67 µs | 2.07 µs | **1.43 µs** |
| `normal` | 50 | 200 | 429 µs | 18.3 µs | 2.29 ms | 20.8 µs | **8.41 µs** |
| `normal` | 20000 | 2 | 249 µs | 1.06 ms | 146 µs | 232 µs | **66.8 µs** |
| `normal` | 20000 | 200 | 179 ms | 3.31 ms | 601 ms | 6.76 ms | **1.49 ms** |
| `poisson` | 50 | 2 | 3.57 µs | 5.53 µs | 3.59 µs | 8.09 µs | **3.31 µs** |
| `poisson` | 50 | 200 | 439 µs | 20.0 µs | 2.34 ms | 27.1 µs | **10.4 µs** |
| `poisson` | 20000 | 2 | **1.10 ms** | 1.86 ms | 1.10 ms | 2.86 ms | 1.30 ms |
| `poisson` | 20000 | 200 | 182 ms | **4.21 ms** | 609 ms | 9.80 ms | 5.10 ms |
| `bernoulli_logit` | 50 | 2 | **3.66 µs** | 7.43 µs | 4.12 µs | 8.62 µs | 5.43 µs |
| `bernoulli_logit` | 50 | 200 | 450 µs | 20.6 µs | 2.32 ms | 25.3 µs | **10.6 µs** |
| `bernoulli_logit` | 20000 | 2 | **1.19 ms** | 2.25 ms | 1.30 ms | 2.96 ms | 1.38 ms |
| `bernoulli_logit` | 20000 | 200 | 184 ms | **4.46 ms** | 610 ms | 9.59 ms | 5.02 ms |
<!-- BENCH:END -->

## Trimming

Pass a literal `Val(N)` rather than `pickchunk(K)`: the trimmer needs the width
statically known, which is why it is a type parameter.

Every established AD package fails under `--trim`, because each builds its
derivative at runtime in some form the trimmer cannot see -- a config, a thunk,
an interpreter. Piste has no tag, no config, and a chunk width fixed as a type
parameter, so the derivative is ordinary compiled Julia.

