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
With few parameters, Piste forward is level with or ahead of everything else on
the Poisson and logistic models. With many parameters, Piste reverse is the
fastest backend on those two models whenever there are many observations, and
in `Float64` with few observations too. Enzyme is the fastest backend on the
normal model (ForwardDiff edges it in the smallest cell). There Piste reverse is
1.3-1.7x behind Enzyme at P=200, and 2.4-7x behind when P is small.
Piste is the only backend in these tables that survives `juliac --trim`.

The timings come from PracticalBayes' benchmark sweep. The corner cells of its
5x5 grid are shown here, and
[`benchmark/make_readme_table.jl`](benchmark/make_readme_table.jl) regenerates
these tables from its output.

<!-- BENCH:START -->
Measured with Julia 1.12.7 on one Julia thread and 1 BLAS thread of an exclusive Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz node: Piste 0.3.0 (92b011b), ForwardDiff 1.4.7, Mooncake 0.5.67, Enzyme 0.13.216. Each figure is the median of up to 200 samples or 3 s.

### Float64

| Model | N | P | Piste fwd | Piste rev | ForwardDiff | Mooncake | Enzyme |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 50 | 2 | 1.39 µs | 2.53 µs | **957 ns** | 2.56 µs | 1.01 µs |
| `normal` | 50 | 200 | 535 µs | 11.4 µs | 1.69 ms | 32.5 µs | **8.39 µs** |
| `normal` | 20000 | 2 | 448 µs | 486 µs | 401 µs | 265 µs | **153 µs** |
| `normal` | 20000 | 200 | 366 ms | 5.34 ms | 616 ms | 19.1 ms | **4.20 ms** |
| `poisson` | 50 | 2 | **3.03 µs** | 5.25 µs | 4.33 µs | 14.1 µs | 3.86 µs |
| `poisson` | 50 | 200 | 574 µs | **15.3 µs** | 1.82 ms | 43.2 µs | 20.2 µs |
| `poisson` | 20000 | 2 | 1.15 ms | 1.80 ms | **1.14 ms** | 2.82 ms | 1.21 ms |
| `poisson` | 20000 | 200 | 373 ms | **6.60 ms** | 622 ms | 19.2 ms | 11.0 ms |
| `bernoulli_logit` | 50 | 2 | **3.23 µs** | 5.96 µs | 3.67 µs | 14.3 µs | 5.75 µs |
| `bernoulli_logit` | 50 | 200 | 588 µs | **15.4 µs** | 1.79 ms | 43.0 µs | 20.1 µs |
| `bernoulli_logit` | 20000 | 2 | **1.22 ms** | 2.12 ms | 1.37 ms | 3.63 ms | 1.83 ms |
| `bernoulli_logit` | 20000 | 200 | 369 ms | **7.62 ms** | 623 ms | 17.8 ms | 11.3 ms |

### Float32

| Model | N | P | Piste fwd | Piste rev | ForwardDiff | Mooncake | Enzyme |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 50 | 2 | 2.51 µs | 2.49 µs | 1.65 µs | 2.10 µs | **1.04 µs** |
| `normal` | 50 | 200 | 429 µs | 12.4 µs | 2.31 ms | 28.1 µs | **8.34 µs** |
| `normal` | 20000 | 2 | 254 µs | 502 µs | 148 µs | 234 µs | **69.7 µs** |
| `normal` | 20000 | 200 | 185 ms | 2.76 ms | 601 ms | 7.01 ms | **1.61 ms** |
| `poisson` | 50 | 2 | 3.57 µs | 5.19 µs | 3.59 µs | 8.53 µs | **3.40 µs** |
| `poisson` | 50 | 200 | 438 µs | 14.6 µs | 2.32 ms | 27.9 µs | **11.5 µs** |
| `poisson` | 20000 | 2 | 1.10 ms | 1.79 ms | **1.10 ms** | 2.75 ms | 1.37 ms |
| `poisson` | 20000 | 200 | 187 ms | **3.99 ms** | 609 ms | 9.90 ms | 5.15 ms |
| `bernoulli_logit` | 50 | 2 | **3.68 µs** | 5.29 µs | 4.59 µs | 8.36 µs | 5.42 µs |
| `bernoulli_logit` | 50 | 200 | 443 µs | 17.9 µs | 2.39 ms | 25.4 µs | **10.4 µs** |
| `bernoulli_logit` | 20000 | 2 | **1.20 ms** | 2.12 ms | 1.30 ms | 3.24 ms | 1.39 ms |
| `bernoulli_logit` | 20000 | 200 | 188 ms | **4.63 ms** | 611 ms | 10.0 ms | 5.24 ms |
<!-- BENCH:END -->

## Trimming

Pass a literal `Val(N)` rather than `pickchunk(K)`: the trimmer needs the width
statically known, which is why it is a type parameter.

Every established AD package fails under `--trim`, because each builds its
derivative at runtime in some form the trimmer cannot see -- a config, a thunk,
an interpreter. Piste has no tag, no config, and a chunk width fixed as a type
parameter, so the derivative is ordinary compiled Julia.

