# Piste.jl

Forward-mode automatic differentiation that survives `juliac --trim`.

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

