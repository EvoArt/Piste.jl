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

Under `--trim`, pass a literal `Val(N)` rather than `pickchunk(K)`: the trimmer
needs the width statically known, which is why it is a type parameter.

