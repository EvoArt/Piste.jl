# Piste.jl

Forward- and reverse-mode AD that survives `juliac --trim`.

```julia
using Piste

f(x) = sum(abs2, x) + exp(x[1]) * log(x[2])
x = [0.7, 1.3, -0.4]
g = similar(x)

gradient!(g, f, x, Val(4))    # forward: ceil(K/4) passes
rev_gradient!(g, f, x)        # reverse: one sweep
```

Forward for small `K`, reverse for large. Pass a workspace
([`GradientWorkspace`](@ref), [`ReverseWorkspace`](@ref)) to allocate nothing
per call.
