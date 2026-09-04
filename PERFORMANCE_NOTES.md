# Performance: two known gaps vs ForwardDiff, both located, neither fixed

Handover note. Piste is **correct** (162 tests, agrees with ForwardDiff bit for
bit) but at K=200 it is ~3x slower than ForwardDiff *at matched chunk width*.
Two concrete causes were found by profiling; neither is fixed yet.

Everything below was measured on Julia 1.12.7, Windows, on AC power with the
machine otherwise idle, using BenchmarkTools with all arguments interpolated.

## The measurement

Objective (the same one every Piste/PracticalBayes probe uses):

```julia
function rosen(v)
    s = zero(eltype(v))
    @inbounds for i in 1:(length(v)-1)
        a = v[i+1] - v[i]^2; b = one(eltype(v)) - v[i]
        s = s + 100*a^2 + b^2
    end
    s
end
```

At K = 200, chunk width swept on both sides:

| width | Piste | ForwardDiff |
|---|---|---|
| 8 | 236 800 ns | 42 700 ns |
| 12 | 186 000 ns | 61 300 ns |
| 16 | 145 100 ns | 33 100 ns |
| **32** | **89 800 ns** | **30 900 ns** |
| 48 | 130 800 ns | 35 700 ns |
| 64 | 162 400 ns | 48 000 ns |

ForwardDiff's own auto-selected chunk (`pickchunksize(200)` = 12) gives 29 400 ns.

**Both peak at width 32**, so width selection is worth having — it takes Piste
from 236 to 90 us — but it is *not* the explanation for the gap. At matched
width Piste is still ~2.9x behind. My first hypothesis (that the earlier
benchmark unfairly pinned Piste at `Val(8)` while ForwardDiff auto-selected)
was **wrong**, and worth stating so nobody re-runs that experiment.

## Gap 1: allocation — 52 887 bytes per gradient, vs ForwardDiff's 0

```
allocations per Piste gradient (Val 32):        52887 bytes
allocations per ForwardDiff gradient (Chunk 32):    0 bytes

Piste       one dual eval: 11400 ns, 0 bytes
ForwardDiff one dual eval:  3400 ns, 0 bytes
```

A single evaluation *on duals* allocates nothing on either side, so the
allocation is all in the driver: `gradient!` builds a fresh
`Vector{Dual{N,T}}(undef, K)` on every call. ForwardDiff keeps that buffer in
its `GradientConfig`.

Profile confirms it — `GenericMemory` is **54.5% self time, flagged for GC**:

```
 self%  total%  gc?  dispatch?   function          file:line
  54.5    54.5   Y       -       GenericMemory     boot.jl:588
  13.6    13.6   -       -       *                 float.jl:497
   6.1     6.1   -       -       ifelse            essentials.jl:799
   4.5     4.5   -       -       pow_body          math.jl:0
   1.5    19.7   -       -       ^                 math.jl:1201
```

**The fix** is a reusable buffer — either a config object like ForwardDiff's, or
an optional preallocated-workspace argument to `gradient!`. Note this must not
cost trimmability: the workspace has to be a concrete struct, not a closure
capture (see `_grad_chunk!`'s comment in `src/Piste.jl` for the `Core.Box`
problem that shape already caused once).

## Gap 2: `x^2` is NOT hitting `literal_pow`

`^` and `pow_body` are **19.7% of total** in the profile above, which should be
near zero — squaring a dual ought to be one multiply.

The cause: **Piste has no `literal_pow` methods registered at all.**

```julia
julia> for m in methods(Base.literal_pow); occursin("Piste", string(m)) && println(m); end
# (prints nothing)

julia> for m in methods(Base.literal_pow); occursin("ForwardDiff", string(m)) && println(m); end
  literal_pow(::typeof(^), x::ForwardDiff.Dual{T}, ::Val{0}) where T
  literal_pow(::typeof(^), x::ForwardDiff.Dual{T}, ::Val{1}) where T
  literal_pow(::typeof(^), x::ForwardDiff.Dual{T}, ::Val{2}) where T
  literal_pow(::typeof(^), x::ForwardDiff.Dual{T}, ::Val{3}) where T
```

`src/Piste.jl:312` *defines* one:

```julia
@inline Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{p}) where {N,T,p} =
    _chain(x, x.v^p, T(p) * x.v^(p - 1))
```

but it is not taking effect, so `x^2` falls through to generic `^` and the
libm power path. Worth checking the signature against ForwardDiff's — theirs
dispatches on `Dual{T}` with the tag as the *first* parameter, and enumerates
small literal exponents individually rather than using a free `Val{p}`.
Note also that the body as written computes `x.v^(p-1)` with a *runtime* `^`,
which would be slow even if the method did fire.

## Suggested order of work

1. **`literal_pow` first** — it is small, self-contained, and worth ~20%.
   Verify with `@code_typed` that `x^2` on a `Dual` no longer reaches
   `pow_body`.
2. **Then the buffer** — the larger win (54% of time is GC/allocation), but it
   is an API change (`gradient!` gains a workspace or a config), so it wants a
   moment's design.
3. **Then chunk selection** — a `pickchunksize`-equivalent so users are not
   pinned to a bad default. 32 was best here at K=200; it will be
   problem-dependent.

After each, re-run the width sweep above and compare against the ForwardDiff
column. The target is parity at matched width; anything better is a bonus.

## Do not regress

- **Correctness first.** Every change must keep `Pkg.test()` green (162 tests,
  all comparing against ForwardDiff or central differences — never a
  hand-computed number, because a wrong gradient does not crash).
- **Trimmability is the whole point of this package.** After optimising, re-run
  the trim check: build `adtrim/ProbeDUAL` (in the PracticalBayes tree) with
  `--trim=safe` and confirm 0 verifier errors, then run the binary with
  `JULIA_LOAD_CODEGEN_LIB=0` and diff against the JIT. A clean verifier pass
  alone proves nothing — that is exactly how ForwardDiff fails under trim.
- Watch for `Core.Box`: a closure capturing a mutated local is what produced 14
  verifier errors before `_grad_chunk!` was split out into its own function.

## Context on where the 3x actually matters

Piste is forward mode, so it is O(K) regardless. Even at parity with
ForwardDiff it loses badly to reverse mode at large K — at K=200 Mooncake is
~10 us against ForwardDiff's ~30 us. Piste's value is that it *trims* and
Mooncake does not.

A trimmable reverse-mode tape has since been prototyped (0 verifier errors,
15.1 us at K=200, within ~1.5x of Mooncake), so the long-term answer for large
K is probably that, with Piste covering small K where a tape's bookkeeping
costs more than it saves. That does not make these two gaps not worth fixing —
small-K is exactly where Piste is meant to win.
