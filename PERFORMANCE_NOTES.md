# Performance: both gaps closed, Piste is at parity with ForwardDiff

Piste was ~3x slower than ForwardDiff at matched chunk width. Two causes were
recorded in the previous version of this note; both are now fixed. At K=200 the
gradient went from **80 400 ns to 27 700 ns, and from 52 887 bytes to 0**.

Measured on Julia 1.12.7, Windows, AC power, machine otherwise idle, with
BenchmarkTools and all arguments interpolated. Contention matters: a sweep run
while a test suite was going gave numbers ~20% off, so these were all re-taken
on an idle machine.

## The measurement

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

At K = 200:

| width | Piste (workspace) | Piste (allocating) | ForwardDiff | was |
|---|---|---|---|---|
| 8 | 52 000 ns / 0 B | 52 900 ns | 40 400 ns | 236 800 |
| 12 | 37 100 ns / 0 B | 38 300 ns | 26 600 ns | 186 000 |
| 16 | 29 000 ns / 0 B | 31 500 ns | 26 100 ns | 145 100 |
| **32** | **27 700 ns / 0 B** | 31 200 ns | 25 200 ns | 89 800 |
| 48 | 28 200 ns / 0 B | 34 700 ns | 29 800 ns | 130 800 |
| 64 | 29 100 ns / 0 B | 34 300 ns | **37 800 ns** | 162 400 |

ForwardDiff's auto-selected chunk gives 26 700 ns.

Piste is now **1.10x of ForwardDiff at its best width**, down from 3.2x, and is
*faster* than ForwardDiff from width 48 up, where ForwardDiff's own performance
falls off. The target in the old note was parity at matched width; that is met.

## What the two fixes were

### 1. `literal_pow` — worth 2.2-2.9x, not the ~20% predicted

The old note guessed the method was unregistered. **That was wrong** — it was
registered and it did fire. The cost was inside its body:

```julia
_chain(x, x.v^p, T(p) * x.v^(p - 1))     # both ^ are RUNTIME powers
```

`^(::Float64, ::Int)` reaches `Base.Math.pow_body` and the libm power path, so
squaring a dual — which should be one multiply — went through a general
exponential. Small exponents are now enumerated (`Val{0}` through `Val{3}`) so
the value work is literal multiplies; `p` is a type parameter, so the branch
resolves at compile time. Verified with `@code_typed` that `x^2` no longer
mentions `pow_body`.

This was by far the larger win, and the profile understated it: `^`/`pow_body`
showed as 19.7% of total, but removing it took the K=200 gradient from 80 400 to
about 36 500 ns, because the libm path was also blocking vectorisation of the
surrounding partials arithmetic.

### 2. `GradientWorkspace` — 52 887 bytes to 0

`gradient!` built a fresh `Vector{Dual{N,T}}(undef, K)` per call; `GenericMemory`
was 54.5% of self time, GC-flagged. ForwardDiff keeps that buffer in its
`GradientConfig`. Piste now has an optional workspace argument:

```julia
ws = GradientWorkspace(x, Val(32))
gradient!(g, f, x, Val(32), ws)      # 0 bytes
```

The allocating methods still exist and still work; the workspace is opt-in for
hot loops such as a sampler's leapfrog step. It is a **concrete struct, not a
closure capture** — a boxed capture infers as `Any` and is exactly the
`Core.Box` problem that `_grad_chunk!` was split out to avoid under `--trim`.

A test asserts `@allocated == 0`, so a regression here fails the suite rather
than showing up later as a slow sampler.

### 3. `pickchunk(K)` — chunk selection

The old fixed default of 8 was poor everywhere except very small problems. See
the docstring for the measured threshold table.

**Caveat, worth knowing before trusting it:** the thresholds were baked from the
contaminated sweep, and are correct but not optimal. `pickchunk(200)` returns 64
(29 500 ns) where 32 is actually best (27 700 ns) — about 6% off. The shape is
right (optimum rises with K, then flattens); the exact cutoffs above K≈100 could
be re-baked on idle numbers if that 6% ever matters.

`pickchunk` returns a `Val`, so where `K` is not a compile-time constant it costs
one dynamic dispatch into `gradient!` — fine under JIT. **Under `--trim`, pass a
literal `Val(N)`:** the trimmer needs the width statically known, which is the
whole reason the width is a type parameter.

## Do not regress

- **Correctness first.** 354 tests, all comparing against ForwardDiff or central
  differences — never a hand-computed number, because a wrong gradient does not
  crash.
- **Trimmability is the whole point of this package.** Nothing in these fixes
  should cost it: `literal_pow` is statically resolved per exponent, and the
  workspace is a concrete struct. But this has **not been re-verified under
  `--trim` since the changes** — build `adtrim/ProbeDUAL` in the PracticalBayes
  tree with `--trim=safe`, confirm 0 verifier errors, then run the binary with
  `JULIA_LOAD_CODEGEN_LIB=0` and diff against the JIT. A clean verifier pass
  alone proves nothing — that is exactly how ForwardDiff fails under trim.
- Watch for `Core.Box`: a closure capturing a mutated local produced 14 verifier
  errors before `_grad_chunk!` was split out.
- **Do not drop a rule for speed.** The lgamma family (`_logabsgamma`) and
  `log1p` are what let Piste express models the other trimmable option cannot:
  STADE rejects `lgamma` outright, and three of the five GLM links in
  PracticalBayes' `GradMode` fast path need lgamma or log1p terms. That coverage
  is a capability advantage, not a nicety.

## Where the remaining effort should go

Forward mode is O(K), so even at parity with ForwardDiff it loses to reverse mode
at large K. Reverse mode now lives in `src/reverse.jl` (trimmable, 0 verifier
errors, ~15 us at K=200). The division of labour is that reverse covers large K
and Piste's forward mode covers small K, where a tape's bookkeeping costs more
than it saves — which is exactly the regime these fixes improved most.
