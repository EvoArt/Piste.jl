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
- **Trimmability is the whole point of this package.** These fixes were checked
  against it and are safe: `adtrim/ProbeDUAL` (PracticalBayes tree) builds with
  **0 verifier errors**, and the binary reproduces the JIT under
  `JULIA_LOAD_CODEGEN_LIB=0` (max |AD − finite difference| 4.31e-9). The probe
  exercises the **workspace path with a literal `Val`**, which is the code that
  needed proving — `literal_pow` resolving per exponent, and the workspace being
  a concrete struct rather than a closure capture. Re-run that probe after any
  change here; a clean verifier pass alone proves nothing, which is exactly how
  ForwardDiff fails under trim.
- Watch for `Core.Box`: a closure capturing a mutated local produced 14 verifier
  errors before `_grad_chunk!` was split out.
- **Do not drop a rule for speed.** The lgamma family (`_logabsgamma`) and
  `log1p` are what let Piste express models the other trimmable option cannot:
  STADE rejects `lgamma` outright, and three of the five GLM links in
  PracticalBayes' `GradMode` fast path need lgamma or log1p terms. Piste is the
  *slowest* of the trimmable options (see below), so coverage is the whole of
  its case — which makes this rule more important, not less.

## Where Piste actually stands, and where effort should go

Forward mode is O(K), so even at parity with ForwardDiff it loses to reverse mode
at large K. Reverse mode now lives in `src/reverse.jl` (trimmable, 0 verifier
errors, ~15 us at K=200). Within Piste, the division of labour is that reverse
covers large K and forward covers small K, where a tape's bookkeeping costs more
than it saves — which is exactly the regime these fixes improved most.

Against the alternatives, measured on PracticalBayes' normal-regression
log-density over an (N,K) grid:

- **STADE beats the best Piste mode in 13 of 16 cells** (1.6x–6.4x, widening
  with K). It is genuinely faster; an earlier framing here that capability
  settled the comparison was too quick.
- **Mooncake beats STADE in 12 of 16.** So the speed ordering is
  Mooncake > STADE > Piste, and Piste is last.

Piste's case therefore rests entirely on being the only option that is **both
general and trimmable**: STADE needs a hand-written kernel per model in a
restricted subset that cannot express `lgamma`, and Mooncake does not trim at
all. That is a real and narrow niche, and it is worth being honest that it is a
coverage argument rather than a speed one.

**The gap worth closing next** is reverse mode, which is ~1.5–4x behind STADE.
That is the mode that should win at large K, and it is where Piste currently
gives up the most.

---

# Reverse mode, second pass: the ordering above is now wrong

The section above ends by calling reverse mode the gap worth closing. It was
closed. **Piste's reverse mode is now faster than Mooncake** on both of the
model shapes that matter here, so "Mooncake > STADE > Piste" no longer
describes reverse mode.

Measured on Julia 1.10.9, Windows, AC power, BenchmarkTools with arguments
interpolated, `BLAS.set_num_threads(1)`. Mooncake is given `prepare_gradient`,
which is the fair comparison to a reused `ReverseWorkspace`. Every gradient is
checked against ForwardDiff before it is timed.

## Regression log-density (the matvec shape)

| N | K | nodes | Piste | Mooncake | Piste is |
|---|---|---|---|---|---|
| 1000 | 50 | 4 159 | 41 µs | 74 µs | 1.8x faster |
| 2000 | 100 | 8 309 | 116 µs | 253 µs | 2.2x faster |
| 5000 | 200 | 20 609 | **797 µs** | 2 996 µs | **3.8x faster** |
| 10000 | 200 | 40 609 | 2 258 µs | 6 289 µs | 2.8x faster |

At N=5000, K=200 this is 4 583 → 797 µs against the previous implementation.

## HMM forward filter (scalar, sequential, no matvec)

Sized to a real cohort (~3 200 sequences, ~39 000 emissions), not a toy.

| case | Piste | Mooncake | ratio |
|---|---|---|---|
| real-like, Se/Sp recomputed | 6.41 ms | 4.78 ms | 1.34x |
| real-like, **Se/Sp hoisted** | 4.15 ms | 3.69 ms | **1.12x** |
| big-like (dense), hoisted | 6.34 ms | 4.81 ms | 1.32x |

Hoisting means computing each `logistic(theta[k])` once instead of once per
emission, which is what a hand-written model does anyway. It is worth ~1.5x
and it is the single most valuable thing a *caller* can do.

## What the five changes were

1. **Array-level matvec on both passes.** `X*beta` was decomposed into per-row
   scalar work in each direction, walking a column-major matrix across its
   stride. Column-major accumulation instead: forward 1345 → 337 µs, backward
   1258 → 373 µs. The adjoint is now deferred into a per-matrix accumulator
   and flushed once after the sweep.

   **Deliberately not BLAS.** `mul!` scored 263/285 µs, so the hand kernel gets
   ~93% of the win with no LinearAlgebra dependency — and an external BLAS call
   is exactly what `--trim` is not guaranteed to survive. The stride was the
   problem, not the absence of BLAS.

2. **A logistic peephole.** `1/(1+exp(-x))` taped as four nodes; it is now one,
   with derivative `s(1-s)` read off the node's own primal.

3. **Detached operands no longer materialise a constant node.** `one(T) - p`
   recorded `OP_CONST` + `OP_SUB` where the mixed-argument rules do it in one.

4. `c/x` is one node; `x*x` is `OP_SQ` rather than an `OP_MUL` that pays two
   read-modify-write updates to the same adjoint slot.

5. `record!` used a **checked** narrowing to `Int32` on every node.

Together: an HMM tape went 17.1 → 7.05 nodes per emission.

## Two things that measurement contradicted

Both are recorded because the reasoning was plausible and wrong, and someone
will have the same idea again.

- **The `elseif` chain in `backward!` is not a linear scan.** The profile puts
  47% of self time in `==` and `OP_MATVEC` is the last of ~20 arms, which looks
  conclusive. Hoisting the hot ops to the front measured **0.86x — slower**.
  The compiler builds a jump table; those samples are attribution artifacts.
  Do not "fix" the ordering.

- **The cost was never arithmetic.** Per-node cost rose 7x (10.6 → 75 ns)
  between K=50 and K=200 while node count scaled linearly. That tracked the
  *data matrix* size, not the tape's — a memory-hierarchy signature. The tape
  was 1–2 MB; `mats` was 8 MB.

## Where reverse mode still loses

**Rosenbrock, and scalar code like it: ~3x slower than Mooncake** (44 µs vs
14 µs at K=1000). Mooncake compiles a rule per call site; a tape pays
per-node bookkeeping that dense scalar chains never amortise. Forward mode is
the better choice at small K anyway, so this is the least valuable gap.

**Against a hand-written analytic gradient, still ~5x.** A hand-derived reverse
sweep for an HMM runs at ~2.0–2.5x the primal; Piste is at ~5.3x. The tape is
the difference — 300k+ nodes is 12 MB streamed twice — and no amount of rule
tuning closes that. The honest framing is that Piste is now competitive with
the best *general* AD, not with a derivation done by hand for one model.

**The next real lever, if it is ever needed,** is shrinking `Node` below 40
bytes: a 24-byte node measured 1.54x on sweep throughput. It means dropping
the saved primal `v` and recomputing it, which is a large change to every rule
and was not attempted here.

## Do not regress

- **Trimmability was verified, not assumed.** `ProbePISTEREV` was extended to
  cover the deferred matvec, the peephole and the detached-operand paths; it
  builds under `--trim=safe` and the binary reproduces the JIT bit-for-bit
  under `JULIA_LOAD_CODEGEN_LIB=0`. Re-run it after any change here.
- **The deferred matvec accumulator is borrowed as primal scratch**, and the
  sweep skips zero-adjoint nodes — so `backward!` MUST re-zero it first, or a
  leftover primal is flushed as a gradient. There is a regression test; it
  exists because this was nearly shipped.
- **A peephole that fires when it should not gives a wrong answer, not an
  error.** The logistic fusion has six near-miss tests for exactly this.
