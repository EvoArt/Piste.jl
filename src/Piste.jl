"""
    Piste

Forward-mode automatic differentiation that survives `juliac --trim=safe`.

A *piste* is a groomed ski run: a slope, trimmed and prepared. That is the whole
package — gradients (slopes), from code slim enough to compile into a standalone
binary.

# Why this exists

Julia 1.12's `--trim` compiles a program to a small native binary with no Julia
install needed at the target. Every established AD package fails under it, each
for the same underlying reason: the derivative code is built at RUNTIME in some
form the trimmer cannot see.

Measured directly, on the same objective:

| package | under `--trim=safe` |
|---|---|
| ForwardDiff | builds with 0 verifier errors, then `MethodError`s at runtime |
| FastDifferentiation | `RuntimeGeneratedFunctions` — an `Expr` in a type parameter |
| Enzyme | `Enzyme.Compiler.thunk(...)::Any` — calls the compiler at runtime |
| Mooncake | reaches `Compiler.InferenceState` — needs the compiler in the binary |
| Differ.jl | 0 verifier errors, then dies: `ContextualInterpreter` |
| **Piste** | **0 verifier errors, and the binary matches the JIT exactly** |

The rule those measurements produced, and the one this package is built to obey:
**the derivative must exist as ordinary compiled Julia, needing nothing from the
compiler or an interpreter at any point the trimmed binary runs.**

ForwardDiff is the instructive case, because duals are *not* what breaks it. The
machinery around them is: a `GradientConfig` built at runtime, per-call-site
`Tag` types that multiply the method table, and dynamically chosen chunk widths.
So Piste has no tag, no config, and a chunk width that is a **type parameter**.

# Usage

```julia
using Piste

f(x) = sum(abs2, x) + exp(x[1]) * log(x[2])
x = [0.7, 1.3, -0.4]
g = similar(x)

gradient!(g, f, x, Val(4))            # 4 directional derivatives per pass
v, g = value_and_gradient!(g, f, x, Val(4))   # primal comes free
```

`Val(N)` is the chunk width: a `K`-parameter gradient costs `ceil(K/N)`
evaluations of `f`, with `N` lanes done at once in SIMD-friendly tuples. 8 is a
reasonable default; small problems can prefer 4.

# Scope, honestly

This is **forward mode**. Cost grows with the number of parameters, so at large
`K` a reverse-mode package will beat it by a wide margin (around `K=200`,
Mooncake is roughly 20× faster). At matched chunk width Piste is within ~15% of
ForwardDiff and agrees with it bit for bit.

Reach for Piste when you want a trimmed standalone binary — currently the only
option that works — or when `K` is small. Otherwise use reverse mode.

# Working with `Distributions.jl` and friends

`Dual <: Real`, which is the door `Distributions.logpdf` and similar generic
code walks through. Note `Distributions` is only a *test* dependency here:
nothing in the engine needs it.

Three methods beyond plain arithmetic turned out to be needed to make a whole
distribution suite work, each found by measurement rather than guessed:

  * `SpecialFunctions._logabsgamma` — the single missing method behind Gamma,
    Beta, Poisson, Binomial, NegativeBinomial, TDist and Chisq. All seven failed
    on it and nothing else.
  * two-argument `atan(y, x)` — its absence was a `StackOverflowError` (infinite
    promotion recursion), not a `MethodError`. Needed by Cauchy's `cdf`, hence
    by any `truncated(Cauchy(...), 0, Inf)` prior.
  * `isinteger` — discrete distributions call it in their support check.

If you hit a `MethodError` on some other function, that is the pattern to
follow: add the rule, and check it against finite differences.
"""
module Piste

using SpecialFunctions: SpecialFunctions

export Dual, value, partials, gradient!, value_and_gradient!, GradientWorkspace, pickchunk
export TapedReal, Tape, ReverseWorkspace, track!, backward!, reset!
export rev_gradient!, rev_value_and_gradient!

# -----------------------------------------------------------------------------
# The number type
# -----------------------------------------------------------------------------

"""
    Dual{N,T}

A value carrying `N` simultaneous directional derivatives.

`<: Real` is load-bearing, not cosmetic: it is what lets `Distributions.logpdf`
accept these at all. Distributions' methods are generic over `Real`, which is
the same door ForwardDiff walks through.
"""
struct Dual{N,T<:Real} <: Real
    v::T
    p::NTuple{N,T}
end

@inline Dual{N,T}(x::Real) where {N,T} = Dual{N,T}(T(x), _zerotup(Val(N), T))
@inline value(x::Dual) = x.v
@inline value(x::Real) = x
@inline partials(x::Dual) = x.p
@inline partials(x::Dual, i::Int) = x.p[i]

# Zero partials, unrolled at compile time (same reason as the arithmetic below).
@generated function _zerotup(::Val{N}, ::Type{T}) where {N,T}
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, [:(zero(T)) for _ in 1:N]...))
    end
end
@inline _zerop(::Type{Dual{N,T}}) where {N,T} = _zerotup(Val(N), T)

# -----------------------------------------------------------------------------
# Promotion. Without these, `2 * x` and `logpdf`'s internal mixed arithmetic
# fail to find a method.
# -----------------------------------------------------------------------------
Base.promote_rule(::Type{Dual{N,T}}, ::Type{S}) where {N,T,S<:Real} =
    Dual{N,promote_type(T, S)}
Base.promote_rule(::Type{Dual{N,T}}, ::Type{Dual{N,S}}) where {N,T,S} =
    Dual{N,promote_type(T, S)}
Base.convert(::Type{Dual{N,T}}, x::Real) where {N,T} = Dual{N,T}(x)
Base.convert(::Type{Dual{N,T}}, x::Dual{N,S}) where {N,T,S} =
    Dual{N,T}(T(x.v), map(T, x.p))
Base.convert(::Type{Dual{N,T}}, x::Dual{N,T}) where {N,T} = x

Base.zero(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(zero(T), _zerotup(Val(N), T))
Base.one(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(one(T), _zerotup(Val(N), T))
Base.zero(x::Dual) = zero(typeof(x))
Base.one(x::Dual) = one(typeof(x))
Base.eltype(::Type{Dual{N,T}}) where {N,T} = T
Base.float(x::Dual) = x
Base.typemin(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(typemin(T))
Base.typemax(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(typemax(T))

# -----------------------------------------------------------------------------
# Arithmetic.
#
# These were originally written with `ntuple(i -> ..., Val(N))`, which is
# correct and reads well, but PROFILING SHOWED IT DOES NOT VECTORIZE: at chunk
# width 8 the scalar `+` on partials was 40% of total runtime, against 8% for
# ForwardDiff, with no dynamic dispatch anywhere on either side. The gap was
# pure code generation.
#
# ForwardDiff solves this with `@generated` + `tupexpr` (see its
# `src/partials.jl`): emit a LITERAL tuple expression with every lane written
# out, so LLVM sees N independent operations it can fold into SIMD registers,
# rather than an `ntuple` closure it has to prove it can unroll first.
#
# `_tupexpr` below is the same idea. Everything stays statically resolvable, so
# it costs nothing under trim.
# -----------------------------------------------------------------------------

"""
    _tupexpr(f, N)

Build `(f(1), f(2), ..., f(N))` as a literal expression at compile time.
`f` is a function from a lane index to an `Expr`.
"""
function _tupexpr(f, N::Int)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, [f(i) for i in 1:N]...))
    end
end

@generated function _padd(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    _tupexpr(i -> :(a[$i] + b[$i]), N)
end
@generated function _psub(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    _tupexpr(i -> :(a[$i] - b[$i]), N)
end
@generated function _pneg(a::NTuple{N,T}) where {N,T}
    _tupexpr(i -> :(-a[$i]), N)
end
@generated function _pscale(a::NTuple{N,T}, s::T) where {N,T}
    _tupexpr(i -> :(a[$i] * s), N)
end
# the product rule's partial: a.p*b.v + a.v*b.p, fused per lane
@generated function _pmuladd(ap::NTuple{N,T}, bv::T, av::T, bp::NTuple{N,T}) where {N,T}
    _tupexpr(i -> :(muladd(ap[$i], bv, av * bp[$i])), N)
end
# the quotient rule's partial: (a.p - q*b.p) * inv_b
@generated function _pdiv(ap::NTuple{N,T}, bp::NTuple{N,T}, q::T, inv_b::T) where {N,T}
    _tupexpr(i -> :((ap[$i] - q * bp[$i]) * inv_b), N)
end

# Mixed Real/Dual fast paths. Without these, `X[i,j] * beta[j]` promotes the
# plain Float64 into a full Dual (materialising N zero partials) before
# multiplying — pure waste on the hottest loop in a regression model, where the
# data is always Float64 and only the parameters carry derivatives.
@inline Base.:+(a::Dual{N,T}, b::Real) where {N,T} = Dual{N,T}(a.v + T(b), a.p)
@inline Base.:+(a::Real, b::Dual{N,T}) where {N,T} = Dual{N,T}(T(a) + b.v, b.p)
@inline Base.:-(a::Dual{N,T}, b::Real) where {N,T} = Dual{N,T}(a.v - T(b), a.p)
@inline Base.:-(a::Real, b::Dual{N,T}) where {N,T} = Dual{N,T}(T(a) - b.v, _pneg(b.p))
@inline Base.:*(a::Dual{N,T}, b::Real) where {N,T} = Dual{N,T}(a.v * T(b), _pscale(a.p, T(b)))
@inline Base.:*(a::Real, b::Dual{N,T}) where {N,T} = Dual{N,T}(T(a) * b.v, _pscale(b.p, T(a)))
@inline Base.:/(a::Dual{N,T}, b::Real) where {N,T} =
    (inv_b = one(T) / T(b); Dual{N,T}(a.v * inv_b, _pscale(a.p, inv_b)))

@inline Base.:+(a::Dual{N,T}, b::Dual{N,T}) where {N,T} =
    Dual{N,T}(a.v + b.v, _padd(a.p, b.p))
@inline Base.:-(a::Dual{N,T}, b::Dual{N,T}) where {N,T} =
    Dual{N,T}(a.v - b.v, _psub(a.p, b.p))
@inline Base.:-(a::Dual{N,T}) where {N,T} =
    Dual{N,T}(-a.v, _pneg(a.p))
@inline Base.:*(a::Dual{N,T}, b::Dual{N,T}) where {N,T} =
    Dual{N,T}(a.v * b.v, _pmuladd(a.p, b.v, a.v, b.p))
@inline function Base.:/(a::Dual{N,T}, b::Dual{N,T}) where {N,T}
    inv_b = one(T) / b.v
    q = a.v * inv_b
    return Dual{N,T}(q, _pdiv(a.p, b.p, q, inv_b))
end

# Three-argument muladd. THIS IS THE ONE THAT MATTERS FOR SPEED.
#
# `X * beta` with `beta::Vector{Dual}` goes through Julia's generic
# `__generic_matvecmul!`, whose inner loop is `s = muladd(A[i,j], x[j], s)`.
# Without a Dual method for the 3-argument form, that call promotes and falls
# back to a separate `*` then `+`, building a temporary Dual per iteration.
#
# Measured on X(200x50) * beta: without this, width 8 took 138.5us against
# ForwardDiff's 22.9us -- a 6x gap, on IDENTICAL memory layout (same sizeof,
# both isbits). Profiling put 79.7% of total gradient time inside
# `__generic_matvecmul!`. ForwardDiff has had `calc_muladd_xyz` (its
# `dual.jl:674`) for exactly this reason.
#
# Fusing all three operands per lane also lets LLVM emit real FMA instructions
# rather than a multiply and an add.
@generated function _pmuladd3(xp::NTuple{N,T}, yp::NTuple{N,T}, zp::NTuple{N,T},
                              xv::T, yv::T) where {N,T}
    _tupexpr(i -> :(muladd(xv, yp[$i], muladd(yv, xp[$i], zp[$i]))), N)
end

@inline function Base.muladd(x::Dual{N,T}, y::Dual{N,T}, z::Dual{N,T}) where {N,T}
    return Dual{N,T}(muladd(x.v, y.v, z.v), _pmuladd3(x.p, y.p, z.p, x.v, y.v))
end
# Mixed forms: a data matrix is plain Float64, so `muladd(::Float64, ::Dual,
# ::Dual)` is the shape the matvec loop actually hits.
@inline Base.muladd(x::Real, y::Dual{N,T}, z::Dual{N,T}) where {N,T} =
    Dual{N,T}(muladd(T(x), y.v, z.v), _pscale_add(y.p, T(x), z.p))
@inline Base.muladd(x::Dual{N,T}, y::Real, z::Dual{N,T}) where {N,T} =
    Dual{N,T}(muladd(x.v, T(y), z.v), _pscale_add(x.p, T(y), z.p))
@inline Base.muladd(x::Dual{N,T}, y::Dual{N,T}, z::Real) where {N,T} =
    Dual{N,T}(muladd(x.v, y.v, T(z)), _pmuladd(x.p, y.v, x.v, y.p))

@generated function _pscale_add(ap::NTuple{N,T}, s::T, zp::NTuple{N,T}) where {N,T}
    _tupexpr(i -> :(muladd(ap[$i], s, zp[$i])), N)
end

# Unary chain rule: given f(v) and f'(v), propagate.
@inline function _chain(x::Dual{N,T}, fv::T, dfv::T) where {N,T}
    return Dual{N,T}(fv, _pscale(x.p, dfv))
end

@inline Base.exp(x::Dual{N,T}) where {N,T} = (e = exp(x.v); _chain(x, e, e))
@inline Base.log(x::Dual{N,T}) where {N,T} = _chain(x, log(x.v), one(T) / x.v)
@inline Base.sqrt(x::Dual{N,T}) where {N,T} = (s = sqrt(x.v); _chain(x, s, one(T) / (2 * s)))
@inline Base.sin(x::Dual{N,T}) where {N,T} = _chain(x, sin(x.v), cos(x.v))
@inline Base.cos(x::Dual{N,T}) where {N,T} = _chain(x, cos(x.v), -sin(x.v))
@inline Base.tan(x::Dual{N,T}) where {N,T} = (t = tan(x.v); _chain(x, t, one(T) + t * t))
@inline Base.tanh(x::Dual{N,T}) where {N,T} = (t = tanh(x.v); _chain(x, t, one(T) - t * t))
@inline Base.atan(x::Dual{N,T}) where {N,T} = _chain(x, atan(x.v), one(T) / (one(T) + x.v^2))
@inline Base.abs(x::Dual{N,T}) where {N,T} = x.v < 0 ? -x : x
@inline Base.inv(x::Dual{N,T}) where {N,T} = _chain(x, inv(x.v), -one(T) / (x.v * x.v))
@inline Base.expm1(x::Dual{N,T}) where {N,T} = _chain(x, expm1(x.v), exp(x.v))
@inline Base.log1p(x::Dual{N,T}) where {N,T} = _chain(x, log1p(x.v), one(T) / (one(T) + x.v))

# Two-argument atan. WITHOUT this, `atan(y, x)` promotes both arguments and
# recurses forever -- a StackOverflowError, not a MethodError, which is why it
# looked scarier than it is. Cauchy's `cdf` needs it, so `truncated(Cauchy(...))`
# depends on it.
#   d/dy atan(y,x) =  x/(x^2+y^2)
#   d/dx atan(y,x) = -y/(x^2+y^2)
@generated function _patan2(yp::NTuple{N,T}, xp::NTuple{N,T}, yv::T, xv::T, den::T) where {N,T}
    _tupexpr(i -> :((xv * yp[$i] - yv * xp[$i]) / den), N)
end

@inline function Base.atan(y::Dual{N,T}, x::Dual{N,T}) where {N,T}
    den = y.v * y.v + x.v * x.v
    # An infinite argument arises from a truncation bound at ±Inf (e.g.
    # `truncated(Cauchy(0,s), 0, Inf)` evaluates `cdf` at Inf). There `den` is
    # Inf and the naive quotient is Inf/Inf = NaN, which then poisons the whole
    # gradient. The mathematically right answer is that atan saturates, so its
    # sensitivity to the parameters is exactly zero.
    if !isfinite(den)
        return Dual{N,T}(atan(y.v, x.v), _zerotup(Val(N), T))
    end
    return Dual{N,T}(atan(y.v, x.v),
                     _patan2(y.p, x.p, y.v, x.v, den))
end
@inline Base.atan(y::Dual{N,T}, x::Real) where {N,T} = atan(y, Dual{N,T}(x))
@inline Base.atan(y::Real, x::Dual{N,T}) where {N,T} = atan(Dual{N,T}(y), x)

@inline function Base.:^(x::Dual{N,T}, n::Integer) where {N,T}
    n == 0 && return one(Dual{N,T})
    return _chain(x, x.v^n, T(n) * x.v^(n - 1))
end
@inline function Base.:^(x::Dual{N,T}, y::T) where {N,T}
    fv = x.v^y
    return _chain(x, fv, y * x.v^(y - one(T)))
end
# Literal integer powers. `x^2` in user code is lowered to
# `literal_pow(^, x, Val(2))`, so this — not the `^(::Dual, ::Integer)` method
# above — is what a squaring in a hot loop actually calls.
#
# The method below USED to be written as `_chain(x, x.v^p, T(p) * x.v^(p-1))`.
# It did fire (it is not, as was once suspected, unregistered), but both `^`s in
# that body are RUNTIME powers on a Float64, and `^(::Float64, ::Int)` goes to
# `Base.Math.pow_body` and the libm power path. Profiling put `^`/`pow_body` at
# 19.7% of a Rosenbrock gradient purely because of this — squaring a dual ought
# to be one multiply.
#
# The fix is to enumerate the small exponents so the value work is literal
# multiplies. `p` is a type parameter, so each is its own specialization and
# the branch is resolved at compile time. Same shape ForwardDiff uses, and it
# stays entirely statically resolvable, so it costs nothing under trim.
@inline Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{0}) where {N,T} =
    one(Dual{N,T})
@inline Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{1}) where {N,T} = x
@inline Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{2}) where {N,T} =
    _chain(x, x.v * x.v, T(2) * x.v)
@inline function Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{3}) where {N,T}
    v2 = x.v * x.v
    return _chain(x, v2 * x.v, T(3) * v2)
end
# General literal exponent: still better than the old body, because
# `Base.literal_pow` on a Float64 has its own optimised path that the plain
# `^(::Float64, ::Int)` call did not reach.
@inline Base.literal_pow(::typeof(^), x::Dual{N,T}, ::Val{p}) where {N,T,p} =
    _chain(x, Base.literal_pow(^, x.v, Val(p)), T(p) * Base.literal_pow(^, x.v, Val(p - 1)))

# Comparisons act on the value only — this is what lets `logpdf`'s support
# checks (`insupport`, `x < 0`, clamping) work unchanged on a Dual.
for op in (:(==), :<, :<=, :>, :>=)
    @eval Base.$op(a::Dual, b::Dual) = $op(a.v, b.v)
    @eval Base.$op(a::Dual, b::Real) = $op(a.v, b)
    @eval Base.$op(a::Real, b::Dual) = $op(a, b.v)
end
Base.isless(a::Dual, b::Dual) = isless(a.v, b.v)
Base.isnan(x::Dual) = isnan(x.v)
Base.isinf(x::Dual) = isinf(x.v)
Base.isfinite(x::Dual) = isfinite(x.v)
# Discrete distributions (Poisson, Binomial) call `isinteger` on the OBSERVED
# value during their support check. Predicates like this always answer about
# the value; a derivative has no bearing on whether a number is an integer.
Base.isinteger(x::Dual) = isinteger(x.v)
Base.round(x::Dual) = round(x.v)
Base.floor(x::Dual) = floor(x.v)
Base.ceil(x::Dual) = ceil(x.v)
Base.trunc(x::Dual) = trunc(x.v)

# -----------------------------------------------------------------------------
# log-gamma, and why it is written the way it is.
#
# `SpecialFunctions._logabsgamma` is THE single missing method behind every
# lgamma-family failure: Gamma, Beta, Poisson, Binomial, NegativeBinomial,
# TDist and Chisq all fail on exactly this one function and nothing else. It is
# a missing rule, not a design problem.
#
# Deliberately implemented from scratch (Lanczos) rather than by calling
# SpecialFunctions on the value and attaching a digamma partial. Two reasons:
#
#   1. It keeps the hot path free of an external call, which matters under trim.
#   2. **It is a prototype for the STADE side.** STADE rejects `lgamma` outright
#      (no derivative rule) but supports user-supplied helper kernels inlined
#      from the same file. The recurrence + Lanczos series below is exactly what
#      that helper kernel needs to contain, and proving its accuracy HERE -- in
#      a normal JIT session, against SpecialFunctions, with no build step -- is
#      far cheaper than discovering an error budget problem inside a compiled
#      binary. Whichever path ships, this is the same arithmetic.
#
# Only the real-argument, x > 0 branch is implemented; that is the whole of what
# a log-density normalizing constant needs.
# -----------------------------------------------------------------------------

# Lanczos g=7, n=9. Standard coefficients.
const _LANCZOS = (
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
)

"""
    lgamma_lanczos(x)

`log Γ(x)` for real `x > 0`, and its derivative `digamma(x)` implicitly via the
dual arithmetic — the point of writing it in terms of `+ - * / log exp` only is
that those all have rules above, so differentiating it needs no extra work.

The same expression tree transcribes directly into a STADE helper kernel.
"""
@inline function lgamma_lanczos(x::Union{Real,Dual})
    # Reflection is not needed for x > 0, so this is the direct branch:
    #   Γ(x) = Γ(x+1)/x, shifted so the series argument is >= 1
    z = x - 1
    a = _LANCZOS[1]
    t = z + 7.5
    # sum_{k=1..8} c_k / (z + k)
    a += _LANCZOS[2] / (z + 1)
    a += _LANCZOS[3] / (z + 2)
    a += _LANCZOS[4] / (z + 3)
    a += _LANCZOS[5] / (z + 4)
    a += _LANCZOS[6] / (z + 5)
    a += _LANCZOS[7] / (z + 6)
    a += _LANCZOS[8] / (z + 7)
    a += _LANCZOS[9] / (z + 8)
    # log(sqrt(2pi)) + (z+0.5)log(t) - t + log(a)
    return 0.9189385332046727 + (z + 0.5) * log(t) - t + log(a)
end

# Hook into SpecialFunctions' internal entry point, which is what Distributions
# actually calls. It returns (logabs, sign); for x > 0 the sign is always +1.
function SpecialFunctions._logabsgamma(x::Dual{N,T}) where {N,T}
    return (lgamma_lanczos(x), 1)
end
SpecialFunctions.loggamma(x::Dual) = lgamma_lanczos(x)
SpecialFunctions.logabsgamma(x::Dual) = (lgamma_lanczos(x), 1)

# -----------------------------------------------------------------------------
# The gradient driver
# -----------------------------------------------------------------------------

"""
    pickchunk(K::Int) -> Val

A reasonable chunk width for a `K`-parameter gradient, as a `Val` ready to hand
to `gradient!`.

```julia
gradient!(g, f, x, pickchunk(length(x)))
```

The width matters a lot — at K=200 the wrong choice is a 5x difference — and the
old fixed default of 8 was a poor one everywhere except very small problems.

The thresholds below are measured, not derived, on the Rosenbrock objective at
Julia 1.12 (minimum-of-samples, zero-allocation workspace path, idle machine):

| K | best width | 8 (old default) | speedup |
|---|---|---|---|
| 2 | 2 | 47 ns | 1.3x |
| 5 | 8 | 60 ns | 1.0x |
| 10 | 16 | 272 ns | 2.0x |
| 20 | 24 | 566 ns | 1.9x |
| 50 | 32 | 3788 ns | 1.7x |
| 100 | 64 | 15800 ns | 1.7x |
| 200 | 64 | 53600 ns | 1.8x |
| 400 | 32 | 222000 ns | 1.8x |

Two forces trade off. A wider chunk means fewer passes over `f`, but an
`NTuple{N,T}` wider than the machine's vector registers spills, and the seeding
loop does `K` work per pass regardless. The optimum therefore rises with `K` and
then flattens — 32 and 64 are within a few percent of each other from K=100 up,
so the exact cutoff there is not critical.

This returns a `Val`, so at a call site where `K` is not a compile-time constant
the result is type-unstable — one dynamic dispatch into `gradient!`, then
everything inside is static again. That is fine for a JIT session, but **under
`--trim` pass a literal `Val(N)` instead**: the trimmer needs the width to be
statically known, which is exactly why the width is a type parameter in the
first place. Treat this as a convenience for interactive and JIT use, and pin
the width explicitly in code you intend to trim.
"""
function pickchunk(K::Int)
    K <= 3 && return Val(2)
    K <= 6 && return Val(4)
    K <= 8 && return Val(8)
    K <= 14 && return Val(16)
    K <= 30 && return Val(24)
    K <= 80 && return Val(32)
    return Val(64)
end

"""
    GradientWorkspace{N,T}(K)
    GradientWorkspace(x::Vector{T}, ::Val{N})

Reusable scratch space for `gradient!` / `value_and_gradient!`.

The drivers need a `Vector{Dual{N,T}}` of length `K` to seed into. Allocating it
per call cost 52 887 bytes on a K=200 gradient and put `GenericMemory` at 54.5%
of self time, flagged for GC — while ForwardDiff, which keeps the same buffer in
its `GradientConfig`, allocated nothing. Passing a workspace closes that gap:

```julia
ws = GradientWorkspace(x, Val(32))
for i in 1:nsteps
    gradient!(g, f, x, Val(32), ws)   # 0 bytes
end
```

The no-workspace methods still work and still allocate; this is an opt-in for
hot loops such as a sampler's leapfrog step.

# Why a struct and not a closure

This is deliberately a plain concrete struct holding a concrete `Vector`, with
`N` and `T` as type parameters. A workspace captured in a closure instead would
reintroduce exactly the `Core.Box` problem that `_grad_chunk!` was split out to
avoid — a boxed capture infers as `Any` and produces verifier errors under
`juliac --trim=safe`. Everything here stays statically resolvable.
"""
struct GradientWorkspace{N,T<:Real}
    xd::Vector{Dual{N,T}}
end

GradientWorkspace{N,T}(K::Int) where {N,T<:Real} =
    GradientWorkspace{N,T}(Vector{Dual{N,T}}(undef, K))
GradientWorkspace(x::Vector{T}, ::Val{N}) where {T<:Real,N} =
    GradientWorkspace{N,T}(length(x))

Base.length(ws::GradientWorkspace) = length(ws.xd)

# The buffer must be at least as long as `x`. Growing it here rather than
# erroring means a workspace stays usable if the caller's problem size changes;
# in the steady state (same `K` every call) this never fires, so the hot path
# keeps its zero-allocation property.
@inline function _ensure!(ws::GradientWorkspace{N,T}, K::Int) where {N,T}
    length(ws.xd) < K && resize!(ws.xd, K)
    return ws.xd
end

"""
    gradient!(g, f, x, ::Val{N})
    gradient!(g, f, x, ::Val{N}, ws::GradientWorkspace{N,T})

Gradient of `f` at `x` into `g`, using `N` partials per pass.

Costs `ceil(length(x)/N)` evaluations of `f`. Chunking is the whole reason for
`N`: one pass with N=8 does 8 directional derivatives in SIMD-width lanes,
rather than 8 separate passes.

Pass a [`GradientWorkspace`](@ref) to reuse the dual buffer across calls and
allocate nothing; without one, a fresh buffer is allocated per call.

Everything about the shape here is static, so the trimmer can see through it.
"""
function gradient!(g::Vector{T}, f::F, x::Vector{T}, ::Val{N},
                   ws::GradientWorkspace{N,T}) where {T,F,N}
    K = length(x)
    xd = _ensure!(ws, K)
    nchunks = cld(K, N)
    for c in 1:nchunks
        _grad_chunk!(g, f, x, xd, (c - 1) * N, Val(N))
    end
    return g
end

function gradient!(g::Vector{T}, f::F, x::Vector{T}, ::Val{N}) where {T,F,N}
    return gradient!(g, f, x, Val(N), GradientWorkspace(x, Val(N)))
end

# One chunk, in its own function ON PURPOSE.
#
# Written inline in a `while` loop with a mutated `offset`, the `ntuple`
# closure captures that variable, so Julia boxes it (`Core.Box`) and every use
# infers as `Any`. Under `juliac --trim=safe` that produced 14 verifier errors
# -- all of the form `Core.getfield(%new()::Core.Box, :contents)::Any`, none of
# them anything to do with the duals themselves.
#
# Passing `offset` as an ARGUMENT makes it a plain immutable local that the
# closure captures by value, so nothing is boxed and every call resolves
# statically. This is a Julia closure-capture detail, not an AD one, but it is
# the difference between trimmable and not.
# Seed tuple: lane `lane` is 1, the rest 0. Runs K times per chunk, so it is
# firmly on the hot path and gets the same unrolling treatment.
@generated function _seedtup(::Val{N}, ::Type{T}, lane::Int) where {N,T}
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, [:(lane == $i ? one(T) : zero(T)) for i in 1:N]...))
    end
end

@inline function _grad_chunk!(g::Vector{T}, f::F, x::Vector{T},
                              xd::Vector{Dual{N,T}}, offset::Int, ::Val{N}) where {T,F,N}
    K = length(x)
    @inbounds for i in 1:K
        # seed lane j iff this element is the j-th in the current chunk
        xd[i] = Dual{N,T}(x[i], _seedtup(Val(N), T, i - offset))
    end
    yd = f(xd)
    p = partials(yd)
    @inbounds for j in 1:N
        idx = offset + j
        idx <= K && (g[idx] = p[j])
    end
    # Returned so `value_and_gradient!` can take the primal from the first pass
    # instead of paying for an extra evaluation.
    return value(yd)
end

"""
    value_and_gradient!(g, f, x, ::Val{N}) -> (value, g)
    value_and_gradient!(g, f, x, ::Val{N}, ws::GradientWorkspace{N,T}) -> (value, g)

As `gradient!`, also returning the primal value.

The primal comes out of the FIRST dual pass rather than a separate `f(x)` call:
every pass already computes it, so re-evaluating would add a whole extra model
evaluation to every gradient — a real cost on the sampler's hot path, where this
is called once per leapfrog step.
"""
function value_and_gradient!(g::Vector{T}, f::F, x::Vector{T}, ::Val{N},
                             ws::GradientWorkspace{N,T}) where {T,F,N}
    K = length(x)
    K == 0 && return (f(x), g)
    xd = _ensure!(ws, K)
    v = _grad_chunk!(g, f, x, xd, 0, Val(N))
    for c in 2:cld(K, N)
        _grad_chunk!(g, f, x, xd, (c - 1) * N, Val(N))
    end
    return v, g
end

function value_and_gradient!(g::Vector{T}, f::F, x::Vector{T}, ::Val{N}) where {T,F,N}
    return value_and_gradient!(g, f, x, Val(N), GradientWorkspace(x, Val(N)))
end

# Reverse mode lives in its own file: forward and reverse share only the
# `SpecialFunctions` import and the `value` accessor, and keeping them apart
# means the two can be worked on independently.
include("reverse.jl")

end # module Piste
