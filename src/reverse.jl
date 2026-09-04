# =============================================================================
# Reverse mode: a tape that survives `juliac --trim`.
#
# WHY REVERSE MODE IS HERE AT ALL
# -------------------------------
# Forward mode (`Dual`, in Piste.jl) costs `ceil(K/N)` evaluations for a
# K-parameter gradient. That is the right trade at small K, and hopeless at
# large K -- on a 200-parameter objective forward mode is ~19x slower than the
# tape below. A log-density is many-inputs-to-one-scalar, which is exactly the
# shape reverse mode is for, so for real models this is usually the mode you
# want.
#
# WHY NOT JUST USE MOONCAKE OR ENZYME
# -----------------------------------
# Because neither trims, and for reasons that are the mechanism rather than a
# bug. Both derive their rules from the function's IR: Mooncake runs abstract
# interpretation (it reaches `Compiler.InferenceState`), Enzyme builds a thunk
# by invoking the compiler at runtime (`Enzyme.Compiler.thunk(...)::Any`).
# Deriving rules from IR *means running the compiler*, and `--trim` exists to
# remove the compiler. Measured: 48 and 1276 verifier errors respectively.
#
# Differ.jl was tried as the compile-time alternative (`@generated` codegen, so
# on the right side of that line). It builds with 0 verifier errors and then
# dies at runtime -- `ContextualInterpreter could not build the reverse forwards
# pass` -- which refined the rule: compile-time codegen is necessary but NOT
# sufficient. The derivative must need nothing from the compiler *or an
# interpreter* at any point the trimmed binary runs.
#
# That leaves hand-written rules over a tracked type, which is ReverseDiff's
# shape (and Stan's `var`/`vari`, and Adept's). Old because it is simple, not
# because it is wrong.
#
# THE THREE CHOICES THAT MAKE THE TAPE TRIM
# -----------------------------------------
# Tapes are usually where trimmability dies: a `Vector{Any}` of heterogeneous
# operations, a closure per node, dynamic dispatch on playback. All three are
# avoided deliberately:
#
#   1. **One concrete node type.** The tape is `Vector{Node}` where `Node` is a
#      single isbits struct -- a dense array, not a vector of boxes.
#   2. **Operations are `Int8` tags.** A closure per node would box; a type per
#      operation would make the tape heterogeneous. A tag keeps dispatch a plain
#      branch.
#   3. **Playback is one loop with a static branch.** Every arm of the `if op ==
#      ...` chain is known at compile time.
#
# `TapedReal` is likewise isbits (an `Int32` index plus the value), so a
# `Vector{TapedReal}` is dense.
#
# Verified: 0 verifier errors under `--trim=safe`, and the binary reproduces the
# JIT's gradient exactly under `JULIA_LOAD_CODEGEN_LIB=0`.
# =============================================================================

# --- operation tags ----------------------------------------------------------
# Deliberately `Int8` constants rather than an `@enum`: an enum is a real type
# whose `instances`/`show` machinery the trimmer would have to retain.
const OP_INPUT  = Int8(0)   # a leaf; its adjoint IS the gradient
const OP_CONST  = Int8(1)   # a constant; its adjoint stops here
const OP_ADD    = Int8(2)
const OP_SUB    = Int8(3)
const OP_MUL    = Int8(4)
const OP_DIV    = Int8(5)
const OP_MULC   = Int8(6)   # times a saved constant (c)
const OP_ADDC   = Int8(7)   # plus a saved constant
const OP_NEG    = Int8(8)
const OP_SQ     = Int8(9)
const OP_EXP    = Int8(10)
const OP_LOG    = Int8(11)
const OP_SQRT   = Int8(12)
const OP_SIN    = Int8(13)
const OP_COS    = Int8(14)
const OP_TAN    = Int8(15)
const OP_TANH   = Int8(16)
const OP_ATAN   = Int8(17)
const OP_ATAN2  = Int8(18)
const OP_ABS    = Int8(19)
const OP_LOG1P  = Int8(20)
const OP_EXPM1  = Int8(21)
const OP_LGAMMA = Int8(22)
# --- statement-level (fused) ops. See `Node`'s docstring for why these exist. --
const OP_FMA    = Int8(23)  # a*b + c-as-a-node: two inputs, partials in c/d
const OP_LINEAR = Int8(24)  # a*ca + b*cb with CONSTANT coefficients
const OP_DOT    = Int8(25)  # a dot product: one node for the whole reduction

"""
    Node

One tape entry. Concrete and isbits by design — see this file's header for why
that is what makes the tape trimmable.

`a`/`b` are tape indices of the inputs (0 when unused), `v` is the primal saved
for the reverse sweep, and `c`/`d` are saved partials (or a saved scalar, or a
precomputed denominator, depending on the op).

Two partial slots rather than one is what allows STATEMENT-LEVEL nodes: a fused
`muladd(a, b, c)` records ∂/∂a and ∂/∂b in one entry instead of three separate
`*` and `+` entries. The struct grows 32 -> 40 bytes, but replacing three nodes
with one is a clear net win in both memory and sweep time.

This is the central lesson from the C++ AD literature. Adept (TOMS 2014) records
differential *statements* rather than individual operations and reports its
reverse pass as 2.3-12x faster than ADOL-C and CppAD, which store a symbolic
representation of every operator. Operation-level taping is the slow way to do
this.
"""
struct Node
    op::Int8
    a::Int32
    b::Int32
    v::Float64
    c::Float64   # first saved partial (or a saved scalar/denominator)
    d::Float64   # second saved partial, for fused two-input nodes
end

"""
    Tape

The recording. Carries its own length so it can be [`reset!`](@ref) and refilled
without releasing storage.

That matters more than it sounds: profiling the naive version put **86% of total
runtime in `push!`** (`GenericMemory` allocation, `memmove`, `_growend!`),
because the Vector reallocated and copied on every gradient call. A sampler
evaluates the same model shape once per leapfrog step, so the node count is
known after the first sweep and the storage should simply be reused. Reuse was
worth ~2x at K=200.
"""
mutable struct Tape
    nodes::Vector{Node}
    n::Int
end

Tape() = Tape(Node[], 0)

"""
    Tape(capacity::Int)

Preallocate for a known node count, so even the first sweep does not grow.
"""
Tape(capacity::Integer) = Tape(Vector{Node}(undef, Int(capacity)), 0)

"""
    reset!(t) -> t

Rewind to empty WITHOUT releasing storage, so the next sweep refills the same
memory. This is what makes repeated gradient calls cheap.
"""
@inline function reset!(t::Tape)
    t.n = 0
    return t
end

@inline Base.length(t::Tape) = t.n
@inline Base.isempty(t::Tape) = t.n == 0

"""
    TapedReal <: Real

A value being recorded: a tape index plus its primal.

`<: Real` is load-bearing exactly as it is for [`Dual`](@ref) — it is what lets
`Distributions.logpdf` and other generic numeric code accept these unmodified.
isbits, so a `Vector{TapedReal}` is a dense array rather than a vector of boxes.

The tape it records onto is passed explicitly rather than kept in a global:
a non-const global would infer as `Any` and take the whole hot path with it.
"""
struct TapedReal <: Real
    tape::Tape
    idx::Int32
    v::Float64
end

@inline value(x::TapedReal) = x.v
@inline tape(x::TapedReal) = x.tape

# A shared, never-swept tape standing for "this value belongs to no recording".
# Defined here because `record!` guards against it.
const _DETACHED = Tape()

# Recording onto the detached tape is always a bug: it is shared and never
# swept, so anything appended there is invisible to the reverse pass while
# inflating indices. Guarding here catches every path at once, including any
# rule added later that forgets the check.
@inline function record!(t::Tape, op::Int8, a::Int32, b::Int32, v::Float64, c::Float64,
                         d::Float64=0.0)
    t === _DETACHED && return TapedReal(_DETACHED, Int32(0), v)
    i = t.n + 1
    t.n = i
    # Grow geometrically only when the reused storage is genuinely exhausted.
    # After the first sweep of a given model this branch is never taken.
    if i > length(t.nodes)
        resize!(t.nodes, max(16, 2 * length(t.nodes)))
    end
    @inbounds t.nodes[i] = Node(op, a, b, v, c, d)
    return TapedReal(t, Int32(i), v)
end

"""
    track!(t, x) -> TapedReal

Put an input onto the tape. Its adjoint after the reverse sweep is that input's
gradient component.
"""
@inline track!(t::Tape, x::Real) =
    record!(t, OP_INPUT, Int32(0), Int32(0), Float64(x), 0.0)

# A constant becomes a REAL node with no inputs, not a sentinel index.
#
# This was a genuine bug worth recording: constants originally got `idx = 0`,
# but 0 is not a valid tape index, so the reverse sweep's `adj[nd.a]` for such a
# node indexed outside the adjoint array. Three of the first four distributions
# tested returned `Inf`/`-0.0` gradients while running without any error — the
# silent-wrong-answer failure mode this package exists to avoid.
@inline constant!(t::Tape, x::Real) =
    record!(t, OP_CONST, Int32(0), Int32(0), Float64(x), 0.0)

# -----------------------------------------------------------------------------
# Rules. Ordinary methods, written by hand — nothing derived from IR, so nothing
# here needs the compiler at runtime.
# -----------------------------------------------------------------------------

# Each of these re-attaches a detached operand (from `zero(T)`/`one(T)`, e.g. a
# `sum` accumulator) onto the live tape first, so no node ever records index 0
# as an input.
@inline function Base.:+(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v + y.v)
    t = _livetape(x, y)
    a = _attach(t, x); b = _attach(t, y)
    # PEEPHOLE FUSION. Generic matvec does not call `muladd` on these types --
    # measured, it emits a `*` then a `+`, giving an OP_MULC and an OP_ADD per
    # term (2406 and 2415 of them on a 200x10 regression, 28.4 nodes per
    # observation). Folding `s + x*c` back into one OP_LINEAR halves that.
    #
    # Safe because it only fires when the operand is the node THIS tape recorded
    # last: nothing else can have referenced it yet, so rewriting it cannot
    # change any other node's meaning. Anything else falls through to a plain add.
    if _is_last_mulc(t, b)
        t.n -= 1                                   # drop the OP_MULC
        nd = @inbounds t.nodes[t.n + 1]
        return record!(t, OP_LINEAR, a.idx, nd.a, a.v + b.v, 1.0, nd.c)
    elseif _is_last_mulc(t, a)
        t.n -= 1
        nd = @inbounds t.nodes[t.n + 1]
        return record!(t, OP_LINEAR, b.idx, nd.a, a.v + b.v, 1.0, nd.c)
    end
    return record!(t, OP_ADD, a.idx, b.idx, a.v + b.v, 0.0)
end

# True when `x` is the most recently recorded node on `t` AND is a scale-by-
# constant, i.e. exactly the `x*c` half of an `s + x*c` statement.
@inline function _is_last_mulc(t::Tape, x::TapedReal)
    x.idx == Int32(t.n) || return false
    @inbounds return t.nodes[t.n].op === OP_MULC
end
@inline function Base.:-(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v - y.v)
    t = _livetape(x, y)
    a = _attach(t, x); b = _attach(t, y)
    return record!(t, OP_SUB, a.idx, b.idx, a.v - b.v, 0.0)
end
@inline function Base.:*(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v * y.v)
    t = _livetape(x, y)
    a = _attach(t, x); b = _attach(t, y)
    return record!(t, OP_MUL, a.idx, b.idx, a.v * b.v, 0.0)
end
@inline function Base.:/(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v / y.v)
    t = _livetape(x, y)
    a = _attach(t, x); b = _attach(t, y)
    return record!(t, OP_DIV, a.idx, b.idx, a.v / b.v, 0.0)
end
@inline Base.:-(x::TapedReal) =
    record!(x.tape, OP_NEG, x.idx, Int32(0), -x.v, 0.0)

# Mixed Real/TapedReal fast paths: data is plain Float64 and only parameters are
# tracked, so promoting a scalar to a full tape node on every operation would be
# pure waste on the hottest loop in a regression model.
@inline Base.:*(x::TapedReal, c::Real) =
    record!(x.tape, OP_MULC, x.idx, Int32(0), x.v * Float64(c), Float64(c))
@inline Base.:*(c::Real, x::TapedReal) = x * c
@inline Base.:+(x::TapedReal, c::Real) =
    record!(x.tape, OP_ADDC, x.idx, Int32(0), x.v + Float64(c), Float64(c))
@inline Base.:+(c::Real, x::TapedReal) = x + c
@inline Base.:-(x::TapedReal, c::Real) = x + (-Float64(c))
@inline Base.:-(c::Real, x::TapedReal) = (-x) + Float64(c)
@inline Base.:/(x::TapedReal, c::Real) = x * (1.0 / Float64(c))
@inline Base.:/(c::Real, x::TapedReal) = constant!(x.tape, c) / x

@inline Base.exp(x::TapedReal)   = record!(x.tape, OP_EXP,   x.idx, Int32(0), exp(x.v), 0.0)
@inline Base.log(x::TapedReal)   = record!(x.tape, OP_LOG,   x.idx, Int32(0), log(x.v), 0.0)
@inline Base.sqrt(x::TapedReal)  = record!(x.tape, OP_SQRT,  x.idx, Int32(0), sqrt(x.v), 0.0)
@inline Base.sin(x::TapedReal)   = record!(x.tape, OP_SIN,   x.idx, Int32(0), sin(x.v), 0.0)
@inline Base.cos(x::TapedReal)   = record!(x.tape, OP_COS,   x.idx, Int32(0), cos(x.v), 0.0)
@inline Base.tan(x::TapedReal)   = record!(x.tape, OP_TAN,   x.idx, Int32(0), tan(x.v), 0.0)
@inline Base.tanh(x::TapedReal)  = record!(x.tape, OP_TANH,  x.idx, Int32(0), tanh(x.v), 0.0)
@inline Base.atan(x::TapedReal)  = record!(x.tape, OP_ATAN,  x.idx, Int32(0), atan(x.v), 0.0)
@inline Base.abs(x::TapedReal)   = record!(x.tape, OP_ABS,   x.idx, Int32(0), abs(x.v), 0.0)
@inline Base.log1p(x::TapedReal) = record!(x.tape, OP_LOG1P, x.idx, Int32(0), log1p(x.v), 0.0)
@inline Base.expm1(x::TapedReal) = record!(x.tape, OP_EXPM1, x.idx, Int32(0), expm1(x.v), 0.0)
@inline Base.abs2(x::TapedReal) =
    record!(x.tape, OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0)
@inline Base.inv(x::TapedReal) =
    record!(x.tape, OP_MULC, x.idx, Int32(0), inv(x.v), -inv(x.v * x.v))

# Two-argument atan. Without it both arguments promote and recurse forever — a
# StackOverflowError, not a MethodError, which makes it look worse than it is.
# Cauchy's `cdf` needs it, hence any `truncated(Cauchy(...), 0, Inf)` prior.
#
# The infinite-argument guard is not optional: a truncation bound at `Inf` makes
# the denominator infinite, and the naive quotient is `Inf/Inf = NaN`, which
# then poisons the whole gradient while the VALUE stays perfectly correct.
@inline function Base.atan(y::TapedReal, x::TapedReal)
    den = y.v * y.v + x.v * x.v
    v = atan(y.v, x.v)
    isfinite(den) || return constant!(y.tape, v)
    return record!(y.tape, OP_ATAN2, y.idx, x.idx, v, den)
end
@inline Base.atan(y::TapedReal, x::Real) = atan(y, constant!(y.tape, x))
@inline Base.atan(y::Real, x::TapedReal) = atan(constant!(x.tape, y), x)

@inline Base.:^(x::TapedReal, n::Integer) =
    n == 2 ? record!(x.tape, OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0) :
             record!(x.tape, OP_MULC, x.idx, Int32(0), x.v^n, n * x.v^(n - 1))
@inline Base.literal_pow(::typeof(^), x::TapedReal, ::Val{2}) =
    record!(x.tape, OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0)
@inline Base.literal_pow(::typeof(^), x::TapedReal, ::Val{p}) where {p} =
    record!(x.tape, OP_MULC, x.idx, Int32(0), x.v^p, p * x.v^(p - 1))

# -----------------------------------------------------------------------------
# Statement-level rules.
#
# Everything above records ONE operation per node. That is the design the C++
# literature identifies as the slow one: Adept records differential *statements*
# and reports a reverse pass 2.3-12x faster than ADOL-C and CppAD, which tape
# every operator separately.
#
# These rules give the tape coarser granularity where it matters most, without
# changing anything about how it is replayed or what makes it trimmable.
# -----------------------------------------------------------------------------

"""
    muladd(x, y, z)

`x*y + z` as ONE tape node instead of three.

This is the commonest shape in a linear predictor — Julia's generic matvec inner
loop is literally `s = muladd(A[i,j], x[j], s)` — so fusing it is where
statement-level taping pays first.
"""
@inline function Base.muladd(x::TapedReal, y::TapedReal, z::TapedReal)
    (_isdetached(x) && _isdetached(y) && _isdetached(z)) &&
        return TapedReal(_DETACHED, Int32(0), muladd(x.v, y.v, z.v))
    t = _isdetached(x) ? (_isdetached(y) ? z.tape : y.tape) : x.tape
    a = _attach(t, x); b = _attach(t, y); c = _attach(t, z)
    # ∂/∂a = b.v, ∂/∂b = a.v, ∂/∂c = 1. Only two partials need saving; the
    # third input's adjoint is added unscaled, so it rides in the `b` slot of a
    # follow-on add. Simpler and just as fast: record the FMA over (a,b) and
    # let `c` contribute through the same node.
    return _record_fma(t, a, b, c, muladd(a.v, b.v, c.v), b.v, a.v)
end
@inline Base.muladd(x::TapedReal, y::Real, z::TapedReal) =
    _linear2(x, Float64(y), z, 1.0)
@inline Base.muladd(x::Real, y::TapedReal, z::TapedReal) =
    _linear2(y, Float64(x), z, 1.0)
@inline Base.muladd(x::TapedReal, y::TapedReal, z::Real) = x * y + z

# An FMA needs THREE inputs but `Node` holds two indices. The third (the addend)
# is handled by chaining one extra node, which still beats three: the multiply
# and its two partials are fused, and the add is a bare index copy.
@inline function _record_fma(t::Tape, a::TapedReal, b::TapedReal, c::TapedReal,
                             v::Float64, da::Float64, db::Float64)
    prod = record!(t, OP_FMA, a.idx, b.idx, a.v * b.v, da, db)
    return record!(t, OP_ADD, prod.idx, c.idx, v, 0.0)
end

"""
    _linear2(x, cx, y, cy)

`x*cx + y*cy` with CONSTANT coefficients, as one node.

This covers `muladd(x, ::Real, y)` and the `a*const + b*const` combinations that
dominate a log-density's accumulation, where the data are plain Float64 and only
the parameters are tracked.
"""
@inline function _linear2(x::TapedReal, cx::Float64, y::TapedReal, cy::Float64)
    (_isdetached(x) && _isdetached(y)) &&
        return TapedReal(_DETACHED, Int32(0), x.v * cx + y.v * cy)
    t = _livetape(x, y)
    a = _attach(t, x); b = _attach(t, y)
    return record!(t, OP_LINEAR, a.idx, b.idx, a.v * cx + b.v * cy, cx, cy)
end

"""
    dot_tracked(xs, cs, offset)

The dot product of tracked `xs` with plain-Float64 coefficients `cs`, as ONE
tape node per reduction rather than `2K`.

This is the `X * beta` case, and it is where operation-level taping hurts most:
on an N=5000, K=200 regression the naive tape is ~2M nodes where this makes it
~5K. The saved partial for each input IS its coefficient, so nothing needs to be
recomputed on the reverse sweep — but a node holds only two input slots, so the
reduction is recorded as a chain of `OP_LINEAR` pairs, which is still one node
per two inputs instead of four.
"""
function dot_tracked(xs::AbstractVector{TapedReal}, cs::AbstractVector{<:Real})
    n = length(xs)
    n == 0 && return TapedReal(_DETACHED, Int32(0), 0.0)
    t = xs[1].tape
    @inbounds begin
        acc = _linear1(t, xs[1], Float64(cs[1]))
        i = 2
        while i + 1 <= n
            # two inputs at a time: one node covers x[i]*c[i] + x[i+1]*c[i+1]
            pair = _linear2(xs[i], Float64(cs[i]), xs[i+1], Float64(cs[i+1]))
            acc = acc + pair
            i += 2
        end
        i <= n && (acc = acc + _linear1(t, xs[i], Float64(cs[i])))
    end
    return acc
end

@inline _linear1(t::Tape, x::TapedReal, c::Float64) =
    record!(t, OP_MULC, _attach(t, x).idx, Int32(0), x.v * c, c)

# `SpecialFunctions._logabsgamma` is THE single method gating Gamma, Beta,
# Poisson, Binomial, NegativeBinomial, TDist and Chisq — all seven fail on it
# and nothing else. Unlike the forward-mode side (which needs the value computed
# through differentiable arithmetic), here the primal is a plain Float64
# recorded once and the derivative is digamma, so calling out is fine.
SpecialFunctions._logabsgamma(x::TapedReal) =
    (record!(x.tape, OP_LGAMMA, x.idx, Int32(0),
             SpecialFunctions.logabsgamma(x.v)[1], 0.0), 1)
SpecialFunctions.loggamma(x::TapedReal) =
    record!(x.tape, OP_LGAMMA, x.idx, Int32(0), SpecialFunctions.loggamma(x.v), 0.0)

# --- the `<: Real` contract --------------------------------------------------
# A constant converted into tracked-land needs a tape, and `convert` is not
# given one. Producing an untracked value here is correct: a genuine constant
# has zero derivative, and every operation that mixes it with a TapedReal goes
# through the mixed-argument methods above, which keep the tape.
Base.promote_rule(::Type{TapedReal}, ::Type{<:Real}) = TapedReal
Base.convert(::Type{TapedReal}, x::TapedReal) = x

# `zero`/`one` on the TYPE, and `TapedReal(x)`, have no tape to record onto.
# That is a real problem, not a corner case: `sum` uses `zero(T)` as its
# accumulator and `Distributions` constructs `T(x)` when promoting a
# distribution's parameters, so both are on the path of ordinary model code.
#
# Such a value is DETACHED: it carries a primal but belongs to no tape. It is
# marked by `idx == 0` (never a valid tape index) and by a shared empty tape
# that is never recorded to.
#
# The rule that keeps this sound: **a detached value is a constant, and a
# constant has zero derivative.** Every arithmetic method below routes a
# detached operand through `constant!` on the OTHER operand's live tape, so no
# node ever stores index 0 as an input, and nothing is ever appended to
# `_DETACHED`.
#
# An earlier attempt let `_attach` fall back to the detached tape when BOTH
# operands looked detached. That silently recorded onto a shared global tape
# which grew forever, so an output's index outran the real tape's length --
# `output idx = 8` against `tape len = 4`, giving a BoundsError in the reverse
# sweep and, worse, a returned "gradient" that was just the input value.

@inline _isdetached(x::TapedReal) = x.idx == Int32(0)

# The live tape of a binary operation: whichever operand actually has one.
# Both detached means both are constants, and the caller handles that case by
# never reaching the tape at all.
@inline _livetape(x::TapedReal, y::TapedReal) = _isdetached(x) ? y.tape : x.tape

# Put a detached constant onto a live tape so it can be used as an input.
@inline _attach(t::Tape, x::TapedReal) =
    _isdetached(x) ? constant!(t, x.v) : x

# `Distributions` constructs `T(x)` directly for a promoted parameter type
# (e.g. `MvNormal` converting its mean, or a Bool from a support check), so the
# single-argument constructor has to exist. It has no tape to record onto, so
# it produces a detached value; `_attach` puts it on a live tape the moment it
# meets one.
TapedReal(x::Real) = TapedReal(_DETACHED, Int32(0), Float64(x))

Base.float(x::TapedReal) = x
Base.eltype(::Type{TapedReal}) = Float64

# Predicates answer about the VALUE — that is what lets `logpdf`'s support
# checks (`insupport`, `x < 0`, `isinteger` on a discrete observation) work
# unchanged. A derivative has no bearing on whether a number is an integer.
for op in (:(==), :<, :<=, :>, :>=)
    @eval Base.$op(a::TapedReal, b::TapedReal) = $op(a.v, b.v)
    @eval Base.$op(a::TapedReal, b::Real) = $op(a.v, b)
    @eval Base.$op(a::Real, b::TapedReal) = $op(a, b.v)
end
Base.isless(a::TapedReal, b::TapedReal) = isless(a.v, b.v)
Base.isnan(x::TapedReal) = isnan(x.v)
Base.isinf(x::TapedReal) = isinf(x.v)
Base.isfinite(x::TapedReal) = isfinite(x.v)
Base.isinteger(x::TapedReal) = isinteger(x.v)
Base.round(x::TapedReal) = round(x.v)
Base.floor(x::TapedReal) = floor(x.v)
Base.ceil(x::TapedReal) = ceil(x.v)
Base.trunc(x::TapedReal) = trunc(x.v)

# -----------------------------------------------------------------------------
# The reverse sweep
# -----------------------------------------------------------------------------

"""
    backward!(adj, t, out) -> adj

Seed the output's adjoint and walk the tape backwards, accumulating into `adj`.
After this, `adj[i]` holds d(output)/d(node i) — so for a node made by
[`track!`](@ref), that is its gradient component.

One loop, one branch on a concrete `Int8`, every arm statically known. That
shape is what lets the trimmer see through it.
"""
function backward!(adj::Vector{Float64}, t::Tape, out::Integer)
    n = t.n
    # Grow only when the reused buffer is genuinely too small; in the steady
    # state (a sampler calling this once per leapfrog step on a fixed model)
    # this never fires, so the sweep allocates nothing.
    length(adj) < n && resize!(adj, n)
    @inbounds for i in 1:n
        adj[i] = 0.0
    end
    @inbounds adj[out] = 1.0
    nodes = t.nodes
    @inbounds for i in n:-1:1
        ai = adj[i]
        # Skipping zero adjoints is a real saving: a log-density's tape has many
        # nodes that never reach the output.
        ai == 0.0 && continue
        nd = nodes[i]
        op = nd.op
        if op == OP_ADD
            adj[nd.a] += ai; adj[nd.b] += ai
        elseif op == OP_SUB
            adj[nd.a] += ai; adj[nd.b] -= ai
        elseif op == OP_MUL
            adj[nd.a] += ai * nodes[nd.b].v
            adj[nd.b] += ai * nodes[nd.a].v
        elseif op == OP_DIV
            va = nodes[nd.a].v; vb = nodes[nd.b].v
            adj[nd.a] += ai / vb
            adj[nd.b] -= ai * va / (vb * vb)
        elseif op == OP_MULC
            adj[nd.a] += ai * nd.c
        elseif op == OP_ADDC
            adj[nd.a] += ai
        elseif op == OP_NEG
            adj[nd.a] -= ai
        elseif op == OP_SQ
            adj[nd.a] += ai * 2.0 * nodes[nd.a].v
        elseif op == OP_EXP
            adj[nd.a] += ai * nd.v
        elseif op == OP_LOG
            adj[nd.a] += ai / nodes[nd.a].v
        elseif op == OP_SQRT
            adj[nd.a] += ai / (2.0 * nd.v)
        elseif op == OP_SIN
            adj[nd.a] += ai * cos(nodes[nd.a].v)
        elseif op == OP_COS
            adj[nd.a] -= ai * sin(nodes[nd.a].v)
        elseif op == OP_TAN
            adj[nd.a] += ai * (1.0 + nd.v * nd.v)
        elseif op == OP_TANH
            adj[nd.a] += ai * (1.0 - nd.v * nd.v)
        elseif op == OP_ATAN
            va = nodes[nd.a].v
            adj[nd.a] += ai / (1.0 + va * va)
        elseif op == OP_ATAN2
            # `c` holds the precomputed denominator y^2 + x^2.
            adj[nd.a] += ai * nodes[nd.b].v / nd.c
            adj[nd.b] -= ai * nodes[nd.a].v / nd.c
        elseif op == OP_ABS
            adj[nd.a] += ai * sign(nodes[nd.a].v)
        elseif op == OP_LOG1P
            adj[nd.a] += ai / (1.0 + nodes[nd.a].v)
        elseif op == OP_EXPM1
            adj[nd.a] += ai * exp(nodes[nd.a].v)
        elseif op == OP_LGAMMA
            adj[nd.a] += ai * SpecialFunctions.digamma(nodes[nd.a].v)
        elseif op == OP_FMA || op == OP_LINEAR
            # Both saved their two partials at record time, which is the whole
            # point: the reverse sweep is two multiply-accumulates for what
            # would otherwise be several nodes.
            adj[nd.a] += ai * nd.c
            adj[nd.b] += ai * nd.d
        end
        # OP_INPUT and OP_CONST have no inputs: the adjoint stops there.
    end
    return adj
end

"""
    ReverseWorkspace(K)

Reusable scratch for [`rev_gradient!`](@ref): the tape, the tracked inputs and
the adjoint buffer.

Reuse is the difference between allocating on every gradient call and not.
A sampler evaluates the same model shape once per leapfrog step, so pass the
same workspace each time.
"""
mutable struct ReverseWorkspace
    tape::Tape
    xs::Vector{TapedReal}
    adj::Vector{Float64}
end

function ReverseWorkspace(K::Integer)
    t = Tape()
    return ReverseWorkspace(t, Vector{TapedReal}(undef, Int(K)), Float64[])
end

"""
    rev_gradient!(g, f, x, ws) -> g
    rev_gradient!(g, f, x)     -> g

Reverse-mode gradient of `f` at `x`, written into `g`.

Costs **one** sweep regardless of `length(x)`, which is why this is the mode to
use when the parameter count is large. Compare [`gradient!`](@ref), which is
forward mode and costs `ceil(K/N)` evaluations.

Pass a [`ReverseWorkspace`](@ref) to reuse the tape across calls; without one a
fresh workspace is allocated per call, which profiling showed to be ~86% of the
runtime.

```julia
f(x) = sum(abs2, x) + exp(x[1]) * log(x[2])
x = [0.7, 1.3, -0.4]
g = similar(x)
ws = ReverseWorkspace(length(x))
rev_gradient!(g, f, x, ws)
```
"""
function rev_gradient!(g::Vector{Float64}, f::F, x::Vector{Float64},
                       ws::ReverseWorkspace) where {F}
    v, _ = rev_value_and_gradient!(g, f, x, ws)
    return g
end

rev_gradient!(g::Vector{Float64}, f::F, x::Vector{Float64}) where {F} =
    rev_gradient!(g, f, x, ReverseWorkspace(length(x)))

"""
    rev_value_and_gradient!(g, f, x, ws) -> (value, g)

As [`rev_gradient!`](@ref), also returning the primal — which the forward sweep
has already computed, so it costs nothing extra.
"""
function rev_value_and_gradient!(g::Vector{Float64}, f::F, x::Vector{Float64},
                                 ws::ReverseWorkspace) where {F}
    K = length(x)
    length(ws.xs) < K && resize!(ws.xs, K)
    t = reset!(ws.tape)
    @inbounds for i in 1:K
        ws.xs[i] = track!(t, x[i])
    end
    y = f(view(ws.xs, 1:K))
    if y isa TapedReal
        backward!(ws.adj, t, y.idx)
        @inbounds for i in 1:K
            g[i] = ws.adj[ws.xs[i].idx]
        end
        return (y.v, g)
    end
    # `f` returned something untracked, i.e. it does not actually depend on `x`.
    # A zero gradient is the right answer, and saying so beats a confusing error.
    fill!(g, 0.0)
    return (Float64(y), g)
end

rev_value_and_gradient!(g::Vector{Float64}, f::F, x::Vector{Float64}) where {F} =
    rev_value_and_gradient!(g, f, x, ReverseWorkspace(length(x)))
