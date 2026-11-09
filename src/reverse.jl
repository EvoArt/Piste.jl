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
const OP_MATVEC = Int8(26)  # one row of X*beta; the matrix lives in the side table
# `1/(1+exp(-x))` as ONE node. Not a convenience: it is the commonest link
# function in applied work, and spelled out it taped as NEG -> EXP -> ADDC ->
# MULC, four nodes whose partials all depend on the same `exp(-x)`. On an HMM
# log-likelihood that chain was ~57% of the whole tape. The derivative
# `s * (1 - s)` is a function of the OUTPUT, so nothing extra need be saved.
const OP_LOGISTIC = Int8(27)
# A whole-array reduction as ONE node: `sum(x)` and `sum(abs2, x)` over a vector
# of tracked values. `a` is the start of the operands' indices in `Tape.ilist`,
# `b` their count. Recorded elementwise, a sum of N terms is N-1 OP_ADD nodes
# (and N OP_SQ before them for `abs2`) -- on a normal regression that was half
# of the tape. See `_record_sum`.
const OP_SUM   = Int8(28)
const OP_SUMSQ = Int8(29)

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
    # Side table for ARRAY-level operations. A `Node` is deliberately isbits and
    # fixed-size, so it cannot hold a matrix; instead an array-level node stores
    # an index into here, and the reverse sweep looks the operand up.
    #
    # This is what lets `X * beta` be ~N nodes instead of ~2NK: the adjoint of a
    # matrix-vector product is another matrix-vector product (X' * adj), one
    # BLAS call, rather than N*K scalar accumulations. Stan does the same thing
    # with its matrix `vari` types, and it is what keeps it competitive.
    #
    # It holds only plain data (the DATA matrix and the tracked inputs' tape
    # indices), never a closure, so nothing here costs trimmability.
    # Concrete element types on purpose. A `Vector{AbstractMatrix}` boxes and
    # dispatches on every access in the reverse sweep -- measured, it made the
    # sweep slower and allocations higher than storing a copy did.
    #
    # Hence two tables: `Matrix{Float64}` and `Matrix{Float32}` data are both
    # stored by REFERENCE, each in its own concretely typed vector. Float32
    # used to go through `convert(Matrix{Float64}, X)`, which copied the whole
    # data matrix on every gradient call: a Float32 model ran 1.3-2.5x slower
    # than the same model in Float64, the gap growing with N*K. Any other
    # element type still converts into `mats`.
    mats::Vector{Matrix{Float64}}
    mats32::Vector{Matrix{Float32}}
    # Where matrix `m` lives: `mref[m] > 0` is an index into `mats`, `< 0`
    # minus an index into `mats32`. `idxs` and `accs` stay indexed by `m`.
    mref::Vector{Int32}
    idxs::Vector{UnitRange{Int32}}
    # Per-matrix DEFERRED adjoint accumulator, one entry per `mats` entry.
    #
    # An OP_MATVEC node's adjoint contribution is `ai * X[row, :]`. Applying that
    # as it comes -- a strided walk across one row of X, once per node -- is a
    # scalar decomposition of what is mathematically a single `X' * acc`. It was
    # measured at 925 us against 304 us for the equivalent gemv (3.0x) on a
    # 5000x200 problem, because a row walk touches the whole 8 MB matrix with no
    # blocking while BLAS runs a cache-tiled kernel.
    #
    # So the sweep only ACCUMULATES `acc[row] = ai` here, and `backward!` applies
    # it in one column-major pass per matrix when the sweep reaches the
    # matvec's operands (see `backward!` for why then and not at the end).
    accs::Vector{Vector{Float64}}
    # Operand indices of the array-level reductions (OP_SUM, OP_SUMSQ), packed
    # end to end; each such node records where its run starts and how long it
    # is. `ni` is the used length -- like `n` for `nodes`, the storage itself is
    # kept across `reset!`, so the steady state allocates nothing.
    ilist::Vector{Int32}
    ni::Int
    # This tape's entry in `_SLOTS`, or 0 while it has none. A TapedReal names
    # its tape by this number rather than by reference; see `TapedReal`.
    slot::Int32
end

Tape() = Tape(Node[], 0, Matrix{Float64}[], Matrix{Float32}[], Int32[],
              UnitRange{Int32}[], Vector{Float64}[], Int32[], 0, Int32(0))

"""
    Tape(capacity::Int)

Preallocate for a known node count, so even the first sweep does not grow.
"""
Tape(capacity::Integer) = Tape(Vector{Node}(undef, Int(capacity)), 0,
                              Matrix{Float64}[], Matrix{Float32}[], Int32[],
                              UnitRange{Int32}[], Vector{Float64}[], Int32[], 0, Int32(0))

"""
    reset!(t) -> t

Rewind to empty WITHOUT releasing storage, so the next sweep refills the same
memory. This is what makes repeated gradient calls cheap.
"""
@inline function reset!(t::Tape)
    t.n = 0
    t.ni = 0
    empty!(t.mats)
    empty!(t.mats32)
    empty!(t.mref)
    empty!(t.idxs)
    # `accs` keeps its buffers: they are pooled by position and re-zeroed when
    # claimed in `_claim_acc!`, so repeated sweeps of the same model reuse the
    # same storage rather than allocating a fresh accumulator each time.
    return t
end

# Hand out the accumulator for side-table slot `m`, sized `n` and zeroed.
# Buffers persist across `reset!` so the steady state allocates nothing.
@inline function _claim_acc!(t::Tape, m::Int, n::Int)
    if length(t.accs) < m
        push!(t.accs, zeros(Float64, n))
    else
        a = t.accs[m]
        length(a) < n ? resize!(a, n) : nothing
        @inbounds for i in 1:n
            a[i] = 0.0
        end
    end
    return nothing
end

@inline Base.length(t::Tape) = t.n
@inline Base.isempty(t::Tape) = t.n == 0

"""
    TapedReal <: Real

A value being recorded: which tape, its index on it, and its primal.

`<: Real` is load-bearing exactly as it is for [`Dual`](@ref) — it is what lets
`Distributions.logpdf` and other generic numeric code accept these unmodified.

**isbits, 16 bytes.** The tape is named by a slot number into the const table
`_SLOTS`, not held by reference. Holding a `Tape` reference (the original
design) made every `Vector{TapedReal}` -- each matvec output, each broadcast
such as `y .- mu` -- an array of pointers, which Julia must zero on
allocation, write-barrier on every store, and scan on every collection.
Measured on an N=20000 normal regression, that was a third of gradient time in
allocation, a sixth in stores and a fifth in GC. Named by number, those arrays
are plain bits.

The table is concretely typed and const, so `tape(x)` infers and trims like a
field read did. A gradient call borrows a slot for its tape and hands it back
on exit (see [`rev_value_and_gradient!`](@ref)), so the table never grows past
the number of gradients in flight at once.
"""
struct TapedReal <: Real
    slot::Int32
    idx::Int32
    v::Float64
end

# Record sites still name the tape itself; this stores its slot.
@inline TapedReal(t::Tape, idx::Int32, v::Float64) = TapedReal(t.slot, idx, v)

@inline value(x::TapedReal) = x.v

# --- the slot table ------------------------------------------------------------
# Fixed capacity and never resized: a gradient on one thread reads its own slot
# while another thread may be claiming a different one, and a resize would move
# the storage out from under the reader. Each slot is written only by the task
# that holds it, before that task records anything.
const _MAX_SLOTS = 4096
const _SLOTS = Vector{Tape}(undef, _MAX_SLOTS)
const _FREE_SLOTS = Int32[]
const _NEXT_SLOT = Ref{Int32}(Int32(1))   # slot 1 is `_DETACHED`'s
const _SLOT_LOCK = Threads.SpinLock()

@inline tape(x::TapedReal) = @inbounds _SLOTS[x.slot]

# A shared, never-swept tape standing for "this value belongs to no recording".
# Defined here because `record!` guards against it. It owns slot 1 for good.
const _DETACHED = Tape()
_DETACHED.slot = Int32(1)
_SLOTS[1] = _DETACHED

# Give `t` a slot. Returns true if it had none, i.e. the caller now owns one and
# should `_release_slot!` it when done.
function _claim_slot!(t::Tape)
    t.slot != Int32(0) && return false
    lock(_SLOT_LOCK)
    s = Int32(0)
    try
        if isempty(_FREE_SLOTS)
            if _NEXT_SLOT[] < Int32(_MAX_SLOTS)
                s = _NEXT_SLOT[] + Int32(1)
                _NEXT_SLOT[] = s
            end
        else
            s = pop!(_FREE_SLOTS)
        end
        s != Int32(0) && (@inbounds _SLOTS[s] = t)
    finally
        unlock(_SLOT_LOCK)
    end
    s == Int32(0) && error("Piste: more than $(_MAX_SLOTS - 1) reverse-mode tapes in use at once")
    t.slot = s
    return true
end

function _release_slot!(t::Tape)
    s = t.slot
    s <= Int32(1) && return nothing
    t.slot = Int32(0)
    lock(_SLOT_LOCK)
    try
        @inbounds _SLOTS[s] = _DETACHED    # drop the reference so the tape can be freed
        push!(_FREE_SLOTS, s)
    finally
        unlock(_SLOT_LOCK)
    end
    return nothing
end

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
        # A tape index is an Int32, so wrapping past 2^31 would silently corrupt
        # every later node's operands. Only reachable on a ~80 GB tape, but a
        # wrong gradient is exactly the failure this package exists to avoid, so
        # it is checked once per growth rather than once per node.
        i > 2_147_483_647 && throw(OverflowError("tape exceeds 2^31 nodes"))
        resize!(t.nodes, max(16, 2 * length(t.nodes)))
    end
    @inbounds t.nodes[i] = Node(op, a, b, v, c, d)
    # `Int32(i)` would emit a CHECKED narrowing (`checked_trunc_sint`, 4.5% of
    # self time on an HMM tape) on every single node. `i` is a tape position
    # that has just been bounds-grown, so it is positive and far below 2^31 for
    # any tape that fits in memory -- 2^31 nodes is 80 GB. The unchecked
    # conversion is the same instruction without the trap.
    return TapedReal(t, i % Int32, v)
end

"""
    track!(t, x) -> TapedReal

Put an input onto the tape. Its adjoint after the reverse sweep is that input's
gradient component.
"""
@inline function track!(t::Tape, x::Real)
    # A tape driven by hand (rather than through `rev_gradient!`) gets a slot
    # here on first use and keeps it; `rev_gradient!` borrows one per call.
    t.slot == Int32(0) && _claim_slot!(t)
    return record!(t, OP_INPUT, Int32(0), Int32(0), Float64(x), 0.0)
end

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
    # One detached operand is a shift by a constant -- one OP_ADDC, not
    # CONST+ADD. `zero(T)` from a `sum` accumulator takes this path on the first
    # term of every reduction.
    _isdetached(y) && return x + y.v
    _isdetached(x) && return y + x.v
    t = tape(x)
    a = x; b = y
    # PEEPHOLE FUSION. `s + x*c` is recorded as one OP_LINEAR reading `x`
    # directly, so on the sweep the `x*c` node gets a zero adjoint and is skipped.
    #
    # The OP_MULC is deliberately KEPT, not dropped. Dropping it (rewinding
    # `t.n` and reusing its slot) was the original version and it was WRONG:
    # the caller can still hold the `x*c` value, and its index then pointed at
    # the OP_LINEAR. `p = v[1]*2; s = v[2] + p; s + 3p` returned
    # d/dv[2] = 4 instead of 1, with no error. A tape slot that a live
    # TapedReal refers to must never be rewritten.
    if _is_last_mulc(t, b)
        nd = @inbounds t.nodes[b.idx]
        return record!(t, OP_LINEAR, a.idx, nd.a, a.v + b.v, 1.0, nd.c)
    elseif _is_last_mulc(t, a)
        nd = @inbounds t.nodes[a.idx]
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
    # ONE detached operand is a scalar, not a tape node. Attaching it recorded an
    # OP_CONST *and* an OP_SUB -- two nodes plus a primal lookup on the sweep --
    # where the mixed-argument rules already do it in one.
    #
    # `one(T) - p` is the shape this is for: `one(T)` has no tape, so every
    # complement in a likelihood (`1 - Se`, `1 - Sp`, `1 - lambda`) was paying
    # double. On an HMM tape CONST+SUB was 31% of all nodes.
    _isdetached(y) && return x + (-y.v)
    _isdetached(x) && return x.v - y
    t = tape(x)
    return record!(t, OP_SUB, x.idx, y.idx, x.v - y.v, 0.0)
end
@inline function Base.:*(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v * y.v)
    # One detached operand is a scalar multiply -- one OP_MULC, not CONST+MUL.
    _isdetached(y) && return x * y.v
    _isdetached(x) && return y * x.v
    t = tape(x)
    # `x * x` is a square. As an OP_MUL it is correct but pays two adjoint
    # updates to the SAME slot (`adj[a] += ai*v; adj[a] += ai*v`), a
    # read-modify-write dependency the CPU cannot overlap. OP_SQ does it in one.
    # `r * r` on a squared residual is the commonest shape in a log-density.
    x.idx === y.idx && return record!(t, OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0)
    return record!(t, OP_MUL, x.idx, y.idx, x.v * y.v, 0.0)
end
@inline function Base.:/(x::TapedReal, y::TapedReal)
    # Two constants stay a constant: nothing to record, nothing to differentiate.
    (_isdetached(x) && _isdetached(y)) && return TapedReal(_DETACHED, Int32(0), x.v / y.v)
    # One detached operand: scale by a constant, or the reciprocal rule. Both
    # are single nodes via the mixed-argument methods.
    _isdetached(y) && return x * (1.0 / y.v)
    _isdetached(x) && return x.v / y
    t = tape(x)
    return record!(t, OP_DIV, x.idx, y.idx, x.v / y.v, 0.0)
end
@inline Base.:-(x::TapedReal) =
    record!(tape(x), OP_NEG, x.idx, Int32(0), -x.v, 0.0)

# Mixed Real/TapedReal fast paths: data is plain Float64 and only parameters are
# tracked, so promoting a scalar to a full tape node on every operation would be
# pure waste on the hottest loop in a regression model.
@inline Base.:*(x::TapedReal, c::Real) =
    record!(tape(x), OP_MULC, x.idx, Int32(0), x.v * Float64(c), Float64(c))
@inline Base.:*(c::Real, x::TapedReal) = x * c
@inline Base.:+(x::TapedReal, c::Real) =
    record!(tape(x), OP_ADDC, x.idx, Int32(0), x.v + Float64(c), Float64(c))
@inline Base.:+(c::Real, x::TapedReal) = x + c
@inline Base.:-(x::TapedReal, c::Real) = x + (-Float64(c))
# `c - x` is ONE affine node, not a negate followed by an add.
#
# Written as `(-x) + c` this recorded an OP_NEG and an OP_ADDC for every term.
# On a normal-regression log-density the residual `y[i] - mu[i]` is the
# innermost expression, so those two nodes were 39% of the entire tape -- more
# than the matvec they were wrapped around. OP_MULC already carries a
# coefficient and its adjoint is `ai * c`, which is exactly what is needed with
# c = -1; the constant addend shifts the primal only and has no adjoint.
@inline Base.:-(c::Real, x::TapedReal) =
    record!(tape(x), OP_MULC, x.idx, Int32(0), Float64(c) - x.v, -1.0)
@inline Base.:/(x::TapedReal, c::Real) = x * (1.0 / Float64(c))
# `c / x` is ONE node, not a materialised constant plus a full division.
#
# Written as `constant!(t, c) / x` it recorded an OP_CONST *and* an OP_DIV, and
# the OP_DIV then had to look up both operands' primals on the reverse sweep.
# d(c/x)/dx = -c/x^2 is known at record time, so OP_MULC carries it directly.
#
# This is on the hot path of every logistic: `1/(1 + exp(-x))` is the commonest
# link function there is, and on an HMM log-likelihood the CONST+DIV pair alone
# was ~36% of the tape.
@inline function Base.:/(c::Real, x::TapedReal)
    cf = Float64(c)
    t = tape(x)
    # PEEPHOLE: `1 / (1 + exp(-u))` as an OP_LOGISTIC reading `u` directly, so
    # the sweep does one multiply-accumulate and skips the chain below it,
    # whose adjoints stay zero.
    #
    # The NEG/EXP/ADDC chain is KEPT on the tape. Dropping it, as this
    # originally did, reused slots the caller could still hold: with
    # `e = exp(-u); s = 1/(1+e); s + e` the gradient came back as +0.244 instead
    # of -0.496, silently. A tape slot a live TapedReal refers to must never be
    # rewritten.
    if cf == 1.0
        u = _logistic_src(t, x)
        u != Int32(0) && return record!(t, OP_LOGISTIC, u, Int32(0), 1.0 / x.v, 0.0)
    end
    return record!(t, OP_MULC, x.idx, Int32(0), cf / x.v, -cf / (x.v * x.v))
end

# If `x` is the `1 + exp(-u)` at the very end of the tape, return u's index.
# Returns 0 when the pattern does not match, so the caller falls through.
#
# The three nodes must be the last three recorded AND consecutive: node n is
# `ADDC(1.0)` over node n-1, which is `EXP` over node n-2, which is `MULC(-1)`
# -- the shape `-u` takes since `c - x` became a single scaled node.
@inline function _logistic_src(t::Tape, x::TapedReal)
    n = t.n
    (x.idx == Int32(n) && n >= 3) || return Int32(0)
    @inbounds begin
        addc = t.nodes[n]
        (addc.op === OP_ADDC && addc.c == 1.0 && addc.a == Int32(n - 1)) || return Int32(0)
        ex = t.nodes[n - 1]
        (ex.op === OP_EXP && ex.a == Int32(n - 2)) || return Int32(0)
        ng = t.nodes[n - 2]
        # `-u` records as OP_NEG; `0 - u` and `c - u` record as OP_MULC(-1).
        if ng.op === OP_NEG || (ng.op === OP_MULC && ng.c == -1.0)
            return ng.a
        end
    end
    return Int32(0)
end

@inline Base.exp(x::TapedReal)   = record!(tape(x), OP_EXP,   x.idx, Int32(0), exp(x.v), 0.0)
@inline Base.log(x::TapedReal)   = record!(tape(x), OP_LOG,   x.idx, Int32(0), log(x.v), 0.0)
@inline Base.sqrt(x::TapedReal)  = record!(tape(x), OP_SQRT,  x.idx, Int32(0), sqrt(x.v), 0.0)
@inline Base.sin(x::TapedReal)   = record!(tape(x), OP_SIN,   x.idx, Int32(0), sin(x.v), 0.0)
@inline Base.cos(x::TapedReal)   = record!(tape(x), OP_COS,   x.idx, Int32(0), cos(x.v), 0.0)
@inline Base.tan(x::TapedReal)   = record!(tape(x), OP_TAN,   x.idx, Int32(0), tan(x.v), 0.0)
@inline Base.tanh(x::TapedReal)  = record!(tape(x), OP_TANH,  x.idx, Int32(0), tanh(x.v), 0.0)
@inline Base.atan(x::TapedReal)  = record!(tape(x), OP_ATAN,  x.idx, Int32(0), atan(x.v), 0.0)
@inline Base.abs(x::TapedReal)   = record!(tape(x), OP_ABS,   x.idx, Int32(0), abs(x.v), 0.0)
@inline Base.log1p(x::TapedReal) = record!(tape(x), OP_LOG1P, x.idx, Int32(0), log1p(x.v), 0.0)
@inline Base.expm1(x::TapedReal) = record!(tape(x), OP_EXPM1, x.idx, Int32(0), expm1(x.v), 0.0)
@inline Base.abs2(x::TapedReal) =
    record!(tape(x), OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0)
@inline Base.inv(x::TapedReal) =
    record!(tape(x), OP_MULC, x.idx, Int32(0), inv(x.v), -inv(x.v * x.v))

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
    isfinite(den) || return constant!(tape(y), v)
    return record!(tape(y), OP_ATAN2, y.idx, x.idx, v, den)
end
@inline Base.atan(y::TapedReal, x::Real) = atan(y, constant!(tape(y), x))
@inline Base.atan(y::Real, x::TapedReal) = atan(constant!(tape(x), y), x)

@inline Base.:^(x::TapedReal, n::Integer) =
    n == 2 ? record!(tape(x), OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0) :
             record!(tape(x), OP_MULC, x.idx, Int32(0), x.v^n, n * x.v^(n - 1))
@inline Base.literal_pow(::typeof(^), x::TapedReal, ::Val{2}) =
    record!(tape(x), OP_SQ, x.idx, Int32(0), x.v * x.v, 0.0)
@inline Base.literal_pow(::typeof(^), x::TapedReal, ::Val{p}) where {p} =
    record!(tape(x), OP_MULC, x.idx, Int32(0), x.v^p, p * x.v^(p - 1))

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
    t = _isdetached(x) ? (_isdetached(y) ? tape(z) : tape(y)) : tape(x)
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
    t = tape(xs[1])
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

"""
    *(X::AbstractMatrix{<:Real}, b::AbstractVector{TapedReal})

Matrix-vector product recorded at ARRAY level: one node per output element,
carrying an index to the matrix, instead of ~2K scalar nodes each.

This is the single biggest saving available on a regression log-density.
Elementwise, `X * beta` with N=1000, K=50 taped ~58 000 nodes and profiling put
**80% of gradient time in building the tape**, `_setindex!` alone at 29.9%. The
adjoint of a matvec is just another matvec — `X' * adj` — so there is no reason
to record the scalar decomposition at all.

The saving is asymptotic, not constant-factor: N*K nodes become N.
"""
# NOTE ON DISPATCH. `AbstractMatrix * AbstractVector{TapedReal}` is NOT specific
# enough: LinearAlgebra defines
#     *(A::StridedMatrix{T}, x::StridedVector{S}) where {T<:Float64, S<:Real}
# and since `TapedReal <: Real` that method is MORE specific for the common case
# of a plain `Matrix{Float64}` times a `Vector{TapedReal}`. It wins, and the
# product silently falls back to the elementwise path — measured, the node count
# went back to scaling with N*K instead of N, with no error to notice.
#
# Matching `StridedVecOrMat` on the left restores precedence for exactly the
# shapes that method claims, and the AbstractMatrix method below still catches
# everything else.
# LinearAlgebra's method is
#     *(A::StridedMatrix{T}, x::StridedVector{S}) where {T<:Union{Float32,Float64,...}, S<:Real}
# which pins the ELEMENT type of A. To take precedence for a tracked right-hand
# side, this has to be at least as specific on the left and strictly more
# specific on the right (`TapedReal` rather than `S<:Real`).
function Base.:*(X::StridedMatrix{Float64}, b::StridedVector{TapedReal})
    return _matvec_taped(X, b)
end
function Base.:*(X::StridedMatrix{Float32}, b::StridedVector{TapedReal})
    return _matvec_taped(X, b)
end
function Base.:*(X::StridedMatrix{<:Real}, b::StridedVector{TapedReal})
    return _matvec_taped(X, b)
end
function Base.:*(X::AbstractMatrix{<:Real}, b::AbstractVector{TapedReal})
    return _matvec_taped(X, b)
end

function _matvec_taped(X::AbstractMatrix{<:Real}, b::AbstractVector{TapedReal})
    n, k = size(X)
    k == length(b) || throw(DimensionMismatch("matrix has $k columns, vector has $(length(b))"))
    n == 0 && return TapedReal[]
    t = _livetape_vec(b)
    t === _DETACHED && return [TapedReal(_DETACHED, Int32(0), _dotv(X, b, i)) for i in 1:n]

    # The tracked inputs must occupy a CONTIGUOUS index range for the reverse
    # sweep to find them, so attach any detached entries and re-record if they
    # are not already contiguous. In the normal case (a parameter vector fresh
    # from `track!`) they are, and this is a no-op scan.
    lo, hi, contiguous = _index_span(b)
    if !contiguous
        bb = similar(b)
        @inbounds for j in 1:k
            bb[j] = record!(t, OP_MULC, _attach(t, b[j]).idx, Int32(0), b[j].v, 1.0)
        end
        lo, hi, _ = _index_span(bb)
    end

    # Store a REFERENCE, not a copy. `Matrix{Float64}(X)` here copied the whole
    # data matrix on every gradient call, which profiling showed to be 62% of
    # runtime (memmove 42.2%, GenericMemory 19.6% and GC-flagged) -- far more
    # than the taping it was meant to replace. The matrix is only read during
    # the reverse sweep, so a reference is sufficient; the caller must not
    # mutate their data between the forward and reverse passes, which is the
    # same contract every AD package assumes.
    #
    # Float64 and Float32 matrices are both kept as they are (see `Tape.mats32`);
    # only other element types are converted, and so copied.
    push!(t.idxs, lo:hi)
    m = length(t.idxs)
    _claim_acc!(t, m, n)

    # The PRIMAL is computed COLUMN-MAJOR, not as `n` row dot products.
    #
    # `_dotv` walked one row of X per output, i.e. across the stride of a
    # column-major array, n times. Accumulating down each column instead is unit
    # stride and vectorises: 1345 us -> 337 us on a 5000x200 problem. BLAS
    # `mul!` scored 263 us, so the hand kernel gives ~93% of the win WITHOUT a
    # LinearAlgebra dependency -- which matters here, because an external BLAS
    # call is exactly the kind of thing `--trim` is not guaranteed to survive.
    # The stride was the whole problem, not the absence of BLAS.
    #
    # The primal is written into this matrix's ADJOINT accumulator, which is
    # already sized `n` and pooled across sweeps, so this allocates nothing.
    # The buffer is not needed as an adjoint until `backward!` runs, and
    # `backward!` re-zeroes it first -- which it MUST, because the sweep skips
    # zero-adjoint nodes and would otherwise flush a leftover primal as a
    # gradient contribution.
    mu = t.accs[m]
    if X isa Matrix{Float32}
        push!(t.mats32, X)
        push!(t.mref, -Int32(length(t.mats32)))
        _matvec_primal!(mu, X, b)
    else
        Xm = convert(Matrix{Float64}, X)
        push!(t.mats, Xm)
        push!(t.mref, Int32(length(t.mats)))
        _matvec_primal!(mu, Xm, b)
    end
    out = Vector{TapedReal}(undef, n)
    @inbounds for i in 1:n
        # `c` carries the side-table slot, `d` the row -- both as Float64, which
        # is exact for any realistic N.
        out[i] = record!(t, OP_MATVEC, lo, hi, mu[i], Float64(m), Float64(i))
    end
    return out
end

# `X * b` over the primal values, accumulated down each column so every inner
# loop is unit stride. Plain Julia, so nothing here costs trimmability.
# In-place into a caller-owned buffer, so repeated sweeps allocate nothing.
function _matvec_primal!(mu::Vector{Float64}, Xm::Matrix{T},
                         b::AbstractVector{TapedReal}) where {T<:Union{Float64,Float32}}
    n, k = size(Xm)
    @inbounds for i in 1:n
        mu[i] = 0.0
    end
    @inbounds for j in 1:k
        bj = b[j].v
        bj == 0.0 && continue
        @simd for i in 1:n
            mu[i] = muladd(Float64(Xm[i, j]), bj, mu[i])
        end
    end
    return mu
end

@inline function _dotv(X, b, i)
    s = 0.0
    @inbounds for j in 1:length(b)
        s = muladd(Float64(X[i, j]), b[j].v, s)
    end
    return s
end

@inline function _livetape_vec(b::AbstractArray{TapedReal})
    @inbounds for x in b
        _isdetached(x) || return tape(x)
    end
    return _DETACHED
end

# Are the tracked entries a contiguous ascending run of tape indices?
#
# Written with an EARLY EXIT on purpose, not as a full scan with a flag. Since
# TapedReal became isbits, LLVM vectorises the flag-and-scan form with 512-bit
# (zmm) instructions, and on AVX-512 Xeons that drops the core's clock: on a
# Xeon Silver 4216 a fixed scalar loop ran 1.5-2x slower after each gradient,
# and the gradient itself went bimodal (14 us or 20 us at N=50, P=200 where it
# had been a steady 18). A loop with an early exit is not vectorised. Check
# `code_native` for zmm before changing this.
function _index_span(b::AbstractVector{TapedReal})
    n = length(b)
    @inbounds lo = b[1].idx
    # A detached first entry (index 0) is never a valid span start, even when
    # the next entry happens to be tape index 1: the flush would write adj[0].
    lo == Int32(0) && return lo, (@inbounds b[n].idx), false
    @inbounds for j in 2:n
        b[j].idx == lo + Int32(j - 1) || return lo, b[n].idx, false
    end
    return lo, lo + Int32(n - 1), true
end

# Flush one matrix's DEFERRED matvec adjoint: `adj[base:base+k-1] += X' * acc`.
#
# Called once per side-table matrix at the end of the sweep, replacing the
# per-row scatter that used to run inside the backward loop. The old form walked
# a row of X per node -- across the stride of a column-major array -- and was
# measured at 1258 us against 373 us for this column-major form on a 5000x200
# problem. Each column is one unit-stride dot product that vectorises.
function _flush_matvec!(adj::Vector{Float64}, Xm::Matrix{T},
                        acc::Vector{Float64}, base::Int, k::Int) where {T<:Union{Float64,Float32}}
    n = size(Xm, 1)
    @inbounds for j in 1:k
        s = 0.0
        @simd for i in 1:n
            s = muladd(Float64(Xm[i, j]), acc[i], s)
        end
        adj[base + j - 1] += s
    end
    return nothing
end

# `sum(x)` and `sum(abs2, x)` over tracked values: a whole-array reduction
# recorded as ONE node, its operands' tape indices kept in the tape's `ilist`
# side buffer.
#
# Recorded elementwise, `sum(abs2, r)` over N residuals is N OP_SQ and N-1
# OP_ADD nodes. That is the inner product of every normal log-density --
# PDMats' `invquad` for an isotropic covariance is exactly
# `sum(abs2, x .- mu)` -- and on an N=20000 regression those two node kinds
# were half of the tape. As one node, the forward pass writes a 4-byte index per
# term instead of two 40-byte nodes, and the reverse sweep is one tight loop.
#
# The primal is summed left to right rather than pairwise. The difference is
# rounding in the last bits of a Float64; the gradient is exact either way.
# (A comment, not a docstring: a docstring here would attach to `Base.sum`.)
Base.sum(x::AbstractArray{TapedReal}) = _record_sum(x, false)
Base.sum(::typeof(abs2), x::AbstractArray{TapedReal}) = _record_sum(x, true)

function _record_sum(x::AbstractArray{TapedReal}, sq::Bool)
    t = _livetape_vec(x)
    s = 0.0
    if t === _DETACHED            # all constants (or empty): nothing to record
        @inbounds for xi in x
            s += sq ? xi.v * xi.v : xi.v
        end
        return TapedReal(_DETACHED, Int32(0), s)
    end
    start = t.ni
    need = start + length(x)
    length(t.ilist) < need && resize!(t.ilist, max(need, 2 * length(t.ilist)))
    m = start
    il = t.ilist
    @inbounds for xi in x
        v = xi.v
        s += sq ? v * v : v
        # A detached entry is a constant: it moves the value, not the gradient.
        _isdetached(xi) && continue
        m += 1
        il[m] = xi.idx
    end
    t.ni = m
    return record!(t, sq ? OP_SUMSQ : OP_SUM, Int32(start + 1), Int32(m - start), s, 0.0)
end

@inline _linear1(t::Tape, x::TapedReal, c::Float64) =
    record!(t, OP_MULC, _attach(t, x).idx, Int32(0), x.v * c, c)

# `SpecialFunctions._logabsgamma` is THE single method gating Gamma, Beta,
# Poisson, Binomial, NegativeBinomial, TDist and Chisq — all seven fail on it
# and nothing else. Unlike the forward-mode side (which needs the value computed
# through differentiable arithmetic), here the primal is a plain Float64
# recorded once and the derivative is digamma, so calling out is fine.
SpecialFunctions._logabsgamma(x::TapedReal) =
    (record!(tape(x), OP_LGAMMA, x.idx, Int32(0),
             SpecialFunctions.logabsgamma(x.v)[1], 0.0), 1)
SpecialFunctions.loggamma(x::TapedReal) =
    record!(tape(x), OP_LGAMMA, x.idx, Int32(0), SpecialFunctions.loggamma(x.v), 0.0)

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
@inline _livetape(x::TapedReal, y::TapedReal) = _isdetached(x) ? tape(y) : tape(x)

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
# Asked of the TYPE by some Distributions paths (BernoulliLogit's logpdf
# reaches it via logistic/log1pexp). A property of the stored float.
Base.precision(::Type{TapedReal}) = precision(Float64)
Base.precision(x::TapedReal) = precision(Float64)
# `nextfloat`/`prevfloat` step to the adjacent representable value -- a guard
# against landing exactly on a boundary (BernoulliLogit's logpdf uses them).
# The step is infinitesimal, so the derivative is unchanged: only the primal
# moves. Returning the value with its partials intact is the right answer;
# treating it as a constant would silently zero a gradient.
@inline Base.nextfloat(x::TapedReal) = TapedReal(x.slot, x.idx, nextfloat(x.v))
@inline Base.prevfloat(x::TapedReal) = TapedReal(x.slot, x.idx, prevfloat(x.v))
@inline Base.eps(::Type{TapedReal}) = eps(Float64)
# Float introspection, all answering about the primal -- see the matching block
# in Piste.jl. Queries, not operations, so nothing is recorded.
@inline Base.exponent(x::TapedReal) = exponent(x.v)
@inline Base.significand(x::TapedReal) = significand(x.v)
@inline Base.frexp(x::TapedReal) = frexp(x.v)
@inline Base.signbit(x::TapedReal) = signbit(x.v)
@inline Base.sign(x::TapedReal) = sign(x.v)
@inline Base.issubnormal(x::TapedReal) = issubnormal(x.v)
@inline Base.decompose(x::TapedReal) = Base.decompose(x.v)
# ldexp scales by a power of two, so the partials scale identically.
@inline Base.ldexp(x::TapedReal, e::Integer) =
    record!(tape(x), OP_MULC, x.idx, Int32(0), ldexp(x.v, e), ldexp(1.0, e))
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
    _zero_adjoints!(adj, t)
    @inbounds adj[out] = 1.0
    return _reverse_sweep!(adj, t)
end

# Split out of `backward!` so a pullback can seed SEVERAL outputs (one adjoint
# per element of an array-valued `f`) before the one sweep. See
# `rev_value_and_pullback!`.
function _zero_adjoints!(adj::Vector{Float64}, t::Tape)
    n = t.n
    # Grow only when the reused buffer is genuinely too small; in the steady
    # state (a sampler calling this once per leapfrog step on a fixed model)
    # this never fires, so the sweep allocates nothing.
    length(adj) < n && resize!(adj, n)
    @inbounds for i in 1:n
        adj[i] = 0.0
    end
    # Zero the deferred matvec accumulators before the sweep.
    #
    # This is NOT redundant with `_claim_acc!`. The forward pass borrows each
    # accumulator as scratch for the primal (see `_matvec_taped`), and the sweep
    # below skips any node whose adjoint is zero (`ai == 0.0 && continue`), so a
    # skipped row would otherwise keep its PRIMAL and be flushed as though it
    # were an adjoint -- a silently wrong gradient, not a crash.
    @inbounds for m in 1:length(t.mref)
        a = t.accs[m]
        for i in 1:length(a)
            a[i] = 0.0
        end
    end
    return adj
end

# Walk the tape backwards from whatever adjoints have been seeded.
function _reverse_sweep!(adj::Vector{Float64}, t::Tape)
    n = t.n
    # Each matvec's deferred adjoint is applied when the sweep reaches the END
    # of its operand span (`hi`), not after the whole loop.
    #
    # Every row of a matvec is recorded after its operands, so by the time the
    # sweep is at `hi` all rows have been visited and the accumulator is
    # complete -- and the operands themselves have not been processed yet, so
    # their adjoints flow on to whatever produced them. Flushing after the loop
    # (the original version) was only right when the operands were OP_INPUT
    # leaves: `X * (2 .* b)` returned a ZERO gradient, silently, because the
    # re-recorded operands were never visited again. For leaf operands this is
    # the same arithmetic, applied earlier.
    #
    # The tape is swept in SEGMENTS between flush points, rather than testing
    # `i == hi` on every node: that per-node test measured 6-8% on a scalar
    # tape with no matvec at all (Rosenbrock, K=200). A tape without a matvec
    # is one segment, i.e. exactly the plain loop.
    hi = n
    nexthi = _next_flush(t, typemax(Int))
    while true
        _sweep!(adj, t, hi, nexthi + 1)
        nexthi == 0 && break
        @inbounds for m in 1:length(t.mref)
            Int(last(t.idxs[m])) == nexthi && _flush_one!(adj, t, m)
        end
        hi = nexthi
        nexthi = _next_flush(t, nexthi)
    end
    return adj
end

# Process nodes `hi` down to `lo`. One loop, one branch on a concrete `Int8`,
# every arm statically known -- the shape that lets the trimmer see through it.
function _sweep!(adj::Vector{Float64}, t::Tape, hi::Int, lo::Int)
    nodes = t.nodes
    accs = t.accs
    @inbounds for i in hi:-1:lo
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
        elseif op == OP_LOGISTIC
            # d/du s(u) = s(1-s), and `nd.v` IS s -- so the whole four-node
            # chain costs one multiply-accumulate and needs no saved partial.
            adj[nd.a] += ai * nd.v * (1.0 - nd.v)
        elseif op == OP_LGAMMA
            adj[nd.a] += ai * SpecialFunctions.digamma(nodes[nd.a].v)
        elseif op == OP_MATVEC
            # One row of X*beta. `c` is the side-table slot, `d` the row index.
            # DEFER: just record this row's adjoint. The whole contribution
            # `adj[base:base+k-1] += X' * acc` is applied once per matrix after
            # the loop, which is the same arithmetic as one BLAS-shaped pass
            # instead of a strided row walk per node. See `_flush_matvec!`.
            # Unchecked: both were stored from exact integers at record time,
            # and the checked `Int(::Float64)` (isinteger + trap) showed in the
            # profile on every row.
            accs[unsafe_trunc(Int, nd.c)][unsafe_trunc(Int, nd.d)] = ai
        elseif op == OP_SUM
            # `a` is where this reduction's operand indices start in `ilist`,
            # `b` how many there are.
            il = t.ilist
            for j in Int(nd.a):(Int(nd.a) + Int(nd.b) - 1)
                adj[il[j]] += ai
            end
        elseif op == OP_SUMSQ
            il = t.ilist
            ai2 = 2.0 * ai
            for j in Int(nd.a):(Int(nd.a) + Int(nd.b) - 1)
                k = il[j]
                adj[k] += ai2 * nodes[k].v
            end
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

# The largest operand-span end `hi` among the matvecs whose `hi` is below
# `bound`, or 0 if there is none: the next point at which the sweep must flush.
@inline function _next_flush(t::Tape, bound::Int)
    best = 0
    @inbounds for m in 1:length(t.mref)
        h = Int(last(t.idxs[m]))
        (h < bound && h > best) && (best = h)
    end
    return best
end

# Apply matrix `m`'s deferred adjoint: `adj[span] += X' * acc`. `mref` says
# which typed table holds the matrix; each branch calls the kernel on a
# concrete matrix type, so there is no dynamic dispatch here.
@inline function _flush_one!(adj::Vector{Float64}, t::Tape, m::Int)
    @inbounds begin
        r = t.idxs[m]
        base = Int(first(r)); k = Int(last(r)) - base + 1
        j = t.mref[m]
        if j > 0
            _flush_matvec!(adj, t.mats[j], t.accs[m], base, k)
        else
            _flush_matvec!(adj, t.mats32[-j], t.accs[m], base, k)
        end
    end
    return nothing
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
function rev_gradient!(g::AbstractVector{<:Real}, f::F, x::AbstractVector{<:Real},
                       ws::ReverseWorkspace) where {F}
    v, _ = rev_value_and_gradient!(g, f, x, ws)
    return g
end

rev_gradient!(g::AbstractVector{<:Real}, f::F, x::AbstractVector{<:Real}) where {F} =
    rev_gradient!(g, f, x, ReverseWorkspace(length(x)))

"""
    rev_value_and_gradient!(g, f, x, ws) -> (value, g)

As [`rev_gradient!`](@ref), also returning the primal — which the forward sweep
has already computed, so it costs nothing extra.
"""
# `g` and `x` are only required to be real vectors, not Float64 ones:
# PracticalBayes is Float32-first, so a Float32 theta has to work. The TAPE
# stays Float64 -- that is where the primals and partials live, and running the
# recording in reduced precision would cost accuracy for no benefit, since the
# tape is not the thing being stored in the model's parameter vector. Only the
# interface converts, so `eltype(g)` is preserved exactly.
function rev_value_and_gradient!(g::AbstractVector{<:Real}, f::F, x::AbstractVector{<:Real},
                                 ws::ReverseWorkspace) where {F}
    t = ws.tape
    # Borrow a slot for this call, unless the tape already holds one for good
    # (it was driven by hand through `track!`). Released in `finally`, so an
    # error inside `f` cannot leak it.
    owned = _claim_slot!(t)
    try
        return _rev_value_and_gradient!(g, f, x, ws, t)
    finally
        owned && _release_slot!(t)
    end
end

function _rev_value_and_gradient!(g::AbstractVector{<:Real}, f::F, x::AbstractVector{<:Real},
                                  ws::ReverseWorkspace, t::Tape) where {F}
    K = length(x)
    length(ws.xs) < K && resize!(ws.xs, K)
    reset!(t)
    @inbounds for i in 1:K
        ws.xs[i] = track!(t, Float64(x[i]))
    end
    y = f(view(ws.xs, 1:K))
    if y isa TapedReal && y.idx != Int32(0)
        backward!(ws.adj, t, y.idx)
        @inbounds for i in 1:K
            g[i] = convert(eltype(g), ws.adj[ws.xs[i].idx])
        end
        return (convert(eltype(g), y.v), g)
    end
    # `f` returned something untracked, i.e. it does not actually depend on `x`.
    # A zero gradient is the right answer, and saying so beats a confusing error.
    # That includes a DETACHED TapedReal (e.g. `zero(eltype(x))`): its index is
    # 0, and seeding `adj[0]` under `@inbounds` would write outside the buffer.
    fill!(g, zero(eltype(g)))
    return (convert(eltype(g), y isa TapedReal ? y.v : y), g)
end

rev_value_and_gradient!(g::AbstractVector{<:Real}, f::F, x::AbstractVector{<:Real}) where {F} =
    rev_value_and_gradient!(g, f, x, ReverseWorkspace(length(x)))

"""
    rev_value_and_pullback!(dx, f, x, dy, ws) -> (y, dx)
    rev_value_and_pullback!(dx, f, x, dy)     -> (y, dx)
    rev_value_and_pullback(f, x::Real, dy, ws) -> (y, dx)
    rev_value_and_pullback(f, x::Real, dy)     -> (y, dx)

Vector-Jacobian product in reverse mode: `dx = J' * dy`, where `J` is the
Jacobian of `f` at `x`, returned with the primal `y = f(x)`.

`f` may return a real number or an array of reals, and `dy` has the same shape
as that output. `x` may be an array of any shape, in which case the result is
written into `dx` (same shape as `x`), or a real number, in which case use the
non-mutating form. Either way it costs one recording of `f` and one reverse
sweep, however large the output is.

[`rev_value_and_gradient!`](@ref) is the special case of a scalar `f` with
`dy = 1`.

The returned primal is `Float64` (an array of them if `f` returns an array),
since that is what the tape holds. `dx` keeps its own element type.

```julia
f(x) = [x[1] * x[2], sin(x[1])]
x = [0.7, 1.3]
dx = similar(x)
ws = ReverseWorkspace(length(x))
y, dx = rev_value_and_pullback!(dx, f, x, [1.0, -2.0], ws)
```
"""
function rev_value_and_pullback!(dx::AbstractArray{<:Real}, f::F, x::AbstractArray{<:Real},
                                 dy, ws::ReverseWorkspace) where {F}
    length(dx) == length(x) ||
        throw(DimensionMismatch("rev_value_and_pullback!: dx and x must have the same length"))
    t = ws.tape
    owned = _claim_slot!(t)
    try
        K = length(x)
        length(ws.xs) < K && resize!(ws.xs, K)
        reset!(t)
        for (i, xi) in enumerate(x)
            @inbounds ws.xs[i] = track!(t, Float64(xi))
        end
        y = f(_taped_input(x, ws.xs, K))
        _seed_and_sweep!(ws.adj, t, y, dy)
        for (i, I) in enumerate(eachindex(dx))
            @inbounds dx[I] = convert(eltype(dx), ws.adj[ws.xs[i].idx])
        end
        return (_primal(y), dx)
    finally
        owned && _release_slot!(t)
    end
end

rev_value_and_pullback!(dx::AbstractArray{<:Real}, f::F, x::AbstractArray{<:Real}, dy) where {F} =
    rev_value_and_pullback!(dx, f, x, dy, ReverseWorkspace(length(x)))

function rev_value_and_pullback(f::F, x::Real, dy, ws::ReverseWorkspace) where {F}
    t = ws.tape
    owned = _claim_slot!(t)
    try
        reset!(t)
        xt = track!(t, Float64(x))
        y = f(xt)
        _seed_and_sweep!(ws.adj, t, y, dy)
        return (_primal(y), oftype(float(x), @inbounds ws.adj[xt.idx]))
    finally
        owned && _release_slot!(t)
    end
end

rev_value_and_pullback(f::F, x::Real, dy) where {F} =
    rev_value_and_pullback(f, x, dy, ReverseWorkspace(1))

# A vector input is handed to `f` as a view of the tracked buffer, exactly as
# `rev_value_and_gradient!` does; any other shape is reshaped to match `x`, so
# `f` sees the array shape it was written for.
@inline _taped_input(::AbstractVector, xs::Vector{TapedReal}, K::Int) = view(xs, 1:K)
@inline _taped_input(x::AbstractArray, xs::Vector{TapedReal}, K::Int) =
    reshape(view(xs, 1:K), size(x))

# Seed every output's adjoint with its cotangent, then sweep once. Seeding with
# `+=` is what makes an output that appears twice (`[x[1], x[1]]`) count twice.
function _seed_and_sweep!(adj::Vector{Float64}, t::Tape, y, dy)
    _zero_adjoints!(adj, t)
    _seed!(adj, y, dy)
    _reverse_sweep!(adj, t)
    return adj
end

# A detached or untracked output is a constant: it has nothing to seed, and its
# index 0 must never be written (see `_rev_value_and_gradient!`).
@inline function _seed!(adj::Vector{Float64}, y::TapedReal, dy::Real)
    _isdetached(y) || (@inbounds adj[y.idx] += Float64(dy))
    return nothing
end
@inline _seed!(::Vector{Float64}, ::Real, ::Real) = nothing
function _seed!(adj::Vector{Float64}, y::AbstractArray, dy::AbstractArray)
    length(y) == length(dy) ||
        throw(DimensionMismatch("rev_value_and_pullback!: dy must have the shape of f(x)"))
    for (yi, dyi) in zip(y, dy)
        _seed!(adj, yi, dyi)
    end
    return nothing
end

@inline _primal(y::TapedReal) = y.v
@inline _primal(y::Real) = y
_primal(y::AbstractArray) = map(_primal, y)
