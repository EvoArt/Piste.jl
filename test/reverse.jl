# Reverse mode.
#
# Same discipline as the forward-mode suite: nothing here checks a
# hand-computed number. Every test compares against ForwardDiff, central
# differences, or Piste's own forward mode — because a wrong gradient does not
# crash, it silently gives a wrong answer.
#
# Two testsets are regressions for bugs that actually occurred and produced
# plausible-looking wrong output rather than errors. They are named as
# regressions so nobody removes them as redundant.

@testset "reverse: arithmetic rules vs finite differences" begin
    fs = [
        ("+",     x -> x + 2.0),        ("-",     x -> x - 2.0),
        ("radd",  x -> 2.0 + x),        ("rsub",  x -> 2.0 - x),
        ("*",     x -> x * 3.0),        ("rmul",  x -> 3.0 * x),
        ("/",     x -> x / 3.0),        ("rdiv",  x -> 3.0 / x),
        ("neg",   x -> -x),             ("^2",    x -> x^2),
        ("^3",    x -> x^3),            ("exp",   x -> exp(x)),
        ("log",   x -> log(x)),         ("sqrt",  x -> sqrt(x)),
        ("sin",   x -> sin(x)),         ("cos",   x -> cos(x)),
        ("tan",   x -> tan(x)),         ("tanh",  x -> tanh(x)),
        ("atan",  x -> atan(x)),        ("abs",   x -> abs(x)),
        ("inv",   x -> inv(x)),         ("log1p", x -> log1p(x)),
        ("expm1", x -> expm1(x)),
        ("chain", x -> exp(sin(x) * log(x + 3.0)) / sqrt(x + 1.0)),
        ("x*x",   x -> x * x),          ("x/x",   x -> x / x),
    ]
    for (name, f) in fs
        # wrap the scalar function as a 1-vector objective
        vf = v -> f(v[1])
        x = [1.3]
        g = zeros(1)
        rev_gradient!(g, vf, x)
        h = 1e-6
        fd = (f(1.3 + h) - f(1.3 - h)) / (2h)
        @test isapprox(g[1], fd; atol=1e-5)
    end
end

@testset "reverse: agrees with forward mode and ForwardDiff" begin
    rng = Xoshiro(11)
    X = randn(rng, 40, 6)
    yv = randn(rng, 40)
    f = b -> sum(abs2, yv .- X * b) + sum(exp, b) / 10
    x = randn(Xoshiro(3), 6)

    gref = ForwardDiff.gradient(f, x)
    gr = zeros(6)
    rev_gradient!(gr, f, x)
    @test gr ≈ gref rtol=1e-10

    gfwd = zeros(6)
    gradient!(gfwd, f, x, Val(4))
    @test gr ≈ gfwd rtol=1e-10
end

@testset "reverse: constants are real tape nodes (regression)" begin
    # `convert`/`zero`/`one` originally gave constants tape index 0, but 0 is
    # not a valid index, so the reverse sweep's `adj[nd.a]` for such a node
    # indexed OUTSIDE the adjoint array. Three of the first four distributions
    # tested came back with `Inf`/`-0.0` gradients while running without error
    # — the silent-wrong-answer mode, not a crash.
    fs = [v -> 2.0 - v[1],
          v -> 1.0 / v[1],
          v -> zero(v[1]) + v[1] * 3.0,
          v -> one(v[1]) * v[1],
          v -> v[1] - 5.0 + 5.0]
    for f in fs
        x = [1.7]; g = zeros(1)
        rev_gradient!(g, f, x)
        h = 1e-6
        fd = (f([1.7 + h]) - f([1.7 - h])) / (2h)
        @test isfinite(g[1])
        @test isapprox(g[1], fd; atol=1e-5)
    end
end

@testset "reverse: scalar logpdf derivatives" begin
    cases = [
        ("Normal loc",   mu -> logpdf(Normal(mu, 1.0), 0.7),   0.3),
        ("Normal scale", s  -> logpdf(Normal(0.0, s), 0.7),    1.4),
        ("Exponential",  s  -> logpdf(Exponential(s), 0.7),    1.4),
        ("LogNormal",    mu -> logpdf(LogNormal(mu, 1.0), 0.7), 0.3),
        ("Cauchy",       x0 -> logpdf(Cauchy(x0, 5.0), 0.7),   0.3),
        ("Uniform",      b  -> logpdf(Uniform(0.0, b), 0.7),   2.0),
    ]
    for (name, f, x0) in cases
        vf = v -> f(v[1])
        g = zeros(1)
        rev_gradient!(g, vf, [x0])
        h = 1e-6
        fd = (f(x0 + h) - f(x0 - h)) / (2h)
        @test isapprox(g[1], fd; atol=1e-5)
    end
end

@testset "reverse: lgamma family (regression: _logabsgamma)" begin
    # As in forward mode, `SpecialFunctions._logabsgamma` is the SINGLE method
    # gating all seven of these. STADE, the other trimmable path, cannot do
    # them at all.
    cases = [
        ("Gamma",       a -> logpdf(Gamma(a, 1.0), 0.7),          2.0),
        ("Beta",        a -> logpdf(Beta(a, 2.0), 0.5),           2.0),
        ("Poisson",     l -> logpdf(Poisson(l), 3),               2.0),
        ("Binomial",    p -> logpdf(Binomial(10, p), 4),          0.4),
        ("NegBinomial", p -> logpdf(NegativeBinomial(3.0, p), 4), 0.4),
        ("TDist",       v -> logpdf(TDist(v), 0.7),               5.0),
        ("Chisq",       k -> logpdf(Chisq(k), 1.5),               3.0),
    ]
    for (name, f, x0) in cases
        vf = v -> f(v[1])
        g = zeros(1)
        rev_gradient!(g, vf, [x0])
        h = 1e-6
        fd = (f(x0 + h) - f(x0 - h)) / (2h)
        @test isapprox(g[1], fd; atol=1e-4)
    end
end

@testset "reverse: truncated Cauchy (regression: two-argument atan)" begin
    # Missing `atan(y, x)` is a StackOverflowError, not a MethodError — both
    # arguments promote and recurse. And without the infinite-argument guard
    # the truncation bound at Inf gives a NaN derivative beside a correct value.
    f = s -> logpdf(truncated(Cauchy(0.0, s), 0, Inf), 0.7)
    vf = v -> f(v[1])
    g = zeros(1)
    rev_gradient!(g, vf, [2.0])
    @test isfinite(g[1])
    h = 1e-6
    @test isapprox(g[1], (f(2.0 + h) - f(2.0 - h)) / (2h); atol=1e-5)
end

@testset "reverse: multivariate likelihoods" begin
    cases = [
        (mu -> logpdf(MvNormal(fill(mu, 3), I), [0.1, 0.2, 0.3]), 0.3),
        (s  -> logpdf(MvNormal(zeros(3), Diagonal(fill(s, 3))), [0.1, 0.2, 0.3]), 1.5),
        (mu -> loglikelihood(Normal(mu, 1.0), [0.1, 0.2, 0.3]), 0.3),
    ]
    for (f, x0) in cases
        vf = v -> f(v[1])
        g = zeros(1)
        rev_gradient!(g, vf, [x0])
        h = 1e-6
        @test isapprox(g[1], (f(x0 + h) - f(x0 - h)) / (2h); atol=1e-5)
    end
end

@testset "reverse: workspace reuse gives identical answers" begin
    # Reuse is what makes repeated calls cheap (profiling put 86% of runtime in
    # `push!` without it), so it must not change the answer.
    f = x -> sum(abs2, x) + exp(x[1]) * log(x[2] + 3.0)
    x = [0.7, 1.3, -0.4]
    ws = ReverseWorkspace(3)
    g1 = zeros(3); rev_gradient!(g1, f, x, ws)
    for _ in 1:5
        g2 = zeros(3); rev_gradient!(g2, f, x, ws)
        @test g2 == g1
    end
    # a fresh workspace must agree with the reused one
    g3 = zeros(3); rev_gradient!(g3, f, x)
    @test g3 ≈ g1
    # and the tape is actually being rewound, not grown
    n_after = length(ws.tape)
    rev_gradient!(g1, f, x, ws)
    @test length(ws.tape) == n_after
end

@testset "reverse: value_and_gradient! returns the true primal" begin
    f = x -> sum(abs2, x) + exp(x[1])
    x = [0.7, 1.3, -0.4]
    g = zeros(3)
    v, g2 = rev_value_and_gradient!(g, f, x)
    @test v ≈ f(x)
    @test g2 === g
    @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-10
end

@testset "reverse: scales better than forward mode" begin
    # The reason reverse mode is here: cost is one sweep regardless of K, so on
    # a many-parameter objective it must produce the same answer as forward mode
    # while doing far less work. (Correctness is what is asserted; the timing
    # claim lives in the benchmarks, not the test suite.)
    f = v -> sum(abs2, v) + exp(v[1]) * log(abs(v[end]) + 2.0)
    for K in (1, 2, 7, 50, 200)
        x = [0.3 + 0.01 * i for i in 1:K]
        gr = zeros(K); rev_gradient!(gr, f, x)
        @test gr ≈ ForwardDiff.gradient(f, x) rtol=1e-9
    end
end

@testset "reverse: TapedReal is a well-behaved Real" begin
    t = Tape()
    x = track!(t, 1.5)
    @test x isa Real
    @test value(x) == 1.5
    @test x > 1.0 && x < 2.0
    @test isfinite(x) && !isnan(x) && !isinf(x)
    @test isinteger(track!(t, 3.0))
    @test !isinteger(track!(t, 3.5))
    @test promote_type(TapedReal, Int) === TapedReal
    # reset! rewinds without freeing
    n = length(t)
    @test n > 0
    reset!(t)
    @test length(t) == 0
    @test isempty(t)
end

@testset "reverse: a function ignoring its input has zero gradient" begin
    # Returns an untracked value; a zero gradient is the right answer and beats
    # a confusing error.
    g = zeros(3)
    v, _ = rev_value_and_gradient!(g, x -> 42.0, [1.0, 2.0, 3.0])
    @test v == 42.0
    @test all(iszero, g)
end

@testset "reverse: array-level matvec" begin
    # `X * beta` is recorded as one node per output row rather than ~2K scalar
    # nodes. This is the largest single saving in the tape (58 281 -> 5 281
    # nodes on a 1000x50 regression), so it is worth checking across shapes --
    # especially the degenerate ones, and the reuse case where a stale side
    # table would show up.
    for (n, k) in [(1,1), (1,5), (5,1), (3,3), (7,4), (50,10)]
        rng = Xoshiro(n*100 + k)
        X = randn(rng, n, k); yv = randn(rng, n)
        f = b -> sum(abs2, yv .- X*b) + sum(exp, b)/10
        x = randn(Xoshiro(k), k)
        g = zeros(k)
        rev_gradient!(g, f, x, ReverseWorkspace(k))
        @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-9
    end

    # A workspace carries the side table too; reuse must not leave it stale.
    rng = Xoshiro(7); X = randn(rng, 30, 6); yv = randn(rng, 30)
    f = b -> sum(abs2, yv .- X*b)
    x = randn(Xoshiro(2), 6)
    ws = ReverseWorkspace(6)
    g1 = zeros(6); rev_gradient!(g1, f, x, ws)
    for _ in 1:5
        g2 = zeros(6); rev_gradient!(g2, f, x, ws)
        @test g2 == g1
    end
    @test g1 ≈ ForwardDiff.gradient(f, x) rtol=1e-9

    # The node count must scale with N and NOT with N*K. Asserted by holding N
    # fixed and varying K: if the elementwise path comes back, the count grows
    # with K. This caught a real dispatch bug — LinearAlgebra's
    # `*(::StridedMatrix{Float64}, ::StridedVector{S<:Real})` is more specific
    # than a plain `AbstractVector{TapedReal}` signature, so it silently won and
    # the array-level rule never fired, with no error to notice.
    counts = map((5, 20, 80)) do kk
        Xb = randn(Xoshiro(4), 100, kk); yb = randn(Xoshiro(5), 100)
        fb = b -> sum(abs2, yb .- Xb*b)
        wsb = ReverseWorkspace(kk)
        gb = zeros(kk); rev_gradient!(gb, fb, randn(Xoshiro(6), kk), wsb)
        @test gb ≈ ForwardDiff.gradient(fb, randn(Xoshiro(6), kk)) rtol=1e-9
        length(wsb.tape)
    end
    # 16x more parameters must not mean materially more nodes
    @test counts[3] < 2 * counts[1]
end

@testset "reverse: matvec rows with zero adjoint (regression)" begin
    # The matvec adjoint is DEFERRED: the sweep records each row's adjoint into
    # a per-matrix accumulator and applies `X' * acc` once at the end. That
    # accumulator is also borrowed as scratch for the forward primal, and the
    # sweep SKIPS nodes whose adjoint is zero -- so a row the output does not
    # depend on never overwrites its borrowed primal. Unless the accumulator is
    # re-zeroed first, that leftover primal is flushed as if it were a gradient
    # contribution: a silently wrong answer, with no error raised.
    #
    # Every objective here uses only SOME rows of `X * b`, so the rest keep a
    # zero adjoint and the bug, if reintroduced, shows up as a wrong gradient.
    rng = Xoshiro(11)
    X = randn(rng, 40, 5)
    x = randn(Xoshiro(12), 5)
    for f in (b -> sum(abs2, (X*b)[1:3]),        # a leading slice
              b -> (X*b)[1],                      # exactly one row
              b -> (X*b)[end],                    # exactly one row, at the end
              b -> sum(abs2, (X*b)[2:2:end]))     # every other row
        g = zeros(5)
        rev_gradient!(g, f, x, ReverseWorkspace(5))
        @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-9
    end

    # The same workspace reused across DIFFERENT objectives: the second call's
    # zero-adjoint rows must not pick up the first call's values either.
    ws = ReverseWorkspace(5)
    f1 = b -> sum(abs2, X*b)
    f2 = b -> sum(abs2, (X*b)[1:3])
    g1 = zeros(5); rev_gradient!(g1, f1, x, ws)
    g2 = zeros(5); rev_gradient!(g2, f2, x, ws)
    @test g2 ≈ ForwardDiff.gradient(f2, x) rtol=1e-9
end

@testset "reverse: logistic fusion (peephole)" begin
    # `1/(1+exp(-u))` is collapsed from four nodes to one OP_LOGISTIC by a
    # peephole that rewrites the tail of the tape. A peephole that fires when it
    # should not gives a WRONG GRADIENT with no error, so this checks both the
    # fused shape and every near-miss that must NOT fuse.
    logistic(x) = 1 / (1 + exp(-x))

    # The fused shape, over a range that covers both tails.
    for x0 in (-8.0, -1.0, 0.0, 0.5, 3.0, 9.0)
        f = v -> logistic(v[1])
        g = zeros(1); rev_gradient!(g, f, [x0], ReverseWorkspace(1))
        s = logistic(x0)
        @test g[1] ≈ s * (1 - s) rtol=1e-10
        @test g[1] ≈ ForwardDiff.gradient(f, [x0])[1] rtol=1e-10
    end

    # Fusion must not lose the chain rule through its argument.
    f2 = v -> logistic(2.5 * v[1] + v[2] * v[2])
    x2 = [0.4, -1.1]
    g2 = zeros(2); rev_gradient!(g2, f2, x2, ReverseWorkspace(2))
    @test g2 ≈ ForwardDiff.gradient(f2, x2) rtol=1e-10

    # Near misses: each differs from the fused pattern in one way, so each must
    # fall through to the unfused path and still be right.
    nearmiss = [
        v -> 2 / (1 + exp(-v[1])),        # numerator is not 1
        v -> 1 / (2 + exp(-v[1])),        # addend is not 1
        v -> 1 / (1 + exp(v[1])),         # argument is not negated
        v -> 1 / (1 + exp(-v[1]) + 0.0*v[2]),  # extra node after the chain
        v -> 1 / (1 - exp(-v[1])),        # minus, not plus
        v -> 1 / (1 + 2*exp(-v[1])),      # scaled exponential
    ]
    xn = [0.7, 0.3]
    for f in nearmiss
        g = zeros(2); rev_gradient!(g, f, xn, ReverseWorkspace(2))
        @test g ≈ ForwardDiff.gradient(f, xn) rtol=1e-9
    end

    # The fusion must actually fire -- otherwise the test above passes for the
    # wrong reason and the speedup silently disappears. The chain below it stays
    # on the tape (see the reuse regression below), so check the output node's op.
    ws = ReverseWorkspace(1)
    g = zeros(1); rev_gradient!(g, v -> logistic(v[1]), [0.3], ws)
    @test ws.tape.nodes[length(ws.tape)].op == Piste.OP_LOGISTIC
    rev_gradient!(g, v -> 1 / (2 + exp(-v[1])), [0.3], ws)
    @test ws.tape.nodes[length(ws.tape)].op != Piste.OP_LOGISTIC

    # Reused inside a sequential loop, the shape the HMM likelihood has.
    f3 = v -> begin
        s = zero(eltype(v)); a = one(eltype(v))
        for i in 1:5
            p = 1 / (1 + exp(-(v[1] * i + v[2])))
            a = a * p
            s = s + log(a)
        end
        s
    end
    x3 = [0.25, -0.4]
    g3 = zeros(2); rev_gradient!(g3, f3, x3, ReverseWorkspace(2))
    @test g3 ≈ ForwardDiff.gradient(f3, x3) rtol=1e-9
end

@testset "reverse: fused intermediates stay valid when reused (regression)" begin
    # Both peepholes used to DROP the nodes they fused and reuse their tape
    # slots. A value the caller still held then pointed at the fused node, and
    # the gradient came back wrong with no error: 4.0 instead of 1.0 for the
    # first case below, +0.244 instead of -0.496 for the second.
    fs = [
        v -> (p = v[1] * 2.0; s = v[2] + p; s + p * 3.0),           # s + x*c, then x*c reused
        v -> (p = v[1] * 2.0; s = p + v[2]; s * p),                 # operands swapped
        v -> (e = exp(-v[1]); s = 1.0 / (1.0 + e); s + e * v[2]),   # exp(-u) reused after logistic
        v -> (d = 1.0 + exp(-v[1]); s = 1.0 / d; s * d + v[2]),     # 1 + exp(-u) reused
    ]
    x = [0.3, 0.7]
    for f in fs
        g = zeros(2); rev_gradient!(g, f, x, ReverseWorkspace(2))
        @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-10
    end
end

@testset "reverse: TapedReal is isbits and slots are returned" begin
    # Arrays of TapedReal must be plain bits: a pointer field made every
    # matvec output and broadcast a GC-scanned, write-barriered, zero-filled
    # array, a third of gradient time on a normal regression.
    @test isbitstype(TapedReal)
    @test sizeof(TapedReal) == 16

    # Each call borrows a slot and returns it. More calls than there are slots
    # must therefore work, both normally and when `f` throws partway through.
    f = v -> sum(abs2, v) + exp(v[1])
    x = [0.3, -0.2]
    g = zeros(2)
    for _ in 1:(Piste._MAX_SLOTS + 10)
        rev_gradient!(g, f, x)                 # a fresh workspace (and tape) per call
    end
    @test g ≈ ForwardDiff.gradient(f, x)
    bad = v -> (v[1] > 0 ? error("boom") : v[1])
    nthrown = 0
    for _ in 1:(Piste._MAX_SLOTS + 10)
        try
            rev_gradient!(g, bad, [1.0])
        catch e
            e isa ErrorException && (nthrown += 1)
        end
    end
    @test nthrown == Piste._MAX_SLOTS + 10
    rev_gradient!(g, f, x)
    @test g ≈ ForwardDiff.gradient(f, x)

    # Concurrent gradients on separate workspaces each read only their own slot.
    X = randn(Xoshiro(5), 60, 4); yv = randn(Xoshiro(6), 60)
    h = b -> sum(abs2, yv .- X * b) + sum(exp, b)
    xs = [randn(Xoshiro(i), 4) for i in 1:64]
    out = [zeros(4) for _ in 1:64]
    Threads.@threads for i in 1:64
        rev_gradient!(out[i], h, xs[i], ReverseWorkspace(4))
    end
    @test all(out[i] ≈ ForwardDiff.gradient(h, xs[i]) for i in 1:64)

    # A DETACHED result (idx 0) used to seed adj[0] under @inbounds.
    g3 = ones(3)
    v, _ = rev_value_and_gradient!(g3, z -> zero(eltype(z)) + 2.0, [1.0, 2.0, 3.0])
    @test v == 2.0
    @test g3 == zeros(3)
end

@testset "reverse: matvec over a COMPUTED vector (regression)" begin
    # The deferred matvec adjoint used to be flushed after the whole sweep,
    # which is only right when the operands are the raw parameters. Over any
    # computed vector the gradient came back as exactly ZERO, silently:
    # `X * (2 .* b)` gave [0, 0] for [-162.4, -208.0]. Every case here goes
    # through that path.
    X = randn(Xoshiro(21), 12, 4); X32 = Float32.(randn(Xoshiro(22), 12, 4))
    Y = randn(Xoshiro(23), 12, 4); yv = randn(Xoshiro(24), 12)
    fs = [
        b -> sum(abs2, X * (2.0 .* b)),
        b -> sum(abs2, yv .- X * b[end:-1:1]),                 # permuted
        b -> sum(abs2, X * exp.(b)) + sum(b),                  # nonlinear
        b -> (u = b .* b; sum(X * u) + sum(abs2, Y * u)),      # two matvecs, one operand
        b -> sum(abs2, X * b) + sum(abs2, Y * (b .+ 1.0)),     # raw and computed
        b -> sum(abs2, X32 * (0.5 .* b)) + sum(X * b),         # Float32 data
        b -> (z = X * (b .* 3.0); sum(abs2, Y * (z[1:4] .- b))),  # matvec of a matvec
    ]
    x = [0.3, -0.7, 0.2, 1.1]
    for f in fs
        g = zeros(4); rev_gradient!(g, f, x, ReverseWorkspace(4))
        @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-6
    end
end

@testset "reverse: matvec over a vector starting with a constant (regression)" begin
    # [constant, x1, x2]: the tracked part is tape indices 1..2, so a span check
    # that accepted index 0 as its start saw "contiguous 0:2" and the flush
    # wrote adj[0].
    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    f = v -> sum(abs2, X * TapedReal[TapedReal(0.5), v[1], v[2]])
    ref = v -> sum(abs2, X * [0.5, v[1], v[2]])
    x = [0.3, -0.7]
    g = zeros(2); rev_gradient!(g, f, x, ReverseWorkspace(2))
    @test g ≈ ForwardDiff.gradient(ref, x) rtol=1e-12
end

@testset "reverse: array-level sum and sum(abs2)" begin
    fs = [
        v -> sum(v),
        v -> sum(abs2, v),
        v -> sum(abs2, [1.5, 2.0, 0.5] .- v),        # the normal log-density shape
        v -> sum(v[[3, 1, 2]] .* 2.0) * v[1],         # operands not contiguous on the tape
        v -> sum(abs2, reshape([v[1], v[2], v[3], v[1] * v[2]], 2, 2)),  # a matrix
        v -> sum(abs2, v) * sum(v) + sum(abs2, v),    # reused, and two reductions
    ]
    x = [0.4, -1.3, 0.8]
    for f in fs
        g = zeros(3); rev_gradient!(g, f, x, ReverseWorkspace(3))
        @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-12
        v, _ = rev_value_and_gradient!(g, f, x, ReverseWorkspace(3))
        @test v ≈ f(x) rtol=1e-14
    end
    # A detached (constant) entry inside the array. These build TapedReal
    # arrays explicitly, which ForwardDiff cannot run, so each has a plain twin
    # as the reference.
    pairs = [
        (v -> sum(TapedReal[v[1], TapedReal(2.0), v[3]]),          v -> v[1] + 2.0 + v[3]),
        (v -> sum(abs2, TapedReal[TapedReal(3.0), v[2]]) + sum(abs2, v),
         v -> 9.0 + v[2]^2 + sum(abs2, v)),
    ]
    for (f, ref) in pairs
        g = zeros(3); v, _ = rev_value_and_gradient!(g, f, x, ReverseWorkspace(3))
        @test g ≈ ForwardDiff.gradient(ref, x) rtol=1e-12
        @test v ≈ ref(x) rtol=1e-14
    end
    # Empty and all-constant reductions have no tape and a zero gradient.
    g = ones(3); rev_gradient!(g, v -> sum(v[1:0]) + v[1], x)
    @test g == [1.0, 0.0, 0.0]

    # One node for the whole reduction, not N-1 adds -- otherwise the tests above
    # pass for the wrong reason and the speedup is gone.
    ws = ReverseWorkspace(200)
    xb = randn(Xoshiro(9), 200)
    rev_gradient!(zeros(200), v -> sum(abs2, v), xb, ws)
    @test length(ws.tape) == 201

    # Workspace reuse across different reduction sizes.
    for n in (5, 200, 3, 200)
        f = v -> sum(abs2, v[1:n]) + sum(v[1:n])
        g = zeros(200); rev_gradient!(g, f, xb, ws)
        @test g ≈ ForwardDiff.gradient(f, xb) rtol=1e-12
    end
end

@testset "reverse: LogExpFunctions rules (extension)" begin
    LEF = LogExpFunctions
    @test Base.get_extension(Piste, :PisteLogExpFunctionsExt) !== nothing
    cases = [
        (LEF.log1pexp, (-800.0, -50.0, -20.0, -1.0, 0.0, 1.0, 20.0, 35.0, 50.0)),
        (LEF.logistic, (-50.0, -3.0, 0.0, 0.7, 40.0)),
        (LEF.logit,    (1e-6, 0.2, 0.5, 0.9, 1 - 1e-6)),
        (LEF.log1mexp, (-50.0, -2.0, -0.5, -1e-3)),
        (LEF.logexpm1, (1e-3, 0.5, 2.0, 25.0, 50.0)),
        (LEF.log1psq,  (-3.0, 0.0, 0.4, 1e5)),
    ]
    for (fn, xs) in cases, x0 in xs
        f = v -> fn(v[1]) * 1.5
        g = zeros(1); v, _ = rev_value_and_gradient!(g, f, [x0])
        @test v ≈ f([x0]) rtol=1e-14
        @test g[1] ≈ ForwardDiff.gradient(f, [x0])[1] rtol=1e-9 atol=1e-300
    end
    # Each is ONE node on top of its input.
    ws = ReverseWorkspace(1)
    rev_gradient!(zeros(1), v -> LEF.log1pexp(v[1]), [0.3], ws)
    @test length(ws.tape) == 2

    # The model it was written for: BernoulliLogit's logpdf is -log1pexp(-x).
    X = randn(Xoshiro(7), 40, 3); yb = rand(Xoshiro(8), Bool, 40)
    f = b -> sum(logpdf.(BernoulliLogit.(X * b), yb))
    b0 = [0.3, -0.5, 1.1]
    g = zeros(3); rev_gradient!(g, f, b0)
    @test g ≈ ForwardDiff.gradient(f, b0) rtol=1e-10
end

@testset "reverse: matvec scratch is pooled, not reallocated" begin
    # The array-level matvec keeps its primal and adjoint scratch in the tape's
    # pooled accumulators, so neither grows with the number of ROWS once warm.
    #
    # This cannot assert zero: `X * b` returns a `Vector{TapedReal}` and the
    # objective's own `yv .- (...)` allocates a broadcast result, both of which
    # belong to the user's code rather than the tape. What it CAN assert is that
    # the tape adds nothing on top -- a regression that reintroduces a per-call
    # `mu` buffer (or stops pooling the accumulator) makes the total scale with
    # N a second time, which is what the comparison below catches.
    rng = Xoshiro(21)
    X = randn(rng, 500, 20); yv = randn(rng, 500)
    f = b -> sum(abs2, yv .- X*b)
    x = randn(Xoshiro(22), 20)
    g = zeros(20); ws = ReverseWorkspace(20)
    rev_gradient!(g, f, x, ws)          # warm: sizes the pooled buffers
    rev_gradient!(g, f, x, ws)
    a = @allocated rev_gradient!(g, f, x, ws)

    # The unavoidable part: one Vector{TapedReal} of length N (the matvec's
    # output) plus one Vector{Float64} of length N (the broadcast). Anything
    # much beyond that means scratch is being reallocated per call.
    unavoidable = 500 * sizeof(TapedReal) + 500 * sizeof(Float64)
    @test a < 2 * unavoidable

    # The decisive check: DOUBLE the rows. The per-call cost must grow by only
    # the two user-visible arrays above, not by a third buffer of the tape's
    # own. If the pooled accumulator were reallocated per call this ratio would
    # be visibly larger than the array-size ratio.
    X2 = randn(Xoshiro(23), 1000, 20); yv2 = randn(Xoshiro(24), 1000)
    f2 = b -> sum(abs2, yv2 .- X2*b)
    ws2 = ReverseWorkspace(20)
    rev_gradient!(g, f2, x, ws2); rev_gradient!(g, f2, x, ws2)
    a2 = @allocated rev_gradient!(g, f2, x, ws2)
    @test a2 < 2.3 * a
end

@testset "reverse: Float32 data matrix is not copied (regression)" begin
    # A `Matrix{Float32}` used to be converted to `Matrix{Float64}` on every
    # gradient call, copying all of X, so a Float32 model ran 1.3-2.5x slower
    # than the same model in Float64. It is now kept by reference in its own
    # side table; the answer must match the Float64 path, which the testsets
    # above already check against ForwardDiff.
    rng = Xoshiro(31)
    X32 = randn(rng, Float32, 500, 40); X64 = Float64.(X32)
    yv = randn(rng, 500)
    x = randn(Xoshiro(32), 40)
    g32 = zeros(40); g64 = zeros(40)
    f32 = b -> sum(abs2, yv .- X32*b) + sum(exp, b)/10
    f64 = b -> sum(abs2, yv .- X64*b) + sum(exp, b)/10
    ws32 = ReverseWorkspace(40); ws64 = ReverseWorkspace(40)
    rev_gradient!(g32, f32, x, ws32); rev_gradient!(g64, f64, x, ws64)
    @test g32 ≈ g64 rtol=1e-12

    # Both a Float32 and a Float64 matrix in one objective, so the two side
    # tables are used in the same sweep.
    fmix = b -> sum(abs2, yv .- X32*b) + sum(abs2, X64*b) / 7
    fref = b -> sum(abs2, yv .- X64*b) + sum(abs2, X64*b) / 7
    gm = zeros(40); gr = zeros(40)
    rev_gradient!(gm, fmix, x, ReverseWorkspace(40))
    rev_gradient!(gr, fref, x, ReverseWorkspace(40))
    @test gm ≈ gr rtol=1e-12

    # No per-call copy: the same allocation budget as the pooling test above.
    # A copy of X would add 500*40*8 = 160 kB, several times this bound.
    rev_gradient!(g32, f32, x, ws32)
    a = @allocated rev_gradient!(g32, f32, x, ws32)
    @test a < 2 * (500 * sizeof(TapedReal) + 500 * sizeof(Float64))
end

@testset "reverse: pullback (vector-Jacobian product) vs ForwardDiff" begin
    rng = Xoshiro(41)
    X = randn(rng, 5, 3)
    x = randn(Xoshiro(42), 3)

    # Vector in, vector out, through a matvec so the deferred-adjoint path is used.
    f = b -> exp.(X * b) .+ sin(b[1]) .* b[2]
    dy = randn(Xoshiro(43), 5)
    dx = zeros(3)
    y, dx = rev_value_and_pullback!(dx, f, x, dy, ReverseWorkspace(3))
    @test y ≈ f(x)
    @test dx ≈ ForwardDiff.jacobian(f, x)' * dy rtol=1e-10

    # Scalar output: the pullback is the gradient scaled by the cotangent.
    h = b -> sum(abs2, X * b) + exp(b[1])
    g = zeros(3)
    rev_gradient!(g, h, x)
    v, dx = rev_value_and_pullback!(zeros(3), h, x, -2.5)
    @test v ≈ h(x)
    @test dx ≈ -2.5 .* g rtol=1e-12

    # Matrix in, matrix out: `f` must see the input's own shape.
    M = randn(Xoshiro(44), 2, 3)
    fm = A -> A' * A .+ log1p.(A .^ 2)[1]
    dY = randn(Xoshiro(45), 3, 3)
    dM = zeros(2, 3)
    Y, dM = rev_value_and_pullback!(dM, fm, M, dY)
    @test Y ≈ fm(M)
    @test vec(dM) ≈ ForwardDiff.jacobian(A -> vec(fm(A)), M)' * vec(dY) rtol=1e-10

    # Scalar input.
    fs = t -> [sin(t), t^3, 2t]
    yv, dt = rev_value_and_pullback(fs, 0.7, [1.0, -1.0, 0.5])
    @test yv ≈ fs(0.7)
    @test dt ≈ cos(0.7) - 3 * 0.7^2 + 1.0

    # A workspace is reusable across calls of different shapes.
    ws = ReverseWorkspace(1)
    _, dx1 = rev_value_and_pullback!(zeros(3), f, x, dy, ws)
    _, dx2 = rev_value_and_pullback!(zeros(3), f, x, dy, ws)
    @test dx1 == dx2
end

@testset "reverse: pullback seeds repeated and constant outputs (regression)" begin
    # An output that appears twice must count twice, and an output that does
    # not depend on `x` (here a detached `zero`) must not be seeded at all:
    # its index is 0, and writing `adj[0]` under `@inbounds` is out of bounds.
    f = b -> [b[1], b[1], zero(eltype(b)), b[2] * b[1]]
    x = [0.3, -1.1]
    dy = [1.0, 2.0, 100.0, -1.0]
    y, dx = rev_value_and_pullback!(zeros(2), f, x, dy)
    @test y ≈ [0.3, 0.3, 0.0, -0.33]
    @test dx ≈ [1.0 + 2.0 - x[2], -x[1]]
end

@testset "broadcasting with a scalar Dual / TapedReal (regression)" begin
    # `eltype` of a Number type must be the type itself: broadcasting reads it
    # to size the output, so `eltype(Dual) == Float64` allocated a Float64
    # array for `[1, 2] .* d` and then failed to store a Dual in it.
    @test eltype(Dual{2,Float64}) === Dual{2,Float64}
    @test eltype(TapedReal) === TapedReal
    d = Dual{1,Float64}(0.5, (1.0,))
    y = sin.([1, 2] .* d)
    @test y isa Vector{Dual{1,Float64}}
    @test value.(y) ≈ sin.([1, 2] .* 0.5)
    @test [partials(yi, 1) for yi in y] ≈ [1, 2] .* cos.([1, 2] .* 0.5)
    y, dx = rev_value_and_pullback(x -> sin.([1, 2] .* x), 0.5, [1.0, 1.0])
    @test y ≈ sin.([1, 2] .* 0.5)
    @test dx ≈ sum([1, 2] .* cos.([1, 2] .* 0.5))
end
