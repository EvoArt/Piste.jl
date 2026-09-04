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
