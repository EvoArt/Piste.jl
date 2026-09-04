# Piste's test suite.
#
# The governing principle: a wrong gradient does not crash. It silently produces
# a wrong answer, which in a sampler means a wrong posterior. So nothing here
# checks a hand-computed number — every test compares against either
# ForwardDiff (a mature reference implementation) or central differences.
#
# Three of these testsets are regressions for methods that were silently missing
# during development, each found by measurement rather than reasoning. They are
# kept named as regressions so nobody removes them as redundant.

using Piste
using Piste: Dual, value, partials, gradient!, value_and_gradient!
using Piste: GradientWorkspace, pickchunk
using Piste: TapedReal, Tape, ReverseWorkspace, track!, reset!, rev_gradient!, rev_value_and_gradient!
import SpecialFunctions
using Test
using Distributions
using LinearAlgebra: I, Diagonal
using Random: Xoshiro
import ForwardDiff

# A dual seeded in direction `i`, for exercising scalar rules directly.
_d(x, i, ::Val{N}) where {N} = Dual{N,Float64}(x, ntuple(j -> j == i ? 1.0 : 0.0, Val(N)))

# Central difference — the neutral referee.
_fd(f, x; h=1e-6) = (value(f(x + h)) - value(f(x - h))) / (2h)

# K test points. `range(a, b; length=1)` is an error, hence the special case.
_pts(K) = K == 1 ? [0.7] : collect(range(0.3, 1.4; length=K))

# A non-separable objective with cross terms, for the driver-level testsets:
# every element of the gradient depends on its neighbours, so a chunking or
# workspace bug cannot hide behind a coincidentally-right answer.
function _rosen(v)
    s = zero(eltype(v))
    @inbounds for i in 1:(length(v) - 1)
        a = v[i + 1] - v[i]^2
        b = one(eltype(v)) - v[i]
        s = s + 100 * a^2 + b^2
    end
    return s
end

@testset "Piste.jl" begin

    @testset "arithmetic rules vs finite differences" begin
        fs = [
            ("+",     x -> x + 2.0),        ("-",     x -> x - 2.0),
            ("*",     x -> x * 3.0),        ("/",     x -> x / 3.0),
            ("rdiv",  x -> 3.0 / x),        ("neg",   x -> -x),
            ("^int",  x -> x^3),            ("exp",   x -> exp(x)),
            ("log",   x -> log(x)),         ("sqrt",  x -> sqrt(x)),
            ("sin",   x -> sin(x)),         ("cos",   x -> cos(x)),
            ("tan",   x -> tan(x)),         ("tanh",  x -> tanh(x)),
            ("atan",  x -> atan(x)),        ("inv",   x -> inv(x)),
            ("expm1", x -> expm1(x)),       ("log1p", x -> log1p(x)),
            ("abs",   x -> abs(x)),
            ("chain", x -> exp(sin(x) * log(x + 3.0)) / sqrt(x + 1.0)),
        ]
        for (name, f) in fs
            d = f(_d(1.3, 1, Val(4)))
            @test isapprox(partials(d, 1), _fd(f, 1.3); atol=1e-5)
            @test value(d) ≈ f(1.3)
        end
    end

    @testset "scalar logpdf derivatives" begin
        # Differentiating w.r.t. a PARAMETER, which is what a sampler does.
        cases = [
            ("Normal loc",   mu -> logpdf(Normal(mu, 1.0), 0.7),  0.3),
            ("Normal scale", s  -> logpdf(Normal(0.0, s), 0.7),   1.4),
            ("Exponential",  s  -> logpdf(Exponential(s), 0.7),   1.4),
            ("LogNormal",    mu -> logpdf(LogNormal(mu, 1.0), 0.7), 0.3),
            ("Cauchy",       x0 -> logpdf(Cauchy(x0, 5.0), 0.7),  0.3),
            ("Uniform",      b  -> logpdf(Uniform(0.0, b), 0.7),  2.0),
        ]
        for (name, f, x0) in cases
            d = f(_d(x0, 1, Val(4)))
            @test isapprox(partials(d, 1), _fd(f, x0); atol=1e-5)
        end
    end

    @testset "lgamma family (regression: SpecialFunctions._logabsgamma)" begin
        # All seven of these failed with a MethodError on `_logabsgamma` and
        # nothing else — one missing method gated the entire family.
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
            d = f(_d(x0, 1, Val(4)))
            @test isapprox(partials(d, 1), _fd(f, x0); atol=1e-5)
        end
    end

    @testset "lgamma matches SpecialFunctions" begin
        # The Lanczos series is our own, so it is checked against the real thing
        # across the range a log-density normalizing constant actually visits.
        # `atol` as well as `rtol`: loggamma(1) and loggamma(2) are exactly 0,
        # where a pure relative test is meaningless (our value there is -8.9e-16,
        # i.e. correct to machine epsilon, but rel.err against 0 is infinite).
        # Measured worst relative error away from the zeros is ~2e-15.
        for x in (0.01, 0.05, 0.5, 1.0, 1.5, 2.0, 5.0, 12.7, 50.0, 300.0, 1e4)
            @test isapprox(Piste.lgamma_lanczos(x), SpecialFunctions.loggamma(x);
                           rtol=1e-13, atol=1e-13)
        end
        # and its derivative is digamma
        for x in (0.5, 1.5, 4.0, 20.0)
            d = Piste.lgamma_lanczos(_d(x, 1, Val(2)))
            @test isapprox(partials(d, 1), SpecialFunctions.digamma(x); rtol=1e-8)
        end
    end

    @testset "truncated Cauchy (regression: two-argument atan)" begin
        # Missing `atan(y, x)` was a StackOverflowError, not a MethodError: both
        # arguments promoted and recursed forever. Then a naive atan2 rule gave
        # the right VALUE with a NaN derivative, because the truncation bound at
        # Inf makes the denominator infinite — a wrong-answer mode, not a crash.
        f = s -> logpdf(truncated(Cauchy(0.0, s), 0, Inf), 0.7)
        d = f(_d(2.0, 1, Val(4)))
        @test isfinite(partials(d, 1))
        @test isapprox(partials(d, 1), _fd(f, 2.0); atol=1e-5)

        # the infinite-argument guard itself
        a = atan(_d(1.0, 1, Val(2)), Inf)
        @test all(iszero, partials(a))
        @test isfinite(value(a))
    end

    @testset "discrete support checks (regression: isinteger)" begin
        x = _d(3.0, 1, Val(2))
        @test isinteger(x)
        @test !isinteger(_d(3.5, 1, Val(2)))
        @test floor(_d(3.7, 1, Val(2))) == 3.0
        @test ceil(_d(3.2, 1, Val(2))) == 4.0
    end

    @testset "multivariate likelihoods" begin
        f1 = mu -> logpdf(MvNormal(fill(mu, 3), I), [0.1, 0.2, 0.3])
        f2 = s  -> logpdf(MvNormal(zeros(3), Diagonal(fill(s, 3))), [0.1, 0.2, 0.3])
        f3 = mu -> loglikelihood(Normal(mu, 1.0), [0.1, 0.2, 0.3])
        for (f, x0) in ((f1, 0.3), (f2, 1.5), (f3, 0.3))
            d = f(_d(x0, 1, Val(4)))
            @test isapprox(partials(d, 1), _fd(f, x0); atol=1e-5)
        end
    end

    @testset "agrees with ForwardDiff, at every chunk width" begin
        # A matrix-vector product is the shape that matters: it goes through
        # generic matvec, whose inner loop is `muladd(A[i,j], x[j], s)`. Missing
        # the three-argument `muladd` method on Dual cost 6x there.
        rng = Xoshiro(11)
        X = randn(rng, 40, 6)
        yv = randn(rng, 40)
        f = b -> sum(abs2, yv .- X * b) + sum(exp, b) / 10

        x = randn(Xoshiro(3), 6)
        gref = ForwardDiff.gradient(f, x)
        # Every chunk width is a separately compiled path (the width is a type
        # parameter), so each is checked rather than assumed equivalent.
        for N in (1, 2, 3, 4, 8, 12, 16)
            g = zeros(6)
            gradient!(g, f, x, Val(N))
            @test g ≈ gref rtol=1e-10
        end
    end

    @testset "value_and_gradient! returns the true primal" begin
        f = x -> sum(abs2, x) + exp(x[1])
        x = [0.7, 1.3, -0.4]
        for N in (1, 2, 4, 8)
            g = zeros(3)
            v, g2 = value_and_gradient!(g, f, x, Val(N))
            @test v ≈ f(x)          # taken from the first pass, not recomputed
            @test g2 === g          # written in place
            @test g ≈ ForwardDiff.gradient(f, x) rtol=1e-10
        end
    end

    @testset "chunk width need not divide the parameter count" begin
        # The last chunk is partial; its unused lanes must not write past the end
        # or leave entries stale.
        f = x -> sum(abs2, x) + prod(x)
        for K in (1, 2, 3, 5, 7, 9)
            # `range(a, b; length=1)` errors unless the endpoints agree, so the
            # single-parameter case is built directly.
            x = K == 1 ? [0.7] : collect(range(0.3, 1.1, length=K))
            gref = ForwardDiff.gradient(f, x)
            for N in (1, 2, 4, 8)
                g = fill(NaN, K)
                gradient!(g, f, x, Val(N))
                @test g ≈ gref rtol=1e-10
                @test all(isfinite, g)
            end
        end
    end

    @testset "Float32 stays Float32" begin
        f = x -> sum(abs2, x) + exp(x[1])
        x32 = Float32[0.7, 1.3, -0.4]
        g32 = zeros(Float32, 3)
        gradient!(g32, f, x32, Val(2))
        @test eltype(g32) === Float32
        @test Float64.(g32) ≈ ForwardDiff.gradient(f, Float64.(x32)) rtol=1e-5
    end

    @testset "Dual is a well-behaved Real" begin
        x = _d(1.5, 1, Val(4))
        @test x isa Real
        @test isbitstype(Dual{4,Float64})
        @test sizeof(Dual{4,Float64}) == sizeof(Float64) * 5
        @test zero(Dual{4,Float64}) == 0
        @test one(Dual{4,Float64}) == 1
        @test value(zero(x)) == 0.0
        # comparisons act on the value, which is what makes support checks work
        @test x > 1.0
        @test x < 2.0
        @test x == _d(1.5, 2, Val(4))
        @test isfinite(x) && !isnan(x) && !isinf(x)
        # promotion
        @test promote_type(Dual{4,Float64}, Int) === Dual{4,Float64}
        @test convert(Dual{4,Float64}, 2) isa Dual{4,Float64}
    end

    @testset "no partials means no derivative" begin
        # A constant carries zero partials; propagating one must not invent a
        # derivative anywhere.
        c = Dual{4,Float64}(2.0)
        @test all(iszero, partials(c))
        @test all(iszero, partials(exp(c) * log(c) + sqrt(c)))
    end

    @testset "workspace path matches the allocating path" begin
        # The workspace is purely an allocation optimisation: it must not change
        # a single bit of the answer, at any width, including widths that do not
        # divide K. Reference is ForwardDiff, per the rule at the top of this file.
        for K in (1, 5, 13, 40)
            x = _pts(K)
            gref = ForwardDiff.gradient(_rosen, x)
            for N in (1, 4, 8, 16, 32)
                ga = similar(x); gradient!(ga, _rosen, x, Val(N))
                ws = GradientWorkspace(x, Val(N))
                gw = similar(x); gradient!(gw, _rosen, x, Val(N), ws)
                @test ga == gw
                @test gw ≈ gref
                # reusing the SAME workspace must give the same answer again --
                # a stale buffer would show up here and nowhere else
                gw2 = similar(x); gradient!(gw2, _rosen, x, Val(N), ws)
                @test gw2 == gw
            end
        end
    end

    @testset "workspace allocates nothing after warmup" begin
        # The whole point of the workspace. If this regresses, the driver has
        # started allocating again.
        x = _pts(50)
        g = similar(x)
        ws = GradientWorkspace(x, Val(16))
        gradient!(g, _rosen, x, Val(16), ws)          # warm up / compile
        @test (@allocated gradient!(g, _rosen, x, Val(16), ws)) == 0
        value_and_gradient!(g, _rosen, x, Val(16), ws)
        @test (@allocated value_and_gradient!(g, _rosen, x, Val(16), ws)) == 0
    end

    @testset "value_and_gradient! with a workspace" begin
        for K in (1, 7, 20)
            x = _pts(K)
            ws = GradientWorkspace(x, Val(8))
            g = similar(x)
            v, _ = value_and_gradient!(g, _rosen, x, Val(8), ws)
            @test v == _rosen(x)
            @test g ≈ ForwardDiff.gradient(_rosen, x)
        end
    end

    @testset "a workspace grows for a larger problem" begin
        # Reusing a workspace against a longer x must resize rather than read
        # past the end -- silently wrong gradients are the failure mode here.
        x_small = _pts(5)
        ws = GradientWorkspace(x_small, Val(4))
        g_small = similar(x_small)
        gradient!(g_small, _rosen, x_small, Val(4), ws)
        @test g_small ≈ ForwardDiff.gradient(_rosen, x_small)

        x_big = _pts(30)
        g_big = similar(x_big)
        gradient!(g_big, _rosen, x_big, Val(4), ws)
        @test g_big ≈ ForwardDiff.gradient(_rosen, x_big)
    end

    @testset "literal_pow (regression: was routing through pow_body)" begin
        # `x^2` lowers to literal_pow. The rule fired but computed its value
        # with a runtime `^`, which cost ~2x on the whole gradient. These check
        # the fast paths agree with the general one and with ForwardDiff.
        for p in 0:6, xv in (0.4, 1.0, 1.7, 3.2)
            f = v -> Base.literal_pow(^, v[1], Val(p))
            @test partials(f([_d(xv, 1, Val(1))]))[1] ≈
                  ForwardDiff.gradient(f, [xv])[1]
        end
        # x^1 must be the identity, x^0 a true constant
        d = _d(1.7, 1, Val(4))
        @test Base.literal_pow(^, d, Val(1)) === d
        @test all(iszero, partials(Base.literal_pow(^, d, Val(0))))
        @test value(Base.literal_pow(^, d, Val(0))) == 1.0
    end

    @testset "pickchunk returns a usable width" begin
        # It must always give a Val, and gradients through it must be right.
        for K in (1, 2, 5, 10, 20, 50, 100, 300)
            x = _pts(K)
            g = similar(x)
            gradient!(g, _rosen, x, pickchunk(K))
            @test g ≈ ForwardDiff.gradient(_rosen, x)
        end
    end

    include("reverse.jl")
end
