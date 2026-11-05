# Render the README benchmark tables from `sweep.json`.
#
# The timings come from PracticalBayes' benchmark sweep
# (PracticalBayes.jl/benchmarks/sweep.jl), which differentiates the same
# PracticalBayes log-density with every backend. To refresh them, rerun that
# sweep, copy its `results/sweep.json` over the one here, and run
#
#     julia --project=benchmark benchmark/make_readme_table.jl
#
# which rewrites the block between the BENCH markers in README.md.

import JSON3
using Printf: @sprintf

const README_PATH = normpath(joinpath(@__DIR__, "..", "README.md"))
const SWEEP_PATH = joinpath(@__DIR__, "sweep.json")
const MARKER_START = "<!-- BENCH:START -->"
const MARKER_END = "<!-- BENCH:END -->"

const BACKENDS = [("piste_fwd", "Piste fwd"), ("piste_rev", "Piste rev"),
                  ("forwarddiff", "ForwardDiff"), ("mooncake", "Mooncake"), ("enzyme", "Enzyme")]

# Three significant figures, in whichever unit keeps the number short.
function fmt_time(ns::Float64)
    v, unit = ns < 1e3 ? (ns, "ns") : ns < 1e6 ? (ns / 1e3, "µs") :
              ns < 1e9 ? (ns / 1e6, "ms") : (ns / 1e9, "s")
    s = v < 10 ? @sprintf("%.2f", v) : v < 100 ? @sprintf("%.1f", v) : @sprintf("%.0f", v)
    return "$s $unit"
end

nsof(x) = x === nothing ? NaN : Float64(x)

function precision_table(lookup, prec, likelihoods, corners)
    io = IOBuffer()
    println(io, "| Model | N | P | ", join(last.(BACKENDS), " | "), " |")
    println(io, "|---|---:|---:|", repeat("---:|", length(BACKENDS)))
    for lik in likelihoods, (n, k) in corners
        row = get(lookup, (prec, lik, n, k), nothing)
        times = row === nothing ? fill(NaN, length(BACKENDS)) :
                [nsof(row[b]["pb_ns"]) for (b, _) in BACKENDS]
        finite = filter(isfinite, times)
        best = isempty(finite) ? NaN : minimum(finite)
        cells = map(times) do t
            !isfinite(t) ? "n/a" : t == best ? "**$(fmt_time(t))**" : fmt_time(t)
        end
        println(io, "| `", lik, "` | ", n, " | ", k, " | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

function environment_line(meta)
    env = meta["environment"]
    pk = env["packages"]
    pkgs = join(["$p $(pk[Symbol(p)])" for p in ("Piste", "ForwardDiff", "Mooncake", "Enzyme")
                 if haskey(pk, Symbol(p))], ", ")
    return "Measured with Julia $(env["julia"]) on one Julia thread and " *
           "$(env["blas_threads"]) BLAS thread of an exclusive $(env["cpu"]) node: $pkgs. " *
           "Each figure is the median of up to $(meta["bench_samples"]) samples or " *
           "$(meta["bench_seconds"]) s."
end

function main()
    payload = JSON3.read(read(SWEEP_PATH, String))
    meta = payload["meta"]
    lookup = Dict((String(r["precision"]), String(r["likelihood"]), Int(r["N"]), Int(r["NPARAMS"])) => r
                  for r in payload["rows"])
    nvals = Int.(meta["n_values"])
    kvals = Int.(meta["nparams_values"])
    likelihoods = String.(meta["likelihoods"])
    corners = [(first(nvals), first(kvals)), (first(nvals), last(kvals)),
               (last(nvals), first(kvals)), (last(nvals), last(kvals))]

    block = join([
        environment_line(meta),
        "",
        "### Float64",
        "",
        precision_table(lookup, "Float64", likelihoods, corners),
        "### Float32",
        "",
        precision_table(lookup, "Float32", likelihoods, corners),
    ], "\n")

    text = read(README_PATH, String)
    i1 = findfirst(MARKER_START, text)
    i2 = findfirst(MARKER_END, text)
    (i1 === nothing || i2 === nothing) && error("README.md is missing the BENCH markers")
    write(README_PATH, text[1:first(i1)-1] * MARKER_START * "\n" * block * MARKER_END *
                       text[last(i2)+1:end])
    println("Updated README benchmark tables from ", SWEEP_PATH)
end

main()
