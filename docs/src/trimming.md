# Trimmed binaries

A CLI that prints the gradient of `x^3 + 2x` at the value you give it.

`src/GradCLI.jl`:

```julia
module GradCLI

using Piste: rev_gradient!

f(x) = x[1]^3 + 2 * x[1]

function (@main)(args::Vector{String})::Cint
    x = [parse(Float64, args[1])]
    g = zeros(1)
    rev_gradient!(g, f, x)
    println(Core.stdout, g[1])
    return 0
end

end
```

`Project.toml` needs only `Piste`. Build with Julia 1.12+:

```
julia --project=juliac-env -e 'using JuliaC; JuliaC.main(ARGS)' -- \
    --output-exe gradcli --bundle build --trim=safe --experimental ./GradCLI
```

```
$ ./build/bin/gradcli.exe 2.0
14.0
```

2.4 MB, 0 verifier errors, no Julia install needed at the target.

## Two rules

Pass a **literal** `Val(N)` to `gradient!`, never `pickchunk(K)` — the trimmer
needs the chunk width statically known.

Always run the binary once with `JULIA_LOAD_CODEGEN_LIB=0` and diff against the
JIT. A clean verifier pass proves nothing on its own: ForwardDiff builds with
zero errors and then dies at runtime.
