using Documenter, Piste

makedocs(
    sitename = "Piste.jl",
    modules = [Piste],
    pages = [
        "Home" => "index.md",
        "Trimmed binaries" => "trimming.md",
        "API" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(repo = "github.com/EvoArt/Piste.jl.git")
