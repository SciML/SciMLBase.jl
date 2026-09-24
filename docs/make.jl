using Pkg

Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.instantiate()

using Documenter, SciMLBase

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(;
    sitename = "SciMLBase.jl",
    authors = "Chris Rackauckas",
    modules = [SciMLBase],
    clean = true, doctest = true, linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SciMLBase/stable"
    ),
    linkcheck_ignore = [
        "https://www.sciencedirect.com/science/article/abs/pii/S0045782523007156",
    ],
    pages
)

deploydocs(
    repo = "github.com/SciML/SciMLBase.jl.git";
    push_preview = true
)
