using Documenter, SciMLBase, ModelingToolkit

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "SciMLBase.jl",
    authors = "Chris Rackauckas",
    modules = [SciMLBase, ModelingToolkit],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SciMLBase/stable"),
    pages = pages)

deploydocs(repo = "github.com/SciML/SciMLBase.jl.git";
    push_preview = true)
