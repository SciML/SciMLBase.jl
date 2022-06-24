using Documenter, SciMLBase, ModelingToolkit

include("pages.jl")

makedocs(sitename = "SciMLBase.jl",
         authors = "Chris Rackauckas",
         modules = [SciMLBase, ModelingToolkit],
         clean = true, doctest = false,
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://scimlbase.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/SciMLBase.jl.git";
           push_preview = true)
