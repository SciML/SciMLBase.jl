using Documenter, SciMLBase, ModelingToolkit

makedocs(
    sitename="SciMLBase.jl",
    authors="Chris Rackauckas",
    modules=[SciMLBase,ModelingToolkit],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://scimlbase.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Interfaces" => Any[
            "interfaces/Problems.md",
            "interfaces/SciMLFunctions.md",
            "interfaces/Algorithms.md",
            "interfaces/Solutions.md",
            "interfaces/Common_Keywords.md",
            "interfaces/Differentiation.md",
            "interfaces/PDE.md",
        ],
        "Fundamentals" => Any[
            "fundamentals/FAQ.md"
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/SciMLBase.jl.git";
   push_preview = true
)
