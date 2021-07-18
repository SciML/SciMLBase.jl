using Documenter, SciMLBase

makedocs(
    sitename="SciMLBase.jl",
    authors="Chris Rackauckas",
    modules=[SciMLBase],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://scimlbase.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Fundamentals" => Any[
            "fundamentals/Problems.md",
            "fundamentals/PDE.md",
            "fundamentals/SciMLFunctions.md",
            "fundamentals/Differentiation.md",
            "fundamentals/FAQ.md"
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/SciMLBase.jl.git";
   push_preview = true
)
