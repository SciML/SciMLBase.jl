using Pkg
using SafeTestsets
using Test
using SciMLBase

@test isempty(detect_ambiguities(SciMLBase))

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin

if GROUP == "Core" || GROUP == "All"
    @time @safetestset "Display" begin include("display.jl") end
    @time @safetestset "Existence functions" begin include("existence_functions.jl") end
    @time @safetestset "Integrator interface" begin include("integrator_tests.jl") end
    @time @safetestset "Ensemble functionality" begin include("ensemble_tests.jl") end
    @time @safetestset "DiffEqOperator tests" begin include("diffeqoperator.jl") end
end

if !is_APPVEYOR && GROUP == "Downstream"
    activate_downstream_env()
    @time @safetestset "Ensembles of Zero Length Solutions" begin include("downstream/ensemble_zero_length.jl") end
end
end
