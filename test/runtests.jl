using Pkg
using SafeTestsets
using Test
using SciMLBase

# https://github.com/JuliaArrays/FillArrays.jl/pull/163
@test_broken isempty(detect_ambiguities(SciMLBase))

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = (Sys.iswindows() && haskey(ENV, "APPVEYOR"))

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "Core" || GROUP == "All"
        @time @safetestset "Display" include("display.jl")
        @time @safetestset "Existence functions" include("existence_functions.jl")
        @time @safetestset "Function Building Error Messages" include("function_building_error_messages.jl")
        @time @safetestset "Solver Missing Error Messages" include("solver_missing_error_messages.jl")
        @time @safetestset "Integrator interface" include("integrator_tests.jl")
        @time @safetestset "Table Traits" include("traits.jl")
        @time @safetestset "Ensemble functionality" include("ensemble_tests.jl")
        @time @safetestset "DiffEqOperator tests" include("diffeqoperator.jl")
        @time @safetestset "Solution interface" include("solution_interface.jl")
        @time @safetestset "DE function conversion" include("convert_tests.jl")
    end

    if !is_APPVEYOR && GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "Ensembles of Zero Length Solutions" include("downstream/ensemble_zero_length.jl")
        @time @safetestset "Timing first batch when solving Ensembles" include("downstream/ensemble_first_batch.jl")
        @time @safetestset "Symbol and integer based indexing of interpolated solutions" include("downstream/symbol_indexing.jl")
        if VERSION >= v"1.8"
            @time @safetestset "Symbol and integer based indexing of integrators" include("downstream/integrator_indexing.jl")
            @time @safetestset "Problem Indexing" include("downstream/problem_interface.jl")
            @time @safetestset "Solution Indexing" include("downstream/solution_interface.jl")
        end
        @time @safetestset "Unitful interpolations" include("downstream/unitful_interpolations.jl")
        @time @safetestset "Integer idxs" include("downstream/integer_idxs.jl")
        @time @safetestset "Autodiff Remake" include("downstream/remake_autodiff.jl")
    end
end
