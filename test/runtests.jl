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
        @time @safetestset "Display" begin include("display.jl") end
        @time @safetestset "Existence functions" begin include("existence_functions.jl") end
        @time @safetestset "Function Building Error Messages" begin include("function_building_error_messages.jl") end
        @time @safetestset "Solver Missing Error Messages" begin include("solver_missing_error_messages.jl") end
        @time @safetestset "Integrator interface" begin include("integrator_tests.jl") end
        @time @safetestset "Table Traits" begin include("traits.jl") end
        @time @safetestset "Ensemble functionality" begin include("ensemble_tests.jl") end
        @time @safetestset "DiffEqOperator tests" begin include("diffeqoperator.jl") end
        @time @safetestset "Solution interface" begin include("solution_interface.jl") end
        @time @safetestset "DE functon conversion" begin include("convert_tests.jl") end
    end

    if !is_APPVEYOR && GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "Ensembles of Zero Length Solutions" begin include("downstream/ensemble_zero_length.jl") end
        @time @safetestset "Timing first batch when solving Ensembles" begin include("downstream/ensemble_first_batch.jl") end
        @time @safetestset "Symbol and integer based indexing of interpolated solutions" begin include("downstream/symbol_indexing.jl") end
        @time @safetestset "Unitful interoplations" begin include("downstream/unitful_interpolations.jl") end
        @time @safetestset "Integer idxs" begin include("downstream/integer_idxs.jl") end
    end
end
