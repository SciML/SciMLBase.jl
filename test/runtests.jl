using Pkg
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = (Sys.iswindows() && haskey(ENV, "APPVEYOR"))

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_python_env()
    Pkg.activate("python")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "QA" || GROUP == "All"
        @time @safetestset "Aqua" begin
            include("aqua.jl")
        end
    end
    if GROUP == "Core" || GROUP == "All"
        @time @safetestset "Display" begin
            include("display.jl")
        end
        @time @safetestset "Existence functions" begin
            include("existence_functions.jl")
        end
        @time @safetestset "Function Building Error Messages" begin
            include("function_building_error_messages.jl")
        end
        @time @safetestset "Integrator interface" begin
            include("integrator_tests.jl")
        end
        @time @safetestset "Ensemble functionality" begin
            include("ensemble_tests.jl")
        end
        @time @safetestset "Solution interface" begin
            include("solution_interface.jl")
        end
        @time @safetestset "DE function conversion" begin
            include("convert_tests.jl")
        end
        @time @safetestset "Performance warnings" begin
            include("performance_warnings.jl")
        end
        @time @safetestset "Problem building tests" begin
            include("problem_building_test.jl")
        end
        @time @safetestset "Serialization tests" begin
            include("serialization_tests.jl")
        end
        @time @safetestset "Clocks" begin
            include("clock.jl")
        end
    end

    if !is_APPVEYOR &&
       (GROUP == "Core" || GROUP == "All" || GROUP == "SymbolicIndexingInterface")
        @time @safetestset "Remake" begin
            include("remake_tests.jl")
        end
    end

    if !is_APPVEYOR && GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "Ensembles of Zero Length Solutions" begin
            include("downstream/ensemble_zero_length.jl")
        end
        @time @safetestset "Timing first batch when solving Ensembles" begin
            include("downstream/ensemble_first_batch.jl")
        end
        @time @safetestset "solving Ensembles with multiple problems" begin
            include("downstream/ensemble_multi_prob.jl")
        end
        @time @safetestset "Ensemble solution statistics" begin
            include("downstream/ensemble_stats.jl")
        end
        @time @safetestset "Ensemble Optimization and Nonlinear problems" begin
            include("downstream/ensemble_nondes.jl")
        end
        @time @safetestset "Ensemble with DifferentialEquations automatic algorithm selection" begin
            include("downstream/ensemble_diffeq.jl")
        end
        @time @safetestset "Solution Indexing" begin
            include("downstream/solution_interface.jl")
        end
        @time @safetestset "Unitful interpolations" begin
            include("downstream/unitful_interpolations.jl")
        end
        @time @safetestset "Integer idxs" begin
            include("downstream/integer_idxs.jl")
        end
        if VERSION < v"1.12" # Zygote is incredibly broken on 1.12
            @time @safetestset "Autodiff Remake" begin
                include("downstream/remake_autodiff.jl")
            end
        end
        @time @safetestset "Partial Functions" begin
            include("downstream/partial_functions.jl")
        end
        @time @safetestset "Autodiff Observable Functions" begin
            include("downstream/observables_autodiff.jl")
        end
        @time @safetestset "ODE Solution Stripping" begin
            include("downstream/ode_stripping.jl")
        end
        @time @safetestset "Tables interface with MTK" begin
            include("downstream/tables.jl")
        end
        @time @safetestset "Initialization" begin
            include("downstream/initialization.jl")
        end
        @time @safetestset "Table Traits" begin
            include("downstream/traits.jl")
        end
        @time @safetestset "SplitODEProblem cache" begin
            include("downstream/splitodeproblem_cache.jl")
        end
    end

    if !is_APPVEYOR && GROUP == "SymbolicIndexingInterface"
        if GROUP != "Downstream"
            activate_downstream_env()
        end
        @time @safetestset "Symbol and integer based indexing of interpolated solutions" begin
            include("downstream/comprehensive_indexing.jl")
        end
        @time @safetestset "Symbol and integer based indexing of integrators" begin
            include("downstream/integrator_indexing.jl")
        end
        @time @safetestset "Problem Indexing" begin
            include("downstream/problem_interface.jl")
        end
        if VERSION < v"1.12" # Zygote is incredibly broken on 1.12
            @time @safetestset "Adjoints" begin
                include("downstream/adjoints.jl")
            end
        end
        @time @safetestset "ModelingToolkit Remake" begin
            include("downstream/modelingtoolkit_remake.jl")
        end
    end

    if !is_APPVEYOR && GROUP == "Python"
        activate_python_env()
        @time @safetestset "PythonCall" begin
            include("python/pythoncall.jl")
        end
    end
end
