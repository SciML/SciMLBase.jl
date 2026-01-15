using OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D
using Test

# DifferentiationInterface with version-dependent backends
using DifferentiationInterface
using ADTypes
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
if VERSION < v"1.12"
    using Zygote: Zygote
    using Enzyme: Enzyme
end

function backend_name(backend::ADTypes.AbstractADType)
    return string(typeof(backend).name.name)
end

@variables x(t) o(t)
function lotka_volterra(; name = name)
    unknowns = @variables x(t) = 1.0 y(t) = 1.0 o(t)
    params = @parameters p1 = 1.5 p2 = 1.0 p3 = 3.0 p4 = 1.0
    eqs = [
        D(x) ~ p1 * x - p2 * x * y,
        D(y) ~ -p3 * y + p4 * x * y,
        o ~ x * y,
    ]
    return System(eqs, t, unknowns, params; name = name)
end

@named lotka_volterra_sys = lotka_volterra()
lotka_volterra_sys = mtkcompile(lotka_volterra_sys, split = false)
prob = ODEProblem(lotka_volterra_sys, [], (0.0, 10.0))
sol = solve(prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6)
setter = setsym_oop(prob, [unknowns(lotka_volterra_sys); parameters(lotka_volterra_sys)])
u0, p = setter(prob, [1.0, 1.0, 1.5, 1.0, 1.0, 1.0])

# These tests use Zygote-specific sensealg (ZygoteVJP), so only run on Julia < 1.12
if VERSION < v"1.12"
    # Define loss functions that take single arguments for DifferentiationInterface
    function make_sum_of_solution_u0(p_fixed)
        return function (u0)
            _prob = remake(prob, u0 = u0, p = p_fixed)
            return sum(
                solve(
                    _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                    sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
                )
            )
        end
    end

    function make_sum_of_solution_p(u0_fixed)
        return function (p)
            _prob = remake(prob, u0 = u0_fixed, p = p)
            return sum(
                solve(
                    _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                    sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
                )
            )
        end
    end

    @testset "Basic remake autodiff" begin
        backend = AutoZygote()
        @testset "$(backend_name(backend))" begin
            du01 = DifferentiationInterface.gradient(make_sum_of_solution_u0(p), backend, u0)
            dp1 = DifferentiationInterface.gradient(make_sum_of_solution_p(u0), backend, p)
            @test du01 !== nothing
            @test dp1 !== nothing
        end
    end

    # These tests depend on a ZygoteRule in a package extension
    function make_symbolic_indexing_u0(p_fixed)
        return function (u0)
            _prob = remake(prob, u0 = u0, p = p_fixed)
            soln = solve(
                _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
            )
            return sum(soln[x])
        end
    end

    function make_symbolic_indexing_p(u0_fixed)
        return function (p)
            _prob = remake(prob, u0 = u0_fixed, p = p)
            soln = solve(
                _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
            )
            return sum(soln[x])
        end
    end

    @testset "Symbolic indexing autodiff" begin
        backend = AutoZygote()
        @testset "$(backend_name(backend))" begin
            du01 = DifferentiationInterface.gradient(make_symbolic_indexing_u0(p), backend, u0)
            dp1 = DifferentiationInterface.gradient(make_symbolic_indexing_p(u0), backend, p)
            @test du01 !== nothing
            @test dp1 !== nothing
        end
    end

    function make_symbolic_indexing_observed_u0(p_fixed)
        return function (u0)
            _prob = remake(prob, u0 = u0, p = p_fixed)
            soln = solve(
                _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
            )
            return sum(soln[o, i] for i in 1:length(soln))
        end
    end

    function make_symbolic_indexing_observed_p(u0_fixed)
        return function (p)
            _prob = remake(prob, u0 = u0_fixed, p = p)
            soln = solve(
                _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
            )
            return sum(soln[o, i] for i in 1:length(soln))
        end
    end

    @testset "Symbolic indexing observed autodiff" begin
        backend = AutoZygote()
        @testset "$(backend_name(backend))" begin
            du01 = DifferentiationInterface.gradient(
                make_symbolic_indexing_observed_u0(p), backend, u0)
            dp1 = DifferentiationInterface.gradient(
                make_symbolic_indexing_observed_p(u0), backend, p)
            @test du01 !== nothing
            @test dp1 !== nothing
        end
    end
end
