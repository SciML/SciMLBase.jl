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

function make_sum_of_solution_u0(prob)
    return function (u0)
        # If `p` is passed to `remake`, MTK won't copy `u0` to initials
        # and it will be reset to the previous value
        _prob = remake(prob, u0 = u0)
        return sum(
            solve(
                _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
                sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
            )
        )
    end
end

function make_sum_of_solution_p(prob, u0_fixed)
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

function make_symbolic_indexing_u0(prob)
    return function (u0)
        _prob = remake(prob, u0 = u0)
        soln = solve(
            _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
        )
        return sum(soln[x])
    end
end

function make_symbolic_indexing_p(prob, u0_fixed)
    return function (p)
        _prob = remake(prob, u0 = u0_fixed, p = p)
        soln = solve(
            _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
        )
        return sum(soln[x])
    end
end

function make_symbolic_indexing_observed_u0(prob)
    return function (u0)
        _prob = remake(prob, u0 = u0)
        soln = solve(
            _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
        )
        return sum(soln[o, i] for i in 1:length(soln))
    end
end

function make_symbolic_indexing_observed_p(prob, u0_fixed)
    return function (p)
        _prob = remake(prob, u0 = u0_fixed, p = p)
        soln = solve(
            _prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6, saveat = 0.1,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
        )
        return sum(soln[o, i] for i in 1:length(soln))
    end
end

if VERSION < v"1.12"
    backend = AutoZygote()
    derivatives = []
    @testset "`split = $split`" for split in (false, true)
        cur_ders = Dict()
        push!(derivatives, cur_ders)

        # `split = true` runs into https://github.com/JuliaDiff/ChainRules.jl/issues/830
        # for Zygote. It can be re-enabled for other backends if they are tested here.
        if split
            continue
            @test_broken false
        end
        sys = mtkcompile(lotka_volterra_sys; split)
        prob = ODEProblem(sys, [], (0.0, 10.0))
        sol = solve(prob, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-6)
        setter = setsym_oop(prob, [unknowns(sys); parameters(sys)])
        u0, p = setter(prob, [1.0, 1.0, 1.5, 1.0, 1.0, 1.0])

        du01 = DifferentiationInterface.gradient(
            make_sum_of_solution_u0(prob), backend, u0
        )
        @test du01 !== nothing
        cur_ders["sum_of_solution_u0"] = du01
        dp1 = DifferentiationInterface.gradient(
            make_sum_of_solution_p(prob, u0), backend, p
        )
        @test dp1 !== nothing
        cur_ders["sum_of_solution_p"] = dp1

        du01 = DifferentiationInterface.gradient(
            make_symbolic_indexing_u0(prob), backend, u0
        )
        @test du01 !== nothing
        cur_ders["symbolic_indexing_u0"] = du01
        dp1 = DifferentiationInterface.gradient(
            make_symbolic_indexing_p(prob, u0), backend, p
        )
        @test dp1 !== nothing
        cur_ders["symbolic_indexing_p"] = dp1

        du01 = DifferentiationInterface.gradient(
            make_symbolic_indexing_observed_u0(prob), backend, u0
        )
        @test du01 !== nothing
        cur_ders["symbolic_indexing_observed_u0"] = du01
        dp1 = DifferentiationInterface.gradient(
            make_symbolic_indexing_observed_p(prob, u0), backend, p
        )
        @test dp1 !== nothing
        cur_ders["symbolic_indexing_observed_p"] = dp1
    end

    @testset "Consistent gradients" begin
        nosplit_ders, split_ders = derivatives
        ks = intersect(keys(nosplit_ders), keys(split_ders))
        @test length(ks) == 6 broken = true
        @testset "$k" for k in ks
            @test nosplit_ders[k] ≈ split_ders[k]
        end
    end
end
