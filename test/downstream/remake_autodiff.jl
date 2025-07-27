using OrdinaryDiffEq, ModelingToolkit, Zygote, SciMLSensitivity
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) o(t)
function lotka_volterra(; name = name)
    unknowns = @variables x(t)=1.0 y(t)=1.0 o(t)
    params = @parameters p1=1.5 p2=1.0 p3=3.0 p4=1.0
    eqs = [
        D(x) ~ p1 * x - p2 * x * y,
        D(y) ~ -p3 * y + p4 * x * y,
        o ~ x * y
    ]
    return System(eqs, t, unknowns, params; name = name)
end

@named lotka_volterra_sys = lotka_volterra()
lotka_volterra_sys = mtkcompile(lotka_volterra_sys, split = false)
prob = ODEProblem(lotka_volterra_sys, [], (0.0, 10.0))
sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
setter = setsym_oop(prob, [unknowns(lotka_volterra_sys); parameters(lotka_volterra_sys)])
u0, p = setter(prob, [1.0, 1.0, 1.5, 1.0, 1.0, 1.0])

function sum_of_solution(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    sum(solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())))
end

du01, dp1 = Zygote.gradient(sum_of_solution, u0, p)

# These tests depend on a ZygoteRule in a package extension
function symbolic_indexing(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    soln = solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
    sum(soln[x])
end

du01, dp1 = Zygote.gradient(symbolic_indexing, u0, p)

function symbolic_indexing_observed(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    soln = solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
    sum(soln[o, i] for i in 1:length(soln))
end

du01, dp1 = Zygote.gradient(symbolic_indexing_observed, u0, p)
