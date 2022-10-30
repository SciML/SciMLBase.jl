using OrdinaryDiffEq, ModelingToolkit, Zygote, SciMLSensitivity

@variables t
D = Differential(t)
function lotka_volterra(; name = name)
    states = @variables x(t)=1.0 y(t)=1.0
    params = @parameters p1=1.5 p2=1.0 p3=3.0 p4=1.0
    eqs = [
        D(x) ~ p1 * x - p2 * x * y,
        D(y) ~ -p3 * y + p4 * x * y,
    ]
    return ODESystem(eqs, t, states, params; name = name)
end

@named lotka_volterra_sys = lotka_volterra()
prob = ODEProblem(lotka_volterra_sys, [], (0.0, 10.0), [])
sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)

function sum_of_solution(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    sum(solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1,
              sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())))
end

u0 = [1.0 1.0]
p = [1.5 1.0 1.0 1.0]
du01, dp1 = Zygote.gradient(sum_of_solution, u0, p)
