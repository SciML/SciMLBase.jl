using ModelingToolkit, OrdinaryDiffEq, SymbolicIndexingInterface, Zygote, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs, t)
@named lorenz2 = ODESystem(eqs, t)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@mtkbuild sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])

u0 = [lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0,
    a => 2.0]

p = [lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, Rodas4())

gs_sym, = Zygote.gradient(sol) do sol
    sum(sol[lorenz1.x])
end
idx_sym = SymbolicIndexingInterface.variable_index(sys, lorenz1.x)
true_grad_sym = zeros(length(ModelingToolkit.unknowns(sys)))
true_grad_sym[idx_sym] = 1.0

@test all(map(x -> x == true_grad_sym, gs_sym))

gs_vec, = Zygote.gradient(sol) do sol
    sum(sum.(sol[[lorenz1.x, lorenz2.x]]))
end
idx_vecsym = SymbolicIndexingInterface.variable_index.(Ref(sys), [lorenz1.x, lorenz2.x])
true_grad_vecsym = zeros(length(ModelingToolkit.unknowns(sys)))
true_grad_vecsym[idx_vecsym] .= 1.0

@test all(map(x -> x == true_grad_vecsym, gs_vec))

gs_tup, = Zygote.gradient(sol) do sol
    sum(sum.(collect.(sol[(lorenz1.x, lorenz2.x)])))
end
idx_tupsym = SymbolicIndexingInterface.variable_index.(Ref(sys), [lorenz1.x, lorenz2.x])
true_grad_tupsym = zeros(length(ModelingToolkit.unknowns(sys)))
true_grad_tupsym[idx_tupsym] .= 1.0

@test all(map(x -> x == true_grad_tupsym, gs_tup))

gs_ts, = Zygote.gradient(sol) do sol
    sum(sum.(sol[[lorenz1.x, lorenz2.x], :]))
end

@test all(map(x -> x == true_grad_vecsym, gs_ts))

# BatchedInterface AD
@variables x(t)=1.0 y(t)=1.0 z(t)=1.0 w(t)=1.0
@named sys1 = ODESystem([D(x) ~ x + y, D(y) ~ y * z, D(z) ~ z * t * x], t)
sys1 = complete(sys1)
prob1 = ODEProblem(sys1, [], (0.0, 10.0))
@named sys2 = ODESystem([D(x) ~ x + w, D(y) ~ w * t, D(w) ~ x + y + w], t)
sys2 = complete(sys2)
prob2 = ODEProblem(sys2, [], (0.0, 10.0))

bi = BatchedInterface((sys1, [x, y, z]), (sys2, [x, y, w]))
getter = getu(bi)

p1grad, p2grad = Zygote.gradient(prob1, prob2) do prob1, prob2
    sum(getter(prob1, prob2))
end

@test p1grad.u0 ≈ ones(3)
testp2grad = zeros(3)
testp2grad[variable_index(prob2, w)] = 1.0
@test p2grad.u0 ≈ testp2grad
