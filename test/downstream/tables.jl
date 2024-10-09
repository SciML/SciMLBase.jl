using ModelingToolkit, OrdinaryDiffEq, Tables, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) y(t)
@parameters p q
@mtkbuild sys = ODESystem(
    [D(x) ~ p * x + q * y, 0 ~ y^2 - 2x * y - 4], t, guesses = [y => 1.0])
prob = ODEProblem(sys, [x => 1.0], (0.0, 5.0), [p => 1.0, q => 2.0])
sol = solve(prob, Rodas5P())

@test sort(collect(Tables.columnnames(Tables.columns(sol)))) ==
      [:timestamp, Symbol("x(t)"), Symbol("y(t)")]
