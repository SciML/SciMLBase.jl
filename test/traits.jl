using SciMLBase, Test
using ModelingToolkit, OrdinaryDiffEq, DataFrames
using ModelingToolkit: t_nounits as t, D_nounits as D

@test SciMLBase.Tables.isrowtable(ODESolution)
@test SciMLBase.Tables.isrowtable(RODESolution)
@test SciMLBase.Tables.isrowtable(DAESolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.NonlinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.LinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.QuadratureSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.OptimizationSolution)

@variables x(t)=1
eqs = [D(x) ~ -x]
@named sys = ODESystem(eqs, t)
sys = complete(sys)
prob = ODEProblem(sys)
sol = solve(prob, Tsit5(), tspan = (0.0, 1.0))
df = DataFrame(sol)
@test size(df) == (length(sol.u), 2)
@test df.timestamp == sol.t
@test df.x == sol[x]
