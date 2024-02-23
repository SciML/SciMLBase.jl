using SciMLBase, Test
using ModelingToolkit, OrdinaryDiffEq, DataFrames

@test SciMLBase.Tables.isrowtable(ODESolution)
@test SciMLBase.Tables.isrowtable(RODESolution)
@test SciMLBase.Tables.isrowtable(DAESolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.NonlinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.LinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.QuadratureSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.OptimizationSolution)

@variables t x(t)=1
D = Differential(t)
eqs = [D(x) ~ -x]
@named sys = ODESystem(eqs)
prob = ODEProblem(sys)
sol = solve(prob, Tsit5(), tspan = (0.0, 1.0))
df = DataFrame(sol)
@test size(df) == (length(sol.u), 2)
@test df.timestamp == sol.t
@test df.x == sol[x]
