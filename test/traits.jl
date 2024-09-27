using SciMLBase, Test
using OrdinaryDiffEq, DataFrames, SymbolicIndexingInterface

@test SciMLBase.Tables.isrowtable(ODESolution)
@test SciMLBase.Tables.isrowtable(RODESolution)
@test SciMLBase.Tables.isrowtable(DAESolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.NonlinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.LinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.QuadratureSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.OptimizationSolution)

function rhs(u, p, t)
    return -u
end
sys = SymbolCache([:x], Symbol[], :t)
prob = ODEProblem(ODEFunction(rhs; sys), [1.0], (0.0, 1.0))
sol = solve(prob, Tsit5())
df = DataFrame(sol)
@test size(df) == (length(sol.u), 2)
@test df.timestamp == sol.t
@test df.x == sol[:x]
