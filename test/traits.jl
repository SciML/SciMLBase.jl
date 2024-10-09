using SciMLBase, Tables, Test
using OrdinaryDiffEq, DataFrames, SymbolicIndexingInterface

@test Tables.isrowtable(ODESolution)
@test Tables.isrowtable(RODESolution)
@test Tables.isrowtable(DAESolution)
@test !Tables.isrowtable(SciMLBase.NonlinearSolution)
@test !Tables.isrowtable(SciMLBase.LinearSolution)
@test !Tables.isrowtable(SciMLBase.QuadratureSolution)
@test !Tables.isrowtable(SciMLBase.OptimizationSolution)

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
