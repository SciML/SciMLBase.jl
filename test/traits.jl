using SciMLBase, Test

@test SciMLBase.Tables.isrowtable(ODESolution)
@test SciMLBase.Tables.isrowtable(RODESolution)
@test SciMLBase.Tables.isrowtable(DAESolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.NonlinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.LinearSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.QuadratureSolution)
@test !SciMLBase.Tables.isrowtable(SciMLBase.OptimizationSolution)
