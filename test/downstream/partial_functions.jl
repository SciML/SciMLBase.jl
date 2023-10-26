using SciMLBase, PartialFunctions, Test

partial_add = @$ +(_, _)
partial_add2 = @$ +(_, 2)
partial_add3 = @$ +(3, _)

@test only(SciMLBase.numargs(partial_add)) == 2
@test only(SciMLBase.numargs(partial_add2)) == 1
@test only(SciMLBase.numargs(partial_add3)) == 1

f1(args...) = args

@test only(SciMLBase.numargs(@$ f1(_, _, _))) == 3
@test only(SciMLBase.numargs(@$ f1(_, _, _, _, 4))) == 4
@test only(SciMLBase.numargs(@$ f1(_, _, _, _, _))) == 5
