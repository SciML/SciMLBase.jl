module SciMLBaseStaticArraysExt

using SciMLBase: AbstractNoTimeSolution
using StaticArrays: StaticMatrix

# Disambiguate `*(::StaticMatrix, ::AbstractVecOrMat)` against SciMLBase's
# `*(::AbstractMatrix, ::AbstractNoTimeSolution)`. Forward to `A * sol.u` so the
# StaticArrays fast path is preserved.
Base.:*(A::StaticMatrix, sol::AbstractNoTimeSolution) = A * sol.u

end
