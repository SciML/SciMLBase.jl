module SciMLBaseStaticArraysExt

using SciMLBase: AbstractNoTimeSolution
using StaticArrays: StaticMatrix

# Disambiguate against StaticArrays' `*(::StaticMatrix, ::AbstractVector)` and
# SciMLBase's `*(::AbstractMatrix, ::AbstractNoTimeSolution)`.
Base.:*(A::StaticMatrix, sol::AbstractNoTimeSolution{<:Any, 1}) = A * sol.u

end
