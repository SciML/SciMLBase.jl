module SciMLBaseReactantCoreExt

using SciMLBase: SciMLBase, ODEFunction, AutoSpecialize, FullSpecialize
using ReactantCore: ReactantCore

SciMLBase.specialization(::ODEFunction{iip, AutoSpecialize}) where {iip} =
    ReactantCore.within_compile() ? FullSpecialize : AutoSpecialize
SciMLBase.specialization(::Type{<:ODEFunction{iip, AutoSpecialize}}) where {iip} =
    ReactantCore.within_compile() ? FullSpecialize : AutoSpecialize

end
