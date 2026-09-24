module SciMLBasePartialFunctionsExt

using PartialFunctions: PartialFunctions
using SciMLBase: SciMLBase

SciMLBase.numargs(::PartialFunctions.PartialFunction{KL, UL}) where {KL, UL} = [length(UL)]

end
