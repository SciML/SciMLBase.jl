module SciMLBasePartialFunctionsExt

using PartialFunctions, SciMLBase

SciMLBase.numargs(::PartialFunctions.PartialFunction{KL, UL}) where {KL, UL} = [length(UL)]

end
