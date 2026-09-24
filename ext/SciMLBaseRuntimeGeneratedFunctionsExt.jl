module SciMLBaseRuntimeGeneratedFunctionsExt

using SciMLBase: SciMLBase
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions

function SciMLBase.numargs(
        f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{
            T,
            V,
            W,
            I,
        }
    ) where {
        T,
        V,
        W,
        I,
    }
    return (length(T),)
end

end
