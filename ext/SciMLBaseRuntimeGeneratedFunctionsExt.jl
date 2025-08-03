module SciMLBaseRuntimeGeneratedFunctionsExt

using SciMLBase
using RuntimeGeneratedFunctions

function SciMLBase.numargs(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{
        T,
        V,
        W,
        I
}) where {
        T,
        V,
        W,
        I
}
    (length(T),)
end

end
