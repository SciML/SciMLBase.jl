module SciMLBaseDifferentiationInterfaceExt

using SciMLBase, DifferentiationInterface

import SciMLBase: anyeltypedual

# Opt out since these are using for preallocation, not differentiation
function anyeltypedual(
        x::DifferentiationInterface.Prep,
        ::Type{Val{counter}} = Val{0}
    ) where {counter}
    return Any
end
function anyeltypedual(
        x::Type{T},
        ::Type{Val{counter}} = Val{0}
    ) where {counter} where {
        T <:
        DifferentiationInterface.Prep,
    }
    return Any
end

end
