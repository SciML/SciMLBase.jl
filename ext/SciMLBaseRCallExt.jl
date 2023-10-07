module SciMLBaseRCallExt

using RCall: RFunction
using SciMLBase

# Always assume a function from R is not in-place because copy-on-write disallows it!
function SciMLBase.isinplace(f::RFunction, args...; kwargs...)
    false
end

end