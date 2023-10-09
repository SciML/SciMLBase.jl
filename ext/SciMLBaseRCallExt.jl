module SciMLBaseRCallExt

using RCall: RFunction
using SciMLBase

function SciMLBase.numargs(f::RFunction)
    R"formals"(f)
end

end
