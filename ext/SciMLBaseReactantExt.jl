module SciMLBaseReactantExt

using Reactant: Reactant
using SciMLBase: SciMLBase, ReturnCode

# Must list the codes `successful_retcode(::ReturnCode.T)` accepts; test/reactant_retcodes.jl
# checks both methods agree on every `ReturnCode`.
const SUCCESSFUL_RETCODES = (
    ReturnCode.Success, ReturnCode.Terminated, ReturnCode.ExactSolutionLeft,
    ReturnCode.ExactSolutionRight, ReturnCode.FloatingPointLimit, ReturnCode.StalledSuccess,
)

# A traced return code has no host value to branch on, so the comparisons are combined
# with the non-short-circuiting `|` into a traced `Bool`.
function SciMLBase.successful_retcode(retcode::Reactant.TracedEnum{ReturnCode.T})
    return mapreduce(code -> retcode == code, |, SUCCESSFUL_RETCODES)
end

end
