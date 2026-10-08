module SciMLBaseReactantExt

using Reactant: Reactant
using SciMLBase: SciMLBase, ReturnCode

# A traced return code has no host value to branch on, so the comparisons are combined
# with the non-short-circuiting `|` into a traced `Bool`.
function SciMLBase.successful_retcode(retcode::Reactant.TracedEnum{ReturnCode.T})
    return mapreduce(code -> retcode == code, |, SciMLBase.SUCCESSFUL_RETCODES)
end

end
