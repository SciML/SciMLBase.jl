function getsyms(sol::AbstractSciMLSolution)
    if has_syms(sol.prob.f)
        return sol.prob.f.syms
    else
        return keys(sol.u[1])
    end
end

function getsyms(prob::AbstractSciMLProblem)
    if has_syms(prob.f)
        return prob.f.syms
    else
        return []
    end
end

function getsyms(sol::AbstractOptimizationSolution)
    if has_syms(sol)
        return get_syms(sol)
    else
        return keys(sol.u[1])
    end
end

function getindepsym(prob::AbstractSciMLProblem)
    if has_indepsym(prob.f)
        return prob.f.indepsym
    else
        return nothing
    end
end

getindepsym(sol::AbstractSciMLSolution) = getindepsym(sol.prob)

function getparamsyms(prob::AbstractSciMLProblem)
    if has_paramsyms(prob.f)
        return prob.f.paramsyms
    else
        return nothing
    end
end

getparamsyms(sol) = getparamsyms(sol.prob)

function getparamsyms(sol::AbstractOptimizationSolution)
    if has_paramsyms(sol)
        return get_paramsyms(sol)
    else
        return nothing
    end
end

# Only for compatibility!
function getindepsym_defaultt(sol)
    if has_indepsym(sol.prob.f)
        return sol.prob.f.indepsym
    else
        return :t
    end
end

function getobserved(prob::AbstractSciMLProblem)
    if has_observed(prob.f)
        return prob.f.observed
    else
        return DEFAULT_OBSERVED
    end
end

getobserved(sol) = getobserved(sol.prob)

function getobserved(sol::AbstractOptimizationSolution)
    if has_observed(sol)
        return get_observed(sol)
    else
        return DEFAULT_OBSERVED
    end
end

cleansyms(syms::Nothing) = nothing
cleansyms(syms::Tuple) = collect(cleansym(sym) for sym in syms)
cleansyms(syms::Vector{Symbol}) = cleansym.(syms)
cleansyms(syms::LinearIndices) = nothing
cleansyms(syms::CartesianIndices) = nothing
cleansyms(syms::Base.OneTo) = nothing

function cleansym(sym::Symbol)
    str = String(sym)
    # MTK generated names
    rules = ("₊" => ".", "⦗" => "(", "⦘" => ")")
    for r in rules
        str = replace(str, r)
    end
    return str
end

function sym_to_index(sym, prob::AbstractSciMLProblem)
    if has_sys(prob.f) && is_state_sym(prob.f.sys, sym)
        return state_sym_to_index(prob.f.sys, sym)
    else
        return sym_to_index(sym, getsyms(prob))
    end
end

function sym_to_index(sym, sol::AbstractSciMLSolution)
    if has_sys(sol.prob.f) && is_state_sym(sol.prob.f.sys, sym)
        return state_sym_to_index(sol.prob.f.sys, sym)
    else
        return sym_to_index(sym, getsyms(sol))
    end
end

sym_to_index(sym, syms) = findfirst(isequal(Symbol(sym)), syms)
const issymbollike = RecursiveArrayTools.issymbollike
