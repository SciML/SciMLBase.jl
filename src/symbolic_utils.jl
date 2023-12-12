function getsyms(sol::AbstractSciMLSolution)
    syms = variable_symbols(sol)
    if isempty(syms)
        syms = keys(sol.u[1])
    end
    return syms
end

function getsyms(prob::AbstractSciMLProblem)
    return variable_symbols(prob.f)
end

function getsyms(sol::AbstractOptimizationSolution)
    syms = variable_symbols(sol)
    if isempty(syms)
        syms = keys(sol.u[1])
    end
    return syms
end

function getindepsym(prob::AbstractSciMLProblem)
    syms = independent_variable_symbols(prob.f)
    if isempty(syms)
        return nothing
    else
        return syms[1]
    end
end

getindepsym(sol::AbstractSciMLSolution) = getindepsym(sol.prob)

function getparamsyms(prob::AbstractSciMLProblem)
    psyms = parameter_symbols(prob.f)
    if isempty(psyms)
        return nothing
    end
    return psyms
end

getparamsyms(sol) = getparamsyms(sol.prob)

function getparamsyms(sol::AbstractOptimizationSolution)
    psyms = parameter_symbols(sol)
    if isempty(psyms)
        return nothing
    end
    return psyms
end

# Only for compatibility!
function getindepsym_defaultt(sol)
    return something(getindepsym(sol), :t)
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
    return variable_index(prob.f, sym)
end
function sym_to_index(sym, sol::AbstractSciMLSolution)
    idx = variable_index(sol.prob.f, sym)
    if idx === nothing
        idx = findfirst(isequal(sym), keys(sol.u[1]))
    end
    return idx
end
sym_to_index(sym, syms) = findfirst(isequal(Symbol(sym)), syms)
