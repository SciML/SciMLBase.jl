Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, sym)
    if issymbollike(sym)
        if sym isa AbstractArray
            return map(s -> prob[s], sym)
        end
    end

    if issymbollike(sym)
        if has_sys(prob.f) && is_indep_sym(prob.f.sys, sym) ||
            Symbol(sym) == getindepsym(prob)
            return getindepsym(prob)
        elseif has_sys(prob.f) && is_param_sym(prob.f.sys, sym)
            return prob.p[param_sym_to_index(prob.f.sys, sym)]
        elseif has_paramsyms(prob.f) && Symbol(sym) in getparamsyms(prob)
            return prob.p[findfirst(x -> isequal(x, Symbol(sym)), getparamsyms(prob))]
        elseif Symbol(sym) in getsyms(prob)
            return prob.u0[sym_to_index(sym, prob)]
        else
            error("Invalid indexing of problem: $sym is not a state, parameter, or independent variable")
        end
    else
        error("Invalid indexing of problem: $sym is not a symbol")
    end
end

function Base.setindex!(prob::AbstractSciMLProblem, val, sym)
    if has_sys(prob.f)
        if issymbollike(sym)
            params = getparamsyms(prob)
            states = Symbol.(setdiff(getsyms(prob), params))
            params = Symbol.(params)

            i = findfirst(isequal(Symbol(sym)), states)
            if !isnothing(i)
                prob.u0[i] = val
                return prob
            end

            i = findfirst(isequal(Symbol(sym)), params)
            if !isnothing(i)
                prob.p[i] = val
                return prob
            end
            error("Invalid indexing of problem: $sym is not a state or parameter, it may be an observed variable.")
        else
            error("Invalid indexing of problem: $sym is not a symbol")
        end
    else
        error("Invalid indexing of problem: Problem does not support indexing without a system")
    end
end
