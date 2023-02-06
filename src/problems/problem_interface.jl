Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, sym)
    if issymbollike(sym)
        if sym isa AbstractArray
            return map(s -> prob[s], sym)
        end
        i = sym_to_index(sym, prob)
    elseif all(issymbollike, sym)
        if has_sys(prob.f) && all(Base.Fix1(is_param_sym, prob.f.sys), sym) ||
           !has_sys(prob.f) && has_paramsyms(prob.f) &&
           all(in(getparamsyms(prob)), Symbol.(sym))
            return getindex.((prob,), sym)
        else
            error("Invalid indexing of problem: Problem does not support numeric indexing")
        end
    else
        i = sym
    end

    if i === nothing
        if issymbollike(sym)
            if has_sys(prob.f) && is_indep_sym(prob.f.sys, sym) ||
               Symbol(sym) == getindepsym(prob)
                return getindepsym(prob)
            elseif has_sys(prob.f) && is_param_sym(prob.f.sys, sym)
                return prob.p[param_sym_to_index(prob.f.sys, sym)]
            elseif has_paramsyms(prob.f) && Symbol(sym) in getparamsyms(prob)
                return prob.p[findfirst(x -> isequal(x, Symbol(sym)), getparamsyms(prob))]
            else Symbol(sym) in getsyms(prob)
                return prob.u0[sym_to_index(sym, prob)]
            else
                error("Invalid indexing of problem: $sym is not a state, parameter, or independent variable")
            end
        else
            error("Invalid indexing of problem: $sym is not a symbol")
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        error("Invalid indexing of problem: Problem does not support numeric indexing")
    else
        error("Invalid indexing of problem")
    end
end

function Base.setindex!(prob::AbstractSciMLProblem, val, sym)
    if has_sys(prob.f)
        if issymbollike(sym)
            if is_state_sym(prob.f.sys, sym)
                remake!(prob, u0 = [sym => val])
            elseif is_param_sym(prob.f.sys, sym)
                remake!(prob, ps = [sym => val])
            else
                error("Invalid indexing of problem: $sym is not a state or parameter")
            end
        else
            error("Invalid indexing of problem: $sym is not a symbol")
        end
    else
        error("Invalid indexing of problem: Problem does not support indexing without a system")
    end
end
