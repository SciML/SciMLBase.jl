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
        elseif has_sys(prob.f) && count('₊', String(Symbol(sym))) == 1   # Handles input like sys.X. 
            s_names = Symbol.(prob.f.sys.name,:₊,getsyms(prob))
            p_names = Symbol.(prob.f.sys.name,:₊,getparamsyms(prob))
            if count(isequal(Symbol(sym)), s_names) == 1
                return prob.u0[findfirst(isequal(Symbol(sym)), s_names)]
            elseif count(isequal(Symbol(sym)), p_names) == 1
                return prob.p[findfirst(isequal(Symbol(sym)), p_names)]
            end
        elseif (sym isa Symbol) && has_sys(prob.f)   # Handles input like :X (where X is a state). 
            s_f = Symbol.(getfield.(states(prob.f.sys),:f))
            s_count = count(isequal(sym), s_f)
            if s_count==1
                return prob.u0[findfirst(isequal(sym), s_f)]
            elseif s_count>1
                error("Tried to index with a Symbol (:$(sym)) that could represent several different possible states.")
            end
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
            s = Symbol.(states(prob.f.sys))
            params = Symbol.(params)

            i = findfirst(isequal(Symbol(sym)), s)
            if !isnothing(i)
                prob.u0[i] = val
                return prob
            elseif sym isa Symbol  # Hanldes input like :X.
                s_f = Symbol.(getfield.(states(prob.f.sys),:f))
                if count(isequal(Symbol(sym)), s_f) == 1
                    i = findfirst(isequal(sym), s_f)
                    prob.u0[i] = val
                    return prob
                elseif count(isequal(Symbol(sym)), s_f) > 1
                    error("The input symbol $(sym) occurs several times among problem states. Please avoid use Symbol form (:$(sym)).")
                end              
            elseif count('₊', String(Symbol(sym))) == 1  # Handles input like sys.X. 
                s_names = Symbol.(prob.f.sys.name,:₊,s)
                if count(isequal(Symbol(sym)), s_names) == 1
                    i = findfirst(isequal(Symbol(sym)), s_names)
                    prob.u0[i] = val
                    return prob
                end
            end

            i = findfirst(isequal(Symbol(sym)), params)
            if !isnothing(i)
                prob.p[i] = val
                return prob
            elseif count('₊', String(Symbol(sym))) == 1  # Handles input like sys.X. 
                p_names = Symbol.(prob.f.sys.name,:₊,params)
                if count(isequal(Symbol(sym)), p_names) == 1
                    i = findfirst(isequal(Symbol(sym)), p_names)
                    prob.p[i] = val
                    return prob
                end
            end
            error("Invalid indexing of problem: $sym is not a state or parameter, it may be an observed variable.")
        else
            error("Invalid indexing of problem: $sym is not a symbol")
        end
    else
        println("HERE 2")
        error("Invalid indexing of problem: Problem does not support indexing without a system")
    end
end
