Base.@propagate_inbounds function Base.getproperty(prob::AbstractSciMLProblem, sym::Symbol)
    if sym === :ps
        return ParameterIndexingProxy(prob)
    end
    return getfield(prob, sym)
end

SymbolicIndexingInterface.symbolic_container(prob::AbstractSciMLProblem) = prob.f

SymbolicIndexingInterface.parameter_values(prob::AbstractSciMLProblem) = prob.p
SymbolicIndexingInterface.state_values(prob::AbstractSciMLProblem) = prob.u0
SymbolicIndexingInterface.current_time(prob::AbstractSciMLProblem) = prob.tspan[1]

Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, ::SymbolicIndexingInterface.SolvedVariables)
    return getindex(prob, variable_symbols(prob))
end

Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, ::SymbolicIndexingInterface.AllVariables)
    return getindex(prob, all_variable_symbols(prob))
end

Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, sym)
    if symbolic_type(sym) == ScalarSymbolic()
        if is_variable(prob.f, sym)
            return prob.u0[variable_index(prob.f, sym)]
        elseif is_parameter(prob.f, sym)
        error("Indexing with parameters is deprecated. Use `getp(prob, $sym)(prob)` for parameter indexing.")
        elseif is_independent_variable(prob.f, sym)
            return getindepsym(prob)
        elseif is_observed(prob.f, sym)
            obs = SymbolicIndexingInterface.observed(prob, sym)
            if is_time_dependent(prob.f)
                return obs(prob.u0, prob.p, 0.0)
            else
                return obs(prob.u0, prob.p)
            end
        else
            error("Invalid indexing of problem: $sym is not a state, parameter, or independent variable")
        end
    elseif symbolic_type(sym) == ArraySymbolic()
        return map(s -> prob[s], collect(sym))
    else
        sym isa AbstractArray || error("Invalid indexing of problem")
        return map(s -> prob[s], sym)
    end
end

function Base.setindex!(prob::AbstractSciMLProblem, args...; kwargs...)
    ___internal_setindex!(prob::AbstractSciMLProblem, args...; kwargs...)
end
function ___internal_setindex!(prob::AbstractSciMLProblem, val, sym)
    has_sys(prob.f) || error("Invalid indexing of problem: Problem does not support indexing without a system")
    if symbolic_type(sym) == ScalarSymbolic()
        if is_variable(prob.f, sym)
            prob.u0[variable_index(prob.f, sym)] = val
        elseif is_parameter(prob.f, sym)
            error("Indexing with parameters is deprecated. Use `setp(prob, $sym)(prob, $val)` to set parameter value.")
        else
            error("Invalid indexing of problem: $sym is not a state or parameter, it may be an observed variable.")
        end
        return prob
    elseif symbolic_type(sym) == ArraySymbolic()
        setindex!.((prob,), val, collect(sym))
        return prob
    else
        sym isa AbstractArray || error("Invalid indexing of problem")
        setindex!.((prob,), val, sym)
        return prob
    end
end
