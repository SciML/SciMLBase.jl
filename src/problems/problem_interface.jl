Base.@propagate_inbounds function Base.getproperty(prob::AbstractSciMLProblem, sym::Symbol)
    if sym === :ps
        return ParameterIndexingProxy(prob)
    end
    return getfield(prob, sym)
end

SymbolicIndexingInterface.symbolic_container(prob::AbstractSciMLProblem) = prob.f
SymbolicIndexingInterface.symbolic_container(prob::AbstractJumpProblem) = prob.prob
SymbolicIndexingInterface.symbolic_container(prob::AbstractEnsembleProblem) = prob.prob

SymbolicIndexingInterface.parameter_values(prob::AbstractSciMLProblem) = prob.p
function SymbolicIndexingInterface.parameter_values(prob::AbstractJumpProblem)
    parameter_values(prob.prob)
end
function SymbolicIndexingInterface.parameter_values(prob::AbstractEnsembleProblem)
    parameter_values(prob.prob)
end
SymbolicIndexingInterface.state_values(prob::AbstractSciMLProblem) = prob.u0
SymbolicIndexingInterface.state_values(prob::AbstractJumpProblem) = state_values(prob.prob)
function SymbolicIndexingInterface.state_values(prob::AbstractEnsembleProblem)
    state_values(prob.prob)
end
SymbolicIndexingInterface.current_time(prob::AbstractSciMLProblem) = prob.tspan[1]
SymbolicIndexingInterface.current_time(prob::AbstractJumpProblem) = current_time(prob.prob)
function SymbolicIndexingInterface.current_time(prob::AbstractEnsembleProblem)
    current_time(prob.prob)
end
SymbolicIndexingInterface.current_time(::AbstractSteadyStateProblem) = Inf

Base.@propagate_inbounds function Base.getindex(
        prob::AbstractSciMLProblem, ::SymbolicIndexingInterface.SolvedVariables)
    return getindex(prob, variable_symbols(prob))
end

Base.@propagate_inbounds function Base.getindex(
        prob::AbstractSciMLProblem, ::SymbolicIndexingInterface.AllVariables)
    return getindex(prob, all_variable_symbols(prob))
end

Base.@propagate_inbounds function Base.getindex(prob::AbstractSciMLProblem, sym)
    if symbolic_type(sym) == ScalarSymbolic()
        if is_variable(prob, sym)
            return state_values(prob, variable_index(prob, sym))
        elseif is_parameter(prob, sym)
            error("Indexing with parameters is deprecated. Use `getp(prob, $sym)(prob)` for parameter indexing.")
        elseif is_independent_variable(prob, sym)
            return current_time(prob)
        elseif is_observed(prob, sym)
            obs = SymbolicIndexingInterface.observed(prob, sym)
            if is_time_dependent(prob)
                return obs(state_values(prob), parameter_values(prob), current_time(prob))
            else
                return obs(state_values(prob), parameter_values(prob))
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
    if symbolic_type(sym) == ScalarSymbolic()
        if is_variable(prob, sym)
            set_state!(prob, val, variable_index(prob, sym))
        elseif is_parameter(prob, sym)
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
