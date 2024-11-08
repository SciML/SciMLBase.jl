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

Base.@propagate_inbounds function Base.getindex(A::AbstractSciMLProblem, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `prob.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractSciMLProblem, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `prob.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

function Base.setindex!(prob::AbstractSciMLProblem, args...; kwargs...)
    ___internal_setindex!(prob::AbstractSciMLProblem, args...; kwargs...)
end

function ___internal_setindex!(A::AbstractSciMLProblem, val, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `prob.ps[$sym] = $val` for parameter indexing.")
    end
    return setsym(A, sym)(A, val)
end

function ___internal_setindex!(
        A::AbstractSciMLProblem, val, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `prob.ps[$sym] = $val` for parameter indexing.")
    end
    return setsym(A, sym)(A, val)
end
