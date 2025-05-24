function adapt_structure(to,
        prob::Union{NonlinearProblem{<:Any, <:Any, iip},
            ImmutableNonlinearProblem{<:Any, <:Any, iip}}) where {iip}
    ImmutableNonlinearProblem{iip}(NonlinearFunction{iip}(adapt(to, prob.f.f)),
        adapt(to, prob.u0),
        adapt(to, prob.p);
        adapt(to, prob.kwargs)...)
end

function adapt_structure(to,
        prob::Union{ODEProblem{<:Any, <:Any, iip}, ImmutableODEProblem{<:Any, <:Any, iip}}) where {iip}
    ImmutableODEProblem{iip, FullSpecialize}(adapt(to, prob.f),
        adapt(to, prob.u0),
        adapt(to, prob.tspan),
        adapt(to, prob.p);
        adapt(to, prob.kwargs)...)
end

function adapt_structure(to, f::ODEFunction{iip}) where {iip}
    if f.mass_matrix !== I && f.initialization_data !== nothing
        error("Adaptation to GPU failed: DAEs of ModelingToolkit currently not supported.")
    end
    ODEFunction{iip, FullSpecialize}(f.f, jac = f.jac, mass_matrix = f.mass_matrix)
end
