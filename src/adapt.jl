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

# Allow DAE adaptation for GPU kernels
function adapt_structure(to, f::SciMLBase.ODEFunction{iip}) where {iip}
    # For GPU kernels, we now support DAEs with mass matrices and initialization
    SciMLBase.ODEFunction{iip, SciMLBase.FullSpecialize}(
        f.f,
        jac = f.jac,
        mass_matrix = f.mass_matrix,
        initialization_data = f.initialization_data
    )
end

# Adapt OverrideInitData for GPU compatibility
function adapt_structure(to, f::SciMLBase.OverrideInitData)
    SciMLBase.OverrideInitData(
        adapt(to, f.initializeprob),  # Also adapt initializeprob
        f.update_initializeprob!,
        f.initializeprobmap,
        f.initializeprobpmap,
        nothing,  # Set metadata to nothing for GPU compatibility
        f.is_update_oop
    )
end
