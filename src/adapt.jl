function adapt_structure(
        to,
        prob::Union{
            NonlinearProblem{<:Any, iip},
            ImmutableNonlinearProblem{<:Any, iip},
        }
    ) where {iip}
    return ImmutableNonlinearProblem{iip}(
        NonlinearFunction{iip}(adapt(to, prob.f.f)),
        adapt(to, prob.u0),
        adapt(to, prob.p);
        adapt(to, prob.kwargs)...
    )
end

function adapt_structure(
        to,
        prob::Union{ODEProblem{<:Any, <:Any, iip}, ImmutableODEProblem{<:Any, <:Any, iip}}
    ) where {iip}
    return ImmutableODEProblem{iip, FullSpecialize}(
        adapt(to, prob.f),
        adapt(to, prob.u0),
        adapt(to, prob.tspan),
        adapt(to, prob.p);
        adapt(to, prob.kwargs)...
    )
end

# Allow DAE adaptation for GPU kernels
function adapt_structure(to, f::ODEFunction{iip}) where {iip}
    # For GPU kernels, we now support DAEs with mass matrices and initialization
    return ODEFunction{iip, FullSpecialize}(
        adapt(to, f.f),
        jac = adapt(to, f.jac),
        mass_matrix = adapt(to, f.mass_matrix),
        initialization_data = adapt(to, f.initialization_data)
    )
end

# Adapt OverrideInitData for GPU compatibility
function adapt_structure(to, f::OverrideInitData)
    return OverrideInitData(
        adapt(to, f.initializeprob),  # Also adapt initializeprob
        f.update_initializeprob!,
        f.initializeprobmap,
        f.initializeprobpmap,
        nothing,  # Set metadata to nothing for GPU compatibility
        f.is_update_oop
    )
end
