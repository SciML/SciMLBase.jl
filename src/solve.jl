# Skip the DiffEqBase handling
function solve(prob::OptimizationProblem, opt, args...;
               scale::Bool = false,
               scaling_functions::Union{Nothing, Tuple{Function, Function}} = nothing,
               cb = (args...) -> (false),
               kwargs...)
    !scale && return __solve(prob, opt, args...; kwargs...)

    θ_start = copy(prob.u0)
    
    scale_function, inv_scale_function =
        isnothing(scaling_functions) ? (x -> x ./ θ_start, x -> x .* θ_start) : scaling_functions

    if isnothing(scaling_functions) && any(iszero.(θ_start))
        error("Default Inverse Scaling is not compatible with `0` as initial guess")
    end

    normalized_f(α, args...) = prob.f.f(inv_scale_function(α), args...)
    normalized_cb(α, args...) = cb(inv_scale_function(α), args...)

    lb = isnothing(prob.lb) ? nothing : scale_function(prob.lb)
    ub = isnothing(prob.ub) ? nothing : scale_function(prob.ub)

    _prob = remake(prob, u0 = scale_function(prob.u0), lb = lb,
                   ub = ub,
                   f = OptimizationFunction(normalized_f, prob.f.adtype,
                                            grad = prob.f.grad, hess = prob.f.hess,
                                            hv = prob.f.hv, cons = prob.f.cons,
                                            cons_j = prob.f.cons_j, cons_h = prob.f.cons_h))

    optsol = solve(_prob, opt, args...; cb = normalized_cb, kwargs...)
    optsol.u .= inv_scale_function(optsol.u)
    optsol
end
