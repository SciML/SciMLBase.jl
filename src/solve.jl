# Skip the DiffEqBase handling
struct InverseScale{T}
    scale::T
end

(inv_scale::InverseScale)(x, compute_inverse::Bool = false) =
    compute_inverse ? x .* inv_scale.scale : x ./ inv_scale.scale

function solve(prob::OptimizationProblem, opt, args...;
               scale::Bool = false, scaling_function = nothing,
               cb = (args...) -> (false), kwargs...)
    !scale && return __solve(prob, opt, args...; kwargs...)

    θ_start = copy(prob.u0)

    if isnothing(scaling_function) && any(iszero.(θ_start))
        error("Default Inverse Scaling is not compatible with `0` as initial guess")
    end

    scaling_function = if isnothing(scaling_function)
            InverseScale(θ_start)
        else
            # Check if arguments are compatible
            # First arg is the parameter
            # 2nd one denotes inverse computation or not
            scaling_function(θ_start, false)
            scaling_function
        end

    normalized_f(α, args...) = prob.f.f(scaling_function(α, true), args...)
    normalized_cb(α, args...) = cb(scaling_function(α, true), args...)

    lb = isnothing(prob.lb) ? nothing : scaling_function(prob.lb, false)
    ub = isnothing(prob.ub) ? nothing : scaling_function(prob.ub, false)

    _prob = remake(prob, u0 = scaling_function(prob.u0, false), lb = lb,
                   ub = ub,
                   f = OptimizationFunction(normalized_f, prob.f.adtype,
                                            grad = prob.f.grad, hess = prob.f.hess,
                                            hv = prob.f.hv, cons = prob.f.cons,
                                            cons_j = prob.f.cons_j, cons_h = prob.f.cons_h))

    optsol = solve(_prob, opt, args...; cb = normalized_cb, kwargs...)
    optsol.u .= scaling_function(optsol.u, true)
    optsol
end
