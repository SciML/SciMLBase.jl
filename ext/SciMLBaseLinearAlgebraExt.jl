module SciMLBaseLinearAlgebraExt

using SciMLBase
using LinearAlgebra

# Override the identity matrix constant after module loading
function __init__()
    SciMLBase.I = LinearAlgebra.I
end

# Extension implementations
SciMLBase.default_identity_matrix() = LinearAlgebra.I

function SciMLBase.has_non_trivial_mass_matrix(prob)
    hasproperty(prob.f, :mass_matrix) && !(prob.f.mass_matrix isa UniformScaling{Bool})
end

function SciMLBase.get_initial_values(
        prob::SciMLBase.AbstractDEProblem, integrator::SciMLBase.DEIntegrator, f, alg::SciMLBase.CheckInit,
        isinplace::Union{Val{true}, Val{false}}; abstol, kwargs...)
    u0 = SciMLBase.state_values(integrator)
    p = SciMLBase.parameter_values(integrator)
    t = SciMLBase.current_time(integrator)
    M = f.mass_matrix

    M == LinearAlgebra.I && return u0, p, true
    algebraic_vars = [all(iszero, x) for x in eachcol(M)]
    algebraic_eqs = [all(iszero, x) for x in eachrow(M)]
    (iszero(algebraic_vars) || iszero(algebraic_eqs)) && return u0, p, true
    SciMLBase.update_coefficients!(M, u0, p, t)
    tmp = SciMLBase.evaluate_f(integrator, prob, f, isinplace, u0, p, t)
    tmp .= SciMLBase.ArrayInterface.restructure(tmp, algebraic_eqs .* SciMLBase._vec(tmp))

    normresid = isdefined(integrator.opts, :internalnorm) ?
                integrator.opts.internalnorm(tmp, t) : norm(tmp)
    if normresid > abstol
        isdae = prob isa SciMLBase.AbstractDAEProblem
        throw(SciMLBase.CheckInitFailureError(normresid, abstol, isdae))
    end
    u0, p, true
end

function SciMLBase.calculate_solution_errors!(
        sol::SciMLBase.AbstractODESolution; fill_uanalytic = true,
        timeseries_errors = true, dense_errors = true)
    f = sol.prob.f

    if fill_uanalytic
        for i in 1:size(sol.u, 1)
            if sol.prob isa SciMLBase.AbstractDDEProblem
                push!(sol.u_analytic,
                    f.analytic(sol.prob.u0, sol.prob.h, sol.prob.p, sol.t[i]))
            else
                push!(sol.u_analytic, f.analytic(sol.prob.u0, sol.prob.p, sol.t[i]))
            end
        end
    end

    save_everystep = length(sol.u) > 2
    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(SciMLBase.recursive_mean(abs.(sol.u[end] .-
                                                                sol.u_analytic[end])))

        if save_everystep && timeseries_errors
            sol.errors[:l∞] = norm(maximum(SciMLBase.vecvecapply((x) -> abs.(x),
                sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(SciMLBase.recursive_mean(SciMLBase.vecvecapply(
                (x) -> float.(x) .^ 2,
                sol.u - sol.u_analytic))))
            if sol.dense && dense_errors
                densetimes = collect(range(sol.t[1], stop = sol.t[end], length = 100))
                interp_u = sol(densetimes)
                interp_analytic = SciMLBase.VectorOfArray([f.analytic(sol.prob.u0, sol.prob.p, t)
                                                           for t in densetimes])
                sol.errors[:L∞] = norm(maximum(SciMLBase.vecvecapply((x) -> abs.(x),
                    interp_u - interp_analytic)))
                sol.errors[:L2] = norm(sqrt(SciMLBase.recursive_mean(SciMLBase.vecvecapply(
                    (x) -> float.(x) .^ 2,
                    interp_u -
                    interp_analytic))))
            end
        end
    end
end

function SciMLBase.calculate_solution_errors!(
        sol::SciMLBase.AbstractDAESolution; fill_uanalytic = true,
        timeseries_errors = true, dense_errors = true)
    prob = sol.prob
    f = prob.f

    if fill_uanalytic
        for i in 1:size(sol.u, 1)
            push!(sol.u_analytic, f.analytic(prob.du0, prob.u0, prob.p, sol.t[i]))
        end
    end

    save_everystep = length(sol.u) > 2
    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(SciMLBase.recursive_mean(abs.(sol.u[end] -
                                                                sol.u_analytic[end])))

        if save_everystep && timeseries_errors
            sol.errors[:l∞] = norm(maximum(SciMLBase.vecvecapply(x -> abs.(x),
                sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(SciMLBase.recursive_mean(SciMLBase.vecvecapply(
                x -> float.(x) .^ 2,
                sol.u - sol.u_analytic))))
            if sol.dense && dense_errors
                densetimes = collect(range(sol.t[1]; stop = sol.t[end], length = 100))
                interp_u = sol(densetimes)
                interp_analytic = SciMLBase.VectorOfArray([f.analytic(prob.du0, prob.u0, prob.p, t)
                                                           for t in densetimes])
                sol.errors[:L∞] = norm(maximum(SciMLBase.vecvecapply(x -> abs.(x),
                    interp_u - interp_analytic)))
                sol.errors[:L2] = norm(sqrt(SciMLBase.recursive_mean(SciMLBase.vecvecapply(
                    x -> float.(x) .^ 2,
                    interp_u .-
                    interp_analytic))))
            end
        end
    end
end

function SciMLBase.calculate_solution_errors!(
        sol::SciMLBase.AbstractRODESolution; fill_uanalytic = true,
        timeseries_errors = true)
    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(SciMLBase.recursive_mean(abs.(sol.u[end] -
                                                                sol.u_analytic[end])))
        if timeseries_errors
            sol.errors[:l∞] = norm(maximum(SciMLBase.vecvecapply((x) -> abs.(x),
                sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(SciMLBase.recursive_mean(SciMLBase.vecvecapply(
                (x) -> float.(x) .^ 2,
                sol.u - sol.u_analytic))))
        end
    end
end

end
