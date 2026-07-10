"""
    EnsembleAnalysis

Namespace for summary statistics over
[`SciMLBase.AbstractEnsembleSolution`](@ref SciMLBase.AbstractEnsembleSolution)
trajectories. Import it with `using SciMLBase.EnsembleAnalysis`; its functions
provide componentwise, timestep, timepoint, and time-series summaries,
including weighted covariance operations where supported.
"""
module EnsembleAnalysis

using SciMLBase
using Statistics: Statistics, cov, median, quantile
using RecursiveArrayTools: RecursiveArrayTools, DiffEqArray, VectorOfArray,
    vecarr_to_vectors
using StaticArraysCore: StaticArraysCore
using DocStringExtensions: DocStringExtensions, SIGNATURES

# Getters
"""
$(SIGNATURES)

Return a lazy iterator over `sol.u[i]` for every trajectory in an ensemble
solution.

This is a step-index based accessor: it assumes that the `i`th saved value is the
quantity to compare across trajectories. That is appropriate for fixed-step
solutions or ensembles saved with common `saveat` values. Use
[`get_timepoint`](@ref) when trajectories should be compared at a physical time
through interpolation.
"""
get_timestep(sim, i) = (sol.u[i] for sol in sim.u)

"""
$(SIGNATURES)

Return a lazy iterator over `sol(t)` for every trajectory in an ensemble
solution.

This is a time-point based accessor: each trajectory is evaluated at the same
independent-variable value `t`, using the solution's callable interpolation
interface. Use [`get_timestep`](@ref) when comparing the same saved index instead
of the same physical time.
"""
get_timepoint(sim, t) = (sol(t) for sol in sim.u)

"""
$(SIGNATURES)

Collect the values at saved step index `i` into componentwise trajectory vectors.

For scalar-valued trajectories, the result is a vector of scalar values. For
array-valued trajectories, the result is a vector whose entries contain the
values of one state component across all trajectories, preserving the component
layout needed by the summary-statistic helpers.
"""
function componentwise_vectors_timestep(sim, i)
    arr = [get_timestep(sim, i)...]
    if arr[1] isa AbstractArray
        return vecarr_to_vectors(VectorOfArray(arr))
    else
        return arr
    end
end

"""
$(SIGNATURES)

Collect interpolated values at time `t` into componentwise trajectory vectors.

For scalar-valued trajectories, the result is a vector of scalar values. For
array-valued trajectories, the result is a vector whose entries contain the
values of one state component across all trajectories, using the same component
layout as [`componentwise_vectors_timestep`](@ref).
"""
function componentwise_vectors_timepoint(sim, t)
    arr = [get_timepoint(sim, t)...]
    if arr[1] isa AbstractArray
        return vecarr_to_vectors(VectorOfArray(arr))
    else
        return arr
    end
end

# Timestep statistics
"""
$(SIGNATURES)

Compute the ensemble mean at saved step index `i`.

For array-valued states, the returned value has the same shape as a single state
and contains the componentwise mean across trajectories. For scalar states, the
returned value is a scalar mean. Passing `:` computes the full step-indexed mean
timeseries via [`timeseries_steps_mean`](@ref).
"""
timestep_mean(sim, i) = componentwise_mean(get_timestep(sim, i))
timestep_mean(sim, ::Colon) = timeseries_steps_mean(sim)

"""
$(SIGNATURES)

Compute the ensemble median at saved step index `i`.

For array-valued states, the result is reshaped to match the state at
`sim.u[1].u[i]`; for scalar states, it is the scalar median across trajectories.
Passing `:` computes medians for every saved step.
"""
function timestep_median(sim, i)
    arr = componentwise_vectors_timestep(sim, i)
    if typeof(first(arr)) <: AbstractArray
        return reshape([median(x) for x in arr], size(sim.u[1].u[i])...)
    else
        return median(arr)
    end
end
timestep_median(sim, ::Colon) = timeseries_steps_median(sim)

"""
$(SIGNATURES)

Compute the componentwise quantile `q` at saved step index `i`.

`q` is passed to `Statistics.quantile` for each state component across
trajectories. Array-valued states are reshaped to match the state at
`sim.u[1].u[i]`; scalar states return a scalar quantile. Passing `:` computes the
quantile for every saved step.
"""
function timestep_quantile(sim, q, i)
    arr = componentwise_vectors_timestep(sim, i)
    if typeof(first(arr)) <: AbstractArray
        return reshape([quantile(x, q) for x in arr], size(sim.u[1].u[i])...)
    else
        return quantile(arr, q)
    end
end
timestep_quantile(sim, q, ::Colon) = timeseries_steps_quantile(sim, q)

"""
$(SIGNATURES)

Compute the ensemble mean and variance at saved step index `i`.

The result is `(mean, variance)`, computed componentwise across trajectories with
Bessel correction by the shared componentwise statistics helper. Passing `:`
computes the full step-indexed mean and variance timeseries.
"""
timestep_meanvar(sim, i) = componentwise_meanvar(get_timestep(sim, i))
timestep_meanvar(sim, ::Colon) = timeseries_steps_meanvar(sim)

"""
$(SIGNATURES)

Compute componentwise means and covariance between saved step indices `i` and `j`.

The result is `(mean_i, mean_j, covariance)`, where each entry is scalar-valued
for scalar states or shaped componentwise for array-valued states. Passing
`(:, :)` computes the full step-indexed covariance matrix.
"""
function timestep_meancov(sim, i, j)
    return componentwise_meancov(get_timestep(sim, i), get_timestep(sim, j))
end
timestep_meancov(sim, ::Colon, ::Colon) = timeseries_steps_meancov(sim)

"""
$(SIGNATURES)

Compute componentwise means and correlation between saved step indices `i` and
`j`.

The result is `(mean_i, mean_j, correlation)`, using the covariance and variance
computed across trajectories. Passing `(:, :)` computes the full step-indexed
correlation matrix.
"""
function timestep_meancor(sim, i, j)
    return componentwise_meancor(get_timestep(sim, i), get_timestep(sim, j))
end
timestep_meancor(sim, ::Colon, ::Colon) = timeseries_steps_meancor(sim)

"""
$(SIGNATURES)

Compute componentwise weighted means and covariance between saved step indices
`i` and `j`.

`W` supplies the trajectory weights used by the weighted covariance calculation.
The result is `(mean_i, mean_j, weighted_covariance)`. Passing `(:, :)` computes
the full step-indexed weighted covariance matrix.
"""
function timestep_weighted_meancov(sim, W, i, j)
    return componentwise_weighted_meancov(get_timestep(sim, i), get_timestep(sim, j), W)
end
function timestep_weighted_meancov(sim, W, ::Colon, ::Colon)
    return timeseries_steps_weighted_meancov(sim, W)
end

"""
$(SIGNATURES)

Compute the ensemble mean at every saved step.

The result is a `DiffEqArray` with the same time vector as the first trajectory,
where each saved value is the componentwise mean across trajectories at the same
saved step index.
"""
function timeseries_steps_mean(sim)
    return DiffEqArray([timestep_mean(sim, i) for i in 1:length(sim.u[1].t)], sim.u[1].t)
end

"""
$(SIGNATURES)

Compute the ensemble median at every saved step.

The result is a `DiffEqArray` with the first trajectory's time vector and
componentwise median values at each saved step index.
"""
function timeseries_steps_median(sim)
    return DiffEqArray([timestep_median(sim, i) for i in 1:length(sim.u[1].t)], sim.u[1].t)
end

"""
$(SIGNATURES)

Compute the componentwise quantile `q` at every saved step.

The result is a `DiffEqArray` with the first trajectory's time vector and
componentwise quantile values at each saved step index.
"""
function timeseries_steps_quantile(sim, q)
    return DiffEqArray([timestep_quantile(sim, q, i) for i in 1:length(sim.u[1].t)], sim.u[1].t)
end

"""
$(SIGNATURES)

Compute the ensemble mean and variance at every saved step.

The result is `(means, variances)`, where both entries are `DiffEqArray`s sharing
the first trajectory's time vector.
"""
function timeseries_steps_meanvar(sim)
    m, v = timestep_meanvar(sim, 1)
    means = [m]
    vars = [v]
    for i in 2:length(sim.u[1].t)
        m, v = timestep_meanvar(sim, i)
        push!(means, m)
        push!(vars, v)
    end
    return DiffEqArray(means, sim.u[1].t), DiffEqArray(vars, sim.u[1].t)
end

"""
$(SIGNATURES)

Compute the step-indexed matrix of componentwise mean/covariance summaries.

Entry `(i, j)` contains the result of [`timestep_meancov`](@ref). This
assumes saved step indices are comparable across trajectories.
"""
function timeseries_steps_meancov(sim)
    return [
        timestep_meancov(sim, i, j) for i in 1:length(sim.u[1].t),
            j in 1:length(sim.u[1].t)
    ]
end

"""
$(SIGNATURES)

Compute the step-indexed matrix of componentwise mean/correlation summaries.

Entry `(i, j)` contains the result of [`timestep_meancor`](@ref). This
assumes saved step indices are comparable across trajectories.
"""
function timeseries_steps_meancor(sim)
    return [
        timestep_meancor(sim, i, j) for i in 1:length(sim.u[1].t),
            j in 1:length(sim.u[1].t)
    ]
end

"""
$(SIGNATURES)

Compute the step-indexed matrix of componentwise weighted covariance summaries.

Entry `(i, j)` contains the weighted mean/covariance summary for saved step
indices `i` and `j` using trajectory weights `W`.
"""
function timeseries_steps_weighted_meancov(sim, W)
    return [
        timestep_weighted_meancov(sim, W, i, j) for i in 1:length(sim.u[1].t),
            j in 1:length(sim.u[1].t)
    ]
end

"""
$(SIGNATURES)

Compute the ensemble mean at physical time `t`.

Each trajectory is evaluated with `sol(t)`, so this requires a callable solution
at `t`. For array-valued states, the result has the same shape as a single state;
for scalar states, it is a scalar mean.
"""
timepoint_mean(sim, t) = componentwise_mean(get_timepoint(sim, t))

"""
$(SIGNATURES)

Compute the componentwise ensemble median at physical time `t`.

Each trajectory is evaluated with `sol(t)`. Array-valued states are reshaped to
match a single saved state layout; scalar states return a scalar median.
"""
function timepoint_median(sim, t)
    arr = componentwise_vectors_timepoint(sim, t)
    if typeof(first(arr)) <: AbstractArray
        return reshape([median(x) for x in arr], size(sim.u[1].u[1])...)
    else
        return median(arr)
    end
end

"""
$(SIGNATURES)

Compute the componentwise quantile `q` at physical time `t`.

Each trajectory is evaluated with `sol(t)`, then `Statistics.quantile` is applied
componentwise across trajectories.
"""
function timepoint_quantile(sim, q, t)
    arr = componentwise_vectors_timepoint(sim, t)
    if typeof(first(arr)) <: AbstractArray
        return reshape([quantile(x, q) for x in arr], size(sim.u[1].u[1])...)
    else
        return quantile(arr, q)
    end
end

"""
$(SIGNATURES)

Compute the ensemble mean and variance at physical time `t`.

The result is `(mean, variance)`, computed componentwise across interpolated
trajectory values at `t`.
"""
timepoint_meanvar(sim, t) = componentwise_meanvar(get_timepoint(sim, t))

"""
$(SIGNATURES)

Compute componentwise means and covariance between physical times `t1` and `t2`.

Each trajectory is evaluated at both times. The result is
`(mean_t1, mean_t2, covariance)`.
"""
function timepoint_meancov(sim, t1, t2)
    return componentwise_meancov(get_timepoint(sim, t1), get_timepoint(sim, t2))
end

"""
$(SIGNATURES)

Compute componentwise means and correlation between physical times `t1` and
`t2`.

Each trajectory is evaluated at both times. The result is
`(mean_t1, mean_t2, correlation)`.
"""
function timepoint_meancor(sim, t1, t2)
    return componentwise_meancor(get_timepoint(sim, t1), get_timepoint(sim, t2))
end

"""
$(SIGNATURES)

Compute componentwise weighted means and covariance between physical times `t1`
and `t2`.

`W` supplies the trajectory weights used by the weighted covariance calculation.
The result is `(mean_t1, mean_t2, weighted_covariance)`.
"""
function timepoint_weighted_meancov(sim, W, t1, t2)
    return componentwise_weighted_meancov(get_timepoint(sim, t1), get_timepoint(sim, t2), W)
end

function SciMLBase.EnsembleSummary(
        sim::SciMLBase.AbstractEnsembleSolution{T, N},
        t = sim.u[1].t; quantiles = [0.05, 0.95]
    ) where {T, N}
    if sim.u[1] isa SciMLBase.AbstractSciMLSolution
        m, v = timeseries_point_meanvar(sim, t)
        med = timeseries_point_median(sim, t)
        qlow = timeseries_point_quantile(sim, quantiles[1], t)
        qhigh = timeseries_point_quantile(sim, quantiles[2], t)
    else
        m, v = timeseries_steps_meanvar(sim)
        med = timeseries_steps_median(sim)
        qlow = timeseries_steps_quantile(sim, quantiles[1])
        qhigh = timeseries_steps_quantile(sim, quantiles[2])
    end

    trajectories = length(sim.u)
    return EnsembleSummary{
        T, N, typeof(t), typeof(m), typeof(v), typeof(med), typeof(qlow),
        typeof(qhigh),
    }(
        t, m, v, med, qlow, qhigh, trajectories, sim.elapsedTime,
        sim.converged
    )
end

"""
$(SIGNATURES)

Compute the ensemble mean at each physical time in `ts`.

The result is a `DiffEqArray` whose time axis is `ts` and whose values are the
componentwise means of `sol(t)` across trajectories.
"""
function timeseries_point_mean(sim, ts)
    return DiffEqArray([timepoint_mean(sim, t) for t in ts], ts)
end

"""
$(SIGNATURES)

Compute the componentwise ensemble median at each physical time in `ts`.

The result is a `DiffEqArray` over `ts`; each value is computed from the
interpolated trajectory values at that time.
"""
function timeseries_point_median(sim, ts)
    return DiffEqArray([timepoint_median(sim, t) for t in ts], ts)
end

"""
$(SIGNATURES)

Compute the componentwise quantile `q` at each physical time in `ts`.

The result is a `DiffEqArray` over `ts`; each value is computed from the
interpolated trajectory values at that time.
"""
function timeseries_point_quantile(sim, q, ts)
    return DiffEqArray([timepoint_quantile(sim, q, t) for t in ts], ts)
end

"""
$(SIGNATURES)

Compute the ensemble mean and variance at each physical time in `ts`.

The result is `(means, variances)`, where both entries are `DiffEqArray`s over
`ts`.
"""
function timeseries_point_meanvar(sim, ts)
    m, v = timepoint_meanvar(sim, first(ts))
    means = [m]
    vars = [v]
    for t in Iterators.drop(ts, 1)
        m, v = timepoint_meanvar(sim, t)
        push!(means, m)
        push!(vars, v)
    end
    return DiffEqArray(means, ts), DiffEqArray(vars, ts)
end

"""
$(SIGNATURES)

Compute the time-point covariance summary matrix for adjacent entries of `ts`.

This method pairs `ts[1:end-1]` with `ts[2:end]` and returns the same matrix form
as `timeseries_point_meancov(sim, ts1, ts2)`.
"""
function timeseries_point_meancov(sim, ts)
    return timeseries_point_meancov(sim, ts[1:(end - 1)], ts[2:end])
end

"""
$(SIGNATURES)

Compute the time-point covariance summary matrix between two time collections.

Entry `(i, j)` contains the result of `timepoint_meancov(sim, ts1[i], ts2[j])`.
"""
function timeseries_point_meancov(sim, ts1, ts2)
    return [timepoint_meancov(sim, t1, t2) for t1 in ts1, t2 in ts2]
end

"""
$(SIGNATURES)

Compute the time-point correlation summary matrix for adjacent entries of `ts`.

This method pairs `ts[1:end-1]` with `ts[2:end]` and returns the same matrix form
as `timeseries_point_meancor(sim, ts1, ts2)`.
"""
function timeseries_point_meancor(sim, ts)
    return timeseries_point_meancor(sim, ts[1:(end - 1)], ts[2:end])
end

"""
$(SIGNATURES)

Compute the time-point correlation summary matrix between two time collections.

Entry `(i, j)` contains the result of `timepoint_meancor(sim, ts1[i], ts2[j])`.
"""
function timeseries_point_meancor(sim, ts1, ts2)
    return [timepoint_meancor(sim, t1, t2) for t1 in ts1, t2 in ts2]
end

"""
$(SIGNATURES)

Compute the weighted covariance summary matrix for adjacent entries of `ts`.

This method pairs `ts[1:end-1]` with `ts[2:end]` and uses weights `W` for each
trajectory.
"""
function timeseries_point_weighted_meancov(sim, W, ts)
    return timeseries_point_weighted_meancov(sim, W, ts[1:(end - 1)], ts[2:end])
end

"""
$(SIGNATURES)

Compute the weighted covariance summary matrix between two time collections.

Entry `(i, j)` contains the weighted mean/covariance summary for `ts1[i]` and
`ts2[j]` using trajectory weights `W`.
"""
function timeseries_point_weighted_meancov(sim, W, ts1, ts2)
    return [timepoint_weighted_meancov(sim, W, t1, t2) for t1 in ts1, t2 in ts2]
end

function componentwise_mean(A)
    x0 = first(A)
    n = 0
    mean = zero(x0) ./ 1
    for x in A
        n += 1
        if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
            mean .+= x
        else
            mean += x
        end
    end
    if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
        mean ./= n
    else
        mean /= n
    end
    return mean
end

# Welford algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
function componentwise_meanvar(A; bessel = true)
    x0 = first(A)
    n = 0
    mean = zero(x0) ./ 1
    M2 = zero(x0) ./ 1
    delta = zero(x0) ./ 1
    delta2 = zero(x0) ./ 1
    for x in A
        n += 1
        if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
            delta .= x .- mean
            mean .+= delta ./ n
            delta2 .= x .- mean
            M2 .+= delta .* delta2
        else
            delta = x .- mean
            mean += delta ./ n
            delta2 = x .- mean
            M2 += delta .* delta2
        end
    end
    if n < 2
        return NaN
    else
        if bessel
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                M2 .= M2 ./ (n .- 1)
            else
                M2 = M2 ./ (n .- 1)
            end
        else
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                M2 .= M2 ./ n
            else
                M2 = M2 ./ n
            end
        end
        return mean, M2
    end
end

function componentwise_meancov(A, B; bessel = true)
    x0 = first(A)
    y0 = first(B)
    n = 0
    meanx = zero(x0) ./ 1
    meany = zero(y0) ./ 1
    C = zero(x0) ./ 1
    dx = zero(x0) ./ 1
    for (x, y) in zip(A, B)
        n += 1
        if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
            dx .= x .- meanx
            meanx .+= dx ./ n
            meany .+= (y .- meany) ./ n
            C .+= dx .* (y .- meany)
        else
            dx = x .- meanx
            meanx += dx ./ n
            meany += (y .- meany) ./ n
            C += dx .* (y .- meany)
        end
    end
    if n < 2
        return NaN
    else
        if bessel
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                C .= C ./ (n .- 1)
            else
                C = C ./ (n .- 1)
            end
        else
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                C .= C ./ n
            else
                C = C ./ n
            end
        end
        return meanx, meany, C
    end
end

function componentwise_meancor(A, B; bessel = true)
    mx, my, cov = componentwise_meancov(A, B; bessel = bessel)
    mx, vx = componentwise_meanvar(A; bessel = bessel)
    my, vy = componentwise_meanvar(B; bessel = bessel)
    if vx isa AbstractArray
        vx .= sqrt.(vx)
        vy .= sqrt.(vy)
    else
        vx = sqrt.(vx)
        vy = sqrt.(vy)
    end
    return mx, my, cov ./ (vx .* vy)
end

function componentwise_weighted_meancov(A, B, W; weight_type = :reliability)
    x0 = first(A)
    y0 = first(B)
    w0 = first(W)
    n = 0
    meanx = zero(x0) ./ 1
    meany = zero(y0) ./ 1
    wsum = zero(w0)
    wsum2 = zero(w0)
    C = zero(x0) ./ 1
    dx = zero(x0) ./ 1
    for (x, y, w) in zip(A, B, W)
        n += 1
        if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
            wsum .+= w
            wsum2 .+= w .* w
            dx .= x .- meanx
            meanx .+= (w ./ wsum) .* dx
            meany .+= (w ./ wsum) .* (y .- meany)
            C .+= w .* dx .* (y .- meany)
        else
            wsum += w
            wsum2 += w .* w
            dx = x .- meanx
            meanx += (w ./ wsum) .* dx
            meany += (w ./ wsum) .* (y .- meany)
            C += w .* dx .* (y .- meany)
        end
    end
    if n < 2
        return NaN
    else
        if weight_type == :population
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                C .= C ./ wsum
            else
                C = C ./ wsum
            end
        elseif weight_type == :reliability
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                C .= C ./ (wsum .- wsum2 ./ wsum)
            else
                C = C ./ (wsum .- wsum2 ./ wsum)
            end
        elseif weight_type == :frequency
            if x0 isa AbstractArray && !(x0 isa StaticArraysCore.SArray)
                C .= C ./ (wsum .- 1)
            else
                C = C ./ (wsum .- 1)
            end
        else
            error("The weight_type which was chosen is not allowed.")
        end
        return meanx, meany, C
    end
end

export get_timestep,
    get_timepoint,
    componentwise_vectors_timestep, componentwise_vectors_timepoint

export componentwise_mean, componentwise_meanvar

export timestep_mean, timestep_median, timestep_quantile, timestep_meanvar,
    timestep_meancov, timestep_meancor, timestep_weighted_meancov

export timeseries_steps_mean, timeseries_steps_median, timeseries_steps_quantile,
    timeseries_steps_meanvar, timeseries_steps_meancov,
    timeseries_steps_meancor, timeseries_steps_weighted_meancov

export timepoint_mean, timepoint_median, timepoint_quantile,
    timepoint_meanvar, timepoint_meancov,
    timepoint_meancor, timepoint_weighted_meancov

export timeseries_point_mean, timeseries_point_median, timeseries_point_quantile,
    timeseries_point_meanvar, timeseries_point_meancov,
    timeseries_point_meancor, timeseries_point_weighted_meancov

end
