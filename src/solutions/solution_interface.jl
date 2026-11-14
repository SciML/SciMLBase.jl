### Abstract Interface
const AllObserved = RecursiveArrayTools.AllObserved

# No Time Solution : Forward to `A.u`
Base.getindex(A::AbstractNoTimeSolution) = A.u[]
Base.getindex(A::AbstractNoTimeSolution, i::Int) = A.u[i]
Base.getindex(A::AbstractNoTimeSolution, I::Vararg{Int, N}) where {N} = A.u[I...]
Base.getindex(A::AbstractNoTimeSolution, I::AbstractArray{Int}) = A.u[I]
Base.getindex(A::AbstractNoTimeSolution, I::CartesianIndex) = A.u[I]
Base.getindex(A::AbstractNoTimeSolution, I::Colon) = A.u[I]
Base.getindex(A::AbstractTimeseriesSolution, I::AbstractArray{Int}) = solution_slice(A, I)
Base.setindex!(A::AbstractNoTimeSolution, v, i::Int) = (A.u[i] = v)
Base.setindex!(A::AbstractNoTimeSolution, v, I::Vararg{Int, N}) where {N} = (A.u[I] = v)
Base.size(A::AbstractNoTimeSolution) = size(A.u)

function Base.show(io::IO, m::MIME"text/plain", A::AbstractNoTimeSolution)
    if hasfield(typeof(A), :retcode)
        println(io, string("retcode: ", A.retcode))
    end
    print(io, "u: ")
    return show(io, m, A.u)
end

# For augmenting system information to enable symbol based indexing of interpolated solutions
function augment(
        A::DiffEqArray{T, N, Q, B}, sol::AbstractODESolution;
        discretes = nothing
    ) where {T, N, Q, B}
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol; discretes,
        interp = sol.interp, dense = sol.dense)
end

# SymbolicIndexingInterface.jl
const AbstractSolution = Union{AbstractTimeseriesSolution, AbstractNoTimeSolution}

Base.@propagate_inbounds function Base.getproperty(A::AbstractSolution, sym::Symbol)
    if sym === :ps
        return ParameterIndexingProxy(A)
    else
        return getfield(A, sym)
    end
end

SymbolicIndexingInterface.symbolic_container(A::AbstractSolution) = A.prob
SymbolicIndexingInterface.parameter_values(A::AbstractSolution) = parameter_values(A.prob)
function SymbolicIndexingInterface.parameter_values(A::AbstractSolution, i)
    return parameter_values(parameter_values(A), i)
end

SymbolicIndexingInterface.symbolic_container(A::AbstractPDESolution) = A.disc_data.pdesys

SymbolicIndexingInterface.is_independent_variable(::AbstractNoTimeSolution, sym) = false

SymbolicIndexingInterface.independent_variable_symbols(::AbstractNoTimeSolution) = []

SymbolicIndexingInterface.is_time_dependent(::AbstractTimeseriesSolution) = true

SymbolicIndexingInterface.is_time_dependent(::AbstractNoTimeSolution) = false

# TODO make this nontrivial once dynamic state selection works
SymbolicIndexingInterface.constant_structure(::AbstractSolution) = true
SymbolicIndexingInterface.state_values(A::AbstractNoTimeSolution) = A.u

function get_saved_subsystem(sol::T) where {T <: AbstractTimeseriesSolution}
    return hasfield(T, :saved_subsystem) ? sol.saved_subsystem : nothing
end

for fn in [
        is_timeseries_parameter, timeseries_parameter_index,
        with_updated_parameter_timeseries_values, get_saveable_values,
    ]
    fname = nameof(fn)
    mod = parentmodule(fn)

    @eval function $(mod).$(fname)(sol::AbstractTimeseriesSolution, args...)
        ss = get_saved_subsystem(sol)
        return if ss === nothing
            $(fn)(symbolic_container(sol), args...)
        else
            $(fn)(SavedSubsystemWithFallback(ss, symbolic_container(sol)), args...)
        end
    end
end

function SymbolicIndexingInterface.state_values(sol::AbstractTimeseriesSolution, i)
    ss = get_saved_subsystem(sol)
    ss === nothing && return sol.u[i]

    original = state_values(sol.prob)
    saved = sol.u[i]
    if !(saved isa AbstractArray)
        saved = [saved]
    end
    idxs = similar(saved, eltype(keys(ss.state_map)))
    for (k, v) in ss.state_map
        idxs[v] = k
    end
    replaced = remake_buffer(sol, original, idxs, saved)
    return replaced
end

function SymbolicIndexingInterface.state_values(sol::AbstractTimeseriesSolution)
    ss = get_saved_subsystem(sol)
    ss === nothing && return sol.u
    return map(Base.Fix1(state_values, sol), eachindex(sol.u))
end

# Ambiguity resolution
function SymbolicIndexingInterface.state_values(sol::AbstractTimeseriesSolution, ::Colon)
    return state_values(sol)
end

Base.@propagate_inbounds function Base.getindex(A::AbstractTimeseriesSolution, ::Colon)
    return A.u[:]
end

Base.@propagate_inbounds function Base.getindex(A::AbstractNoTimeSolution, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, sym::Union{AbstractArray, Tuple}
    )
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
            is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, ::SymbolicIndexingInterface.SolvedVariables
    )
    return getindex(A, variable_symbols(A))
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, ::SymbolicIndexingInterface.AllVariables
    )
    return getindex(A, all_variable_symbols(A))
end

function observed(A::AbstractTimeseriesSolution, sym, i::Int)
    return getobserved(A)(sym, A[i], parameter_values(A), A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::AbstractArray{Int})
    return getobserved(A).((sym,), A.u[i], (parameter_values(A),), A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::Colon)
    return getobserved(A).((sym,), A.u, (parameter_values(A),), A.t)
end

function observed(A::AbstractNoTimeSolution, sym)
    return getobserved(A)(sym, A.u, parameter_values(A))
end

## AbstractTimeseriesSolution Interface

function Base.summary(io::IO, A::AbstractTimeseriesSolution)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(A)),
        no_color, " with uType ",
        type_color, eltype(A.u),
        no_color, " and tType ",
        type_color, eltype(A.t), no_color
    )
end

function Base.show(io::IO, m::MIME"text/plain", A::AbstractTimeseriesSolution)
    println(io, string("retcode: ", A.retcode))
    println(io, string("Interpolation: "), interp_summary(A.interp))
    print(io, "t: ")
    show(io, m, A.t)
    println(io)
    print(io, "u: ")
    return show(io, m, A.u)
end


function Base.iterate(sol::AbstractTimeseriesSolution, state = 0)
    state >= length(sol) && return nothing
    state += 1
    return (solution_new_tslocation(sol, state), state)
end

function Base.show(io::IO, m::MIME"text/plain", A::AbstractPDESolution)
    println(io, string("retcode: ", A.retcode))
    print(io, "t: ")
    show(io, m, A.t)
    println(io)
    print(io, "u: ")
    return show(io, m, A.u)
end

function isdenseplot(sol)
    return (sol.dense || sol.prob isa AbstractDiscreteProblem) &&
        !(sol isa AbstractRODESolution) &&
        !(
        hasfield(typeof(sol), :interp) &&
            sol.interp isa SensitivityInterpolation
    )
end

@recipe function f(
        sol::AbstractTimeseriesSolution;
        plot_analytic = false,
        denseplot = isdenseplot(sol),
        plotdensity = min(
            Int(1.0e5),
            sol.tslocation == 0 ?
                (
                    sol.prob isa AbstractDiscreteProblem ?
                    max(1000, 100 * length(sol)) :
                    max(1000, 10 * length(sol))
                ) :
                1000 * sol.tslocation
        ), plotat = nothing,
        tspan = nothing,
        vars = nothing, idxs = nothing
    )
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true
        )
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    if plot_analytic && (sol.u_analytic === nothing)
        throw(ArgumentError("No analytic solution was found but `plot_analytic` was set to `true`."))
    end

    idxs = idxs === nothing ? plottable_indices(sol.u[1]) : idxs
    idxs_input = idxs
    if !(idxs isa Union{Tuple, AbstractArray})
        vars = interpret_vars([idxs], sol)
    else
        vars = interpret_vars(idxs, sol)
    end
    disc_vars = Tuple[]
    cont_vars = Tuple[]
    for var in vars
        tsidxs = union(
            var[2] === 0 ? () : get_all_timeseries_indexes(sol, var[2]),
            get_all_timeseries_indexes(sol, var[3])
        )
        if ContinuousTimeseries() in tsidxs || isempty(tsidxs)
            push!(cont_vars, var)
        else
            push!(disc_vars, (var..., only(tsidxs)))
        end
    end
    idxs = identity.(cont_vars)
    vars = identity.(cont_vars)
    tdir = sign(sol.t[end] - sol.t[1])
    xflip --> tdir < 0
    seriestype --> :path

    @series begin
        if idxs isa Union{AbstractArray, Tuple} && isempty(idxs)
            label --> nothing
            ([], [])
        else
            tscale = get(plotattributes, :xscale, :identity)
            plot_vecs, labels = solution_plot_arrays(
                sol, plot_analytic, denseplot, plotdensity, tspan, vars, tscale, plotat
            )

            if idxs_input isa Tuple && vars[1][1] === DEFAULT_PLOT_FUNC
                xguide --> plot_var_label(sol, vars[1][2])
                yguide --> plot_var_label(sol, vars[1][3])
                if length(vars[1]) > 3
                    zguide --> plot_var_label(sol, vars[1][4])
                end
            end

            if all(
                    x -> (x[2] isa Integer && x[2] == 0) ||
                        isequal(x[2], getindepsym_defaultt(sol)), vars
                )
                xguide --> "$(getindepsym_defaultt(sol))"
                if tspan === nothing
                    if tdir > 0
                        xlims --> (sol.t[1], sol.t[end])
                    else
                        xlims --> (sol.t[end], sol.t[1])
                    end
                else
                    xlims --> (tspan[1], tspan[end])
                end
            end

            label --> reshape(labels, 1, length(labels))
            (plot_vecs...,)
        end
    end

    for (func, xvar, yvar, tsidx) in disc_vars
        partition = sol.discretes[tsidx]
        ts = current_time(partition)
        if tspan !== nothing
            tstart = searchsortedfirst(ts, tspan[1])
            tend = searchsortedlast(ts, tspan[2])
            if tstart == lastindex(ts) + 1 || tend == firstindex(ts) - 1
                continue
            end
        else
            tstart = firstindex(ts)
            tend = lastindex(ts)
        end
        ts = ts[tstart:tend]

        if symbolic_type(xvar) == NotSymbolic() && xvar == 0
            xvar = only(independent_variable_symbols(sol))
        end
        xvals = sol(ts; idxs = xvar).u
        yvals = getp(sol, yvar)(sol, tstart:tend)
        tmpvals = map(func, xvals, yvals)
        xvals = getindex.(tmpvals, 1)
        yvals = getindex.(tmpvals, 2)
        @series begin
            seriestype := :line
            linestyle --> :dash
            markershape --> :o
            markersize --> repeat([2, 0], length(ts) - 1)
            markeralpha --> repeat([1, 0], length(ts) - 1)
            label --> string(hasname(yvar) ? getname(yvar) : yvar)

            x = vec([xvals[1:(end - 1)]'; xvals[2:end]'])
            y = repeat(yvals, inner = 2)[1:(end - 1)]
            x, y
        end
    end
end

"""
    plot_var_label(sol, x)

Axis and legend label of the plotting variable `x` of `sol`: `"t"` for index `0`,
the variable name for symbolic or named indices, and `"u[x]"` otherwise.
"""
function plot_var_label(sol, x)
    symbolic_type(x) == NotSymbolic() || return string(x)
    x isa Integer || return string(x)
    x == 0 && return "t"
    varsyms = variable_symbols(sol)
    return 0 < x <= length(varsyms) ? string(varsyms[x]) : "u[$x]"
end

"""
    plot_times(sol, denseplot, plotdensity, tspan, tscale, plotat)

Time points at which the plot recipes sample `sol`. Without an explicit `tspan`
or `plotat`, an animation frame (`sol.tslocation != 0`) is sampled only up to
`sol.t[sol.tslocation]`.
"""
function plot_times(sol, denseplot, plotdensity, tspan, tscale, plotat)
    plotat === nothing || return plotat
    if tspan === nothing
        start_idx = 1
        end_idx = sol.tslocation == 0 ? lastindex(sol.t) : sol.tslocation
    else
        start_idx = searchsortedfirst(sol.t, tspan[1])
        end_idx = searchsortedlast(sol.t, tspan[end])
    end
    denseplot || return sol.t[start_idx:end_idx]
    tstart, tstop = if sol isa AbstractAnalyticalSolution
        sol.prob.tspan[1], sol.prob.tspan[end]
    elseif tspan === nothing
        sol.t[start_idx], sol.t[end_idx]
    else
        tspan[1], tspan[end]
    end
    if tscale in (:ln, :log10, :log2)
        return exp10.(range(log10(tstart), log10(tstop); length = plotdensity))
    end
    return collect(range(tstart, tstop; length = plotdensity))
end

"""
    solution_plot_arrays(sol, plot_analytic, denseplot, plotdensity, tspan, vars, tscale, plotat)

Plot coordinates and labels of the interpreted plotting variables `vars` of `sol`,
as `(plot_vecs, labels)` with one column per curve. With `plot_analytic`, the
analytic solution of `sol.prob` follows the numerical curves, labelled `"True ..."`.
"""
function solution_plot_arrays(
        sol, plot_analytic, denseplot, plotdensity, tspan, vars, tscale, plotat
    )
    plott = plot_times(sol, denseplot, plotdensity, tspan, tscale, plotat)
    plot_vecs, labels = diffeq_to_arrays(
        sol, denseplot, plotdensity, tspan, vars, tscale, plott
    )
    plot_analytic || return plot_vecs, labels
    analytic_vecs, analytic_labels = analytic_plot_arrays(sol, vars, plott)
    return [hcat(a, b) for (a, b) in zip(plot_vecs, analytic_vecs)],
        vcat(labels, analytic_labels)
end

function analytic_plot_arrays(sol, vars, plott)
    f = sol.prob.f isa Tuple ? sol.prob.f[1] : sol.prob.f
    analytic = [f.analytic(sol.prob.u0, sol.prob.p, t) for t in plott]
    plot_vecs = []
    labels = String[]
    for x in vars
        tmp = []
        strs = String[]
        for j in 2:length(x)
            if (x[j] isa Integer && x[j] == 0) || isequal(x[j], getindepsym_defaultt(sol))
                push!(tmp, plott)
                push!(strs, "t")
            else
                push!(tmp, [u isa Number ? u : u[x[j]] for u in analytic])
                push!(strs, plot_var_label(sol, x[j]))
            end
        end
        tmp = map(x[1], tmp...)
        tmp = tuple((getindex.(tmp, i) for i in eachindex(tmp[1]))...)
        for i in eachindex(tmp)
            length(plot_vecs) < i && push!(plot_vecs, [])
            push!(plot_vecs[i], tmp[i])
        end
        series_labels = add_labels!(String[], x, length(tmp), sol, strs)
        push!(labels, "True " * only(series_labels))
    end
    return [hcat(x...) for x in plot_vecs], labels
end
