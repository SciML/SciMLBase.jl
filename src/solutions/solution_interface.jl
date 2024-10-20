### Abstract Interface
const AllObserved = RecursiveArrayTools.AllObserved

# No Time Solution : Forward to `A.u`
Base.getindex(A::AbstractNoTimeSolution) = A.u[]
Base.getindex(A::AbstractNoTimeSolution, i::Int) = A.u[i]
Base.getindex(A::AbstractNoTimeSolution, I::Vararg{Int, N}) where {N} = A.u[I]
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
    show(io, m, A.u)
end

# For augmenting system information to enable symbol based indexing of interpolated solutions
function augment(A::DiffEqArray{T, N, Q, B}, sol::AbstractODESolution;
        discretes = nothing) where {T, N, Q, B}
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol; discretes)
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

SymbolicIndexingInterface.symbolic_container(A::AbstractSolution) = A.prob.f
SymbolicIndexingInterface.parameter_values(A::AbstractSolution) = A.prob.p
function SymbolicIndexingInterface.parameter_values(A::AbstractSolution, i)
    parameter_values(parameter_values(A), i)
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
    hasfield(T, :saved_subsystem) ? sol.saved_subsystem : nothing
end

for fn in [is_timeseries_parameter, timeseries_parameter_index,
    with_updated_parameter_timeseries_values, get_saveable_values]
    fname = nameof(fn)
    mod = parentmodule(fn)

    @eval function $(mod).$(fname)(sol::AbstractTimeseriesSolution, args...)
        ss = get_saved_subsystem(sol)
        if ss === nothing
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
    state_values(sol)
end

Base.@propagate_inbounds function Base.getindex(A::AbstractTimeseriesSolution, ::Colon)
    return A.u[:]
end

Base.@propagate_inbounds function Base.getindex(A::AbstractNoTimeSolution, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getu(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getu(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, ::SymbolicIndexingInterface.SolvedVariables)
    return getindex(A, variable_symbols(A))
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, ::SymbolicIndexingInterface.AllVariables)
    return getindex(A, all_variable_symbols(A))
end

function observed(A::AbstractTimeseriesSolution, sym, i::Int)
    getobserved(A)(sym, A[i], parameter_values(A), A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::AbstractArray{Int})
    getobserved(A).((sym,), A.u[i], (parameter_values(A),), A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::Colon)
    getobserved(A).((sym,), A.u, (parameter_values(A),), A.t)
end

function observed(A::AbstractNoTimeSolution, sym)
    getobserved(A)(sym, A.u, parameter_values(A))
end

## AbstractTimeseriesSolution Interface

function Base.summary(io::IO, A::AbstractTimeseriesSolution)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(A)),
        no_color, " with uType ",
        type_color, eltype(A.u),
        no_color, " and tType ",
        type_color, eltype(A.t), no_color)
end

function Base.show(io::IO, m::MIME"text/plain", A::AbstractTimeseriesSolution)
    println(io, string("retcode: ", A.retcode))
    println(io, string("Interpolation: "), interp_summary(A.interp))
    print(io, "t: ")
    show(io, m, A.t)
    println(io)
    print(io, "u: ")
    show(io, m, A.u)
end

RecursiveArrayTools.tuples(sol::AbstractTimeseriesSolution) = tuple.(sol.u, sol.t)

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
    show(io, m, A.u)
end

DEFAULT_PLOT_FUNC(x, y) = (x, y)
DEFAULT_PLOT_FUNC(x, y, z) = (x, y, z) # For v0.5.2 bug

function isdenseplot(sol)
    (sol.dense || sol.prob isa AbstractDiscreteProblem) &&
        !(sol isa AbstractRODESolution) &&
        !(hasfield(typeof(sol), :interp) &&
          sol.interp isa SensitivityInterpolation)
end

@recipe function f(sol::AbstractTimeseriesSolution;
        plot_analytic = false,
        denseplot = isdenseplot(sol),
        plotdensity = min(Int(1e5),
            sol.tslocation == 0 ?
            (sol.prob isa AbstractDiscreteProblem ?
             max(1000, 100 * length(sol)) :
             max(1000, 10 * length(sol))) :
            1000 * sol.tslocation), plotat = nothing,
        tspan = nothing,
        vars = nothing, idxs = nothing)
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    if plot_analytic && (sol.u_analytic === nothing)
        throw(ArgumentError("No analytic solution was found but `plot_analytic` was set to `true`."))
    end

    idxs = idxs === nothing ? (1:length(sol.u[1])) : idxs
    if !(idxs isa Union{Tuple, AbstractArray})
        vars = interpret_vars([idxs], sol)
    else
        vars = interpret_vars(idxs, sol)
    end
    disc_vars = Tuple[]
    cont_vars = Tuple[]
    for var in vars
        tsidxs = union(get_all_timeseries_indexes(sol, var[2]),
            get_all_timeseries_indexes(sol, var[3]))
        if ContinuousTimeseries() in tsidxs
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
            plot_vecs, labels = diffeq_to_arrays(sol, plot_analytic, denseplot,
                plotdensity, tspan, vars, tscale, plotat)

            # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ...
            if idxs isa Tuple && vars[1][1] === DEFAULT_PLOT_FUNC
                val = hasname(vars[1][2]) ? String(getname(vars[1][2])) : vars[1][2]
                if val isa Integer
                    if val == 0
                        val = "t"
                    else
                        val = "u[$val]"
                    end
                end
                xguide --> val
                val = hasname(vars[1][3]) ? String(getname(vars[1][3])) : vars[1][3]
                if val isa Integer
                    if val == 0
                        val = "t"
                    else
                        val = "u[$val]"
                    end
                end
                yguide --> val
                if length(idxs) > 2
                    val = hasname(vars[1][4]) ? String(getname(vars[1][4])) : vars[1][4]
                    if val isa Integer
                        if val == 0
                            val = "t"
                        else
                            val = "u[$val]"
                        end
                    end
                    zguide --> val
                end
            end

            if (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 1))) &&
                getindex.(vars, 1) == zeros(length(vars))) ||
               (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 2))) &&
                getindex.(vars, 2) == zeros(length(vars))) ||
               all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(vars, 1)) ||
               all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(vars, 2))
                xguide --> "$(getindepsym_defaultt(sol))"
            end
            if length(vars[1]) >= 3 &&
               ((!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 3))) &&
                 getindex.(vars, 3) == zeros(length(vars))) ||
                all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(vars, 3)))
                yguide --> "$(getindepsym_defaultt(sol))"
            end
            if length(vars[1]) >= 4 &&
               ((!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 4))) &&
                 getindex.(vars, 4) == zeros(length(vars))) ||
                all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(vars, 4)))
                zguide --> "$(getindepsym_defaultt(sol))"
            end

            if (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 2))) &&
                getindex.(vars, 2) == zeros(length(vars))) ||
               all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(vars, 2))
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
        # xvals = getu(sol, xvar)(sol, tstart:tend)
        yvals = getp(sol, yvar)(sol, tstart:tend)
        tmpvals = map(func, xvals, yvals)
        xvals = getindex.(tmpvals, 1)
        yvals = getindex.(tmpvals, 2)
        # Scatterplot of points
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

function diffeq_to_arrays(sol, plot_analytic, denseplot, plotdensity, tspan,
        vars, tscale, plotat)
    if tspan === nothing
        if sol.tslocation == 0
            end_idx = length(sol)
        else
            end_idx = sol.tslocation
        end
        start_idx = 1
    else
        start_idx = searchsortedfirst(sol.t, tspan[1])
        end_idx = searchsortedlast(sol.t, tspan[end])
    end

    # determine type of spacing for plott
    densetspacer = if tscale in [:ln, :log10, :log2]
        (start, stop, n) -> exp10.(range(log10(start), stop = log10(stop), length = n))
    else
        (start, stop, n) -> range(start; stop = stop, length = n)
    end

    if plotat !== nothing
        plott = plotat
        plot_analytic_timeseries = nothing
    elseif denseplot
        # Generate the points from the plot from dense function
        if sol isa AbstractAnalyticalSolution
            tspan = sol.prob.tspan
            plott = collect(densetspacer(tspan[1], tspan[end], plotdensity))
        elseif tspan === nothing
            plott = collect(densetspacer(sol.t[start_idx], sol.t[end_idx], plotdensity))
        else
            plott = collect(densetspacer(tspan[1], tspan[end], plotdensity))
        end
        if plot_analytic
            if sol.prob.f isa Tuple
                plot_analytic_timeseries = [sol.prob.f[1].analytic(sol.prob.u0, sol.prob.p,
                                                t) for t in plott]
            else
                plot_analytic_timeseries = [sol.prob.f.analytic(sol.prob.u0, sol.prob.p, t)
                                            for t in plott]
            end
        else
            plot_analytic_timeseries = nothing
        end
    else
        # Plot for sparse output: use the timeseries itself
        if sol.tslocation == 0
            plott = sol.t
            plot_timeseries = DiffEqArray(sol.u, sol.t)
            if plot_analytic
                plot_analytic_timeseries = sol.u_analytic
            else
                plot_analytic_timeseries = nothing
            end
        else
            if tspan === nothing
                plott = sol.t[start_idx:end_idx]
            else
                plott = collect(densetspacer(tspan[1], tspan[2], plotdensity))
            end

            if plot_analytic
                plot_analytic_timeseries = sol.u_analytic[start_idx:end_idx]
            else
                plot_analytic_timeseries = nothing
            end
        end
    end

    dims = length(vars[1]) - 1
    for var in vars
        @assert length(var) - 1 == dims
    end
    # Should check that all have the same dims!
    plot_vecs, labels = solplot_vecs_and_labels(dims, vars, plott, sol,
        plot_analytic, plot_analytic_timeseries)
end

function interpret_vars(vars, sol)
    if vars === nothing
        # Default: plot all timeseries
        if sol[:, 1] isa Union{Tuple, AbstractArray}
            vars = collect((DEFAULT_PLOT_FUNC, 0, i) for i in plot_indices(sol[:, 1]))
        else
            vars = [(DEFAULT_PLOT_FUNC, 0, 1)]
        end
    end

    if vars isa Base.Integer
        vars = [(DEFAULT_PLOT_FUNC, 0, vars)]
    end

    if vars isa AbstractArray
        # If list given, its elements should be tuples, or we assume x = time
        tmp = Tuple[]
        for x in vars
            if x isa Tuple
                if x[1] isa Int
                    push!(tmp, tuple(DEFAULT_PLOT_FUNC, x...))
                else
                    push!(tmp, x)
                end
            else
                push!(tmp, (DEFAULT_PLOT_FUNC, 0, x))
            end
        end
        vars = tmp
    end

    if vars isa Tuple
        # If tuple given...
        if vars[end - 1] isa AbstractArray
            if vars[end] isa AbstractArray
                # If both axes are lists we zip (will fail if different lengths)
                vars = collect(zip([DEFAULT_PLOT_FUNC for i in eachindex(vars[end - 1])],
                    vars[end - 1], vars[end]))
            else
                # Just the x axis is a list
                vars = [(DEFAULT_PLOT_FUNC, x, vars[end]) for x in vars[end - 1]]
            end
        else
            if vars[2] isa AbstractArray
                # Just the y axis is a list
                vars = [(DEFAULT_PLOT_FUNC, vars[end - 1], y) for y in vars[end]]
            else
                # Both axes are numbers
                if vars[1] isa Int || symbolic_type(vars[1]) != NotSymbolic()
                    vars = [tuple(DEFAULT_PLOT_FUNC, vars...)]
                else
                    vars = [vars]
                end
            end
        end
    end

    # Here `vars` should be a list of tuples (x, y).
    @assert(typeof(vars)<:AbstractArray)
    @assert(eltype(vars)<:Tuple)
    vars
end

function add_labels!(labels, x, dims, sol, strs)
    if ((x[2] isa Integer && x[2] == 0) || isequal(x[2], getindepsym_defaultt(sol))) &&
       dims == 2
        push!(labels, strs[end])
    elseif x[1] !== DEFAULT_PLOT_FUNC
        push!(labels, "f($(join(strs, ',')))")
    else
        push!(labels, "($(join(strs, ',')))")
    end
    labels
end

function add_analytic_labels!(labels, x, dims, sol, strs)
    if ((x[2] isa Integer && x[2] == 0) || isequal(x[2], getindepsym_defaultt(sol))) &&
       dims == 2
        push!(labels, "True $(strs[end])")
    elseif x[1] !== DEFAULT_PLOT_FUNC
        push!(labels, "True f($(join(strs, ',')))")
    else
        push!(labels, "True ($(join(strs, ',')))")
    end
    labels
end

function solplot_vecs_and_labels(dims, vars, plott, sol, plot_analytic,
        plot_analytic_timeseries)
    plot_vecs = []
    labels = String[]
    varsyms = variable_symbols(sol)
    batch_symbolic_vars = []
    for x in vars
        for j in 2:length(x)
            if (x[j] isa Integer && x[j] == 0) || isequal(x[j], getindepsym_defaultt(sol))
            else
                push!(batch_symbolic_vars, x[j])
            end
        end
    end
    batch_symbolic_vars = identity.(batch_symbolic_vars)
    indexed_solution = sol(plott; idxs = batch_symbolic_vars)
    idxx = 0
    for x in vars
        tmp = []
        strs = String[]
        for j in 2:length(x)
            if (x[j] isa Integer && x[j] == 0) || isequal(x[j], getindepsym_defaultt(sol))
                push!(tmp, plott)
                push!(strs, "t")
            else
                idxx += 1
                push!(tmp, indexed_solution[idxx, :])
                if !isempty(varsyms) && x[j] isa Integer
                    push!(strs, String(getname(varsyms[x[j]])))
                elseif hasname(x[j])
                    push!(strs, String(getname(x[j])))
                else
                    push!(strs, "u[$(x[j])]")
                end
            end
        end

        f = x[1]

        tmp = map(f, tmp...)

        tmp = tuple((getindex.(tmp, i) for i in eachindex(tmp[1]))...)
        for i in eachindex(tmp)
            if length(plot_vecs) < i
                push!(plot_vecs, [])
            end
            push!(plot_vecs[i], tmp[i])
        end
        add_labels!(labels, x, dims, sol, strs)
    end

    if plot_analytic
        for x in vars
            tmp = []
            strs = String[]
            for j in 2:length(x)
                if (x[j] isa Integer && x[j] == 0)
                    push!(tmp, plott)
                    push!(strs, "t")
                elseif isequal(x[j], getindepsym_defaultt(sol))
                    push!(tmp, plott)
                    push!(strs, String(getname(x[j])))
                elseif x[j] == 1 && !(sol[:, 1] isa Union{AbstractArray, ArrayPartition})
                    push!(tmp, plot_analytic_timeseries)
                    if !isempty(varsyms) && x[j] isa Integer
                        push!(strs, String(getname(varsyms[x[j]])))
                    elseif hasname(x[j])
                        push!(strs, String(getname(x[j])))
                    else
                        push!(strs, "u[$(x[j])]")
                    end
                else
                    _tmp = Vector{eltype(sol.u[1])}(undef, length(plot_analytic_timeseries))
                    for n in 1:length(plot_analytic_timeseries)
                        _tmp[n] = plot_analytic_timeseries[n][x[j]]
                    end
                    push!(tmp, _tmp)
                    if !isempty(varsyms) && x[j] isa Integer
                        push!(strs, String(getname(varsyms[x[j]])))
                    elseif hasname(x[j])
                        push!(strs, String(getname(x[j])))
                    else
                        push!(strs, "u[$(x[j])]")
                    end
                end
            end
            f = x[1]
            tmp = map(f, tmp...)
            tmp = tuple((getindex.(tmp, i) for i in eachindex(tmp[1]))...)
            for i in eachindex(tmp)
                push!(plot_vecs[i], tmp[i])
            end
            add_analytic_labels!(labels, x, dims, sol, strs)
        end
    end
    plot_vecs = [hcat(x...) for x in plot_vecs]
    plot_vecs, labels
end

plot_indices(A::AbstractArray) = eachindex(A)
plot_indices(A::ArrayPartition) = eachindex(A)
