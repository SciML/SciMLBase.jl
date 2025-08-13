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

SymbolicIndexingInterface.symbolic_container(A::AbstractSolution) = A.prob
SymbolicIndexingInterface.parameter_values(A::AbstractSolution) = parameter_values(A.prob)
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
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::AbstractNoTimeSolution, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
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

"""
    $(TYPEDSIGNATURES)

Given the first element in a timeseries solution, return an `AbstractArray` of
indices that can be plotted as continuous variables. This is useful for systems
that store auxiliary variables in the state vector which are not meant to be
used for plotting.
"""
plottable_indices(x::AbstractArray) = 1:length(x)
plottable_indices(x::Number) = 1


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
    plot_vecs,
    labels = solplot_vecs_and_labels(dims, vars, plott, sol,
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
