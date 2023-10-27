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
function augment(A::DiffEqArray{T, N, Q, B}, sol::AbstractODESolution) where {T, N, Q, B}
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol)
end

# SymbolicIndexingInterface.jl
const AbstractSolution = Union{AbstractTimeseriesSolution,AbstractNoTimeSolution}
SymbolicIndexingInterface.symbolic_container(A::AbstractSolution) = A.prob.f
SymbolicIndexingInterface.parameter_values(A::AbstractSolution) = A.prob.p

SymbolicIndexingInterface.is_independent_variable(::AbstractNoTimeSolution, sym) = false

SymbolicIndexingInterface.independent_variable_symbols(::AbstractNoTimeSolution) = []

function SymbolicIndexingInterface.is_observed(A::AbstractSolution, sym)
    return !is_variable(A, sym) && !is_parameter(A, sym) && !is_independent_variable(A, sym) && symbolic_type(sym) == ScalarSymbolic()
end

function SymbolicIndexingInterface.observed(A::AbstractTimeseriesSolution, sym)
    (u, p, t) -> getobserved(A)(sym, u, p, t)
end

function SymbolicIndexingInterface.observed(A::AbstractNoTimeSolution, sym)
    (u, p) -> getobserved(A)(sym, u, p)
end

for soltype in [AbstractTimeseriesSolution, AbstractNoTimeSolution]
    @eval function SymbolicIndexingInterface.observed(A::$(soltype), sym::Symbol)
        has_sys(A.prob.f) || error("Cannot use observed without system")
        return SymbolicIndexingInterface.observed(A, getproperty(A.prob.f.sys, sym))
    end
end

SymbolicIndexingInterface.is_time_dependent(::AbstractTimeseriesSolution) = true

SymbolicIndexingInterface.is_time_dependent(::AbstractNoTimeSolution) = false

# TODO make this nontrivial once dynamic state selection works
SymbolicIndexingInterface.constant_structure(::AbstractSolution) = true

Base.@propagate_inbounds function Base.getindex(A::AbstractTimeseriesSolution, ::Colon)
    return A.u[:]
end

Base.@propagate_inbounds function Base.getindex(A::AbstractNoTimeSolution, sym)
    if symbolic_type(sym) == ScalarSymbolic()
        if is_variable(A, sym)
            return A[variable_index(A, sym)]
        elseif is_parameter(A, sym)
            Base.depwarn("Indexing with parameters is deprecated. Use `getp(sys, $sym)(sol)` for parameter indexing.", :parameter_getindex)
            return getp(A, sym)(A)
        elseif is_observed(A, sym)
            return SymbolicIndexingInterface.observed(A, sym)(A.u, A.prob.p)
        else
            error("Tried to index solution with a Symbol that was not found in the system.")
        end
    elseif symbolic_type(sym) == ArraySymbolic()
        return A[collect(sym)]
    else
        sym isa AbstractArray || error("Invalid indexing of solution")
        return getindex.((A,), sym)
    end
end

function observed(A::AbstractTimeseriesSolution, sym, i::Int)
    getobserved(A)(sym, A[i], A.prob.p, A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::AbstractArray{Int})
    getobserved(A).((sym,), A.u[i], (A.prob.p,), A.t[i])
end

function observed(A::AbstractTimeseriesSolution, sym, i::Colon)
    getobserved(A).((sym,), A.u, (A.prob.p,), A.t)
end

function observed(A::AbstractNoTimeSolution, sym)
    getobserved(A)(sym, A.u, A.prob.p)
end

function observed(A::AbstractOptimizationSolution, sym)
    getobserved(A)(sym, A.u, get_p(A))
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

DEFAULT_PLOT_FUNC(x...) = (x...,)
DEFAULT_PLOT_FUNC(x, y, z) = (x, y, z) # For v0.5.2 bug

@recipe function f(sol::AbstractTimeseriesSolution;
    plot_analytic = false,
    denseplot = (sol.dense ||
                 sol.prob isa AbstractDiscreteProblem) &&
                !(sol isa AbstractRODESolution) &&
                !(hasfield(typeof(sol), :interp) &&
                  sol.interp isa SensitivityInterpolation),
    plotdensity = min(Int(1e5),
        sol.tslocation == 0 ?
        (sol.prob isa AbstractDiscreteProblem ?
         max(1000, 100 * length(sol)) :
         max(1000, 10 * length(sol))) :
        1000 * sol.tslocation),
    tspan = nothing, axis_safety = 0.1,
    vars = nothing, idxs = nothing)
    if vars !== nothing
        Base.depwarn("To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    syms = getsyms(sol)
    if idxs isa Symbol
        int_vars = interpret_vars([idxs], sol, syms)
    else
        int_vars = interpret_vars(idxs, sol, syms)
    end
    strs = cleansyms(syms)

    tscale = get(plotattributes, :xscale, :identity)
    plot_vecs, labels = diffeq_to_arrays(sol, plot_analytic, denseplot,
        plotdensity, tspan, axis_safety,
        idxs, int_vars, tscale, strs)

    tdir = sign(sol.t[end] - sol.t[1])
    xflip --> tdir < 0
    seriestype --> :path

    # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ...
    if idxs isa Tuple && (symbolic_type(idxs[1]) != NotSymbolic() && symbolic_type(idxs[2]) != NotSymbolic())
        val = symbolic_type(int_vars[1][2]) != NotSymbolic() ? String(Symbol(int_vars[1][2])) :
              strs[int_vars[1][2]]
        xguide --> val
        val = symbolic_type(int_vars[1][3]) != NotSymbolic() ? String(Symbol(int_vars[1][3])) :
              strs[int_vars[1][3]]
        yguide --> val
        if length(idxs) > 2
            val = symbolic_type(int_vars[1][4]) != NotSymbolic() ? String(Symbol(int_vars[1][4])) :
                  strs[int_vars[1][4]]
            zguide --> val
        end
    end

    if (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(int_vars, 1))) &&
        getindex.(int_vars, 1) == zeros(length(int_vars))) ||
       (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(int_vars, 2))) &&
        getindex.(int_vars, 2) == zeros(length(int_vars))) ||
       all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(int_vars, 1)) ||
       all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(int_vars, 2))
        xguide --> "$(getindepsym_defaultt(sol))"
    end
    if length(int_vars[1]) >= 3 && ((!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(int_vars, 3))) &&
         getindex.(int_vars, 3) == zeros(length(int_vars))) ||
        all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(int_vars, 3)))
        yguide --> "$(getindepsym_defaultt(sol))"
    end
    if length(int_vars[1]) >= 4 && ((!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(int_vars, 4))) &&
         getindex.(int_vars, 4) == zeros(length(int_vars))) ||
        all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(int_vars, 4)))
        zguide --> "$(getindepsym_defaultt(sol))"
    end

    if (!any(!isequal(NotSymbolic()), symbolic_type.(getindex.(int_vars, 2))) &&
        getindex.(int_vars, 2) == zeros(length(int_vars))) ||
       all(t -> Symbol(t) == getindepsym_defaultt(sol), getindex.(int_vars, 2))
        if tspan === nothing
            if tdir > 0
                xlims --> (sol.t[1], sol.t[end])
            else
                xlims --> (sol.t[end], sol.t[1])
            end
        else
            xlims --> (tspan[1], tspan[end])
        end
    else
        mins = minimum(sol[int_vars[1][2], :])
        maxs = maximum(sol[int_vars[1][2], :])
        for iv in int_vars
            mins = min(mins, minimum(sol[iv[2], :]))
            maxs = max(maxs, maximum(sol[iv[2], :]))
        end
        xlims -->
        ((1 - sign(mins) * axis_safety) * mins, (1 + sign(maxs) * axis_safety) * maxs)
    end

    # Analytical solutions do not save enough information to have a good idea
    # of the axis ahead of time
    # Only set axis for animations
    if sol.tslocation != 0 && !(sol isa AbstractAnalyticalSolution)
        if all(getindex.(int_vars, 1) .== DEFAULT_PLOT_FUNC)
            mins = minimum(sol[int_vars[1][3], :])
            maxs = maximum(sol[int_vars[1][3], :])
            for iv in int_vars
                mins = min(mins, minimum(sol[iv[3], :]))
                maxs = max(maxs, maximum(sol[iv[3], :]))
            end
            ylims -->
            ((1 - sign(mins) * axis_safety) * mins, (1 + sign(maxs) * axis_safety) * maxs)

            if length(int_vars[1]) >= 4
                mins = minimum(sol[int_vars[1][4], :])
                maxs = maximum(sol[int_vars[1][4], :])
                for iv in int_vars
                    mins = min(mins, minimum(sol[iv[4], :]))
                    maxs = max(mins, maximum(sol[iv[4], :]))
                end
                zlims --> ((1 - sign(mins) * axis_safety) * mins,
                    (1 + sign(maxs) * axis_safety) * maxs)
            end
        end
    end

    label --> reshape(labels, 1, length(labels))
    (plot_vecs...,)
end

function diffeq_to_arrays(sol, plot_analytic, denseplot, plotdensity, tspan, axis_safety,
    vars, int_vars, tscale, strs)
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

    if denseplot
        # Generate the points from the plot from dense function
        if tspan === nothing && !(sol isa AbstractAnalyticalSolution)
            plott = collect(densetspacer(sol.t[start_idx], sol.t[end_idx], plotdensity))
        elseif sol isa AbstractAnalyticalSolution
            tspan = sol.prob.tspan
            plott = collect(densetspacer(tspan[1], tspan[end], plotdensity))
        else
            plott = collect(densetspacer(tspan[1], tspan[end], plotdensity))
        end
        plot_timeseries = sol(plott)
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

            plot_timeseries = sol.u[start_idx:end_idx]
            if plot_analytic
                plot_analytic_timeseries = sol.u_analytic[start_idx:end_idx]
            else
                plot_analytic_timeseries = nothing
            end
        end
    end

    dims = length(int_vars[1])
    for var in int_vars
        @assert length(var) == dims
    end
    # Should check that all have the same dims!
    plot_vecs, labels = solplot_vecs_and_labels(dims, int_vars, plot_timeseries, plott, sol,
        plot_analytic, plot_analytic_timeseries,
        strs)
end

function interpret_vars(vars, sol, syms)
    if vars !== nothing && syms !== nothing
        # Do syms conversion
        tmp_vars = []
        for var in vars
            if var isa Union{Tuple, AbstractArray} #eltype(var) <: Symbol # Some kind of iterable
                tmp = []
                for x in var
                    if symbolic_type(x) != NotSymbolic()
                        found = sym_to_index(x, syms)
                        push!(tmp,
                            found == nothing && getindepsym_defaultt(sol) == x ? 0 :
                            something(found, x))
                    else
                        push!(tmp, x)
                    end
                end
                if var isa Tuple
                    var_int = tuple(tmp...)
                else
                    var_int = tmp
                end
            elseif symbolic_type(var) != NotSymbolic()
                found = sym_to_index(var, syms)
                if (var isa Symbol) && has_sys(sol.prob.f)
                    var_int = if found === nothing && getindepsym_defaultt(sol) == var
                        0
                    elseif found !== nothing
                        found
                    elseif is_variable(sol, var)
                        variable_symbols(sol)[variable_index(sol, var)]
                    elseif is_parameter(sol, var)
                        parameter_symbols(sol)[parameter_index(sol, var)]
                    elseif is_independent_variable(sol, var)
                        independent_variable_symbols(sol)[1]
                    else
                        error("Tried to index solution with a Symbol that was not found in the system using `getproperty`.")
                    end 
                else
                    var_int = found == nothing && getindepsym_defaultt(sol) == var ? 0 :
                              something(found, var)
                end
            else
                var_int = var
            end
            push!(tmp_vars, var_int)
        end
        if vars isa Tuple
            vars = tuple(tmp_vars...)
        else
            vars = tmp_vars
        end
    end

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
    lys = []
    for j in 3:dims
        if symbolic_type(x[j]) == NotSymbolic() && x[j] == 0
            push!(lys, "$(getindepsym_defaultt(sol)),")
        elseif symbolic_type(x[j]) != NotSymbolic()
            push!(lys, "$(x[j]),")
        else
            if strs !== nothing
                push!(lys, "$(strs[x[j]]),")
            else
                push!(lys, "u$(x[j]),")
            end
        end
    end
    lys[end] = chop(lys[end]) # Take off the last comma
    if symbolic_type(x[2]) == NotSymbolic() && x[2] == 0 && dims == 3
        # if there are no dependence in syms, then we add "(t)"
        if strs !== nothing && (x[3] isa Int && endswith(strs[x[3]], r"(.*)")) ||
           (symbolic_type(x[3]) != NotSymbolic() && endswith(string(x[3]), r"(.*)"))
            tmp_lab = "$(lys...)"
        else
            tmp_lab = "$(lys...)($(getindepsym_defaultt(sol)))"
        end
    else
        if strs !== nothing && symbolic_type(x[2]) == NotSymbolic() && x[2] != 0
            tmp = strs[x[2]]
            tmp_lab = "($tmp,$(lys...))"
        else
            if symbolic_type(x[2]) == NotSymbolic() && x[2] == 0
                tmp_lab = "($(getindepsym_defaultt(sol)),$(lys...))"
            elseif symbolic_type(x[2]) != NotSymbolic()
                tmp_lab = "($(x[2]),$(lys...))"
            else
                tmp_lab = "(u$(x[2]),$(lys...))"
            end
        end
    end
    if x[1] != DEFAULT_PLOT_FUNC
        push!(labels, "$(x[1])$(tmp_lab)")
    else
        push!(labels, tmp_lab)
    end
    labels
end

function add_analytic_labels!(labels, x, dims, sol, strs)
    lys = []
    for j in 3:dims
        if x[j] == 0 && dims == 3
            push!(lys, "$(getindepsym_defaultt(sol)),")
        else
            if strs !== nothing
                push!(lys, string("True ", strs[x[j]], ","))
            else
                push!(lys, "True u$(x[j]),")
            end
        end
    end
    lys[end] = lys[end][1:(end - 1)] # Take off the last comma
    if x[2] == 0
        tmp_lab = "$(lys...)($(getindepsym_defaultt(sol)))"
    else
        if strs !== nothing
            tmp = string("True ", strs[x[2]])
            tmp_lab = "($tmp,$(lys...))"
        else
            tmp_lab = "(True u$(x[2]),$(lys...))"
        end
    end
    if x[1] != DEFAULT_PLOT_FUNC
        push!(labels, "$(x[1])$(tmp_lab)")
    else
        push!(labels, tmp_lab)
    end
end

function u_n(timeseries::Union{AbstractArray,RecursiveArrayTools.AbstractVectorOfArray}, n::Int, sol, plott, plot_timeseries)
    # Returns the nth variable from the timeseries, t if n == 0
    if n == 0
        return plott
    elseif n == 1 && !(sol[:, 1] isa Union{AbstractArray, ArrayPartition})
        return timeseries
    else
        tmp = Vector{eltype(sol[1])}(undef, length(plot_timeseries))
        for j in 1:length(plot_timeseries)
            tmp[j] = plot_timeseries[j][n]
        end
        return tmp
    end
end

function u_n(timeseries::Union{AbstractArray,RecursiveArrayTools.AbstractVectorOfArray}, sym, sol, plott, plot_timeseries)
    @assert symbolic_type(sym) != NotSymbolic()
    if getindepsym_defaultt(sol) == Symbol(sym)
        return plott
    else
        getobserved(sol).((sym,), eachcol(timeseries), (sol.prob.p,), plott)
    end
end

function solplot_vecs_and_labels(dims, vars, plot_timeseries, plott, sol, plot_analytic,
    plot_analytic_timeseries, strs)
    plot_vecs = []
    labels = String[]
    for x in vars
        tmp = []
        for j in 2:dims
            push!(tmp, u_n(plot_timeseries, x[j], sol, plott, plot_timeseries))
        end

        f = x[1]

        tmp = f.(tmp...)

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
        @assert sol.u_analytic !== Nothing
        analytic_plot_vecs = []
        for x in vars
            tmp = []
            for j in 2:dims
                push!(tmp,
                    u_n(plot_analytic_timeseries, x[j], sol, plott,
                        plot_analytic_timeseries))
            end
            f = x[1]
            tmp = f.(tmp...)
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
