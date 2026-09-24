module SciMLBaseRecipesBaseExt

using SciMLBase: SciMLBase
using RecipesBase: RecipesBase, @recipe, @series
using SymbolicIndexingInterface: SymbolicIndexingInterface, ContinuousTimeseries,
    NotSymbolic, current_time, get_all_timeseries_indexes, getname, getp, getsym,
    hasname, independent_variable_symbols, symbolic_type, variable_symbols
using RecursiveArrayTools: VectorOfArray, vecarr_to_vectors

@recipe function f(
        sol::SciMLBase.AbstractTimeseriesSolution;
        plot_analytic = false,
        denseplot = SciMLBase.isdenseplot(sol),
        plotdensity = min(
            Int(1.0e5),
            sol.tslocation == 0 ?
                (
                    sol.prob isa SciMLBase.AbstractDiscreteProblem ?
                    max(1000, 100 * length(sol.t)) :
                    max(1000, 10 * length(sol.t))
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

    idxs = idxs === nothing ? SciMLBase.plottable_indices(sol.u[1]) : idxs
    if !(idxs isa Union{Tuple, AbstractArray})
        vars = SciMLBase.interpret_vars([idxs], sol)
    else
        vars = SciMLBase.interpret_vars(idxs, sol)
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
    # Avoid creating a redundant series when we're only plotting discrete variables
    if !(idxs isa Union{AbstractArray, Tuple} && isempty(idxs)) || isempty(disc_vars)
        @series begin
            if idxs isa Union{AbstractArray, Tuple} && isempty(idxs)
                label --> nothing
                ([], [])
            else
                tscale = get(plotattributes, :xscale, :identity)
                plot_vecs,
                    labels = SciMLBase.diffeq_to_arrays(
                    sol, plot_analytic, denseplot,
                    plotdensity, tspan, vars, tscale, plotat
                )

                # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ...
                if idxs isa Tuple && vars[1][1] === SciMLBase.DEFAULT_PLOT_FUNC
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

                if (
                        !any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 1))) &&
                            getindex.(vars, 1) == zeros(length(vars))
                    ) ||
                        (
                        !any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 2))) &&
                            getindex.(vars, 2) == zeros(length(vars))
                    ) ||
                        all(
                        t -> Symbol(t) == SciMLBase.getindepsym_defaultt(sol),
                        getindex.(vars, 1)
                    ) ||
                        all(
                        t -> Symbol(t) == SciMLBase.getindepsym_defaultt(sol),
                        getindex.(vars, 2)
                    )
                    xguide --> "$(SciMLBase.getindepsym_defaultt(sol))"
                end
                if length(vars[1]) >= 3 &&
                        (
                        (
                            !any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 3))) &&
                                getindex.(vars, 3) == zeros(length(vars))
                        ) ||
                            all(
                            t -> Symbol(t) == SciMLBase.getindepsym_defaultt(sol),
                            getindex.(vars, 3)
                        )
                    )
                    yguide --> "$(SciMLBase.getindepsym_defaultt(sol))"
                end
                if length(vars[1]) >= 4 &&
                        (
                        (
                            !any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 4))) &&
                                getindex.(vars, 4) == zeros(length(vars))
                        ) ||
                            all(
                            t -> Symbol(t) == SciMLBase.getindepsym_defaultt(sol),
                            getindex.(vars, 4)
                        )
                    )
                    zguide --> "$(SciMLBase.getindepsym_defaultt(sol))"
                end

                if (
                        !any(!isequal(NotSymbolic()), symbolic_type.(getindex.(vars, 2))) &&
                            getindex.(vars, 2) == zeros(length(vars))
                    ) ||
                        all(
                        t -> Symbol(t) == SciMLBase.getindepsym_defaultt(sol),
                        getindex.(vars, 2)
                    )
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
    end
    for (func, xvar, yvar, tsidx) in disc_vars
        partition = sol.discretes[tsidx]
        ts = current_time(partition)
        if tspan !== nothing
            tstart, tend = SciMLBase.tspan_indices(ts, tspan)
            if tstart > tend
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
        # xvals = getsym(sol, xvar)(sol, tstart:tend)
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

function integplot_vecs_and_labels(dims, vars, plott, integrator, denseplot)
    varsyms = variable_symbols(integrator)

    batch_symbolic_vars = []
    for x in vars
        for j in 2:length(x)
            SciMLBase.is_independent_variable_index(integrator, x[j]) && continue
            push!(batch_symbolic_vars, x[j])
        end
    end
    batch_symbolic_vars = identity.(batch_symbolic_vars)

    if isempty(batch_symbolic_vars)
        timevals = denseplot ? plott : [integrator.t]
        indexed_values = [[] for _ in timevals]
    elseif denseplot
        timevals = plott
        indexed_values = SciMLBase.symbolic_interpolation(
            integrator, plott, batch_symbolic_vars
        ).u
    else
        timevals = [integrator.t]
        indexed_values = [getsym(integrator, batch_symbolic_vars)(integrator)]
    end

    plot_vecs = []
    labels = String[]
    idxx = 0
    for x in vars
        tmp = []
        strs = String[]
        for j in 2:length(x)
            if SciMLBase.is_independent_variable_index(integrator, x[j])
                push!(tmp, timevals)
                push!(strs, "t")
            else
                idxx += 1
                push!(tmp, [vals[idxx] for vals in indexed_values])
                if !isempty(varsyms) && x[j] isa Integer
                    push!(strs, String(getname(varsyms[x[j]])))
                elseif hasname(x[j])
                    push!(strs, String(getname(x[j])))
                else
                    push!(strs, "u[$(x[j])]")
                end
            end
        end

        tmp = map(x[1], tmp...)
        tmp = tuple((getindex.(tmp, i) for i in eachindex(tmp[1]))...)
        for i in eachindex(tmp)
            if length(plot_vecs) < i
                push!(plot_vecs, [])
            end
            push!(plot_vecs[i], tmp[i])
        end
        SciMLBase.add_labels!(labels, x, dims, integrator, strs)
    end

    return [hcat(x...) for x in plot_vecs], labels
end

@recipe function f(
        integrator::SciMLBase.DEIntegrator;
        denseplot = (
            integrator.opts.calck ||
                integrator isa SciMLBase.AbstractSDEIntegrator
        ) &&
            integrator.iter > 0,
        plotdensity = 10,
        plot_analytic = false, vars = nothing, idxs = nothing
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
    if plot_analytic
        throw(
            ArgumentError(
                "`plot_analytic` is not supported when plotting an integrator. Plot `integrator.sol` instead."
            )
        )
    end

    idxs = idxs === nothing ? SciMLBase.plottable_indices(integrator.u) : idxs
    int_vars = if idxs isa Union{Tuple, AbstractArray}
        SciMLBase.interpret_vars(idxs, integrator.sol)
    else
        SciMLBase.interpret_vars([idxs], integrator.sol)
    end

    plott = if denseplot
        collect(range(integrator.tprev, integrator.t; length = plotdensity))
    else
        nothing
    end

    dims = length(int_vars[1]) - 1
    for var in int_vars
        @assert length(var) - 1 == dims
    end

    plot_vecs,
        labels = integplot_vecs_and_labels(
        dims, int_vars, plott, integrator, denseplot
    )

    xflip --> integrator.tdir < 0

    if denseplot
        seriestype --> :path
    else
        seriestype --> :scatter
    end

    # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ...
    if idxs isa Tuple && idxs[1] isa Symbol && idxs[2] isa Symbol
        xlabel --> idxs[1]
        ylabel --> idxs[2]
        if length(idxs) > 2
            zlabel --> idxs[3]
        end
    end
    if all(x -> SciMLBase.is_independent_variable_index(integrator, x[2]), int_vars)
        xlabel --> "$(SciMLBase.getindepsym_defaultt(integrator))"
    end

    linewidth --> 3
    label --> reshape(labels, 1, length(labels))
    (plot_vecs...,)
end

@recipe function f(
        sim::SciMLBase.AbstractEnsembleSolution;
        zcolors = sim.u isa AbstractArray ? fill(nothing, length(sim.u)) :
            nothing,
        trajectories = eachindex(sim.u)
    )
    for i in trajectories
        size(sim.u[i].u, 1) == 0 && continue
        @series begin
            legend := false
            xlims --> (-Inf, Inf)
            ylims --> (-Inf, Inf)
            zlims --> (-Inf, Inf)
            marker_z --> zcolors[i]
            sim.u[i]
        end
    end
end

@recipe function f(
        sim::SciMLBase.EnsembleSummary;
        idxs = sim.u.u[1] isa AbstractArray ? eachindex(sim.u.u[1]) :
            1,
        error_style = :ribbon, ci_type = :quantile
    )
    if ci_type == :SEM
        if sim.u.u[1] isa AbstractArray
            u = vecarr_to_vectors(sim.u)
        else
            u = [sim.u.u]
        end
        if sim.u.u[1] isa AbstractArray
            ci_low = vecarr_to_vectors(
                VectorOfArray(
                    [
                        sqrt.(sim.v.u[i] / sim.num_monte) .*
                            1.96 for i in 1:length(sim.v)
                    ]
                )
            )
            ci_high = ci_low
        else
            ci_low = [
                [
                    sqrt(sim.v.u[i] / length(sim.num_monte)) .* 1.96
                        for i in 1:length(sim.v)
                ],
            ]
            ci_high = ci_low
        end
    elseif ci_type == :quantile
        if sim.med.u[1] isa AbstractArray
            u = vecarr_to_vectors(sim.med)
        else
            u = [sim.med.u]
        end
        if sim.u.u[1] isa AbstractArray
            ci_low = u - vecarr_to_vectors(sim.qlow)
            ci_high = vecarr_to_vectors(sim.qhigh) - u
        else
            ci_low = [u[1] - sim.qlow.u]
            ci_high = [sim.qhigh.u - u[1]]
        end
    else
        error("ci_type choice not valid. Must be `:SEM` or `:quantile`")
    end
    for i in idxs
        @series begin
            legend --> false
            linewidth --> 3
            fillalpha --> 0.2
            if error_style == :ribbon
                ribbon --> (ci_low[i], ci_high[i])
            elseif error_style == :bars
                yerror --> (ci_low[i], ci_high[i])
            elseif error_style == :none
                nothing
            else
                error("error_style not recognized")
            end
            sim.t, u[i]
        end
    end
end

function SciMLBase.anyeltypedual(
        x::RecipesBase.AbstractPlot,
        ::Type{Val{counter}} = Val{0}
    ) where {counter}
    return Any
end

end
