module SciMLBaseRecipesBaseExt

using SciMLBase
using RecipesBase
import RecursiveArrayTools

# Need to import the plotting-related functions
import SciMLBase: DEFAULT_PLOT_FUNC, isdenseplot, plottable_indices, interpret_vars, 
                  get_all_timeseries_indexes, ContinuousTimeseries, DiscreteTimeseries,
                  solution_slice, add_labels!, AbstractTimeseriesSolution, AbstractEnsembleSolution,
                  AbstractNoTimeSolution, EnsembleSummary, DEIntegrator, AbstractSDEIntegrator,
                  getindepsym_defaultt, getname, hasname, u_n, AbstractDEAlgorithm

# Recipe for AbstractTimeseriesSolution
@recipe function f(sol::AbstractTimeseriesSolution;
        plot_analytic = false,
        denseplot = isdenseplot(sol),
        plotdensity = min(Int(1e5),
            sol.tslocation == 0 ?
            (sol.prob isa SciMLBase.AbstractDiscreteProblem ?
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

    idxs = idxs === nothing ? plottable_indices(sol.u[1]) : idxs
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
        if ContinuousTimeseries() in tsidxs || isempty(tsidxs)
            push!(cont_vars, var)
        else
            push!(disc_vars, var)
        end
    end

    plot_vecs = []
    labels = []

    # Handle continuous variables
    if !isempty(cont_vars)
        int_vars = cont_vars

        if tspan === nothing
            if plotat === nothing
                if denseplot
                    # Generate the points from the plot from dense function
                    start_idx = sol.tslocation == 0 ? 1 : sol.tslocation
                    end_idx = length(sol.t)
                    plott = collect(range(sol.t[start_idx], sol.t[end_idx]; length = plotdensity))
                    plot_timeseries = sol(plott)
                    if plot_analytic
                        plot_analytic_timeseries = [sol.prob.f.analytic(sol.prob.u0,
                                                        sol.prob.p, t)
                                                    for t in plott]
                    end
                else
                    plot_timeseries = sol.u
                    plott = sol.t
                    if plot_analytic
                        plot_analytic_timeseries = sol.u_analytic
                    end
                end
            else
                plot_timeseries = sol(plotat)
                plott = plotat
                if plot_analytic
                    plot_analytic_timeseries = [sol.prob.f.analytic(sol.prob.u0,
                                                    sol.prob.p, t) for t in plott]
                end
            end
        else
            _tspan = tspan isa Number ? (sol.t[1], tspan) : tspan
            start_idx = findfirst(x -> x >= _tspan[1], sol.t)
            end_idx = findlast(x -> x <= _tspan[2], sol.t)
            if denseplot
                plott = collect(range(_tspan...; length = plotdensity))
                plot_timeseries = sol(plott)
                if plot_analytic
                    plot_analytic_timeseries = [sol.prob.f.analytic(sol.prob.u0,
                                                    sol.prob.p, t) for t in plott]
                end
            else
                if start_idx === nothing
                    start_idx = 1
                end
                if end_idx === nothing
                    end_idx = length(sol.t)
                end
                plott = @view sol.t[start_idx:end_idx]
                plot_timeseries = @view sol.u[start_idx:end_idx]
                if plot_analytic
                    plot_analytic_timeseries = @view sol.u_analytic[start_idx:end_idx]
                end
            end
        end

        dims = length(int_vars[1])
        for var in int_vars
            @assert length(var) == dims
        end
        # Should check that all have the same dims!

        for i in 2:dims
            push!(plot_vecs, [])
        end

        labels = String[]# Array{String, 2}(1, length(int_vars)*(1+plot_analytic))
        strs = String[]
        varsyms = SciMLBase.variable_symbols(sol)

        for x in int_vars
            for j in 2:dims
                if denseplot
                    if (x[j] isa Integer && x[j] == 0) ||
                       isequal(x[j], SciMLBase.getindepsym_defaultt(sol))
                        push!(plot_vecs[j - 1], plott)
                    else
                        # For the dense plotting case, we use getindex on the timeseries
                        if plot_timeseries isa AbstractArray
                            if x[j] isa Integer
                                # Simple integer indexing
                                push!(plot_vecs[j - 1], [u[x[j]] for u in plot_timeseries])
                            else
                                # Symbol indexing
                                push!(plot_vecs[j - 1], Vector(sol(plott; idxs = x[j])))
                            end
                        else
                            # Single value case
                            push!(plot_vecs[j - 1], Vector(sol(plott; idxs = x[j])))
                        end
                    end
                else # just get values
                    if x[j] == 0
                        push!(plot_vecs[j - 1], plott)
                    elseif x[j] == 1 && !(eltype(plot_timeseries) <: AbstractArray)
                        push!(plot_vecs[j - 1], plot_timeseries)
                    else
                        if x[j] isa Integer
                            push!(plot_vecs[j - 1], [u[x[j]] for u in plot_timeseries])
                        else
                            # Symbol indexing
                            push!(plot_vecs[j - 1], [sol(t, idxs = x[j]) for t in plott])
                        end
                    end
                end

                if !isempty(varsyms) && x[j] isa Integer
                    push!(strs, String(getname(varsyms[x[j]])))
                elseif hasname(x[j])
                    push!(strs, String(getname(x[j])))
                else
                    push!(strs, "u[$(x[j])]")
                end
            end
            add_labels!(labels, x, dims, sol, strs)
        end

        if plot_analytic
            for x in int_vars
                for j in 2:dims
                    if denseplot
                        if (x[j] isa Integer && x[j] == 0) ||
                           isequal(x[j], SciMLBase.getindepsym_defaultt(sol))
                            push!(plot_vecs[j - 1], plott)
                        else
                            push!(plot_vecs[j - 1],
                                u_n(plot_analytic_timeseries, x[j], sol, plott,
                                    plot_analytic_timeseries))
                        end
                    else # Just get values
                        if x[j] == 0
                            push!(plot_vecs[j - 1], plott)
                        elseif x[j] == 1 &&
                               !(eltype(plot_analytic_timeseries) <: AbstractArray)
                            push!(plot_vecs[j - 1], plot_analytic_timeseries)
                        else
                            push!(plot_vecs[j - 1],
                                u_n(plot_analytic_timeseries, x[j], sol, plott,
                                    plot_analytic_timeseries))
                        end
                    end
                end
                add_labels!(labels, x, dims, sol, strs)
            end
        end

        xflip --> sol.tdir < 0

        if denseplot
            seriestype --> :path
        else
            seriestype --> :scatter
        end

        # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ..
        if idxs isa Tuple && (typeof(idxs[1]) == Symbol && typeof(idxs[2]) == Symbol)
            xlabel --> idxs[1]
            ylabel --> idxs[2]
            if length(idxs) > 2
                zlabel --> idxs[3]
            end
        end
        if getindex.(int_vars, 1) == zeros(length(int_vars)) ||
           getindex.(int_vars, 2) == zeros(length(int_vars))
            xlabel --> "t"
        end

        linewidth --> 3
        #xtickfont --> font(11)
        #ytickfont --> font(11)
        #legendfont --> font(11)
        #guidefont  --> font(11)
        label --> reshape(labels, 1, length(labels))
        (plot_vecs...,)

    # Handle discrete variables
    elseif !isempty(disc_vars)
        int_vars = disc_vars

        if sol.tslocation != 0
            start_idx = sol.tslocation
        else
            start_idx = 1
        end

        if tspan === nothing
            end_idx = length(sol.t)
        else
            _tspan = tspan isa Number ? (sol.t[1], tspan) : tspan
            end_idx = findlast(x -> x <= _tspan[2], sol.t)
            if end_idx === nothing
                end_idx = length(sol.t)
            end
        end

        plott = sol.t[start_idx:end_idx]
        plot_timeseries = sol.u[start_idx:end_idx]

        dims = length(int_vars[1])
        for var in int_vars
            @assert length(var) == dims
        end

        for i in 2:dims
            push!(plot_vecs, [])
        end

        labels = String[]
        strs = String[]
        varsyms = SciMLBase.variable_symbols(sol)

        for x in int_vars
            for j in 2:dims
                if x[j] == 0
                    push!(plot_vecs[j - 1], plott)
                elseif x[j] == 1 && !(eltype(plot_timeseries) <: AbstractArray)
                    push!(plot_vecs[j - 1], plot_timeseries)
                else
                    if x[j] isa Integer
                        push!(plot_vecs[j - 1], [u[x[j]] for u in plot_timeseries])
                    else
                        # Symbol indexing for discrete case  
                        push!(plot_vecs[j - 1], [sol[ti, x[j]] for ti in 1:length(plott)])
                    end
                end

                if !isempty(varsyms) && x[j] isa Integer
                    push!(strs, String(getname(varsyms[x[j]])))
                elseif hasname(x[j])
                    push!(strs, String(getname(x[j])))
                else
                    push!(strs, "u[$(x[j])]")
                end
            end
            add_labels!(labels, x, dims, sol, strs)
        end

        seriestype --> :steppost
        if getindex.(int_vars, 1) == zeros(length(int_vars)) ||
           getindex.(int_vars, 2) == zeros(length(int_vars))
            xlabel --> "t"
        end

        linewidth --> 3
        label --> reshape(labels, 1, length(labels))
        (plot_vecs...,)
    end
end

# Recipe for AbstractEnsembleSolution
@recipe function f(sim::AbstractEnsembleSolution; idxs = nothing, 
                   summarize = true, error_style = :ribbon, ci_type = :quantile, linealpha = 0.4, zorder = 1)

    if idxs === nothing
        if sim.u[1] isa SciMLBase.AbstractTimeseriesSolution
            idxs = 1:length(sim.u[1].u[1])
        else
            idxs = 1:length(sim.u[1])
        end
    end

    if !(idxs isa Union{Tuple, AbstractArray})
        idxs = [idxs]
    end

    if summarize
        summary = EnsembleSummary(sim; quantiles = [0.05, 0.95])
        if error_style == :ribbon
            ribbon --> (summary.qlow[:, idxs], summary.qhigh[:, idxs])
        elseif error_style == :bars
            yerror --> (summary.qlow[:, idxs], summary.qhigh[:, idxs])
        end
        summary.t, summary.med[:, idxs]
    else
        alpha --> linealpha
        # Plot all trajectories
        for i in eachindex(sim.u)
            @series begin
                if sim.u[i] isa SciMLBase.AbstractTimeseriesSolution
                    idxs --> idxs
                    sim.u[i]
                else
                    # For non-timeseries solutions
                    sim.u[i][idxs]
                end
            end
        end
    end
end

# Recipe for EnsembleSummary
@recipe function f(sim::EnsembleSummary; idxs = nothing, error_style = :ribbon)
    if idxs === nothing
        idxs = 1:size(sim.med, 2)
    end

    if !(idxs isa Union{Tuple, AbstractArray})
        idxs = [idxs]
    end

    if error_style == :ribbon
        ribbon --> (sim.qlow[:, idxs], sim.qhigh[:, idxs])
    elseif error_style == :bars
        yerror --> (sim.qlow[:, idxs], sim.qhigh[:, idxs])
    end
    sim.t, sim.med[:, idxs]
end

# Recipe for DEIntegrator
@recipe function f(integrator::DEIntegrator;
        denseplot = (integrator.opts.calck ||
                     integrator isa AbstractSDEIntegrator) &&
                    integrator.iter > 0,
        plotdensity = 10,
        plot_analytic = false, vars = nothing, idxs = nothing)
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    int_vars = interpret_vars(idxs, integrator.sol)

    if denseplot
        # Generate the points from the plot from dense function
        plott = collect(range(integrator.tprev, integrator.t; length = plotdensity))
        if plot_analytic
            plot_analytic_timeseries = [integrator.sol.prob.f.analytic(
                                            integrator.sol.prob.u0,
                                            integrator.sol.prob.p,
                                            t) for t in plott]
        end
    else
        plott = nothing
    end

    dims = length(int_vars[1])
    for var in int_vars
        @assert length(var) == dims
    end
    # Should check that all have the same dims!

    plot_vecs = []
    for i in 2:dims
        push!(plot_vecs, [])
    end

    labels = String[]# Array{String, 2}(1, length(int_vars)*(1+plot_analytic))
    strs = String[]
    varsyms = SciMLBase.variable_symbols(integrator)

    for x in int_vars
        for j in 2:dims
            if denseplot
                if (x[j] isa Integer && x[j] == 0) ||
                   isequal(x[j], getindepsym_defaultt(integrator))
                    push!(plot_vecs[j - 1], plott)
                else
                    push!(plot_vecs[j - 1], Vector(integrator(plott; idxs = x[j])))
                end
            else # just get values
                if x[j] == 0
                    push!(plot_vecs[j - 1], integrator.t)
                elseif x[j] == 1 && !(integrator.u isa AbstractArray)
                    push!(plot_vecs[j - 1], integrator.u)
                else
                    push!(plot_vecs[j - 1], integrator.u[x[j]])
                end
            end

            if !isempty(varsyms) && x[j] isa Integer
                push!(strs, String(getname(varsyms[x[j]])))
            elseif hasname(x[j])
                push!(strs, String(getname(x[j])))
            else
                push!(strs, "u[$(x[j])]")
            end
        end
        add_labels!(labels, x, dims, integrator.sol, strs)
    end

    if plot_analytic
        for x in int_vars
            for j in 1:dims
                if denseplot
                    push!(plot_vecs[j],
                        u_n(plot_timeseries, x[j], sol, plott, plot_timeseries))
                else # Just get values
                    if x[j] == 0
                        push!(plot_vecs[j], integrator.t)
                    elseif x[j] == 1 && !(integrator.u isa AbstractArray)
                        push!(plot_vecs[j],
                            integrator.sol.prob.f(Val{:analytic}, integrator.t,
                                integrator.sol[1]))
                    else
                        push!(plot_vecs[j],
                            integrator.sol.prob.f(Val{:analytic}, integrator.t,
                                integrator.sol[1])[x[j]])
                    end
                end
            end
            add_labels!(labels, x, dims, integrator.sol, strs)
        end
    end

    xflip --> integrator.tdir < 0

    if denseplot
        seriestype --> :path
    else
        seriestype --> :scatter
    end

    # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ..
    if idxs isa Tuple && (typeof(idxs[1]) == Symbol && typeof(idxs[2]) == Symbol)
        xlabel --> idxs[1]
        ylabel --> idxs[2]
        if length(idxs) > 2
            zlabel --> idxs[3]
        end
    end
    if getindex.(int_vars, 1) == zeros(length(int_vars)) ||
       getindex.(int_vars, 2) == zeros(length(int_vars))
        xlabel --> "t"
    end

    linewidth --> 3
    #xtickfont --> font(11)
    #ytickfont --> font(11)
    #legendfont --> font(11)
    #guidefont  --> font(11)
    label --> reshape(labels, 1, length(labels))
    (plot_vecs...,)
end

end