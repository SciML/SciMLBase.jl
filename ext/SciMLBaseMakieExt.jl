module SciMLBaseMakieExt

using SciMLBase
using Makie

import Makie.SpecApi as S

function ensure_plottrait(PT::Type, arg, desired_plottrait_type::Type)
    if !(Makie.conversion_trait(PT, arg) isa desired_plottrait_type)
        error("""
              `Makie.convert_arguments` for the plot type $PT and its conversion trait $(Makie.conversion_trait(PT, arg)) was unsuccessful.

              There is a recipe for the given arguments and the `$desired_plottrait_type` trait, however.

              The signature that could not be converted was:
              ::$(string(typeof(arg)))

              Makie needs to convert all plot input arguments to types that can be consumed by the backends (typically Arrays with Float32 elements).
              You can define a method for `Makie.convert_arguments` (a type recipe) for these types or their supertypes to make this set of arguments convertible (See http://docs.makie.org/stable/documentation/recipes/index.html).

              Alternatively, you can define `Makie.convert_single_argument` for single arguments which have types that are unknown to Makie but which can be converted to known types and fed back to the conversion pipeline.
              """)
    end
end

# ## `AbstractTimeseriesSolution` recipe

# First, we define the standard plot type for timeseries solutions.
# This is a `Lines` plot, since timeseries solutions are typically
# plotted as lines.
# In the recipe, we use the Makie PlotSpec API to override
Makie.plottype(sol::SciMLBase.AbstractTimeseriesSolution) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, sol::SciMLBase.AbstractTimeseriesSolution)
    (:plot_analytic, :denseplot, :plotdensity, :plotat, :tspan, :tscale, :vars, :idxs)
end

function Makie.convert_arguments(PT::Type{<:Plot},
        sol::SciMLBase.AbstractTimeseriesSolution;
        plot_analytic = false,
        denseplot = Makie.automatic,
        plotdensity = Makie.automatic,
        plotat = nothing,
        tspan = nothing,
        tscale = :identity,
        vars = nothing,
        idxs = nothing)

    # First, this recipe is specifically only for timeseries solutions.
    # This means that the recipe only applies to `PointBased` plot types.
    #
    # However, since we need to know the plot type (lines, scatter, scatterlines, etc)
    # which the user passed in, we have to take the plot type directly instead of using
    # the trait system!
    #
    # So, we first check if the plot type is PointBased, and if not, we throw the standard
    # Makie error message for convert_arguments - just at a different place.
    # TODO: this is a bit of a hack, but of course one can define specific dispatches elsewhere...
    ensure_plottrait(PT, sol, Makie.PointBased)

    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    # Extract indices (this is SOP)

    idxs = idxs === nothing ? (1:length(sol.u[1])) : idxs

    if !(idxs isa Union{Tuple, AbstractArray})
        vars = SciMLBase.interpret_vars([idxs], sol)
    else
        vars = SciMLBase.interpret_vars(idxs, sol)
    end

    # Translate automatics inside the function, for ease of use + passthrough from higher 
    # level recipes
    if denseplot isa Makie.Automatic
        denseplot = (sol.dense ||
                     typeof(sol.prob) <: SciMLBase.AbstractDiscreteProblem) &&
                    !(typeof(sol) <: SciMLBase.AbstractRODESolution) &&
                    !(hasfield(typeof(sol), :interp) &&
                      typeof(sol.interp) <: SciMLBase.SensitivityInterpolation)
    end

    if plotdensity isa Makie.Automatic
        plotdensity = min(Int(1e5),
            sol.tslocation == 0 ?
            (sol.prob isa SciMLBase.AbstractDiscreteProblem ?
             max(1000, 100 * length(sol)) :
             max(1000, 10 * length(sol))) :
            1000 * sol.tslocation)
    end

    # Originally, in the Plots recipe, `tscale` 
    # read the axis's `xscale` attribute, and passed that on.
    #
    # However, we can't lift attributes from the axis since
    # we don't have access to it here, so we allow the user
    # to pass in a `tscale` attribute directly.
    @assert tscale isa Symbol "`tscale` if passed in to `Makie.plot` must be a Symbol, got a $(typeof(tscale))"

    # Convert the solution to arrays - this is the hard part!
    plot_vecs, labels = SciMLBase.diffeq_to_arrays(sol, plot_analytic, denseplot,
        plotdensity, tspan, vars, tscale, plotat)

    # We must convert from plot Type to symbol here, for plotspec use
    # since PlotSpecs are defined based on symbols
    plot_type_sym = Makie.plotsym(PT) # TODO this is still a bit hacky!

    # Finally, generate a vector of PlotSpecs (one per variable pair)
    # TODO: broadcast across all input attributes, or figure out how to 
    # allow customizable colors/labels/etc if required
    makie_plotspecs = if length(plot_vecs) == 2
        map((x, y, label) -> PlotSpec(plot_type_sym, Point2f.(x, y); label),
            eachcol(plot_vecs[1]),
            eachcol(plot_vecs[2]),
            labels)
    elseif length(plot_vecs) == 3
        map((x, y, z, label) -> PlotSpec(plot_type_sym, Point3f.(x, y, z); label),
            eachcol(plot_vecs[1]),
            eachcol(plot_vecs[2]),
            eachcol(plot_vecs[3]),
            labels)
    end

    return makie_plotspecs
end

# ## Integrator recipes

# This will be very similar to the timeseries solution recipe, but without some of the conveniences.

Makie.plottype(integrator::SciMLBase.DEIntegrator) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, integrator::SciMLBase.DEIntegrator)
    (:plot_analytic, :denseplot, :plotdensity, :vars, :idxs)
end

function Makie.convert_arguments(PT::Type{<:Makie.Plot},
        integrator::SciMLBase.DEIntegrator;
        plot_analytic = false,
        denseplot = Makie.automatic,
        plotdensity = 10,
        vars = nothing,
        idxs = nothing)
    ensure_plottrait(PT, integrator, Makie.PointBased)

    # Interpret keyword arguments
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    if denseplot isa Makie.Automatic
        denseplot = (integrator.opts.calck ||
                     integrator isa AbstractSDEIntegrator) &&
                    integrator.iter > 0
    end

    # Begin deconstructing the integrator

    int_vars = SciMLBase.interpret_vars(idxs, integrator.sol)

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
                   isequal(x[j], SciMLBase.getindepsym_defaultt(integrator))
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
                push!(strs, String(SciMLBase.getname(varsyms[x[j]])))
            elseif SciMLBase.hasname(x[j])
                push!(strs, String(SciMLBase.getname(x[j])))
            else
                push!(strs, "u[$(x[j])]")
            end
        end
        SciMLBase.add_labels!(labels, x, dims, integrator.sol, strs)
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
            SciMLBase.add_labels!(labels, x, dims, integrator.sol, strs)
        end
    end

    # @show plot_vecs

    plot_type_sym = Makie.plotsym(PT)

    return if denseplot
        [Makie.PlotSpec(plot_type_sym,
             Point2f.(plot_vecs[1][idx], plot_vecs[2][idx]);
             label,
             color = Makie.Cycled(idx))
         for (idx, label) in zip(1:length(plot_vecs[1]), labels)]
    else
        [S.Scatter([Point2f(plot_vecs[1][idx], plot_vecs[2][idx])]; label)
         for (idx, label) in zip(1:length(plot_vecs[1]), labels)]
    end
end

# ## Ensemble recipes

# Again, we first define the "ideal" plot type for ensemble solutions.
Makie.plottype(sol::SciMLBase.AbstractEnsembleSolution) = Makie.Lines

# We also define the attributes that are used by the ensemble solution recipe:
function Makie.used_attributes(::Type{<:Plot}, sol::SciMLBase.AbstractEnsembleSolution)
    (:trajectories,
        :plot_analytic,
        :denseplot,
        :plotdensity,
        :plotat,
        :tspan,
        :tscale,
        :vars,
        :idxs)
end

function Makie.convert_arguments(PT::Type{<:Lines},
        sim::SciMLBase.AbstractEnsembleSolution;
        trajectories = eachindex(sim),
        plot_analytic = false,
        denseplot = Makie.automatic,
        plotdensity = Makie.automatic,
        plotat = nothing,
        tspan = nothing,
        tscale = :identity,
        vars = nothing,
        idxs = nothing)

    # First, we check if the plot type is PointBased, and if not, we throw the standard
    # Makie error message for convert_arguments - just at a different place.
    ensure_plottrait(PT, sim, Makie.PointBased)

    @assert length(trajectories)>0 "No trajectories to plot"
    @assert length(sim.u)>0 "No solutions to plot"

    plot_type_sym = Makie.plotsym(PT)

    mp = [PlotSpec(plot_type_sym,
              sim.u[i];
              plot_analytic,
              denseplot,
              plotdensity,
              plotat,
              tspan,
              tscale,
              idxs) for i in trajectories]

    # Main.Infiltrator.@infiltrate

    return mp
end

# ## EnsembleSummary recipes

# WARNING: EXPERIMENTAL!

Makie.plottype(sim::SciMLBase.EnsembleSummary) = Makie.Lines

function Makie.used_attributes(::Type{<:Lines}, sim::SciMLBase.EnsembleSummary)
    (:trajectories,
        :error_style,
        :ci_type,
        :plot_analytic,
        :denseplot,
        :plotdensity,
        :plotat,
        :tspan,
        :tscale,
        :vars,
        :idxs)
end

# TODO: should `error_style` be Makie plot types instead?  I.e. `Band`, `Errorbar`, etc
function Makie.convert_arguments(::Type{<:Lines},
        sim::SciMLBase.EnsembleSummary;
        trajectories = sim.u.u[1] isa AbstractArray ? eachindex(sim.u.u[1]) :
                       1,
        error_style = :ribbon, ci_type = :quantile,
        kwargs...)
    if ci_type == :SEM
        if sim.u.u[1] isa AbstractArray
            u = SciMLBase.vecarr_to_vectors(sim.u)
        else
            u = [sim.u.u]
        end
        if sim.u.u[1] isa AbstractArray
            ci_low = SciMLBase.vecarr_to_vectors(VectorOfArray([sqrt.(sim.v.u[i] /
                                                                      sim.num_monte) .*
                                                                1.96
                                                                for i in 1:length(sim.v)]))
            ci_high = ci_low
        else
            ci_low = [[sqrt(sim.v.u[i] / length(sim.num_monte)) .* 1.96
                       for i in 1:length(sim.v)]]
            ci_high = ci_low
        end
    elseif ci_type == :quantile
        if sim.med.u[1] isa AbstractArray
            u = SciMLBase.vecarr_to_vectors(sim.med)
        else
            u = [sim.med.u]
        end
        if sim.u.u[1] isa AbstractArray
            ci_low = u - SciMLBase.vecarr_to_vectors(sim.qlow)
            ci_high = SciMLBase.vecarr_to_vectors(sim.qhigh) - u
        else
            ci_low = [u[1] - sim.qlow.u]
            ci_high = [sim.qhigh.u - u[1]]
        end
    else
        error("ci_type choice not valid. Must be `:SEM` or `:quantile`")
    end

    makie_plotlist = Makie.PlotSpec[]

    for (count, idx) in enumerate(trajectories)
        push!(makie_plotlist,
            S.Lines(sim.t, u[idx]; color = Makie.Cycled(count), label = "u[$idx]"))
        if error_style == :ribbon
            push!(makie_plotlist,
                S.Band(sim.t,
                    u[idx] .- ci_low[idx],
                    u[idx] .+ ci_high[idx];
                    color = Makie.Cycled(count),
                    alpha = 0.1))
        elseif error_style == :bars
            push!(makie_plotlist,
                S.Errorbars(sim.t, u[idx], ci_low[idx], ci_high[idx]))
        elseif error_style == :none
            nothing
        else
            error("error_style `$error_style` not recognized")
        end
    end

    return makie_plotlist
end

end
