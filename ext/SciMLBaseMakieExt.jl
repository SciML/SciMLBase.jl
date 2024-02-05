module SciMLBaseMakieExt

using SciMLBase
using Makie

import Makie.SpecApi as S

# ## `AbstractTimeseriesSolution` recipe

# First, we define the standard plot type for timeseries solutions.
# This is a `Lines` plot, since timeseries solutions are typically
# plotted as lines.
# In the recipe, we use the Makie PlotSpec API to override
Makie.plottype(sol::SciMLBase.AbstractTimeseriesSolution) = Makie.Lines

Makie.used_attributes(::Type{<: Plot}, sol::SciMLBase.AbstractTimeseriesSolution) = (:plot_analytic, :denseplot, :plotdensity, :plotat, :tspan, :tscale, :vars, :idxs)

function Makie.convert_arguments(
    ::Makie.PointBased, 
    sol::SciMLBase.AbstractTimeseriesSolution;
    plot_analytic = false,
    denseplot = (sol.dense ||
                typeof(sol.prob) <: SciMLBase.AbstractDiscreteProblem) &&
                !(typeof(sol) <: SciMLBase.AbstractRODESolution) &&
                !(hasfield(typeof(sol), :interp) &&
                    typeof(sol.interp) <: SciMLBase.SensitivityInterpolation),
    plotdensity = min(Int(1e5),
                        sol.tslocation == 0 ?
                        (sol.prob isa SciMLBase.AbstractDiscreteProblem ?
                        max(1000, 100 * length(sol)) :
                        max(1000, 10 * length(sol))) :
                        1000 * sol.tslocation),
    plotat = nothing,
    tspan = nothing,
    tscale = :identity,
    vars = nothing, 
    idxs = nothing
    )

    if vars !== nothing
        Base.depwarn("To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
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
    plot_type_sym = Makie.plotsym(Makie.Lines) # TODO FIXME: this is a hack, we want to get the plot type from the user input

    # Finally, generate a vector of PlotSpecs (one per variable pair)
    # TODO: broadcast across all input attributes, or figure out how to 
    # allow customizable colors/labels/etc if required
    makie_plotspecs = if length(plot_vecs) == 2
        map((x, y, label) -> PlotSpec(plot_type_sym, Point2f.(x, y); label), eachcol(plot_vecs[1]), eachcol(plot_vecs[2]), labels)
    elseif length(plot_vecs) == 3
        map((x, y, z, label) -> PlotSpec(plot_type_sym, Point3f.(x, y, z); label), eachcol(plot_vecs[1]), eachcol(plot_vecs[2]), eachcol(plot_vecs[3]), labels)
    end

    return makie_plotspecs

end

end
