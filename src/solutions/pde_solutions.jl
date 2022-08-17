"""
$(TYPEDEF)
"""
struct PDESolution{T, N, uType, Disc, Sol, DType, tType, gridType, ivtype, dvType, P, A,
                   IType} <: AbstractPDESolution{T, N, S}
    u::uType
    original_sol::Sol
    errors::DType
    t::tType
    ivgrid::gridType
    ivs::ivType
    dvs::dvType
    disc_data::Disc
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    retcode::Symbol
end

function PDESolution(sol::ODESolution{T}, metadata::MOLMetadata) where {T}
    odesys = sol.prob.f.sys

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace

    ivs = [discretespace.time, discretespace.x̄...]
    ivgrid = (isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)
    # Reshape the solution to flat arrays
    umap = Dict(map(discretespace.ū) do u
                    let discu = discretespace.discvars[u]
                        solu = map(CartesianIndices(discu)) do I
                            i = sym_to_index(discu[I], odesys)
                            # Handle Observed
                            if i !== nothing
                                sol.u[i]
                            else
                                observed(sol, discu[I], :)
                            end
                        end
                        out = zeros(T, length(sol.t), size(discu)...)
                        for I in CartesianIndices(discu)
                            out[:, I] .= solu[I]
                        end
                        u => out
                    end
                end)
    # Build Interpolations
    # TODO: Check if the grid is uniform and use the higher order interpolations that are supported
    interp = Dict(map(keys(umap)) do k
                      args = arguments(k) #! Do without Symbolics
                      nodes = ((map(args) do arg
                                    i = findfirst(arg, ivs)
                                    @assert i !==nothing "Independent variable $arg not found in ivs"
                                    ivgrid[i]
                                end)...)
                      k => interpolate(nodes, umap[k], Gridded(Linear()))
                  end)

    return PDESolution{T, length(discretespace.ū), typeof(umap), typeof(metadata),
                       typeof(sol), typeof(sol.errors), typeof(sol.t), typeof(ivgrid),
                       typeof(ivs), typeof(pdesys.dvs), typeof(sol.prob), typeof(sol.alg),
                       typeof(interp)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
                                       pdesys.dvs, metadata, sol.prob, sol.alg,
                                       interp, sol.dense, sol.tslocation,
                                       sol.retcode)
end

wrap_sol(sol::ODESolution, disc_data::MOLMetadata) = PDESolution(sol, disc_data)

function (sol::PDESolution{T, N, S, D})(args...;
                                        dv = nothing) where {T, N, S, D <: MOLMetadata}
    # Colon reconstructs on gridpoints
    args = map(enumerate(args)) do (i, arg)
        if arg isa Colon
            sol.ivgrid[i]
        else
            arg
        end
    end
    # If no dv is given, return interpolations for every dv
    if dv === nothing
        @assert length(args)==length(sol.ivs) + 1 "Not enough arguments for the number of independent variables (including time), got $(length(args)) expected $(length(sol.ivs) + 1)."
        return map(dvs) do dv
            ivs = arguments(dv) #! Do this without Symbolics
            is = map(ivs) do iv
                i = findfirst(iv, sol.ivs)
                @assert i!==nothing "Independent variable $(iv) in dependent variable $(dv) not found in the solution."
                i
            end

            sol.interp[dv](args[is]...)
        end
    end
    return sol.interp[dv](args...)
end

Base.@propagate_inbounds function Base.getindex(A::PDESolution{T, N, S, D},
                                                sym) where {T, N, S, D <: MOLMetadata}
    if issymbollike(sym) || all(issymbollike, sym)
        if sym isa AbstractArray
            return map(s -> A[s], collect(sym))
        end
        i = sym_to_index(sym, A.original_sol)
    else
        i = sym
    end

    iv = nothing
    dv = nothing
    if i === nothing
        iiv = sym_to_index(sym, A.ivs)
        if iiv !== nothing
            iv = A.ivs[iiv]
        end
        idv = sym_to_index(sym, A.dvs)
        if idv !== nothing
            dv = A.dvs[idv]
        end
        if issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.ivgrid[iiv]
        elseif issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym]
        else
            observed(A.original_sol, sym, :)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, :]
    else
        error("Invalid indexing of solution")
    end
end

Base.@propagate_inbounds function Base.getindex(A::PDESolution{T, N, S, D}, sym,
                                                args...) where {T, N, S, D <: MOLMetadata}
    if issymbollike(sym)
        i = sym_to_index(sym, A.original_sol)
    else
        i = sym
    end

    iv = nothing
    dv = nothing
    if i === nothing
        iiv = sym_to_index(sym, A.ivs)
        if iiv !== nothing
            iv = A.ivs[iiv]
        end
        idv = sym_to_index(sym, A.dvs)
        if idv !== nothing
            dv = A.dvs[idv]
        end
        if issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.ivgrid[iiv]
        elseif issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym][args...]
        else
            observed(A.original_sol, sym, args...)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, args...]
    else
        error("Invalid indexing of solution")
    end
end
