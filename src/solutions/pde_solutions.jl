"""
$(TYPEDEF)
"""
struct PDESolution{T, N, uType, Disc, Sol, DType, tType, ivtype, dvType, P, A, IType} <:
       AbstractPDESolution{T, N, S}
    u::uType
    original_sol::Sol
    errors::DType
    t::tType
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

    umap = Dict(map(discretespace.ū) do u
                    let discu = discretespace.discvars[u]
                        solu = map(CartesianIndices(discu)) do I
                            i = sym_to_index(discu[I], odesys)
                            if i !== nothing
                                sol.u[i]
                            else
                                observed(sol, discu[I])
                            end
                        end
                        out = zeros(T, length(sol.t), size(discu)...)
                        for I in CartesianIndices(discu), i in 1:length(sol.t)
                            out[i, I] = solu[I][i]
                        end
                        u => out
                    end
                end)
    interp = Dict(map(keys(umap)) do k
        k => interpolate(umap[k], BSpline(Cubic()))
    end)
    return PDESolution{T, length(discretespace.ū), typeof(umap), typeof(metadata),
                       typeof(sol), typeof(sol.errors), typeof(sol.t), typeof(pdesys.ivs),
                       typeof(pdesys.dvs), typeof(sol.prob), typeof(sol.alg),
                       typeof(interp)}(umap, sol, sol.errors, sol.t, pdesys.ivs,
                                           pdesys.dvs, metadata, sol.prob, sol.alg,
                                           interp, sol.dense, sol.tslocation,
                                           sol.retcode)
end

wrap_sol(sol::ODESolution, disc_data::MOLMetadata) = PDESolution(sol, disc_data)

function (sol::PDESolution{T, N, S, D})(args...; dv = nothing) where {T, N, S, D <: MOLMetadata}
    args = map(enumerate(args)) do (i, arg)
        if arg isa Colon
            if i == 1
                sol.t
            else
                sol.disc_data.discretespace.grid[ivs[i-1]]
            end
        else
            arg
        end
    end
    if dv === nothing
        @assert length(args) == length(sol.ivs) + 1 "Not enough arguments for the number of independent variables (including time), got $(length(args)) expected $(length(sol.ivs) + 1)."
        return map(dvs) do dv
            ivs = arguments(dv) #! Do this without Symbolics
            is = map(ivs) do iv
                i = findfirst(iv, sol.ivs)
                @assert i !== nothing "Independent variable $(iv) in dependent variable $(dv) not found in the solution."
                i + 1
            end

            sol.interp[dv](first(args), args[is]...)
        end

        return map(dv -> sol.interp[dv](args...), sol.dvs)
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

    indepsym = getindepsym(A)
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
        if issymbollike(sym) && indepsym !== nothing && Symbol(sym) == indepsym
            A.t
        elseif issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.disc_data.discretespace.grid[sym]
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

Base.@propagate_inbounds function Base.getindex(A::PDESolution{T, N, S, D}, sym, args...) where {T, N, S, D <: MOLMetadata}
    if issymbollike(sym)
        i = sym_to_index(sym, A.original_sol)
    else
        i = sym
    end

    indepsym = getindepsym(A)
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
        if issymbollike(sym) && indepsym !== nothing && Symbol(sym) == indepsym
            A.t[args...]
        elseif issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.disc_data.discretespace.grid[sym]
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
