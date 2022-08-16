"""
$(TYPEDEF)
"""
struct PDESolution{T, N, uType, Sol, DType, ivtype, Disc, tType, P, A, IType}  <: AbstractPDESolution{T, N, S}
    u::uType
    original_sol::Sol
    errors::DType
    t::tType
    ivs::ivType
    disc_data::Disc
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    retcode::Symbol
end

function PDESolution(sol::ODESolution{T}, disc_data::MOLMetadata) where T
    odesys = disc_data.odesys
    metadata = disc_data.metadata

    states = odesys.states

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace

    indexof(sym,syms) = findfirst(isequal(sym),syms)
    umap = Dict(map(discretespace.ū) do u
        let discu = discretespace.discvars[u]
            solu = map(CartesianIndices(discu)) do I
                i = indexof(discu[I], states)
                if i !== nothing
                    sol.u[i]
                else
                    sol[discu[I]]
                end
            end
            out = [zeros(T, size(discu)) for i in 1:length(sol.t)]
            for I in CartesianIndices(discu), i in 1:length(sol.t)
                out[i][I] = solu[I][i]
            end
            u => out
        end
    end)
    return PDESolution{T, length(discretespace.ū), typeof(umap), typeof(sol), typeof(sol.errors),
                       typeof(disc_data), typeof(pdesys.ivs), typeof(sol.t), typeof(sol.prob),
                       typeof(sol.alg), typeof(sol.interp)}(umap, sol, sol.errors, sol.t,
                                                            pdesys.ivs, disc_data, sol.prob,
                                                            sol.alg, sol.interp, sol.dense,
                                                            sol.tslocation, sol.retcode)
end

wrap_sol(sol::ODESolution, disc_data::MOLMetadata) = PDESolution(sol, disc_data)
