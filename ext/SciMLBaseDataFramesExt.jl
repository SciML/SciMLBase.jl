module SciMLBaseDataFramesExt
using SciMLBase, DataFrames

function DataFrames.DataFrame(sol::EnsembleSolution, idxs::AbstractVector)
    @assert allequal(getproperty.(sol.u, :t)) "solutions must have shared timesteps"
    data = sol[:, idxs]
    i_max, j_max, _ = size(data)
    v = ["t"=>sol[1].t]
    for (i, s) in enumerate(sol.u)
        for idx in idxs
            push!(v, string("sol ", i, ": ", idx)=>s[idx])
        end
    end
    DataFrame(v)
end

function DataFrames.DataFrame(sol::ODESolution, idxs::AbstractVector)
    @assert allequal(getproperty.(sol.u, :t)) "solutions must have shared timesteps"
    data = sol[:, idxs]
    i_max, j_max, _ = size(data)
    v = ["t"=>sol[1].t]
    for idx in idxs
        push!(v, string("sol ", i, ": ", idx)=>s[idx])
    end
    DataFrame(v)
end

end
