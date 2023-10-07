module SciMLBaseZygoteExt

using Zygote: pullback
using ZygoteRules: @adjoint
using SciMLBase: ODESolution, issymbollike, sym_to_index, remake, getobserved

# This method resolves the ambiguity with the pullback defined in
# RecursiveArrayToolsZygoteExt
# https://github.com/SciML/RecursiveArrayTools.jl/blob/d06ecb856f43bc5e37cbaf50e5f63c578bf3f1bd/ext/RecursiveArrayToolsZygoteExt.jl#L67
@adjoint function getindex(VA::ODESolution, i::Int, j::Int)
    function ODESolution_getindex_pullback(Δ)
        du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
              zero(VA.u[1]) for m in 1:length(VA.u)]
        dp = zero(VA.prob.p)
        dprob = remake(VA.prob, p = dp)
        du, dprob
        T = eltype(eltype(VA.u))
        N = length(VA.prob.p)
        Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, typeof(VA.t),
            typeof(VA.k), typeof(dprob), typeof(VA.alg), typeof(VA.interp),
            typeof(VA.destats), typeof(VA.alg_choice)}(du, nothing, nothing,
            VA.t, VA.k, dprob, VA.alg, VA.interp, VA.dense, 0, VA.destats,
            VA.alg_choice, VA.retcode)
        (Δ′, nothing, nothing)
    end
    VA[i, j], ODESolution_getindex_pullback
end

@adjoint function getindex(VA::ODESolution, sym, j::Int)
    function ODESolution_getindex_pullback(Δ)
        i = issymbollike(sym) ? sym_to_index(sym, VA) : sym
        du, dprob = if i === nothing
            getter = getobserved(VA)
            grz = pullback(getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)]
            dp = grz[3] # pullback for p
            dprob = remake(VA.prob, p = dp)
            du, dprob
        else
            du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
                  zero(VA.u[1]) for m in 1:length(VA.u)]
            dp = zero(VA.prob.p)
            dprob = remake(VA.prob, p = dp)
            du, dprob
        end
        T = eltype(eltype(VA.u))
        N = length(VA.prob.p)
        Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, typeof(VA.t),
            typeof(VA.k), typeof(dprob), typeof(VA.alg), typeof(VA.interp),
            typeof(VA.destats), typeof(VA.alg_choice)}(du, nothing, nothing,
            VA.t, VA.k, dprob, VA.alg, VA.interp, VA.dense, 0, VA.destats,
            VA.alg_choice, VA.retcode)
        (Δ′, nothing, nothing)
    end
    VA[sym, j], ODESolution_getindex_pullback
end

end
