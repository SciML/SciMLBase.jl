@adjoint function getindex(VA::ODESolution, sym, j::Int) 
    function ODESolution_getindex_pullback(Δ)
        i = issymbollike(sym) ? sym_to_index(sym, VA) : sym
        if i === nothing
            getter = getobserved(VA)
            grz = pullback(getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)] 
            dp = grz[3] # pullback for p
            dprob = remake(VA.prob, p = dp)
            T = eltype(eltype(VA.u))
            N = length(VA.prob.p)
            Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, Nothing, Nothing,
                             typeof(dprob), Nothing, Nothing, Nothing}(du, nothing,
                              nothing, nothing, nothing, dprob, nothing, nothing, 
                              VA.dense, 0, nothing, VA.retcode)
            (Δ′, nothing, nothing)
        else
            Δ′ = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] : zero(VA.u[1]) for m in 1:length(VA.u)]
            (Δ′, nothing, nothing)
        end
    end
    VA[sym, j], ODESolution_getindex_pullback
end

@adjoint function getindex(VA::ODESolution, sym) 
    function ODESolution_getindex_pullback(Δ)
        i = issymbollike(sym) ? sym_to_index(sym, VA) : sym
        if i === nothing
            throw("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated.")
        else
            Δ′ = [ [i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)] for (x, j) in zip(VA.u, 1:length(VA))]
            (Δ′, nothing)
        end
    end
    VA[sym], ODESolution_getindex_pullback
end
  