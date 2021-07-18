@adjoint function getindex(VA::ODESolution, i::Int) 
    function ODESolution_getindex_pullback(Δ)
        Δ′ = [ [i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)] for (x, j) in zip(VA.u, 1:length(VA))]
        (Δ′, nothing)
    end
    VA[i], ODESolution_getindex_pullback
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

