"""
$(TYPEDEF)
"""
abstract type BasicEnsembleAlgorithm <: EnsembleAlgorithm end

"""
$(TYPEDEF)
"""
struct EnsembleThreads <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)
"""
struct EnsembleDistributed <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)
"""
struct EnsembleSplitThreads <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)
"""
struct EnsembleSerial <: BasicEnsembleAlgorithm end

function __solve(prob::AbstractEnsembleProblem,
                 alg::Union{DEAlgorithm, Nothing};
                 kwargs...)
    if alg isa EnsembleAlgorithm
        # Assume DifferentialEquations.jl is being used, so default alg
        ensemblealg = alg
        alg = nothing
    else
        ensemblealg = EnsembleThreads()
    end
    __solve(prob, alg, ensemblealg; kwargs...)
end

@noinline function rerun_warn()
    @warn("output_func should return (out,rerun). See docs for updated details")
end
tighten_container_eltype(u::Vector{Any}) = map(identity, u)
tighten_container_eltype(u) = u

function __solve(prob::AbstractEnsembleProblem,
                 alg::Union{DEAlgorithm, Nothing},
                 ensemblealg::BasicEnsembleAlgorithm;
                 trajectories, batch_size = trajectories,
                 pmap_batch_size = batch_size ÷ 100 > 0 ? batch_size ÷ 100 : 1, kwargs...)
    num_batches = trajectories ÷ batch_size
    num_batches < 1 &&
        error("trajectories ÷ batch_size cannot be less than 1, got $num_batches")
    num_batches * batch_size != trajectories && (num_batches += 1)

    if num_batches == 1 && prob.reduction === DEFAULT_REDUCTION
        elapsed_time = @elapsed u = solve_batch(prob, alg, ensemblealg, 1:trajectories,
                                                pmap_batch_size; kwargs...)
        _u = tighten_container_eltype(u)
        return EnsembleSolution(_u, elapsed_time, true)
    end

    converged::Bool = false
    elapsed_time = @elapsed begin
        i = 1
        II = (batch_size * (i - 1) + 1):(batch_size * i)

        batch_data = solve_batch(prob, alg, ensemblealg, II, pmap_batch_size; kwargs...)

        u = prob.u_init === nothing ? similar(batch_data, 0) : prob.u_init
        u, converged = prob.reduction(u, batch_data, II)
        for i in 2:num_batches
            converged && break
            if i == num_batches
                II = (batch_size * (i - 1) + 1):trajectories
            else
                II = (batch_size * (i - 1) + 1):(batch_size * i)
            end
            batch_data = solve_batch(prob, alg, ensemblealg, II, pmap_batch_size; kwargs...)
            u, converged = prob.reduction(u, batch_data, II)
        end
    end
    _u = tighten_container_eltype(u)

    return EnsembleSolution(_u, elapsed_time, converged)
end

function batch_func(i, prob, alg; kwargs...)
    iter = 1
    _prob = prob.safetycopy ? deepcopy(prob.prob) : prob.prob
    new_prob = prob.prob_func(_prob, i, iter)
    rerun = true
    x = prob.output_func(solve(new_prob, alg; kwargs...), i)
    if !(typeof(x) <: Tuple)
        rerun_warn()
        _x = (x, false)
    else
        _x = x
    end
    rerun = _x[2]
    while rerun
        iter += 1
        _prob = prob.safetycopy ? deepcopy(prob.prob) : prob.prob
        new_prob = prob.prob_func(_prob, i, iter)
        x = prob.output_func(solve(new_prob, alg; kwargs...), i)
        if !(typeof(x) <: Tuple)
            rerun_warn()
            _x = (x, false)
        else
            _x = x
        end
        rerun = _x[2]
    end
    _x[1]
end

function solve_batch(prob, alg, ensemblealg::EnsembleDistributed, II, pmap_batch_size;
                     kwargs...)
    wp = CachingPool(workers())

    # Fix the return type of pmap
    #=
    function f()
      batch_func(1,prob,alg;kwargs...)
    end
    T = Core.Compiler.return_type(f,Tuple{})
    =#

    batch_data = pmap(wp, II, batch_size = pmap_batch_size) do i
        batch_func(i, prob, alg; kwargs...)
    end

    tighten_container_eltype(batch_data)
end

function stable_map!(f, dst::AbstractArray{T}, args::Vararg{Any, K}) where {K, T}
    N = length(dst)
    all(==(Base.oneto(N)), map(eachindex, args)) ||
        throw(ArgumentError("All args must have same axes."))
    @inbounds for i in 1:N
        dst[i] = convert(T, f(map(Base.Fix2(Base.unsafe_getindex, i), args)...))
    end
    return dst
end
function narrowing_map!(f,
                        dst::AbstractArray{T},
                        start::Int,
                        args::Vararg{Any, K}) where {K, T}
    N = length(dst)
    all(==(Base.oneto(N)), map(eachindex, args)) ||
        throw(ArgumentError("All args must have same axes."))
    @inbounds for i in start:N
        xi = f(map(Base.Fix2(Base.unsafe_getindex, i), args)...)
        Ti = typeof(xi)
        if Ti <: T
            dst[i] = xi
        else
            PT = promote_type(Ti, T)
            if PT === T
                dst[i] = convert(T, xi)
            elseif Base.isconcretetype(PT)
                dst_promote = Array{PT}(undef, size(dst))
                copyto!(view(dst_promote, Base.OneTo(i - 1)), view(dst, Base.OneTo(i - 1)))
                dst_promote[i] = xi
                return narrowing_map!(f, dst_promote, i + 1, args...)
            else
                dst_union = Array{Union{T, Ti}}(undef, size(dst))
                copyto!(view(dst_union, Base.OneTo(i - 1)), view(dst, Base.OneTo(i - 1)))
                dst_union[i] = xi
                return narrowing_map!(f, dst_union, i + 1, args...)
            end
        end
    end
    return dst
end

function promote_return(f::F, args...) where {F}
    T = Base.promote_op(f, map(eltype, args)...)
    Base.isconcretetype(T) && return T
    TU = Base.promote_union(T)
    Base.isconcretetype(TU) && return TU
    nothing
end
function responsible_map(f, args::Vararg{AbstractArray, K}) where {K}
    first_arg = first(args)
    T = promote_return(f, args...)
    T === nothing || return stable_map!(f, Array{T}(undef, size(first_arg)), args...)
    x = f(map(first, args)...)
    dst = similar(first_arg, typeof(x))
    @inbounds dst[1] = x
    narrowing_map!(f, dst, 2, args...)
end
function responsible_map(f, II...)
    batch_data = Vector{Core.Compiler.return_type(f, Tuple{typeof.(getindex.(II, 1))...})}(undef,
                                                                                           length(II[1]))
    for i in 1:length(II[1])
        batch_data[i] = f(getindex.(II, i)...)
    end
    batch_data
end

function SciMLBase.solve_batch(prob, alg, ::EnsembleSerial, II, pmap_batch_size; kwargs...)
    batch_data = responsible_map(II) do i
        SciMLBase.batch_func(i, prob, alg; kwargs...)
    end
    SciMLBase.tighten_container_eltype(batch_data)
end

function solve_batch(prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size;
                     kwargs...)
    nthreads = min(Threads.nthreads(), length(II))
    if length(II) == 1 || nthreads == 1
        return solve_batch(prob, alg, EnsembleSerial(), II, pmap_batch_size; kwargs...)
    end

    if typeof(prob.prob) <: AbstractJumpProblem && length(II) != 1
        probs = [deepcopy(prob.prob) for i in 1:nthreads]
    else
        probs = prob.prob
    end

    batch_data = tmap(II) do i
        batch_func(i, prob, alg; kwargs...)
    end
    tighten_container_eltype(batch_data)
end

function tmap(f, args...)
    batch_data = Vector{Core.Compiler.return_type(f, Tuple{typeof.(getindex.(args, 1))...})
                        }(undef, length(args[1]))
    Threads.@threads for i in 1:length(args[1])
        batch_data[i] = f(getindex.(args, i)...)
    end
    batch_data
end

function solve_batch(prob, alg, ::EnsembleSplitThreads, II, pmap_batch_size; kwargs...)
    wp = CachingPool(workers())
    N = nworkers()
    batch_size = length(II) ÷ N

    # Fix the return type of pmap
    #=
    function f()
      i = 1
      I_local = II[(batch_size*(i-1)+1):end]
      solve_batch(prob,alg,EnsembleThreads(),I_local,pmap_batch_size;kwargs...)
    end
    T = Core.Compiler.return_type(f,Tuple{})
    =#

    batch_data = let
        pmap(wp, 1:N, batch_size = pmap_batch_size) do i
            if i == N
                I_local = II[(batch_size * (i - 1) + 1):end]
            else
                I_local = II[(batch_size * (i - 1) + 1):(batch_size * i)]
            end
            solve_batch(prob, alg, EnsembleThreads(), I_local, pmap_batch_size; kwargs...)
        end
    end
    reduce(vcat, batch_data)
end
