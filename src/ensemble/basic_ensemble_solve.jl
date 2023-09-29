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

function merge_stats(us)
    mapreduce(x -> x.stats, merge, us)
end

function __solve(prob::AbstractEnsembleProblem,
    alg::Union{AbstractDEAlgorithm, Nothing};
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

function __solve(prob::EnsembleProblem{<:AbstractVector{<:AbstractSciMLProblem}},
    alg::Union{AbstractDEAlgorithm, Nothing},
    ensemblealg::BasicEnsembleAlgorithm; kwargs...)
    # TODO: @invoke
    invoke(__solve, Tuple{AbstractEnsembleProblem, typeof(alg), typeof(ensemblealg)},
        prob, alg, ensemblealg; trajectories = length(prob.prob), kwargs...)
end

function __solve(prob::AbstractEnsembleProblem,
    alg::Union{AbstractDEAlgorithm, Nothing},
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
        stats = merge_stats(_u)
        return EnsembleSolution(_u, elapsed_time, true, stats)
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
    stats = merge_stats(_u)
    return EnsembleSolution(_u, elapsed_time, converged, stats)
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
    }(undef,
        length(args[1]))
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
