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
    st = Iterators.filter(
        !isnothing, (hasproperty(x, :stats) ? x.stats : nothing for x in us))
    isempty(st) && return nothing
    reduce(merge, st)
end

mutable struct AggregateLogger{T <: Logging.AbstractLogger} <: Logging.AbstractLogger
    progress::Dict{Symbol, Float64}
    done_counter::Int
    total::Float64
    print_time::Float64
    lock::ReentrantLock
    logger::T
end
function AggregateLogger(logger::Logging.AbstractLogger)
    AggregateLogger(Dict{Symbol, Float64}(), 0, 0.0, 0.0, ReentrantLock(), logger)
end

function Logging.handle_message(
        l::AggregateLogger, level, message, _module, group, id, file, line; kwargs...)
    if convert(Logging.LogLevel, level) == Logging.LogLevel(-1) && haskey(kwargs, :progress)
        pr = kwargs[:progress]
        if trylock(l.lock) || (pr == "done" && lock(l.lock) === nothing)
            try
                if pr == "done"
                    pr = 1.0
                    l.done_counter += 1
                end
                len = length(l.progress)
                if haskey(l.progress, id)
                    l.total += (pr - l.progress[id]) / len
                else
                    l.total = l.total * (len / (len + 1)) + pr / (len + 1)
                    len += 1
                end
                l.progress[id] = pr
                # validation check (slow)
                # tot = sum(values(l.progress))/length(l.progress)
                # @show tot l.total l.total ≈ tot
                curr_time = time()
                if l.done_counter >= len
                    tot = "done"
                    empty!(l.progress)
                    l.done_counter = 0
                    l.print_time = 0.0
                elseif curr_time - l.print_time > 0.1
                    tot = l.total
                    l.print_time = curr_time
                else
                    return
                end
                id = :total
                message = "Total"
                kwargs = merge(values(kwargs), (progress = tot,))
            finally
                unlock(l.lock)
            end
        else
            return
        end
    end
    Logging.handle_message(
        l.logger, level, message, _module, group, id, file, line; kwargs...)
end
Logging.shouldlog(l::AggregateLogger, args...) = Logging.shouldlog(l.logger, args...)
Logging.min_enabled_level(l::AggregateLogger) = Logging.min_enabled_level(l.logger)
Logging.catch_exceptions(l::AggregateLogger) = Logging.catch_exceptions(l.logger)

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
        alg::A,
        ensemblealg::BasicEnsembleAlgorithm;
        trajectories, batch_size = trajectories, progress_aggregate = true,
        pmap_batch_size = batch_size ÷ 100 > 0 ? batch_size ÷ 100 : 1, kwargs...) where {A}
    logger = progress_aggregate ? AggregateLogger(Logging.current_logger()) :
             Logging.current_logger()

    Logging.with_logger(logger) do
        num_batches = trajectories ÷ batch_size
        num_batches < 1 &&
            error("trajectories ÷ batch_size cannot be less than 1, got $num_batches")
        num_batches * batch_size != trajectories && (num_batches += 1)

        if get(kwargs, :progress, false)
            name = get(kwargs, :progress_name, "Ensemble")
            for i in 1:trajectories
                msg = "$name #$i"
                Logging.@logmsg(Logging.LogLevel(-1), msg, _id=Symbol("SciMLBase_$i"),
                    progress=0)
            end
        end

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
                batch_data = solve_batch(
                    prob, alg, ensemblealg, II, pmap_batch_size; kwargs...)
                u, converged = prob.reduction(u, batch_data, II)
            end
        end
        _u = tighten_container_eltype(u)
        stats = merge_stats(_u)
        return EnsembleSolution(_u, elapsed_time, converged, stats)
    end
end

function batch_func(i, prob, alg; kwargs...)
    iter = 1
    _prob = prob.safetycopy ? deepcopy(prob.prob) : prob.prob
    new_prob = prob.prob_func(_prob, i, iter)
    rerun = true

    progress = get(kwargs, :progress, false)
    if progress
        name = get(kwargs, :progress_name, "Ensemble")
        progress_name = "$name #$i"
        progress_id = Symbol("SciMLBase_$i")
        kwargs = (kwargs..., progress_name = progress_name, progress_id = progress_id)
    end
    x = prob.output_func(solve(new_prob, alg; kwargs...), i)
    if !(x isa Tuple)
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
        if !(x isa Tuple)
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
    batch_data = Vector{Core.Compiler.return_type(f, Tuple{typeof.(getindex.(II, 1))...})}(
        undef,
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

    if prob.prob isa AbstractJumpProblem && length(II) != 1
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
