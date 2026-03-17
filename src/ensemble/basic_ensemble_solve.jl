"""
$(TYPEDEF)
"""
abstract type BasicEnsembleAlgorithm <: EnsembleAlgorithm end

"""
$(TYPEDEF)

Basic ensemble solver which uses no parallelism and runs
the problems in serial
"""
struct EnsembleSerial <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

The default. This uses multithreading. It's local (single computer, shared memory)
parallelism only. Lowest parallelism overhead for small problems.
"""
struct EnsembleThreads <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

Uses `pmap` internally. It will use as many processors as you
have Julia processes. To add more processes, use `addprocs(n)`. These processes
can be placed onto multiple different machines in order to paralleize across
an entire cluster via passwordless SSH. See Julia's
documentation for more details.

Recommended for the case when each trajectory calculation isn't “too quick” (at least about
a millisecond each?), where the calculations of a given problem allocate memory, or when
you have a very large ensemble. This can be true even on a single shared memory system
because distributed process use separate garbage collectors and thus can be even faster
than EnsembleThreads if the computation is complex enough.
"""
struct EnsembleDistributed <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

A mixture of distributed computing with threading. The optimal version of this is to have
a process on each node of a computer and then multithread on each system. However, this
ensembler will simply use the node setup provided by the Julia Distributed processes, and
thus it is recommended that you setup the processes in this fashion before using this
ensembler. See Julia's Distributed documentation for more information
"""
struct EnsembleSplitThreads <: BasicEnsembleAlgorithm end

function merge_stats(us)
    st = Iterators.filter(
        !isnothing, (hasproperty(x, :stats) ? x.stats : nothing for x in us)
    )
    isempty(st) && return nothing
    return reduce(merge, st)
end

"""
    EnsembleContext{S, R, M}

Contextual information about the current simulation within an ensemble solve.
Passed to `rng_func`, `prob_func(prob, ctx)`, and `output_func(sol, ctx)`.

## Fields
- `sim_id::Int` — Unique simulation index (1:trajectories)
- `repeat::Int` — Rerun counter (starts at 1, increments if `output_func` requests rerun)
- `worker_id::Int` — 0 for serial/threaded; `Distributed.myid()` for distributed
- `sim_seed::S` — Pre-generated seed for this simulation, or `nothing`
- `rng::R` — Per-simulation RNG created by `rng_func`, or `nothing`.
  When `rng_func` has not yet been called (i.e. inside `rng_func` itself), this is `nothing`.
- `master_rng::M` — User-provided master RNG, or `nothing`.
  Set to `nothing` in distributed modes (`EnsembleDistributed`, `EnsembleSplitThreads`)
  to avoid serialization issues.

!!! warning "Thread safety"
    In threaded ensemble modes (`EnsembleThreads`, `EnsembleSplitThreads`), `master_rng`
    is shared across tasks. A custom `rng_func` must **not** mutate `master_rng` unless
    it is thread-safe. The default `default_rng_func` does not access `master_rng`.
"""
struct EnsembleContext{S, R, M}
    sim_id::Int
    repeat::Int
    worker_id::Int
    sim_seed::S
    rng::R
    master_rng::M
end

"""
    generate_sim_seeds(rng, seed, trajectories)

Pre-generate an array of per-simulation seeds from a master RNG.
If `rng` is provided it is used directly; otherwise `Xoshiro(seed)` is constructed.
"""
function generate_sim_seeds(rng, seed, trajectories)
    master = rng !== nothing ? rng : Random.Xoshiro(seed)
    return [rand(master, UInt64) for _ in 1:trajectories]
end

"""
    default_rng_func(ctx::EnsembleContext)

Default per-simulation RNG factory. Seeds `TaskLocalRNG` with the simulation seed
(if available) and returns `Random.default_rng()`.

Note: `ctx.rng` is `nothing` when this function is called, since its purpose is to
create the RNG that will be stored in `ctx.rng` for `prob_func` and `output_func`.
"""
function default_rng_func(ctx::EnsembleContext)
    if ctx.sim_seed !== nothing
        Random.seed!(ctx.sim_seed)
    end
    return Random.default_rng()
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
    return AggregateLogger(Dict{Symbol, Float64}(), 0, 0.0, 0.0, ReentrantLock(), logger)
end

function Logging.handle_message(
        l::AggregateLogger, level, message, _module, group, id, file, line; kwargs...
    )
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
    return Logging.handle_message(
        l.logger, level, message, _module, group, id, file, line; kwargs...
    )
end
Logging.shouldlog(l::AggregateLogger, args...) = Logging.shouldlog(l.logger, args...)
Logging.min_enabled_level(l::AggregateLogger) = Logging.min_enabled_level(l.logger)
Logging.catch_exceptions(l::AggregateLogger) = Logging.catch_exceptions(l.logger)

function __solve(
        prob::AbstractEnsembleProblem,
        alg::Union{AbstractDEAlgorithm, Nothing};
        kwargs...
    )
    if alg isa EnsembleAlgorithm
        # Assume DifferentialEquations.jl is being used, so default alg
        ensemblealg = alg
        alg = nothing
    else
        ensemblealg = EnsembleThreads()
    end
    return __solve(prob, alg, ensemblealg; kwargs...)
end

@noinline function rerun_warn()
    return @warn("output_func should return (out,rerun). See docs for updated details")
end
tighten_container_eltype(u::Vector{Any}) = map(identity, u)
tighten_container_eltype(u) = u

function __solve(
        prob::EnsembleProblem{<:AbstractVector{<:AbstractSciMLProblem}},
        alg::Union{AbstractDEAlgorithm, Nothing},
        ensemblealg::BasicEnsembleAlgorithm; kwargs...
    )
    Base.depwarn(
        "This dispatch is deprecated for the standard ensemble syntax. See the Parallel
Ensembles Simulations Interface page for more details",
        :EnsembleProblemSolve
    )
    return invoke(
        __solve, Tuple{AbstractEnsembleProblem, typeof(alg), typeof(ensemblealg)},
        prob, alg, ensemblealg; trajectories = length(prob.prob), kwargs...
    )
end

"""
    sim = solve(enprob, alg, ensemblealg = EnsembleThreads(), kwargs...)

Solves the ensemble problem `enprob` with the algorithm `alg` using the ensembler
`ensemblealg`.

The keyword arguments take in the arguments for the common solver interface and will
pass them to the solver. The `ensemblealg` is optional, and will
default to `EnsembleThreads()`. The special keyword arguments to note are:

  - `trajectories`: The number of simulations to run. This argument is required.
  - `batch_size` : The size of the batches on which the reductions are applies. Defaults to `trajectories`.
  - `pmap_batch_size`: The size of the `pmap` batches. Default is
    `batch_size÷100 > 0 ? batch_size÷100 : 1`
  - `seed`: Master seed for reproducible ensemble solves. Pre-generates per-trajectory seeds.
  - `rng`: Master RNG for reproducible ensemble solves. Takes priority over `seed`.
  - `rng_func`: Custom per-trajectory RNG factory `(ctx::EnsembleContext) -> AbstractRNG`.
    Defaults to `default_rng_func` which seeds the `TaskLocalRNG`.
    Note: `ctx.master_rng` is shared across tasks in threaded modes. A custom `rng_func`
    must not mutate it unless it is thread-safe.
"""
function __solve(
        prob::AbstractEnsembleProblem,
        alg::A,
        ensemblealg::BasicEnsembleAlgorithm;
        trajectories, batch_size = trajectories, progress_aggregate = true,
        pmap_batch_size = batch_size ÷ 100 > 0 ? batch_size ÷ 100 : 1,
        seed = nothing,
        rng = nothing,
        rng_func = default_rng_func,
        kwargs...
    ) where {A}
    logger = progress_aggregate ? AggregateLogger(Logging.current_logger()) :
        Logging.current_logger()

    # Pre-generate simulation seeds if rng or seed is provided
    sim_seeds = (rng !== nothing || seed !== nothing) ?
        generate_sim_seeds(rng, seed, trajectories) : nothing

    # Pre-compute solve-RNG strategy flags from prob.prob's type. These are converted to
    # Val types in _dispatch_ensemble_solve for type-stable dispatch. This assumes prob_func
    # preserves the problem type (e.g., a JumpProblem stays a JumpProblem).
    # For the deprecated Vector-of-problems path, prob.prob is a Vector — default to :seed
    # mode so seed/rng kwargs are forwarded (matching master behavior).
    _is_rng_solver = supports_solve_rng(prob.prob, alg)
    _is_jump_prob = prob.prob isa AbstractJumpProblem ||
        prob.prob isa AbstractVector

    # Function barrier: _dispatch_ensemble_solve converts Bool flags to Val types
    # so the compiler sees concrete types in _solve_ensemble_impl. Val(::Bool) infers
    # as abstract Val; explicit if/else with Val literals gives Union{Val{true}, Val{false}}
    # which the compiler can union-split.
    return _dispatch_ensemble_solve(
        _is_rng_solver, _is_jump_prob,
        prob, alg, ensemblealg, sim_seeds, rng_func, rng, logger;
        trajectories, batch_size, pmap_batch_size, kwargs...
    )
end

# Function barrier: converts Bool flags to Val types with explicit if/else so the
# compiler sees Union{Val{true}, Val{false}} (union-splittable) rather than abstract Val.
function _dispatch_ensemble_solve(
        _is_rng_solver::Bool, _is_jump_prob::Bool,
        prob, alg, ensemblealg, sim_seeds, rng_func, rng, logger;
        kwargs...
    )
    _solve_rng_mode = _is_rng_solver ? Val(:rng) : _is_jump_prob ? Val(:seed) : Val(:none)
    return _solve_ensemble_impl(
        prob, alg, ensemblealg, _solve_rng_mode,
        sim_seeds, rng_func, rng, logger;
        kwargs...
    )
end

function _solve_ensemble_impl(
        prob, alg, ensemblealg, _solve_rng_mode::Val{SM},
        sim_seeds, rng_func, rng, logger;
        trajectories, batch_size, pmap_batch_size, kwargs...
    ) where {SM}
    # Bundle ensemble RNG state to pass through solve_batch -> batch_func.
    # All fields have concrete types here thanks to the function barrier.
    ensemble_rng_state = (;
        sim_seeds, _solve_rng_mode, rng_func,
        master_rng = rng,
    )

    return Logging.with_logger(logger) do
        num_batches = trajectories ÷ batch_size
        num_batches < 1 &&
            error("trajectories ÷ batch_size cannot be less than 1, got $num_batches")
        num_batches * batch_size != trajectories && (num_batches += 1)

        if get(kwargs, :progress, false)
            name = get(kwargs, :progress_name, "Ensemble")
            for i in 1:trajectories
                msg = "$name #$i"
                Logging.@logmsg(
                    Logging.LogLevel(-1), msg, _id = Symbol("SciMLBase_$i"),
                    progress = 0
                )
            end
        end

        if num_batches == 1 && prob.reduction === DEFAULT_REDUCTION
            elapsed_time = @elapsed u = solve_batch(
                prob, alg, ensemblealg, 1:trajectories,
                pmap_batch_size, ensemble_rng_state; kwargs...
            )
            _u = tighten_container_eltype(u)
            stats = merge_stats(_u)
            return EnsembleSolution(_u, elapsed_time, true, stats)
        end

        converged::Bool = false
        elapsed_time = @elapsed begin
            i = 1
            II = (batch_size * (i - 1) + 1):(batch_size * i)

            batch_data = solve_batch(
                prob, alg, ensemblealg, II, pmap_batch_size,
                ensemble_rng_state; kwargs...
            )

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
                    prob, alg, ensemblealg, II, pmap_batch_size,
                    ensemble_rng_state; kwargs...
                )
                u, converged = prob.reduction(u, batch_data, II)
            end
        end
        _u = tighten_container_eltype(u)
        stats = merge_stats(_u)
        return EnsembleSolution(_u, elapsed_time, converged, stats)
    end
end

# Val-dispatch for solve RNG strategy:
#   :rng  → solver supports rng kwarg directly (e.g. JumpProcesses ≥ v10)
#   :seed → solver doesn't support rng kwarg but problem is a JumpProblem;
#           pass seed kwarg so JP v9 can reseed the aggregator's stored RNG
#   :none → non-jump solver; TaskLocalRNG already seeded by rng_func, no extra kwargs needed
_invoke_solve(::Val{:rng}, new_prob, alg, rng, seed; kwargs...) =
    solve(new_prob, alg; rng, kwargs...)
_invoke_solve(::Val{:seed}, new_prob, alg, rng, seed; kwargs...) =
    seed !== nothing ? solve(new_prob, alg; seed, kwargs...) : solve(new_prob, alg; kwargs...)
_invoke_solve(::Val{:none}, new_prob, alg, rng, seed; kwargs...) =
    solve(new_prob, alg; kwargs...)

function batch_func(
        i, prob, alg, ensemble_rng_state, thread_prob;
        worker_id = 0, kwargs...
    )
    (; sim_seeds, _solve_rng_mode, rng_func, master_rng) = ensemble_rng_state

    # Build context for this simulation (rng = nothing before rng_func is called)
    sim_seed = sim_seeds !== nothing ? sim_seeds[i] : nothing
    iter = 1
    pre_ctx = EnsembleContext(i, iter, worker_id, sim_seed, nothing, master_rng)

    # Get per-simulation RNG (always called to seed TaskLocalRNG for this simulation)
    sim_rng = rng_func(pre_ctx)

    # Update context with the created RNG (new variable to avoid type instability
    # since R changes from Nothing to the concrete RNG type)
    ctx = @set pre_ctx.rng = sim_rng

    # Respect safetycopy: if true, deepcopy per simulation (user's explicit choice).
    # If false, use per-task JumpProblem copy if available (fixes race condition),
    # otherwise use the original problem directly.
    _prob = if prob.safetycopy
        deepcopy(prob.prob)
    elseif thread_prob !== nothing
        thread_prob
    else
        prob.prob
    end

    # Call prob_func(prob, ctx) — single 2-arg form, no arity detection needed
    new_prob = prob.prob_func(_prob, ctx)

    # Progress handling
    progress = get(kwargs, :progress, false)
    if progress
        name = get(kwargs, :progress_name, "Ensemble")
        progress_name = "$name #$i"
        progress_id = Symbol("SciMLBase_$i")
        kwargs = (kwargs..., progress_name = progress_name, progress_id = progress_id)
    end

    # Solve — dispatch on pre-computed _solve_rng_mode:
    #   :rng  → pass rng kwarg (new interface)
    #   :seed → pass seed kwarg (JP v9 fallback for explicit-RNG JumpProblems)
    #   :none → no RNG kwargs (non-DE solvers; TaskLocalRNG already seeded by rng_func)
    _solve_call = _invoke_solve(
        _solve_rng_mode, new_prob, alg, sim_rng, sim_seed; kwargs...
    )
    x = prob.output_func(_solve_call, ctx)
    if !(x isa Tuple)
        rerun_warn()
        _x = (x, false)
    else
        _x = x
    end

    # Rerun loop
    rerun = _x[2]
    while rerun
        iter += 1
        ctx = @set ctx.repeat = iter
        _prob2 = if prob.safetycopy
            deepcopy(prob.prob)
        elseif thread_prob !== nothing
            thread_prob
        else
            prob.prob
        end
        new_prob = prob.prob_func(_prob2, ctx)
        _solve_call = _invoke_solve(
            _solve_rng_mode, new_prob, alg, sim_rng, sim_seed; kwargs...
        )
        x = prob.output_func(_solve_call, ctx)
        if !(x isa Tuple)
            rerun_warn()
            _x = (x, false)
        else
            _x = x
        end
        rerun = _x[2]
    end
    return _x[1]
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleDistributed, II, pmap_batch_size,
        ensemble_rng_state; kwargs...
    )
    wp = CachingPool(workers())

    # Sanitize: don't serialize master_rng to workers (may not be serializable,
    # and is unnecessary — trajectory seeds are already pre-generated)
    dist_rng_state = (; ensemble_rng_state..., master_rng = nothing)

    batch_data = pmap(wp, II, batch_size = pmap_batch_size) do i
        # thread_prob = nothing: pmap serializes prob to each worker, so every
        # worker gets its own deserialized copy. No shared memory means no
        # JumpProblem race condition, unlike the threaded case.
        batch_func(
            i, prob, alg, dist_rng_state, nothing;
            worker_id = myid(), kwargs...
        )
    end

    return tighten_container_eltype(batch_data)
end

__getindex(x, i) = x[i]
__getindex(x::AbstractVectorOfArray, i) = x.u[i]

function responsible_map(f, II...)
    batch_data = Vector{
        Core.Compiler.return_type(
            f, Tuple{ntuple(i -> typeof(__getindex(II[i], 1)), Val(length(II)))...}
        ),
    }(
        undef,
        length(II[1])
    )
    for i in 1:length(II[1])
        batch_data[i] = f(ntuple(ii -> __getindex(II[ii], i), Val(length(II)))...)
    end
    return batch_data
end

function SciMLBase.solve_batch(
        prob, alg, ::EnsembleSerial, II, pmap_batch_size,
        ensemble_rng_state; worker_id = 0, kwargs...
    )
    batch_data = responsible_map(II) do i
        SciMLBase.batch_func(
            i, prob, alg, ensemble_rng_state, nothing;
            worker_id, kwargs...
        )
    end
    return SciMLBase.tighten_container_eltype(batch_data)
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size,
        ensemble_rng_state; worker_id = 0, kwargs...
    )
    nthreads = min(Threads.nthreads(), length(II))
    if length(II) == 1 || nthreads == 1
        return solve_batch(
            prob, alg, EnsembleSerial(), II, pmap_batch_size,
            ensemble_rng_state; worker_id, kwargs...
        )
    end

    # Per-task JumpProblem isolation (replaces vestigial threadid()-indexed deepcopy).
    # Only needed when safetycopy is false — when safetycopy is true,
    # batch_func already deepcopies per trajectory which provides isolation.
    # Uses task_local_storage() for safe per-task copies under Julia's M:N scheduler.
    needs_jump_copy = !prob.safetycopy && prob.prob isa AbstractJumpProblem

    batch_data = tmap(II) do i
        _base_prob = if needs_jump_copy
            tls = task_local_storage()
            if !haskey(tls, :_ensemble_jump_prob)
                tls[:_ensemble_jump_prob] = deepcopy(prob.prob)
            end
            tls[:_ensemble_jump_prob]::typeof(prob.prob)
        else
            nothing
        end
        batch_func(
            i, prob, alg, ensemble_rng_state, _base_prob;
            worker_id, kwargs...
        )
    end
    return tighten_container_eltype(batch_data)
end

function tmap(f, args...)
    batch_data = Vector{
        Core.Compiler.return_type(f, Tuple{typeof.(getindex.(args, 1))...}),
    }(
        undef,
        length(args[1])
    )
    Threads.@threads for i in 1:length(args[1])
        batch_data[i] = f(getindex.(args, i)...)
    end
    return batch_data
end

function solve_batch(
        prob, alg, ::EnsembleSplitThreads, II, pmap_batch_size,
        ensemble_rng_state; kwargs...
    )
    wp = CachingPool(workers())
    N = nworkers()
    batch_size = length(II) ÷ N

    # Sanitize master_rng for distributed serialization
    dist_rng_state = (; ensemble_rng_state..., master_rng = nothing)

    batch_data = let
        pmap(wp, 1:N, batch_size = pmap_batch_size) do i
            if i == N
                I_local = II[(batch_size * (i - 1) + 1):end]
            else
                I_local = II[(batch_size * (i - 1) + 1):(batch_size * i)]
            end
            solve_batch(
                prob, alg, EnsembleThreads(), I_local, pmap_batch_size,
                dist_rng_state; worker_id = myid(), kwargs...
            )
        end
    end
    return reduce(vcat, batch_data)
end

# Backwards-compatible 5-arg fallbacks for callers that don't pass ensemble_rng_state
# (e.g. DiffEqGPU's CPU offload path). Constructs a no-op RNG state and forwards.
const _DEFAULT_ENSEMBLE_RNG_STATE = (;
    sim_seeds = nothing,
    _solve_rng_mode = Val(:none),
    rng_func = Returns(nothing),
    master_rng = nothing,
)

function solve_batch(prob, alg, ensemblealg::EnsembleSerial, II, pmap_batch_size; kwargs...)
    return solve_batch(
        prob, alg, ensemblealg, II, pmap_batch_size,
        _DEFAULT_ENSEMBLE_RNG_STATE; kwargs...
    )
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size; kwargs...
    )
    return solve_batch(
        prob, alg, ensemblealg, II, pmap_batch_size,
        _DEFAULT_ENSEMBLE_RNG_STATE; kwargs...
    )
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleDistributed, II, pmap_batch_size; kwargs...
    )
    return solve_batch(
        prob, alg, ensemblealg, II, pmap_batch_size,
        _DEFAULT_ENSEMBLE_RNG_STATE; kwargs...
    )
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleSplitThreads, II, pmap_batch_size; kwargs...
    )
    return solve_batch(
        prob, alg, ensemblealg, II, pmap_batch_size,
        _DEFAULT_ENSEMBLE_RNG_STATE; kwargs...
    )
end

function solve(prob::EnsembleProblem, args...; kwargs...)
    alg = extract_alg(args, kwargs, kwargs)
    return if length(args) > 1
        __solve(prob, alg, Base.tail(args)...; kwargs...)
    else
        __solve(prob, alg; kwargs...)
    end
end

function solve(prob::SciMLBase.WeightedEnsembleProblem, args...; kwargs...)
    return WeightedEnsembleSolution(solve(prob.ensembleprob), prob.weights)
end
