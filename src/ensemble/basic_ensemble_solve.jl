"""
$(TYPEDEF)

Base interface for built-in ensemble execution algorithms. A
`BasicEnsembleAlgorithm` chooses how trajectories from an `EnsembleProblem` are
scheduled: serially, on Julia threads, across Julia distributed workers, or with
a split distributed/threaded strategy.

Concrete subtypes should document their execution backend, worker setup
requirements, serialization assumptions, and random number behavior. They are
passed as the ensemble algorithm in calls such as
`solve(ensembleprob, alg, ensemblealg; trajectories, kwargs...)`, while `alg`
continues to select the numerical solver for each generated trajectory.
"""
abstract type BasicEnsembleAlgorithm <: EnsembleAlgorithm end

"""
$(TYPEDEF)

Ensemble execution algorithm that runs trajectories serially in the current Julia
process.

`EnsembleSerial` has the lowest scheduling complexity and is useful for
debugging, deterministic single-task execution, or small ensembles where
parallel overhead dominates the solve time. Trajectories still receive distinct
[`EnsembleContext`](@ref) values and can use per-trajectory RNG state when
`seed` or `rng` is supplied to `solve`.
"""
struct EnsembleSerial <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

Ensemble execution algorithm that schedules trajectories on Julia threads in the
current process.

`EnsembleThreads` is the default basic ensemble backend. It provides shared-memory
parallelism with low overhead, but user hooks such as `prob_func`, `output_func`,
and `rng_func` must be thread-safe. In particular, mutable objects captured by
closures or shared through the template problem should not be mutated unless each
trajectory receives independent storage.
"""
struct EnsembleThreads <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

Ensemble execution algorithm that distributes trajectory batches with
`Distributed.pmap`.

`EnsembleDistributed` uses the available Julia worker processes; add workers
with `Distributed.addprocs` before solving. The template problem, algorithms,
callbacks, and ensemble hooks must be serializable and available on each worker.
This backend is usually appropriate when each trajectory is relatively expensive,
when trajectories allocate heavily, or when work should be spread across multiple
machines. It can also help on a single machine when separate worker processes and
garbage collectors outweigh distributed communication overhead.
"""
struct EnsembleDistributed <: BasicEnsembleAlgorithm end

"""
$(TYPEDEF)

Ensemble execution algorithm that combines distributed workers with threaded
execution inside each worker process.

`EnsembleSplitThreads` uses the Julia distributed worker setup provided by the
caller and then runs threaded batches on each worker. It is intended for node-based
cluster layouts, for example one Julia worker per node with multiple local threads
per worker. The same serialization requirements as [`EnsembleDistributed`](@ref)
apply, and threaded hooks must also be thread-safe.
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

Contextual information about the current trajectory within an ensemble solve.

An `EnsembleContext` is passed to `rng_func`, `prob_func(prob, ctx)`, and
`output_func(sol, ctx)`. It is the stable interface for selecting
trajectory-specific data without relying on global counters or backend-specific
worker state.

## Fields

- `sim_id::Int`: Unique trajectory index in `1:trajectories`.
- `repeat::Int`: Rerun counter, starting at `1` and incremented when
  `output_func` requests a rerun.
- `worker_id::Int`: `0` for serial and threaded execution, or
  `Distributed.myid()` for distributed execution.
- `sim_seed::S`: Pre-generated seed for this trajectory, or `nothing`.
- `rng::R`: Per-trajectory RNG created by `rng_func`, or `nothing` while
  `rng_func` itself is running.
- `master_rng::M`: User-provided master RNG, or `nothing`. Distributed modes set
  this to `nothing` to avoid serializing mutable RNG state to workers.

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

Pre-generate an array of per-trajectory seeds from a master RNG.

If `rng` is provided it is used directly; otherwise `Random.Xoshiro(seed)` is
constructed. The returned `UInt64` values are stored in each
[`EnsembleContext`](@ref) as `ctx.sim_seed` so serial, threaded, and distributed
ensemble execution can create reproducible per-trajectory RNG state.
"""
function generate_sim_seeds(rng, seed, trajectories)
    master = rng !== nothing ? rng : Random.Xoshiro(seed)
    return [rand(master, UInt64) for _ in 1:trajectories]
end

"""
    default_rng_func(ctx::EnsembleContext)

Default per-trajectory RNG factory.

When `ctx.sim_seed` is available, this seeds Julia's task-local default RNG and
returns `Random.default_rng()`. `ctx.rng` is `nothing` when this function is
called, since the returned RNG is what will be stored as `ctx.rng` for
`prob_func` and `output_func`.
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

Solve an [`AbstractEnsembleProblem`](@ref) by running many trajectories of the
template problem.

`alg` is the numerical algorithm used for each generated trajectory, while
`ensemblealg` chooses the scheduling backend. Keywords that are not consumed by
the ensemble layer are forwarded to each inner `solve` call, so common solver
keywords such as `saveat`, tolerances, callbacks, and progress options retain
their usual meaning for each trajectory.

The special ensemble keywords are:

  - `trajectories`: Required number of trajectories to run.
  - `batch_size`: Number of trajectories processed before `prob.reduction` is
    called. Defaults to `trajectories`.
  - `pmap_batch_size`: Batch size passed to `pmap` for distributed ensemble
    algorithms. Defaults to `div(batch_size, 100) > 0 ? div(batch_size, 100) : 1`.
  - `progress_aggregate`: Whether per-trajectory progress messages are
    aggregated into a single total progress message. Defaults to `true`.
  - `seed`: Master seed for reproducible ensemble solves. Pre-generates
    per-trajectory seeds.
  - `rng`: Master RNG for reproducible ensemble solves. Takes priority over `seed`.
  - `rng_func`: Custom per-trajectory RNG factory `(ctx::EnsembleContext) -> AbstractRNG`.
    Defaults to `default_rng_func` which seeds the `TaskLocalRNG`.

In threaded modes, `ctx.master_rng` is shared across tasks. A custom `rng_func`
must not mutate it unless that mutation is thread-safe.
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

function responsible_map(f, II...)
    batch_data = Vector{
        Core.Compiler.return_type(
            f, Tuple{ntuple(i -> typeof(II[i][1]), Val(length(II)))...}
        ),
    }(
        undef,
        length(II[1])
    )
    for i in 1:length(II[1])
        batch_data[i] = f(ntuple(ii -> II[ii][i], Val(length(II)))...)
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

function solve(prob::EnsembleProblem, args...; kwargs...)
    if haskey(kwargs, :ensemblealg)
        throw(
            ArgumentError(
                "ensemblealg must be passed as a positional argument, not a keyword argument. " *
                    "Correct usage: solve(prob, alg, ensemblealg; kwargs...)"
            )
        )
    end

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
