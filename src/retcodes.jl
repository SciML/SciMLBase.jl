"""
`SciML.ReturnCode`

`SciML.ReturnCode` is the standard return code enum interface for the SciML interface.
Return codes are notes given by the solvers to indicate the state of the solution, for
example whether it successfully solved the equations, whether it failed to solve the
equations, and importantly, why it exited.

## Using `SciML.ReturnCode`

`SciML.ReturnCode` use the interface of [EnumX.jl](https://github.com/fredrikekre/EnumX.jl)
and thus inherits all of the behaviors of being an EnumX. This includes the Enum type itself
being referred to as `SciML.ReturnCode.T`, and each of the constituent enum states being
referred to via `getproperty`, i.e. `SciML.ReturnCode.Success`.

## Note About Success Checking

Previous iterations of the interface suggested using `sol.retcode == :Success`, however,
that is now not advised instead should be replaced with ` SciMLBase.successful_retcode(sol)`. The reason is that there are many different
codes that can be interpreted as successful, such as `ReturnCode.Terminated` which means
successfully used `terminate!(integrator)` to end an integration at a user-specified
condition. As such, `successful_retcode` is the most general way to query for if the solver
did not error.

## Properties

  - `successful_retcode(retcode::ReturnCode.T)`: Determines whether the output enum is
    considered a success state of the solver, i.e. the solver successfully solved the
    equations. `ReturnCode.Success` is the most basic form, simply declaring that it was
    successful, but many more informative success return codes exist as well.
"""
EnumX.@enumx ReturnCode begin
    """
    `ReturnCode.Default`

    The default state of the solver. If this return code is given, then the solving
    process is either still in process or the solver library has not been setup
    with the return code interface and thus the return code is undetermined.

    ## Common Reasons for Seeing this Return Code

      - A common reason for `Default` return codes is that a solver is a non-SciML solver
        which does not fully conform to the interface. Please open an issue if this is seen
        and it will be improved.
      - Another common reason for a `Default` return code is if the solver is probed
        internally before the solving process is done, such as through the callback interface.
        Return codes are set to `Default` to start and are changed to `Success` and other
        return codes upon finishing the solving process or hitting a numerical difficulty.

    ## Properties

      - successful_retcode = false
    """
    Default

    """
    `ReturnCode.Success`

    The success state of the solver. If this return code is given, then the solving
    process was successful, but no extra information about that success is given.

    ## Common Reasons for Seeing this Return Code

      - This is the most common return code and most solvers will give this return code if
        the solving process went as expected without any errors or detected numerical issues.

    ## Properties

      - successful_retcode = true
    """
    Success

    """
    `ReturnCode.Terminated`

    The successful termination state of the solver. If this return code is given,
    then the solving process was successful at terminating the solve, usually
    through a callback `affect!` via `terminate!(integrator)`.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is if a user calls a callback which
        uses `terminate!(integrator)` to halt the integration at a user-chosen stopping point.
      - Another common reason for this return code is due to implicit `terminate!` statements
        in some library callbacks. For example, `SteadyStateCallback` uses `terminate!`
        internally, so solutions which reach steady state will have a `ReturnCode.Terminated`
        state instead of a `ReturnCode.Success` state. Similarly, problems solved via
        SteadyStateDiffEq.jl will have this `ReturnCode.Terminated` state if a timestepping
        method is used to solve to steady state.

    ## Properties

      - successful_retcode = true
    """
    Terminated

    """
    `ReturnCode.DtNaN`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful and exited early because the `dt` of the
    integration was determined to be `NaN` and thus the solver could not continue.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because the automatic `dt`
        selection algorithm is used but the starting derivative has a `NaN` or `Inf`
        derivative term. Double check that the `f(u0,p,t0)` term is well-defined without
        `NaN` or `Inf` values.
      - Another common reason for this return code is because of a user set `dt` which is
        calculated to be a `NaN`. If `solve(prob,alg,dt=x)`, double check that `x` is not
        `NaN`.

    ## Properties

      - successful_retcode = false
    """
    DtNaN

    """
    `ReturnCode.MaxIters`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful and exited early because the solver's
    iterations hit the `maxiters` either set by default or by the user in the
    `solve`/`init` command.

    ## Note about Nonlinear Optimization

    In nonlinear optimization, many solvers (such as `OptimizationOptimisers.Adam`) do not
    have an exit criteria other than `iters == maxiters`. In this case, the solvers will
    iterate until `maxiters` and exit with a `Success` return code, as that is a successful
    run of the solver and not considered to be an error state. Solves with early termination
    criteria, such as `Optim.BFGS` exiting when the gradient is sufficiently close to zero,
    will give `ReturnCode.MaxIters` on exits which require the maximum iteration.

    ## Common Reasons for Seeing this Return Code

      - This commonly occurs in ODE solving if a non-stiff method (e.g. `Tsit5`) is used in
        an algorithm choice for a stiff ODE. It is recommended that in such cases, one tries a
        stiff ODE solver.
      - This commonly occurs in optimization and nonlinear solvers if the tolerance on `solve`
        to too low and cannot be achieved due to floating point error or the condition number
        of the solver matrix. Double check that the chosen tolerance is numerically possible.

    ## Properties

      - successful_retcode = false
    """
    MaxIters

    """
    `ReturnCode.DtLessThanMin`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful and exited early because the `dt` of the
    integration was made to be less than `dtmin`, i.e. `dt < dtmin`.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because the integration
        is going unstable. As `f(u,p,t) -> infinity`, the time steps required by the solver
        to accurately handle the dynamics decreases. When it gets sufficiently small, `dtmin`,
        an exit is thrown as the solution is likely unstable. `dtmin` is also chosen to be
        around the value where floating point issues cause `t + dt == t`, and thus a `dt`
        of that size is impossible at floating point precision.
      - Another common reason for this return code is if domain constraints are set, such as
        by using `isoutofdomain`, but the domain constraint is incorrect. For example, if
        one is solving the ODE `f(u,p,t) = -u - 1`, one may think "but I want a solution with
        `u > 0` and thus I will set `isoutofdomain(u,p,t) = u < 0`. However, the true solution
        of this ODE is not positive, and thus what will occur is that the solver will try to
        decrease `dt` until it can give an accurate solution that is positive. As this is
        impossible, it will continue to shrink the `dt` until `dt < dtmin` and then exit with
        this return code.

    ## Properties

      - successful_retcode = false
    """
    DtLessThanMin

    """
    `ReturnCode.Unstable`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful and exited early because the `unstable_check`
    function, as given by the `unstable_check` common keyword argument (or its default),
    give a `true` at the current state.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because `u` contains a `NaN`
        or `Inf` value. The default `unstable_check` only checks for these values.

    ## Properties

      - successful_retcode = false
    """
    Unstable

    """
    `ReturnCode.InitialFailure`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful because the initialization process failed.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because the initialization
        process of a DAE solver failed to find consistent initial conditions, which can
        occur if the differentiation index of the DAE solver is too high. Most DAE solvers
        only allow for index-1 DAEs, and so an index-2 DAE will fail during this
        initialization. To solve this kind of problem, use `ModelingToolkit.jl` and its
        `structural_simplify` method to reduce the index of the DAE.
      - Another common reason for this return code is if the initial condition was not
        suitable for the numerical solve. For example, the initial point had a `NaN` or `Inf`.
        Or in optimization, this can occur if the initial point is outside of the bound
        constraints given by the user.

    ## Properties

      - successful_retcode = false
    """
    InitialFailure

    """
    `ReturnCode.ConvergenceFailure`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful because internal nonlinear solver iterations
    failed to converge.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because an inappropriate
        nonlinear solver was chosen. If fixed point iteration is used on a stiff problem,
        it will be faster by avoiding the Jacobian but it will make a stiff ODE solver not
        stable for stiff problems!
      - For nonlinear solvers, this can occur if certain threshold was exceeded. For example,
        in approximate jacobian solvers like Broyden, Klement, etc. if the number of jacobian
        resets exceeds the threshold, then this return code is given.

    ## Properties

      - successful_retcode = false
    """
    ConvergenceFailure

    """
    `ReturnCode.Failure`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful but no extra information is given.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for seeing this return code is because the solver is a wrapped
        solver (i.e. a Fortran code) which does not provide any extra information about its
        exit state. If this is from a Julia-based solver, please open an issue.

    ## Properties

      - successful_retcode = false
    """
    Failure

    """
    `ReturnCode.ExactSolutionLeft`

    The success state of the solver. If this return code is given, then the solving
    process was successful, and the left solution was given.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for this return code is via a bracketing nonlinear solver,
        such as bisection, iterating to convergence is unable to give the exact `f(x)=0`
        solution due to floating point precision issues, and thus it gives the first floating
        point value to the left for `x`.

    ## Properties

      - successful_retcode = true
    """
    ExactSolutionLeft

    """
    `ReturnCode.ExactSolutionRight`

    The success state of the solver. If this return code is given, then the solving
    process was successful, and the right solution was given.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for this return code is via a bracketing nonlinear solver,
        such as bisection, iterating to convergence is unable to give the exact `f(x)=0`
        solution due to floating point precision issues, and thus it gives the first floating
        point value to the right for `x`.

    ## Properties

      - successful_retcode = true
    """
    ExactSolutionRight

    """
    `ReturnCode.FloatingPointLimit`

    The success state of the solver. If this return code is given, then the solving
    process was successful, and the closest floating point value to the solution was given.

    ## Common Reasons for Seeing this Return Code

      - The most common reason for this return code is via a nonlinear solver, such as Falsi,
        iterating to convergence is unable to give the exact `f(x)=0` solution due to floating
        point precision issues, and thus it gives the closest floating point value to the
        true solution for `x`.

    ## Properties

      - successful_retcode = true
    """
    FloatingPointLimit

    """
    `ReturnCode.Infeasible`

    The optimization problem was proven to be infeasible by the solver.

    ## Properties

      - successful_retcode = false
    """
    Infeasible

    """
    `ReturnCode.MaxTime`

    A failure exit state of the solver. If this return code is given, then the
    solving process was unsuccessful and exited early because the solver's
    timer hit `maxtime` either set by default or by the user in the
    `solve`/`init` command.

    ## Properties

      - successful_retcode = false
    """
    MaxTime

    """
    `ReturnCode.InternalLineSearchFailed`

    Internal Line Search used by the algorithm has failed.

    ## Properties

      - successful_retcode = false
    """
    InternalLineSearchFailed

    """
    `ReturnCode.ShrinkThresholdExceeded`

    The trust region radius was shrunk more times than the provided threshold.

    ## Properties

      - successful_retcode = false
    """
    ShrinkThresholdExceeded

    """
    `ReturnCode.Stalled`

    The solution has stalled. This is only returned by algorithms for which stalling is a
    failure mode. Certain solvers like Nonlinear Least Squares solvers are considered
    successful if the solution has stalled, in those cases `ReturnCode.Success` is returned.

    ## Properties

      - successful_retcode = false
    """
    Stalled
end

Base.:(!=)(retcode::ReturnCode.T, s::Symbol) = Symbol(retcode) != s

const symtrue = Symbol("true")
const symfalse = Symbol("false")

function Base.convert(::Type{ReturnCode.T}, retcode::Symbol)
    @warn "Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information"

    if retcode == :Default || retcode == :DEFAULT
        ReturnCode.Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT ||
           retcode == :FLOATING_POINT_LIMIT || retcode == symtrue || retcode == :OPTIMAL ||
           retcode == :LOCALLY_SOLVED
        ReturnCode.Success
    elseif retcode == :Terminated
        ReturnCode.Terminated
    elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED
        ReturnCode.MaxIters
    elseif retcode == :MaxTime || retcode == :TIME_LIMIT
        ReturnCode.MaxTime
    elseif retcode == :DtLessThanMin
        ReturnCode.DtLessThanMin
    elseif retcode == :Unstable
        ReturnCode.Unstable
    elseif retcode == :InitialFailure
        ReturnCode.InitialFailure
    elseif retcode == :ConvergenceFailure || retcode == :ITERATION_LIMIT
        ReturnCode.ConvergenceFailure
    elseif retcode == :Failure || retcode == symfalse
        ReturnCode.Failure
    elseif retcode == :Infeasible || retcode == :INFEASIBLE ||
           retcode == :DUAL_INFEASIBLE || retcode == :LOCALLY_INFEASIBLE ||
           retcode == :INFEASIBLE_OR_UNBOUNDED
        ReturnCode.Infeasible
    else
        ReturnCode.Failure
    end
end

# Deprecate ASAP, only to make the deprecation easier
symbol_to_ReturnCode(retcode::ReturnCode.T) = retcode
function symbol_to_ReturnCode(retcode::Symbol)
    if retcode == :Default || retcode == :DEFAULT
        ReturnCode.Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT ||
           retcode == :FLOATING_POINT_LIMIT || retcode == symtrue || retcode == :OPTIMAL ||
           retcode == :LOCALLY_SOLVED
        ReturnCode.Success
    elseif retcode == :Terminated
        ReturnCode.Terminated
    elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED
        ReturnCode.MaxIters
    elseif retcode == :MaxTime || retcode == :TIME_LIMIT
        ReturnCode.MaxTime
    elseif retcode == :DtLessThanMin
        ReturnCode.DtLessThanMin
    elseif retcode == :Unstable
        ReturnCode.Unstable
    elseif retcode == :InitialFailure
        ReturnCode.InitialFailure
    elseif retcode == :ConvergenceFailure || retcode == :ITERATION_LIMIT
        ReturnCode.ConvergenceFailure
    elseif retcode == :Failure || retcode == symfalse
        ReturnCode.Failure
    elseif retcode == :Infeasible || retcode == :INFEASIBLE ||
           retcode == :DUAL_INFEASIBLE || retcode == :LOCALLY_INFEASIBLE ||
           retcode == :INFEASIBLE_OR_UNBOUNDED
        ReturnCode.Infeasible
    else
        ReturnCode.Failure
    end
end

function Base.convert(::Type{ReturnCode.T}, bool::Bool)
    bool ? ReturnCode.Success : ReturnCode.Failure
end

"""
`successful_retcode(retcode::ReturnCode.T)::Bool`
`successful_retcode(sol::AbstractSciMLSolution)::Bool`

Returns a boolean for whether a return code should be interpreted as a form of success.
"""
function successful_retcode end

function successful_retcode(retcode::ReturnCode.T)
    retcode == ReturnCode.Success || retcode == ReturnCode.Terminated ||
        retcode == ReturnCode.ExactSolutionLeft ||
        retcode == ReturnCode.ExactSolutionRight ||
        retcode == ReturnCode.FloatingPointLimit
end
successful_retcode(sol::AbstractSciMLSolution) = successful_retcode(sol.retcode)
