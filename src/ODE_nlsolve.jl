"""
    $(TYPEDEF)

A collection of all the data required for custom ODE Nonlinear problem solving
"""
struct ODE_NLProbData{NLProb, UNLProb, NLProbMap, NLProbPmap}
    """
    The `AbstractNonlinearProblem` to define custom nonlinear problems to be used for
    implicit time discretizations. This allows to use extra structure of the ODE function (e.g.
    multi-level structure). The nonlinear function must match that form of the function implicit
    ODE integration algorithms need do solve the a nonlinear problems,
    specifically of the form `z = outer_tmp + dt⋅f(γ⋅z+inner_tmp,p,t)`.
    Here `z` is the stage solution vector, `p` is the parameter of the ODE problem, `t` is
    the time, `dt` the respective time increment`, `γ` is some scaling factor and the temporary
    variables are some compatible vectors set by the specific solver.
    Note that this field will not be used for integrators such as fully-implicit Runge-Kutta methods
    that need to solve different nonlinear systems.
    The inner nonlinear function of the nonlinear problem is in general of the form `g(z,p') = 0`
    where `p'` is a NamedTuple with all information about the specific nonlinear problem at hand to solve
    for a specific time discretization. Specifically, it is `(;dt, γ, inner_tmp, outer_tmp, t, p)`, such that
    `g(z,p') = dt⋅f(γ⋅z+inner_tmp,p,t) + outer_tmp - z = 0`.
    """
    nlprob::NLProb
    """
    A function which takes `(nlprob, value_provider)` and updates
    the parameters of the former with their values in the latter.
    If absent (`nothing`) this will not be called, and the parameters
    in `nlprob` will be used without modification. `value_provider`
    refers to a value provider as defined by SymbolicIndexingInterface.jl.
    Usually this will refer to a problem or integrator.
    """
    update_nlprob!::UNLProb
    """
    A function which takes the solution of `nlprob` and returns
    the state vector of the original problem.
    """
    nlprobmap::NLProbMap
    """
    A function which takes the solution of `nlprob` and returns
    the parameter object of the original problem. If absent (`nothing`),
    this will not be called and the parameters of the problem being
    solved will be returned as-is.
    """
    nlprobpmap::NLProbPmap
end

