"""
    $(TYPEDEF)

A collection of hooks for custom nonlinear stage solves in implicit ODE
algorithms.

`ODENLStepData` lets an `ODEFunction` provide a structured
`AbstractNonlinearProblem` template for solver packages that form implicit stage
equations. Before each nonlinear solve, the algorithm updates the stage guess,
scaling factors, time information, and temporary vectors through the stored
setter callables. After the nonlinear solve, `nlprobmap` converts the nonlinear
unknown back to the state vector used by the original ODE problem.

The nonlinear problem should represent a stage equation of the form
`M * z = outer_tmp + gamma1 * f(gamma2 * z + inner_tmp, p, t_c)`, equivalently
`g(z, p') = gamma1 * f(gamma2 * z + inner_tmp, p, t_c) + outer_tmp - M * z`.
Here `z` is the nonlinear stage unknown, `p` is the ODE parameter object, `t_c`
is the stage evaluation time, and `gamma1`, `gamma2`, `outer_tmp`, and
`inner_tmp` are supplied by the ODE algorithm.

# Fields

$(TYPEDFIELDS)
"""
struct ODENLStepData{NLProb, SetU0, SetGammaC, SetOuterTmp, SetInnerTmp, NLProbMap}
    """
    The structured `AbstractNonlinearProblem` template solved for each implicit
    ODE stage.
    """
    nlprob::NLProb
    """
    Callable used by the ODE algorithm to update the nonlinear problem's
    initial guess from the current stage data.
    """
    u0perm::SetU0
    """
    Callable used by the ODE algorithm to update the stage scaling factors and
    stage time/abscissa data used by the nonlinear problem.
    """
    set_γ_c::SetGammaC
    """
    Callable used by the ODE algorithm to update the `outer_tmp` vector in the
    nonlinear stage equation.
    """
    set_outer_tmp::SetOuterTmp
    """
    Callable used by the ODE algorithm to update the `inner_tmp` vector in the
    nonlinear stage equation.
    """
    set_inner_tmp::SetInnerTmp
    """
    Callable that maps the solution of `nlprob` back to the state vector or
    stage vector of the original ODE problem.
    """
    nlprobmap::NLProbMap
end
