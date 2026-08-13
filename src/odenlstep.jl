"""
    ODENLStepData(nlprob, u0perm, set_gamma_c, set_outer_tmp, set_inner_tmp, nlprobmap)

A collection of hooks for custom nonlinear stage solves in implicit ODE and DAE
algorithms.

`ODENLStepData` lets an `ODEFunction`, `SplitFunction` or `DAEFunction` provide a structured
`AbstractNonlinearProblem` template for solver packages that form implicit stage
equations. Before each nonlinear solve, the algorithm updates the stage guess,
scaling factors, time information, and temporary vectors through the stored
setter callables. After the nonlinear solve, `nlprobmap` converts the nonlinear
unknown back to the state vector used by the original problem.

## Mass-matrix form

For `M * du/dt = f(u, p, t)` the nonlinear problem should represent a stage equation of the
form `M * z = outer_tmp + gamma1 * f(gamma2 * z + inner_tmp, p, t_c)`, equivalently
`g(z, p') = gamma1 * f(gamma2 * z + inner_tmp, p, t_c) + outer_tmp - M * z`.
Here `z` is the nonlinear stage unknown, `p` is the ODE parameter object, `t_c`
is the stage evaluation time, and `gamma1`, `gamma2`, `outer_tmp`, and
`inner_tmp` are supplied by the ODE algorithm.

## Fully implicit form

For `0 = F(du, u, p, t)` (a `DAEFunction`) the stage equation has the same shape, with both
arguments of `F` affine in the stage unknown:
`g(z, p') = F(gamma1 * z + outer_tmp, gamma2 * z + inner_tmp, p, t_c)`.
`gamma2` and `inner_tmp` build the state argument from the stage unknown exactly as in the
mass-matrix form, while `gamma1` and `outer_tmp` build the derivative argument. Taking the
stage unknown to be the stage state (`gamma2 = 1`, `inner_tmp = 0`), a BDF-type step with
`du ≈ (u - tmp) / (γ * dt)` gives `gamma1 = inv(γ * dt)` and `outer_tmp = -tmp / (γ * dt)`.
With that convention `gamma1` is the `gamma` of the `DAEFunction` Jacobian signature
`jac(J, du, u, p, gamma, t)`: the Jacobian of the stage residual with respect to `z` is
`gamma1 * dF/d(du) + dF/du`.

# Fields

$(TYPEDFIELDS)

# Extension Rules

Symbolic-system packages construct this value and store it as the `nlstep_data` of an
`ODEFunction`, `SplitFunction` or `DAEFunction`. Solver packages may consume the six fields
through their callable contracts, but must not assume concrete callable types or mutate the
container. Each setter must update the object it closes over consistently with `nlprob`, and
`nlprobmap` must map a completed nonlinear solution back to the stage representation of the
original problem.
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
