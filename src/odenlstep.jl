"""
    $(TYPEDEF)

A collection of all the data required for custom ODE Nonlinear problem solving
"""
struct ODENLStepData{NLProb, SetU0, SetGammaC, SetOuterTmp, SetInnerTmp, NLProbMap}
    """
    The `AbstractNonlinearProblem` to define custom nonlinear problems to be used for
    implicit time discretizations. This allows to use extra structure of the ODE function (e.g.
    multi-level structure). The nonlinear function must match that form of the function implicit
    ODE integration algorithms need do solve the a nonlinear problems,
    specifically of the form `M*z = outer_tmp + γ₁⋅f(γ₂⋅z+inner_tmp,p,t_c)`.
    Here `z` is the stage solution vector, `p` is the parameter of the ODE problem, `t_c` is
    the time of evaluation (`t_c = t + c*dt`), `γ₁` and `γ₂` are some scaling factors determined
    by the solver algorithm and the temporary variables are some compatible vectors set by the specific solver.
    The inner nonlinear function of the nonlinear problem is in general of the form `g(z,p') = 0` such that
    `g(z,p') = γ₁⋅f(γ₂⋅z+inner_tmp,p,t_c) + outer_tmp - M*z = 0`.
    """
    nlprob::NLProb
    u0perm::SetU0
    set_γ_c::SetGammaC
    set_outer_tmp::SetOuterTmp
    set_inner_tmp::SetInnerTmp
    """
    A function which takes the solution of `nlprob` and returns
    the state vector of the original problem.
    """
    nlprobmap::NLProbMap
end
