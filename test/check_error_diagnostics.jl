using Test, SciMLBase, Logging

# A minimal integrator that goes through the generic `SciMLBase.check_error`
# rather than specializing it, so every exit branch can be driven directly.
struct DiagSolution
    retcode::SciMLBase.ReturnCode.T
end
SciMLBase.solution_new_retcode(::DiagSolution, code) = DiagSolution(code)

mutable struct DiagOpts
    verbose::Bool
    maxiters::Int
    dtmin::Float64
    adaptive::Bool
    force_dtmin::Bool
    unstable_check::Any
end

mutable struct DiagIntegrator{Alg, IIP, U, T} <: SciMLBase.DEIntegrator{Alg, IIP, U, T}
    u::U
    uprev::U
    t::T
    dt::T
    p::Any
    f::Any
    iter::Int
    accept_step::Bool
    step_failed::Bool
    sol::DiagSolution
    opts::DiagOpts

    function DiagIntegrator(; kwargs...)
        opts = DiagOpts(true, 100, 1.0e-10, true, false, (dt, u, p, t) -> false)
        integ = new{Bool, true, Vector{Float64}, Float64}(
            [1.0], [1.0], 0.5, 0.1, nothing, nothing, 1, true, false,
            DiagSolution(ReturnCode.Default), opts
        )
        for (k, v) in kwargs
            k in fieldnames(DiagOpts) ? setfield!(opts, k, v) : setfield!(integ, k, v)
        end
        return integ
    end
end

SciMLBase.last_step_failed(integrator::DiagIntegrator) = integrator.step_failed
SciMLBase.postamble!(::DiagIntegrator) = nothing

const NUMERIC_MARK = " <numeric>"
const ESTIMATE_MARK = " <estimate>"
SciMLBase.log_numerical_instability(::DiagIntegrator; jacobian_logging = true) = NUMERIC_MARK
SciMLBase.log_error_estimate(::DiagIntegrator) = ESTIMATE_MARK

function logged_check_error(integrator)
    io = IOBuffer()
    code = with_logger(SimpleLogger(io)) do
        SciMLBase.check_error(integrator)
    end
    return code, String(take!(io))
end

# Every non-Success exit reports the extended diagnostics, not just the dtmin and
# instability branches that carried them historically.
@testset "diagnostics on $(name)" for (name, integrator, expected) in (
        ("dt NaN", DiagIntegrator(dt = NaN), ReturnCode.DtNaN),
        ("maxiters", DiagIntegrator(iter = 101), ReturnCode.MaxIters),
        (
            "dt below dtmin", DiagIntegrator(dt = 1.0e-14, accept_step = false),
            ReturnCode.DtLessThanMin,
        ),
        (
            "dt below eps", DiagIntegrator(dt = 1.0e-30, dtmin = 0.0, accept_step = false),
            ReturnCode.Unstable,
        ),
        (
            "unstable check", DiagIntegrator(unstable_check = (dt, u, p, t) -> true),
            ReturnCode.Unstable,
        ),
        ("newton failure", DiagIntegrator(step_failed = true), ReturnCode.ConvergenceFailure),
    )
    code, logs = logged_check_error(integrator)
    @test code == expected
    @test occursin(NUMERIC_MARK, logs)
    @test occursin(ESTIMATE_MARK, logs)
end

@testset "successful check reports nothing" begin
    code, logs = logged_check_error(DiagIntegrator())
    @test code == ReturnCode.Success
    @test isempty(logs)
end

@testset "silent verbosity skips diagnostic work" begin
    integrator = DiagIntegrator(iter = 101, verbose = false)
    code, logs = logged_check_error(integrator)
    @test code == ReturnCode.MaxIters
    @test isempty(logs)
    @test SciMLBase.exit_diagnostic(integrator, :max_iters) == ""
end

@testset "default hooks are empty" begin
    @test SciMLBase.log_error_estimate(nothing) == ""
    @test SciMLBase.log_numerical_instability(nothing) == ""
end
