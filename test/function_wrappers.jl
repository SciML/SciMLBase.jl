using Test, SciMLBase

u = [1.0, 2.0]
p = [3.0]

f_oop(u, p, t) = u .* p[1] .+ t
function f_iip(du, u, p, t)
    du .= u .* p[1] .+ t
    return nothing
end

@testset "time wrappers" begin
    expected = [3.5, 6.5]

    time_gradient_oop = SciMLBase.TimeGradientWrapper(f_oop, u, p)
    @test SciMLBase.isinplace(time_gradient_oop) === false
    @test time_gradient_oop(0.5) == expected

    time_gradient_iip = SciMLBase.TimeGradientWrapper(f_iip, u, p)
    @test SciMLBase.isinplace(time_gradient_iip) === true
    out = similar(u)
    @test time_gradient_iip(out, 0.5) === nothing
    @test out == expected
    @test time_gradient_iip(0.5) == expected

    time_derivative_oop = SciMLBase.TimeDerivativeWrapper(f_oop, u, p)
    @test SciMLBase.isinplace(time_derivative_oop) === false
    @test time_derivative_oop(0.5) == expected

    time_derivative_iip = SciMLBase.TimeDerivativeWrapper(f_iip, u, p)
    @test SciMLBase.isinplace(time_derivative_iip) === true
    out = similar(u)
    @test time_derivative_iip(out, 0.5) === nothing
    @test out == expected
    @test time_derivative_iip(0.5) == expected
end

@testset "state wrappers" begin
    t = 0.5
    expected = [3.5, 6.5]
    override_expected = [3.0, 5.0]

    u_jacobian_oop = SciMLBase.UJacobianWrapper(f_oop, t, p)
    @test SciMLBase.isinplace(u_jacobian_oop) === false
    @test u_jacobian_oop(u) == expected
    @test u_jacobian_oop(u, [2.0], 1.0) == override_expected

    u_jacobian_iip = SciMLBase.UJacobianWrapper(f_iip, t, p)
    @test SciMLBase.isinplace(u_jacobian_iip) === true
    out = similar(u)
    @test u_jacobian_iip(out, u) === nothing
    @test out == expected
    @test u_jacobian_iip(u) == expected
    out = similar(u)
    @test u_jacobian_iip(out, u, [2.0], 1.0) === nothing
    @test out == override_expected
    @test u_jacobian_iip(u, [2.0], 1.0) == override_expected

    u_derivative_oop = SciMLBase.UDerivativeWrapper(f_oop, t, p)
    @test SciMLBase.isinplace(u_derivative_oop) === false
    @test u_derivative_oop(u) == expected

    u_derivative_iip = SciMLBase.UDerivativeWrapper(f_iip, t, p)
    @test SciMLBase.isinplace(u_derivative_iip) === true
    out = similar(u)
    @test u_derivative_iip(out, u) === nothing
    @test out == expected
    @test u_derivative_iip(u) == expected
end
