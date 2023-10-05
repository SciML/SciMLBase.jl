using PyCall, SciMLBase, OrdinaryDiffEq

py""" # This is a mess because normal site-packages is not writeable in CI
import subprocess, sys, site
subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'julia'])
sys.path.append(site.getusersitepackages())
"""

@testset "numargs" begin
    py"""
    def three_arg(a, b, c):
        return a + b + c

    def four_arg(a, b, c, d):
        return a + b + c + d

    class MyClass:
        def three_arg_method(self, a, b, c):
            return a + b + c

        def four_arg_method(self, a, b, c, d):
            return a + b + c + d
    """

    @test SciMLBase.numargs(py"three_arg") === 3
    @test SciMLBase.numargs(py"four_arg") === 4
    x = py"MyClass()"
    @test SciMLBase.numargs(x.three_arg_method) === 3
    @test SciMLBase.numargs(x.four_arg_method) === 4
end

@testset "solution handling" begin
    py"""
    from julia import OrdinaryDiffEq as ode

    def f(u,p,t):
        return -u

    u0 = 0.5
    tspan = (0., 1.)
    prob = ode.ODEProblem(f, u0, tspan)
    sol = ode.solve(prob, ode.Tsit5())
    """
    @test py"sol" isa ODESolution
end
