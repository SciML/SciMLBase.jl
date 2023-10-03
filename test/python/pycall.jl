using PyCall, SciMLBase, OrdinaryDiffEq

# PyCall only works when PyCall is in the default environment :'(
import Pkg
function with_pycall_in_default_environment(f)
    path = Pkg.project().path
    Pkg.activate()
    install = "PyCall" ∉ keys(Pkg.project().dependencies)
    install && Pkg.add("PyCall")
    try
        f()
    finally
        install && Pkg.rm("PyCall")
        Pkg.activate(path)
    end
end

with_pycall_in_default_environment() do
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
end # with_pycall_in_default_environment
