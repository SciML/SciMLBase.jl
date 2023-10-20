# PyCall and PythonCall must use the same Python interpreter. This environment variable
# tells PythonCall to use the same Python interpreter as PyCall. See
# https://github.com/JuliaPy/PythonCall.jl/blob/5f56a9b96b867a9f6742ab1d1e2361abd844e19f/docs/src/pycall.md#tips
ENV["JULIA_PYTHONCALL_EXE"]="@PyCall"

using DifferentialEquations, PythonCall

@testset "Use of DifferentialEquations through PythonCall with user code written in Python" begin
    pyexec("""
    from juliacall import Main
    Main.seval("using DifferentialEquations")
    de = Main.seval("DifferentialEquations")

    def f(u,p,t):
        return -u

    u0 = 0.5
    tspan = (0., 1.)
    prob = de.ODEProblem(f, u0, tspan)
    sol = de.solve(prob)
    """, @__MODULE__)
    @test pyconvert(Any, pyeval("sol", @__MODULE__)) isa ODESolution

    pyexec("""
    def f(u,p,t):
        x, y, z = u
        sigma, rho, beta = p
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,8/3]
    prob = de.ODEProblem(f, u0, tspan, p)
    sol = de.solve(prob,saveat=0.01)
    """, @__MODULE__)
    @test pyconvert(Any, pyeval("sol", @__MODULE__)) isa ODESolution

    # Test that the types and shapes of sol.t and de.transpose(de.stack(sol.u)) are
    # compatible with matplotlib, but don't actually plot anything.
    pyexec("""
    u2 = de.transpose(de.stack(sol.u))
    ok = sol.t.shape == (10001,) and \
         u2.shape == (10001, 3) and \
         sol.t[0] == 0 and \
         sol.t[-1] == 100 and \
         type(u2[4123, 2]) == float
    """, @__MODULE__)
    @test pyconvert(Any, pyeval("ok", @__MODULE__))

    @pyexec """
    jul_f = Main.seval(""\"
    function f(du,u,p,t)
        x, y, z = u
        sigma, rho, beta = p
        du[1] = sigma * (y - x)
        du[2] = x * (rho - z) - y
        du[3] = x * y - beta * z
    end""\")
    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,2.66]
    prob = de.ODEProblem(jul_f, u0, tspan, p)
    sol = de.solve(prob)
    """
    @test pyconvert(Any, pyeval("sol", @__MODULE__)) isa ODESolution

    pyexec("""
    def f(u,p,t):
        return 1.01*u

    def g(u,p,t):
        return 0.87*u

    u0 = 0.5
    tspan = (0.0,1.0)
    prob = de.SDEProblem(f,g,u0,tspan)
    sol = de.solve(prob,reltol=1e-3,abstol=1e-3)
    """, @__MODULE__)
end

@testset "promotion" begin
    _u0 = pyconvert(Any, pyeval("""de.SciMLBase.prepare_initial_state([1.0, 0, 0])""", @__MODULE__))
    @test _u0 isa Vector{Float64}
end
