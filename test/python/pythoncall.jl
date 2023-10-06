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

    # TODO: test the types and shapes of sol.t and de.transpose(de.stack(sol.u)) but don't actually plot them in CI
    # pyexec("""
    # import matplotlib.pyplot as plt

    # plt.plot(sol.t, de.transpose(de.stack(sol.u))) # :( fails without the conversion
    # plt.show()
    # """, @__MODULE__)

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
