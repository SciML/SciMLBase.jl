using FunctionProperties, SciMLBase, Test

# The SciMLBaseFunctionPropertiesExt extension makes FunctionProperties look through an
# `AbstractSciMLFunction` to its wrapped right-hand side `f.f`, rather than analyzing the functor.
@test hasmethod(FunctionProperties.hasbranching, Tuple{SciMLBase.AbstractSciMLFunction, Vararg})
@test hasmethod(FunctionProperties.islinear, Tuple{SciMLBase.AbstractSciMLFunction, Vararg})
@test hasmethod(FunctionProperties.isautonomous, Tuple{SciMLBase.AbstractSciMLFunction, Vararg})

free!(du, u, p, t) = (du[1] = u[1] * u[1]; nothing)
branchy!(du, u, p, t) = (du[1] = u[1] > 0 ? u[1] : -u[1]; nothing)
linear_autonomous(u, p, t) = p[1] * u
linear_nonautonomous(u, p, t) = p[1] * u + t
u = [1.0]
du = [0.0]
p = [2.0]

@test !FunctionProperties.hasbranching(ODEFunction(free!), copy(du), u, p, 0.0)
@test FunctionProperties.hasbranching(ODEFunction(branchy!), copy(du), u, p, 0.0)
@test FunctionProperties.islinear(ODEFunction(linear_autonomous), u, p, 0.0)
@test FunctionProperties.isautonomous(ODEFunction(linear_autonomous), u, p, 0.0)
@test !FunctionProperties.isautonomous(ODEFunction(linear_nonautonomous), u, p, 0.0)
