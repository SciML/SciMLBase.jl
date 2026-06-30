using SciMLBase, FunctionProperties, Test

# The SciMLBaseFunctionPropertiesExt extension makes `hasbranching` look through an
# `AbstractSciMLFunction` to its wrapped right-hand side `f.f`, rather than analyzing the functor
# (whose operator-dispatch/forwarding branches are value-independent plumbing).
@test hasmethod(FunctionProperties.hasbranching, Tuple{SciMLBase.AbstractSciMLFunction, Vararg})

free!(du, u, p, t) = (du[1] = u[1] * u[1]; nothing)
branchy!(du, u, p, t) = (du[1] = u[1] > 0 ? u[1] : -u[1]; nothing)
u = [1.0]
du = [0.0]
p = SciMLBase.NullParameters()

@test !hasbranching(ODEFunction(free!), copy(du), u, p, 0.0)
@test hasbranching(ODEFunction(branchy!), copy(du), u, p, 0.0)
