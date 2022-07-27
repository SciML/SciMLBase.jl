"""
isautodifferentiable(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
isautodifferentiable(alg::AbstractSciMLAlgorithm) = false

"""
forwarddiffs_model(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses ForwardDiff.jl
on the model function is called with ForwardDiff.jl

Defaults to false as only pure-Julia algorithms can have this be true.
"""
forwarddiffs_model(alg::AbstractSciMLAlgorithm) = false

"""
forwarddiffs_model_time(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses ForwardDiff.jl
on the model `f(u,p,t)` function is called with ForwardDiff.jl on the `t` argument.

Defaults to false as only a few pure-Julia algorithms (Rosenbrock methods)
have this as true
"""
forwarddiffs_model_time(alg::AbstractSciMLAlgorithm) = false

"""
allows_arbitrary_number_types(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
allows_arbitrary_number_types(alg::AbstractSciMLAlgorithm) = false

"""
allowscomplex(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
having complex numbers as the state variables.

Defaults to false.
"""
allowscomplex(alg::AbstractSciMLAlgorithm) = false

"""
isadaptive(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses adaptivity,
i.e. has a non-quasi-static compute graph.

Defaults to true.
"""
isadaptive(alg::AbstractDEAlgorithm) = true
# Default to assuming adaptive, safer error("Adaptivity algorithm trait not set.")

"""
isdiscrete(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm allows for
discrete state values, such as integers.

Defaults to false.
"""
isdiscrete(alg::AbstractDEAlgorithm) = false
