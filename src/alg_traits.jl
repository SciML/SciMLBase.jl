"""
isautodifferentiable(alg::DEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
isautodifferentiable(alg::DEAlgorithm) = false

"""
allows_arbitrary_number_types(alg::DEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
allows_arbitrary_number_types(alg::DEAlgorithm) = false

"""
allowscomplex(alg::DEAlgorithm)

Trait declaration for whether an algorithm is compatible with
having complex numbers as the state variables.

Defaults to false.
"""
allowscomplex(alg::DEAlgorithm) = false

"""
isadaptive(alg::DEAlgorithm)

Trait declaration for whether an algorithm uses adaptivity,
i.e. has a non-quasi-static compute graph.

Defaults to true.
"""
isadaptive(alg::DEAlgorithm) = true # Default to assuming adaptive, safer error("Adaptivity algorithm trait not set.")

"""
isdiscrete(alg::DEAlgorithm)

Trait declaration for whether an algorithm allows for
discrete state values, such as integers.

Defaults to false.
"""
isdiscrete(alg::DEAlgorithm) = false
