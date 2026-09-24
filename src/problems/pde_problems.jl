"""
$(TYPEDEF)

Concrete wrapper for PDE problems before discretization.

`PDEProblem` stores a symbolic or package-specific PDE problem object together
with extrapolation and spatial-domain metadata used by discretization backends.
SciMLBase owns only this lightweight wrapper; concrete PDE semantics and the
conversion to ODE, nonlinear, optimization, or linear problems are supplied by
downstream discretization packages through `discretize` and
`symbolic_discretize`.

# Fields

$(TYPEDFIELDS)
"""
struct PDEProblem{P, E, S} <: AbstractPDEProblem
    prob::P
    extrapolation::E
    space::S
end
