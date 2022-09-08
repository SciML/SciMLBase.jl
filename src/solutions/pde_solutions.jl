"""
IMPORTANT:

These types are used by the following packages, please notify with an issue if you need to
modify these structures:
- `SciML/MethodOfLines.jl`
"""

"""
$(TYPEDEF)

Solution to a PDE, solved from an ODEProblem generated by a discretizer.

## Fields
- `u`: the solution to the PDE, as a dictionary of symbols to Arrays of values. The
  Arrays are of the same shape as the domain of the PDE. Time is always the first axis.
- `original_sol`: The original ODESolution that was used to generate this solution.
- `t`: the time points corresponding to the saved values of the ODE solution.
- `ivdomain`: The full list of domains for the independent variables. May be a grid for a
  discrete solution, or a vector/tuple of tuples for a continuous solution.
- `ivs`: The list of independent variables for the solution.
- `dvs`: The list of dependent variables for the solution.
- `disc_data`: Metadata about the discretization process and type.
- `prob`: The ODEProblem that was used to generate this solution.
- `alg`: The algorithm used to solve the ODEProblem.
- `interp`: Interpolations for the solution.
- `retcode`: The return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === :Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the ODEProblem.jl documentation.
"""
struct PDETimeSeriesSolution{T, N, uType, Disc, Sol, DType, tType, domType, ivType, dvType,
                             P, A,
                             IType} <: AbstractPDETimeSeriesSolution{T, N, uType, Disc}
    u::uType
    original_sol::Sol
    errors::DType
    t::tType
    ivdomain::domType
    ivs::ivType
    dvs::dvType
    disc_data::Disc
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    retcode::Symbol
end

"""
Dispatch for the following function should be implemented in each discretizer package, for their relevant metadata type `D`.
"""
function (sol::PDETimeSeriesSolution{T, N, S, D})(args...; kwargs...) where {T, N, S, D}
    error(ArgumentError("Call for PDETimeSeriesSolution not implemented for solution metadata type $D, please post an issue on the relevant discretizer package's github page."))
end

"""
$(TYPEDEF)

Solution to a PDE, solved from an NonlinearProblem generated by a discretizer.

## Fields
- `u`: the solution to the PDE, as a dictionary of symbols to Arrays of values. The
  Arrays are of the same shape as the domain of the PDE. Time is always the first axis.
- `original_sol`: The original NonlinearSolution that was used to generate this solution.
- `ivdomain`: The full list of domains for the independent variables. May be a grid for a
  discrete solution, or a vector/tuple of tuples for a continuous solution.
- `ivs`: The list of independent variables for the solution.
- `dvs`: The list of dependent variables for the solution.
- `disc_data`: Metadata about the discretization process and type.
- `prob`: The NonlinearProblem that was used to generate this solution.
- `alg`: The algorithm used to solve the NonlinearProblem.
- `interp`: Interpolations for the solution.
- `retcode`: The return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === :Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the ODEProblem.jl documentation.
"""
struct PDENoTimeSolution{T, N, uType, Disc, Sol, domType, ivType, dvType, P, A,
    IType} <: AbstractPDENoTimeSolution{T, N, uType, Disc}
    u::uType
    original_sol::Sol
    ivdomain::domType
    ivs::ivType
    dvs::dvType
    disc_data::Disc
    prob::P
    alg::A
    interp::IType
    retcode::Symbol
end

const PDESolution{T, N, S, D} = Union{PDETimeSeriesSolution{T, N, S, D}, PDENoTimeSolution{T, N, S, D}}

"""
Dispatch for the following function should be implemented in each discretizer package, for their relevant metadata type `D`.
"""
function (sol::PDENoTimeSolution{T, N, S, D})(args...; kwargs...) where {T, N, S, D}
    error(ArgumentError("Call for PDENoTimeSolution not implemented for solution metadata type $D, please post an issue on the relevant discretizer package's github page."))
end

"""
Intercept PDE wrapping. Please implement a method for the PDESolution types in your discretizer.
"""
function SciMLBase.wrap_sol(sol, metadata::AbstractDiscretizationMetadata{hasTime}) where {hasTime}
    if hasTime isa Val{true}
        return PDETimeSeriesSolution(sol, metadata)
    else
        return PDENoTimeSolution(sol, metadata)
    end
end
