abstract type DebugMode end
struct NoDebug <: DebugMode end
struct VerboseDebug <: DebugMode end

DETAILED_INFORMATION = """
Note that detailed debugging information adds a small amount of overhead to SciML solves
which can be disabled with the keyword argument `debug = NoDebug()`.

The detailed original error message information from Julia reproduced below:
"""

DEFAULT_DEBUG_MSG = """
Error detected inside of the run of the solver. For more detailed information about the error
with SciML context and recommendations, try adding the keyword argument `debug = VerboseDebug()`

$DETAILED_INFORMATION
"""

UNKNOWN_MSG = """
An unclassified error occured during the solver process and SciML's `VerboseDebug` mode cannot give
any more information. You can help improve the debug mode messages by reporting this error to
https://github.com/SciML/SciMLBase.jl/issues with a reproducer of the error from which a high
level description can be added.

$DETAILED_INFORMATION
"""

DOMAINERROR_COMPLEX_MSG = """
DomainError detected in the user `f` function. This occurs when the domain of a function is violated.
For example, `log(-1.0)` is undefined because `log` of a real number is defined to only output real
numbers, but `log` of a negative number is complex valued and therefore Julia throws a DomainError
by default. Cases to be aware of include:

* `log(x)`, `sqrt(x)`, `cbrt(x)`, etc. where `x<0`
* `x^y` for `x<0` floating point `y` (example: `(-1.0)^(1/2) == im`)

Within the context of SciML, this error can occur within the solver process even if the domain constriant
would not be violated in the solution due to adaptivity. For example, an ODE solver or optimization
routine may check a step at `new_u` which violates the domain constraint, and if violated reject the
step and use a smaller `dt`. However, the throwing of this error will have haulted the solving process.

Thus the recommended fix is to replace this function with the equivalent ones from NaNMath.jl
(https://github.com/JuliaMath/NaNMath.jl) which returns a NaN instead of an error. The solver will then
effectively use the NaN within the error control routines to reject the out of bounds step. Additionally,
one could perform a domain transformation on the variables so that such an issue does not occur in the
definition of `f`.

For more information, check out the following FAQ page:
https://docs.sciml.ai/Optimization/stable/API/FAQ/#The-Solver-Seems-to-Violate-Constraints-During-the-Optimization,-Causing-DomainErrors,-What-Can-I-Do-About-That?

$DETAILED_INFORMATION
"""

struct VerboseDebugFunction{F}
    f::F
end
function (f::VerboseDebugFunction)(args...) 
    try
        f.f(args...)
    catch e
        if e isa DomainError && occursin("will only return a complex result if called with a complex argument. Try ",e.msg)
            println(DOMAINERROR_COMPLEX_MSG)
        else
            println(UNKNOWN_MSG)
        end
        throw(e)
    end
end

debugwrapfun(f,debug::VerboseDebug) = VerboseDebugFunction(f)
debugwrapfun(f,debug::NoDebug) = f