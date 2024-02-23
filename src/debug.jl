DOMAINERROR_COMPLEX_MSG = """

DomainError detected in the user `f` function. This occurs when the domain of a function is violated.
For example, `log(-1.0)` is undefined because `log` of a real number is defined to only output real
numbers, but `log` of a negative number is complex valued and therefore Julia throws a DomainError
by default. Cases to be aware of include:

* `log(x)`, `sqrt(x)`, `cbrt(x)`, etc. where `x<0`
* `x^y` for `x<0` floating point `y` (example: `(-1.0)^(1/2) == im`)

Within the context of SciML, this error can occur within the solver process even if the domain constraint
would not be violated in the solution due to adaptivity. For example, an ODE solver or optimization
routine may check a step at `new_u` which violates the domain constraint, and if violated reject the
step and use a smaller `dt`. However, the throwing of this error will have halted the solving process.

Thus the recommended fix is to replace this function with the equivalent ones from NaNMath.jl
(https://github.com/JuliaMath/NaNMath.jl) which returns a NaN instead of an error. The solver will then
effectively use the NaN within the error control routines to reject the out of bounds step. Additionally,
one could perform a domain transformation on the variables so that such an issue does not occur in the
definition of `f`.

For more information, check out the following FAQ page:
https://docs.sciml.ai/Optimization/stable/API/FAQ/#The-Solver-Seems-to-Violate-Constraints-During-the-Optimization,-Causing-DomainErrors,-What-Can-I-Do-About-That?"""

FUNCTIONWRAPPERSWRAPPERS_MSG = """
No appropriate function wrapper found. This means that the auto-despecialization code used for the reduction
of compile times has failed. This is most likely due to an issue internal to the function `prob.f` that is
found upon evaluation of the model. To work around this issue, use `SciMLBase.FullSpecialize`, like:

```julia
ODEProblem{iip,SciMLBase.FullSpecialize}(f,u0,tspan,p)
```

where `iip` is either true or false depending on the in-placeness of the definition of `f` (i.e. for ODEs
if `f` has 3 arguments `(u,p,t)` then it's false, otherwise `f(du,u,p,t)` is true).

For more information on the control of specialization options, please see the documentation at:

https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Choices

If one wants way more detail than necessary on why the function wrappers exist and what they are doing, see:

https://sciml.ai/news/2022/09/21/compile_time/"""

const NO_PARAMETERS_ARITHMETIC_ERROR_MESSAGE = """

An arithmetic operation was performed on a NullParameters object. This means no parameters were passed
into the AbstractSciMLProblem (e.x.: ODEProblem) but the parameters object `p` was used in an arithmetic
expression. Two common reasons for this issue are:

1. Forgetting to pass parameters into the problem constructor. For example, `ODEProblem(f,u0,tspan)` should
be `ODEProblem(f,u0,tspan,p)` in order to use parameters.

2. Using the wrong function signature. For example, with `ODEProblem`s the function signature is always
`f(du,u,p,t)` for the in-place form or `f(u,p,t)` for the out-of-place form. Note that the `p` argument
will always be in the function signature regardless of if the problem is defined with parameters!
"""

function __init__()
    Base.Experimental.register_error_hint(DomainError) do io, e
        if e isa DomainError &&
           occursin(
            "will only return a complex result if called with a complex argument. Try ",
            e.msg)
            println(io, DOMAINERROR_COMPLEX_MSG)
        end
    end

    Base.Experimental.register_error_hint(MethodError) do io, e, args, kwargs
        if e isa MethodError && NullParameters in args
            println(io, NO_PARAMETERS_ARITHMETIC_ERROR_MESSAGE)
        end
    end

    Base.Experimental.register_error_hint(FunctionWrappersWrappers.NoFunctionWrapperFoundError) do io,
    e
        println(io, FUNCTIONWRAPPERSWRAPPERS_MSG)
    end
end
