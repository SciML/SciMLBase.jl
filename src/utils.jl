"""
$(SIGNATURES)

Returns the number of arguments of `f` for each method.
"""
function numargs(f)
    if hasfield(typeof(f), :r) && typeof(f.r).name.name == :RObject ||
       typeof(f).name.name == :RFunction
        # Uses the RCall form to grab the parameter length
        return [length(unsafe_load(f.r.p).formals)]
    else
        return [num_types_in_tuple(m.sig) - 1 for m in methods(f)] #-1 since f is the first parameter
    end
end

function numargs(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{
    T,
    V,
    W,
    I,
}) where {
    T,
    V,
    W,
    I,
}
    (length(T),)
end

numargs(f::ComposedFunction) = numargs(f.inner)

"""
$(SIGNATURES)

Get the number of parameters of a Tuple type, i.e. the number of fields.
"""
function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end

const NO_METHODS_ERROR_MESSAGE = """
                                 No methods were found for the model function passed to the equation solver.
                                 The function `f` needs to have dispatches, for example, for an ODEProblem
                                 `f` must define either `f(u,p,t)` or `f(du,u,p,t)`. For more information
                                 on how the model function `f` should be defined, consult the docstring for
                                 the appropriate `AbstractSciMLFunction`.
                                 """

struct NoMethodsError <: Exception
    fname::String
end

function Base.showerror(io::IO, e::NoMethodsError)
    println(io, NO_METHODS_ERROR_MESSAGE)
    print(io, "Offending function: ")
    printstyled(io, e.fname; bold = true, color = :red)
end

const TOO_MANY_ARGUMENTS_ERROR_MESSAGE = """
                                         All methods for the model function `f` had too many arguments. For example,
                                         an ODEProblem `f` must define either `f(u,p,t)` or `f(du,u,p,t)`. This error
                                         can be thrown if you define an ODE model for example as `f(du,u,p1,p2,t)`.
                                         For more information on the required number of arguments for the function
                                         you were defining, consult the documentation for the `SciMLProblem` or
                                         `SciMLFunction` type that was being constructed.

                                         A common reason for this occurrence is due to following the MATLAB or SciPy
                                         convention for parameter passing, i.e. to add each parameter as an argument.
                                         In the SciML convention, if you wish to pass multiple parameters, use a
                                         struct or other collection to hold the parameters. For example, here is the
                                         parameterized Lorenz equation:

                                         ```julia
                                         function lorenz(du,u,p,t)
                                           du[1] = p[1]*(u[2]-u[1])
                                           du[2] = u[1]*(p[2]-u[3]) - u[2]
                                           du[3] = u[1]*u[2] - p[3]*u[3]
                                          end
                                          u0 = [1.0;0.0;0.0]
                                          p = [10.0,28.0,8/3]
                                          tspan = (0.0,100.0)
                                          prob = ODEProblem(lorenz,u0,tspan,p)
                                         ```

                                         Notice that `f` is defined with a single `p`, an array which matches the definition
                                         of the `p` in the `ODEProblem`. Note that `p` can be any Julia struct.
                                         """

struct TooManyArgumentsError <: Exception
    fname::String
    f::Any
end

function Base.showerror(io::IO, e::TooManyArgumentsError)
    println(io, TOO_MANY_ARGUMENTS_ERROR_MESSAGE)
    print(io, "Offending function: ")
    printstyled(io, e.fname; bold = true, color = :red)
    println(io, "\nMethods:")
    println(io, methods(e.f))
end

const TOO_FEW_ARGUMENTS_ERROR_MESSAGE_OPTIMIZATION = """
                                        All methods for the model function `f` had too few arguments. For example,
                                        an OptimizationProblem `f` must define `f(u,p)` where `u` is the optimization
                                        state and `p` are the parameters of the optimization (commonly, the hyperparameters
                                        of the simulation).

                                        A common reason for this error is from defining a single-input loss function
                                        `f(u)`. While parameters are not required, a loss function which takes parameters
                                        is required, i.e. `f(u,p)`. If you have a function `f(u)`, ignored parameters
                                        can be easily added using a closure, i.e. `OptimizationProblem((u,_)->f(u),...)`.

                                        For example, here is a parameterized optimization problem:

                                        ```julia
                                        using Optimization, OptimizationOptimJL
                                        rosenbrock(u,p) =  (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
                                        u0 = zeros(2)
                                        p  = [1.0,100.0]

                                        prob = OptimizationProblem(rosenbrock,u0,p)
                                        sol = solve(prob,NelderMead())
                                        ```

                                        and a parameter-less example:

                                        ```julia
                                        using Optimization, OptimizationOptimJL
                                        rosenbrock(u,p) =  (1 - u[1])^2 + (u[2] - u[1]^2)^2
                                        u0 = zeros(2)

                                        prob = OptimizationProblem(rosenbrock,u0)
                                        sol = solve(prob,NelderMead())
                                        ```
                                        """

const TOO_FEW_ARGUMENTS_ERROR_MESSAGE = """
                                        All methods for the model function `f` had too few arguments. For example,
                                        an ODEProblem `f` must define either `f(u,p,t)` or `f(du,u,p,t)`. This error
                                        can be thrown if you define an ODE model for example as `f(u,t)`. The parameters
                                        `p` are not optional in the definition of `f`! For more information on the required
                                        number of arguments for the function you were defining, consult the documentation
                                        for the `SciMLProblem` or `SciMLFunction` type that was being constructed.

                                        For example, here is the no parameter Lorenz equation. The two valid versions
                                        are out of place:

                                        ```julia
                                        function lorenz(u,p,t)
                                          du1 = 10.0*(u[2]-u[1])
                                          du2 = u[1]*(28.0-u[3]) - u[2]
                                          du3 = u[1]*u[2] - 8/3*u[3]
                                          [du1,du2,du3]
                                         end
                                         u0 = [1.0;0.0;0.0]
                                         tspan = (0.0,100.0)
                                         prob = ODEProblem(lorenz,u0,tspan)
                                        ```

                                        and in-place:

                                        ```julia
                                        function lorenz!(du,u,p,t)
                                          du[1] = 10.0*(u[2]-u[1])
                                          du[2] = u[1]*(28.0-u[3]) - u[2]
                                          du[3] = u[1]*u[2] - 8/3*u[3]
                                         end
                                         u0 = [1.0;0.0;0.0]
                                         tspan = (0.0,100.0)
                                         prob = ODEProblem(lorenz!,u0,tspan)
                                        ```
                                        """

struct TooFewArgumentsError <: Exception
    fname::String
    f::Any
    isoptimization::Bool
end

function Base.showerror(io::IO, e::TooFewArgumentsError)
    if e.isoptimization
        println(io, TOO_FEW_ARGUMENTS_ERROR_MESSAGE_OPTIMIZATION)
    else
        println(io, TOO_FEW_ARGUMENTS_ERROR_MESSAGE)
    end
    print(io, "Offending function: ")
    printstyled(io, e.fname; bold = true, color = :red)
    println(io, "\nMethods:")
    println(io, methods(e.f))
end

const ARGUMENTS_ERROR_MESSAGE = """
                                Methods dispatches for the model function `f` do not match the required number.
                                For example, an ODEProblem `f` must define either `f(u,p,t)` or `f(du,u,p,t)`.
                                This error can be thrown if you define an ODE model for example as `f(u,t)`
                                and `f(u,p,t,x,y)` as both of those are not valid dispatches! For more information
                                on the required dispatches for the given model function, consult the documentation
                                for the appropriate `SciMLProblem` or `AbstractSciMLFunction`.
                                """

struct FunctionArgumentsError <: Exception
    fname::String
    f::Any
end

function Base.showerror(io::IO, e::FunctionArgumentsError)
    println(io, ARGUMENTS_ERROR_MESSAGE)
    print(io, "Offending function: ")
    printstyled(io, e.fname; bold = true, color = :red)
    println(io, "\nMethods:")
    println(io, methods(e.f))
end

"""
    isinplace(f, inplace_param_number[,fname="f"])
    isinplace(f::AbstractSciMLFunction[, inplace_param_number])

Check whether a function operates in place by comparing its number of arguments
to the expected number. If `f` is an `AbstractSciMLFunction`, then the type
parameter is assumed to be correct and is used. Otherwise `inplace_param_number`
is checked against the methods table, where `inplace_param_number` is the number
of arguments for the in-place dispatch. The out-of-place dispatch is assumed
to have one less. If neither of these dispatches exist, an error is thrown.
If the error is thrown, `fname` is used to tell the user which function has the
incorrect dispatches.

`iip_preferred` means that if `inplace_param_number=4` and methods of both 3 and
for 4 args exist, then it will be chosen as in-place. `iip_dispatch` flips this
decision.

If `has_two_dispatches = false`, then it is assumed that there is only one correct
dispatch, i.e. `f(u,p)` for OptimizationFunction, and thus the check for the oop
form is disabled and the 2-argument signature is ensured to be matched.

# See also
* [`numargs`](@ref numargs)
"""
function isinplace(f, inplace_param_number, fname = "f", iip_preferred = true;
    has_two_dispatches = true, isoptimization = false)
    nargs = numargs(f)
    iip_dispatch = any(x -> x == inplace_param_number, nargs)
    oop_dispatch = any(x -> x == inplace_param_number - 1, nargs)

    if length(nargs) == 0
        throw(NoMethodsError(fname))
    end

    if !iip_dispatch && !oop_dispatch && !isoptimization
        if all(x -> x > inplace_param_number, nargs)
            throw(TooManyArgumentsError(fname, f))
        elseif all(x -> x < inplace_param_number - 1, nargs) && has_two_dispatches
            # Possible extra safety?
            # Find if there's a `f(args...)` dispatch
            # If so, no error
            _parameters = if methods(f).ms[1].sig isa UnionAll
                Base.unwrap_unionall(methods(f).ms[1].sig).parameters
            else
                methods(f).ms[1].sig.parameters
            end
            
            for i in 1:length(nargs)
                if nargs[i] < inplace_param_number &&
                   any(isequal(Vararg{Any}),_parameters)
                    # If varargs, assume iip
                    return iip_preferred
                end
            end

            # No varargs detected, error that there are dispatches but not the right ones

            throw(TooFewArgumentsError(fname, f, isoptimization))
        else
            throw(FunctionArgumentsError(fname, f))
        end
    elseif oop_dispatch && !iip_dispatch && !has_two_dispatches

        # Possible extra safety?
        # Find if there's a `f(args...)` dispatch
        # If so, no error
        for i in 1:length(nargs)
            if nargs[i] < inplace_param_number &&
               any(isequal(Vararg{Any}), methods(f).ms[1].sig.parameters)
                # If varargs, assume iip
                return iip_preferred
            end
        end

        throw(TooFewArgumentsError(fname, f, isoptimization))
    else
        if iip_preferred
            # Equivalent to, if iip_dispatch exists, treat as iip
            # Otherwise, it's oop
            iip_dispatch
        else
            # Equivalent to, if oop_dispatch exists, treat as oop
            # Otherwise, it's iip
            !oop_dispatch
        end
    end
end

isinplace(f::AbstractSciMLFunction{iip}) where {iip} = iip
function isinplace(f::AbstractSciMLFunction{iip}, inplace_param_number,
    fname = nothing) where {iip}
    iip
end

"""
    @CSI_str cmd

Create an ANSI escape sequence string for the CSI command `cmd`.
"""
macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end

const TYPE_COLOR = CSI"38;2;86;182;194"  # Cyan
const NO_COLOR = CSI"0"

get_colorizers(io::IO) = get(io, :color, false) ? (TYPE_COLOR, NO_COLOR) : ("", "")

"""
    @def name definition
"""
macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

using Base: typename

Base.@pure __parameterless_type(T) = typename(T).wrapper
parameterless_type(x) = __parameterless_type(typeof(x))
parameterless_type(::Type{T}) where {T} = __parameterless_type(T)

# support functions
export check_keywords, warn_compat
function check_keywords(alg, kwargs, warnlist)
    flg = false
    for (kw, val) in kwargs
        if kw in warnlist
            if val !== nothing
                flg = true
                @warn(string("The ", kw, " argument is ignored by ", alg, "."))
            end
        end
    end
    flg
end

"""
$(SIGNATURES)

Emit a warning with a link to the solver compatibility chart in the documentation.
"""
warn_compat() = @warn("https://docs.sciml.ai/DiffEqDocs/stable/basics/compatibility_chart/")

"""
    @add_kwonly function_definition

Define keyword-only version of the `function_definition`.

    @add_kwonly function f(x; y=1)
        ...
    end

expands to:

    function f(x; y=1)
        ...
    end
    function f(; x = error("No argument x"), y=1)
        ...
    end
"""
macro add_kwonly(ex)
    esc(add_kwonly(ex))
end

add_kwonly(ex::Expr) = add_kwonly(Val{ex.head}, ex)

function add_kwonly(::Type{<:Val}, ex)
    error("add_only does not work with expression $(ex.head)")
end

function add_kwonly(::Union{Type{Val{:function}},
        Type{Val{:(=)}}}, ex::Expr)
    body = ex.args[2:end]  # function body
    default_call = ex.args[1]  # e.g., :(f(a, b=2; c=3))
    kwonly_call = add_kwonly(default_call)
    if kwonly_call === nothing
        return ex
    end

    return quote
        begin
            $ex
            $(Expr(ex.head, kwonly_call, body...))
        end
    end
end

function add_kwonly(::Type{Val{:where}}, ex::Expr)
    default_call = ex.args[1]
    rest = ex.args[2:end]
    kwonly_call = add_kwonly(default_call)
    if kwonly_call === nothing
        return nothing
    end
    return Expr(:where, kwonly_call, rest...)
end

function add_kwonly(::Type{Val{:call}}, default_call::Expr)
    # default_call is, e.g., :(f(a, b=2; c=3))
    funcname = default_call.args[1]  # e.g., :f
    required = []  # required positional arguments; e.g., [:a]
    optional = []  # optional positional arguments; e.g., [:(b=2)]
    default_kwargs = []
    for arg in default_call.args[2:end]
        if isa(arg, Symbol)
            push!(required, arg)
        elseif arg.head == :(::)
            push!(required, arg)
        elseif arg.head == :kw
            push!(optional, arg)
        elseif arg.head == :parameters
            @assert default_kwargs == []  # can I have :parameters twice?
            default_kwargs = arg.args
        else
            error("Not expecting to see: $arg")
        end
    end
    if isempty(required) && isempty(optional)
        # If the function is already keyword-only, do nothing:
        return nothing
    end
    if isempty(required)
        # It's not clear what should be done.  Let's not support it at
        # the moment:
        error("At least one positional mandatory argument is required.")
    end

    kwonly_kwargs = Expr(:parameters,
        [Expr(:kw, pa, :(error($("No argument $pa"))))
         for pa in required]..., optional..., default_kwargs...)
    kwonly_call = Expr(:call, funcname, kwonly_kwargs)
    # e.g., :(f(; a=error(...), b=error(...), c=1, d=2))

    return kwonly_call
end

"""
$(SIGNATURES)

List symbols `export`'ed but not actually defined.
"""
function undefined_exports(mod)
    undefined = []
    for name in names(mod)
        if !isdefined(mod, name)
            push!(undefined, name)
        end
    end
    return undefined
end

# Overloaded in other repositories
function unwrap_cache end

struct Void{F}
    f::F
end
function (f::Void)(args...)
    f.f(args...)
    nothing
end

"""
To be overloaded in ModelingToolkit
"""
function handle_varmap end

function mergedefaults(defaults, varmap, vars)
    defs = if varmap isa Dict
        merge(defaults, varmap)
    elseif eltype(varmap) <: Pair
        merge(defaults, Dict(varmap))
    elseif eltype(varmap) <: Number
        merge(defaults, Dict(zip(vars, varmap)))
    else
        defaults
    end
end

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

"""
    prepare_initial_state(u0) = u0

Whenever an initial state is passed to the SciML ecosystem, is passed to
`prepare_initial_state` and the result is used instead. If you define a
type which cannot be used as a state but can be converted to something that
can be, then you may define `prepare_initial_state(x::YourType) = ...`.

!!! warning
    This function is experimental and may be removed in the future.

See also: `prepare_function`.
"""
prepare_initial_state(u0) = u0

"""
    prepare_function(f) = f

Whenever a function is passed to the SciML ecosystem, is passed to
`prepare_function` and the result is used instead. If you define a type which
cannot be used as a function in the SciML ecosystem but can be converted to
something that can be, then you may define `prepare_function(x::YourType) = ...`.

`prepare_function` may be called before or after
the arity of a function is computed with `numargs`

!!! warning
    This function is experimental and may be removed in the future.

See also: `prepare_initial_state`.
"""
prepare_function(f) = f
