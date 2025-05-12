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
        I
}) where {
        T,
        V,
        W,
        I
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

# backward compat in case anyone is using these.
# TODO: remove at next major version
const TooManyArgumentsError = FunctionArgumentsError
const TooFewArgumentsError = FunctionArgumentsError
const NoMethodsError = FunctionArgumentsError

function Base.showerror(io::IO, e::FunctionArgumentsError)
    println(io, ARGUMENTS_ERROR_MESSAGE)
    print(io, "Offending function: ")
    printstyled(io, e.fname; bold = true, color = :red)
    println(io, "\nMethods:")
    println(io, methods(e.f))
end

"""
    isinplace(f, inplace_param_number, fname = "f", iip_preferred = true;
              has_two_dispatches = true,
              outofplace_param_number = inplace_param_number - 1)
    isinplace(f::AbstractSciMLFunction[, inplace_param_number])

Check whether a function operates in place by comparing its number of arguments
to the expected number. If `f` is an `AbstractSciMLFunction`, then the type
parameter is assumed to be correct and is used. Otherwise `inplace_param_number`
is checked against the methods table, where `inplace_param_number` is the number
of arguments for the in-place dispatch. The out-of-place dispatch is assumed
to have `outofplace_param_number` parameters (one less than the inplace version
by default). If neither of these dispatches exist, an error is thrown.
If the error is thrown, `fname` is used to tell the user which function has the
incorrect dispatches.

`iip_preferred` means that if `inplace_param_number=4` and methods of both 3 and
for 4 args exist, then it will be chosen as in-place. `iip_dispatch` flips this
decision.

If `has_two_dispatches = false`, then it is assumed that there is only one correct
dispatch, i.e. `f(u,p)` for OptimizationFunction, and thus the check for the oop
form is disabled and the 2-argument signature is ensured to be matched.

# See also

  - [`numargs`](@ref numargs)
"""
function isinplace(f, inplace_param_number, fname = "f", iip_preferred = true;
        has_two_dispatches = true, isoptimization = false,
        outofplace_param_number = inplace_param_number - 1)
    if iip_preferred
        hasmethod(f, ntuple(_->Any, inplace_param_number)) && return true
        hasmethod(f, ntuple(_->Any, outofplace_param_number)) && return false
    else
        hasmethod(f, ntuple(_->Any, outofplace_param_number)) && return false
        hasmethod(f, ntuple(_->Any, inplace_param_number)) && return true
    end
    throw(FunctionArgumentsError(fname, f))
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

"""
        strip_solution(sol)

Strips a SciMLSolution object and its interpolation of their functions to better accommodate serialization.
"""
function strip_solution(sol::AbstractSciMLSolution)
    sol
end
