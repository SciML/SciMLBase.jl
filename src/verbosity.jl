@data Verbosity begin
    None
    Info
    Warn
    Error
    Level(Int)
    Edge
    All
    Default
end

# Linear Verbosity

linear_defaults = Dict(
    :default_lu_fallback => Verbosity.Warn(),
    :no_right_preconditioning => Verbosity.Warn(),
    :using_iterative_solvers => Verbosity.Warn(),
    :using_IterativeSolvers => Verbosity.Warn(),
    :IterativeSolvers_iterations => Verbosity.Warn(),
    :KrylovKit_verbosity => Verbosity.Warn()
)
mutable struct LinearErrorControlVerbosity
    default_lu_fallback::Verbosity.Type

    function LinearErrorControlVerbosity(;
            default_lu_fallback = linear_defaults[:default_lu_fallback])
        new(default_lu_fallback)
    end

    function LinearErrorControlVerbosity(verbose::Verbosity.Type)
        @match verbose begin
            Verbosity.None() => new(fill(
                Verbosity.None(), length(fieldnames(LinearErrorControlVerbosity)))...)

            Verbosity.Info() => new(fill(
                Verbosity.Info(), length(fieldnames(LinearErrorControlVerbosity)))...)

            Verbosity.Warn() => new(fill(
                Verbosity.Warn(), length(fieldnames(LinearErrorControlVerbosity)))...)

            Verbosity.Error() => new(fill(
                Verbosity.Error(), length(fieldnames(LinearErrorControlVerbosity)))...)

            Verbosity.Default() => LinearErrorControlVerbosity()

            Verbosity.Edge() => LinearErrorControlVerbosity()

            _ => @error "Not a valid choice for verbosity."
        end
    end
end


mutable struct LinearPerformanceVerbosity
    no_right_preconditioning::Verbosity.Type

    function LinearPerformanceVerbosity(;
            no_right_preconditioning = linear_defaults[:no_right_preconditioning])
        new(no_right_preconditioning)
    end

    function LinearPerformanceVerbosity(verbose::Verbosity.Type)
        @match verbose begin
            Verbosity.None() => new(fill(
                Verbosity.None(), length(fieldnames(LinearPerformanceVerbosity)))...)

            Verbosity.Info() => new(fill(
                Verbosity.Info(), length(fieldnames(LinearPerformanceVerbosity)))...)

            Verbosity.Warn() => new(fill(
                Verbosity.Warn(), length(fieldnames(LinearPerformanceVerbosity)))...)

            Verbosity.Error() => new(fill(
                Verbosity.Error(), length(fieldnames(LinearPerformanceVerbosity)))...)

            Verbosity.Default() => LinearPerformanceVerbosity()

            Verbosity.Edge() => LinearPerformanceVerbosity()

            _ => @error "Not a valid choice for verbosity."
        end
    end

end

mutable struct LinearNumericalVerbosity
    using_IterativeSolvers::Verbosity.Type
    IterativeSolvers_iterations::Verbosity.Type
    KrylovKit_verbosity::Verbosity.Type

    function LinearNumericalVerbosity(;
            using_IterativeSolvers = linear_defaults[:using_IterativeSolvers],
            IterativeSolvers_iterations = linear_defaults[:IterativeSolvers_iterations],
            KrylovKit_verbosity = linear_defaults[:KrylovKit_verbosity])
        new(using_IterativeSolvers, IterativeSolvers_iterations, KrylovKit_verbosity)
    end

    function LinearNumericalVerbosity(verbose::Verbosity.Type)
        @match verbose begin
            Verbosity.None() => new(fill(
                Verbosity.None(), length(fieldnames(LinearNumericalVerbosity)))...)

            Verbosity.Info() => new(fill(
                Verbosity.Info(), length(fieldnames(LinearNumericalVerbosity)))...)

            Verbosity.Warn() => new(fill(
                Verbosity.Warn(), length(fieldnames(LinearNumericalVerbosity)))...)

            Verbosity.Error() => new(fill(
                Verbosity.Error(), length(fieldnames(LinearNumericalVerbosity)))...)

            Verbosity.Default() => LinearNumericalVerbosity()

            Verbosity.Edge() => LinearNumericalVerbosity()

            _ => @error "Not a valid choice for verbosity."
        end
    end
end



struct LinearVerbosity{T} <: AbstractVerbositySpecifier{T}
    error_control::LinearErrorControlVerbosity
    performance::LinearPerformanceVerbosity
    numerical::LinearNumericalVerbosity
end

function LinearVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.Default() => LinearVerbosity{true}(
            LinearErrorControlVerbosity(Verbosity.Default()),
            LinearPerformanceVerbosity(Verbosity.Default()),
            LinearNumericalVerbosity(Verbosity.Default())
        )

        Verbosity.None() => LinearVerbosity{false}(
            LinearErrorControlVerbosity(Verbosity.None()),
            LinearPerformanceVerbosity(Verbosity.None()),
            LinearNumericalVerbosity(Verbosity.None()))

        Verbosity.All() => LinearVerbosity{true}(
            LinearErrorControlVerbosity(Verbosity.Info()),
            LinearPerformanceVerbosity(Verbosity.Info()),
            LinearNumericalVerbosity(Verbosity.Info())
        )

        _ => @error "Not a valid choice for LinearVerbosity. Available choices are `Default`, `None`, and `All`."
    end
end

function LinearVerbosity(;
        error_control = Verbosity.Default(), performance = Verbosity.Default(),
        numerical = Verbosity.Default(), kwargs...)
    if error_control isa Verbosity.Type
        error_control_verbosity = LinearErrorControlVerbosity(error_control)
    else
        error_control_verbosity = error_control
    end

    if performance isa Verbosity.Type
        performance_verbosity = LinearPerformanceVerbosity(performance)
    else
        performance_verbosity = performance
    end

    if numerical isa Verbosity.Type
        numerical_verbosity = LinearNumericalVerbosity(numerical)
    else
        numerical_verbosity = numerical
    end

    if !isempty(kwargs)
        for (key, value) in pairs(kwargs)
            if hasfield(LinearErrorControlVerbosity, key)
                setproperty!(error_control_verbosity, key, value)
            elseif hasfield(LinearPerformanceVerbosity, key)
                setproperty!(performance_verbosity, key, value)
            elseif hasfield(LinearNumericalVerbosity, key)
                setproperty!(numerical_verbosity, key, value)
            else
                error("$key is not a recognized verbosity toggle.")
            end
        end
    end

    LinearVerbosity{true}(error_control_verbosity,
        performance_verbosity, numerical_verbosity)
end

# Utilities 

function message_level(verbose::AbstractVerbositySpecifier{true}, option, group)
    group = getproperty(verbose, group)
    opt_level = getproperty(group, option)

    @match opt_level begin
        Verbosity.None() => nothing
        Verbosity.Info() => Logging.Info
        Verbosity.Warn() => Logging.Warn
        Verbosity.Error() => Logging.Error
        Verbosity.Level(i) => Logging.LogLevel(i)
    end
end

function emit_message(
        f::Function, verbose::V, option, group, file, line,
        _module) where {V <: AbstractVerbositySpecifier{true}}
    level = message_level(
        verbose, option, group)
    if !isnothing(level)
        message = f()
        Base.@logmsg level message _file=file _line=line _module=_module
    end
end

function emit_message(message::String, verbose::V,
        option, group, file, line, _module) where {V <: AbstractVerbositySpecifier{true}}
    level = message_level(verbose, option, group)

    if !isnothing(level)
        Base.@logmsg level message _file=file _line=line _module=_module
    end
end

function emit_message(
        f, verbose::AbstractVerbositySpecifier{false}, option, group, file, line, _module)
end

@doc doc"""
A macro that emits a log message based on the log level specified in the `option` and `group` of the `AbstractVerbositySpecifier` supplied. 
    
`f_or_message` may be a message String, or a 0-argument function that returns a String. 

## Usage
To emit a simple string, `@SciMLMessage("message", verbosity, :option, :group)` will emit a log message with the LogLevel specified in `verbosity`, at the appropriate `option` and `group`. 

`@SciMLMessage` can also be used to emit a log message coming from the evaluation of a 0-argument function. This function is resolved in the environment of the macro call.
Therefore it can use variables from the surrounding environment. This may be useful if the log message writer wishes to carry out some calculations using existing variables
and use them in the log message.

```julia
x = 10
y = 20

@SciMLMessage(verbosity, :option, :group) do 
    z = x + y
    "Message is: x + y = \$z"
end
```
"""
macro SciMLMessage(f_or_message, verb, option, group)
    line = __source__.line
    file = string(__source__.file)
    _module = __module__
    return :(emit_message(
        $(esc(f_or_message)), $(esc(verb)), $option, $group, $file, $line, $_module))
end

