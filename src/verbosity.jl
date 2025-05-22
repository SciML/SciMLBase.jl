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

mutable struct LinearErrorControlVerbosity

end

function LinearErrorControlVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => LinearErrorControlVerbosity(fill(Verbosity.None(), length(fieldnames(LinearErrorControlVerbosity)))...)

        Verbosity.Info() => LinearErrorControlVerbosity(fill(Verbosity.Info(), length(fieldnames(LinearErrorControlVerbosity)))...)

        Verbosity.Warn() => LinearErrorControlVerbosity(fill(Verbosity.Warn(), length(fieldnames(LinearErrorControlVerbosity)))...)

        Verbosity.Error() => LinearErrorControlVerbosity(fill(Verbosity.Error(), length(fieldnames(LinearErrorControlVerbosity)))...)

        Verbosity.Default() => LinearErrorControlVerbosity()

        Verbosity.Edge() => LinearErrorControlVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct LinearPerformanceVerbosity

    @add_kwonly function LinearPerformanceVerbosity()
        new()
    end
end



function LinearPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => LinearPerformanceVerbosity(fill(
            Verbosity.None(), length(fieldnames(LinearPerformanceVerbosity)))...)

        Verbosity.Info() => LinearPerformanceVerbosity(fill(
            Verbosity.Info(), length(fieldnames(LinearPerformanceVerbosity)))...)

        Verbosity.Warn() => LinearPerformanceVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(LinearPerformanceVerbosity)))...)

        Verbosity.Error() => LinearPerformanceVerbosity(fill(
            Verbosity.Error(), length(fieldnames(LinearPerformanceVerbosity)))...)

        Verbosity.Default() => LinearPerformanceVerbosity()

        Verbosity.Edge() => LinearPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct LinearNumericalVerbosity
    @add_kwonly function LinearNumericalVerbosity()
        new()
    end

end


function LinearNumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => LinearNumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(LinearNumericalVerbosity)))...)

        Verbosity.Info() => LinearNumericalVerbosity(fill(
            Verbosity.Info(), length(fieldnames(LinearNumericalVerbosity)))...)

        Verbosity.Warn() => LinearNumericalVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(LinearNumericalVerbosity)))...)

        Verbosity.Error() => LinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(LinearNumericalVerbosity)))...)

        Verbosity.Default() => LinearNumericalVerbosity()

        Verbosity.Edge() => LinearNumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
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

        _ => @error "Not a valid choice for verbosity."
    end
end

# Nonlinear Verbosity

mutable struct NonlinearErrorControlVerbosity
    @add_kwonly function NonlinearErrorControlVerbosity()
        new()
    end
end



function NonlinearErrorControlVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearErrorControlVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Info() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Warn() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Default() => NonlinearErrorControlVerbosity()

        Verbosity.Edge() => NonlinearErrorControlVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct NonlinearPerformanceVerbosity
    @add_kwonly function NonlinearPerformanceVerbosity()
        new()
    end

end


function NonlinearPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearPerformanceVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Info() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Warn() => NonlinPerformanceVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Error() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Default() => NonlinearPerformanceVerbosity()

        Verbosity.Edge() => NonlinearPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end


mutable struct NonlinearNumericalVerbosity
    @add_kwonly function NonlinearNumericalVerbosity()
        new()
    end
end

function NonlinearNumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearNumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Info() => NonlinearNumericalVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Warn() => NonlinearNumericalVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Default() => NonlinearNumericalVerbosity()

        Verbosity.Edge() => NonlinearNumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct NonlinearVerbosity{T} <: AbstractVerbositySpecifier{T}
    linear_verbosity::LinearVerbosity 

    error_control::NonlinearErrorControlVerbosity
    performance::NonlinearPerformanceVerbosity
    numerical::NonlinearNumericalVerbosity
end

function NonlinearVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.Default() => NonlinearVerbosity{true}(
            LinearVerbosity(Verbosity.Default()),
            NonlinearErrorControlVerbosity(Verbosity.Default()),
            NonlinearPerformanceVerbosity(Verbosity.Default()),
            NonlinearNumericalVerbosity(Verbosity.Default())
        )

        Verbosity.None() => NonlinearVerbosity{false}(
            LinearVerbosity(Verbosity.None()),
            NonlinearErrorControlVerbosity(Verbosity.None()), 
            NonlinearPerformanceVerbosity(Verbosity.None()), 
            NonlinearNumericalVerbosity(Verbosity.None()))

        Verbosity.All() => NonlinearVerbosity{true}(
            LinearVerbosity(Verbosity.All()),
            NonlinearErrorControlVerbosity(Verbosity.Info()),
            NonlinearPerformanceVerbosity(Verbosity.Info()),
            NonlinearNumericalVerbosity(Verbosity.Info())
        )

        _ => @error "Not a valid choice for verbosity."
    end
end

# ODE Verbosity

mutable struct ODEErrorControlVerbosity
    dt_NaN::Verbosity.Type
    init_NaN::Verbosity.Type

    @add_kwonly function ODEErrorControlVerbosity(dt_NaN, init_NaN)
        new(dt_NaN, init_NaN)
    end
end



function ODEErrorControlVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => ODEErrorControlVerbosity(fill(
            Verbosity.None(), length(fieldnames(ODEErrorControlVerbosity)))...)

        Verbosity.Info() => ODEErrorControlVerbosity(fill(
            Verbosity.Info(), length(fieldnames(ODEErrorControlVerbosity)))...)

        Verbosity.Warn() => ODEErrorControlVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(ODEErrorControlVerbosity)))...)

        Verbosity.Error() => ODEErrorControlVerbosity(fill(
            Verbosity.Error(), length(fieldnames(ODEErrorControlVerbosity)))...)

        Verbosity.Default() => ODEErrorControlVerbosity(Verbosity.Info(), Verbosity.Error())

        Verbosity.Edge() => ODEErrorControlVerbosity(Verbosity.Info(), Verbosity.Warn())

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct ODEPerformanceVerbosity
    alg_switch

    @add_kwonly function ODEPerformanceVerbosity()
        new(alg_switch)
    end
end



function ODEPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => ODEPerformanceVerbosity(fill(
            Verbosity.None(), length(fieldnames(ODEPerformanceVerbosity)))...)

        Verbosity.Info() => ODEPerformanceVerbosity(fill(
            Verbosity.Info(), length(fieldnames(ODEPerformanceVerbosity)))...)

        Verbosity.Warn() => ODEPerformanceVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(ODEPerformanceVerbosity)))...)

        Verbosity.Error() => ODEPerformanceVerbosity(fill(
            Verbosity.Error(), length(fieldnames(ODEPerformanceVerbosity)))...)

        Verbosity.Default() => ODEPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct ODENumericalVerbosity
    @add_kwonly function ODENumericalVerbosity()
        new()
    end
end



function ODENumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => ODENumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(ODENumericalVerbosity)))...)

        Verbosity.Info() => ODENumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(ODENumericalVerbosity)))...)

        Verbosity.Warn() => ODENumericalVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(ODENumericalVerbosity)))...)

        Verbosity.Error() => ODENumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(ODENumericalVerbosity)))...)

        Verbosity.Default() => ODENumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct ODEVerbosity{T} <: AbstractVerbositySpecifier{T}
    linear_verbosity::LinearVerbosity
    nonlinear_verbosity::NonlinearVerbosity

    error_control::ODEErrorControlVerbosity
    performance::ODEPerformanceVerbosity
    numerical::ODENumericalVerbosity
end

function ODEVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.Default() => ODEVerbosity{true}(
            LinearVerbosity(Verbosity.Default()),
            NonlinearVerbosity(Verbosity.Default()),
            ODEErrorControlVerbosity(Verbosity.Default()),
            ODEPerformanceVerbosity(Verbosity.Default()),
            ODENumericalVerbosity(Verbosity.Default())
        )

        Verbosity.None() => ODEVerbosity{false}(
            LinearVerbosity(Verbosity.None()),
            NonlinearVerbosity(Verbosity.None()),
            ODEErrorControlVerbosity(Verbosity.None()),
            ODEPerformanceVerbosity(Verbosity.None()),
            ODENumericalVerbosity(Verbosity.None())
        )

        Verbosity.All() => ODEVerbosity{true}(
            LinearVerbosity(Verbosity.All()),
            NonlinearVerbosity(Verbosity.All()),
            ODEErrorControlVerbosity(Verbosity.Info()),
            ODEPerformanceVerbosity(Verbosity.Info()),
            ODENumericalVerbosity(Verbosity.Info())
        )

        _ => @error "Not a valid choice for verbosity."
    end
end

function ODEVerbosity(; error_control = Verbosity.Default(), performance = Verbosity.Default(), numerical = Verbosity.Default(), linear_verbosity = Verbosity.Default(), nonlinear_verbosity = Verbosity.Default())

    if error_control isa Verbosity.Type 
        error_control_verbosity = ODEErrorControlVerbosity(error_control)
    else 
        error_control_verbosity = error_control
    end

    if performance isa Verbosity.Type
        performance_verbosity = ODEPerformanceVerbosity(performance)
    else
        performance_verbosity = performance
    end

    if numerical isa Verbosity.Type
        numerical_verbosity = ODENumericalVerbosity(numerical)
    else
        numerical_verbosity = numerical
    end

    if linear_verbosity isa Verbosity.Type 
        linear = LinearVerbosity(linear_verbosity)
    else
        linear = linear_verbosity
    end

    if nonlinear_verbosity isa Verbosity.Type
        nonlinear = NonlinearVerbosity(nonlinear_verbosity)
    else
        nonlinear = nonlinear_verbosity
    end

    ODEVerbosity{true}(linear, nonlinear, error_control_verbosity, performance_verbosity, numerical_verbosity)
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
    f::Function, verbose::V, option, group, file, line, _module) where {V<:AbstractVerbositySpecifier{true}}

    level = message_level(
        verbose, option, group) 
    if !isnothing(level)
        message = f()
        Base.@logmsg level message _file=file _line=line _module=_module
    end
end

function emit_message(message::String, verbose::V,
        option, group, file, line, _module) where {V<:AbstractVerbositySpecifier{true}}
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