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
        Verbosity.None() => LinearErrorControlVerbosity(fill(Verbosity.None(), nfields(LinearErrorControlVerbosity))...)

        Verbosity.Warn() => LinearErrorControlVerbosity(fill(Verbosity.Warn(), nfields(LinearErrorControlVerbosity))...)

        Verbosity.Error() => LinearErrorControlVerbosity(fill(Verbosity.Error(), nfields(LinearErrorControlVerbosity))...)

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
            Verbosity.None(), nfields(LinearPerformanceVerbosity))...)

        Verbosity.Warn() => LinearPerformanceVerbosity(fill(
            Verbosity.Warn(), nfields(LinearPerformanceVerbosity))...)

        Verbosity.Error() => LinearPerformanceVerbosity(fill(
            Verbosity.Error(), nfields(LinearPerformanceVerbosity))...)

        Verbosity.Default() => LinearPerformanceVerbosity()

        Verbosity.Edge() => LinearPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct LinearNumericalVerbosity
    @add_kwonly function LinearNumericalVerbosity()
        LinearNumericalVerbosity()
    end

end


function LinearNumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => LinearNumericalVerbosity(fill(
            Verbosity.None(), nfields(LinearNumericalVerbosity))...)

        Verbosity.Warn() => LinearNumericalVerbosity(fill(
            Verbosity.Warn(), nfields(LinearNumericalVerbosity))...)

        Verbosity.Error() => LinearNumericalVerbosity(fill(
            Verbosity.Error(), nfields(LinearNumericalVerbosity))...)

        Verbosity.Default() => LinearNumericalVerbosity()

        Verbosity.Edge() => LinearNumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct LinearVerbosity{T} 
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

        Verbosity.All() => ODEVerbosity{true}(
            LinearErrorControlVerbosity(Verbosity.All()),
            LinearPerformanceVerbosity(Verbosity.All()),
            LinearNumericalVerbosity(Verbosity.All())
        )

        _ => @error "Not a valid choice for verbosity."
    end
end

# Nonlinear Verbosity

mutable struct NonlinearErrorControlVerbosity
    @add_kwonly function NonlinearErrorControlVerbosity()
        NonlinearErrorControlVerbosity()
    end
end



function NonlinearErrorControlVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearErrorControlVerbosity(fill(
            Verbosity.None(), nfields(NonlinearErrorControlVerbosity))...)

        Verbosity.Warn() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Warn(), nfields(NonlinearErrorControlVerbosity))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), nfields(NonlinearErrorControlVerbosity))...)

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
            Verbosity.None(), nfields(NonlinearPerformanceVerbosity))...)

        Verbosity.Warn() => NonlinPerformanceVerbosity(fill(
            Verbosity.Warn(), nfields(NonlinearPerformanceVerbosity))...)

        Verbosity.Error() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Error(), nfields(NonlinearPerformanceVerbosity))...)

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
            Verbosity.None(), nfields(NonlinearPerformanceVerbosity))...)

        Verbosity.Warn() => NonlinearNumericalVerbosity(fill(
            Verbosity.Warn(), nfields(NonlinearPerformanceVerbosity))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), nfields(NonlinearPerformanceVerbosity))...)

        Verbosity.Default() => NonlinearNumericalVerbosity()

        Verbosity.Edge() => NonlinearNumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct NonlinearVerbosity{T}
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

        Verbosity.All() => ODEVerbosity{true}(
            LinearVerbosity(Verbosity.All()),
            NonlinearErrorControlVerbosity(Verbosity.All()),
            NonlinearPerformanceVerbosity(Verbosity.All()),
            NonlinearNumericalVerbosity(Verbosity.All())
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
            Verbosity.None(), nfields(ODEErrorControlVerbosity))...)

        Verbosity.Warn() => ODEErrorControlVerbosity(fill(
            Verbosity.Warn(), nfields(ODEErrorControlVerbosity))...)

        Verbosity.Error() => ODEErrorControlVerbosity(fill(
            Verbosity.Error(), nfields(ODEErrorControlVerbosity))...)

        Verbosity.Default() => ODEErrorControlVerbosity(Verbosity.Info(), Verbosity.Error())

        Verbosity.Edge() => ODEErrorControlVerbosity(Verbosity.Info(), Verbosity.Warn())

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct ODEPerformanceVerbosity
    @add_kwonly function ODEPerformanceVerbosity(dt_NaN, init_NaN)
        new(dt_NaN, init_NaN)
    end
end



function ODEPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => ODEPerformanceVerbosity(fill(
            Verbosity.None(), nfields(ODEPerformanceVerbosity))...)

        Verbosity.Warn() => ODEPerformanceVerbosity(fill(
            Verbosity.Warn(), nfields(ODEPerformanceVerbosity))...)

        Verbosity.Error() => ODEPerformanceVerbosity(fill(
            Verbosity.Error(), nfields(ODEPerformanceVerbosity))...)

        Verbosity.Default() => ODEPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct ODENumericalVerbosity
    @add_kwonly function ODENumericalVerbosity(dt_NaN, init_NaN)
        new(dt_NaN, init_NaN)
    end
end



function ODENumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => ODENumericalVerbosity(fill(
            Verbosity.None(), nfields(ODENumericalVerbosity))...)

        Verbosity.Warn() => ODENumericalVerbosity(fill(
            Verbosity.Warn(), nfields(ODENumericalVerbosity))...)

        Verbosity.Error() => ODENumericalVerbosity(fill(
            Verbosity.Error(), nfields(ODENumericalVerbosity))...)

        Verbosity.Default() => ODENumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct ODEVerbosity{T}
    nonlinear_verbosity::NonlinearVerbosity
    linear_verbosity::LinearVerbosity

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
            ODEErrorControlVerbosity(Verbosity.All()),
            ODEPerformanceVerbosity(Verbosity.All()),
            ODENumericalVerbosity(Verbosity.All())
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

    ODEVerbosity(nonlinear, linear, error_control_verbosity, performance_verbosity, numerical_verbosity)
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
        f::Function, verbose::AbstractVerbositySpecifier{true}, option, group, file, line, _module)
    level = message_level(verbose, option, group)

    if !isnothing(level)
        message = f()
        Base.@logmsg level message _file=file _line=line _module=_module
    end
end

function emit_message(message::String, verbose::AbstractVerbositySpecifier{true},
        option, group, file, line, _module)
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
        $(esc(f_or_message)), $(esc(verb)), $toggle, $group, $file, $line, $_module))
end