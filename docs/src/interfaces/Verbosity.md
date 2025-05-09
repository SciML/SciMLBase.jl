# SciML Verbosity
The SciML verbosity system is designed to make it easy for users to specify what messages are logged and at what level they are logged at during the solution process. 

At the highest level are the `AbstractVerbositySpecifier` subtypes, e.g. `ODEVerbosity`, `LinearVerbosity`, and so on. These hold "group" objects that group the error messages into three broad categories. The first is error control, which contains options related to solver error control and adaptivity algorithms, such as convergence issues and correctness guarantees (e.g. `dt < dtmin`). The numerical group holds options pertaining to performance issues, such as large condition numbers and detection of potential floating point issues. Finally the performance group has options related to potential performance issues. For example, mismatched input/output types in out of place functions, and other known performance traps. An example of a group object would be `ODEPerformanceVerbosity`. 

At the lowest level are the `option` settings. These correspond to either individual messages or groups of messages relating to a specific issue or piece of information. For example, the `dt_NaN` option in the `ODEErrorControl` group sets the level of a log message that states that the adaptive timestepping algorithm set `dt` to `NaN`. 

The system is also hierarchical in the sense that for example `ODEVerbosity` also contains a `NonlinearVerbosity` and a `LinearVerbosity`, because an ODE solve might use NonlinearSolve and LinearSolve. Note that `NonlinearVerbosity` also has a `LinearVerbosity`, because NonlinearSolve also can use LinearSolve. The `LinearVerbosity` in the `ODEVerbosity` handles any calls to LinearSolve that are not inside of a call to NonlinearSolve. 

# Base Verbosity Types 
The base verbosity specifiers are [Moshi.jl](https://rogerluo.dev/Moshi.jl/) algebraic data types. These allow for pattern matching and gives a namespace `Verbosity` to access the types with. 
There are five of these types that are used for the lowest level option toggles:

- `None`: indicates that this message should not be logged.
- `Info`: indicates that this message should be logged with a log level of `Info`
- `Warn`: indicates that this message should be logged with a log level of `Warn`
- `Error`: indicates that this message should be logged with a log level of `Error`
- `Level`: indicates that this message should be logged with a custom log level held by the `Level` type, e.g. `Verbosity.Level(4)` corresponds to `Logging.LogLevel(4)`

Three of these types are only meant to be used for higher level constructors of the verbosity types:

- `Edge`: messages in this group or verbosity specifier are only logged when they relate to edge cases
- `All`: All messages in this group or verbosity specifier are logged
- `Default`: only the default messages of this group or verbosity specifier are logged.

# Constructors

The constructors for the verbosity specifiers are designed to be flexible. For example, they can takes some of the base verbosity types described above:

```julia
ODEVerbosity(Verbosity.None()) #logs nothing
ODEVerbosity(Verbosity.All()) #logs everything
ODEVerbosity(Verbosity.Default()) #logs somethings 
```

They also have keyword argument constructors that can take base verbosity types and automatically construct the correct group object, while also taking a group object for a different 
group:

```julia 
# Doesn't print out any error_control information, warns if there is an `init_dt` issue, errors if there is a `dt_NaN` issue, uses Verbosity.Default() for ODENumericalVerbosity
ODEVerbosity(error_control = Verbosity.None(), performance = ODEPerformanceVerbosity(init_dt = Verbosity.Warn(), dt_NaN = Verbosity.Error()))
```

Similarly, all of the group objects have equally flexible constructors, but take different base verbosity types. 

```julia
ODENumericalVerbosity(Verbosity.None()) #logs nothing
ODENumericalVerbosity(Verbosity.Warn()) #everything is logged as a warning
```

# SciMLMessage Macro
```@docs
SciMLBase.SciMLMessage
```

# Verbosity API

