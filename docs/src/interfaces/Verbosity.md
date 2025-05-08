# SciML Verbosity
The SciML verbosity system is designed to make it easy for users to specify what messages are logged, and at what level they are logged at, during the solution process. 

At the highest level are the `AbstractVerbositySpecifier` subtypes, e.g. `ODEVerbosity`, `LinearVerbosity`, and so on. These hold `group` objects that group the error messages into three broad categories. The first is error control, which contains options related to solver error control and adaptivity algorithms, such as convergence issues and correctness guarantees (e.g. `dt < dtmin) 

At the lowest level are the `option` settings. These correspond to either individual messages or groups of messages in

## Verbosity Types

