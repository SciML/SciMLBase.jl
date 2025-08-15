# Integrator Interface

The integrator interface provides fine-grained control over the solution process of differential equations. This interface allows users to step through solutions manually, modify solver state, manage caches, and control timestepping behavior.

## Overview

The integrator interface is accessed through `init(prob, alg; kwargs...)` which returns an integrator object that can be manipulated step-by-step. This is useful for:
- Interactive solution stepping
- Real-time control and monitoring
- Complex event handling
- Custom solution algorithms
- Debugging solver behavior

## Cache Management

### Basic Cache Access

```julia
get_tmp_cache(integrator)
full_cache(integrator) 
user_cache(integrator)
u_cache(integrator)
du_cache(integrator)
```

These functions provide access to different levels of the integrator's internal cache system:

- `get_tmp_cache`: Returns temporary arrays used by the solver
- `full_cache`: Returns the complete cache object
- `user_cache`: Returns user-accessible cache components
- `u_cache`: Returns the state variable cache
- `du_cache`: Returns the derivative cache

### Cache Resizing and Management

```julia
resize_non_user_cache!(integrator, i)
deleteat_non_user_cache!(integrator, idxs)
addat_non_user_cache!(integrator, i)
```

These functions manage the size and structure of internal caches:

- `resize_non_user_cache!`: Resizes non-user-facing caches to size `i`
- `deleteat_non_user_cache!`: Removes elements at indices `idxs` from caches
- `addat_non_user_cache!`: Adds elements at index `i` to caches

## Timestepping Control

### Time Stop Management

```julia
add_tstop!(integrator, t)
has_tstop(integrator)
first_tstop(integrator)
pop_tstop!(integrator)
```

Control when the integrator stops:

- `add_tstop!`: Adds a time point `t` where the integrator must stop
- `has_tstop`: Returns `true` if there are upcoming stop times
- `first_tstop`: Returns the next scheduled stop time
- `pop_tstop!`: Removes and returns the next stop time

### Save Time Management

```julia
add_saveat!(integrator, t)
```

Adds a time point where the solution should be saved without necessarily stopping the integrator.

### Time and State Modification

```julia
set_t!(integrator, t)
set_u!(integrator, u)
set_ut!(integrator, u, t)
get_dt(integrator)
get_proposed_dt(integrator)
set_proposed_dt!(integrator, dt)
```

Direct manipulation of integrator state:

- `set_t!`: Sets the current time to `t`
- `set_u!`: Sets the current state to `u`  
- `set_ut!`: Sets both state and time simultaneously
- `get_dt`: Returns the current timestep size
- `get_proposed_dt`: Returns the proposed next timestep
- `set_proposed_dt!`: Sets the proposed next timestep

## State Modification and Control

### Solution Modification

```julia
u_modified!(integrator, bool=true)
savevalues!(integrator)
```

- `u_modified!`: Notifies the integrator that the state has been externally modified
- `savevalues!`: Forces the integrator to save the current state

### Integrator Control

```julia
terminate!(integrator)
reinit!(integrator, u0; kwargs...)
auto_dt_reset!(integrator)
reeval_internals_due_to_modification!(integrator)
```

- `terminate!`: Stops the integration process
- `reinit!`: Reinitializes the integrator with new initial conditions
- `auto_dt_reset!`: Resets automatic timestep sizing
- `reeval_internals_due_to_modification!`: Recalculates internal solver state after modifications

## Error Handling and Validation

### Error Checking

```julia
check_error(integrator)
check_error!(integrator)
```

- `check_error`: Returns the current error estimate
- `check_error!`: Checks error and potentially modifies integrator state

### Tolerance Management

```julia
set_abstol!(integrator, abstol)
set_reltol!(integrator, reltol)
```

Dynamically adjust solver tolerances:

- `set_abstol!`: Sets absolute tolerance
- `set_reltol!`: Sets relative tolerance

## Advanced Features

### Derivative Access

```julia
get_du(integrator)
get_du!(out, integrator)
```

Access derivative information:

- `get_du`: Returns the current derivative
- `get_du!`: Writes current derivative into pre-allocated `out`

### Interpolation and Steps

```julia
change_t_via_interpolation!(integrator, t)
addsteps!(integrator)
```

- `change_t_via_interpolation!`: Changes time using interpolation
- `addsteps!`: Adds additional steps to the solution

### Solver Properties

```julia
isdiscrete(integrator)
```

- `isdiscrete`: Returns `true` if the integrator is for a discrete problem

## Usage Example

```julia
using SciMLBase, OrdinaryDiffEq

# Define a simple ODE
function lorenz!(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

prob = ODEProblem(lorenz!, [1.0; 0.0; 0.0], (0.0, 10.0))
integrator = init(prob, Tsit5())

# Manual stepping with control
while integrator.t < 5.0
    step!(integrator)
    
    # Modify state at specific times
    if integrator.t > 2.0 && integrator.t < 2.1
        integrator.u[1] += 0.1  # Add perturbation
        u_modified!(integrator)  # Notify integrator
    end
    
    # Add stop time dynamically
    if integrator.t > 3.0
        add_tstop!(integrator, 4.5)
    end
end
```

This interface provides maximum flexibility for advanced users who need fine control over the solution process while maintaining the performance and stability of the underlying numerical methods.