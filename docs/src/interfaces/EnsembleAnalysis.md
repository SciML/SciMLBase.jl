# Ensemble Analysis

SciMLBase provides a comprehensive suite of statistical analysis functions for ensemble simulations. These functions enable detailed statistical analysis of Monte Carlo simulations, parameter sweeps, and uncertainty quantification studies.

## Overview

Ensemble analysis functions operate on `EnsembleSolution` objects and provide various statistical measures across time series data. The analysis can be performed:
- At specific time steps across all trajectories
- At specific time points through interpolation
- Component-wise across solution variables
- Using various statistical measures (mean, median, quantiles, variance, etc.)

## Data Extraction Functions

### Timestep-Based Extraction

```julia
get_timestep(ensemble_sol, i)
```

Extracts the solution values at the `i`-th timestep across all trajectories in the ensemble.

**Arguments:**
- `ensemble_sol`: An `EnsembleSolution` object
- `i`: Timestep index

**Returns:** Array containing solution values at timestep `i` for all trajectories

### Timepoint-Based Extraction

```julia
get_timepoint(ensemble_sol, t)
```

Extracts solution values at a specific time `t` by interpolating each trajectory in the ensemble.

**Arguments:**
- `ensemble_sol`: An `EnsembleSolution` object  
- `t`: Time value (can be any time within the solution interval)

**Returns:** Array containing interpolated solution values at time `t` for all trajectories

## Timestep Statistics

These functions compute statistics at specific timestep indices across all trajectories:

### Mean and Central Tendency

```julia
timestep_mean(ensemble_sol, i)
timestep_median(ensemble_sol, i)
```

- `timestep_mean`: Computes the mean across all trajectories at timestep `i`
- `timestep_median`: Computes the median across all trajectories at timestep `i`

### Quantiles

```julia
timestep_quantile(ensemble_sol, q, i)
```

Computes the `q`-th quantile across all trajectories at timestep `i`.

**Arguments:**
- `q`: Quantile value between 0 and 1 (e.g., 0.25 for first quartile)
- `i`: Timestep index

### Variance and Moments

```julia
timestep_meanvar(ensemble_sol, i)
```

Computes both mean and variance across all trajectories at timestep `i`.

**Returns:** Tuple `(mean, variance)`

## Timepoint Statistics

These functions compute statistics at specific time values using interpolation:

### Mean and Central Tendency

```julia
timepoint_mean(ensemble_sol, t)
timepoint_median(ensemble_sol, t)
```

- `timepoint_mean`: Computes the mean across all trajectories at time `t`
- `timepoint_median`: Computes the median across all trajectories at time `t`

### Quantiles

```julia
timepoint_quantile(ensemble_sol, q, t)
```

Computes the `q`-th quantile across all trajectories at time `t`.

### Variance and Moments

```julia
timepoint_meanvar(ensemble_sol, t)
```

Computes both mean and variance across all trajectories at time `t`.

## Time Series Analysis

### Complete Time Series Statistics

```julia
timeseries_steps_mean(ensemble_sol)
timeseries_point_mean(ensemble_sol, ts)
```

- `timeseries_steps_mean`: Computes mean at each timestep across all trajectories
- `timeseries_point_mean`: Computes mean at specified time points `ts`

**Returns:** Time series of mean values

## Component-wise Analysis

### Per-Variable Statistics

```julia
componentwise_mean(ensemble_sol)
componentwise_meanvar(ensemble_sol)
```

- `componentwise_mean`: Computes mean for each solution component across time and trajectories
- `componentwise_meanvar`: Computes mean and variance for each component

These functions are particularly useful for analyzing systems with multiple state variables independently.

## Covariance and Correlation Analysis

### Covariance Functions

```julia
timestep_cov(ensemble_sol, i)
timestep_cor(ensemble_sol, i)
timepoint_cov(ensemble_sol, t)
timepoint_cor(ensemble_sol, t)
```

- `timestep_cov`/`timepoint_cov`: Compute covariance matrices between solution components
- `timestep_cor`/`timepoint_cor`: Compute correlation matrices between solution components

These functions help analyze relationships between different state variables in the ensemble.

## Usage Examples

### Basic Statistical Analysis

```julia
using SciMLBase, OrdinaryDiffEq

# Define ensemble problem
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Create ensemble with parameter variations
prob = ODEProblem(lorenz!, [1.0; 0.0; 0.0], (0.0, 10.0), [10.0, 28.0, 8/3])
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p .+ 0.1*randn(3))
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories = 1000)

# Compute statistics
mean_at_t5 = timepoint_mean(ensemble_sol, 5.0)
median_at_t5 = timepoint_median(ensemble_sol, 5.0)
var_at_t5 = timepoint_meanvar(ensemble_sol, 5.0)[2]

# Quantile analysis
q25_at_t5 = timepoint_quantile(ensemble_sol, 0.25, 5.0)
q75_at_t5 = timepoint_quantile(ensemble_sol, 0.75, 5.0)

# Time series analysis
mean_trajectory = timeseries_steps_mean(ensemble_sol)
component_means = componentwise_mean(ensemble_sol)
```

### Correlation Analysis

```julia
# Analyze correlations between state variables
correlation_at_t5 = timepoint_cor(ensemble_sol, 5.0)
covariance_at_t5 = timepoint_cov(ensemble_sol, 5.0)

# Analyze how correlations change over time
times = 0.0:0.5:10.0
correlations_over_time = [timepoint_cor(ensemble_sol, t) for t in times]
```

### Uncertainty Quantification

```julia
# Compute confidence intervals
times = ensemble_sol[1].t
means = [timepoint_mean(ensemble_sol, t) for t in times]
q05 = [timepoint_quantile(ensemble_sol, 0.05, t) for t in times]
q95 = [timepoint_quantile(ensemble_sol, 0.95, t) for t in times]

# 90% confidence bands
lower_bound = q05
upper_bound = q95
```

## Performance Considerations

- **Memory Usage**: Large ensembles can consume significant memory. Consider using iterative analysis for very large ensembles
- **Interpolation Cost**: Timepoint-based functions require interpolation and are more expensive than timestep-based functions
- **Parallel Processing**: Analysis functions can benefit from parallel processing for large ensembles

The ensemble analysis functions provide a powerful toolkit for extracting meaningful statistical insights from Monte Carlo simulations and uncertainty quantification studies in the SciML ecosystem.