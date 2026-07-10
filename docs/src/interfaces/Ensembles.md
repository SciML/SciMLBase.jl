# [Parallel Ensemble Simulations Interface](@id ensemble)

Performing Monte Carlo simulations, solving with a predetermined set of initial conditions, and
GPU-parallelizing a parameter search all fall under the ensemble simulation interface. This
interface allows one to declare a template `AbstractSciMLProblem`, customize it for each
of `trajectories` runs, solve those trajectories in parallel batches, reduce the solutions
down to specific answers, and compute summary statistics on the results.

## Performing an Ensemble Simulation

### Building a Problem

```@docs
SciMLBase.AbstractEnsembleProblem
SciMLBase.EnsembleProblem
SciMLBase.WeightedEnsembleProblem
SciMLBase.DEFAULT_PROB_FUNC
SciMLBase.DEFAULT_OUTPUT_FUNC
SciMLBase.DEFAULT_REDUCTION
```

### Solving the Problem

```@docs
SciMLBase.__solve(prob::SciMLBase.AbstractEnsembleProblem, alg, ensemblealg::SciMLBase.BasicEnsembleAlgorithm)
SciMLBase.EnsembleContext
SciMLBase.generate_sim_seeds
SciMLBase.default_rng_func
```

### EnsembleAlgorithms

The choice of ensemble algorithm allows for control over how the multiple trajectories
are handled. Currently, the ensemble algorithm types are:

```@docs
SciMLBase.BasicEnsembleAlgorithm
SciMLBase.EnsembleSerial
SciMLBase.EnsembleThreads
SciMLBase.EnsembleDistributed
SciMLBase.EnsembleSplitThreads
SciMLBase.AbstractEnsembleEstimator
```

#### GPU Backends

GPU ensemble algorithms are extensions supplied by DiffEqGPU.jl rather than
SciMLBase. `EnsembleGPUArray()` adapts trajectories to GPU arrays, while
`EnsembleGPUKernel(backend)` compiles compatible solves into a device kernel.
Supported problem features and backend constructors depend on DiffEqGPU and the
selected GPU package; see the
[DiffEqGPU documentation](https://docs.sciml.ai/DiffEqGPU/stable/) for the current
compatibility rules.

### Choosing an Ensembler

For example, `EnsembleThreads()` is invoked by:

```julia
solve(ensembleprob, alg, EnsembleThreads(); trajectories = 1000)
```

### Solution Type

The resulting object is an `EnsembleSolution`, which includes the array of
trajectory outputs.

```@docs
SciMLBase.EnsembleSolution
SciMLBase.EnsembleTestSolution
SciMLBase.WeightedEnsembleSolution
```

### Plot Recipe

There is a plot recipe for an `AbstractEnsembleSolution` which composes all
of the plot recipes for the component solutions. The keyword arguments are passed
along. A useful argument to use is `linealpha` which will change the transparency
of the plots. An additional argument is `idxs` which allows you to choose which
components of the solution to plot. For example, if the differential equation
is a vector of 9 values, `idxs=1:2:9` will plot only the solutions
of the odd components. Another additional argument is `zcolors` (an alias of `marker_z`) which allows
you to pass a `zcolor` for each series. For details about `zcolor` see the
[Series documentation for Plots.jl](https://docs.juliaplots.org/stable/attributes/).

## Analyzing an Ensemble Experiment

Analysis tools are included for generating summary statistics and summary plots
for an `AbstractEnsembleSolution`.

To use this functionality, import the analysis module via:

```julia
using SciMLBase.EnsembleAnalysis
```

```@docs
SciMLBase.EnsembleAnalysis
```

### Time steps vs time points

For the summary statistics, there are two types. You can either summarize by
time steps or by time points. Summarizing by time steps assumes that the time steps
are all the same time point, i.e. the integrator used a fixed `dt` or the values were
saved using `saveat`. Summarizing by time points requires interpolating the solution.

```@docs
SciMLBase.EnsembleAnalysis.get_timestep
SciMLBase.EnsembleAnalysis.get_timepoint
SciMLBase.EnsembleAnalysis.componentwise_vectors_timestep
SciMLBase.EnsembleAnalysis.componentwise_vectors_timepoint
```

### Summary Statistics Functions

#### Single Time Statistics

The available functions for time steps are:

```@docs
SciMLBase.EnsembleAnalysis.timestep_mean
SciMLBase.EnsembleAnalysis.timestep_median
SciMLBase.EnsembleAnalysis.timestep_quantile
SciMLBase.EnsembleAnalysis.timestep_meanvar
SciMLBase.EnsembleAnalysis.timestep_meancov
SciMLBase.EnsembleAnalysis.timestep_meancor
SciMLBase.EnsembleAnalysis.timestep_weighted_meancov
```

The available functions for time points are:

```@docs
SciMLBase.EnsembleAnalysis.timepoint_mean
SciMLBase.EnsembleAnalysis.timepoint_median
SciMLBase.EnsembleAnalysis.timepoint_quantile
SciMLBase.EnsembleAnalysis.timepoint_meanvar
SciMLBase.EnsembleAnalysis.timepoint_meancov
SciMLBase.EnsembleAnalysis.timepoint_meancor
SciMLBase.EnsembleAnalysis.timepoint_weighted_meancov
```

#### Full Timeseries Statistics

Additionally, the following functions are provided for analyzing the full timeseries.
The `mean` and `meanvar` versions return a `DiffEqArray` which can be directly plotted.
The `meancov` and `meancor` return a matrix of tuples, where the tuples are the
`(mean_t1,mean_t2,cov or cor)`.

The available functions for the time steps are:

```@docs
SciMLBase.EnsembleAnalysis.timeseries_steps_mean
SciMLBase.EnsembleAnalysis.timeseries_steps_median
SciMLBase.EnsembleAnalysis.timeseries_steps_quantile
SciMLBase.EnsembleAnalysis.timeseries_steps_meanvar
SciMLBase.EnsembleAnalysis.timeseries_steps_meancov
SciMLBase.EnsembleAnalysis.timeseries_steps_meancor
SciMLBase.EnsembleAnalysis.timeseries_steps_weighted_meancov
```

The available functions for the time points are:

```@docs
SciMLBase.EnsembleAnalysis.timeseries_point_mean
SciMLBase.EnsembleAnalysis.timeseries_point_median
SciMLBase.EnsembleAnalysis.timeseries_point_quantile
SciMLBase.EnsembleAnalysis.timeseries_point_meanvar
SciMLBase.EnsembleAnalysis.timeseries_point_meancov
SciMLBase.EnsembleAnalysis.timeseries_point_meancor
SciMLBase.EnsembleAnalysis.timeseries_point_weighted_meancov
```

### EnsembleSummary

```@docs
SciMLBase.EnsembleAnalysis.EnsembleSummary
```

## Example 1: Solving an ODE With Different Initial Conditions

### Random Initial Conditions

Let's test the sensitivity of the linear ODE to its initial condition. To do this,
we would like to solve the linear ODE 100 times and plot what the trajectories
look like. Let's start by opening up some extra processes so that way the computation
will be parallelized. Here we will choose to use distributed parallelism, which means
that the required functions must be made available to all processes. This can be
achieved with
[`@everywhere` macro](https://docs.julialang.org/en/v1.2/stdlib/Distributed/#Distributed.@everywhere):

```julia
using Distributed
using OrdinaryDiffEq
using Plots

addprocs()
@everywhere using OrdinaryDiffEq
```

Now let's define the linear ODE, which is our base problem:

```julia
# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))
```

For our ensemble simulation, we would like to change the initial condition around.
This is done through the `prob_func`. This function takes in the base problem
and modifies it to create the new problem that the trajectory actually solves.
The `prob_func` has the signature `prob_func(prob, ctx)` where:

- `prob` is the base problem to be modified
- `ctx` is an `EnsembleContext` with the trajectory index, rerun count,
  trajectory-local RNG and seed, worker identifier, and optional master RNG.

Here, we will take the base problem, multiply the initial condition by a random draw
from the trajectory-local RNG, and use that for calculating the trajectory:

```julia
@everywhere function prob_func(prob, ctx)
    remake(prob, u0 = rand(ctx.rng) * prob.u0)
end
```

Now we build and solve the `EnsembleProblem` with this base problem and `prob_func`:

```julia
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = 10)
```

We can use the plot recipe to plot what the 10 ODEs look like:

```julia
plot(sim, linealpha = 0.4)
```

We note that if we wanted to find out what the initial condition was for a given
trajectory, we can retrieve it from the solution. `sim[i]` returns the `i`th
solution object. `sim[i].prob` is the problem that specific trajectory solved,
and `sim[i].prob.u0` would then be the initial condition used in the `i`th
trajectory.

With distributed execution, every function and captured value used by `prob_func`,
callbacks, and the model must be serializable and available on each worker. Top-level
named definitions loaded with `@everywhere` are the simplest way to satisfy this rule.

### Using multithreading

The previous ensemble simulation can also be parallelized using a multithreading
approach, which will make use of the different cores within a single computer.
Because the memory is shared across the different threads, it is not necessary to
use the `@everywhere` macro. Instead, the same problem can be implemented simply as:

```@example ensemble1_2
using OrdinaryDiffEq
using SciMLBase
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))
function prob_func(prob, ctx)
    remake(prob, u0 = rand(ctx.rng) * prob.u0)
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10)
using Plots;
plot(sim);
```

Set Julia's thread count when starting the process, for example with `julia --threads=auto`
or the `JULIA_NUM_THREADS` environment variable. See Julia's
[multithreading documentation](https://docs.julialang.org/en/v1/manual/multi-threading/).

### Pre-Determined Initial Conditions

Often, you may already know what initial conditions you want to use. This
can be specified by `ctx.sim_id` in the `prob_func`. The `sim_id` is the unique
index of each trajectory. So, if we have `trajectories=100`, then we have `ctx.sim_id` as
some index in `1:100`, and it's different for each trajectory.

So, if we wanted to use a grid of evenly spaced initial conditions from `0` to `1`,
we can index an evenly spaced `range`:

```@example ensemble1_3
initial_conditions = range(0, stop = 1, length = 100)
function prob_func(prob, ctx)
    remake(prob, u0 = initial_conditions[ctx.sim_id])
end
```

It's worth noting that if you run this code successfully, there will be no visible output.

## Example 2: Solving an SDE with Different Parameters

Let's solve the same SDE, but with varying parameters. Let's create a Lotka-Volterra
system with multiplicative noise. Our Lotka-Volterra system will have as its
drift component:

```@example ensemble2
function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end
```

For our noise function, we will use multiplicative noise:

```@example ensemble2
function g(du, u, p, t)
    du[1] = p[3] * u[1]
    du[2] = p[4] * u[2]
end
```

Now we build the SDE with these functions:

```@example ensemble2
using StochasticDiffEq
p = [1.5, 1.0, 0.1, 0.1]
prob = SDEProblem(f, g, [1.0, 1.0], (0.0, 10.0), p)
```

This is the base problem for our study. What would like to do with this experiment
is keep the same parameters in the deterministic component each time, but vary
the parameters for the amount of noise using `0.3rand(ctx.rng, 2)` as our parameters.
Once again, we do this with a `prob_func`, and here we modify the parameters in
`prob.p`:

```@example ensemble2
# `p` is a global variable, referencing it would be type unstable.
# Using a let block defines a small local scope in which we can
# capture that local `p` which isn't redefined anywhere in that local scope.
# This allows it to be type stable.
prob_func = let p = p
    (prob, ctx) -> begin
        x = 0.3rand(ctx.rng, 2)
        remake(prob, p = [p[1], p[2], x[1], x[2]])
    end
end
```

Now we solve the problem 10 times and plot all of the trajectories in phase space:

```@example ensemble2
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, SRIW1(), trajectories = 10)
using Plots;
plot(sim, linealpha = 0.6, color = :blue, idxs = (0, 1), title = "Phase Space Plot");
plot!(sim, linealpha = 0.6, color = :red, idxs = (0, 2), title = "Phase Space Plot")
```

We can then summarize this information with the mean/variance bounds using a
`EnsembleSummary` plot. We will take the mean/quantile at every `0.1` time
units and directly plot the summary:

```@example ensemble2
summ = EnsembleSummary(sim, 0:0.1:10)
plot(summ, fillalpha = 0.5)
```

Note that here we used the quantile bounds, which default to `[0.05,0.95]` in
the `EnsembleSummary` constructor. We can change to standard error of the mean
bounds using `ci_type=:SEM` in the plot recipe.

## Example 3: Using the Reduction to Halt When Estimator is Within Tolerance

In this problem, we will solve the equation just as many times as needed to get
the relative standard error of the mean for the final time point below our tolerance
`0.5`. Since we only care about the endpoint, we can tell the `output_func`
to discard the rest of the data.

```@example ensemble3
function output_func(sol, ctx)
    last(sol), false
end
```

Our `prob_func` will simply randomize the initial condition:

```@example ensemble3
using OrdinaryDiffEq
using SciMLBase
# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))

function prob_func(prob, ctx)
    remake(prob, u0 = rand(ctx.rng) * prob.u0)
end
```

Our reduction function will append the data from the current batch to the previous
batch, and declare convergence if the standard error of the mean is calculated
as sufficiently small:

```@example ensemble3
using Statistics
function reduction(u, batch, I)
    u = append!(u, batch)
    relative_standard_error = sqrt(var(u) / last(I)) / abs(mean(u))
    finished = relative_standard_error < 0.5
    u, finished
end
```

Then we can define and solve the problem:

```@example ensemble3
prob2 = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func,
    reduction = reduction, u_init = Vector{Float64}())
sim = solve(prob2, Tsit5(), trajectories = 10000, batch_size = 20)
```

Since `batch_size=20`, this means that every 20 simulations, it will take this batch,
append the results to the previous batch, calculate the relative standard error,
and exit the simulation once it is small enough. The criterion is first checked
after 20 simulations and then after every subsequent batch. This can save a lot
of time when convergence occurs before all requested trajectories are solved.

In addition to saving time by checking convergence, we can save memory by reducing
between batches. For example, say we only care about the mean at the end once
again. Instead of saving the solution at the end for each trajectory, we can instead
save the running summation of the endpoints:

```@example ensemble3
function reduction(u, batch, I)
    u + sum(batch), false
end
prob2 = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func,
    reduction = reduction, u_init = 0.0)
sim2 = solve(prob2, Tsit5(), trajectories = 100, batch_size = 20)
```

this will sum up the endpoints after every 20 solutions, and save the running sum.
The final result will have `sim2.u` as simply a number, and thus `sim2.u/100` would
be the mean.

## Example 4: Using the Analysis Tools

In this example, we will show how to analyze a `EnsembleSolution`. First, let's
generate a 10 solution Monte Carlo experiment. For our problem, we will use a `4x2`
system of linear stochastic differential equations:

```@example ensemble4
function f(du, u, p, t)
    for i in 1:length(u)
        du[i] = 1.01 * u[i]
    end
end
function σ(du, u, p, t)
    for i in 1:length(u)
        du[i] = 0.87 * u[i]
    end
end
using StochasticDiffEq
prob = SDEProblem(f, σ, ones(4, 2) / 2, (0.0, 1.0)) #prob_sde_2Dlinear
```

To solve this 10 times, we use the `EnsembleProblem` constructor and solve
with `trajectories=10`. Since we wish to compare values at the timesteps, we need
to make sure the steps all hit the same times. We thus set `adaptive=false` and
explicitly give a `dt`.

```@example ensemble4
prob2 = EnsembleProblem(prob)
sim = solve(prob2, SRIW1(), dt = 1 // 2^(3), trajectories = 10, adaptive = false)
```

**Note that if you don't do the `timeseries_steps` calculations, this code is
compatible with adaptive timestepping. Using adaptivity is usually more efficient!**

We can compute the mean and the variance at the 3rd timestep using:

```@example ensemble4
using SciMLBase.EnsembleAnalysis
m, v = timestep_meanvar(sim, 3)
```

or we can compute the mean and the variance at the `t=0.5` using:

```@example ensemble4
m, v = timepoint_meanvar(sim, 0.5)
```

We can get a series for the mean and the variance at each time step using:

```@example ensemble4
m_series, v_series = timeseries_steps_meanvar(sim)
```

or at chosen values of `t`:

```@example ensemble4
ts = 0:0.1:1
m_series = timeseries_point_mean(sim, ts)
```

Note that these mean and variance series can be directly plotted. We can
compute covariance matrices similarly:

```@example ensemble4
timeseries_steps_meancov(sim) # Use the time steps, assume fixed dt
timeseries_point_meancov(sim, 0:(1 // 2 ^ (3)):1, 0:(1 // 2 ^ (3)):1) # Use time points, interpolate
```

For general analysis, we can build an `EnsembleSummary`.

```@example ensemble4
summ = EnsembleSummary(sim)
```

will summarize at each time step, while

```@example ensemble4
summ = EnsembleSummary(sim, 0.0:0.1:1.0)
```

will summarize at the `0.1` time points using the interpolations. To
visualize the results, we can plot it. Since there are 8 components to
the differential equation, this can get messy, so let's only plot the
3rd component:

```@example ensemble4
using Plots;
plot(summ; idxs = 3);
```

We can change to errorbars instead of ribbons and plot two different
indices:

```@example ensemble4
plot(summ; idxs = (3, 5), error_style = :bars)
```

Or we can simply plot the mean of every component over time:

```@example ensemble4
plot(summ; error_style = :none)
```
