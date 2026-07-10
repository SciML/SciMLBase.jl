# SciML Clock Interfaces

Clock objects describe time domains used by solution interpolation and symbolic
indexing metadata. A saved time-series solution can be queried at indexed clock
ticks, such as `sol(Clock(0.1)[2:4]; idxs = :x)`. The indexed clock is converted
to concrete independent-variable values for the solution before interpolation.

```@docs
SciMLBase.Clocks
SciMLBase.AbstractClock
SciMLBase.ContinuousClock
SciMLBase.PeriodicClock
SciMLBase.SolverStepClock
SciMLBase.EventClock
SciMLBase.TimeDomain
SciMLBase.Continuous
SciMLBase.Clock
SciMLBase.isclock
SciMLBase.issolverstepclock
SciMLBase.iscontinuous
SciMLBase.iseventclock
SciMLBase.is_discrete_time_domain
SciMLBase.first_clock_tick_time
SciMLBase.IndexedClock
SciMLBase.canonicalize_indexed_clock
```
