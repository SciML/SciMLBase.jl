using SciMLBase
using Serialization
using Test

for clock in [
    SciMLBase.Clock(0.5),
    SciMLBase.Clock(0.5; phase = 0.1),
    SciMLBase.SolverStepClock,
    SciMLBase.Continuous
]
    serialize("_tmp.jls", clock)
    newclock = deserialize("_tmp.jls")
    @test newclock == clock
end

if isfile("_tmp.jls")
    rm("_tmp.jls")
end
