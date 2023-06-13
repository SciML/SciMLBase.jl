using SciMLBase, Test
using SciMLBase: should_warn_paramtype, warn_paramtype, WARN_PARAMTYPE_MESSAGE

@test should_warn_paramtype([]) == true
@test should_warn_paramtype(Float64[]) == false
@test should_warn_paramtype([1, "2"]) == true
@test should_warn_paramtype([1, 2.0]) == false
@test should_warn_paramtype(SciMLBase.NullParameters()) == false
@test should_warn_paramtype(nothing) == false
@test should_warn_paramtype(()) == false
@test should_warn_paramtype((1, "2")) == false
@test should_warn_paramtype(Dict(:a => 1, :b => "2")) == true
@test should_warn_paramtype(((1, 2.0), (3, "4"))) == false
@test should_warn_paramtype(([1, 2.0], [3, "4"])) == true
@test should_warn_paramtype([(1, 2.0), (3, "4")]) == true # uh oh
@test should_warn_paramtype([[1, 2.0], [3, "4"]]) == true

@test_logs (:info, WARN_PARAMTYPE_MESSAGE) warn_paramtype([1, "2"])
@test_logs warn_paramtype((1, "2"))
@test_logs warn_paramtype([1, 2])
@test_logs warn_paramtype([1, "2"], false)
@test_logs warn_paramtype((1, "2"), false)

# mock functions
f_2(a, b) = nothing
f_3(a, b, c) = nothing
f_4(a, b, c, d) = nothing

x = [0.0]
tspan = (0.0, 1.0)
p = [1, "2"]
# Test all Basic Problem types
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) AnalyticalProblem(f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) LinearProblem(f_4, x, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) IntervalNonlinearProblem(f_2, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) NonlinearProblem(f_3, x, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) IntegralProblem(f_3, x, x, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) OptimizationProblem(f_2, x, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) BVProblem(f_4, f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) DAEProblem(f_4, x, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) DDEProblem(f_4, f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) DiscreteProblem(f_4, x, tspan, p)
#@test_logs (:info, WARN_PARAMTYPE_MESSAGE) ImplicitDiscreteProblem(f_3, x, tspan, p) # Base constructor is apparently broken?
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) ODEProblem(f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) RODEProblem(f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) SDDEProblem(f_4, f_4, x, f_2, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) SDEProblem(f_4, f_4, x, tspan, p)
@test_logs (:info, WARN_PARAMTYPE_MESSAGE) SteadyStateProblem(f_4, x, p)
