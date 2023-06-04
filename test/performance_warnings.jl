using SciMLBase, Test
using SciMLBase: should_warn_paramtype, warn_paramtype, WARN_PARAMTYPE_MESSAGE

@test should_warn_paramtype([]) == true
@test should_warn_paramtype(Float64[]) == false
@test should_warn_paramtype([1,"2"]) == true
@test should_warn_paramtype([1,2.0]) == false
@test should_warn_paramtype(SciMLBase.NullParameters()) == false
@test should_warn_paramtype(nothing) == false
@test should_warn_paramtype(()) == false
@test should_warn_paramtype((1,"2")) == false
@test should_warn_paramtype(Dict(:a => 1, :b => "2")) == true
@test should_warn_paramtype(((1,2.0),(3,"4"))) == false
@test should_warn_paramtype(([1,2.0],[3,"4"])) == true
@test should_warn_paramtype([(1,2.0),(3,"4")]) == true # uh oh
@test should_warn_paramtype([[1,2.0],[3,"4"]]) == true

@test_logs (:info, WARN_PARAMTYPE_MESSAGE) warn_paramtype([1,"2"])
@test_logs warn_paramtype((1,"2"))
@test_logs warn_paramtype([1,2])
@test_logs warn_paramtype([1,"2"], false)
@test_logs warn_paramtype((1,"2"), false)


f(x,p,t) = x
x0 = [0.0]
tspan = (0.0,1.0)

@test_logs (:info, WARN_PARAMTYPE_MESSAGE) ODEProblem(f, x0, tspan, [1,"2"])
