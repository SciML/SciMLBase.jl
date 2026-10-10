using SciMLBase, Reactant, Test

function traced_successful(flag)
    retcode = ifelse(flag > 0, ReturnCode.Success, ReturnCode.Failure)
    stalled = ifelse(flag > 0, ReturnCode.StalledSuccess, ReturnCode.Default)
    return SciMLBase.successful_retcode(retcode), SciMLBase.successful_retcode(stalled),
        retcode
end

@testset "successful_retcode of traced return codes: $expected" for (flag, expected) in
    ((1.0, true), (-1.0, false))
    ok, stalled_ok, retcode = Reactant.@jit traced_successful(Reactant.ConcreteRNumber(flag))
    @test Bool(ok) == expected
    @test Bool(stalled_ok) == expected
    @test SciMLBase.successful_retcode(retcode) == expected
end

# `ifelse` on a traced condition turns the code into a `Reactant.TracedEnum`.
traced_successful_code(flag, code) = SciMLBase.successful_retcode(ifelse(flag > 0, code, code))

@testset "traced and host successful_retcode agree on $code" for code in
    instances(ReturnCode.T)
    traced = Reactant.@jit traced_successful_code(Reactant.ConcreteRNumber(1.0), code)
    @test Bool(traced) == SciMLBase.successful_retcode(code)
end
