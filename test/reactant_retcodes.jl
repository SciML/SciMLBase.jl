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
