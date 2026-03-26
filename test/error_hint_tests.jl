using SciMLBase, Test

@testset "DomainError hint only in SciML context" begin
    # DomainError outside any SciML context should NOT show the SciML hint
    @testset "standalone log(-1) does not trigger hint" begin
        output = try
            log(-1.0)
        catch e
            sprint(showerror, e)
        end
        @test occursin("DomainError", output)
        @test !occursin("DomainError detected in the user", output)
    end

    @testset "standalone sqrt(-1) does not trigger hint" begin
        output = try
            sqrt(-1.0)
        catch e
            sprint(showerror, e)
        end
        @test occursin("DomainError", output)
        @test !occursin("DomainError detected in the user", output)
    end

    # DomainError through an ODEFunction (SciMLBase wrapper) SHOULD show the hint
    @testset "DomainError through ODEFunction triggers hint" begin
        f_oop(u, p, t) = log(u)
        odefun = ODEFunction(f_oop)
        output = try
            odefun(-1.0, nothing, 0.0)
        catch e
            sprint(showerror, e)
        end
        @test occursin("DomainError", output)
        @test occursin("DomainError detected in the user", output)
    end

    @testset "DomainError through in-place ODEFunction triggers hint" begin
        f_iip(du, u, p, t) = (du .= log.(u))
        odefun = ODEFunction(f_iip)
        du = [0.0]
        output = try
            odefun(du, [-1.0], nothing, 0.0)
        catch e
            sprint(showerror, e)
        end
        @test occursin("DomainError", output)
        @test occursin("DomainError detected in the user", output)
    end

    # Non-complex DomainError should never show the hint regardless of context
    @testset "unrelated DomainError does not trigger hint" begin
        output = try
            throw(DomainError(-1, "some other message"))
        catch e
            sprint(showerror, e)
        end
        @test occursin("DomainError", output)
        @test !occursin("DomainError detected in the user", output)
    end
end

@testset "NullParameters MethodError hint" begin
    @testset "arithmetic on NullParameters triggers hint" begin
        output = try
            SciMLBase.NullParameters() + 1
        catch e
            sprint(showerror, e)
        end
        @test occursin("NullParameters", output)
    end
end
