using QuantileEffects
using Test

@testset "foo_checks" begin
    @test foo(0)<1E-16
end
