include("WindyGridWorld.jl")
import Main.WindyGrid

using Test

@testset "validprobability" begin
    @test WindyGrid.validprobability(.5)
end
