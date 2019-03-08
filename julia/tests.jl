include("WindyGridWorld.jl")
import Main.WindyGrid

using Test

@testset "validprobability" begin
    @test WindyGrid.validprobability(.5)
    @test WindyGrid.validprobability(1)
    @test WindyGrid.validprobability(0)
    @test !WindyGrid.validprobability(-1)
    @test !WindyGrid.validprobability(10)
end

@testset "validdiscount" begin
    @test WindyGrid.validdiscount(.5)
    @test WindyGrid.validdiscount(1)
    @test !WindyGrid.validdiscount(0)
    @test !WindyGrid.validdiscount(-1)
    @test !WindyGrid.validdiscount(10)
end

@testset "validlocation" begin
    gridshape = [4, 5]
    @test WindyGrid.validlocation(gridshape, [1, 1])
    @test !WindyGrid.validlocation(gridshape, [6, 1])
    @test !WindyGrid.validlocation(gridshape, [1, 6])
    @test !WindyGrid.validlocation(gridshape, [-6, 1])
    @test !WindyGrid.validlocation(gridshape, [1, -1])

    @test WindyGrid.validlocation(gridshape, [1 1])
    @test !WindyGrid.validlocation(gridshape, [6 1])
    @test !WindyGrid.validlocation(gridshape, [1 6])
    @test !WindyGrid.validlocation(gridshape, [-6 1])
    @test !WindyGrid.validlocation(gridshape, [1 -1])
end

@testset "validobstacles" begin
    gridshape = [4, 5]
    @test WindyGrid.validobstacles(gridshape, [1 1])
    @test WindyGrid.validobstacles(gridshape, [1 1; 2 2])
    @test !WindyGrid.validobstacles(gridshape, [1 1; 2 -2])
end
