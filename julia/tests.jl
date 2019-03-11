include("WindyGridWorld.jl")
import Main.WindyGrid

using Test

const target = [1, 1]
const obstacles = [1 2; 4 3]
const world = WindyGrid.WindyGridWorld([4, 4], target, obstacles, .9, .1, .2)

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

@testset "istarget" begin
    @test WindyGrid.istarget(world, [1, 1])
    @test !WindyGrid.istarget(world, [1, 4])
end

@testset "isobstacle" begin
    @test WindyGrid.isobstacle(world, [1, 2])
    @test !WindyGrid.isobstacle(world, [1, 1])
end

@testset "getreward" begin
    @test WindyGrid.getreward(world, target) == WindyGrid.targetreward
    @test WindyGrid.getreward(world, obstacles[1, :]) == WindyGrid.obstaclereward
    @test WindyGrid.getreward(world, [2, 2]) == WindyGrid.defaultreward
end

@testset "getnextlocation" begin
    @test WindyGrid.getnextlocation(world, [1, 1], [0, 1]) == [1, 2]
    @test WindyGrid.getnextlocation(world, [1, 1], [1, 0]) == [2, 1]
    @test WindyGrid.getnextlocation(world, [1, 1], [-1, 0]) == [1, 1]
    @test WindyGrid.getnextlocation(world, [4, 4], [-1, 0]) == [3, 4]
    @test WindyGrid.getnextlocation(world, [4, 4], [1, 0]) == [4, 4]
    @test WindyGrid.getnextlocation(world, [1, 1], [0, -1]) == [1, 1]
end

@testset "getcontinuationvalue" begin
    valuefn = zeros(Float64, world.shape...)
    valuefn[1, 1] = 1.
    location = [1, 2]
    action = [0, -1]
    p = .5
    wind_y = 0
    actual = WindyGrid.getcontinuationvalue(valuefn, world, location, action, wind_y, p)
    expected = .5
    @test actual == expected
    location = [1, 1]
    action = [0, -1]
    p = .5
    wind_y = 1
    actual = WindyGrid.getcontinuationvalue(valuefn, world, location, action, wind_y, p)
    expected = .5
    @test actual == expected
    location = [1, 1]
    action = [0, -1]
    p = .5
    wind_y = 1
    actual = WindyGrid.getcontinuationvalue(valuefn, world, location, action, wind_y, p)
    expected = .5
    @test actual == expected
end
