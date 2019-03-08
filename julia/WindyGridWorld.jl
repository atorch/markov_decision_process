module WindyGrid
const actions = [0 0; -1 0; 1 0; 0 -1; 0 1]
const windactions = [1, 0, -1]

const defaultreward = -1
const obstaclereward = -10
const targetreward = 0

const shape = [10, 12]
const target = [6, 8]
const obstacles = [5 8; 6 6]
const prpushup = .1
const prpushdown = .2
const discount = .9


struct WindyGridWorld
    shape::Array{Int, 1}
    target::Array{Int, 1}
    obstacles::Array{Int, 2}
    discount::Float64
    prpushup::Float64
    prpushdown::Float64
    prstay::Float64

    function WindyGridWorld(shape, target, obstacles, discount, prpushup, prpushdown)
        validdiscount(discount) || error("Invalid discount $discount")

        prstay = 1. - prpushup - prpushdown
        all(map(validprobability, [prpushdown, prpushup, prstay])) || error("Invalid wind probability.")

        validlocation(target) || error("Target $target out of grid.")
        validobstacles(obstacles) || error("Obstacles out of grid.")

        new(shape, target, obstacles, discount, prpushup, prpushdown, prstay)
    end
end


function istarget(world::WindyGridWorld, location::Array{Int, 1})
    world.target == location
end


function isobstacle(world::WindyGridWorld, location::Array{Int, 1})
    n_obstacles = size(world.obstacles)[2]
    any([ location == world.obstacles[:, i] for i=1:n_obstacles ])
end

function getreward(world::WindyGridWorld, location::Array{Int, 1})
    if istarget(world, location)
        targetreward

    elseif isobstacle(world, location)
        obstaclereward

    else
        defaultreward
    end
end

function getnextlocation(
        world::WindyGridWorld,
        location::Array{Int, 1},
        action::Array{Int, 1}
    )
    nextlocation = location + action
    min.(max.(nextlocation, ones(Int, size(location))), world.shape)
end

function getcontinuationvalue(
        valuefn,
        world::WindyGridWorld,
        location::Array{Int, 1},
        action::Array{Int, 1},
        windy::Int,
        p::Float64
    )
    action = action + [0, windy]
    nextlocation = getnextlocation(world, location, action)
    p * valuefn[nextlocation...]
end


function getvalue(
    valuefn,
    world::WindyGridWorld,
    location::Array{Int, 1},
    action::Array{Int, 1}
)
    if istarget(world, location)
        0.0
    end

    windprobs = [world.prpushup, world.prpushdown, world.prstay]

    continuationvalue = sum(
        [
            getcontinuationvalue(
                valuefn,
                world,
                location,
                action,
                windaction,
                p
            )
            for (windaction, p) in zip(windactions, windprobs)
        ]
    )

    reward = getreward(world, location)

    reward + world.discount * continuationvalue
end

function updatevaluefn(valuefn, policyfn, world::WindyGridWorld)
    updatedvaluefn = copy(valuefn)

    for i = 1:world.shape[1], j=1:world.shape[2]
        action = policyfn[i, j, :]
        updatedgrid[i, j] = getvalue(valuefn, world, [i, j], action)
    end
    updatedvaluefn
end

function solvevaluefn!(valuefn, policyfn, world::WindyGridWorld, maxiterations::Int=50)
    for _ = 1:maxiterations
        valuefn = updatevaluefn(world)
    end
    valuefn
end


function updatepolicyfn(valuefn, policyfn, world::WindyGridWorld)
    updatedgrid = copy(policyfn)

    for i = 1:world.shape[1], j=world.shape[2]
        candidates = [ getvalue(valuefn, world, [i, j], actions[i, :]) for i = 1:size(actions)[1] ]

        updatedgrid[i, j] = actions[argmax(candidates), :]
    end

    updatedgrid
end

function policyiteration(maxiterations::Int=50, verbose::Bool=true)
    world = WindyGridWorld(shape, target, obstacles, discount, prpushup, prpushdown)

    valuefn = zeros(Int, world.shape)
    policyfn = zeros(Int, world.shape..., 2)

    for i in 1:maxiterations
        valuefn = solvevaluefn!(valuefn, policyfn, world)

        if verbose
            display(round.(valuefn, digits=3))
        end

        updatedpolicy = updatepolicyfn(world)
        if all(updated_policy .= policyfn)
            println("Converged after $i iterations.")
            break
        end

        policyfn = updatedpolicy
    end
    world
end


function validprobability(p)
    0 <= p <= 1
end


function validdiscount(d)
    0 < d <= 1
end

function validlocation(gridshape::Array{Int, 1}, location::Array{Int, 1})
    all(location .>= 1) && all(location .<= gridshape)
end

function validlocation(gridshape::Array{Int, 1}, location::Array{Int, 2})
    all(location .>= 1) && all(location .<= gridshape)
end


function validobstacles(gridshape::Array{Int, 1}, obstacles::Array{Int, 2})
    all(mapslices(obstacle -> validlocation(gridshape, obstacle), obstacles, dims=[2]))
end

end
