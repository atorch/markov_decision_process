module WindyGridWorld
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


mutable struct WindyGridWorld
    shape::Array{Int, 1}
    target::Array{Int, 1}
    obstacles::Array{Int, 2}
    discount::Float64
    prpushup::Float64
    prpushdown::Float64
    prstay::Float64
    valuegrid::Array{Float64, 2}
    policygrid::Array{Int, 3}
    function WindyGridWorld(shape, target, obstacles, discount, prpushup, prpushdown)
        prstay = 1. - prpushup - prpushdown
        valuegrid = zeros(Int, shape...)
        policygrid = zeros(Int, shape..., 2)
        new(shape, target, obstacles, discount, prpushup, prpushdown, prstay, valuegrid, policygrid)
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

function getnextlocation(world::WindyGridWorld, location::Array{Int, 1}, action::Array{Int, 1})
    nextlocation = location + action
    min.(max.(nextlocation, ones(Int, size(location))), world.shape)
end

function getcontinuationvalue(world::WindyGridWorld, location::Array{Int, 1}, action::Array{Int, 1}, windy::Int, p::Float64)
    action = action + [0, windy]
    nextlocation = getnextlocation(world, location, action)
    p * world.valuegrid[nextlocation...]
end


function getvalue(world::WindyGridWorld, location::Array{Int, 1}, action::Array{Int, 1})
    if istarget(world, location)
        0.0
    end
    windprobs = [world.prpushup, world.prpushdown, world.prstay]
    continuationvalue = sum([ getcontinuationvalue(world, location, action, windaction, p) for (windaction, p) in zip(windactions, windprobs) ])
    reward = getreward(world, location)
    reward + world.discount * continuationvalue
end

function updatevaluegrid(world::WindyGridWorld)
    updatedgrid = copy(world.valuegrid)

    for j = 1:world.shape[2], i=world.shape[1]
        action = world.policygrid[i, j, :]
        updatedgrid[i, j] = getvalue(world, [i, j], action)
    end

    updatedgrid
end

function solvevaluefn!(world::WindyGridWorld, maxiterations::Int=50)
    for _ = 1:maxiterations
        world.valuegrid = updatevaluegrid(world)
    end
end


function updatepolicygrid(world::WindyGridWorld)
    updatedgrid = copy(world.policygrid)

    for j = 1:world.shape[2], i=world.shape[1]
        candidates = [ getvalue(world, [i, j], actions[i, :]) for i = 1:size(actions)[1] ]
        display(candidates)
        display(argmax(candidates))

        updatedgrid[i, j] = actions[argmax(candidates), :]
    end

    updatedgrid
end

function policyiteration(maxiterations::Int=50, verbose::Bool=true)
    world = WindyGridWorld(shape, target, obstacles, discount, prpushup, prpushdown)

    for i in 1:maxiterations
        solvevaluefn!(world)

        if verbose
            display(round.(world.valuegrid, digits=3))
        end

        updatedpolicy = updatepolicygrid(world)
        if all(updated_policy .= world.policygrid)
            println("Converged after $i iterations.")
            break
        end

        world.policygrid = updatedpolicy
    end
    world
end


function validprobability(p)
    0 <= p <= 1
end


function validdiscount(d)
    0 < d <= 1
end

end

policyiteration()
