module WindyGrid

using Plots

const actions = Dict(
    [
        "stay" => [0, 0],
        "left" => [-1, 0],
        "right" => [1, 0],
        "down" => [0, 1],
        "up" => [0, -1]
    ]
)

const windactions = [1, -1, 0]

const defaultreward = -1
const obstaclereward = -10
const targetreward = 0

const shape = [10, 12]
const target = [6, 8]
const obstacles = [5 8; 6 6]
const prpushup = .1
const prpushdown = .2
const discount = .9


Obstacles = Union{Matrix, Nothing}


struct WindyGridWorld
    shape::Vector{Int}
    target::Vector{Int}
    obstacles::Obstacles
    discount::Float64
    prpushup::Float64
    prpushdown::Float64
    prstay::Float64


    function WindyGridWorld(shape, target, obstacles, discount, prpushup, prpushdown)
        validdiscount(discount) || error("Invalid discount $discount")

        prstay = 1. - prpushup - prpushdown
        all(map(validprobability, [prpushdown, prpushup, prstay])) || error("Invalid wind probability.")

        validlocation(shape, target) || error("Target $target out of grid.")
        validobstacles(shape, obstacles) || error("Obstacles out of grid.")

        new(shape, target, obstacles, discount, prpushup, prpushdown, prstay)
    end
end


function validprobability(p)
    0 <= p <= 1
end


function validdiscount(d)
    0 < d <= 1
end


function validlocation(gridshape::Vector{Int}, location::Vector{Int})
    all(location .>= 1) && all(location .<= gridshape)
end


function validlocation(gridshape::Vector{Int}, location::Matrix{Int})
    all(location .>= 1) && all(location .<= gridshape)
end


function validobstacles(gridshape::Vector{Int}, obstacles::Obstacles)
    obstacles == nothing || all(mapslices(obs -> validlocation(gridshape, obs), obstacles, dims=[2]))
end


function istarget(world::WindyGridWorld, location::Vector{Int})
    world.target == location
end


function isobstacle(world::WindyGridWorld, location::Vector{Int})
    if world.obstacles == nothing
        false
    else
        any(mapslices(obs -> location == obs, world.obstacles, dims = [2]))
    end
end


function getreward(world::WindyGridWorld, location::Vector{Int})
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
        location::Vector{Int},
        action::Vector{Int}
)
    nextlocation = location + action
    min.(max.(nextlocation, ones(Int, size(location))), world.shape)
end


function getcontinuationvalue(
        valuefn,
        world::WindyGridWorld,
        location::Vector{Int},
        action::Vector{Int},
        wind_y::Int,
        p::Float64
)
    action = action .+ [0, wind_y]

    nextlocation = getnextlocation(world, location, action)
    p * valuefn[nextlocation...]
end


function getvalue(
    valuefn,
    world::WindyGridWorld,
    location::Vector{Int},
    action::String
)
    if istarget(world, location)
        targetreward
    end

    windprobs = [world.prpushup, world.prpushdown, world.prstay]

    continuationvalue = sum(
        [
            getcontinuationvalue(
                valuefn,
                world,
                location,
                actions[action],
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
        action = policyfn[i, j]
        updatedvaluefn[i, j] = getvalue(valuefn, world, [i, j], action)
    end

    updatedvaluefn
end


function solvevaluefn!(
    valuefn::Matrix,
    policyfn::Matrix,
    world::WindyGridWorld,
    maxiterations::Int=50
)
    for _ = 1:maxiterations
        valuefn = updatevaluefn(valuefn, policyfn, world)
    end

    valuefn
end


function updatepolicyfn(valuefn, policyfn, world::WindyGridWorld)
    updatedpolicyfn = copy(policyfn)

    for i = 1:world.shape[1], j=1:world.shape[2]
        candidates = Dict(
            [
                (k, getvalue(valuefn, world, [i, j], k))
                for k in keys(actions)
            ]
        )

        updatedpolicyfn[i, j] = argmax(candidates)
    end

    updatedpolicyfn
end


function policyiteration(maxiterations::Int=50, verbose::Bool=true)
    world = WindyGridWorld(
        shape,
        target,
        obstacles,
        discount,
        prpushup,
        prpushdown
    )

    valuefn = zeros(Float64, world.shape...)
    policyfn = fill("stay", world.shape...)

    for i in 1:maxiterations
        valuefn = solvevaluefn!(valuefn, policyfn, world)

        if verbose
            display(round.(valuefn, digits=3))
        end

        updatedpolicy = updatepolicyfn(valuefn, policyfn, world)
        if all(updatedpolicy .== policyfn)
            println("Converged after $i iterations.")
            break
        end

        policyfn = updatedpolicy
    end

    valuefn, policyfn
end


function plotfns(valuefn, policyfn)
    scale = .5
    heatmap(transpose(valuefn))
    xmax, ymax = size(valuefn)
    xs = repeat(1:xmax, inner=ymax)
    ys = repeat(1:ymax, xmax)
    actionvec = [policyfn[xs[i], ys[i]] for i in 1:length(xs)]
    us = [scale * actions[a][1] for a in actionvec]
    vs = [scale * actions[a][2] for a in actionvec]
    quiver!(xs, ys, quiver=(us, vs), color=:blue)
    savefig("policyiteration.png")
end

end


function main()
    valuefn, policyfn = WindyGrid.policyiteration()
    WindyGrid.plotfns(valuefn, policyfn)
end

main()
