module Main exposing (main)

import Browser
import Color exposing (Color)
import Dict exposing (Dict)
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import List.Extra exposing (groupsOf, maximumBy, transpose, zip)


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = \s -> Sub.none
        , view = view
        }


type alias Point =
    ( Int, Int )


type alias Points =
    List Point


type alias Target =
    Point


type alias Obstacle =
    Point


type alias Wind =
    Point


type alias Movement =
    Point


type alias Obstacles =
    List Obstacle


type alias Values =
    Dict Point Float


type alias Policies =
    Dict Point Action


type Action
    = Stay
    | Left
    | Right
    | Up
    | Down


type Msg
    = Iterate
    | Reset


type alias Model =
    { width : Int
    , height : Int
    , target : Target
    , obstacles : Obstacles
    , windUp : Float
    , windDown : Float
    , windStay : Float
    , discount : Float
    , values : Values
    , policies : Policies
    , targetReward : Float
    , obstacleReward : Float
    , defaultReward : Float
    , iterations : Int
    }


init : () -> ( Model, Cmd Msg )
init _ =
    let
        width =
            10

        height =
            12

        target =
            ( 6, 8 )

        obstacles =
            [ ( 5, 8 ), ( 6, 6 ) ]

        windUp =
            0.1

        windDown =
            0.2

        windStay =
            1.0 - windUp - windDown

        discount =
            0.9

        values =
            Dict.empty

        policies =
            Dict.empty

        targetReward =
            0.0

        obstacleReward =
            -10.0

        defaultReward =
            -1.0

        iterations =
            0

        model =
            Model
                width
                height
                target
                obstacles
                windUp
                windDown
                windStay
                discount
                values
                policies
                targetReward
                obstacleReward
                defaultReward
                iterations
    in
    ( model, Cmd.none )


isTargetLocation : Point -> Model -> Bool
isTargetLocation point model =
    point == model.target


isObstacleLocation : Point -> Model -> Bool
isObstacleLocation point model =
    List.any (\obstacle -> obstacle == point) model.obstacles


actionToMovement : Action -> Movement
actionToMovement action =
    case action of
        Stay ->
            ( 0, 0 )

        Left ->
            ( -1, 0 )

        Right ->
            ( 1, 0 )

        Up ->
            ( 0, 1 )

        Down ->
            ( 0, -1 )


getNextPoint : Point -> Action -> Wind -> Model -> Point
getNextPoint currentPoint action wind model =
    let
        ( dx, dy ) =
            actionToMovement action

        -- Keep next point within the grid
        nextX =
            clamp 0 (model.width - 1) (Tuple.first currentPoint + dx + Tuple.first wind)

        nextY =
            clamp 0 (model.height - 1) (Tuple.second currentPoint + dy + Tuple.second wind)
    in
    ( nextX, nextY )


getReward : Point -> Model -> Float
getReward point model =
    if isTargetLocation point model then
        model.targetReward

    else if isObstacleLocation point model then
        model.obstacleReward

    else
        model.defaultReward


getUpdatedValueAtLocation : Point -> Action -> Model -> Float
getUpdatedValueAtLocation point action model =
    if isTargetLocation point model then
        model.targetReward

    else
        let
            winds =
                [ ( model.windUp, 1 )
                , ( model.windStay, 0 )
                , ( model.windDown, -1 )
                ]

            continuationValue =
                List.foldl
                    (\( probability, dy ) accumulator ->
                        let
                            wind =
                                ( 0, dy )

                            nextPoint =
                                getNextPoint point action wind model

                            value =
                                Maybe.withDefault 0 (Dict.get nextPoint model.values)
                        in
                        accumulator + probability * value
                    )
                    0.0
                    winds

            reward =
                getReward point model
        in
        reward + model.discount * continuationValue


getUpdatedValueFunction : Model -> Values
getUpdatedValueFunction model =
    let
        allPoints =
            getAllPoints model

        values =
            List.map
                (\point ->
                    let
                        action =
                            Maybe.withDefault Stay (Dict.get point model.policies)
                    in
                    ( point, getUpdatedValueAtLocation point action model )
                )
                allPoints
    in
    values |> Dict.fromList


solveValueFunction : Model -> Model
solveValueFunction model =
    let
        iterations =
            List.range 0 50
    in
    List.foldl
        (\_ currentModel -> { currentModel | values = getUpdatedValueFunction currentModel })
        model
        iterations


getUpdatedPolicyFunction : Model -> Policies
getUpdatedPolicyFunction model =
    let
        allPoints =
            getAllPoints model

        allActions =
            getAllActions
    in
    List.map
        (\point ->
            let
                candidatePairs =
                    List.map (\action -> ( action, getUpdatedValueAtLocation point action model )) allActions

                ( optimalAction, _ ) =
                    Maybe.withDefault ( Stay, largeNegativeNumber ) (maximumBy Tuple.second candidatePairs)
            in
            ( point, optimalAction )
        )
        allPoints
        |> Dict.fromList


runPolicyIteration : Model -> Model
runPolicyIteration model =
    let
        modelWithUpdatedValues =
            solveValueFunction model

        policies =
            getUpdatedPolicyFunction modelWithUpdatedValues

        iterations =
            modelWithUpdatedValues.iterations + 1
    in
    { modelWithUpdatedValues | policies = policies, iterations = iterations }


getAllPoints : Model -> Points
getAllPoints model =
    let
        xs =
            List.range 0 (model.width - 1)

        ys =
            List.range 0 (model.height - 1)
    in
    cartesian xs ys


getAllActions : List Action
getAllActions =
    [ Stay, Left, Right, Up, Down ]


largeNegativeNumber : Float
largeNegativeNumber =
    -1000000000


cartesian : List a -> List b -> List ( a, b )
cartesian xs ys =
    List.concatMap (\x -> List.map (\y -> ( x, y )) ys) xs


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Iterate ->
            ( runPolicyIteration model, Cmd.none )

        Reset ->
            init ()


view : Model -> Html Msg
view model =
    let
        message =
            "Iterations so far: " ++ String.fromInt model.iterations

        buttonDivStyle =
            [ style "padding-top" "15px" ]

        buttonStyle =
            [ style "margin-right" "15px" ]
    in
    div []
        [ div [] [ text message ]
        , div buttonDivStyle
            [ button (buttonStyle ++ [ onClick Iterate ]) [ text "Iterate" ]
            , button (buttonStyle ++ [ onClick Reset ]) [ text "Reset" ]
            ]
        , viewGrid model
        ]


viewGrid : Model -> Html Msg
viewGrid model =
    let
        valuesAndPolicies =
            zip (Dict.values model.values) (Dict.values model.policies)

        divStyle =
            [ style "padding-top" "50px" ]

        tableStyle =
            [ style "table-layout" "fixed"
            , style "width" "50%"
            , style "border-collapse" "collapse"
            , style "border" "1px solid black"
            ]

        cells =
            List.map
                (\( value, action ) -> td (cellStyle value) [ text (actionToArrow action) ])
                valuesAndPolicies

        chunkedCells =
            groupsOf model.height cells |> transpose

        rows =
            List.map (\row -> tr [] row) chunkedCells |> List.reverse
    in
    div divStyle [ table tableStyle [ tbody [] rows ] ]


cellStyle : Float -> List (Attribute msg)
cellStyle value =
    let
        backgroundColor =
            value |> valueToColor |> Color.toCssString
    in
    [ style "border" "1px solid black"
    , style "text-align" "center"
    , style "background-color" backgroundColor
    ]


valueToColor : Float -> Color
valueToColor value =
    let
        scaledGreen =
            (10.0 + value) / 10

        clampedGreen =
            clamp 0.0 1.0 scaledGreen
    in
    Color.rgb 1 clampedGreen 1


actionToArrow : Action -> String
actionToArrow action =
    case action of
        Stay ->
            "✗"

        Left ->
            "←"

        Right ->
            "→"

        Up ->
            "↑"

        Down ->
            "↓"
