# for Debug previous line (%%writefile submission.py) should be commented out, uncomment to write submission.py

#FUNCTIONS###################################################
def get_map_and_average_halite(obs):
    """
        get average amount of halite per halite source
        and map as two dimensional array of objects and set amounts of halite in each cell
    """
    game_map = []
    halite_sources_amount = 0
    halite_total_amount = 0
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]
            })
            if game_map[x][y]["halite"] > 0:
                halite_total_amount += game_map[x][y]["halite"]
                halite_sources_amount += 1
    average_halite = halite_total_amount / halite_sources_amount
    return game_map, average_halite

def get_swarm_units_coords_and_update_map(s_env):
    """ get lists of coords of Swarm's units and update locations of ships and shipyards on the map """
    # arrays of (x, y) coords
    swarm_shipyards_coords = []
    swarm_ships_coords = []
    # place on the map locations of units of every player
    for player in range(len(s_env["obs"].players)):
        # place on the map locations of every shipyard of the player
        shipyards = list(s_env["obs"].players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map
            s_env["map"][x][y]["shipyard"] = player
            if player == s_env["obs"].player:
                swarm_shipyards_coords.append((x, y))
        # place on the map locations of every ship of the player
        ships = list(s_env["obs"].players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            s_env["map"][x][y]["ship"] = player
            s_env["map"][x][y]["ship_cargo"] = ship[1]
            if player == s_env["obs"].player:
                swarm_ships_coords.append((x, y))
    return swarm_shipyards_coords, swarm_ships_coords

def get_c(c):
    """ get coordinate, considering donut type of the map """
    return c % conf.size

def clear(x, y, player, game_map):
    """ check if cell is safe to move in """
    # if there is no shipyard, or there is player's shipyard
    # and there is no ship
    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and
            game_map[x][y]["ship"] == None):
        return True
    return False

def move_ship(x_initial, y_initial, actions, s_env, ship_index):
    """ move the ship according to first acceptable tactic """
    ok, actions = go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    return standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)

def go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ ship will go to safe cell with enough halite, if it is found """
    # if current cell has enough halite
    if (s_env["map"][x_initial][y_initial]["halite"] > s_env["low_amount_of_halite"] and
            not hostile_ship_near(x_initial, y_initial, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
        most_halite = s_env["map"][x_initial][y_initial]["halite"]
    else:
        # biggest amount of halite among scanned cells
        most_halite = s_env["low_amount_of_halite"]
    direction = None
    for d in range(len(directions_list)):
        x = directions_list[d]["x"](x_initial)
        y = directions_list[d]["y"](y_initial)
        # if cell is safe to move in
        if (clear(x, y, s_env["obs"].player, s_env["map"]) and
                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
            # if current cell has more than biggest amount of halite
            if s_env["map"][x][y]["halite"] > most_halite:
                most_halite = s_env["map"][x][y]["halite"]
                direction = directions_list[d]["direction"]
                direction_x = x
                direction_y = y
    # if cell is safe to move in and has substantial amount of halite
    if most_halite > s_env["low_amount_of_halite"] and direction != None:
        actions[ship_id] = direction
        s_env["map"][x_initial][y_initial]["ship"] = None
        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player
        return True, actions
    # if current cell has biggest amount of halite
    elif most_halite == s_env["map"][x_initial][y_initial]["halite"]:
        return True, actions
    return False, actions

def standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ 
        ship will move in expanding circles clockwise or counterclockwise
        until reaching maximum radius, then radius will be minimal again
    """
    directions = ships_data[ship_id]["directions"]
    # set index of direction
    i = ships_data[ship_id]["directions_index"]
    direction_found = False
    for j in range(len(directions)):
        x = directions[i]["x"](x_initial)
        y = directions[i]["y"](y_initial)
        # if cell is ok to move in
        if (clear(x, y, s_env["obs"].player, s_env["map"]) and
                (s_env["map"][x][y]["shipyard"] == s_env["obs"].player or
                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1]))):
            ships_data[ship_id]["moves_done"] += 1
            # apply changes to game_map, to avoid collisions of player's ships next turn
            s_env["map"][x_initial][y_initial]["ship"] = None
            s_env["map"][x][y]["ship"] = s_env["obs"].player
            # if it was last move in this direction
            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:
                ships_data[ship_id]["moves_done"] = 0
                ships_data[ship_id]["directions_index"] += 1
                # if it is last direction in a list
                if ships_data[ship_id]["directions_index"] >= len(directions):
                    ships_data[ship_id]["directions_index"] = 0
                    ships_data[ship_id]["ship_max_moves"] += 1
                    # if ship_max_moves reached maximum radius expansion
                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:
                        ships_data[ship_id]["ship_max_moves"] = 1
            actions[ship_id] = directions[i]["direction"]
            direction_found = True
            break
        else:
            # loop through directions
            i += 1
            if i >= len(directions):
                i = 0
    # if ship is not on shipyard and hostile ship is near
    if (not direction_found and s_env["map"][x_initial][y_initial]["shipyard"] == None and
            hostile_ship_near(x_initial, y_initial, s_env["obs"].player, s_env["map"],
                              s_env["ships_values"][ship_index][1])):
        # if there is enough halite to convert
        if s_env["ships_values"][ship_index][1] >= conf.convertCost:
            actions[ship_id] = "CONVERT"
            s_env["map"][x_initial][y_initial]["ship"] = None
        else:
            for i in range(len(directions)):
                x = directions[i]["x"](x_initial)
                y = directions[i]["y"](y_initial)
                # if it is opponent's shipyard
                if s_env["map"][x][y]["shipyard"] != None:
                    # apply changes to game_map, to avoid collisions of player's ships next turn
                    s_env["map"][x_initial][y_initial]["ship"] = None
                    s_env["map"][x][y]["ship"] = s_env["obs"].player
                    actions[ship_id] = directions[i]["direction"]
                    return actions
            for i in range(len(directions)):
                x = directions[i]["x"](x_initial)
                y = directions[i]["y"](y_initial)
                # if it is empty place
                if s_env["map"][x][y]["ship"] == None:
                    # apply changes to game_map, to avoid collisions of player's ships next turn
                    s_env["map"][x_initial][y_initial]["ship"] = None
                    s_env["map"][x][y]["ship"] = s_env["obs"].player
                    actions[ship_id] = directions[i]["direction"]
                    return actions
    return actions

def get_directions(i0, i1, i2, i3):
    """ get list of directions in a certain sequence """
    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]

def hostile_ship_near(x, y, player, m, cargo):
    """ check if hostile ship is in one move away from game_map[x][y] and has less or equal halite """
    # m = game map
    n = get_c(y - 1)
    e = get_c(x + 1)
    s = get_c(y + 1)
    w = get_c(x - 1)
    if (
            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] <= cargo) or
            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] <= cargo) or
            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] <= cargo) or
            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] <= cargo)
        ):
        return True
    return False

def spawn_ship(actions, s_env, ships_amount, i):
    """ spawn ship, if possible """
    if s_env["swarm_halite"] >= conf.spawnCost and ships_amount < s_env["ships_max_amount"]:
        x = s_env["swarm_shipyards_coords"][i][0]
        y = s_env["swarm_shipyards_coords"][i][1]
        # if there is currently no ship at shipyard
        if clear(x, y, s_env["obs"].player, s_env["map"]):
            s_env["swarm_halite"] -= conf.spawnCost
            actions[s_env["shipyards_keys"][i]] = "SPAWN"
            s_env["map"][x][y]["ship"] = s_env["obs"].player
            ships_amount += 1
        return True, actions, s_env, ships_amount
    return False, actions, s_env, ships_amount

def this_is_new_ship(s_env, i):
    """ add this ship to ships_data """
    global movement_tactics_index
    ships_data[s_env["ships_keys"][i]] = {
        "moves_done": 0,
        "ship_max_moves": 1,
        "directions": movement_tactics[movement_tactics_index]["directions"],
        "directions_index": 0
    }
    movement_tactics_index += 1
    if movement_tactics_index >= movement_tactics_amount:
        movement_tactics_index = 0
    return s_env

def this_is_last_step(x, y, actions, s_env, i):
    """ actions of ship, if it is last step """
    if s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost:
        actions[s_env["ships_keys"][i]] = "CONVERT"
        s_env["map"][x][y]["ship"] = None
        return True, actions, s_env
    return False, actions, s_env

def no_shipyards(x, y, actions, s_env, shipyards_amount, i):
    """ actions of ship, if there is no shipyard and conversion of ship is possible """
    if (shipyards_amount == 0 and
            not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1]) and
            (s_env["swarm_halite"] + s_env["ships_values"][i][1]) >= convert_threshold):
        s_env["swarm_halite"] = s_env["swarm_halite"] + s_env["ships_values"][i][1] - conf.convertCost
        actions[s_env["ships_keys"][i]] = "CONVERT"
        s_env["map"][x][y]["ship"] = None
        shipyards_amount += 1
        return True, actions, s_env, shipyards_amount
    return False, actions, s_env, shipyards_amount

def to_spawn_or_not_to_spawn(s_env):
    """ to spawn, or not to spawn, that is the question """
    # get ships_max_amount to decide whether to spawn new ships or not
    ships_max_amount = 9
    # if ships_max_amount is less than minimal allowed amount of ships in the Swarm
    if ships_max_amount < ships_min_amount or s_env["obs"].step >= spawn_stop_step:
        ships_max_amount = ships_min_amount
    return ships_max_amount

def define_some_globals(configuration):
    """ define some of the global variables """
    global conf
    global convert_threshold
    global max_moves_amount
    global spawn_stop_step
    global globals_not_defined
    conf = configuration
    convert_threshold = conf.convertCost + conf.spawnCost * 2
    max_moves_amount = 3
    spawn_stop_step = conf.episodeSteps - 150
    globals_not_defined = False

def adapt_environment(observation, configuration):
    """ adapt environment for the Swarm """
    s_env = {}
    s_env["obs"] = observation
    if globals_not_defined:
        define_some_globals(configuration)
    s_env["map"], s_env["average_halite"] = get_map_and_average_halite(s_env["obs"])
    s_env["low_amount_of_halite"] = s_env["average_halite"] * 0.75
    s_env["swarm_halite"] = s_env["obs"].players[s_env["obs"].player][0]
    s_env["swarm_shipyards_coords"], s_env["swarm_ships_coords"] = get_swarm_units_coords_and_update_map(s_env)
    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())
    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())
    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())
    s_env["ships_max_amount"] = to_spawn_or_not_to_spawn(s_env)
    return s_env
    
def actions_of_ships(s_env):
    """ actions of every ship of the Swarm """
    actions = {}
    shipyards_amount = len(s_env["shipyards_keys"])
    for i in range(len(s_env["swarm_ships_coords"])):
        x = s_env["swarm_ships_coords"][i][0]
        y = s_env["swarm_ships_coords"][i][1]
        # if this is a new ship
        if s_env["ships_keys"][i] not in ships_data:
            s_env = this_is_new_ship(s_env, i)
        # if it is last step
        ok, actions, s_env = this_is_last_step(x, y, actions, s_env, i)
        if ok:
            continue
        # if there is no shipyards, necessity to have shipyard, no hostile ships near
        # and enough halite to spawn few ships
        ok, actions, s_env, shipyards_amount = no_shipyards(x, y, actions, s_env, shipyards_amount, i)
        if ok:
            continue
        actions = move_ship(x, y, actions, s_env, i)
    return actions
     
def actions_of_shipyards(actions, s_env):
    """ actions of every shipyard of the Swarm """
    ships_amount = len(s_env["ships_keys"])
    # spawn ships from every shipyard, if possible
    for i in range(len(s_env["shipyards_keys"])):
        ok, actions, s_env, ships_amount = spawn_ship(actions, s_env, ships_amount, i)
        if not ok:
            break
    return actions


#GLOBAL_VARIABLES#############################################
conf = None
# max amount of moves in one direction before turning
max_moves_amount = None
# threshold of harvested by a ship halite to convert
convert_threshold = None
# no ship spawning above ships_min_amount after this step
spawn_stop_step = None
# object with ship ids and their data
ships_data = {}
# initial movement_tactics index
movement_tactics_index = 0
# minimum amount of ships that should be in the Swarm at any time
ships_min_amount = 3
# not all global variables are defined
globals_not_defined = True

# list of directions
directions_list = [
    {
        "direction": "NORTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z - 1)
    },
    {
        "direction": "EAST",
        "x": lambda z: get_c(z + 1),
        "y": lambda z: z
    },
    {
        "direction": "SOUTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z + 1)
    },
    {
        "direction": "WEST",
        "x": lambda z: get_c(z - 1),
        "y": lambda z: z
    }
]

# list of movement tactics
movement_tactics = [
    # N -> E -> S -> W
    {"directions": get_directions(0, 1, 2, 3)},
    # S -> E -> N -> W
    {"directions": get_directions(2, 1, 0, 3)},
    # N -> W -> S -> E
    {"directions": get_directions(0, 3, 2, 1)},
    # S -> W -> N -> E
    {"directions": get_directions(2, 3, 0, 1)},
    # E -> N -> W -> S
    {"directions": get_directions(1, 0, 3, 2)},
    # W -> S -> E -> N
    {"directions": get_directions(3, 2, 1, 0)},
    # E -> S -> W -> N
    {"directions": get_directions(1, 2, 3, 0)},
    # W -> N -> E -> S
    {"directions": get_directions(3, 0, 1, 2)},
]
movement_tactics_amount = len(movement_tactics)


#THE_SWARM####################################################
def swarm_agent(observation, configuration):
    """ RELEASE THE SWARM!!! """
    s_env = adapt_environment(observation, configuration)
    actions = actions_of_ships(s_env)
    actions = actions_of_shipyards(actions, s_env)
    return actions
