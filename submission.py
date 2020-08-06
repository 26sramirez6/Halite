'''
Created on Jul 26, 2020

@author: 26sra
'''
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import make
from random import choice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations, product

EPISODE_STEPS = 200
STARTING = 1000
BOARD_SIZE = 10
PLAYERS = 2
GAMMA = 0.9
EGREEDY = 0.4
EGREEDY_DECAY = 0.4
BATCH_SIZE = 32
LEARNING_RATE = 0.1
CHANNELS = 7
MOMENTUM  = 0.9
EPOCHS = 3
MAX_ACTION_SPACE = 1000
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]

class DQN(nn.Module):
    def __init__(self, conv_layers, fc_layers, fc_volume, filters, kernel, stride, pad, ts_ftrs):
        super(DQN, self).__init__()
        
        self._conv_layers = []
        self._relus = []
        
        for i in range(conv_layers):
            layer = nn.Conv2d(
                CHANNELS if i==0 else filters,   # number of in channels (depth of input)
                filters,    # out channels (depth, or number of filters)
                kernel,     # size of convolving kernel
                stride,     # stride of kernel
                pad)        # padding
            nn.init.xavier_uniform(layer.weight)
            relu = nn.ReLU()
            
            self._conv_layers.append(layer)
            self._relus.append(relu)
            # necessary to register layer
            setattr(self, "_conv{0}".format(i), layer)
            setattr(self, "_relu{0}".format(i), relu)
            
        volume = DQN._compute_output_dim(BOARD_SIZE, kernel, stride, pad)
        self._fc_layers = []
        self._sigmoids = []
        for i in range(fc_layers):
            layer = nn.Linear(
                (volume * volume * filters + ts_ftrs) if i==0 else fc_volume, # number of neurons from previous layer
                fc_volume # number of neurons in output layer
                )
            nn.init.xavier_uniform(layer.weight)
            sigmoid = nn.Sigmoid()
            self._sigmoids.append(sigmoid)
            
            # necessary to register layer
            setattr(self, "_fc{0}".format(i), layer)
            setattr(self, "_sigmoid{0}".format(i), sigmoid)
            
        self._final_layer = nn.Linear(
                fc_volume,
                1)
        
    def forward(self, geometric_x, ts_x):
        y = self._conv_layers[0](geometric_x)
        y = self._relus[0](y)
        for layer, activation in zip(self._conv_layers[1:], self._relus[1:]):
            y = layer(y)
            y = activation(y)
        
        y = y.view(-1)
        y = torch.cat((y, ts_x), dim=0)
        for layer, activation in zip(self._fc_layers, self._sigmoids):
            y = layer(y)
            y = activation(y)
        
        return self._final_layer(y)
    
    @staticmethod
    def _compute_output_dim(w, f, s, p):
        for el in [w,f,s,p]:
            DQN._check_and_convert(el)
            if not isinstance(el, int):
                raise ValueError()
        ret = (w-f+2*p)/s + 1
        return DQN._check_and_convert(ret)
    
    @staticmethod
    def _check_and_convert(f):
        if not float(f).is_integer():
            raise ValueError("error on {0}".format(f))
        return int(f)


halite_ftr = BOARD_SIZE**2
my_ship_exist_ftr = BOARD_SIZE**2
my_ship_halite_ftr = BOARD_SIZE**2
my_shipyard_exist_ftr = BOARD_SIZE**2
enemy_ship_exist_ftr = BOARD_SIZE**2
enemy_ship_halite_ftr = BOARD_SIZE**2
enemy_shipyard_exist_ftr = BOARD_SIZE**2
turns_remaining_ftr = 1 
halite_deposited_by_player_ftr = PLAYERS
halite_mined_by_player_ftr = PLAYERS

ftr_count = (
    halite_ftr + 
    my_ship_exist_ftr + 
    my_ship_halite_ftr + 
    my_shipyard_exist_ftr +
    enemy_ship_exist_ftr + 
    enemy_ship_halite_ftr + 
    enemy_shipyard_exist_ftr +
    turns_remaining_ftr +
    halite_deposited_by_player_ftr +
    halite_mined_by_player_ftr)

ts_ftr_count = (
    turns_remaining_ftr +
    halite_deposited_by_player_ftr +
    halite_mined_by_player_ftr)

geometric_ftr_count = (
    halite_ftr + 
    my_ship_exist_ftr + 
    my_ship_halite_ftr + 
    my_shipyard_exist_ftr +
    enemy_ship_exist_ftr + 
    enemy_ship_halite_ftr + 
    enemy_shipyard_exist_ftr)

game_step = 0
geometric_ftrs = torch.zeros((EPISODE_STEPS, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float)
time_series_ftrs = torch.zeros((EPISODE_STEPS, ts_ftr_count), dtype=torch.float)
episode_rewards = torch.zeros(EPISODE_STEPS, dtype=torch.float)

halite_mined = np.zeros(PLAYERS)    
ship_halite_prev = np.zeros(PLAYERS)
ship_halite_cur = np.zeros(PLAYERS)
player_zeros = np.zeros(PLAYERS)
ships_spawned = 0
shipyards_spawned = 0
halite_lost = 0
halite_stolen = 0

# criterion = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss() # huber
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DQN(
    10, # number of conv layers
    2,  # number of fully connected layers at end
    32, # number of neurons in fully connected layers at end
    2,  # number of filters for conv layers (depth)
    5,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    3)  # number of extra time series features

optimizer = torch.optim.SGD(
    net.parameters(), 
    lr=LEARNING_RATE, 
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)


class Features:
    def __init__(self, board_size):
        self.geometric = torch.zeros((3, board_size, board_size), dtype=torch.float)
        self.time_series = torch.zeros((), dtype=torch.float)
        
        
class ModelPropagator:
    def __init__(self, board_size, player_count):
        self._state = np.zeros(ftr_count)
        self._ship_halite_prev = np.zeros(player_count)
        self._ship_halite_cur = np.zeros(player_count)
        # assume a max amount of ships and shipyards to
        # so we can allocate upfront
        self._ship_actions = np.zeros(board_size**2, dtype=np.int32)
        self._shipyard_actions = np.zeros(board_size**2, dtype=np.int32)
        self._best_ship_actions = np.zeros(board_size**2, dtype=np.int32)
        self._best_shipyard_actions = np.zeros(board_size**2, dtype=np.int32)
        
    def set_ship_halite_prev(self, ship_halite_prev):
        self._ship_halite_prev[:] = ship_halite_prev
        
    def select_action(self, board, model):
        ship_count = len(board.current_player.ships)
        shipyard_count = len(board.current_player.shipyards)
        action_space = (len(SHIP_ACTIONS)**ship_count) * (len(SHIPYARD_ACTIONS)**shipyard_count)
        current_halite = board.current_player.halite
        max_q_value = float('-inf')
        if action_space > MAX_ACTION_SPACE:
            # choose a random ship, find the best action for this ship while holding the rest at 0
            # then choose another ship, holding the previous ship at it's best action. 
            # continue for all ships
            self._ship_actions.fill(0)
            self._shipyard_actions.fill(0)
            ship_idxs = list(range(ship_count))
            shipyard_idxs = list(range(shipyard_count))
            np.random.shuffle(ship_idxs)
            np.random.shuffle(shipyard_idxs)
            for i in ship_idxs:
                for j, _ in enumerate(SHIP_ACTIONS):
                    self._ship_actions[i] = j
                
                    new_board = step_forward(board, self._ship_actions, self._shipyard_actions)
                    update_state(self._state, new_board, self._ship_halite_cur, self._ship_halite_prev)
                    q_value = model(self._state)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        self._best_ship_actions[:ship_count] = self._ship_actions[:ship_count]
                        self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[:shipyard_count]
                        
                self._ship_actions[i] = self._best_ship_actions[i]
                
            for i in shipyard_idxs:
                for j, _ in enumerate(SHIPYARD_ACTIONS):
                    self._shipyard_actions[i] = j
                
                    new_board = step_forward(board, self._ship_actions, self._shipyard_actions)
                    update_state(self._state, new_board, self._ship_halite_cur, self._ship_halite_prev)
                    q_value = model(self._state)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        self._best_ship_actions[:ship_count] = self._ship_actions[:ship_count]
                        self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[:shipyard_count]
                        
                self._shipyard_actions[i] = self._best_shipyard_actions[i]
        else:
            for l in product(range(len(SHIP_ACTIONS)), repeat=ship_count):
                ship_halite = current_halite // 2
                for y in product(range(len(SHIPYARD_ACTIONS)), repeat=shipyard_count):
                    shipyard_halite = current_halite // 2
                    for i, j in enumerate(l):
                        if SHIP_ACTIONS[j]==ShipAction.CONVERT:
                            if ship_halite < board.configuration.convert_cost:
                                j = 0 # set to none action
                            else:
                                ship_halite -= board.configuration.convert_cost
                                
                        self._ship_actions[i] = j
                    for i, j in enumerate(y):
                        if SHIPYARD_ACTIONS[j]==ShipyardAction.SPAWN:
                            if shipyard_halite < board.configuration.spawn_cost:
                                j = 0 # set to none
                            else
                                shipyard_halite -= board.configuration.spawn_cost
                        self._shipyard_actions[i] = j
                    
                    new_board = step_forward(board, self._ship_actions, self._shipyard_actions)
                    update_state(self._state, new_board, self._ship_halite_cur, self._ship_halite_prev)
                    q_value = model(self._state)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        self._best_ship_actions[:ship_count] = self._ship_actions[:ship_count]
                        self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[:shipyard_count]
        
        set_next_actions(board, self._best_ship_actions, self._best_shipyard_actions)
        
mp = ModelPropagator(PLAYERS)
    
def train(model, device, training, optimizer, epoch, criterion):
    model.train()
    training = torch.from_numpy(training).float()
    for idx, batch_idx in enumerate(range(0, training.size()[0], BATCH_SIZE)):
        X = training[batch_idx:batch_idx+BATCH_SIZE, :-1].to(device)
        y = training[batch_idx:batch_idx+BATCH_SIZE, -1].to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))
            
def compute_reward(board):
    if (board.step==0): return -100
        
    halite_mined_reward = (
        episodes[board.step, -halite_mined_by_player_ftr: ] * 
    ([2] + [-1]*(PLAYERS-1))).sum()
    
    halite_deposited_reward = (
        (episodes[board.step, -(halite_mined_by_player_ftr + halite_deposited_by_player_ftr):  -halite_mined_by_player_ftr] -
         episodes[board.step-1, -(halite_mined_by_player_ftr + halite_deposited_by_player_ftr):  -halite_mined_by_player_ftr])*
    ([10] + [-1]*(PLAYERS-1))).sum()
        
    reward = (halite_deposited_reward +
              halite_mined_reward + 
              ships_spawned*5 + 
              shipyards_spawned*5 +
              halite_lost*-20 + 
              halite_stolen*10 -
              100)
    
    return reward

def randomize_action(board):
    current_halite = board.current_player.halite
    for my_ship in board.current_player.ships:
        if current_halite > board.configuration.convert_cost:
            my_ship.next_action = choice(SHIP_ACTIONS)
            if my_ship.next_action==ShipAction.CONVERT:
                current_halite -= board.configuration.convert_cost
        else:
            my_ship.next_action = choice(SHIP_MOVE_ACTIONS)
            
    for my_shipyard in board.current_player.next_action.shipyards:
        if current_halite > board.configuration.spawn_cost:
            my_shipyard.next_action = choice(SHIPYARD_ACTIONS)
            current_halite -= board.configuration.spawn_cost
        else:
            my_shipyard.next_action = None

def new_position(size, ship, action):
    ret = ship.position
    if action==ShipAction.NORTH:
        if ret.y==size-1:
            ret = Point(ret.x, 0)
        else:
            ret = ret + Point(0,1)
    elif action==ShipAction.EAST:
        if ret.x==size-1:
            ret = Point(0, ret.y)
        else:
            ret = ret + Point(1,0)
    elif action==ShipAction.SOUTH:
        if ret.y==0:
            ret = Point(ret.x, size-1)
        else:
            ret = ret - Point(1,0)
    elif action==ShipAction.WEST:
        if ret.x==0:
            ret = Point(size-1, ret.y)
        else:
            ret = ret - Point(1,0)
        
    return ret

def min_max_norm(arr):
    arr[:] = 2 * ((arr - arr[0]) / (arr[-1] - arr[0])) - 1

def set_next_actions(board, ship_actions, shipyard_actions):
    cp = board.current_player
    for i, my_ship in enumerate(cp.ships):
        my_ship.next_action = None if ship_actions[i]==0 else ShipAction(ship_actions[i]) 
    for i, my_shipyard in enumerate(cp.shipyards):
        my_shipyard.next_action = None if shipyard_actions[i]==0 else ShipyardAction(ship_actions[i]) 
          
def step_forward(board, ship_actions, shipyard_actions):
    set_next_actions(board, ship_actions, shipyard_actions)
    new_board = board.next()
    return new_board

def update_tensors(geometric_tensor, ts_tensor, board, ship_halite_cur, ship_halite_prev):        
    cp = board.current_player
    ship_halite_cur.fill(0)
    # there doesn't seem to be a good way to assign a list/array to an existing tensor
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            geometric_tensor[0, i, j] = board.observation["halite"][i * BOARD_SIZE + j]
                
    for my_ship in cp.ships:
        geometric_tensor[1, my_ship.position.x, my_ship.position.y] = 1
        geometric_tensor[2, my_ship.position.x, my_ship.position.y] = my_ship.halite
        ship_halite_cur[0] += my_ship.halite
    for my_shipyard in cp.shipyards:
        geometric_tensor[3, my_shipyard.position.x, my_shipyard.position.y] = 1
    
    for i, player in enumerate(board.opponents):
        for enemy_ship in player.ships:
            geometric_tensor[4, enemy_ship.position.x, enemy_ship.position.y] = 1
            geometric_tensor[5, enemy_ship.position.x, enemy_ship.position.y] = enemy_ship.halite
        for enemy_shipyard in player.shipyards:
            geometric_tensor[6, enemy_shipyard.position.x, enemy_shipyard.position.y] = 1
    
    ts_tensor[0] = board.configuration.episode_steps - board.step
#     for i, player in enumerate(board.players.values()):
#         state[ftr_index + i] = player.halite
#     min_max_norm(state[ftr_index: ftr_index+halite_deposited_by_player_ftr])
#     ftr_index += halite_deposited_by_player_ftr
#     
#     # should we also min_max_norm the amount of halite mined?
#     state[-halite_mined_by_player_ftr:] = np.maximum(player_zeros, ship_halite_cur - ship_halite_prev)
#     min_max_norm(state[-halite_mined_by_player_ftr:])
    
    return

def update_state_old(state, board, ship_halite_cur, ship_halite_prev):        
    cp = board.current_player
    state[:halite_ftr] = board.observation["halite"]
    ship_halite_cur.fill(0)
    
    for my_ship in cp.ships:
        pos = my_ship.position
        ship_index = (pos.y)*BOARD_SIZE + pos.x
        state[ halite_ftr + ship_index] = 1
        state[ halite_ftr + my_ship_exist_ftr + ship_index] = my_ship.halite
        ship_halite_cur[0] += my_ship.halite
    for my_shipyard in cp.shipyards:
        pos = my_shipyard.position
        shipyard_index = (pos.y)*BOARD_SIZE + pos.x
        state[ halite_ftr + my_ship_exist_ftr + my_ship_halite_ftr + shipyard_index] = 1
        
    ftr_index = halite_ftr + my_ship_exist_ftr + my_ship_halite_ftr + my_shipyard_exist_ftr
    for i, player in enumerate(board.opponents):
        for enemy_ship in player.ships:
            pos = enemy_ship.position
            ship_index = (pos.y)*BOARD_SIZE + pos.x
            state[ ftr_index + ship_index] = 1
            state[ ftr_index + enemy_ship_exist_ftr + ship_index] = enemy_ship.halite
            ship_halite_cur[i+1] += enemy_ship.halite
        for enemy_shipyard in player.shipyards:
            pos = enemy_shipyard.position
            shipyard_index = (pos.y)*BOARD_SIZE + pos.x
            state[ ftr_index + enemy_ship_exist_ftr + enemy_ship_halite_ftr + shipyard_index] = 1
    
    ftr_index += enemy_ship_exist_ftr + enemy_ship_halite_ftr + enemy_shipyard_exist_ftr
    state[ftr_index] = board.configuration.episode_steps - board.step
    ftr_index += 1
    # the halite score changes based on board configuration. Probably best to
    # not use the current score and instead use the amount deposited by
    # each of the players in the previous round? or apply min-max normalization
    # to create a better signal.
    for i, player in enumerate(board.players.values()):
        state[ftr_index + i] = player.halite
    min_max_norm(state[ftr_index: ftr_index+halite_deposited_by_player_ftr])
    ftr_index += halite_deposited_by_player_ftr
    
    # should we also min_max_norm the amount of halite mined?
    state[-halite_mined_by_player_ftr:] = np.maximum(player_zeros, ship_halite_cur - ship_halite_prev)
    min_max_norm(state[-halite_mined_by_player_ftr:])
    
    return

    
def agent(obs, config):
    board = Board(obs, config)
    update_tensor(geometric_states[board.step], board)
#     ship_halite_prev[:] = ship_halite_cur
#     mp.set_ship_halite_prev(ship_halite_cur)
    episode_rewards[board.step] = compute_reward(board)
    print(board.step, episode_rewards[board.step])
    
    if np.random.rand() < EGREEDY:
        randomize_action(board)
    else:
        mp.select_action(board, net)
        
    return board.current_player.next_actions
    
env = make("halite", configuration={"size": BOARD_SIZE, "startingHalite": STARTING, "episodeSteps": EPISODE_STEPS})
env.reset(PLAYERS)
print("starting")
env.run([agent, "random"])
print("complete")

out = env.render(mode="html", width=800, height=600)
with open("test.html", "w") as f:
    f.write(out)