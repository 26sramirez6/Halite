'''
Created on Jul 26, 2020

@author: 26sra
'''
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #@UnusedImport
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

EPISODE_STEPS = 200
STARTING = 1000
BOARD_SIZE = 21
PLAYERS = 2
GAMMA = 0.9
EGREEDY = 0.4
EGREEDY_DECAY = 0.4
BATCH_SIZE = 32
LEARNING_RATE = 0.1
CHANNELS = 7
MOMENTUM  = 0.9
SAMPLE_CYCLE = 10
EPOCHS = 3
MAX_ACTION_SPACE = 1000
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
TS_FTR_COUNT = 1 + PLAYERS*2 

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
            nn.init.xavier_uniform_(layer.weight)
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
            nn.init.xavier_uniform_(layer.weight)
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
        print(y.size(), file=sys.__stdout__)
        for layer, activation in zip(self._conv_layers[1:], self._relus[1:]):            
            y = layer(y)
            print(y.size(), file=sys.__stdout__)
            y = activation(y)
        
        y = y.view(-1)
        y = torch.cat((y, ts_x), dim=0) #@UndefinedVariable
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #@UndefinedVariable
geometric_ftrs = torch.zeros((EPISODE_STEPS, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float).to(device) #@UndefinedVariable
time_series_ftrs = torch.zeros((EPISODE_STEPS, TS_FTR_COUNT), dtype=torch.float).to(device) #@UndefinedVariable
episode_rewards = torch.zeros(EPISODE_STEPS, dtype=torch.float).to(device) #@UndefinedVariable
q_values = torch.zeros(EPISODE_STEPS, dtype=torch.float).to(device) #@UndefinedVariable
ship_halite_cur = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
ship_halite_prev = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
player_zeros = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
halite_mined = np.zeros(PLAYERS)    
ships_spawned = 0
shipyards_spawned = 0
halite_lost = 0
halite_stolen = 0

# criterion = nn.CrossEntropyLoss()
huber = nn.SmoothL1Loss()
dqn = DQN(
    10, # number of conv layers
    2,  # number of fully connected layers at end
    32, # number of neurons in fully connected layers at end
    2,  # number of filters for conv layers (depth)
    3,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    TS_FTR_COUNT# number of extra time series features
    ).to(device)  

optimizer = torch.optim.SGD( #@UndefinedVariable
    dqn.parameters(), 
    lr=LEARNING_RATE, 
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)
        
def randomize_action(board):
    current_halite = board.current_player.halite
    for my_ship in board.current_player.ships:
        if current_halite > board.configuration.convert_cost:
            my_ship.next_action = choice(SHIP_ACTIONS)
            if my_ship.next_action==ShipAction.CONVERT:
                current_halite -= board.configuration.convert_cost
        else:
            my_ship.next_action = choice(SHIP_MOVE_ACTIONS)
            
    for my_shipyard in board.current_player.shipyards:
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

def min_max_norm(tensor):
    tensor[:] = 2 * ((tensor - tensor[0]) / (tensor[-1] - tensor[0])) - 1

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
    ship_halite_cur.fill_(0)
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
            ship_halite_cur[i+1] += enemy_ship.halite
        for enemy_shipyard in player.shipyards:
            geometric_tensor[6, enemy_shipyard.position.x, enemy_shipyard.position.y] = 1
    
    ts_tensor[0] = board.configuration.episode_steps - board.step
    for i, player in enumerate(board.players.values()):
        ts_tensor[i+1] = player.halite
    ts_tensor[1: PLAYERS+1] = ts_tensor[1: PLAYERS+1] / ts_tensor[1: PLAYERS+1].max()
     
    ts_tensor[-PLAYERS:] = torch.max(player_zeros, ship_halite_cur - ship_halite_prev) #@UndefinedVariable
    ts_tensor[-PLAYERS:] = ts_tensor[-PLAYERS:] / ts_tensor[-PLAYERS:].max()
    
    return

class BoardEmulator:
    def __init__(self):
        self._geometric_ftrs = torch.zeros((1, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float).to(device) #@UndefinedVariable
        self._ts_ftrs = torch.zeros((1,TS_FTR_COUNT), dtype=torch.float).to(device) #@UndefinedVariable
        
        self._ship_halite_prev = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        self._ship_halite_cur = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        
        # assume a max amount of ships and shipyards to
        # so we can allocate upfront
        self._ship_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        self._shipyard_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        self._best_ship_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        self._best_shipyard_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        
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
                    update_tensors(self._geometric_ftrs[0], self._ts_ftrs[0], new_board, self._ship_halite_cur, self._ship_halite_prev)
                    with torch.no_grad():
                        q_value = model(self._geometric_ftrs, self._ts_ftrs)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        self._best_ship_actions[:ship_count] = self._ship_actions[:ship_count]
                        self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[:shipyard_count]
                        
                self._ship_actions[i] = self._best_ship_actions[i]
                
            for i in shipyard_idxs:
                for j, _ in enumerate(SHIPYARD_ACTIONS):
                    self._shipyard_actions[i] = j
                
                    new_board = step_forward(board, self._ship_actions, self._shipyard_actions)
                    update_tensors(self._geometric_ftrs[0], self._ts_ftrs[0], new_board, self._ship_halite_cur, self._ship_halite_prev)
                    with torch.no_grad():
                        q_value = model(self._geometric_ftrs, self._ts_ftrs)
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
                            else:
                                shipyard_halite -= board.configuration.spawn_cost
                        self._shipyard_actions[i] = j
                    
                    new_board = step_forward(board, self._ship_actions, self._shipyard_actions)
                    update_tensors(self._geometric_ftrs[0], self._ts_ftrs[0], new_board, self._ship_halite_cur, self._ship_halite_prev)
                    with torch.no_grad():
                        q_value = model(self._geometric_ftrs, self._ts_ftrs)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        self._best_ship_actions[:ship_count] = self._ship_actions[:ship_count]
                        self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[:shipyard_count]
        
        set_next_actions(board, self._best_ship_actions, self._best_shipyard_actions)
    
def train(model, geometric_sample, ts_samples, q_values, optimizer, epoch, criterion):
    model.train()
    y_pred = model(geometric_sample, ts_samples)
    loss = criterion(y_pred, q_values)
    # SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
mined_reward_weights = torch.tensor([2] + [-1]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
deposited_reward_weights = torch.tensor([10] + [-1]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
def compute_reward(board):
    if board.step==0: return -100
        
    halite_mined_reward = (time_series_ftrs[board.step, -PLAYERS: ] * 
                           mined_reward_weights).sum().item()
    
    halite_deposited_reward = (
        (time_series_ftrs[board.step, 1:  1 + PLAYERS] -
         time_series_ftrs[board.step-1, 1:  1 + PLAYERS])*
    deposited_reward_weights).sum().item()
        
    reward = (halite_deposited_reward +
              halite_mined_reward + 
              ships_spawned*5 + 
              shipyards_spawned*5 +
              halite_lost*-20 + 
              halite_stolen*10 -
              100)
    
    return reward

emulator = BoardEmulator()
def agent(obs, config):
    board = Board(obs, config)
    update_tensors(geometric_ftrs[board.step], time_series_ftrs[board.step], board, ship_halite_cur, ship_halite_prev)
    ship_halite_prev[:] = ship_halite_cur
    emulator.set_ship_halite_prev(ship_halite_cur)
    episode_rewards[board.step] = compute_reward(board)
        
    print(board.step, episode_rewards[board.step])
    
    if np.random.rand() < EGREEDY:
        randomize_action(board)
    else:
        emulator.select_action(board, dqn)
    return board.current_player.next_actions

gamma_vec = torch.tensor([GAMMA**i for i in range(EPISODE_STEPS)], dtype=torch.float).to(device) #@UndefinedVariable
gamma_mat = torch.zeros((EPISODE_STEPS, EPISODE_STEPS), dtype=torch.float).to(device) #@UndefinedVariable
for i in range(EPISODE_STEPS):
    gamma_mat[i, i:] = gamma_vec[:EPISODE_STEPS-i]
def compute_q_post_game():
    torch.matmul(gamma_mat, episode_rewards, out=q_values) #@UndefinedVariable
    
env = make("halite", configuration={"size": BOARD_SIZE, "startingHalite": STARTING, "episodeSteps": EPISODE_STEPS})
env.reset(PLAYERS)
print("starting")
env.run([agent, "random"])
print("complete")
compute_q_post_game()

out = env.render(mode="html", width=800, height=600)
with open("test.html", "w") as f:
    f.write(out)