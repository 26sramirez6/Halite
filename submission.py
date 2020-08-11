'''
Created on Jul 26, 2020

@author: 26sra
'''
import sys #@UnusedImport
import numpy as np
import torch
import base64
import io
import torch.nn as nn
import datetime
import torch.nn.functional as F #@UnusedImport
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

EPISODE_STEPS = 400
STARTING = 5000
BOARD_SIZE = 21
PLAYERS = 4
GAMMA = 0.9
GAME_BATCH_SIZE = 1
CHANNELS = 7
MAX_ACTION_SPACE = 500
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
TS_FTR_COUNT = 1 + PLAYERS*2 
TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.').replace('-',"_")

class AgentStateManager:
    @classmethod
    def init_gamma_mat(cls, device):
        cls.gamma_vec = torch.tensor([GAMMA**i for i in range(EPISODE_STEPS)], dtype=torch.float).to(device) #@UndefinedVariable
        cls.gamma_mat = torch.zeros((EPISODE_STEPS, EPISODE_STEPS), dtype=torch.float).to(device) #@UndefinedVariable
        for i in range(EPISODE_STEPS):
            cls.gamma_mat[i, (1+i):] = cls.gamma_vec[:EPISODE_STEPS-(1+i)]
            
    def __init__(self, player_id):
        self.player_id = player_id
        self.game_id = 0
        self.prior_board = None
        self.total_episodes_seen = 0
        self.in_game_episodes_seen = 0
        self.geometric_ftrs = torch.zeros(
            (EPISODE_STEPS*GAME_BATCH_SIZE, CHANNELS, BOARD_SIZE, BOARD_SIZE), 
            dtype=torch.float).to(device) #@UndefinedVariable
            
        self.time_series_ftrs = torch.zeros(
            (EPISODE_STEPS*GAME_BATCH_SIZE, TS_FTR_COUNT), 
            dtype=torch.float).to(device) #@UndefinedVariable
            
        self.episode_rewards = torch.zeros(
            EPISODE_STEPS*GAME_BATCH_SIZE, 
            dtype=torch.float).to(device) #@UndefinedVariable
        
        self.q_values = torch.zeros(
            EPISODE_STEPS*GAME_BATCH_SIZE, 
            dtype=torch.float).to(device) #@UndefinedVariable
            
        self.current_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        self.prior_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        
        self.emulator = BoardEmulator(self.time_series_ftrs)
        
    def set_prior_board(self, prior_board):
        self.prior_board = prior_board
    
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self.prior_ship_cargo.copy_(prior_ship_cargo)
        self.emulator.set_prior_ship_cargo(prior_ship_cargo)
    
    def compute_q_post_game(self):
        torch.matmul(
            self.gamma_mat[:self.in_game_episodes_seen, :self.in_game_episodes_seen], 
            self.episode_rewards[self.total_episodes_seen: self.total_episodes_seen + self.in_game_episodes_seen], 
            out=self.q_values[self.total_episodes_seen: self.total_episodes_seen + self.in_game_episodes_seen]) #@UndefinedVariable
        
    def serialize(self):
        append = 'p{0}g{1}_{2}.tensor'.format(self.player_id, self.game_id, TIMESTAMP)
        torch.save(self.geometric_ftrs[:self.total_episodes_seen], 'geo_ftrs_{0}.tensor'.format(append))
        torch.save(self.time_series_ftrs[:self.total_episodes_seen], 'ts_ftrs_{0}.tensor'.format(append))
        torch.save(self.q_values[:self.total_episodes_seen], 'q_values_{0}.tensor'.format(append))
    
    def clear_data(self):
        self.total_episodes_seen = 0
        self.in_game_episodes_seen = 0
        self.geometric_ftrs.fill_(0)
        self.time_series_ftrs.fill_(0)
        self.episode_rewards.fill_(0)
        self.q_values.fill_(0)
    
    
class DQN(nn.Module):
    def __init__(self, conv_layers, fc_layers, fc_volume, filters, kernel, stride, pad, ts_ftrs):
        super(DQN, self).__init__()
        
        self._conv_layers = []
        self._relus = []
        self.trained_examples = 0
        height = DQN._compute_output_dim(BOARD_SIZE, kernel, stride, pad)
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
            if i!=0:
                height = DQN._compute_output_dim(height, kernel, stride, pad)
            
        
        self._fc_layers = []
        self._sigmoids = []
        for i in range(fc_layers):
            layer = nn.Linear(
                (height * height * filters + ts_ftrs) if i==0 else fc_volume, # number of neurons from previous layer
                fc_volume # number of neurons in output layer
                )
            nn.init.xavier_uniform_(layer.weight)
            sigmoid = nn.Sigmoid()
            self._fc_layers.append(layer)
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
        
        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = torch.cat((y, ts_x), dim=1) #@UndefinedVariable
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
player_zeros = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable

dqn = DQN(
    10, # number of conv layers
    2,  # number of fully connected layers at end
    32, # number of neurons in fully connected layers at end
    8,  # number of filters for conv layers (depth)
    3,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    TS_FTR_COUNT# number of extra time series features
    ).to(device)  
    
encoded_weights = '''gAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAA
AGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAA
aW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAmNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxAClScQEo
WA0AAABfY29udjAud2VpZ2h0cQJjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxAygo
WAcAAABzdG9yYWdlcQRjdG9yY2gKRmxvYXRTdG9yYWdlCnEFWA0AAAAyODQ5NDEwMzM2ODk2cQZY
AwAAAGNwdXEHTfgBTnRxCFFLAChLCEsHSwNLA3RxCShLP0sJSwNLAXRxColoAClScQt0cQxScQ1Y
CwAAAF9jb252MC5iaWFzcQ5oAygoaARoBVgNAAAAMjg0OTQxMDMzMTEzNnEPaAdLCE50cRBRSwBL
CIVxEUsBhXESiWgAKVJxE3RxFFJxFVgNAAAAX2NvbnYxLndlaWdodHEWaAMoKGgEaAVYDQAAADI4
NDk0MTAzMzQ3ODRxF2gHTUACTnRxGFFLAChLCEsISwNLA3RxGShLSEsJSwNLAXRxGoloAClScRt0
cRxScR1YCwAAAF9jb252MS5iaWFzcR5oAygoaARoBVgNAAAAMjg0OTQxMDMzMTMyOHEfaAdLCE50
cSBRSwBLCIVxIUsBhXEiiWgAKVJxI3RxJFJxJVgNAAAAX2NvbnYyLndlaWdodHEmaAMoKGgEaAVY
DQAAADI4NDk0MTAzMzQwMTZxJ2gHTUACTnRxKFFLAChLCEsISwNLA3RxKShLSEsJSwNLAXRxKolo
AClScSt0cSxScS1YCwAAAF9jb252Mi5iaWFzcS5oAygoaARoBVgNAAAAMjg0OTQxMDMzMzE1MnEv
aAdLCE50cTBRSwBLCIVxMUsBhXEyiWgAKVJxM3RxNFJxNVgNAAAAX2NvbnYzLndlaWdodHE2aAMo
KGgEaAVYDQAAADI4NDk0MTAzMzIwOTZxN2gHTUACTnRxOFFLAChLCEsISwNLA3RxOShLSEsJSwNL
AXRxOoloAClScTt0cTxScT1YCwAAAF9jb252My5iaWFzcT5oAygoaARoBVgNAAAAMjg0OTQxMDMz
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
'''
decoded = base64.b64decode(encoded_weights)
buffer = io.BytesIO(decoded)
dqn.load_state_dict(torch.load(buffer))
dqn.eval()

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
        my_shipyard.next_action = None if shipyard_actions[i]==0 else ShipyardAction(shipyard_actions[i]) 

def step_forward(board, ship_actions, shipyard_actions):
    set_next_actions(board, ship_actions, shipyard_actions)
    new_board = board.next()
    return new_board

def update_tensors(geometric_tensor, ts_tensor, board, current_ship_cargo, prior_ship_cargo):        
    cp = board.current_player
    current_ship_cargo.fill_(0)
    halite = board.observation["halite"]
    geometric_tensor[0] = torch.as_tensor(
        [halite[i:i+BOARD_SIZE] for i in 
         range(0, len(halite), BOARD_SIZE)], 
        dtype=torch.float) #@UndefinedVariable    
    for my_ship in cp.ships:
        geometric_tensor[1, my_ship.position.x, my_ship.position.y] = 1
        geometric_tensor[2, my_ship.position.x, my_ship.position.y] = my_ship.halite
        current_ship_cargo[0] += my_ship.halite
    for my_shipyard in cp.shipyards:
        geometric_tensor[3, my_shipyard.position.x, my_shipyard.position.y] = 1
    
    for i, player in enumerate(board.opponents):
        for enemy_ship in player.ships:
            geometric_tensor[4, enemy_ship.position.x, enemy_ship.position.y] = 1
            geometric_tensor[5, enemy_ship.position.x, enemy_ship.position.y] = enemy_ship.halite
            current_ship_cargo[i+1] += enemy_ship.halite
        for enemy_shipyard in player.shipyards:
            geometric_tensor[6, enemy_shipyard.position.x, enemy_shipyard.position.y] = 1
    
    ts_tensor[0] = board.configuration.episode_steps - board.step
    for i, player in enumerate(board.players.values()):
        ts_tensor[i+1] = player.halite
#     ts_tensor[1: PLAYERS+1] = ts_tensor[1: PLAYERS+1] / ts_tensor[1: PLAYERS+1].max()
     
    ts_tensor[-PLAYERS:] = torch.max(player_zeros, current_ship_cargo - prior_ship_cargo) #@UndefinedVariable
#     ts_tensor[-PLAYERS:] = ts_tensor[-PLAYERS:] / (ts_tensor[-PLAYERS:].max()
    
    return

class BoardEmulator:
    def __init__(self, agent_ts_ftrs):
        self._agent_ts_ftrs = agent_ts_ftrs
        self._geometric_ftrs = torch.zeros((MAX_ACTION_SPACE, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float).to(device) #@UndefinedVariable
        self._ts_ftrs = torch.zeros((MAX_ACTION_SPACE, TS_FTR_COUNT), dtype=torch.float).to(device) #@UndefinedVariable
        
        self._prior_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        self._current_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(device) #@UndefinedVariable
        
        self._rewards = torch.zeros(MAX_ACTION_SPACE, dtype=torch.float).to(device) #@UndefinedVariable
        # assume a max amount of ships and shipyards to
        # so we can allocate upfront
        self._ship_actions = np.zeros((MAX_ACTION_SPACE, BOARD_SIZE**2), dtype=np.int32)
        self._shipyard_actions = np.zeros((MAX_ACTION_SPACE, BOARD_SIZE**2), dtype=np.int32)
        self._best_ship_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        self._best_shipyard_actions = np.zeros(BOARD_SIZE**2, dtype=np.int32)
        
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self._prior_ship_cargo[:] = prior_ship_cargo
    
    def _permutate_all_states(self, board, ship_count, shipyard_count, current_halite):
        forward_batch_size = 0
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
                            
                    self._ship_actions[forward_batch_size, i] = j
                    
                for i, j in enumerate(y):
                    if SHIPYARD_ACTIONS[j]==ShipyardAction.SPAWN:
                        if shipyard_halite < board.configuration.spawn_cost:
                            j = 0 # set to none
                        else:
                            shipyard_halite -= board.configuration.spawn_cost
                    self._shipyard_actions[forward_batch_size, i] = j
                
                new_board = step_forward(
                    board, 
                    self._ship_actions[forward_batch_size], 
                    self._shipyard_actions[forward_batch_size])
                
                update_tensors(
                    self._geometric_ftrs[forward_batch_size], 
                    self._ts_ftrs[forward_batch_size], 
                    new_board, 
                    self._current_ship_cargo, 
                    self._prior_ship_cargo)

                self._rewards[forward_batch_size] = compute_reward(
                    new_board, 
                    board, 
                    self._ts_ftrs[forward_batch_size, -PLAYERS: ], 
                    self._ts_ftrs[forward_batch_size, 1:  1 + PLAYERS], 
                    self._agent_ts_ftrs[board.step, 1:  1 + PLAYERS])
                forward_batch_size += 1
        return forward_batch_size
    
    def _guess_best_states(self, model, board, ship_count, shipyard_count):
        # choose a random ship, find the best action for this ship while holding the rest at 0
        # then choose another ship, holding the previous ship at it's best action. 
        # continue for all ships
        max_q_value = float('-inf')
        forward_batch_size = 0
        self._ship_actions.fill(0)
        self._shipyard_actions.fill(0)
        ship_idxs = list(range(ship_count))
        shipyard_idxs = list(range(shipyard_count))
        np.random.shuffle(ship_idxs)
        np.random.shuffle(shipyard_idxs)
        for i in ship_idxs:
            for j, _ in enumerate(SHIP_ACTIONS):
                self._ship_actions[0, i] = j
            
                new_board = step_forward(
                    board, 
                    self._ship_actions[0], 
                    self._shipyard_actions[0])
                
                update_tensors(
                    self._geometric_ftrs[0], 
                    self._ts_ftrs[0], 
                    new_board, 
                    self._current_ship_cargo, 
                    self._prior_ship_cargo)
                
                reward = compute_reward(
                    new_board, 
                    board, 
                    self._ts_ftrs[0, -PLAYERS: ], 
                    self._ts_ftrs[0, 1:  1 + PLAYERS], 
                    self._agent_ts_ftrs[board.step, 1:  1 + PLAYERS])
                
                forward_batch_size += 1
                
                with torch.no_grad():
                    q_value = reward + model(self._geometric_ftrs[0:1], self._ts_ftrs[0:1]).item()
                if q_value > max_q_value:
                    max_q_value = q_value
                    self._best_ship_actions[:ship_count] = self._ship_actions[0, :ship_count]
                    self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[0, :shipyard_count]
                    
            self._ship_actions[0, i] = self._best_ship_actions[i]
            
        for i in shipyard_idxs:
            for j, _ in enumerate(SHIPYARD_ACTIONS):
                self._shipyard_actions[0,i] = j
            
                new_board = step_forward(board, self._ship_actions[0], self._shipyard_actions[0])
                update_tensors(
                    self._geometric_ftrs[0], 
                    self._ts_ftrs[0], 
                    new_board, 
                    self._current_ship_cargo, 
                    self._prior_ship_cargo)
                
                reward = compute_reward(
                    new_board, 
                    board, 
                    self._ts_ftrs[0, -PLAYERS: ], 
                    self._ts_ftrs[0, 1:  1 + PLAYERS], 
                    self._agent_ts_ftrs[board.step, 1:  1 + PLAYERS])
                                
                with torch.no_grad():
                    q_value = reward + model(self._geometric_ftrs[0:1], self._ts_ftrs[0:1]).item()
                if q_value > max_q_value:
                    max_q_value = q_value
                    self._best_ship_actions[:ship_count] = self._ship_actions[0, :ship_count]
                    self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[0, :shipyard_count]
                    
            self._shipyard_actions[0,i] = self._best_shipyard_actions[i]
    
    def select_action(self, board, model):
        model.eval()
        ship_count = len(board.current_player.ships)
        shipyard_count = len(board.current_player.shipyards)
        action_space = (len(SHIP_ACTIONS)**ship_count) * (len(SHIPYARD_ACTIONS)**shipyard_count)
        current_halite = board.current_player.halite
        
        if action_space > MAX_ACTION_SPACE:
            self._guess_best_states(model, board, ship_count, shipyard_count)
        else:
            forward_batch_size = self._permutate_all_states(
                board, ship_count, shipyard_count, current_halite)
            with torch.no_grad():
                q_values = (self._rewards[:forward_batch_size] + 
                    model(self._geometric_ftrs[:forward_batch_size], 
                          self._ts_ftrs[:forward_batch_size]).view(-1))
                    
            max_q_idx = q_values.argmax()
            self._best_ship_actions[:ship_count] = self._ship_actions[max_q_idx, :ship_count]
            self._best_shipyard_actions[:shipyard_count] = self._shipyard_actions[max_q_idx, :shipyard_count]
        set_next_actions(board, self._best_ship_actions, self._best_shipyard_actions)
            
mined_reward_weights = torch.tensor([2] + [-1]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
deposited_reward_weights = torch.tensor([10] + [-1]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
def compute_reward(
        current_board, 
        prior_board, 
        current_mined_halite,
        current_deposited_halite, 
        prior_deposited_halite):
    if current_board.step==0: return -100
        
    halite_cargo_reward = (current_mined_halite * 
                           mined_reward_weights).sum().item()
    
    halite_deposited_reward = max(0, (
        (current_deposited_halite -
         prior_deposited_halite)*
    deposited_reward_weights).sum().item())
    
    my_ships_lost_from_collision = (
        len(current_board.current_player.ships) - len(prior_board.current_player.ships) +
        len(current_board.current_player.shipyards) - len(prior_board.current_player.shipyards) 
    )
     
    reward = (halite_deposited_reward +
              halite_cargo_reward + 
              my_ships_lost_from_collision*-5 - 
              100)
    
    return reward

asm = AgentStateManager(0)
def agent(obs, config):
    global asm
    current_board = Board(obs, config)
    
    ftr_index = asm.total_episodes_seen + asm.in_game_episodes_seen
    update_tensors(
        asm.geometric_ftrs[ftr_index], 
        asm.time_series_ftrs[ftr_index], 
        current_board, 
        asm.current_ship_cargo, 
        asm.prior_ship_cargo)
    
    asm.set_prior_ship_cargo(asm.current_ship_cargo)
    asm.emulator.select_action(current_board, dqn)
    asm.set_prior_board(current_board)
    asm.in_game_episodes_seen += 1
    print("board step {} complete".format(asm.in_game_episodes_seen))
    return current_board.current_player.next_actions

env = make("halite", configuration={
    "size": BOARD_SIZE, "startingHalite": STARTING, "episodeSteps": EPISODE_STEPS,
    "runTimeout": 1e6, "actTimeout":1e6})
steps = env.run([agent, "random", "random", "random"])
out = env.render(mode="html", width=800, height=600)
with open("game_{0}.html".format(TIMESTAMP), "w") as f:
    f.write(out)