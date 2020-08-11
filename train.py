'''
Created on Jul 26, 2020

@author: 26sra
'''
import sys #@UnusedImport
import numpy as np
import torch
import os
import time
import functools
import torch.nn as nn
import torch.nn.functional as F #@UnusedImport
import datetime
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

EPISODE_STEPS = 50
STARTING = 5000
BOARD_SIZE = 21
PLAYERS = 4
GAMMA = 0.9
EGREEDY = 1
EGREEDY_DECAY = 0.0005
EGREEDY_LOWER_BOUND = 0.2
GAME_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 0.1
CHANNELS = 7
MOMENTUM  = 0.9
SAMPLE_CYCLE = 10
OUTPUT_LOGS = True
EPOCHS = 3
MAX_ACTION_SPACE = 500
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
TS_FTR_COUNT = 1 + PLAYERS*2 
GAME_COUNT = 10000
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
        try:
            torch.matmul(
                self.gamma_mat[:self.in_game_episodes_seen, :self.in_game_episodes_seen], 
                self.episode_rewards[self.total_episodes_seen: self.total_episodes_seen + self.in_game_episodes_seen], 
                out=self.q_values[self.total_episodes_seen: self.total_episodes_seen + self.in_game_episodes_seen]) #@UndefinedVariable
        except Exception as e:
            raise e
        
    def serialize(self):
        append = 'p{0}g{1}_{2}'.format(self.player_id, self.game_id, TIMESTAMP)
        torch.save(self.geometric_ftrs[:self.total_episodes_seen], '{0}/geo_ftrs_{1}.tensor'.format(TIMESTAMP, append))
        torch.save(self.time_series_ftrs[:self.total_episodes_seen], '{0}/ts_ftrs_{1}.tensor'.format(TIMESTAMP, append))
        torch.save(self.q_values[:self.total_episodes_seen], '{0}/q_values_{1}.tensor'.format(TIMESTAMP, append))
    
    def clear_data(self):
        self.total_episodes_seen = 0
        self.in_game_episodes_seen = 0
        self.geometric_ftrs.fill_(0)
        self.time_series_ftrs.fill_(0)
        self.episode_rewards.fill_(0)
        self.q_values.fill_(0)
    
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

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
halite_lost_from_collision = 0
halite_stolen_from_collision = 0
# criterion = nn.CrossEntropyLoss()
huber = nn.SmoothL1Loss()
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
    
# dqn.load_state_dict(torch.load("{0}/dqn_{0}.nn".format(TIMESTAMP)))

optimizer = torch.optim.SGD( #@UndefinedVariable
    dqn.parameters(), 
    lr=LEARNING_RATE, 
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)

AgentStateManager.init_gamma_mat(device)

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
        
#         print("action_space:", action_space, file=sys.__stdout__)
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
    
def train(model, criterion, agent_managers):
    model.train()
    for e in range(EPOCHS):
        for asm in agent_managers.values():
            idxs = list(range(asm.total_episodes_seen))
            np.random.shuffle(idxs)
            for j, i in enumerate(range(0, len(idxs), TRAIN_BATCH_SIZE)):
                train_idx = idxs[i:i+TRAIN_BATCH_SIZE]
                y_pred = model(
                    asm.geometric_ftrs[train_idx], 
                    asm.time_series_ftrs[train_idx])
                loss = criterion(y_pred.view(-1), asm.q_values[train_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print("Epoch: {}, Train batch iteration: {}, Loss: {}".format(e, j, loss.item()))
            model.trained_examples += len(idxs)    
    torch.save(model.state_dict(), "{0}/dqn_{0}.nn".format(TIMESTAMP))
        
mined_reward_weights = torch.tensor([1] + [-.25]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
# deposited_reward_weights = torch.tensor([1] + [-.1]*(PLAYERS-1), dtype=torch.float).to(device) #@UndefinedVariable
def compute_reward(
        current_board, 
        prior_board, 
        current_mined_halite,
        current_deposited_halite, 
        prior_deposited_halite):
    if current_board.step==0: return 0
        
    halite_cargo_reward = (current_mined_halite * 
                           mined_reward_weights).sum().item()
    
    halite_deposited_score_diff = (max(current_deposited_halite) - current_deposited_halite[0]).item()
    
    halite_deposited_reward = max(0,(current_deposited_halite[0] - prior_deposited_halite[0]).item())
    
    prior_ships_set = set([ship.id for ship in prior_board.current_player.ships])
    current_ships_set = set([ship.id for ship in current_board.current_player.ships])
    prior_shipyards_set = set([shipyard.id for shipyard in prior_board.current_player.shipyards])
    current_shipyards_set = set([shipyard.id for shipyard in current_board.current_player.shipyards])
    
    my_ships_lost_from_collision = len(prior_ships_set.difference(current_ships_set))
    my_shipyards_lost_from_collision = len(prior_shipyards_set.difference(current_shipyards_set))
    
    reward = (halite_deposited_score_diff*-1 +
              halite_deposited_reward*1 +
              halite_cargo_reward*1 + 
              my_ships_lost_from_collision*-500 +
              my_shipyards_lost_from_collision*-500)
    
    return reward

agent_managers = {}
def agent(obs, config):
    global agent_managers
    current_board = Board(obs, config)
    if current_board.current_player.id not in agent_managers:
        agent_managers[current_board.current_player.id] = AgentStateManager(current_board.current_player.id)
    
    asm = agent_managers.get(current_board.current_player.id)
    ftr_index = asm.total_episodes_seen + asm.in_game_episodes_seen
    update_tensors(
        asm.geometric_ftrs[ftr_index], 
        asm.time_series_ftrs[ftr_index], 
        current_board, 
        asm.current_ship_cargo, 
        asm.prior_ship_cargo)
    
    asm.set_prior_ship_cargo(asm.current_ship_cargo)
    
    reward = compute_reward(
        current_board, 
        asm.prior_board, 
        asm.time_series_ftrs[ftr_index, -PLAYERS: ], 
        asm.time_series_ftrs[ftr_index, 1:  1 + PLAYERS], 
        None if current_board.step==0 else asm.time_series_ftrs[ftr_index-1, 1:  1 + PLAYERS])
    
    asm.episode_rewards[ftr_index] = reward
#     print("board step:", 
#           current_board.step, 
#           ", reward:", 
#           reward, 
#           file=sys.__stdout__)
    
    epsilon = max(EGREEDY_LOWER_BOUND, EGREEDY*((1-EGREEDY_DECAY)**dqn.trained_examples))
    randomize = np.random.rand() < epsilon
#     print("epsilon: ", epsilon, "randomize: ", randomize)
    if randomize:
        randomize_action(current_board)
    else:
        asm.emulator.select_action(current_board, dqn)
        
    asm.set_prior_board(current_board)
    asm.in_game_episodes_seen += 1
    
    return current_board.current_player.next_actions

if not os.path.exists(TIMESTAMP):
    os.makedirs(TIMESTAMP)
env = make("halite", configuration={"size": BOARD_SIZE, "startingHalite": STARTING, "episodeSteps": EPISODE_STEPS, "actTimeout": 1e8, "runTimeout":1e8})

def output_logs(env, steps, agent_managers):
    if hasattr(env, "logs") and env.logs is not None:
        with open("{0}/log_{0}.txt".format(TIMESTAMP), "w") as f:
            f.write('\n'.join([str(t) for t in env.logs]))
    with open("{0}/steps_{0}.txt".format(TIMESTAMP), "w") as f:
        f.write('\n'.join([str(l) for l in steps]))
    out = env.render(mode="html", width=800, height=600)
    with open("{0}/game_{0}.html".format(TIMESTAMP), "w") as f:
        f.write(out)
    
    for asm in agent_managers.values():
        with open("{0}/p{1}_rewards_{0}.txt".format(TIMESTAMP, asm.player_id), "w") as f:
            f.write(str(asm.episode_rewards[asm.total_episodes_seen:
                                        asm.total_episodes_seen+asm.in_game_episodes_seen]))
        
        with open("{0}/p{1}_qvals_{0}.txt".format(TIMESTAMP, asm.player_id), "w") as f:
            f.write(str(asm.q_values[asm.total_episodes_seen:
                                     asm.total_episodes_seen+asm.in_game_episodes_seen]))
            
print(env.configuration)
i = 1
from Halite_Swarm_Intelligence import swarm_agent
agents = [agent, agent, swarm_agent, "random"]
while i < GAME_COUNT:
    env.reset(PLAYERS)
    np.random.shuffle(agents)
    active_agents = set([j for j, el in enumerate(agents) if el==agent])
    print("starting game {0} with agent order: {1}".format(i, agents))
    active_agent_managers = {j:asm for j,asm in agent_managers.items() if j in active_agents}
    steps = env.run(agents)
    print("completed game {0}".format(i))
    
    for asm in active_agent_managers.values():
        asm.compute_q_post_game()
    
    if OUTPUT_LOGS:
        output_logs(env, steps, active_agent_managers)
        print("outputted data files")
        
    for asm in agent_managers.values():
        asm.total_episodes_seen += asm.in_game_episodes_seen
        asm.in_game_episodes_seen = 0
        
    if i % GAME_BATCH_SIZE == 0:
        train(dqn, huber, agent_managers)
        for asm in agent_managers.values():
            asm.serialize()
            asm.clear_data()
            asm.game_id += GAME_BATCH_SIZE
    
    i += 1