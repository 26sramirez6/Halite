'''
Created on Aug 21, 2020

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
import matplotlib.pyplot as plt
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

EPISODE_STEPS = 400
MAX_EPISODES_MEMORY = 4000
MAX_SHIPS = 100
STARTING = 5000
BOARD_SIZE = 21
PLAYERS = 4
GAMMA = 0.4
EGREEDY = 0.1
EGREEDY_LOWER_BOUND = 0.1
EGREEDY_DECAY = 0.0001
GAME_BATCH_SIZE = 1
TRAIN_BATCH_SIZE = 48
LEARNING_RATE = 0.0001
CHANNELS = 1
MOMENTUM  = 0.9
EPOCHS = 1
MAX_ACTION_SPACE = 50
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
TS_FTR_COUNT = 1 + PLAYERS*2 
GAME_COUNT = 100
TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.').replace('-',"_")
SERIALIZE = False
OUTPUT_LOGS = True
PRINT_STATEMENTS = True
RANDOM_SEED = 123; 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #@UndefinedVariable
if RANDOM_SEED > -1: 
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class AgentStateManager:        
    def __init__(self, 
            player_id,
            ship_model, 
            shipyard_model, 
            ship_model_criterion, 
            shipyard_model_criterion, 
            ship_model_optimizer, 
            shipyard_model_optimizer):
        
        self.player_id = player_id
        self.game_id = 0
        self.prior_board = None
        self.prior_ships_converted = 0
        self.ship_model = ship_model
        self.shipyard_model = shipyard_model
        self._ship_model_samples = 0
        self._shipyard_model_samples = 0
        self.total_episodes_seen = 0
        self.in_game_episodes_seen = 0
        self._ship_buffer_filled = False
        self._shipyard_buffer_filled = False
        
        self._geo_ship_ftrs = torch.zeros(
            (MAX_EPISODES_MEMORY, CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
        
        self._geo_shipyard_ftrs = torch.zeros(
            (MAX_EPISODES_MEMORY, CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
        
        self._ts_ftrs = torch.zeros(
            (MAX_EPISODES_MEMORY, TS_FTR_COUNT),
            dtype=torch.float).to(DEVICE)
        
        self._Q_ship = torch.zeros(
            MAX_EPISODES_MEMORY*(MAX_SHIPS),
            dtype=torch.float).to(DEVICE)
        
        self._Q_shipyard = torch.zeros(
            MAX_EPISODES_MEMORY*(MAX_SHIPS),
            dtype=torch.float).to(DEVICE)
        
        self._target_Q_ship = torch.zeros(
            MAX_EPISODES_MEMORY*(MAX_SHIPS),
            dtype=torch.float).to(DEVICE)
        
        self._target_Q_shipyard = torch.zeros(
            MAX_EPISODES_MEMORY*(MAX_SHIPS),
            dtype=torch.float).to(DEVICE)
        
        self._tdiffs_ship = torch.zeros(
            MAX_EPISODES_MEMORY,
            dtype=torch.float64).to(DEVICE)
        self._tdiffs_ship.fill_(1e-6)
        
        self._tdiffs_shipyard = torch.zeros(
            MAX_EPISODES_MEMORY,
            dtype=torch.float64).to(DEVICE)
        self._tdiffs_shipyard.fill_(1e-6)
        self.current_ship_cargo = torch.zeros(
            PLAYERS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
            
        self.prior_ship_cargo = torch.zeros(
            PLAYERS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self.episode_rewards = torch.zeros(
            MAX_EPISODES_MEMORY, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self.action_selector = ActionSelector(
            self, 
            self._ts_ftrs, 
            ship_model, 
            shipyard_model, 
            ship_model_criterion, 
            shipyard_model_criterion, 
            ship_model_optimizer, 
            shipyard_model_optimizer)     
        
    def set_prior_board(self, prior_board):
        self.prior_board = prior_board
    
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self.prior_ship_cargo.copy_(prior_ship_cargo)
        self.action_selector.set_prior_ship_cargo(prior_ship_cargo)
    
    def compute_total_reward_post_game(self):
        return self.episode_rewards[self.total_episodes_seen: 
            self.total_episodes_seen + self.in_game_episodes_seen].sum().item()
    
    def priority_sample(self, batch_size, is_ship_model):
        if is_ship_model:
            ship_end = MAX_EPISODES_MEMORY if self._ship_buffer_filled else self._ship_model_samples
            try:
                ship_idxs = np.random.choice(
                    list(range(ship_end)), 
                    size=batch_size, 
                    replace=True, 
                    p=self._target_Q_ship_priorities)
            except Exception as e:
                raise e
            return (self._geo_ship_ftrs[ship_idxs], 
                    self._ts_ftrs[ship_idxs], 
                    self._target_Q_ship[ship_idxs])
        else:
            shipyard_end = MAX_EPISODES_MEMORY if self._shipyard_buffer_filled else self._shipyard_model_samples
            shipyard_idxs = np.random.choice(
                list(range(shipyard_end)), 
                size=batch_size, 
                replace=True, 
                p=self._target_Q_shipyard_priorities) 
            return (self._geo_shipyard_ftrs[shipyard_idxs], 
                    self._ts_ftrs[shipyard_idxs], 
                    self._target_Q_shipyard[shipyard_idxs])       
        
    def store(self,
        count,
        geo_ftrs, 
        pred_Q,
        target_Q,
        is_ship_model) :   
             
        if is_ship_model:        
            if self._ship_model_samples + count > MAX_EPISODES_MEMORY:
                self._ship_buffer_filled = True
                self._ship_model_samples = 0
            
            start_ship = self._ship_model_samples
            end_ship = self._ship_model_samples + count
            
            self._geo_ship_ftrs[start_ship:end_ship] = geo_ftrs
            self._Q_ship[start_ship:end_ship] = pred_Q
            self._target_Q_ship[start_ship:end_ship] = target_Q
            
            ### ship priorities ###
            self._tdiffs_ship[start_ship+1 : end_ship] = (
                self._target_Q_ship[start_ship + 1 : end_ship] - 
                self._target_Q_ship[start_ship : end_ship-1] ).abs()
            
            if start_ship==0:
                self._tdiffs_ship[0] = self._tdiffs_ship.mean().item()
            else:
                self._tdiffs_ship[0] = abs(self._tdiffs_ship[start_ship] - self._tdiffs_ship[start_ship-1])
            
            pi = (self._tdiffs_ship[0: self._tdiffs_ship.shape[0] if self._ship_buffer_filled else end_ship]**0.9) # alpha=0.9
            self._target_Q_ship_priorities = pi / pi.sum()
            self._ship_model_samples += count
        else:
            if self._shipyard_model_samples + count > MAX_EPISODES_MEMORY:
                self._shipyard_buffer_filled = True
                self._shipyard_model_samples = 0
                
            start_shipyard = self._shipyard_model_samples
            end_shipyard = self._shipyard_model_samples + count
            try:
                self._geo_shipyard_ftrs[start_shipyard:end_shipyard] = geo_ftrs
            except Exception as e:
                raise e
            self._Q_shipyard[start_shipyard:end_shipyard] = pred_Q
            self._target_Q_shipyard[start_shipyard:end_shipyard] = target_Q
            
            ### shipyard priorities ###
            self._tdiffs_shipyard[start_shipyard+1 : end_shipyard] = (
                self._target_Q_shipyard[start_shipyard + 1 : end_shipyard] - 
                self._target_Q_shipyard[start_shipyard : end_shipyard-1] ).abs()
            
            if start_shipyard==0:
                self._tdiffs_shipyard[0] = self._tdiffs_shipyard.mean().item()
            else:
                self._tdiffs_shipyard[0] = abs(self._tdiffs_shipyard[start_shipyard] - self._tdiffs_shipyard[start_shipyard-1]) 
            
            pi = (self._tdiffs_shipyard[0 : self._tdiffs_shipyard.shape[0] if self._shipyard_buffer_filled else end_shipyard]**0.9) # alpha=0.9
            self._target_Q_shipyard_priorities = pi / pi.sum()
            self._shipyard_model_samples += count
            
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
    def __init__(self, 
         conv_layers, 
         fc_layers, 
         fc_volume, 
         filters, 
         kernel, 
         stride, 
         pad, 
         ts_ftrs,
         out_neurons):
        
        super(DQN, self).__init__()
        
        self._conv_layers = []
        self._conv_act = []
        self.trained_examples = 0
        height = DQN._compute_output_dim(BOARD_SIZE, kernel, stride, pad)
        for i in range(conv_layers):
            channels_in = CHANNELS if i==0 else filters * (2**(i-1))
            channels_out = filters * (2**i)
            layer = nn.Conv2d(
                channels_in,   # number of in channels (depth of input)
                channels_out,    # out channels (depth, or number of filters)
                kernel,     # size of convolving kernel
                stride,     # stride of kernel
                pad)        # padding
            nn.init.xavier_uniform_(layer.weight)
            activation = nn.ReLU() if i!=0 else nn.ReLU()
            
            self._conv_layers.append(layer)
            self._conv_act.append(activation)
            # necessary to register layer
            setattr(self, "_conv{0}".format(i), layer)
            setattr(self, "_conv_act{0}".format(i), activation)
            if i!=0:
                height = DQN._compute_output_dim(height, kernel, stride, pad)
        
        self._fc_layers = []
        self._fc_act = []
        for i in range(fc_layers):
            layer = nn.Linear(
                (height * height * channels_out + ts_ftrs) if i==0 else fc_volume, # number of neurons from previous layer
                fc_volume # number of neurons in output layer
                )
            nn.init.xavier_uniform_(layer.weight)
            act = nn.ReLU()
            self._fc_layers.append(layer)
            self._fc_act.append(act)
            
            # necessary to register layer
            setattr(self, "_fc{0}".format(i), layer)
            setattr(self, "_fc_act{0}".format(i), act)
            
        self._final_layer = nn.Linear(
                fc_volume,
                out_neurons)
        nn.init.xavier_uniform_(self._final_layer.weight)
        
    def forward(self, geometric_x, ts_x):
        try:
            y = self._conv_layers[0](geometric_x)
        except Exception as e:
            raise e
        y = self._conv_act[0](y)
        for layer, activation in zip(self._conv_layers[1:], self._conv_act[1:]):            
            y = layer(y)
            y = activation(y)
        
        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = torch.cat((y, ts_x[:,0:1]), dim=1) #@UndefinedVariable
        for layer, activation in zip(self._fc_layers, self._fc_act):
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

def randomize_action(board):
    current_halite = board.current_player.halite
    ships_converted = 0
    for my_ship in board.current_player.ships:
        if current_halite > board.configuration.convert_cost:
            my_ship.next_action = choice(SHIP_ACTIONS)
            if my_ship.next_action==ShipAction.CONVERT:
                current_halite -= board.configuration.convert_cost
                ships_converted += 1
        else:
            my_ship.next_action = choice(SHIP_MOVE_ACTIONS)
        
    for my_shipyard in board.current_player.shipyards:
        if current_halite > board.configuration.spawn_cost:
            my_shipyard.next_action = choice(SHIPYARD_ACTIONS)
            current_halite -= board.configuration.spawn_cost
        else:
            my_shipyard.next_action = None
    
    return ships_converted


def set_next_actions(board, ship_actions, shipyard_actions):
    cp = board.current_player
    ships_converted = 0
    for i, my_ship in enumerate(cp.ships):
        if ship_actions[i]==0:
            my_ship.next_action = None
        else:
            my_ship.next_action = ShipAction(ship_actions[i]) 
            if my_ship.next_action == ShipAction.CONVERT:
                ships_converted += 1

    for i, my_shipyard in enumerate(cp.shipyards):
        my_shipyard.next_action = None if shipyard_actions[i]==0 else ShipyardAction(shipyard_actions[i]) 
    
    return ships_converted

def step_forward(board, ship_actions, shipyard_actions):
    set_next_actions(board, ship_actions, shipyard_actions)
    new_board = board.next()
    return new_board

spatial = torch.tensor((
    [[(i,j) for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]), 
    dtype=torch.float).reshape(-1,2)
from scipy.spatial import distance
fleet_heat = np.full((BOARD_SIZE, BOARD_SIZE), BOARD_SIZE*2, dtype=np.float)
player_zeros = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
def update_tensors_v2(
    geometric_ship_ftrs, 
    geometric_shipyard_ftrs, 
    ts_ftrs, 
    board, 
    current_ship_cargo, 
    prior_ship_cargo):
    
    global fleet_heat
    geometric_ship_ftrs.fill_(0)
    geometric_shipyard_ftrs.fill_(0)
    ts_ftrs.fill_(0)
    current_ship_cargo.fill_(0)
    fleet_heat.fill((BOARD_SIZE*2)**2)
    cp = board.current_player
    halite = board.observation["halite"]
    remaining = float(board.configuration.episode_steps - board.step) / board.configuration.episode_steps
    ts_ftrs[:, 0] = remaining
    halite_tensor = torch.as_tensor(
            [halite[i:i+BOARD_SIZE] for i in 
             range(0, len(halite), BOARD_SIZE)], 
            dtype=torch.float) 
    if len(cp.ships)>0:        
        geometric_ship_ftrs[:, 0] = halite_tensor
        geometric_shipyard_ftrs[:, 0] = halite_tensor
    
        for i, my_ship in enumerate(cp.ships):
            heat = np.square(distance.cdist(
                spatial, [(my_ship.position.x, my_ship.position.y)], 
                metric="cityblock").reshape(BOARD_SIZE, BOARD_SIZE))
            
            heat[my_ship.position.y, my_ship.position.x] = .1
            
            fleet_heat = np.minimum(fleet_heat, heat, out=fleet_heat)
            
            np.divide(0.5, heat, out=heat)
            
            flipped = torch.flip(torch.as_tensor(heat, dtype=torch.float), dims=(0,))
            torch.mul(geometric_ship_ftrs[i, 0], flipped, out=geometric_ship_ftrs[i, 0])
            current_ship_cargo[0] += my_ship.halite
    
    if len(cp.shipyards)>0:
        np.divide(1, fleet_heat, out=fleet_heat)
        for i, my_shipyard in enumerate(cp.shipyards):
            heat = distance.cdist(
                spatial, [(my_shipyard.position.x, my_shipyard.position.y)], 
                metric="cityblock").reshape(BOARD_SIZE, BOARD_SIZE)
                        
            heat[my_shipyard.position.y, my_shipyard.position.x] = 0.25
            
            np.minimum(heat, fleet_heat, out=heat)
            
            np.divide(1, heat, out=heat)
            
            flipped = torch.flip(torch.as_tensor(heat, dtype=torch.float), dims=(0,))
            torch.mul(remaining, flipped, out=flipped)
            torch.mul(halite_tensor, flipped, out=geometric_shipyard_ftrs[i, 0])
            
            geometric_ship_ftrs[:, 0, BOARD_SIZE - my_shipyard.position.y - 1, my_shipyard.position.x] = max(100, current_ship_cargo[0])
                    
    ts_ftrs[:, 1] = cp.halite
    for i, enemy in enumerate(board.opponents):
        ts_ftrs[:, 2 + i] = enemy.halite
        
    ts_ftrs[:, -PLAYERS:] = torch.max(player_zeros, current_ship_cargo - prior_ship_cargo) #@UndefinedVariable
    
    return

class ActionSelector:
    def __init__(self, 
            agent_manager,
            agent_ts_ftrs, 
            ship_model, 
            shipyard_model, 
            ship_model_criterion,
            shipyard_model_criterion,
            ship_model_optimizer,
            shipyard_model_optimizer):
        
        self._agent_manager = agent_manager
        self._ship_model = ship_model
        self._shipyard_model = shipyard_model
        self._ship_model_criterion = ship_model_criterion
        self._shipyard_model_criterion = shipyard_model_criterion
        self._ship_model_optimizer = ship_model_optimizer
        self._shipyard_model_optimizer = shipyard_model_optimizer
        self._reward_engine = RewardEngine()
        self._agent_ts_ftrs = agent_ts_ftrs
        
        self._prior_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
        self._current_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self._best_ship_actions = np.zeros(MAX_SHIPS, dtype=np.int32)
        self._best_shipyard_actions = np.zeros(MAX_SHIPS, dtype=np.int32)
        
        self._geometric_ship_ftrs_v2 = torch.zeros(#@UndefinedVariable
            (2, MAX_SHIPS, CHANNELS, BOARD_SIZE, BOARD_SIZE), 
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._geometric_shipyard_ftrs_v2 = torch.zeros(#@UndefinedVariable
            (2, MAX_SHIPS, CHANNELS, BOARD_SIZE, BOARD_SIZE), 
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
            
        self._ts_ftrs_v2 = torch.zeros(#@UndefinedVariable
            (2, MAX_SHIPS, TS_FTR_COUNT), 
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._Q_ships_current_adj = torch.zeros(#@UndefinedVariable
            (MAX_SHIPS, len(SHIP_ACTIONS)),
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._Q_shipyards_current_adj = torch.zeros(#@UndefinedVariable
            (MAX_SHIPS, len(SHIPYARD_ACTIONS)),
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._Q_ships_next = torch.zeros(#@UndefinedVariable
            MAX_SHIPS,
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._Q_shipyards_next = torch.zeros(#@UndefinedVariable
            MAX_SHIPS,
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._ship_rewards = torch.zeros(#@UndefinedVariable
            MAX_SHIPS, dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._shipyard_rewards = torch.zeros(#@UndefinedVariable
            MAX_SHIPS, dtype=torch.float).to(DEVICE)#@UndefinedVariable
            
        self._non_terminal_ships = torch.zeros(#@UndefinedVariable
            MAX_SHIPS, dtype=torch.bool).to(DEVICE)#@UndefinedVariable
        
        self._non_terminal_shipyards = torch.zeros(#@UndefinedVariable
            MAX_SHIPS, dtype=torch.bool).to(DEVICE)#@UndefinedVariable
            
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self._prior_ship_cargo[:] = prior_ship_cargo
    
    def _ship_iterator_model(
            self, 
            board, 
            ship_count, 
            shipyard_count, 
            current_halite):
        
        self._ship_model.eval()
        self._shipyard_model.eval()
        self._best_ship_actions.fill(0)
        self._best_shipyard_actions.fill(0)
        self._Q_ships_current_adj.fill_(0)
        self._Q_shipyards_current_adj.fill_(0)
        self._non_terminal_ships.fill_(1)
        self._non_terminal_shipyards.fill_(1)
        
        update_tensors_v2(
            self._geometric_ship_ftrs_v2[0], 
            self._geometric_shipyard_ftrs_v2[0], 
            self._ts_ftrs_v2[0], 
            board, 
            self._current_ship_cargo, 
            self._prior_ship_cargo)
        
        max_new_shipyards_allowed = int(current_halite * .2) // board.configuration.convert_cost
        max_new_ships_allowed = int(current_halite * .8) // board.configuration.spawn_cost
        
        with torch.no_grad():
            if ship_count > 0:
                Q_ships_current = self._ship_model(
                    self._geometric_ship_ftrs_v2[0, :ship_count], 
                    self._ts_ftrs_v2[0, :ship_count])
            if shipyard_count > 0:
                Q_shipyards_current = self._shipyard_model(
                    self._geometric_shipyard_ftrs_v2[0, :shipyard_count], 
                    self._ts_ftrs_v2[0, :shipyard_count])
        
        if ship_count > 0:
            self._Q_ships_current_adj[:ship_count] = Q_ships_current
            retmax = Q_ships_current.max(dim=1)
            Q_ships_current_max = retmax.values
            conversion_indices = np.where(retmax.indices==5)[0][max_new_shipyards_allowed:]
            self._Q_ships_current_adj[conversion_indices, 5] = float('-inf')
            self._best_ship_actions[:ship_count] = self._Q_ships_current_adj[:ship_count].argmax(dim=1)
        if shipyard_count > 0:
            retmax = Q_shipyards_current.max(dim=1)
            Q_shipyards_current_max = retmax.values
            shipyard_actions_f = retmax.indices.type(dtype=torch.float)
            torch.mul(shipyard_actions_f, retmax.values, out=shipyard_actions_f)
            shipyard_actions_f[shipyard_actions_f==0] = float('-inf')
            self._best_shipyard_actions[shipyard_actions_f.argsort(descending=True) < max_new_ships_allowed] = 1
        self._prior_ship_cargo.copy_(self._current_ship_cargo)
        
        new_board = step_forward(
            board, 
            self._best_ship_actions, 
            self._best_shipyard_actions)
                
        update_tensors_v2(
            self._geometric_ship_ftrs_v2[1], 
            self._geometric_shipyard_ftrs_v2[1], 
            self._ts_ftrs_v2[1], 
            new_board, 
            self._current_ship_cargo, 
            self._prior_ship_cargo)
        
        self._reward_engine.compute_ship_rewards(
            new_board, 
            board, 
            self._ship_rewards, 
            self._non_terminal_ships)
        
        self._reward_engine.compute_shipyard_rewards(
            new_board, 
            board, 
            self._shipyard_rewards, 
            self._non_terminal_shipyards)
        
        self._Q_ships_next[:ship_count].add_(self._ship_rewards[:ship_count])
        self._Q_shipyards_next[:shipyard_count].add_(self._shipyard_rewards[:shipyard_count])
        
        new_ship_count = len(new_board.current_player.ships)
        new_shipyard_count = len(new_board.current_player.shipyards)
        with torch.no_grad():
            if new_ship_count > 0:
                Q_ships_next = self._ship_model(
                    self._geometric_ship_ftrs_v2[1, :new_ship_count], 
                    self._ts_ftrs_v2[1, :new_ship_count])
                
                self._Q_ships_next[self._non_terminal_ships][:ship_count].add_(Q_ships_next.max(dim=1).values)
                
            if new_shipyard_count > 0:
                Q_shipyards_next = self._shipyard_model(
                    self._geometric_shipyard_ftrs_v2[1, :new_shipyard_count], 
                    self._ts_ftrs_v2[1, :new_shipyard_count])
                
                self._Q_shipyards_next[self._non_terminal_shipyards][:shipyard_count].add_(Q_shipyards_next.max(dim=1).values)
                    
        if ship_count > 0:
            self._agent_manager.store(
                ship_count,
                self._geometric_ship_ftrs_v2[0, :ship_count], 
                Q_ships_current_max,
                self._Q_ships_next[:ship_count],
                True)
        
        if shipyard_count > 0:
            self._agent_manager.store(
                shipyard_count,
                self._geometric_shipyard_ftrs_v2[0, :shipyard_count], 
                Q_shipyards_current_max,
                self._Q_shipyards_next[:shipyard_count],
                False)
            
        if self._agent_manager.total_episodes_seen + self._agent_manager.in_game_episodes_seen > TRAIN_BATCH_SIZE:
            geo_ship_ftrs, ts_ftrs_ship, ship_targets = \
                self._agent_manager.priority_sample(TRAIN_BATCH_SIZE, True)
            
            train_model(
                self._ship_model, 
                "ship_dqn",
                self._ship_model_criterion, 
                self._ship_model_optimizer, 
                geo_ship_ftrs, 
                ts_ftrs_ship, 
                ship_targets,
                board.step)
                    
            geo_shipyards_ftrs, ts_ftrs_shipyard, shipyards_targets = \
                self._agent_manager.priority_sample(TRAIN_BATCH_SIZE, False)
            
            train_model(
                self._shipyard_model, 
                "shipyard_dqn",
                self._shipyard_model_criterion,
                self._shipyard_model_optimizer,
                geo_shipyards_ftrs, 
                ts_ftrs_shipyard, 
                shipyards_targets,
                board.step)
                    
    def get_action_space(self, board):
        ship_count = len(board.current_player.ships)
        shipyard_count = len(board.current_player.shipyards)
        action_space = (len(SHIP_ACTIONS)**ship_count) * (len(SHIPYARD_ACTIONS)**shipyard_count)
        return action_space
    
    def select_action(self, board):
        ship_count = len(board.current_player.ships)
        shipyard_count = len(board.current_player.shipyards)
        current_halite = board.current_player.halite
        
        self._ship_iterator_model(
            board, 
            ship_count, 
            shipyard_count, 
            current_halite)            
            
        ships_converted = set_next_actions(board, self._best_ship_actions, self._best_shipyard_actions)
        return ships_converted

def reward_ship(board, new_board):
    pass
    
def train_model(model, model_name, criterion, optimizer, geo_ftrs, ts_ftrs, labels, step):
    model.train()
    for e in range(EPOCHS):
        y_pred = model(geo_ftrs, ts_ftrs)
        loss = criterion(y_pred.max(dim=1).values, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {}, Board step {}, Train batch loss: {}".format(e, step, loss.item()))
    torch.save(model.state_dict(), "{0}/{1}_{0}.nn".format(TIMESTAMP, model_name))


def train(model, criterion, agent_managers):
    model.train()
    for e in range(EPOCHS):
        for asm in agent_managers.values():
            if asm.total_episodes_seen > 0:
                idxs = asm.generate_priority_samples()
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
            if e==0:
                model.trained_examples += len(idxs) 
#     for layer in model._conv_layers:
#         print(layer.weight.mean())   
#     for layer in model._fc_layers:
#         print(layer.weight.mean())  
#     print(model._final_layer.weight.mean())
    torch.save(model.state_dict(), "{0}/dqn_{0}.nn".format(TIMESTAMP))

class RewardEngine:
    @classmethod
    def init_weights(cls):
        cls.mined_reward_weights = torch.tensor([1] + [-.25]*(PLAYERS-1), dtype=torch.float).to(DEVICE) #@UndefinedVariable
        cls.deposited_reward_weights = torch.tensor([1] + [-.1]*(PLAYERS-1), dtype=torch.float).to(DEVICE) #@UndefinedVariable
        cls.reward_step = 1 / EPISODE_STEPS
        
    def __init__(self):
        self._prior_ships_dict = {}
        self._prior_ships_set = set()
        self._prior_shipyards_dict = {}
        self._prior_shipyards_set = set()
        
    def update_player_ships(self, board):
        self._ships = {ship.id : ship for ship in board.current_player.ships}
        
    def compute_ship_rewards(self, current_board, prior_board, ship_rewards, non_terminal_ships):
        current_player = current_board.current_player
        prior_player = prior_board.current_player
        if current_board.step==1:
            self._prior_ships_set = set([ship.id for ship in prior_player.ships])
            self._prior_ships_dict = {sid: prior_board.ships[sid] for sid in self._prior_ships_set}
        
        
        my_ship_ids = {ship.id:i for i, ship in enumerate(prior_player.ships)}
        
        current_ships_set = set([ship.id for ship in current_player.ships])
        current_halite_cargo = {ship.id: ship.halite for ship in current_player.ships}
        ships_converted = set([ship.id for ship in prior_player.ships if ship.next_action==ShipAction.CONVERT])
        ships_lost_from_collision = self._prior_ships_set.difference(ships_converted).difference(current_ships_set)
        
        retained_ships = current_ships_set.intersection(self._prior_ships_set)
        deposit_weight = (.5 + current_board.step*self.reward_step)
        
        non_terminal_ships[[my_ship_ids[sid] for sid in ships_converted.union(ships_lost_from_collision)]] = 0
        
        rewards = {ship.id: 
            max(0, current_halite_cargo[ship.id] - ship.halite) + # diff halite cargo
            (ship.halite - current_halite_cargo[ship.id])*deposit_weight + # diff halite deposited
            ship.halite==current_halite_cargo[ship.id]*-25 # inactivity
            if ship.id in retained_ships else 0 
            for ship in self._prior_ships_dict.values()}
        rewards.update({sid: -500*deposit_weight for sid in ships_converted})
        rewards.update({sid: -750 for sid in ships_lost_from_collision})
        
        ship_rewards[:len(self._prior_ships_set)] = torch.tensor(list(rewards.values()), dtype=torch.float)     
        ship_rewards[:len(self._prior_ships_set)].mul_(GAMMA)
        self._prior_ships_set = current_ships_set
        self._prior_ships_dict = {sid: current_board.ships[sid] for sid in current_ships_set}
    
    def compute_shipyard_rewards(self, current_board, prior_board, shipyard_rewards, non_terminal_shipyards):
        current_player = current_board.current_player
        prior_player = prior_board.current_player
        
        my_shipyard_ids = {shipyard.id:i for i, shipyard in enumerate(prior_player.shipyards)}
        current_shipyards_set = set([shipyard.id for shipyard in current_player.shipyards])
        shipyards_lost_from_collision = self._prior_shipyards_set.difference(current_shipyards_set)
        non_terminal_shipyards[[my_shipyard_ids[sid] for sid in shipyards_lost_from_collision]] = 0
        
        rewards = {shipyard.id: 
           (shipyard.next_action==ShipyardAction.SPAWN)*500 +
           (shipyard.next_action==None)*(-50*(1-current_board.step*self.reward_step))
           for shipyard in self._prior_shipyards_dict.values()}
        shipyard_rewards[:len(self._prior_shipyards_set)] = torch.tensor(list(rewards.values()), dtype=torch.float)
        shipyard_rewards[:len(self._prior_shipyards_set)].mul_(GAMMA)
        
        self._prior_shipyards_set = current_shipyards_set
        self._prior_shipyards_dict = {sid: current_board.shipyards[sid] for sid in current_shipyards_set}
        
        
RewardEngine.init_weights()
    
    
def agent(obs, config):
    global agent_managers
    current_board = Board(obs, config)
    asm = agent_managers.get(current_board.current_player.id)
    step = current_board.step        
    ftr_index = asm.total_episodes_seen + asm.in_game_episodes_seen
    print(ftr_index, file=sys.__stdout__)
    asm.set_prior_ship_cargo(asm.current_ship_cargo)
    
    epsilon = max(EGREEDY_LOWER_BOUND, EGREEDY*((1-EGREEDY_DECAY)**asm.ship_model.trained_examples))
    randomize = np.random.rand() < epsilon
    
#     if PRINT_STATEMENTS:
#         reward = compute_reward(
#             current_board, 
#             asm.prior_board, 
#             asm.current_ship_cargo,
#             asm._ts_ftrs[ftr_index, -PLAYERS: ],
#             asm._ts_ftrs[ftr_index, 1:  1 + PLAYERS], 
#             None if step==0 else asm._ts_ftrs[ftr_index-1, 1:  1 + PLAYERS],
#             asm.prior_ships_converted)
#         
#         print("board step:", 
#               step, 
#               ",realized reward:", 
#               reward, 
#               ",epsilon:",
#               epsilon,
#               "randomize:",
#               randomize,
#               "action_space",
#               asm.action_selector.get_action_space(current_board),
#               file=sys.__stdout__)
    
    if randomize:
        ships_converted = randomize_action(current_board)
    else:
        ships_converted = asm.action_selector.select_action(current_board)
    
    asm.prior_ships_converted = ships_converted
    asm.set_prior_board(current_board)
    asm.in_game_episodes_seen += 1
    
    return current_board.current_player.next_actions

def output_logs(env, steps, agent_managers, game_id):
    if hasattr(env, "logs") and env.logs is not None:
        with open("{0}/log_{0}.txt".format(TIMESTAMP), "w") as f:
            f.write('\n'.join([str(t) for t in env.logs]))
    with open("{0}/steps_{0}.txt".format(TIMESTAMP), "w") as f:
        f.write('\n'.join([str(l) for l in steps]))
    
    append_p = ""
#     for asm in agent_managers.values():
#         append_p += "p{}".format(asm.player_id)
#         with open("{0}/p{1}_rewards_{0}.txt".format(TIMESTAMP, asm.player_id), "w") as f:
#             f.write(str(asm.episode_rewards[asm.total_episodes_seen:
#                                         asm.total_episodes_seen+asm.in_game_episodes_seen]))
#         
#         with open("{0}/p{1}_qvals_{0}.txt".format(TIMESTAMP, asm.player_id), "w") as f:
#             f.write(str(asm.q_values[asm.total_episodes_seen:
#                                      asm.total_episodes_seen+asm.in_game_episodes_seen]))
    out = env.render(mode="html", width=800, height=600)
    with open("{0}/{1}_{0}g{2}.html".format(TIMESTAMP, append_p, game_id), "w") as f:
        f.write(out)

if not os.path.exists(TIMESTAMP):
    os.makedirs(TIMESTAMP)
config = {
    "size": BOARD_SIZE, 
    "startingHalite": STARTING, 
    "episodeSteps": EPISODE_STEPS, 
    "actTimeout": 1e8, 
    "runTimeout":1e8}

if RANDOM_SEED > -1: 
    config["randomSeed"] = RANDOM_SEED

i = 1
agents = [agent]
ship_huber = nn.SmoothL1Loss()
shipyard_huber = nn.SmoothL1Loss()
ship_dqn = DQN(
    3,  # number of conv layers
    1,  # number of fully connected layers at end
    64, # number of neurons in fully connected layers at end
    8,  # number of start filters for conv layers (depth)
    3,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    1,  # number of extra time series features
    6   # out nerouns
).to(DEVICE)  
shipyard_dqn = DQN(
    3,  # number of conv layers
    1,  # number of fully connected layers at end
    64, # number of neurons in fully connected layers at end
    8,  # number of start filters for conv layers (depth)
    3,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    1,  # number of extra time series features
    2   # out nerouns
).to(DEVICE)  
ship_optimizer = torch.optim.SGD( #@UndefinedVariable
    ship_dqn.parameters(), 
    lr=LEARNING_RATE,
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)
shipyard_optimizer = torch.optim.SGD( #@UndefinedVariable
    shipyard_dqn.parameters(), 
    lr=LEARNING_RATE,
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)

if os.path.exists("{0}/a_dqn_{0}.nn".format(TIMESTAMP)):
    ship_dqn.load_state_dict(torch.load("{0}/a_dqn_{0}.nn".format(TIMESTAMP)))
if os.path.exists("{0}/b_dqn_{0}.nn".format(TIMESTAMP)):
    ship_dqn.load_state_dict(torch.load("{0}/b_dqn_{0}.nn".format(TIMESTAMP)))
    
print("ship dqn:", ship_dqn)
print("shipyard dqn:", shipyard_dqn)

agent_managers = {i: 
    AgentStateManager(
        i, 
        ship_dqn, 
        shipyard_dqn, 
        ship_huber, 
        shipyard_huber, 
        ship_optimizer, 
        shipyard_optimizer) for i in range(4)}

while i < GAME_COUNT:
    env = make("halite", configuration=config)
    print(env.configuration)
    env.reset(PLAYERS)
    np.random.shuffle(agents)
    active_agents = set([j for j, el in enumerate(agents) if el in (agent,)])
    print("starting game {0} with agent order: {1}".format(i, agents))
    active_agent_managers = {j:asm for j,asm in agent_managers.items() if j in active_agents}
    
    steps = env.run(agents)
    print("completed game {0}".format(i))
    
    for asm in active_agent_managers.values():
        print("agent {0} total reward: {1}".format(asm.player_id, asm.compute_total_reward_post_game()))
    
    if OUTPUT_LOGS:
        output_logs(env, steps, active_agent_managers, i)
        print("outputted data files")
        
    for asm in agent_managers.values():
        asm.game_id += 1
    i += 1