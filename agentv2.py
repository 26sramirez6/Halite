'''
Created on Aug 21, 2020

@author: 26sra
'''

import sys #@UnusedImport
import numpy as np 
import torch
import os
import math
import time
import functools
import torch.nn as nn
import torch.nn.functional as F #@UnusedImport
import datetime
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

EPISODE_STEPS = 400
MAX_EPISODES_MEMORY = 500
MAX_SHIPS = 100
STARTING = 5000
CONVERT_COST = 500
BOARD_SIZE = 21
PLAYERS = 4
GAMMA = 0.9
TRAIN_BATCH_SIZE = 48
TARGET_MODEL_UPDATE = 100
LEARNING_RATE = 0.01
SHIP_CHANNELS = 1
SHIPYARD_CHANNELS = 1
MOMENTUM  = 0.9
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
TS_FTR_COUNT = 1 + PLAYERS*2 
GAME_COUNT = 1000
TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.').replace('-',"_")
SERIALIZE = False
OUTPUT_LOGS = True
PRINT_STATEMENTS = False
TRAIN_MODELS = True
RANDOM_SEED = -1; 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #@UndefinedVariable
if RANDOM_SEED > -1: 
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class AgentStateManager:        
    def __init__(self, 
            player_id,        
            ship_cur_model, 
            shipyard_cur_model,
            ship_tar_model, 
            shipyard_tar_model, 
            ship_model_criterion, 
            shipyard_model_criterion, 
            ship_model_optimizer, 
            shipyard_model_optimizer):
        
        self.player_id = player_id
        self.game_id = 0
        self.prior_board = None
        self.prior_ships_converted = 0
        self._alpha = 0.2
        self.ship_cur_model = ship_cur_model
        self.shipyard_cur_model = shipyard_cur_model
        self.ship_tar_model = ship_tar_model
        self.shipyard_tar_model = shipyard_tar_model
        self.ship_model_criterion = ship_model_criterion
        self.shipyard_model_criterion = shipyard_model_criterion 
        self.ship_model_optimizer = ship_model_optimizer
        self.shipyard_model_optimizer = shipyard_model_optimizer
        
        self.ship_model_samples = 0
        self._ship_losses_stored = 0
        self.shipyard_model_samples = 0
        self._shipyard_losses_stored = 0
        self.total_episodes_seen = 0
        self.in_game_episodes_seen = 0
        self._ship_buffer_filled = False
        self._shipyard_buffer_filled = False
        
        self._geo_ship_ftrs_t0 = torch.zeros(
            (MAX_EPISODES_MEMORY, SHIP_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
        
        self._geo_shipyard_ftrs_t0 = torch.zeros(
            (MAX_EPISODES_MEMORY, SHIPYARD_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
        
        self._geo_ship_ftrs_t1 = torch.zeros(
            (MAX_EPISODES_MEMORY, SHIP_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
        
        self._geo_shipyard_ftrs_t1 = torch.zeros(
            (MAX_EPISODES_MEMORY, SHIPYARD_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float).to(DEVICE)
            
        self._ts_ship_ftrs_t0 = torch.zeros(
            (MAX_EPISODES_MEMORY, TS_FTR_COUNT),
            dtype=torch.float).to(DEVICE)
        
        self._ts_ship_ftrs_t1 = torch.zeros(
            (MAX_EPISODES_MEMORY, TS_FTR_COUNT),
            dtype=torch.float).to(DEVICE)
            
        self._ts_shipyard_ftrs_t0 = torch.zeros(
            (MAX_EPISODES_MEMORY, TS_FTR_COUNT),
            dtype=torch.float).to(DEVICE)
        
        self._ts_shipyard_ftrs_t1 = torch.zeros(
            (MAX_EPISODES_MEMORY, TS_FTR_COUNT),
            dtype=torch.float).to(DEVICE)
        
        self._ship_rewards = torch.zeros(
            MAX_EPISODES_MEMORY,
            dtype=torch.float).to(DEVICE)
        
        self._ship_actions = torch.LongTensor(
            MAX_EPISODES_MEMORY).to(DEVICE)
        self._ship_actions.fill_(0)
        self._shipyard_rewards = torch.zeros(
            MAX_EPISODES_MEMORY,
            dtype=torch.float).to(DEVICE)
        
        self._shipyard_actions = torch.LongTensor(
            MAX_EPISODES_MEMORY).to(DEVICE)
        self._shipyard_actions.fill_(0)
        self._ship_non_terminals = torch.ones(
            MAX_EPISODES_MEMORY,
            dtype=torch.float).to(DEVICE)
        
        self._shipyard_non_terminals = torch.ones(
            MAX_EPISODES_MEMORY,
            dtype=torch.float).to(DEVICE)
#         
#         self._tdiffs_ship = torch.zeros(
#             MAX_EPISODES_MEMORY,
#             dtype=torch.float64).to(DEVICE)
#         self._tdiffs_ship.fill_(1e-6)
#         
#         self._tdiffs_shipyard = torch.zeros(
#             MAX_EPISODES_MEMORY,
#             dtype=torch.float64).to(DEVICE)
#         self._tdiffs_shipyard.fill_(1e-6)
        self.current_ship_cargo = torch.zeros(
            PLAYERS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
            
        self.prior_ship_cargo = torch.zeros(
            PLAYERS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self.episode_rewards = torch.zeros(
            EPISODE_STEPS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self.ship_losses = torch.zeros(
            EPISODE_STEPS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
            
        self.shipyard_losses = torch.zeros(
            EPISODE_STEPS, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
            
        self.action_selector = ActionSelector(self)     
    
    def set_prior_board(self, prior_board):
        self.prior_board = prior_board
    
    def post_game_state_handle(self):
        self.action_selector.reset_state()
        self.total_episodes_seen += self.in_game_episodes_seen
        self.in_game_episodes_seen = 0
        self._ship_losses_stored = 0
        self._shipyard_losses_stored = 0
        if self.total_episodes_seen + EPISODE_STEPS > MAX_EPISODES_MEMORY:
            self.total_episodes_seen = 0
        torch.save(self.ship_cur_model.state_dict(), "{0}/{1}_{0}.nn".format(TIMESTAMP, "ship_dqn"))
        torch.save(self.shipyard_cur_model.state_dict(), "{0}/{1}_{0}.nn".format(TIMESTAMP, "shipyard_dqn"))
            
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self.prior_ship_cargo.copy_(prior_ship_cargo)
        self.action_selector.set_prior_ship_cargo(prior_ship_cargo)
    
    def compute_total_reward_post_game(self):
        return self.episode_rewards.sum().item()
    
    def priority_sample(self, batch_size, is_ship_model):
        if is_ship_model:
            ship_end = MAX_EPISODES_MEMORY if self._ship_buffer_filled else self.ship_model_samples
#             ship_idxs = list(range(ship_end - batch_size, ship_end))
#             try:
#                 
#                 ship_idxs = np.random.choice(
#                     list(range(ship_end)), 
#                     size=batch_size,
#                     replace=False, 
#                     p=self._target_Q_ship_priorities)
#             except Exception as e:
#                 print("error using priorities, using uniform", file=sys.__stdout__)
            ship_idxs = np.random.choice(
                list(range(ship_end)), 
                size=batch_size)
            
            return (self._geo_ship_ftrs_t0[ship_idxs], 
                    self._ts_ship_ftrs_t0[ship_idxs],
                    self._geo_ship_ftrs_t1[ship_idxs], 
                    self._ts_ship_ftrs_t1[ship_idxs],
                    self._ship_rewards[ship_idxs],
                    self._ship_actions[ship_idxs],
                    self._ship_non_terminals[ship_idxs])
        else:
            shipyard_end = MAX_EPISODES_MEMORY if self._shipyard_buffer_filled else self.shipyard_model_samples
#             shipyard_idxs = list(range(shipyard_end - batch_size, shipyard_end))
#             try:
#                 shipyard_idxs = np.random.choice(
#                     list(range(shipyard_end)), 
#                     size=batch_size,
#                     replace=False, 
#                     p=self._target_Q_shipyard_priorities) 
#             except Exception as e:
            shipyard_idxs = np.random.choice(
                list(range(shipyard_end)), 
                size=batch_size) 
#                 print("error using priorities, using uniform", file=sys.__stdout__)
            return (self._geo_shipyard_ftrs_t0[shipyard_idxs], 
                    self._ts_shipyard_ftrs_t0[shipyard_idxs],
                    self._geo_shipyard_ftrs_t1[shipyard_idxs], 
                    self._ts_shipyard_ftrs_t1[shipyard_idxs],
                    self._shipyard_rewards[shipyard_idxs],
                    self._shipyard_actions[shipyard_idxs],
                    self._shipyard_non_terminals[shipyard_idxs])       
    
    def train_current_models(self):
        if self.ship_model_samples > TRAIN_BATCH_SIZE:
            geo_ship_ftrs_t0, \
            ts_ftrs_ship_t0, \
            geo_ship_ftrs_t1, \
            ts_ftrs_ship_t1, \
            ship_rewards, \
            ship_actions, \
            non_terminals = self.priority_sample(TRAIN_BATCH_SIZE, True)
                    
            mini_batch_loss = AgentStateManager.train_model(
                self.ship_cur_model, 
                self.ship_tar_model,
                self.ship_model_criterion, 
                self.ship_model_optimizer, 
                geo_ship_ftrs_t0,
                ts_ftrs_ship_t0,
                geo_ship_ftrs_t1,
                ts_ftrs_ship_t1,
                ship_rewards,
                ship_actions,
                non_terminals)
            
            self.save_loss(mini_batch_loss, True)
            
        if self.shipyard_model_samples > TRAIN_BATCH_SIZE:
            geo_shipyard_ftrs_t0, \
            ts_ftrs_shipyard_t0, \
            geo_shipyard_ftrs_t1, \
            ts_ftrs_shipyard_t1, \
            shipyard_rewards, \
            shipyard_actions, \
            non_terminals = self.priority_sample(TRAIN_BATCH_SIZE, False)
        
            mini_batch_loss = AgentStateManager.train_model(
                self.shipyard_cur_model, 
                self.shipyard_tar_model,
                self.shipyard_model_criterion, 
                self.shipyard_model_optimizer, 
                geo_shipyard_ftrs_t0,
                ts_ftrs_shipyard_t0,
                geo_shipyard_ftrs_t1,
                ts_ftrs_shipyard_t1,
                shipyard_rewards,
                shipyard_actions,
                non_terminals)
            
            self.save_loss(mini_batch_loss, False)
            
        self.ship_cur_model.reset_noise()
        self.shipyard_cur_model.reset_noise()
        
        if self.ship_model_samples > 0 and self.ship_model_samples % TARGET_MODEL_UPDATE == 0:
            self.update_target_model(self.ship_cur_model, self.ship_tar_model)
        
        if self.shipyard_model_samples > 0 and self.shipyard_model_samples % TARGET_MODEL_UPDATE == 0:
            self.update_target_model(self.shipyard_cur_model, self.shipyard_tar_model)
        
    def update_target_model(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    @staticmethod
    def train_model(
        current_model, 
        target_model,
        criterion, 
        optimizer, 
        geo_ftrs_t0,
        ts_ftrs_t0,
        geo_ftrs_t1,
        ts_ftrs_t1,
        rewards,
        actions,
        non_terminals):
        
        current_model.train()
        optimizer.zero_grad()
        Q_current = current_model(geo_ftrs_t0, ts_ftrs_t0).gather(1, actions.unsqueeze(1))
        
        if non_terminals.sum() > 0:
            Q_next_state_cur_model = current_model(
                geo_ftrs_t1, 
                ts_ftrs_t1).detach().mul_(GAMMA)                
            
            Q_next_state_tar_model = target_model(
                geo_ftrs_t1, 
                ts_ftrs_t1).detach().mul_(GAMMA)
            
            Q_next_state_targets = Q_next_state_tar_model.gather(1, Q_next_state_cur_model.max(dim=1).indices.unsqueeze(1))
        else:
            Q_next_state_targets = torch.zeros(rewards.shape[0], dtype=torch.float).unsqueeze(1).detach()
            
        Q_next_state_targets.add_(rewards.unsqueeze(1))
                       
        loss = criterion(Q_current, Q_next_state_targets)
        loss.backward()
        optimizer.step()
        return loss.item()
        
    def save_loss(self, loss, is_ship_model):
        if is_ship_model:
            self.ship_losses[self._ship_losses_stored] = loss
            self._ship_losses_stored += 1
        else:
            self.shipyard_losses[self._shipyard_losses_stored] = loss
            self._shipyard_losses_stored += 1
        
    def store(self,
        count,
        geo_ftrs_t0, 
        ts_ftrs_t0,
        actions,
        rewards,
        geo_ftrs_t1, 
        ts_ftrs_t1,
        non_terminals,
        is_ship_model) :   
             
        if is_ship_model:        
            if self.ship_model_samples + count > MAX_EPISODES_MEMORY:
                self._ship_buffer_filled = True
                self.ship_model_samples = 0
            
            start_ship = self.ship_model_samples
            end_ship = self.ship_model_samples + count
            
            self._geo_ship_ftrs_t0[start_ship:end_ship] = geo_ftrs_t0
            self._geo_ship_ftrs_t1[start_ship:end_ship] = geo_ftrs_t1
            self._ts_ship_ftrs_t0[start_ship:end_ship] = ts_ftrs_t0
            self._ts_ship_ftrs_t1[start_ship:end_ship] = ts_ftrs_t1
            self._ship_actions[start_ship:end_ship] = actions
            self._ship_rewards[start_ship:end_ship] = rewards
            self._ship_non_terminals[start_ship:end_ship] = non_terminals
#             self._Q_ship[start_ship:end_ship] = pred_Q
#             self._target_Q_ship[start_ship:end_ship] = target_Q
#             
#             ### ship priorities ###
#             self._tdiffs_ship[start_ship+1 : end_ship] = (
#                 self._target_Q_ship[start_ship + 1 : end_ship] - 
#                 self._target_Q_ship[start_ship : end_ship-1] ).abs()
#             
#             if start_ship==0:
#                 self._tdiffs_ship[0] = self._tdiffs_ship.mean().item()
#             else:
#                 self._tdiffs_ship[start_ship] = abs(self._target_Q_ship[start_ship] - self._target_Q_ship[start_ship-1])
#             
#             pi = (self._tdiffs_ship[0: self._tdiffs_ship.shape[0] if self._ship_buffer_filled else end_ship]**self._alpha)
#             self._target_Q_ship_priorities = pi / pi.sum()
            self.ship_model_samples += count
        else:
            if self.shipyard_model_samples + count > MAX_EPISODES_MEMORY:
                self._shipyard_buffer_filled = True
                self.shipyard_model_samples = 0
                
            start_shipyard = self.shipyard_model_samples
            end_shipyard = self.shipyard_model_samples + count
            
            self._geo_shipyard_ftrs_t0[start_shipyard:end_shipyard] = geo_ftrs_t0
            self._geo_shipyard_ftrs_t1[start_shipyard:end_shipyard] = geo_ftrs_t1
            self._ts_shipyard_ftrs_t0[start_shipyard:end_shipyard] = ts_ftrs_t0
            self._ts_shipyard_ftrs_t1[start_shipyard:end_shipyard] = ts_ftrs_t1
            self._shipyard_actions[start_shipyard:end_shipyard] = actions
            self._shipyard_rewards[start_shipyard:end_shipyard] = rewards
            self._shipyard_non_terminals[start_shipyard:end_shipyard] = non_terminals
#             self._Q_shipyard[start_shipyard:end_shipyard] = pred_Q
#             self._target_Q_shipyard[start_shipyard:end_shipyard] = target_Q
            
            ### shipyard priorities ###
#             self._tdiffs_shipyard[start_shipyard+1 : end_shipyard] = (
#                 self._target_Q_shipyard[start_shipyard + 1 : end_shipyard] - 
#                 self._target_Q_shipyard[start_shipyard : end_shipyard-1] ).abs()
#             
#             if start_shipyard==0:
#                 self._tdiffs_shipyard[0] = self._tdiffs_shipyard.mean().item()
#             else:
#                 self._tdiffs_shipyard[start_shipyard] = abs(self._target_Q_shipyard[start_shipyard] - self._target_Q_shipyard[start_shipyard-1]) 
#             
#             pi = (self._tdiffs_shipyard[0 : self._tdiffs_shipyard.shape[0] if self._shipyard_buffer_filled else end_shipyard]**self._alpha)
#             self._target_Q_shipyard_priorities = pi / pi.sum()
            self.shipyard_model_samples += count
            
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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(torch.autograd.Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(torch.autograd.Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    
class DQN(nn.Module):
    def __init__(self, 
         channels,
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
        self._conv_bn = []
        self.trained_examples = 0
        height = DQN._compute_output_dim(BOARD_SIZE, kernel, stride, pad)
        for i in range(conv_layers):
            channels_in = channels if i==0 else filters * (2**(i-1))
            channels_out = filters * (2**i)
            layer = nn.Conv2d(
                channels_in,   # number of in channels (depth of input)
                channels_out,    # out channels (depth, or number of filters)
                kernel,     # size of convolving kernel
                stride,     # stride of kernel
                pad)        # padding
            nn.init.xavier_uniform_(layer.weight)
            bn = nn.BatchNorm2d(channels_out)
            activation = nn.ReLU() if i!=0 else nn.ReLU()
            
            self._conv_layers.append(layer)
            self._conv_bn.append(bn)
            self._conv_act.append(activation)
            # necessary to register layer
            setattr(self, "_conv{0}".format(i), layer)
            setattr(self, "_conv_bn{0}".format(i), bn)
            setattr(self, "_conv_act{0}".format(i), activation)
            if i!=0:
                height = DQN._compute_output_dim(height, kernel, stride, pad)
        
        self._fc_layers = []
        self._fc_act = []
        for i in range(fc_layers):
            layer = NoisyLinear(
                (height * height * channels_out + ts_ftrs) if i==0 else fc_volume, # number of neurons from previous layer
                fc_volume # number of neurons in output layer
                )
#             nn.init.xavier_uniform_(layer.weight)
            act = nn.ReLU()
            self._fc_layers.append(layer)
            self._fc_act.append(act)
            
            # necessary to register layer
            setattr(self, "_fc{0}".format(i), layer)
            setattr(self, "_fc_act{0}".format(i), act)
            
        self._final_layer = NoisyLinear(
                fc_volume,
                out_neurons)
#         nn.init.xavier_uniform_(self._final_layer.weight)
    
    def reset_noise(self):
        for noisy_layer in self._fc_layers:
            noisy_layer.reset_noise()
        self._final_layer.reset_noise()
            
    def forward(self, geometric_x, ts_x):
        y = self._conv_layers[0](geometric_x)
        y = self._conv_bn[0](y)
        y = self._conv_act[0](y)
        for layer, bn, activation in zip(self._conv_layers[1:], self._conv_bn[1:], self._conv_act[1:]):       
            y = layer(y)
            y = bn(y)
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


class FeatureCalculator:
    @classmethod
    def init(cls):
        cls.spatial = torch.tensor((
            [[(i,j) for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]), 
            dtype=torch.float).reshape(-1,2)
        
        cls.fleet_heat = np.full((BOARD_SIZE, BOARD_SIZE), BOARD_SIZE*2, dtype=np.float)
        cls.player_zeros = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
        cls.middle_heat = distance.cdist(
            cls.spatial, [(BOARD_SIZE//2, BOARD_SIZE//2)], 
            metric="cityblock").reshape(BOARD_SIZE, BOARD_SIZE)
        cls.middle_heat[BOARD_SIZE//2, BOARD_SIZE//2] = .75
        np.divide(1, cls.middle_heat, out=cls.middle_heat)
        cls.middle_heat_flipped = torch.flip(torch.as_tensor(cls.middle_heat, dtype=torch.float), dims=(0,))
        cls.ship_positions = torch.tensor((MAX_SHIPS,2), dtype=torch.int)
        cls.selector = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        np.fill_diagonal(cls.selector, False)
        
    @classmethod
    def update_tensors_v2(
        cls,
        geometric_ship_ftrs, 
        geometric_shipyard_ftrs, 
        ts_ftrs, 
        board, 
        current_ship_cargo, 
        prior_ship_cargo):
        
        geometric_ship_ftrs.zero_()
        geometric_shipyard_ftrs.zero_()
        ts_ftrs.zero_()
        current_ship_cargo.zero_()
        cls.fleet_heat.fill(BOARD_SIZE*2)
        cp = board.current_player
        halite = board.observation["halite"]
        remaining = float(board.configuration.episode_steps - board.step) / board.configuration.episode_steps
        ts_ftrs[:, 0] = remaining
        halite_tensor = torch.as_tensor(
                [halite[i:i+BOARD_SIZE] for i in 
                 range(0, len(halite), BOARD_SIZE)], 
                dtype=torch.float) 
        
        current_ship_cargo[0] = sum([ship.halite for ship in cp.ships])                    
        max_halite_cell = halite_tensor.max()
        min_halite_cell = halite_tensor.min()
        diff = max_halite_cell - min_halite_cell
        halite_tensor.sub_(min_halite_cell)
        halite_tensor.div_(diff) 
        halite_tensor.mul_(2)
        halite_tensor.sub_(1)
        
        ship_count = len(cp.ships)
        shipyard_count = len(cp.shipyards)
        
        if ship_count>0:        
            geometric_ship_ftrs[:, 0] = halite_tensor
            ship_positions_yx = np.array([(my_ship.position.y, my_ship.position.x) for my_ship in cp.ships])
        
        if shipyard_count>0 and ship_count>0:
            if ship_count>1:
                pause = True
            shipyard_positions_yx = np.array([(my_shipyard.position.y, my_shipyard.position.x) for my_shipyard in cp.shipyards])
            shift_shipyard_positions_yx = np.array([
                (BOARD_SIZE//2 - my_shipyard.position.y, 
                 BOARD_SIZE//2 - my_shipyard.position.x) 
                for my_shipyard in cp.shipyards])
            
            ending_ship_positions_yx = (
                np.tile(ship_positions_yx, (shipyard_count,1)) +
                shift_shipyard_positions_yx.repeat(ship_count,axis=0)
            ).reshape(-1, 2)
            
            heats = distance.cdist(
                cls.spatial, ending_ship_positions_yx, 
                metric="cityblock").reshape(BOARD_SIZE, BOARD_SIZE, ship_count, shipyard_count)
            heats = heats.min(2)
            
            np.divide(1, heats + 1, out=heats)
            flipped_heats = torch.flip(torch.as_tensor(heats, dtype=torch.float), dims=(0,)).transpose(2,0)
            torch.mul(1 - remaining, flipped_heats, out=geometric_shipyard_ftrs[:shipyard_count, 0])
        
        if ship_count>0:
            for i, my_ship in enumerate(cp.ships):
                shift = (BOARD_SIZE//2 - my_ship.position.x, my_ship.position.y - BOARD_SIZE//2) 
                
                geometric_ship_ftrs[
                    i, 0, 
                    BOARD_SIZE - 1 - ship_positions_yx[cls.selector[i, :ship_count], 0], 
                    ship_positions_yx[cls.selector[i, :ship_count], 1]] = -2
                
                if shipyard_count>0:
                    geometric_ship_ftrs[
                        i, 0,
                        BOARD_SIZE - 1 - shipyard_positions_yx[:, 0],
                        shipyard_positions_yx[:, 1]] = ((my_ship.halite / diff) * 2 - 1)**(2-remaining)
                
                geometric_ship_ftrs[i, 0] = torch.roll(
                    geometric_ship_ftrs[i, 0], 
                    shifts=(shift[0], shift[1]), 
                    dims=(1,0))
                
                torch.mul(
                    geometric_ship_ftrs[i, 0], 
                    cls.middle_heat_flipped, 
                    out=geometric_ship_ftrs[i, 0])                
                
#                 flipped_fleet_heat_rolled = torch.roll(
#                     flipped_fleet_heat, 
#                     shifts=(shift[0], shift[1]), 
#                     dims=(1,0))
#                 
#                 torch.div(1, flipped_fleet_heat_rolled, out=flipped_fleet_heat_rolled)
#                 diff = flipped_fleet_heat_rolled.max() - flipped_fleet_heat_rolled.min()
#                 flipped_fleet_heat_rolled.sub_(halite_tensor.min())
#                 flipped_fleet_heat_rolled.div_(diff) 
#                 flipped_fleet_heat_rolled.mul_(2)
#                 flipped_fleet_heat_rolled.sub_(1)
                
                
#                 geometric_ship_ftrs[i, 1] = flipped_fleet_heat_rolled
    #             
    #             np.divide(1, heat, out=heat)
    #             
    #             heat_tensor = torch.as_tensor(heat, dtype=torch.float)
    #             flipped = torch.flip(heat_tensor, dims=(0,))
                
                
                
    #             geometric_ship_ftrs[i, 1] = flipped
                                    
    #             current_ship_cargo[0] += my_ship.halite
            
            # have to do this after since it will get affected by the flip multiplication
    #         for i, my_ship in enumerate(cp.ships):
    #             geometric_ship_ftrs[[j for j in range(len(cp.ships)) if j!= i], 0, 
    #                 BOARD_SIZE - my_ship.position.y - 1, my_ship.position.x] = -500
                    
#         if len(cp.shipyards)>0:
#             for i, my_shipyard in enumerate(cp.shipyards):
#                 
#                 heat = distance.cdist(
#                     cls.spatial, [(my_shipyard.position.x, my_shipyard.position.y)], 
#                     metric="cityblock").reshape(BOARD_SIZE, BOARD_SIZE)
#                             
#                 heat[my_shipyard.position.y, my_shipyard.position.x] = 0.75
#                 
#                 np.minimum(heat, cls.fleet_heat, out=heat)
#                 
#                 np.divide(1, heat, out=heat)
#                 
#                 flipped = torch.flip(torch.as_tensor(heat, dtype=torch.float), dims=(0,))
#                 
#                 torch.mul(remaining, flipped, out=flipped)
#                 torch.mul(halite_tensor, flipped, out=geometric_shipyard_ftrs[i, 0])
#                 
#                 shift = (BOARD_SIZE//2 - my_shipyard.position.x, my_shipyard.position.y - BOARD_SIZE//2) 
#                 
#                 geometric_shipyard_ftrs[i, 0] = torch.roll(
#                     geometric_shipyard_ftrs[i, 0], 
#                     shifts=(shift[0], shift[1]), 
#                     dims=(1,0))
#                 
#                 shifted_ship_positions_yx = (
#                     np.tile(ship_positions_yx, (shipyard_count,1)) +
#                     shifted_shipyard_positions_yx.repeat(ship_count,axis=0)
#                 ).reshape(shipyard_count, ship_count, 2)
                
                
        ts_ftrs[:, 1] = cp.halite
        for i, enemy in enumerate(board.opponents):
            ts_ftrs[:, 2 + i] = enemy.halite
            
        ts_ftrs[:, -PLAYERS:] = torch.max(cls.player_zeros, current_ship_cargo - prior_ship_cargo) #@UndefinedVariable
        
        return
FeatureCalculator.init()

class ActionSelector:
    def __init__(self, agent_manager):
        
        self._agent_manager = agent_manager
        self._reward_engine = RewardEngine()
        
        self._prior_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
        self._current_ship_cargo = torch.zeros(PLAYERS, dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self._best_ship_actions = np.zeros(MAX_SHIPS, dtype=np.int32)
        self._best_shipyard_actions = np.zeros(MAX_SHIPS, dtype=np.int32)
        
        self._geometric_ship_ftrs_v2 = torch.zeros(#@UndefinedVariable
            (2, MAX_SHIPS, SHIP_CHANNELS, BOARD_SIZE, BOARD_SIZE), 
            dtype=torch.float).to(DEVICE)#@UndefinedVariable
        
        self._geometric_shipyard_ftrs_v2 = torch.zeros(#@UndefinedVariable
            (2, MAX_SHIPS, SHIPYARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), 
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
            MAX_SHIPS, dtype=torch.uint8).to(DEVICE)#@UndefinedVariable
        
        self._non_terminal_shipyards = torch.zeros(#@UndefinedVariable
            MAX_SHIPS, dtype=torch.uint8).to(DEVICE)#@UndefinedVariable
            
    def set_prior_ship_cargo(self, prior_ship_cargo):
        self._prior_ship_cargo[:] = prior_ship_cargo
    
    def _ship_iterator_model(
            self, 
            board, 
            ship_count, 
            shipyard_count, 
            current_halite):
        if not TRAIN_MODELS:
            self._agent_manager.ship_cur_model.eval()
            self._agent_manager.shipyard_cur_model.eval()
            self._agent_manager.ship_tar_model.eval()
            self._agent_manager.shipyard_tar_model.eval()
            
        self._best_ship_actions.fill(0)
        self._best_shipyard_actions.fill(0)
        self._Q_ships_current_adj.zero_()
        self._Q_shipyards_current_adj.zero_()
        self._Q_ships_next.zero_()
        self._Q_shipyards_next.zero_()
        self._non_terminal_ships[:ship_count].fill_(1)
        self._non_terminal_ships[ship_count:].zero_()
        self._non_terminal_shipyards[:shipyard_count].fill_(1)
        self._non_terminal_shipyards[shipyard_count:].zero_()
        
        FeatureCalculator.update_tensors_v2(
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
                Q_ships_current = self._agent_manager.ship_cur_model(
                    self._geometric_ship_ftrs_v2[0, :ship_count], 
                    self._ts_ftrs_v2[0, :ship_count])
                if PRINT_STATEMENTS:
                    print(Q_ships_current, file=sys.__stdout__)
            if shipyard_count > 0:
                Q_shipyards_current = self._agent_manager.shipyard_cur_model(
                    self._geometric_shipyard_ftrs_v2[0, :shipyard_count], 
                    self._ts_ftrs_v2[0, :shipyard_count])
        
        if ship_count > 0:
            self._Q_ships_current_adj[:ship_count] = Q_ships_current
            retmax = Q_ships_current.max(dim=1)
            Q_ships_current_max = retmax.values
            conversion_indices = np.where(retmax.indices==5)[0][max_new_shipyards_allowed:]
            self._Q_ships_current_adj[conversion_indices, 5] = float('-inf')
            self._best_ship_actions[:ship_count] = self._Q_ships_current_adj[:ship_count].argmax(dim=1)
            if 5 in self._best_ship_actions:
                pause = True 
        if shipyard_count > 0:
            retmax = Q_shipyards_current.max(dim=1)
            Q_shipyards_current_max = retmax.values
            if max_new_ships_allowed > 0:
                if retmax.indices.sum() > max_new_ships_allowed:
                    shipyard_actions_f = retmax.indices.type(dtype=torch.float)
                    torch.mul(shipyard_actions_f, retmax.values, out=shipyard_actions_f)
                    shipyard_actions_f[shipyard_actions_f==0] = float('-inf')
                    self._best_shipyard_actions[:shipyard_count] = shipyard_actions_f.argsort(descending=True) < max_new_ships_allowed
                else:
                    self._best_shipyard_actions[:shipyard_count] = retmax.indices
        self._prior_ship_cargo.copy_(self._current_ship_cargo)
        
        new_board = step_forward(
            board, 
            self._best_ship_actions, 
            self._best_shipyard_actions)
                
        FeatureCalculator.update_tensors_v2(
            self._geometric_ship_ftrs_v2[1], 
            self._geometric_shipyard_ftrs_v2[1], 
            self._ts_ftrs_v2[1], 
            new_board, 
            self._current_ship_cargo, 
            self._prior_ship_cargo)
        
        if ship_count > 0:
            ships_lost_from_collision = self._reward_engine.compute_ship_rewards(
                new_board, 
                board, 
                ship_count,
                self._ship_rewards, 
                self._non_terminal_ships)
        else:
            ships_lost_from_collision = set()
            
        if shipyard_count > 0:
            self._reward_engine.compute_shipyard_rewards(
                new_board, 
                board, 
                ships_lost_from_collision,
                shipyard_count,
                self._shipyard_rewards, 
                self._non_terminal_shipyards)
                
        if ship_count > 0:
            self._agent_manager.store(
                ship_count,
                self._geometric_ship_ftrs_v2[0, :ship_count],
                self._ts_ftrs_v2[0, :ship_count],
                torch.LongTensor(self._best_ship_actions[:ship_count]),
                self._ship_rewards[:ship_count],
                self._geometric_ship_ftrs_v2[1, :ship_count],
                self._ts_ftrs_v2[1, :ship_count],
                self._non_terminal_ships[:ship_count],
                True)
        
        if shipyard_count > 0:
            self._agent_manager.store(
                shipyard_count,
                self._geometric_shipyard_ftrs_v2[0, :shipyard_count],
                self._ts_ftrs_v2[0, :shipyard_count],
                torch.LongTensor(self._best_shipyard_actions[:shipyard_count]),
                self._shipyard_rewards[:shipyard_count],
                self._geometric_shipyard_ftrs_v2[1, :shipyard_count],
                self._ts_ftrs_v2[1, :shipyard_count],
                self._non_terminal_shipyards[:shipyard_count],
                False)
        
        if TRAIN_MODELS:
            self._agent_manager.train_current_models()
                                
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
        if PRINT_STATEMENTS:   
            print("next action:", self._best_ship_actions[0])
        set_next_actions(board, self._best_ship_actions, self._best_shipyard_actions)
        
        return self._ship_rewards[:ship_count], self._shipyard_rewards[:shipyard_count]
    
    def reset_state(self):
        self._reward_engine.reset_state()

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
                
    def compute_ship_rewards(self, current_board, prior_board, prior_ship_count, ship_rewards, non_terminal_ships):
        current_player = current_board.current_player
        prior_player = prior_board.current_player        
        
        my_ship_ids = {ship.id:i for i, ship in enumerate(prior_player.ships)}
        my_ship_ids_rev = {k:v for k,v in my_ship_ids.items()}
        current_ships_set = set([ship.id for ship in current_player.ships])
        prior_ships_set = set([ship.id for ship in prior_player.ships])
        prior_ships_dict = {sid: prior_board.ships[sid] for sid in prior_ships_set}
        current_halite_cargo = {ship.id: ship.halite for ship in current_player.ships}
        ships_converted = set([ship.id for ship in prior_player.ships if ship.next_action==ShipAction.CONVERT])
        ships_lost_from_collision = prior_ships_set.difference(ships_converted).difference(current_ships_set)
        
        retained_ships = current_ships_set.intersection(prior_ships_set)
        deposit_weight = (.5 + current_board.step*self.reward_step)
        non_terminal_ships[[my_ship_ids[sid] for sid in ships_converted.union(ships_lost_from_collision)]] = 0
        max_halite = float(max(prior_board.observation['halite']))
        rewards = {ship.id: 
            max(0, current_halite_cargo[ship.id] - ship.halite)/max_halite + # diff halite cargo
            max(0, ship.halite - current_halite_cargo[ship.id])/max_halite # diff halite deposited
#             int(ship.halite==current_halite_cargo[ship.id] and ship.next_action==None)*-5 # inactivity
            if ship.id in retained_ships else 0 
            for ship in prior_ships_dict.values()}
#         rewards.update({sid: -500*deposit_weight for sid in ships_converted})
        rewards.update({sid: -1 for sid in ships_lost_from_collision})
        rewards = {my_ship_ids_rev[sid]:r for sid, r in rewards.items()}
#         rewards = {sid: v*(1-self.reward_step*current_board.step) for sid, v in rewards.items()}
        
        ship_rewards[:prior_ship_count] = torch.tensor(list(rewards.values()), dtype=torch.float)     
#         self._prior_ships_set = current_ships_set
#         self._prior_ships_dict = {sid: current_board.ships[sid] for sid in current_ships_set}
        return ships_lost_from_collision
    
    def compute_shipyard_rewards(self, 
             current_board, 
             prior_board, 
             ships_lost_from_collision,
             prior_shipyard_count, 
             shipyard_rewards, 
             non_terminal_shipyards):
        current_player = current_board.current_player
        prior_player = prior_board.current_player
        

        my_shipyard_ids = {shipyard.id:i for i, shipyard in enumerate(prior_player.shipyards)}
        current_shipyards_set = set([shipyard.id for shipyard in current_player.shipyards])
        prior_shipyards_set = set([shipyard.id for shipyard in prior_player.shipyards])
        prior_shipyards_dict = {sid: prior_board.shipyards[sid] for sid in prior_shipyards_set}
        shipyards_lost_from_collision = prior_shipyards_set.difference(current_shipyards_set)
        non_terminal_shipyards[[my_shipyard_ids[sid] for sid in shipyards_lost_from_collision]] = 0
        ship_collision_positions = set([prior_board.ships[sid].position for sid in ships_lost_from_collision])
        
        max_halite = float(max(prior_board.observation['halite']))
                
        rewards = {shipyard.id: 
           (current_player.halite - prior_player.halite)/max_halite +
           int(shipyard.position in ship_collision_positions)*-250./max_halite
           for shipyard in prior_shipyards_dict.values()}
        
        shipyard_rewards[:prior_shipyard_count] = torch.tensor(list(rewards.values()), dtype=torch.float)
    
    def reset_state(self):
        self._prior_ships_dict = {}
        self._prior_ships_set = set()
        self._prior_shipyards_dict = {}
        self._prior_shipyards_set = set()
        
RewardEngine.init_weights()
    
    
def agent(obs, config):
    global agent_managers
    sys.stdout = sys.__stdout__
    current_board = Board(obs, config)
    asm = agent_managers.get(current_board.current_player.id)
    step = current_board.step        
    asm.set_prior_ship_cargo(asm.current_ship_cargo)
    
    ship_reward, shipyard_reward = asm.action_selector.select_action(current_board)
    if PRINT_STATEMENTS:
        print("board step:", 
          step, 
          ",ship reward:", 
          ship_reward, 
          ",shipyard reward",
          shipyard_reward,
          file=sys.__stdout__)
    asm.episode_rewards[asm.in_game_episodes_seen] = (
        ship_reward.sum().item() + shipyard_reward.sum().item())
    
    asm.set_prior_board(current_board)
    asm.in_game_episodes_seen += 1
    
    return current_board.current_player.next_actions


class Outputter:
    def __init__(self):
        self._all_rewards = torch.zeros(
            0, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self._ship_losses = torch.zeros(
            0, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
        
        self._shipyard_losses = torch.zeros(
            0, 
            dtype=torch.float).to(DEVICE) #@UndefinedVariable
            
    def _moving_average(self, a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def output_logs(self, agent_managers, game_id):        
        append_p = ""
        for asm in agent_managers.values():
            self._all_rewards = torch.cat((self._all_rewards, asm.episode_rewards))        
            self._ship_losses = torch.cat((self._ship_losses, asm.ship_losses))  
            self._shipyard_losses = torch.cat((self._shipyard_losses, asm.shipyard_losses)) 
            
        fig, axis = plt.subplots(1, 3, figsize=(60, 20))
        arr = np.array(self._all_rewards)
        ma1_len = 100
        ma2_len = 1000
        ma1 = self._moving_average(arr, ma1_len)
        ma2 = self._moving_average(arr, ma2_len)
        axis[0].plot(range(len(arr)), arr, label="Step Reward")
        axis[0].plot(range(ma1_len,len(ma1)+ma1_len), ma1, label="MA{0}".format(ma1_len))
        axis[0].plot(range(ma2_len,len(ma2)+ma2_len), ma2, label="MA{0}".format(ma2_len))
        
        axis[1].plot(range(len(self._ship_losses)), np.array(self._ship_losses), label="Ship Model Loss", color="red")
        axis[2].plot(range(len(self._shipyard_losses)), np.array(self._shipyard_losses), label="Shipyard Model Loss", color="green")
        
#         axis[0].legend()
#         axis[1].legend()
#         axis[2].legend()
        axis[0].grid()
        axis[1].grid()
        axis[2].grid()
        fig.savefig("{0}/rewards.png".format(TIMESTAMP))
        plt.close(fig)
        with open("{0}/{1}_{0}g{2}.html".format(TIMESTAMP, append_p, game_id), "w") as f:
            f.write(env.render(mode="html", width=800, height=600))

if not os.path.exists(TIMESTAMP):
    os.makedirs(TIMESTAMP)
config = {
    "size": BOARD_SIZE, 
    "startingHalite": STARTING, 
    "episodeSteps": EPISODE_STEPS, 
    "convertCost": CONVERT_COST,
    "actTimeout": 1e8, 
    "runTimeout":1e8}

if RANDOM_SEED > -1: 
    config["randomSeed"] = RANDOM_SEED

i = 1
agents = [agent]

ship_dqn_cur = DQN(
    SHIP_CHANNELS,  # channels in
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
shipyard_dqn_cur = DQN(
    SHIPYARD_CHANNELS,  # channels in
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
 
ship_dqn_tar = DQN(
    SHIP_CHANNELS, # channels in
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
shipyard_dqn_tar = DQN(
    SHIPYARD_CHANNELS, #channels in
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
 
ship_huber = nn.SmoothL1Loss()
shipyard_huber = nn.SmoothL1Loss()
ship_optimizer = torch.optim.SGD( #@UndefinedVariable
    ship_dqn_cur.parameters(), 
    lr=LEARNING_RATE,
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)
shipyard_optimizer = torch.optim.SGD( #@UndefinedVariable
    shipyard_dqn_cur.parameters(), 
    lr=LEARNING_RATE,
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)

if os.path.exists("{0}/ship_dqn_{0}.nn".format(TIMESTAMP)):
    ship_dqn_tar.load_state_dict(torch.load("{0}/ship_dqn_{0}.nn".format(TIMESTAMP)))
if os.path.exists("{0}/shipyard_dqn_{0}.nn".format(TIMESTAMP)):
    shipyard_dqn_tar.load_state_dict(torch.load("{0}/shipyard_dqn_{0}.nn".format(TIMESTAMP)))

print("ship dqn:", ship_dqn_tar, file=sys.__stdout__)
print("shipyard dqn:", shipyard_dqn_tar, file=sys.__stdout__)

agent_managers = {i: 
    AgentStateManager(
        i, 
        ship_dqn_cur, 
        shipyard_dqn_cur,
        ship_dqn_tar, 
        shipyard_dqn_tar, 
        ship_huber, 
        shipyard_huber, 
        ship_optimizer, 
        shipyard_optimizer) for i in range(4)}

outputter = Outputter()
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
        outputter.output_logs(active_agent_managers, i)
        print("outputted data files")
        
    for asm in agent_managers.values():
        asm.game_id += 1
        asm.post_game_state_handle()
        
    i += 1
