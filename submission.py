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
MOMENTUM  = 0.9
EPOCHS = 3
MAX_ACTION_SPACE = 1000
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [None, ShipyardAction.SPAWN]
SHIP_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [None, ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]

class Net(nn.Module):
    def __init__(self, in_features):
        super(Net, self).__init__()
        self.l1 = nn.Linear(in_features, 20)
        self.s1 = nn.Sigmoid()
        self.l2 = nn.Linear(20, 100)
        self.s2 = nn.Sigmoid()
        self.l3 = nn.Linear(100, 100)
        self.s3 = nn.Sigmoid()
        self.l4 = nn.Linear(100, 1)
        nn.init.xavier_uniform(self.l1)
        nn.init.xavier_uniform(self.l2)
        nn.init.xavier_uniform(self.l3)
        nn.init.xavier_uniform(self.l4)
        
    def forward(self, x):
        y = self.l1(x)
        y = self.s1(y)
        
        y = self.l2(y)
        y = self.s2(y)
        
        y = self.l3(y)
        y = self.s3(y)
        return self.l4(y)


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

game_step = 0
episodes = np.zeros((EPISODE_STEPS, ftr_count))
episode_rewards = np.zeros(EPISODE_STEPS)
halite_mined = np.zeros(PLAYERS)
    
ship_halite_prev = np.zeros(PLAYERS)
ship_halite_cur = np.zeros(PLAYERS)
player_zeros = np.zeros(PLAYERS)
ships_spawned = 0
shipyards_spawned = 0
halite_lost = 0
halite_stolen = 0

# criterion = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(ftr_count)
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=LEARNING_RATE, momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)

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

def update_state(state, board, ship_halite_cur, ship_halite_prev):        
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
    update_state(episodes[board.step], board, ship_halite_cur, ship_halite_prev)
    ship_halite_prev[:] = ship_halite_cur
    mp.set_ship_halite_prev(ship_halite_cur)
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