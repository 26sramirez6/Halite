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

EPISODE_STEPS = 200
STARTING = 1000
BOARD_SIZE = 10
PLAYERS = 2
EGREEDY = 0.4
BATCH_SIZE = 32
LEARNING_RATE = 0.1
MOMENTUM  = 0.9
EPOCHS = 3
WEIGHT_DECAY = 5e-4
SHIPYARD_ACTIONS = [ShipyardAction.SPAWN, None]
SHIP_ACTIONS = [ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None,ShipAction.CONVERT]
SHIP_MOVE_ACTIONS = [ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None]

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
 
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(ftr_count)
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=LEARNING_RATE, momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)

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
            my_ship.next_action = choice(SHIP_ACTIONS)
            
    for my_shipyard in board.current_player.next_action.shipyards:
        if current_halite > board.configuration.spawn_cost:
            my_shipyard.next_action = choice(SHIPYARD_ACTIONS)
            current_halite -= board.configuration.spawn_cost
        else:
            my_shipyard.next_action = None

def model_select_action(board, model):
    pass
    
def agent(obs, config):
    board = Board(obs, config)
    current_player = board.current_player
    episodes[board.step, :halite_ftr] = board.observation["halite"]
    ship_halite_cur.fill(0)
    
    for my_ship in current_player.ships:
        pos = my_ship.position
        ship_index = (pos.y)*BOARD_SIZE + pos.x
        episodes[board.step, halite_ftr + ship_index] = 1
        episodes[board.step, halite_ftr + my_ship_exist_ftr + ship_index] = my_ship.halite
        ship_halite_cur[0] += my_ship.halite
    for my_shipyard in current_player.shipyards:
        pos = my_shipyard.position
        shipyard_index = (pos.y)*BOARD_SIZE + pos.x
        episodes[board.step, halite_ftr + my_ship_exist_ftr + my_ship_halite_ftr + shipyard_index] = 1
        
    ftr_index = halite_ftr + my_ship_exist_ftr + my_ship_halite_ftr + my_shipyard_exist_ftr
    for i, player in enumerate(board.opponents):
        for enemy_ship in player.ships:
            pos = enemy_ship.position
            ship_index = (pos.y)*BOARD_SIZE + pos.x
            episodes[board.step, ftr_index + ship_index] = 1
            episodes[board.step, ftr_index + enemy_ship_exist_ftr + ship_index] = enemy_ship.halite
            ship_halite_cur[i+1] += enemy_ship.halite
        for enemy_shipyard in player.shipyards:
            pos = enemy_shipyard.position
            shipyard_index = (pos.y)*BOARD_SIZE + pos.x
            episodes[board.step, ftr_index + enemy_ship_exist_ftr + enemy_ship_halite_ftr + shipyard_index] = 1
    
    ftr_index += enemy_ship_exist_ftr + enemy_ship_halite_ftr + enemy_shipyard_exist_ftr
    episodes[board.step, ftr_index] = board.configuration.episode_steps - board.step
    ftr_index += 1
    for i, player in enumerate(board.players.values()):
        episodes[board.step, ftr_index + i] = player.halite
    ftr_index += halite_deposited_by_player_ftr
    
    episodes[board.step, -halite_mined_by_player_ftr:] = np.maximum(player_zeros, ship_halite_cur - ship_halite_prev)
    ship_halite_prev[:] = ship_halite_cur
    episode_rewards[board.step] = compute_reward(board)
    print(board.step)
    
    randomize = np.random.rand() < EGREEDY
    if randomize:
        randomize_action(board)
    else:
        model_select_action(board, net)
        
    return current_player.next_actions
    
env = make("halite", configuration={"size": BOARD_SIZE, "startingHalite": STARTING, "episodeSteps": EPISODE_STEPS})
env.reset(PLAYERS)
print("starting")
env.run([agent, "random"])
print("complete")

out = env.render(mode="html", width=800, height=600)
with open("test.html", "w") as f:
    f.write(out)