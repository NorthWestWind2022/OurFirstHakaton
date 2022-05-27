import torch


DEVICE = torch.device('cuda')
ACTOR_LR = 0.001
CRITIC_LR = 0.002
TAU = 0.05
BUFFER_SIZE = 100
HIDDEN_SIZE = 256
GAMMA = 0.99