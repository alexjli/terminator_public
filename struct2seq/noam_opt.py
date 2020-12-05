import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        optimizer_state = self.optimizer.state_dict()
        return_state = {'step': self._step, 'warmup': self.warmup, 'factor': self.factor, 'model_size': self.model_size, 'rate': self._rate, 'optimizer_state': optimizer_state}
        return return_state

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.warmup = state_dict['warmup']
        self.factor = state_dict['factor']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['rate']
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        
def get_std_opt(parameters, d_model, lr_multiplier = 1, regularization = 1e-3):
    return NoamOpt(
        d_model, 2*lr_multiplier, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=regularization)
    )
