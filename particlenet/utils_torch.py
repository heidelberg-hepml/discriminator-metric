import numpy as np
import torch


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, initial_lr):
        self.initial_lr = initial_lr
        self._optimizer = optimizer
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()
    
    def lr_schedule(self):
      #  self.initial_lr = 3 * 1e-4
        lr = self.initial_lr
        if self.n_steps <= 8:
            lr = self.initial_lr + (3 * 1e-3 - self.initial_lr) * self.n_steps / 8
        elif (self.n_steps <= 16) & (self.n_steps > 8):
            lr = 3 * 1e-3 - (3 * 1e-3 - 3 * 1e-4) * (self.n_steps - 8) / 8
        elif self.n_steps > 16:
            lr = 3 * 1e-4 - (3 * 1e-4 - 5 * 1e-7) * (self.n_steps - 16) / 4   
        return lr

    
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_schedule(self.n_steps)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


