
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden_dim,
        n_layers,
        activation=nn.ReLU(),
        bnorm=False,
        input_norm=None,
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.net = []

        dims = [in_channel] + [hidden_dim] * n_layers + [out_channel]
        net = []
        if input_norm:
            net.append(input_norm)
        for h1, h2 in zip(dims[:-1], dims[1:]):
            net.append(nn.Linear(h1, h2))
            if bnorm:
                net.append(nn.BatchNorm1d(h2))
            net.append(activation)
        net.pop()
        
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MLP_ED(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_embd: int,
        n_layers: int,
    ):
        """ed: environment dynamics"""

        super().__init__()
        
        self.out_shape = state_dim + 1 # next_state_dim + reward_dim
        self.in_shape = state_dim + action_dim # state_dim + action_dim 
        
        self.mlp = MLP(
            in_channel=self.in_shape,
            out_channel=self.out_shape,
            hidden_dim=n_embd,
            n_layers=n_layers,
            activation=nn.GELU(),
        )
        

    def forward(self, states, actions):
        B, T, states_dim = states.shape
        input_states = states.reshape(B * T, states_dim)
        
        B, T, actions_dim = actions.shape
        input_actions = actions.reshape(B * T, actions_dim)
        
        input_features = torch.cat([input_states, input_actions], dim=1)
        pred = self.mlp(input_features)
        
        next_state_preds = pred[:, :states_dim]
        reward_preds = pred[:, states_dim]
                
        return next_state_preds, reward_preds