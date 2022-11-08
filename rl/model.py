import torch
import torch.nn as nn
import numpy as np

from rl.distributions import Bernoulli, Categorical, DiagGaussian
from rl.pas_rnn_model import PASRNN


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, action_space, config=None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.action_space = action_space
        self.config = config
        if base_kwargs is None:
            base_kwargs = {}
        self.name = base

        if config.robot.policy == 'pas_rnn':
            base=PASRNN
            self.base = base(base_kwargs, config)

            if config.pas.encoder_type == 'vae':
                if config.sim.train_val_sim == "turtlebot":
                    vae_weight_file = 'data/Turtlebot_LabelVAE_CircleFOV30/label_vae_ckpt/label_vae_weight_60.pth'
                elif config.sim.train_val_sim == "circle_crossing":
                    vae_weight_file = 'data/LabelVAE_CircleFOV30/label_vae_ckpt/label_vae_weight_300.pth'
                self.base.Label_VAE.load_state_dict(torch.load(vae_weight_file), strict=True)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError
    
    def compute_position(self, pxy, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            action = action.cpu().numpy()
            theta = np.arctan2(action[1], action[0])
            px = pxy[0] + action[0] * delta_t
            py = pxy[1] + action[1] * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px.cpu().numpy(), py.cpu().numpy()


    def act(self, inputs, rnn_hxs, masks, deterministic=False, visualize=False):
        if self.name == 'pas_rnn':
            value, actor_features, rnn_hxs, decoded = self.base(inputs, rnn_hxs, masks, infer=True)
        else:
            raise NotImplementedError

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        
        action_log_probs = dist.log_probs(action)
        if self.name == 'pas_rnn':
            return value, action, action_log_probs, rnn_hxs, decoded
        else:
            return value, action, action_log_probs, rnn_hxs



    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        return value

        

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        if self.name == 'pas_rnn':
            value, actor_features, rnn_hxs, z_l, z, decoded, mu, logvar = self.base(inputs, rnn_hxs, masks, infer=False) 
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=False) 
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        if self.name == 'pas_rnn':
            return value, action_log_probs, dist_entropy, rnn_hxs, z_l, z, decoded, mu, logvar
        else:
            return value, action_log_probs, dist_entropy, rnn_hxs


