import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.autograd import Variable
import torch
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from rl.utils import init



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),1, -1)  #(seq_len*nenv,1 -1)


class RNNBase(nn.Module):
    def __init__(self, args, input_size):
        super(RNNBase, self).__init__()
        self.args = args

        self.gru = nn.GRU(args.rnn_input_size, args.rnn_hidden_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def _forward_gru(self, x, hxs, masks):
        '''
        Forward pass for the model
        params:
        x : input position (sequence, nenv, 1, args.rnn_embedding_size)
        h : hidden state of the current RNN (sequence, nenv, 1, args.rnn_hidden_size)
        masks : cell state of the current RNN (sequence, nenv, 1)
        '''
        # for acting model, input shape[0] == hidden state shape[0]
        if x.size(0) == hxs.size(0):
            # use env dimension as batch
            # [1, 12, 6, ?] -> [1, 12*6, ?] or [30, 6, 6, ?] -> [30, 6*6, ?]
            seq_len, nenv, agent_num, _ = x.size() 
            x = x.view(seq_len, nenv*agent_num, -1) 
            hxs_times_masks = hxs * (masks.view(seq_len, nenv, 1, 1))
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv*agent_num, -1) # (1, 12, *)            
            x, hxs = self.gru(x, hxs_times_masks) # we already unsqueezed the inputs in SRNN forward function
            x = x.view(seq_len, nenv, agent_num, -1) 
            hxs = hxs.view(seq_len, nenv, agent_num, -1) 

        # during update, input shape[0]=1 * nsteps (30) = hidden state shape[0]
        else:
            seq_len, nenv, agent_num, _ = x.size()

            # Same deal with masks
            masks = masks.view(seq_len, nenv)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # for the [29, num_env] boolean array, if any entry in the second axis (num_env) is True -> True
            # to make it [29, 1], then select the indices of True entries
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1) 
                            .nonzero()
                            .squeeze()
                            .cpu())
                    #(29,6) --> (29)

            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [seq_len]

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                x_in = x[start_idx:end_idx]
                x_in = x_in.view(x_in.size(0), x_in.size(1)*x_in.size(2), x_in.size(3))
                hxs = hxs.view(hxs.size(0), nenv, agent_num, -1)
                hxs = hxs * (masks[start_idx].view(1, -1, 1,1))
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                rnn_scores, hxs = self.gru(x_in, hxs) 

                outputs.append(rnn_scores) 

            x = torch.cat(outputs, dim=0) 
            # flatten
            x = x.view(seq_len, nenv, agent_num, -1) 
            hxs = hxs.view(1, nenv, agent_num, -1)
        return x, hxs


class FeatureRNN(RNNBase):

    def __init__(self, args, input_size):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(FeatureRNN, self).__init__(args, input_size)

        self.args = args

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(input_size, args.rnn_input_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Output linear layer
        self.output_linear = nn.Linear(args.rnn_hidden_size, args.rnn_output_size)



    def forward(self, pos, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position (sequence, nenv, args.rnn_input_size)
        h : hidden state of the current RNN (sequence, nenv, 1, args.rnn_hidden_size)
        masks : cell state of the current RNN (sequence, nenv, 1)
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(pos) # (seq_len, nenv,1, args.rnn_embedding_size)
        encoded_input = self.relu(encoded_input)

        x, h_new = self._forward_gru(encoded_input, h, masks) # x : (seq_len, nenv, 1, 256)

        outputs = self.output_linear(x)

        return outputs, h_new

def mlp(input_dim, mlp_dims, last_relu=False):
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(init_(nn.Linear(mlp_dims[i], mlp_dims[i + 1])))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Label_VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        Args:
            latent_size : output latent vector size for encoder (args.rnn_input_size)
        """
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.encoder = nn.Sequential(
            init_(nn.Conv2d(1, 32, 4, 2, 1)),          
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            init_(nn.Conv2d(32, 64, 4, 2, 1)),          
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 128, 4, 2, 1)),          
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            init_(nn.Conv2d(128, 64, 4, 2, 1)),         
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 32, 4, 2, 1)),       
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32*3*3, args.rnn_output_size*2))           
        )
        
        self.decoder = nn.Sequential(
            init_(nn.Linear(args.rnn_output_size, 32*3*3)),               
            View((-1, 32, 3, 3)),              
            nn.ReLU(), 
            nn.BatchNorm2d(32),
            init_(nn.ConvTranspose2d(32, 64, 4, 2, 1)),      
            nn.ReLU(), 
            nn.BatchNorm2d(64),
            init_(nn.ConvTranspose2d(64, 128, 4, 2, 1)),    
            nn.ReLU(),
            nn.BatchNorm2d(128),
            init_(nn.ConvTranspose2d(128, 64, 4, 2, 1,1)),
            nn.ReLU(), 
            nn.BatchNorm2d(64),
            init_(nn.ConvTranspose2d(64, 32, 4, 2, 1)), 
            nn.ReLU(), 
            nn.BatchNorm2d(32),
            init_(nn.ConvTranspose2d(32, 1, 4, 2, 1)), 
            nn.Sigmoid()
        )

        self.linear_mu = nn.Linear(args.rnn_output_size*2, args.rnn_output_size)
        self.linear_var = nn.Linear(args.rnn_output_size*2, args.rnn_output_size)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_()).cuda()
        return mu + std*eps
    
    def encode(self, grid):        
        x = self.encoder(grid) # z:(1, 1, args.rnn_output_size)
        z_mu = self.linear_mu(x)
        z_log_variance = torch.log(self.softplus(self.linear_var(x)))
        z = self.reparameterize(z_mu, z_log_variance)
        return z_mu.squeeze(1), z_log_variance.squeeze(1), z
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded


    def forward(self, grid):
        """
        Args:
            grid:(seq_len*nenv, 1, *grid_shape)
        Return:
            z : (seq_len*nenv, 1, args.rnn_output_size)
            decoded : (seq_len*nenv, 1, *grid_shape)
        """
        z_mu, z_log_variance, z = self.encode(grid) # z:(1, 1, args.rnn_output_size)
        decoded = self.decode(z) 

        return z_mu.squeeze(1), z_log_variance.squeeze(1), z, decoded 
    

class Sensor_VAE(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        """
        Args:
            latent_size : output latent vector size for encoder (args.rnn_input_size)
        """
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.sequence = config.pas.sequence
        
        self.encoder = nn.Sequential(
            init_(nn.Conv2d(self.sequence, 32, 4, 2, 1)),         
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            init_(nn.Conv2d(32, 64, 4, 2, 1)),          
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 128, 4, 2, 1)),          
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            init_(nn.Conv2d(128, 64, 4, 2, 1)),          
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 32, 4, 2, 1)),          
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32*3*3, args.rnn_output_size*2))          
        )
        
        self.linear_mu = nn.Linear(args.rnn_output_size*2, args.rnn_output_size)
        self.linear_var = nn.Linear(args.rnn_output_size*2, args.rnn_output_size)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_()).cuda()
        return mu + std*eps
    
    def encode(self, grid):        
        x = self.encoder(grid) 
        z_mu = self.linear_mu(x)
        z_log_variance = torch.log(self.softplus(self.linear_var(x)))
        z = self.reparameterize(z_mu, z_log_variance)
        return z_mu.squeeze(1), z_log_variance.squeeze(1), z
       

class Sensor_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        Args:
            latent_size : output latent vector size for encoder (args.rnn_input_size)
        """
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.encoder = nn.Sequential(
            init_(nn.Conv2d(1, 32, 4, 2, 1)),         
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            init_(nn.Conv2d(32, 64, 4, 2, 1)),         
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 128, 4, 2, 1)),         
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            init_(nn.Conv2d(128, 64, 4, 2, 1)),         
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 32, 4, 2, 1)),         
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32*3*3, args.rnn_output_size))
        )
        
    def forward(self, grid):
        """
        Args:
            grid:(seq_len*nenv, 1, *grid_shape)
        Return:
            z : (seq_len*nenv, 1, args.rnn_output_size)
        """
        z = self.encoder(grid) 

        return z
    
class Sensor_CNN_seq(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        """
        Args:
            latent_size : output latent vector size for encoder (args.rnn_input_size)
        """
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        
        self.encoder = nn.Sequential(
            init_(nn.Conv2d(config.pas.sequence, 32, 4, 2, 1)),         
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            init_(nn.Conv2d(32, 64, 4, 2, 1)),         
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 128, 4, 2, 1)),         
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            init_(nn.Conv2d(128, 64, 4, 2, 1)),         
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            init_(nn.Conv2d(64, 32, 4, 2, 1)),          
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32*3*3, args.rnn_output_size))          
        )        


    def forward(self, grid):
        """
        Args:
            grid:(seq_len*nenv, S, *grid_shape)
        Return:
            z : (seq_len*nenv, 1, args.rnn_output_size)
        """
        z = self.encoder(grid) 

        return z
    


class PASRNN(nn.Module):
    """
    Class representing the PASRNN model
    """
    def __init__(self, args, config, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(PASRNN, self).__init__()
        self.args=args
        self.config = config

        self.seq_length = config.pas.sequence
        self.num_steps = args.num_steps
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.rnn_input_size = args.rnn_input_size
        self.rnn_hidden_size = args.rnn_hidden_size
        self.output_size = args.rnn_output_size

        self.latent_size = args.rnn_output_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        
        num_inputs = output_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, output_size)), nn.Tanh(),
            init_(nn.Linear(output_size, output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, output_size)), nn.Tanh(),
            init_(nn.Linear(output_size, 32)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(32, 1))

        if self.config.pas.gridtype == 'global':
            robot_state_length = 9
        
        elif self.config.pas.gridtype == 'local':
            if self.config.action_space.kinematics == 'holonomic':    
                robot_state_length = 4 # robot state vector : rel_px, rel_py, vx, vy
            else:
                robot_state_length = 5 # robot state vector : rel_px, rel_py, theta, v, w
        vector_feat_dim = 16
        self.vector_linear = init_(nn.Linear(robot_state_length, vector_feat_dim)) 
        embed_input_size = args.rnn_output_size+vector_feat_dim
        
            
        if config.pas.encoder_type == 'vae':
            self.Sensor_VAE = Sensor_VAE(args, config)

            self.Label_VAE = Label_VAE(args)
        else:
            if config.pas.seq_flag:
                self.Sensor_CNN_seq = Sensor_CNN_seq(args, config)
            else:
                self.Sensor_CNN = Sensor_CNN(args)

        self.FeatureRNN = FeatureRNN(args, embed_input_size)




    def forward(self, inputs, rnn_hxs, masks, infer=False):
        """[summary]
        Args:
            inputs  ['vector': (1*nenv, 1, vec_length) , 'grid':(seq_len*nenv, 1, *grid_shape) or (seq_len*nenv, S, *grid_shape)]   
            rnn_hxs ([type]): ['vector': (1*nenv, 1, hidden_size) , 'grid':(1*nenv, 1, hidden_size)]
            masks ([type]): [description] (seq_len*nenv, 1) or  (seq_len*nenv,seq_length)             
            infer (bool, optional): [description]. Defaults to False.
        """

        vector = inputs['vector'] 
        grid = inputs['grid'] 
        
        if infer: # for validation too
            # Test time
            num_steps = 1
            nenv = inputs['grid'].shape[0]

        else:
            num_steps = self.num_steps 
            nenv = int(inputs['grid'].shape[0]/num_steps)
            
        vec_feat = self.vector_linear(vector)          
        
        if self.config.pas.encoder_type == 'vae':
            mu, logvar, z = self.Sensor_VAE.encode(grid)
            with torch.no_grad():
                decoded = self.Label_VAE.decoder(z)


            feat = torch.cat((vec_feat, z), -1)

            current_masks = masks[:,[-1]]
            feat = reshapeT(feat, num_steps, nenv)
            hidden_states_vector_RNNs = reshapeT(rnn_hxs['policy'], 1, nenv)
            outputs, h_nodes = self.FeatureRNN(feat, hidden_states_vector_RNNs, current_masks)
            x = outputs[:, :, 0, :] 
            rnn_hxs['policy'] = h_nodes

        else:   
            if self.config.pas.seq_flag:
                z = self.Sensor_CNN_seq(grid)
            else:
                z = self.Sensor_CNN(grid)
            feat = torch.cat((vec_feat, z), -1)
            if self.config.pas.seq_flag:
                decoded = deepcopy(grid[:,-1])
            else:
                decoded = deepcopy(grid)                

            feat = reshapeT(feat, num_steps, nenv)
            hidden_states_vector_RNNs = reshapeT(rnn_hxs['policy'], 1, nenv)
            outputs, h_nodes = self.FeatureRNN(feat, hidden_states_vector_RNNs, masks)
            x = outputs[:, :, 0, :] # x: (seq_len, nenv, args.rnn_output_size)
            rnn_hxs['policy'] = h_nodes



        hidden_critic = self.critic(x)  
        hidden_actor = self.actor(x) 


        if not infer:
            if self.config.pas.encoder_type == 'vae' :
                with torch.no_grad():
                    _,_, z_l, _ = self.Label_VAE(inputs['label_grid'][:,[0]])


            else:
                z_l = torch.zeros(z.shape).cuda().reshape(-1, 1, self.latent_size)
                decoded = None 
                mu, logvar = None, None

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0) 
        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs, decoded
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs, z_l, z.reshape(-1, 1, self.latent_size), decoded, mu, logvar


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))