import os
import matplotlib.pyplot as plt
import glob 
import numpy as np
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from arguments import get_args
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from rl.pas_rnn_model import Label_VAE
from crowd_nav.configs.config import Config


def make_sequence(data_path, phase, sequence=1):    
    label_grid_files = sorted(glob.glob(data_path+'*label_grid.npy')) 
    sensor_grid_files = sorted(glob.glob(data_path+'*sensor_grid.npy')) 
    if phase !='train':
        vector_files = sorted(glob.glob(data_path+'*vector.npy')) 
        id_grid_files = sorted(glob.glob(data_path+'*id_grid.npy')) 

    vector = []
    label_grid = []
    sensor_grid = []
    id_grid = []
    for i in range(len(label_grid_files)):
        if phase != 'train':
            v_f, lg_f, sg_f, id_f = vector_files[i], label_grid_files[i], sensor_grid_files[i], id_grid_files[i] 
        else:
            lg_f, sg_f = label_grid_files[i], sensor_grid_files[i]
        
        epi_label_grid = np.load(lg_f, mmap_mode='r')
        epi_sensor_grid = np.load(sg_f, mmap_mode='r')
        if phase != 'train':
            epi_vector = np.load(v_f, mmap_mode='r')
            epi_id_grid = np.load(id_f, mmap_mode='r')
        timestamp = 0
        for k in range(len(epi_label_grid)):   
            if phase == 'train':  
                lg, sg = epi_label_grid[k], epi_sensor_grid[k]
            else:
                v, lg,sg, ig = epi_vector[k], epi_label_grid[k], epi_sensor_grid[k], epi_id_grid[k]
            if timestamp == 0.:
                lg = lg # (100, 100)
                sg = sg
                if phase != 'train':
                    v = v
                    ig = ig

                labelG = deque(maxlen=1)
                sensorG = deque(maxlen=sequence)
                if phase != 'train':
                    vec = deque(maxlen=sequence)
                    idG = deque(maxlen=1)

                sensorG.extend(np.vstack([sg for i in range(sequence)]))
                if phase != 'train':
                    vec.extend(np.vstack([v for i in range(sequence)])) 

            else:
                sensorG.extend(sg)
                if phase != 'train':
                    vec.extend(v)
            
            labelG.extend(lg)
            if phase != 'train':
                idG.extend(ig)
            
            label_grid.append(np.array(labelG, dtype=np.float32).copy())
            sensor_grid.append(np.array(sensorG, dtype=np.float32).copy())
            if phase != 'train':
                vector.append(np.array(vec, dtype=np.float32).copy())
                id_grid.append(np.array(idG, dtype=np.float32).copy())
            
            timestamp+=1
            
        if i % 50 == 0:
            print(i, 'file sequence has been made.')
            
    if phase == 'train':
        return label_grid, sensor_grid
    else:        
        return vector, label_grid, sensor_grid, id_grid 




class DATA(Dataset):
    def __init__(self,logging, phase, sequence=1):
        data_path = 'VAEdata_CircleFOV30/'+phase +'/' 
        self.phase = phase

        if phase =='train':
            self.label_grid, self.sensor_grid = make_sequence(data_path, phase, sequence=sequence)
        else:
            self.vector, self.label_grid, self.sensor_grid, self.id_grid = make_sequence(data_path, phase, sequence=sequence)
        logging.info('Phase : {}, sequential data : {:d}'. format(phase, len(self.label_grid)))   

    def __len__(self):
        return len(self.label_grid)

    def __getitem__(self, index):
        label_grid = self.label_grid[index].copy()
        sensor_grid = self.sensor_grid[index].copy()
        if self.phase != 'train':
            vector = self.vector[index].copy()
            id_grid = self.id_grid[index].copy()
            return (vector, label_grid, sensor_grid, id_grid)
        else:
            return (label_grid, sensor_grid)

def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])

def rearrange(data):
    # after loaded from the dataloader
    # (N, T, ~) --> (T, N, ~) --> (T*N, ~)
    data = data.unsqueeze(2).to(device)
    N = data.size()[0]
    T = data.size()[1]
    data = data.transpose(1,0)   
    return _flatten_helper(T, N, data.squeeze(1)) 

def stack_tensors(label_grid, sensor_grid, id_grid=None, mask=None):
    label_grid = rearrange(label_grid)
    sensor_grid = rearrange(sensor_grid)
    if id_grid is not None:
        id_grid = rearrange(id_grid)
    # if mask is not None:
    #     mask = rearrange(mask)
    return label_grid, sensor_grid, id_grid# ,  mask


def reconstruction_loss(x, x_recon):
    """[summary]

    Args:
        x ([N,1,120,120]): [description]
        x_recon ([N,1,120,120]): [description]
        distribution (str, optional): [description]. Defaults to 'gaussian'.
        overest (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    batch_size = x.size(0)
    assert batch_size != 0

    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size) 
    return recon_loss


def MSE(x, x_prime):
    batch_size = x.size(0)
    return F.mse_loss(x, x_prime, reduction='mean').div(batch_size)

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

def KL_loss(mu, logvar):
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0) 
    return KLD



def VAE_loss(x, recon_x, mu, logvar):
    MSE = reconstruction_loss(x, recon_x)
    KLD = KL_loss(mu, logvar)
    return MSE, KLD




def Label_vae_evaluate(beta, logging, loader, model, epoch=None):
    recon_loss_epoch = []
    k_loss_epoch = []

    model.eval()

    with torch.no_grad():
        for vector, label_grid, sensor_grid, id_grid in loader:
            """
            vector : (T*N, 36)
            *_grid : (T*N, 120, 120)
            """     
            label_grid, sensor_grid, id_grid = stack_tensors(label_grid, sensor_grid, id_grid)  # (N,T, ...) --> (T*N, ...)       

            mu_l, logvar_l, z_l, decoded_l = model(label_grid)    

            recon_loss, k_loss = VAE_loss(label_grid, decoded_l, mu_l, logvar_l)
            loss = recon_loss + k_loss * beta


            recon_loss_epoch.append(recon_loss.item())  
            k_loss_epoch.append(k_loss.item())            
        
        avg_recon_loss = average(recon_loss_epoch)
        avg_k_loss = average(k_loss_epoch)
        
        if epoch == None:    
            loss = logging.info('(Test) recon_loss: {:.4f}, k_loss: {:.4f}'. format(avg_recon_loss, avg_k_loss))  
            save_path = out_dir+'/Label_VAE_test_sample/'
        else:
            loss = logging.info('(Eval Epoch {:d}) recon_loss: {:.4f}, k_loss: {:.4f}'. format(epoch, avg_recon_loss, avg_k_loss))   
            save_path = out_dir+'/Label_VAE_val_sample/'

            writer.add_scalar('Label_VAE_val_recon_loss', avg_recon_loss, epoch)
            writer.add_scalar('Label_VAE_val_k_loss', avg_k_loss, epoch)

            
        if not os.path.exists(save_path):
            os.makedirs(save_path) 


        vectors = vector.cpu().numpy()
        for k in range(batch_size):
            fig, axes = plt.subplots(ncols=3, figsize=(6*3+2,6))
            ax = axes[0]
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)            
            
            robot_traj = []
            a1_traj = []
            a2_traj = []
            a3_traj = []
            a4_traj = []
            a5_traj = []
            a6_traj = []
            # vector length : robot 9, human 5
            robot_traj.append(vectors[k, 0, :2]) 
            a1_traj.append(vectors[k, 0, 9:11]) 
            a2_traj.append(vectors[k, 0, 14:16]) 
            a3_traj.append(vectors[k, 0, 19:21]) 
            a4_traj.append(vectors[k, 0, 24:26]) 
            a5_traj.append(vectors[k, 0, 29:31]) 
            a6_traj.append(vectors[k, 0, 34:36])

            robot_traj = np.array(robot_traj) 
            a1_traj = np.array(a1_traj)
            a2_traj = np.array(a2_traj)
            a3_traj = np.array(a3_traj)
            a4_traj = np.array(a4_traj)
            a5_traj = np.array(a5_traj)
            a6_traj = np.array(a6_traj)
            ax.plot(robot_traj[:,0], robot_traj[:,1], 'r+')
            ax.plot(a1_traj[:,0], a1_traj[:,1], 'bo')
            ax.plot(a2_traj[:,0], a2_traj[:,1], 'go')
            ax.plot(a3_traj[:,0], a3_traj[:,1], 'yo')
            ax.plot(a4_traj[:,0], a4_traj[:,1], 'co')
            ax.plot(a5_traj[:,0], a5_traj[:,1], 'mo')
            ax.plot(a6_traj[:,0], a6_traj[:,1], 'ko')
            ax.set_title('traj')
            i = 2 
            for grid in [label_grid[k], decoded_l[k]]:
                ax = axes[i-1]
                Con = ax.contourf(grid.squeeze(0).cpu().numpy(), cmap='binary', vmin = 0.0, vmax = 1.0)
                i+=1     
            fig.colorbar(Con, ax = axes.ravel().tolist())

            if epoch == None:
                plt.savefig(save_path + 'ex_'+str(k)+'.png')
            else:
                plt.savefig(save_path + 'epoch_'+str(epoch)+'_ex_'+str(k)+'.png')
            plt.close()
    return loss


def Label_vae_train(beta, logging, train_loader, validation_loader, model, ckpt_path, num_epochs, learning_rate = 0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    model.train()
    for epoch in range(1, num_epochs+1):
        recon_loss_epoch = []
        k_loss_epoch = []
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        if epoch % 20 == 0 :
            loop.set_postfix(loss = Label_vae_evaluate(beta, logging, validation_loader, model, epoch))
            ckpt_file = os.path.join(ckpt_path, 'label_vae_weight_'+str(epoch)+'.pth')
            torch.save(model.state_dict(), ckpt_file)
            model.train()
        for label_grid, sensor_grid in loop: 

            label_grid, sensor_grid, _ = stack_tensors(label_grid, sensor_grid)  # (N,T, ...) --> (T*N, ...)       

            mu_l, logvar_l, z_l, decoded_l = model(label_grid)    
            recon_loss, k_loss = VAE_loss(label_grid, decoded_l, mu_l, logvar_l)
            loss = recon_loss + k_loss * beta
        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(recon_loss=recon_loss.item(), k_loss=k_loss.item())
            recon_loss_epoch.append(recon_loss.item())
            k_loss_epoch.append(k_loss.item())

        avg_recon_loss = average(recon_loss_epoch)
        avg_k_loss = average(k_loss_epoch)

        logging.info('(Epoch {:d}) recon_loss: {:.4f}, k_loss: {:.4f}'.
                     format(epoch, avg_recon_loss, avg_k_loss))      

        writer.add_scalar('Label_VAE_train_recon_loss', avg_recon_loss, epoch)
        writer.add_scalar('Label_VAE_train_val_k_loss', avg_k_loss, epoch)




if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 300 # 60 for the turtlebot experiment
    beta = 0.00025
    vae_learning_rate =  0.001 

    algo_args = get_args()
    config = Config()
    encoder_type = 'vae'     
    sequence = config.pas.sequence
    
    max_grad_norm = algo_args.max_grad_norm
    batch_size = 16    
    grid_shape = [100, 100] 

    

    
    # ###########################################
    # Label_AE training
    
    import logging

    output_path = 'LabelVAE_CircleFOV30'

    # configure logging
    out_dir = 'data/'+ output_path  
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'label_vae_train.log')
    mode = 'a'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %%M:%S")

    summary_path = out_dir+'/runs_label_vae_train' 
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path) 
                        

    logging.info('(config) encodertype:{},beta: {:.6f}, seq_length: {:d}, '. format(encoder_type, beta, sequence))    


    label_vae_ckpt_path = out_dir+'/label_vae_ckpt/'    

    logging.info('Learning rate:{:.6f}'. format(vae_learning_rate))   
    

    
    if not os.path.exists(label_vae_ckpt_path):
        os.makedirs(label_vae_ckpt_path)

    
    if encoder_type == 'vae':
        label_vae = Label_VAE(algo_args)
        
    label_vae.to(device)

    ## loading checkpoint for label_vae_train
    # # if resume:
    # label_vae_ckpt_file = os.path.join(label_vae_ckpt_path, 'label_vae_weight_'+str(300)+'.pth')
    # label_vae.load_state_dict(torch.load(label_vae_ckpt_file))

    
    train_set = DATA(logging, 'train', sequence=1) 
    val_set = DATA(logging, 'val', sequence=1) 
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size,num_workers=1,pin_memory=True, drop_last=True)
    validation_loader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)

    Label_vae_train(beta, logging, train_loader, validation_loader, label_vae, label_vae_ckpt_path, num_epochs, vae_learning_rate)


    test_set = DATA(logging,'test', sequence=1)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)  # batch_size=100
     
    Label_vae_evaluate(beta, logging, test_loader, label_vae)

    