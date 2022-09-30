import numpy as np
import torch
from crowd_sim.envs.utils.info import *
import os


def CollectingStep(args, config, model_dir, data_envs, device, test_size, logging, visualize=False):
    data_total_timesteps = 0
    baseEnv = data_envs.venv.envs[0].env
    device = torch.device("cuda" if args.cuda else "cpu")
    video_dir = model_dir+"/data_video/"

    for k in range(test_size):
        ego_data = [['episode', 'timestamp', 'obs', 'label_grid', 'sensor_grid', 'id_grid', 'mask']]
        done = False
        all_videoGrids = []
        stepCounter = 0
        obs = data_envs.reset()
        
        masks = torch.FloatTensor([[1.0]]).to(device)
        global_time = 0.0
        total_timesteps = int(config.env.time_limit/config.env.time_step)

        
        ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['sensor_grid'].cpu().numpy(),\
            obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()] 
        ego_data.append(ego_step_data)
        data_total_timesteps += 1
        

        while not done:
            with torch.no_grad():
                stepCounter = stepCounter + 1 
                if stepCounter==total_timesteps:
                    break

                    
                if visualize and k<20:       
                    all_videoGrids.append(obs['sensor_grid'][0].cpu().numpy())
                else:
                    all_videoGrids = torch.Tensor([99.])
                
                action = torch.Tensor([-99., -99.])

                # Obser reward and next obs
                obs, rew, done, infos = data_envs.step(action)
                
                if not done:
                    global_time = baseEnv.global_time
    
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(device)
                
                if done and visualize and k<20:
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir) 
                    data_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=video_dir+'data_epi_'+str(k))
                        
                ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['sensor_grid'].cpu().numpy(), \
                    obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()]
                ego_data.append(ego_step_data)
                data_total_timesteps += 1

  
        # if visualize and k<20:
        #     if not os.path.exists(video_dir):
        #         os.makedirs(video_dir) 
        #     data_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=video_dir+'data_epi_'+str(k))
                
            
        ego_data = np.array(ego_data, dtype=object)
        np.save(model_dir+'/epi_'+format(k, '05d')+'_vector', np.array(np.vstack(ego_data[1:,2]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_label_grid', np.array(np.vstack(ego_data[1:,3]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_sensor_grid', np.array(np.vstack(ego_data[1:,4]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_id_grid', np.array(np.vstack(ego_data[1:,5]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_mask', np.array(np.vstack(ego_data[1:,-1]), dtype=np.float32))

        print('Episode', k, 'ends in', stepCounter)    
    
    logging.info('Total data timesteps: (%d,) ',data_total_timesteps)            
    data_envs.close()