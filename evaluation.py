import numpy as np
import torch
import os
import pdb

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.grid_utils import MapSimilarityMetric


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
    

def evaluate(rollouts, config, model_dir, actor_critic,eval_envs, device, test_size, logging, visualize=False,
             phase=None, j=None):

    eval_episode_rewards = []

    if actor_critic is not None:
        actor_critic.base.eval()
    
    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    chc_total = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    gamma = 0.99
    baseEnv = eval_envs.venv.envs[0].env    
    
    total_similarity = []
    total_occupied_similarity = []
    total_free_similarity = []
    total_occluded_similarity = []
    total_base_similarity = []
    total_base_occupied_similarity = []
    total_base_free_similarity = []
    total_base_occluded_similarity = []
    

    for k in range(test_size):
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        all_videoGrids = []

        obs = eval_envs.reset()
            
        if rollouts is not None:
            rollouts.reset()
            if isinstance(obs, dict):
                for key in obs:
                    rollouts.obs[key][0].copy_(obs[key])
            else:
                rollouts.obs[0].copy_(obs)

            rollouts.to(device)

            
            eval_recurrent_hidden_states = {}
            for key in rollouts.recurrent_hidden_states:
                eval_recurrent_hidden_states[key] = rollouts.recurrent_hidden_states[key][stepCounter]
        
        global_time = 0.0
        path = 0.0
        chc = 0.0

    
        last_pos = obs['vector'][0, 0, :2].cpu().numpy()  # robot px, py
        if config.action_space.kinematics == 'unicycle':
            last_angle = obs['vector'][0, 0, 2].cpu().numpy() 
                

        
        while not done:         
            with torch.no_grad():
                if rollouts is not None:
                    masks = rollouts.masks[stepCounter]               

                if config.robot.policy == 'pas_rnn':
                    value, action, action_log_prob, eval_recurrent_hidden_states, decoded = actor_critic.act(obs, eval_recurrent_hidden_states, masks, deterministic=True, visualize=visualize)
                else: # if robot's policy is ORCA 
                    action = torch.Tensor([-99., -99.])

                if phase == 'test' and config.pas.encoder_type == 'vae':                  
                    gt_grid = obs['label_grid'][:,0] 
                    
                    pas_grid = decoded.squeeze(0).squeeze(0).cpu().numpy() 
                    sensor_grid = obs['grid'][:,-1].squeeze(0).squeeze(0).cpu().numpy()
                    label_grid = gt_grid.squeeze(0).squeeze(0).cpu().numpy()
                    similarity, base_similarity = MapSimilarityMetric(pas_grid, sensor_grid, label_grid)
                    
                    total_similarity.append(similarity[0])
                    total_occupied_similarity.append(similarity[1])
                    total_free_similarity.append(similarity[2])
                    total_occluded_similarity.append(similarity[3])
                    
                    total_base_similarity.append(base_similarity[0])
                    total_base_occupied_similarity.append(base_similarity[1])
                    total_base_free_similarity.append(base_similarity[2])
                    total_base_occluded_similarity.append(base_similarity[3])                       


            if not done:
                global_time = baseEnv.global_time
                
            if visualize:
                if config.robot.policy == 'pas_rnn':
                    if config.pas.encoder_type == 'vae' and config.pas.gridsensor == 'sensor':
                        all_videoGrids.append(decoded.squeeze(0).squeeze(1).cpu().numpy())
                    else:
                        all_videoGrids.append(obs['grid'][:,-1].cpu().numpy())
                        
                else:
                    all_videoGrids = torch.Tensor([99.])

            obs, rew, done, infos = eval_envs.step(action)

            path = path + np.linalg.norm(np.array([last_pos[0] - obs['vector'][0, 0, 0].cpu().numpy(),last_pos[1] - obs['vector'][0, 0, 1].cpu().numpy()]))


            if config.action_space.kinematics == 'unicycle':
                chc = chc + abs(obs['vector'][0, 0, 2].cpu().numpy() - last_angle)
            last_angle = obs['vector'][0, 0, 2].cpu().numpy() 

            last_pos = obs['vector'][0, 0, :2].cpu().numpy()  
            

            rewards.append(rew)


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]

            # If done then clean the history of observations.
            mask = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done]).cuda()


            if config.robot.policy=='pas_rnn' and config.pas.encoder_type != 'cnn':
                masks = torch.cat([masks[ :, 1:], mask],-1)
            else:
                masks = mask

            if rollouts is not None:
                rollouts.insert(obs, eval_recurrent_hidden_states, action,
                                action_log_prob, value, rew, masks)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])                    
            
            if done and visualize:        
                video_dir = model_dir+"/"+phase+"_render/"
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir) 
                    
                output_file=video_dir+str(j)+'_'+'eval_epi'+str(k)+'_'+str(infos[0]['info'])
                
                if phase == 'val':
                    eval_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=output_file) 
                else:
                    eval_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=output_file)  # mode='video'  or  mode=None for render_traj          
            
            stepCounter = stepCounter + 1            
           
        if phase=='test':
            print('')
            print('Reward={}'.format(episode_rew))              
            print('Episode', k, 'ends in', stepCounter+1)
        

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            if phase=='test':
                print('Success')
            path_lengths.append(path)
            if config.action_space.kinematics == 'unicycle':
                chc_total.append(chc)
            
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            if phase=='test':
                print('Collision')
        elif isinstance(infos[0]['info'], Timeout):  
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            if phase=='test':
                print('Time out')
        else:
            print(infos[0]['info'])
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]).item())
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size

    extra_info = ''
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time (mean/var): {:.2f}/{:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, np.mean(success_times), np.var(success_times),
                np.average((cumulative_rewards))))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                    too_close * baseEnv.robot.time_step / total_time, avg_min_dist)

    if phase == 'test' and config.pas.encoder_type != 'cnn':

        avg_occupied_smiliarity = average(total_occupied_similarity)
        avg_free_similarity = average(total_free_similarity)
        avg_occluded_similarity  = average(total_occluded_similarity)
        avg_base_occupied_smiliarity = average(total_base_occupied_similarity)
        avg_base_free_similarity = average(total_base_free_similarity)
        avg_base_occluded_similarity  = average(total_base_occluded_similarity)
            
        avg_similarity = average(total_similarity)
        avg_base_similarity = average(total_base_similarity)
        
        logging.info(
            '{:<5} {}has image similarity(pas/sensor): {:.3f}/{:.3f}'.
                format(phase.upper(), extra_info, avg_similarity, avg_base_similarity))
        if len(similarity) > 1:
            logging.info(
            ' occupied image similarity(pas/sensor): {:.3f}/{:.3f} and free image similarity(pas/sensor): {:.3f}/{:.3f} and occluded image similarity(pas/sensor): {:.3f}/{:.3f} '.
                format(avg_occupied_smiliarity, avg_base_occupied_smiliarity, avg_free_similarity, avg_base_free_similarity, avg_occluded_similarity, avg_base_occluded_similarity))

    logging.info(
        '{:<5} {}has average path length (mean/var): {:.2f}/{:.2f}'.
            format(phase.upper(), extra_info, np.mean(path_lengths) , np.var(path_lengths)))
    if config.action_space.kinematics == 'unicycle':
        chc_total.append(chc)
        logging.info(
        '{:<5} {}has average rotational radius (mean/var): {:.2f}/{:.2f}'.
            format(phase.upper(), extra_info, np.mean(chc_total) , np.var(chc_total)))
        
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return eval_episode_rewards, success_rate
