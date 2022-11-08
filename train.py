import sys
import logging
import os
import shutil
import time
from collections import deque
from rl import utils
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from rl.ppo import PPO
from rl.vec_env.envs import make_vec_envs
from rl.model import Policy
from rl.storage import RolloutStorage
from evaluation import evaluate
from test import test
from crowd_sim import *

import warnings
warnings.filterwarnings("ignore")

def main():

	from arguments import get_args
	algo_args = get_args()

	# save policy to output_dir
	if os.path.exists(algo_args.output_dir) and algo_args.overwrite: # if I want to overwrite the directory
		shutil.rmtree(algo_args.output_dir)  # delete an entire directory tree

	if not os.path.exists(algo_args.output_dir):
		os.makedirs(algo_args.output_dir)

		shutil.copytree('crowd_nav/configs', os.path.join(algo_args.output_dir, 'configs'))
		shutil.copy('arguments.py', algo_args.output_dir)
	from crowd_nav.configs.config import Config
	config = Config()



	# configure logging
	log_file = os.path.join(algo_args.output_dir, 'output.log')
	mode = 'a' # if algo_args.resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")\

	
	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
     
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	device = torch.device("cuda" if algo_args.cuda else "cpu")



	summary_path = algo_args.output_dir+'/runs_gradient'
	if not os.path.exists(summary_path):
		os.makedirs(summary_path)


	# For fastest training: use GRU
	env_name = algo_args.env_name
	recurrent_cell = 'GRU'

	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
						algo_args.gamma, device, False, envConfig=config, phase='train')

	val_envs = make_vec_envs(env_name, algo_args.seed, 1,
						 algo_args.gamma, device, allow_early_resets=True,
						 envConfig=config, phase='val')


	actor_critic = Policy(
		envs.action_space,
		config = config,
		base_kwargs=algo_args,
		base=config.robot.policy)

	if config.robot.policy == 'pas_rnn' or config.robot.policy == 'srnn':
		rollouts = RolloutStorage(algo_args.num_steps,
								algo_args.num_processes,
								envs.observation_space.spaces,
								envs.action_space,
								algo_args.rnn_hidden_size,
								recurrent_cell_type=recurrent_cell,
								base=config.robot.policy, encoder_type=config.pas.encoder_type, seq_length=algo_args.seq_length, gridsensor=config.pas.gridsensor)
		eval_rollouts = RolloutStorage(int(config.env.time_limit/config.env.time_step),
								1,
								envs.observation_space.spaces,
								envs.action_space,
								algo_args.rnn_hidden_size,
								recurrent_cell_type=recurrent_cell,
								base=config.robot.policy, encoder_type=config.pas.encoder_type, seq_length=algo_args.seq_length, gridsensor=config.pas.gridsensor)

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	agent = PPO(
		actor_critic,
		algo_args.clip_param,
		algo_args.ppo_epoch,
		algo_args.num_mini_batch,
		algo_args.value_loss_coef,
		algo_args.entropy_coef,
		PaS_coef = config.pas.PaS_coef,
		lr=algo_args.lr,
		eps=algo_args.eps,
		max_grad_norm=algo_args.max_grad_norm)

	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)


	recurrent_hidden_states = {}
	for key in rollouts.recurrent_hidden_states:
		recurrent_hidden_states[key] = rollouts.recurrent_hidden_states[key][0]


	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes


	for j in range(num_updates): 
		if algo_args.use_linear_lr_decay:
			utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)

		for step in range(algo_args.num_steps):
			with torch.no_grad():
				masks = rollouts.masks[step]

				value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
					obs, recurrent_hidden_states,
					masks)		

			obs, reward, done, infos = envs.step(action) 
			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			mask = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done]).cuda()

			if config.pas.encoder_type =='vae' :
				masks = torch.cat([masks[ :, 1:], mask],-1)
			else:
				masks = mask

			
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks)

		
		with torch.no_grad():
			masks = rollouts.masks[-1]

			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]

				
			next_value = actor_critic.get_value(
				obs, rollouts_hidden_s,
				masks).detach()


		rollouts.compute_returns(next_value, algo_args.gamma,
								 algo_args.gae_lambda)

		
		if config.robot.policy=='pas_rnn' and config.pas.encoder_type !='cnn':
			value_loss, action_loss, dist_entropy, PaS_loss = agent.update(rollouts)


		else:
			value_loss, action_loss, dist_entropy = agent.update(rollouts)
   

		rollouts.after_update()
  

		# save the model for every interval-th episode or for the last epoch
		if (j % algo_args.save_interval == 0
		or j == num_updates - 1) and j!=0:
			save_path = os.path.join(algo_args.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)


			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i'%j + ".pt"))

			## Validation for the saving intervals
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
			visualize=False
			val_episode_rewards, success_rate = evaluate(eval_rollouts, config, algo_args.output_dir, actor_critic, val_envs, device, config.env.val_size, logging, visualize, 'val', j)


			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
								'eprewmean': [np.mean(val_episode_rewards)],'successrate': [success_rate]})

			if os.path.exists(os.path.join(algo_args.output_dir, 'val_progress.csv')) and j > 20:
				df.to_csv(os.path.join(algo_args.output_dir, 'val_progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'val_progress.csv'), mode='w', header=True, index=False)
			actor_critic.base.train()

		if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
			end = time.time()
			logging.info(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, \n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			if config.robot.policy=='pas_rnn' and config.pas.encoder_type =='vae' :

				df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
								'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
								'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
								'loss/value_loss': value_loss, 'loss/PaS_loss': PaS_loss})

			else:
				df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
								'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
								'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
								'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='w', header=True, index=False)


if __name__ == '__main__':
	main()
