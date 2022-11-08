import logging
import argparse
import os
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pdb

from rl.storage import RolloutStorage
from rl.model import Policy
from rl.vec_env.envs import make_vec_envs
from evaluation import evaluate
from crowd_sim import *


def test(output_dir=None, test_ckpt=None, success_rate=None):
    # the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the model directory that we are testing
	parser.add_argument('--output_dir', type=str, default=output_dir)
	parser.add_argument('--visualize', default=False, action='store_true')
	# model weight file you want to test
	parser.add_argument('--test_ckpt', type=str, default= format(test_ckpt, '05d')+'.pt')
	test_args = parser.parse_args()


	from importlib import import_module
	output_dir_temp = test_args.output_dir
	if output_dir_temp.endswith('/'):
		output_dir_temp = output_dir_temp[:-1]

	# import arguments.py from saved directory
	# if not found, import from the default directory
	try:
		output_dir_string = output_dir_temp.replace('/', '.') + '.arguments'
		model_arguments = import_module(output_dir_string)
		get_args = getattr(model_arguments, 'get_args')  
		algo_args = get_args()
	except:
		print('Failed to get get_args function from ', test_args.output_dir, '/arguments.py')
		pdb.set_trace()

	# import config #class from saved directory
	# if not found, import from the default directory
	try:
		output_dir_string = output_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(output_dir_string)
		Config = getattr(model_arguments, 'Config')
		config = Config
	except:
		print('Failed to get Config function from ', test_args.output_dir, '/configs/config.py')
		pdb.set_trace()


	# configure logging and device

	mode = 'a'
	log_file = os.path.join(test_args.output_dir,'test.log') 

	seed = algo_args.seed

	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
	logging.info('----------------------------------------')
	logging.info('test seed %f', seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False
	

	torch.set_num_threads(1)
	device = torch.device("cuda" if algo_args.cuda else "cpu")
	logging.info('Val success rate : '+str(success_rate))
	logging.info('Restored from checkpoint : '+test_args.test_ckpt)




	if test_args.visualize:
		fig, ax = plt.subplots(figsize=(7, 7))
		if config.sim.test_sim == 'turtlebot':
			ax.set_xlim(-2, 10)
			ax.set_ylim(-6, 6)
		else:
			ax.set_xlim(-6, 6)
			ax.set_ylim(-6, 6)

		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)
	else:
		ax = None
	

	env_name = algo_args.env_name
	recurrent_cell = 'GRU'

	eval_dir = os.path.join(test_args.output_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)


	envs = make_vec_envs(env_name, seed, 1,
						 algo_args.gamma, device, allow_early_resets=True,
						 envConfig=config, ax=ax, phase='test')

	actor_critic = Policy(
		envs.action_space,
		config = config,
		base_kwargs=algo_args,
		base=config.robot.policy)
	actor_critic.base.nenv = 1

	load_path=os.path.join(test_args.output_dir,'checkpoints', test_args.test_ckpt)
 
	if os.path.exists(load_path):
		actor_critic.load_state_dict(torch.load(load_path), strict=True)
		actor_critic.base.nenv = 1
		actor_critic.config = config
		
	else:
		print('Path does not exsits. Type c+enter to continue without loading.')
		pdb.set_trace()
  

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	test_size = config.env.test_size


	rollouts = RolloutStorage(int(config.env.time_limit/config.env.time_step),
								1,
								envs.observation_space.spaces,
								envs.action_space,
								algo_args.rnn_hidden_size,
								recurrent_cell_type=recurrent_cell,
								base=config.robot.policy, encoder_type=config.pas.encoder_type, seq_length=algo_args.seq_length, gridsensor=config.pas.gridsensor)

	test_episode_rewards, success_rate = evaluate(rollouts, config, test_args.output_dir, actor_critic, envs, device, test_size, logging, test_args.visualize, 'test', j=test_ckpt)
	return test_episode_rewards, success_rate

if __name__ == '__main__':     
	test_rewards = []
	test_success = []
  
	output_dir = 'data/pasrl'
	ckpt = [38800, 0.95]                             

	test_episode_rewards, test_success_rate = test(output_dir=output_dir, test_ckpt= ckpt[0] , success_rate=ckpt[1])
	test_rewards.append(test_episode_rewards)
	test_success.append(test_success_rate)
