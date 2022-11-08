import logging
import os
import sys
from matplotlib import pyplot as plt
import torch
import shutil


from rl.vec_env.envs import make_vec_envs
from collecting_step import CollectingStep
from crowd_sim import *
from arguments import get_args
from crowd_nav.configs.config import Config

###########
# Things to change in config
# robot.policy = "orca' 
# sim.collectingdata = True
# sim.human_num = 6

# Things to change in arguments
# 'VAEdata/train' or 'VAEdata/val' or 'VAEdata/test'
#############

def main():
    data_args = get_args()

    # save policy to output_dir
    if os.path.exists(data_args.output_dir) and data_args.overwrite: # if I want to overwrite the directory
        shutil.rmtree(data_args.output_dir)  # delete an entire directory tree

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir)


    config = Config()
    data_args = get_args()

    # configure logging and device
    # print data result in log file
    log_file = os.path.join(data_args.output_dir,'data')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file = os.path.join(data_args.output_dir, 'data_visual.log')


    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('robot FOV %f', config.robot.FOV)
    logging.info('humans FOV %f', config.humans.FOV)

    torch.manual_seed(data_args.seed)
    torch.cuda.manual_seed_all(data_args.seed)
    if data_args.cuda:
        if data_args.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


    torch.set_num_threads(1)
    device = torch.device("cuda" if data_args.cuda else "cpu")

    logging.info('Create other envs with new settings')


    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.set_xlabel('x(m)', fontsize=16)
    ax.set_ylabel('y(m)', fontsize=16)


    env_name = data_args.env_name
    phase = data_args.output_dir.split('/')[1]
    print('phase', phase)    

    #! collectingtraindata : When collecting test data, this should be false.
    envs = make_vec_envs(env_name, data_args.seed, 1,
                            data_args.gamma, device, allow_early_resets=True,
                            envConfig=config, ax=ax, phase=phase)


    if phase =='train':
        data_size = 1000  # 500 for turtlebot exp w/ 4 humans
    elif phase =='val':
        data_size = 100 # 50 for turtlebot exp w/ 4 humans
    elif phase =='test':
        data_size = 200 # 100 for turtlebot exp w/ 4 humans

    else:
        raise NotImplementedError


    visualize = True
    
    CollectingStep(data_args, config, data_args.output_dir, envs,  device, data_size, logging, visualize)


if __name__ == '__main__':
    main()
