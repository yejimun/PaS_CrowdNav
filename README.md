# PaS_CrowdNav
This repository contains the code for our paper titled "Occlusion-Aware Crowd Navigation Using People as Sensors". 
For more details, please refer to our [arXiv preprint](https://arxiv.org/abs/2210.00552)
and [YouTube video](https://youtu.be/BG5s7w5BdME).

<p align="center">
<img src="/figures/Intro_turtlebot.jpg" width="500" />
</p>

## Abstract
Autonomous navigation in crowded spaces poses a challenge for mobile robots due to the highly dynamic, 
partially observable environment. Occlusions are highly prevalent in such settings due to a limited sensor field of view 
and obstructing human agents. Previous work has shown that observed interactive behaviors of human agents can be 
used to estimate potential obstacles despite occlusions. We propose integrating such social inference techniques ("People as Sensors" also known as PaS) into the planning pipeline. We use a variational autoencoder with a specially designed loss function to learn representations that are meaningful for occlusion inference. This work adopts a deep reinforcement learning approach to incorporate the learned representation for occlusion-aware planning. In simulation, our occlusion-aware policy achieves comparable collision avoidance 
performance to fully observable navigation by estimating agents in occluded spaces. We demonstrate successful policy transfer 
from simulation to the real-world Turtlebot 2i. To the best of our knowledge, this work is the first to use social occlusion 
inference for crowd navigation. 

## Method Overview
<p align="center">
<img src="/figures/Algorithm_Structure_Scene4.png" width="800" />
</p>

## Setup
1. Install crowd_sim and crowd_nav into pip
```
pip install -e .
pip install -r requirements.txt
```
2. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
</p>

## Getting started
This repository is organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains code for training and testing the policies. The folder rl contains the code
for the network and PPO algorithm.
Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations: modify `crowd_nav/configs/config.py`.
- For perception level (ground-truth or sensor): set `pas.gridsensor` to `gt` or `sensor`.
- For PaS perception with occlusion inference: set `pas.encoder_type` to `vae` (otherwise `cnn`).
- For a sequential grid (or single) input  : sets `pas.seq_flag` to `True` (or `False`).

2. PPO configurations: modify arguments.py 

### Run the code

1. Collect data for training GT-VAE.
- In `crowd_nav/configs/config.py`, set (i) `robot.policy` to `orca` and (ii) `sim.collectingdata` to `True`
- In `arguments.py`, set `output_dir` to `VAEdata_CircleFOV30/{phase}` where phase is `train` or `val` or `test`
Run the following commands for all three phases.
```
python collect_data.py 
```
2. Pretrain GT-VAE with collected data.
```
python vae_pretrain.py 
```
3. Train policies.
- In `crowd_nav/configs/config.py`, set (i) `robot.policy` to `pas_rnn` and (ii) `sim.collectingdata` to `False`
- In `arguments.py`, set `output_dir` to `data/{foldername}` (i.e. 'data/pas_rnn')
```
python train.py 
```
4. Test policies.
- In `test.py`, set `output_dir` to `data/pasrl`
- In `test.py`, set `ckpt` to [{checkpoint}, {success rate}] (i.e. [38800, 0.95]) 
```
python test.py 
```

5. Plot training curve.
```
python plot.py
```

(We only tested our code in Ubuntu 18.04 with Python 3.8.)

## Results 
**Illustration of our occlusion inference performance.**
The figure shows human agent trajectories in our sequential observation input 
for 1 s and the reconstructed OGMs from our PaS encoding. The observed and occluded humans are denoted as blue and red circles, 
respectively. If an agent is temporarily occluded but has been seen in the past 1 s, it is denoted in magenta. In the OGM, higher 
occupied probability cells are darker in shade. The estimated OGMs from our approach show properties that promote better navigation. 
In (a), the PaS encoding favors estimation of occluded agents that may pose a potential danger to the robot such as an approaching 
agent (humans 2 and 3) rather than human 4 who is moving away at a distance from the robot. In (b), despite fewer observed 
interactions at the boundary of the robot’s FOV, temporarily occluded human 4 is successfully inferred by extrapolating from previous 
observations. In (c), the slow speed of observed human 0 can be used to infer occluded agents ahead like humans 3 and 5. In 
(d), human 3 making an avoidance maneuver (i.e. a turn) provides insight that occluded obstacles like human 5 may be to its left. 
Our algorithm successfully provides insight for better crowd navigation in the presence of occlusion based on observed social behaviors.

<p align="center">
<img src="/figures/PaSOGM.png" width="800" >
</p>

**Collision avoidance of the robot (yellow) with (a) the limited view (baseline), (b) our PaS occlusion inference, and (c) the ground-truth view (oracle).** 
Our approach takes a comparable route to the oracle ground-truth view. While the limited sensor view baseline is highly reactive to unexpected agents resulting in sharp turns, our algorithm reaches the goal using a more efficient and smooth trajectory. 

<p align="center">
<img src="/figures/PaS_CrowdNav_github_video.gif" width="800" >
</p>

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
Y.-J. Mun, M. Itkina, S. Liu, and K. Driggs-Campbell. "Occlusion-Aware Crowd Navigation Using People as Sensors". ArXiv, 2022.
```

## Credits
Part of the code is based on the following repositories:  
[1] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, “Decentralized structural-RNN for robot crowd navigation 
with deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), IEEE, 2021.
(Github:https://github.com/Shuijing725/CrowdNav_Prediction)

[2] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[3] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

