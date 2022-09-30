import numpy as np

from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY


class PASRNN(Policy):
	def __init__(self, config):
		super().__init__(config)
		self.time_step = self.config.env.time_step # Todo: is this needed?
		self.name = 'pas_rnn'
		self.trainable = True
		self.multiagent_training = True
		self.onedim_action = self.config.robot.onedim_action
		self.time_step = config.env.time_step


	# clip the self.raw_action and return the clipped action
	def clip_action(self, raw_action, v_pref, prev_v, time_step, a_pref=1.):
		"""
		Input state is the joint state of robot concatenated by the observable state of other agents

		To predict the best action, agent samples actions and propagates one step to see how good the next state is
		thus the reward function is needed
		"""
		# quantize the action
		holonomic = True if self.config.action_space.kinematics == 'holonomic' else False
		# clip the action
		if holonomic:
			raw_action = np.array(raw_action)      
			# clip acceleration
			a_norm = np.linalg.norm(raw_action-prev_v) 
			a_norm = np.linalg.norm((raw_action-prev_v))
			if a_norm > a_pref:
				v_action = np.zeros(2)
				raw_ax = raw_action[0]-prev_v[0] 
				raw_ay = raw_action[1]-prev_v[1] 
				v_action[0] = (raw_ax / a_norm * a_pref)*time_step + prev_v[0]
				v_action[1] = (raw_ay / a_norm * a_pref)*time_step + prev_v[1]	
			else:
				v_action = raw_action		
			
			# clip velocity				
			v_norm = np.linalg.norm(v_action)
			if v_norm > v_pref:
				v_action[0] = v_action[0] / v_norm * v_pref
				v_action[1] = v_action[1] / v_norm * v_pref
			return ActionXY(v_action[0], v_action[1])
		else:
			# for sim2real   
			raw_action[0] = np.clip(raw_action[0], -0.1, 0.1) # action[0] is change of v
			raw_action[1] = np.clip(raw_action[1], -0.25, 0.25) # action[1] is change of theta

			return ActionRot(raw_action[0], raw_action[1])


