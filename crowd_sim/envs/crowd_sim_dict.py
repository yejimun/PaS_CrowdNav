import gym
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import CrowdSim
from crowd_sim.envs.generateLabelGrid import generateLabelGrid
from crowd_sim.envs.generateSensorGrid import generateSensorGrid
import torch
import glob
import imageio
import os
from collections import deque
from copy import deepcopy


class CrowdSimDict(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.desiredVelocity=[0.0,0.0]
        self.robot_states_copy = None
        self.human_states_copy = None

    def set_robot(self, robot):
        self.robot = robot
        """[summary]
        """
        if self.collectingdata:   
            robot_vec_length = 9
        else:
            if self.config.action_space.kinematics=="holonomic":
                robot_vec_length = 4
            else:
                robot_vec_length = 5
        d={}         
        if self.collectingdata:    
            vec_length = robot_vec_length+5*self.human_num
            d['vector'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, vec_length,), dtype = np.float32)                    
            d['label_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, *self.grid_shape), dtype = np.float32) 
            d['sensor_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *self.grid_shape), dtype = np.float32) 
        else:
            d['vector'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, robot_vec_length,), dtype = np.float32)                
            if self.config.pas.seq_flag or self.config.pas.encoder_type != 'cnn' :
                d['grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.pas.sequence, *self.grid_shape), dtype = np.float32) 
                d['label_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, *self.grid_shape), dtype = np.float32) 
            else:
                d['grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *self.grid_shape), dtype = np.float32)


        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    def generate_ob(self, reset):
        ob = {}
        ego_id = 100 # robot_id
        self.dict_update()
        ego_dict, other_dict = self.ego_other_dict(ego_id)
        
        label_grid, x_map, y_map = generateLabelGrid(ego_dict, other_dict, res=self.grid_res)
        map_xy = [x_map, y_map]

        if self.gridsensor == 'sensor' or self.collectingdata:
            visible_id, sensor_grid = generateSensorGrid(label_grid, ego_dict, other_dict, map_xy, self.FOV_radius, res=self.grid_res)
            self.visible_ids.append(visible_id)
        else:
            visible_id = np.unique(label_grid[1])[:-1]
            self.visible_ids.append(visible_id)
        
        
        human_visibility = [True for i in range(self.human_num)]               
        self.xy_local_grid.append([x_map, y_map])


        if self.config.pas.gridtype == 'global' or self.collectingdata:
            ob['vector'] = self.robot.get_full_state_list()
        elif self.config.pas.gridtype == 'local':
            
            # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
            robot_state_np = np.array(self.robot.get_full_state_list()).copy()
            if self.robot.kinematics == 'holonomic':               
                robot_vector = np.zeros(4)
                robot_vector[:2] = robot_state_np[:2]-robot_state_np[5:7]
                robot_vector[2:4] = robot_state_np[2:4]
                ob['vector'] = robot_vector  # (px-gx, py-gy, vx, vy)     
            else: 
                robot_vector = np.zeros(5) 
                robot_vector[:5] = robot_state_np[:5]
                robot_vector[:2] = deepcopy(robot_state_np[:2]-robot_state_np[5:7])
                robot_vector[2] = robot_state_np[-1]                    
                ob['vector'] = robot_vector # unicycle # (px-gx, py-gy, theta, v, w)
            
        
        self.update_last_human_states(human_visibility, reset=reset)
        # For collecting data
        if self.collectingdata:
            # When robot is executed by ORCA
            for i, human in enumerate(self.humans):                    
                # observation for robot. Don't include robot's state
                self.ob = [] 
                for other_human in self.humans:
                    if other_human != human:
                        # Chance for one human to be blind to some other humans
                        if self.random_unobservability and i == 0:
                            if np.random.random() <= self.unobservable_chance or not self.detect_visible(human,
                                                                                                        other_human):
                                self.ob.append(self.dummy_human.get_observable_state())
                            else:
                                self.ob.append(other_human.get_observable_state())
                        # Else detectable humans are always observable to each other
                        elif self.detect_visible(human, other_human):
                            self.ob.append(other_human.get_observable_state())
                        else:
                            self.ob.append(self.dummy_human.get_observable_state())
                            
            ob['vector'].extend(list(np.ravel(self.last_human_states))) # add human states to vector
            ob['grid_xy'] = map_xy
            ob['label_grid'] = label_grid # both the label grid and id grid
            ob['sensor_grid'] = sensor_grid 

        else:    
            if self.gridsensor == 'sensor':
                if self.config.pas.encoder_type != 'cnn' :
                    self.sequence_grid.append(sensor_grid)
                    if len(self.sequence_grid) < self.config.pas.sequence:
                        gd = deepcopy(self.sequence_grid)
                        gd1 = np.stack([gd[0] for i in range(self.config.pas.sequence-len(self.sequence_grid))])
                        stacked_grid = deepcopy(np.concatenate([gd1, gd]))
                    else:
                        stacked_grid = deepcopy(np.stack(self.sequence_grid))
                        
                    ob['label_grid'] = label_grid
                    ob['grid'] = stacked_grid
                else:
                    ob['label_grid'] = label_grid
                    ob['grid'] = sensor_grid
            elif self.gridsensor == 'gt' :
                if self.config.pas.seq_flag or self.config.pas.encoder_type != 'cnn' :
                    self.sequence_grid.append(label_grid[0])
                    if len(self.sequence_grid) < self.config.pas.sequence:
                        gd = deepcopy(self.sequence_grid)
                        gd1 = np.stack([gd[0] for i in range(self.config.pas.sequence-len(self.sequence_grid))])
                        stacked_grid = deepcopy(np.concatenate([gd1, gd]))
                    else:
                        stacked_grid = deepcopy(np.stack(self.sequence_grid))                            
                    ob['grid'] = stacked_grid   
                    ob['label_grid'] = label_grid                 
                    
                else:
                    ob['grid'] = label_grid[0]
        return ob



    def dict_update(self,):
        """[summary]
        Updates the current state dictionary (self.robot_dict, self.human_dict)
        For creating label/sensor grid
        """
        human_id = []
        human_pos = []
        human_v = []
        human_a = []
        human_radius = []
        human_theta = []
        human_goal = []

        for i, human in enumerate(self.humans):
            theta = human.theta 
            human_v.append([human.vx, human.vy])
            human_pos.append([human.px, human.py])
            human_radius.append(human.radius)
            human_goal.append([human.gx, human.gy])
            human_a.append([human.ax, human.ay])
            human_id.append(i)
            human_theta.append(theta)

        robot_pos = np.array([self.robot.px,self.robot.py])
        robot_theta = self.robot.theta
        robot_radius = self.robot.radius
        robot_v = np.array([self.robot.vx, self.robot.vy])
        robot_goal = np.array([self.robot.gx, self.robot.gy])
        robot_a = np.array([self.robot.ax, self.robot.ay])

        keys = ['id','pos', 'v', 'a', 'r', 'theta', 'goal']
        self.robot_values = [100, robot_pos, robot_v, robot_a, robot_radius, robot_theta, robot_goal]
        self.robot_dict = dict(zip(keys, self.robot_values))

        self.human_values = [human_id, human_pos, human_v, human_a, human_radius, human_theta, human_goal]
        self.humans_dict = dict(zip(keys, self.human_values))


    def ego_other_dict(self, ego_id):
        """[summary]
        For creating label/sensor grid
        Ego can be either the robot or one of the pedestrians
        Args:
            ego_id (int): 100 for robot and 0~(n-1) for n pedestrians
        Returns:
            [type]: [description]
        """
        keys = ['id','pos', 'v', 'a', 'r', 'theta', 'goal']
        if ego_id == 100: # When the ego is robot
            ego_dict = self.robot_dict
            other_dict = self.humans_dict
        else: # When the ego is a pedestrian
            human_values = np.array(copy.copy(self.human_values) ,dtype=object)
            ego_dict = dict(zip(keys, human_values[:,ego_id].tolist()))

            other_values = np.delete(human_values, ego_id, 1)  # deleting the ego info from the pedestrians' info
            robot_values = np.reshape(self.robot_values, [other_values.shape[0],-1])
            other_values = np.hstack((other_values, robot_values)) # combining robot and other pedestrians info (for grid w/ robot)
            other_dict = dict(zip(keys, other_values))
        return ego_dict, other_dict

    def update_robot_goals(self,):
        ## Update robot goal from generated goal list to promote exploration during collecting data (policy: ORCA)
        step = int(self.global_time/self.time_step)
        self.robot.gx = self.robot_goals[step][0]
        self.robot.gy = self.robot_goals[step][1]

    def reset(self, ):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.visible_ids = []
        self.Con = None
        self.cbar = None

        if self.phase is not None:
            phase = self.phase

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0    

        self.all_videoGrids = []  # Store the FOV grid for rendering
        self.xy_local_grid = [] # Store the FOV grid coordinate for rendering

        if self.collectingdata and self.sim == 'crosswalk': 
            # During the data collection, episodes terminate when there's no reference agent (e.g. both reference agent reached their goals in crosswalk scenario)
            self.ref_goal_reaching = [False, False]
        
        self.sequence_grid = deque(maxlen=self.config.pas.sequence)
        self.states = list()
        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]


        # get robot observation
        ob = self.generate_ob(reset=True)
        if self.robot.kinematics == 'unicycle':    
            ob['vector'][3:5] =  np.array([0,0])

        # initialize potential
        self.potential = - abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))
        return ob


    def step(self, action):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if np.all(action==[-99., -99.]) or np.all(action==[-99.]): 
            # Input dummy action values for data collection. To use ORCA planner for robot 
            # Robot's action or state is actually not important for data collection because we only train AE with pedestrian infos.
            with torch.no_grad():
                action = self.robot.act(self.ob)  
        else:
            robot_v_prev = np.array([self.robot.vx, self.robot.vy])
            action = self.robot.policy.clip_action(action, self.robot.v_pref, robot_v_prev, self.time_step) # Use previous action to clip
        
            if self.robot.kinematics == 'unicycle':
                self.desiredVelocity[0] = np.clip(self.desiredVelocity[0]+action.v,-self.robot.v_pref,self.robot.v_pref)
                self.desiredVelocity[1] = action.r
                action=ActionRot(self.desiredVelocity[0], self.desiredVelocity[1])          

        human_actions = self.get_human_actions()

        # ! currently reward is calculated according to the current human and robot state, not the updated state with current action?
        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)
        
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        
        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step

        # compute the observation
        ob = self.generate_ob(reset=False)
        if self.robot.kinematics == 'unicycle':    
            ob['vector'][3:5] =  np.array(action)        

        info={'info':episode_info}

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()
            
        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human, i=i)
        return ob, reward, done, info    
    


    def render(self, mode='video', all_videoGrids=99., output_file='data/my_model/eval.mp4'):
        """[summary]

        Args:
            mode (str, optional): Haven't updated the 'humna mode'. Use the video mode to plot the FOV grids. Defaults to 'video'.
            all_videoGrids (list, optional): list of FOV grids for the whole episode. Defaults to 99..
            output_file (str, optional): video path/filename  Defaults to 'data/my_model/eval.mp4'.

        Returns:
            [type]: [description]
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches
        if all_videoGrids == 99.:
            pass
        else:
            self.all_videoGrids = all_videoGrids
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        robot_color = '#FFD300' #'yellow'
        goal_color = 'yellow'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode is None: # to pass all_videoGrids to render_traj()
            pass
        elif mode == 'human':
            def calcFOVLineEndPoint(ang, point, extendFactor):
                # choose the extendFactor big enough
                # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
                FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                    [np.sin(ang), np.cos(ang), 0],
                                    [0, 0, 1]])
                point.extend([1])
                # apply rotation matrix
                newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
                # increase the distance between the line start point and the end point
                newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
                return newPoint


            ax=self.render_axis
            artists=[]

            # add goal
            goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX,robotY=self.robot.get_position()

            robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            artists.append(robot)

            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper left')


            # compute orientation in each step and add arrow to show the direction
            radius = self.robot.radius
            arrowStartEnd=[]

            robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

            arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

            for i, human in enumerate(self.humans):
                theta = np.arctan2(human.vy, human.vx)
                arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

            arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                    for arrow in arrowStartEnd]
            for arrow in arrows:
                ax.add_artist(arrow)
                artists.append(arrow)


            # draw FOV for the robot
            # add robot FOV
            if self.robot_fov < np.pi * 2:
                FOVAng = self.robot_fov / 2
                FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
                FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


                startPointX = robotX
                startPointY = robotY
                endPointX = robotX + radius * np.cos(robot_theta)
                endPointY = robotY + radius * np.sin(robot_theta)

                # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
                # the start point of the FOVLine is the center of the robot
                FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
                FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
                FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
                FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

                ax.add_artist(FOVLine1)
                ax.add_artist(FOVLine2)
                artists.append(FOVLine1)
                artists.append(FOVLine2)

            if all_videoGrids == 99.:
                pass
            else:
                self.Con = plt.contourf(self.xy_local_grid[-1][0],self.xy_local_grid[-1][1], self.all_videoGrids[-1][0], cmap='BuPu',alpha=0.8, levels=np.linspace(0, 2, 9)) #, cmap='coolwarm'
                self.cbar = plt.colorbar(self.Con) 
            

            # add humans and change the color of them based on visibility
            human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]

            for i in range(len(self.humans)):
                ax.add_artist(human_circles[i])
                artists.append(human_circles[i])

                # green: visible; red: invisible
                if self.detect_visible(self.robot, self.humans[i], robot1=True):
                    human_circles[i].set_color(c='g')
                else:
                    human_circles[i].set_color(c='r')
                plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(i), color='black', fontsize=12)

            plt.pause(0.1)
            for item in artists:
                item.remove() # there should be a better way to do this. For example,
                # initially use add_artist and draw_artist later on
            for t in ax.texts:
                t.set_visible(False)
            plt.savefig(output_file+'_'+format(int(self.global_time/self.time_step), '03d')+'.png')

        elif mode == 'video':
            from matplotlib import animation
            import itertools
            self.visible_ids = np.array(self.visible_ids)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            # FOV = plt.Circle(robot_positions[0], 3, fill=False, color='grey',  linestyle='--')
            
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            # ax.add_artist(FOV)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper left')

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            
                    
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])
                if i in list(self.visible_ids[0]):
                    human.set_color(c='blue') # green if seen in the current timestep
                else:
                    human.set_color(c='r') # red if not seen in the current timestep


            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)


            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            plt.savefig(output_file+'_'+format(0, '03d')+'.png')

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal humans
                nonlocal human_numbers
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                # FOV.center = robot_positions[frame_num]
                for human in humans:
                    human.remove()
                for txt in human_numbers:
                    txt.set_visible(False)
                
                if frame_num >= self.config.pas.sequence:
                    sequence = self.config.pas.sequence
                else:
                    sequence = frame_num+1 # frame_num= 0,1,2,3
                for j in range(sequence):     # frame_num-j >=0 always
                    if j ==0:
                        alpha = 1
                        humans = [plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                for i, human in enumerate(self.states[frame_num][1])]
                        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
                    elif j ==1:
                        alpha = 0.4
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-1][1])])
                    elif j ==2:
                        alpha = 0.3          
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-2][1])])          
                    elif j == 3:
                        alpha = 0.1
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-3][1])])
                    end_frame = frame_num-j                                 
                    
                    
                    for i, human in enumerate(humans[self.human_num*j:self.human_num*(j+1)]):
                        ax.add_artist(human)                
                    
                        # plt.text(self.human_states_copy[frame_num-j][i][0]+0.3, self.human_states_copy[frame_num-j][i][1]+0.3, str((frame_num-j)*self.time_step), fontsize=14, color='black', ha='center', va='center')
                        # Observation history with colored pedestrians
                        if self.config.pas.gridsensor == 'sensor':
                            if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
                                if frame_num-j<self.config.pas.sequence:
                                    start_frame = 0
                                else:
                                    start_frame = frame_num-j - self.config.pas.sequence+1
                                if frame_num-j == 0:
                                    end_frame = 1
                                    
                            
                                past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:end_frame])))
                                if i in list(self.visible_ids[frame_num-j]): # green if seen in current timestep.
                                    human.set_color(c='blue')

                                else:
                                    human.set_color(c='r')
                            else:
                                if i in list(self.visible_ids[frame_num-j]):
                                    human.set_color(c='blue') # green if seen in current timestep.
                                else:
                                    human.set_color(c='r') # red if unseen in current timestep.
                        else: # all agents are observable
                            human.set_color(c='blue') # green if seen in current timestep.

                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                 
                if self.Con!=None:
                    for coll in self.Con.collections:
                        ax.collections.remove(coll)
                if self.cbar!= None:
                    self.cbar.remove()
                    

                if all_videoGrids == 99.:
                    pass
                else:
                    if np.any(self.all_videoGrids[frame_num] != None):
                        self.Con = plt.contourf(self.xy_local_grid[frame_num][0],self.xy_local_grid[frame_num][1], self.all_videoGrids[frame_num][0], cmap='binary',alpha=0.8, levels=np.linspace(0, 1, 9)) #, cmap='coolwarm' 'BuPu'
                        self.cbar = plt.colorbar(self.Con) 
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                plt.savefig(output_file+'_'+format(frame_num, '03d')+'.png')
                
            def getint(name):
                basename = name.partition('.')
                phase, epi_num, termination, num = basename.split('_')
                return int(num)
            

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()           

            # # Save to mp4
            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000, repeat=False)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file+'.mp4', writer=writer)
            else:
                pass
                # plt.show()
            
            # # # # combine saved images to gif
            filenames = glob.glob(output_file+'*.png')
            filenames.sort() # sort by timestep
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
                os.remove(filename)
            # for i in range(10):
            #     images.append(imageio.imread(last_filename))
            imageio.mimsave(output_file+'.gif', images, fps=5)
            plt.close()
            
    
    def render_traj(self, path, episode_num=0):
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines
        import itertools
        import os
        
        self.Con = None
        self.cbar = None
        arrows = []
        x_offset = 0.11
        y_offset = 0.11
        # Save current live figure number
        curr_fig = plt.gcf().number
        plt.figure(figsize=(7, 7))
        

        # Set Constants
        cmap = plt.cm.get_cmap('jet', self.human_num)
        robot_color = '#FFD300'

        # Set Axes
        plt.tick_params(labelsize=16)
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.xlabel('x(m)', fontsize=16)
        plt.ylabel('y(m)', fontsize=16)
        ax = plt.axes()
        
        
        # # Begin drawing
        for k in range(len(self.states)-1): # all_videoGrids does not have the last OGM    
            # Save current live figure number
            curr_fig = plt.gcf().number
            plt.figure(figsize=(7, 7))            

            # Set Constants
            cmap = plt.cm.get_cmap('jet', self.human_num)
            robot_color = '#FFD300'
            arrow_color = 'red'
            arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

            # Set Axes
            plt.tick_params(labelsize=16)
            plt.xlim(-6, 6)
            plt.ylim(-6, 6)
            plt.xlabel('x(m)', fontsize=16)
            plt.ylabel('y(m)', fontsize=16)
            ax = plt.axes()
                    
         
            robot = plt.Circle((self.states[k][0].px,self.states[k][0].py), self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            theta = np.arctan2(self.states[k][0].vy, self.states[k][0].vx)
            orientation = [[((self.states[k][0].px,self.states[k][0].py), (self.states[k][0].px + self.states[k][0].radius * np.cos(theta),
                                self.states[k][0].py + self.states[k][0].radius * np.sin(theta)))]]
        
            if k >= self.config.pas.sequence:
                sequence = self.config.pas.sequence
            else:
                sequence = k+1 # k= 0,1,2,3
            for j in range(sequence):     # k-j >=0 always
                if j ==0:
                    alpha = 1
                elif j ==1:
                    alpha = 0.4
                elif j ==2:
                    alpha = 0.3                    
                elif j == 3:
                    alpha = 0.1
                end_frame = k-j                 
                humans = [plt.Circle((human.px, human.py), human.radius, fill=False, color=cmap(i), linewidth=1.5, alpha=alpha)
                            for i, human in enumerate(self.states[end_frame][1])]
                
                for i, human in enumerate(humans):
                    ax.add_artist(human)                
                
                    # Observation history with colored pedestrians
                    if self.config.pas.gridsensor == 'sensor':
                        if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
                            if k-j<self.config.pas.sequence:
                                start_frame = 0
                            else:
                                start_frame = k-j - self.config.pas.sequence+1
                            if k-j == 0:
                                end_frame = 1                                
                        
                            past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:end_frame])))
                            if i in list(self.visible_ids[k-j]): # green if seen in current timestep.
                                human.set_color(c='blue')
                            elif i in list(np.unique(past_vis_ids)): # red if the not seen in past sequence either.
                                human.set_color(c='magenta')
                            else:
                                human.set_color(c='r')
                        else:
                            if i in list(self.visible_ids[k-j]):
                                human.set_color(c='blue') # green if seen in current timestep.
                            else:
                                human.set_color(c='r') # red if unseen in current timestep.
                    else: # all agents are observable
                        human.set_color(c='blue') # green if seen in current timestep.
                    

                        
                if j == 0:                        
                    human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                            color='black', fontsize=12) for i in range(len(self.humans))]
            for h in range(self.human_num):
                ax.add_artist(human_numbers[h])
                agent_state = copy.deepcopy(self.states[k][1][h])
                theta = np.arctan2(agent_state.vy, agent_state.vx)
                orientation.append([((agent_state.px, agent_state.py), (agent_state.px + agent_state.radius * np.cos(theta),
                                    agent_state.py + agent_state.radius * np.sin(theta)))])

            for arrow in arrows:
                arrow.remove()
            arrows = [patches.FancyArrowPatch(*orient[0], color=arrow_color, arrowstyle=arrow_style)
                    for orient in orientation]
            
            for arrow in arrows:
                ax.add_artist(arrow)

                
            # # (i) Drawing for some time intervals
            # # Draw circle for robot and humans every 7 timesteps and at the end of episode
            # # plt.contourf(self.xy_local_grid[k][0],self.xy_local_grid[k][1], self.all_videoGrids[k].cpu().numpy()[0], cmap='binary',alpha=0.8, levels=np.linspace(0, 1, 9)) #, cmap='coolwarm'
            # # plt.colorbar(self.Con)

        
            # # # Draw robot and humans' goals
            # goal = mlines.Line2D([self.states[k][0].gx], [self.states[k][0].gy], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')
            # ax.add_artist(goal)
            # # human_goals = [human.get_goal_position() for human in self.humans]
            # # for i, point in enumerate(human_goals):
            # #     if not self.humans[i].isObstacle:
            # #         curr_goal = mlines.Line2D([point[0]], [point[1]], color='black', markerfacecolor=cmap(i), marker='*', linestyle='None', markersize=15)
            # #         ax.add_artist(curr_goal)

            # # # Draw robot and humans' start positions
            # # for pos in [robot_start_pos] + human_start_poses:
            # #     plt.text(pos[0], pos[1], 'S', fontsize=14, color='black', ha='center', va='center')

            # # Set legend
            # plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # # Save trajectory for current episode
            # new_fig = plt.gcf().number
            # plt.savefig(os.path.join(path, str(episode_num) + '_traj'+'_'+str(k*self.time_step*100)+'.png'), dpi=300)

            # # Close trajectory figure and switch back to live figure
            # plt.close(new_fig)
            # plt.figure(curr_fig)  
                
        
        # ## (ii) Drawing the whole episode in one fig
        self.Con = None
        self.cbar = None
        # Save current live figure number
        curr_fig = plt.gcf().number
        plt.figure(figsize=(7, 7))
        

        # Set Constants
        cmap = plt.cm.get_cmap('jet', self.human_num)
        robot_color = '#FFD300'

        # Set Axes
        plt.tick_params(labelsize=16)
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.xlabel('x(m)', fontsize=16)
        plt.ylabel('y(m)', fontsize=16)
        ax = plt.axes()
        
        
        # Get robot and humans' start positions
        robot_start_pos = [self.states[0][0].px, self.states[0][0].py] #self.robot_states_copy[0]
        human_start_poses = [[human.px, human.py] for human in self.states[0][1]]  #self.human_states_copy[0]       
        
        
        for k in range(len(self.states)-1): # all_videoGrids does not have the last OGM 
            # Draw circle for robot and humans every 7 timesteps and at the end of episode
            if k % 16 == 0 or k == len(self.states) - 1:
                robot = plt.Circle((self.states[k][0].px,self.states[k][0].py), self.robot.radius, fill=True, color=robot_color) 
                # plt.text(self.states[k][0].px+0.3, self.states[k][0].py+0.3, str(k*self.time_step), fontsize=14, color='black', ha='center', va='center')
                humans = [plt.Circle((human.px, human.py), 0.1, fill=False, color=cmap(i), linewidth=1.5) # human.radius
                            for i, human in enumerate(self.states[k][1])]
                ax.add_artist(robot)
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    
                    if self.config.pas.gridsensor == 'sensor':
                        if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
                            if k<self.config.pas.sequence:
                                start_frame = 0
                            else:
                                start_frame = k - self.config.pas.sequence+1
                            past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:k])))
                            if i in list(self.visible_ids[k]): # green if seen in current timestep.
                                human.set_color(c='blue')
                            elif i in list(np.unique(past_vis_ids)): # red if the not seen in past sequence either.
                                human.set_color(c='magenta')
                            else:
                                human.set_color(c='r')
                        else:
                            if i in list(self.visible_ids[k]):
                                human.set_color(c='blue') # green if seen in current timestep.
                            else:
                                human.set_color(c='r') # red if unseen in current timestep.
                    else: # all agents are observable
                        human.set_color(c='blue') # green if seen in current timestep.        

            # Draw lines for trajectory every step of the episode for all agents
            if k != 0:
                nav_direction = plt.Line2D((self.states[k-1][0].px, self.states[k][0].px),
                                            (self.states[k-1][0].py, self.states[k][0].py),
                                            color=robot_color, ls='solid')
                human_directions = [plt.Line2D((self.states[k-1][1][i].px, self.states[k][1][i].px),
                                                (self.states[k-1][1][i].py, self.states[k][1][i].py),
                                                color=cmap(i), ls='solid')
                                    for i in range(self.human_num)]
                ax.add_artist(nav_direction)
                for human_direction in human_directions:
                    ax.add_artist(human_direction)

        # Draw robot and humans' goals
        goal = mlines.Line2D([self.states[k][0].px], [self.states[k][0].py], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
      
        # Set legend
        plt.legend([robot, goal], ['Robot', 'Goal'], loc='upper right', fontsize=16)

        # Save trajectory for current episode
        new_fig = plt.gcf().number
        plt.savefig(os.path.join(path, str(episode_num) + '_traj.png'), dpi=300)

        # Close trajectory figure and switch back to live figure
        plt.close(new_fig)
        plt.figure(curr_fig)

            



