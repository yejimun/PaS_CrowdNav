import gym
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import *


class CrowdSim(gym.Env):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None # a Robot instance representing the robot
        self.humans = None # a list of Human instances, representing all humans in the environment
        self.global_time = None

        # reward function
        self.success_reward = None
        self.timeout_penalty = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None

        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        #seed
        self.thisSeed=None # the seed will be set when the env is created

        #nenv
        self.nenv=None # the number of env will be set when the env is created.
        # Because the human crossing cases are controlled by random seed, we will calculate unique random seed for each
        # parallel env.

        self.phase=None # set the phase to be train, val or test
        # self.test_case=None # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # for render
        self.render_axis=None

        self.humans = []

        self.potential = None


    def configure(self, config):
        self.config = config
        
        self.collectingdata = config.sim.collectingdata
        if self.collectingdata:
            self.ref_goal_reaching = [False, False]
        self.sim = config.sim.train_val_sim
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes

        self.success_reward = config.reward.success_reward
        self.timeout_penalty = config.reward.timeout_penalty
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor

        self.grid_res = config.pas.grid_res
        self.gridsensor = config.pas.gridsensor
        self.gridtype = config.pas.gridtype
        self.FOV_radius = config.robot.FOV_radius

        self.grid_shape = [100, 100] 
            
        minx, miny, maxx, maxy = -6+self.grid_res/2, -6+self.grid_res/2, 6, 6
        x_coords = np.arange(minx,maxx,self.grid_res)
        y_coords = np.arange(miny,maxy,self.grid_res)
        global_x, global_y = np.meshgrid(x_coords,y_coords)
        self.global_xy = [global_x, global_y]

        if self.config.humans.policy == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': self.config.env.val_size,
                              'test': self.config.env.test_size}
            self.circle_radius = config.sim.circle_radius
            self.human_num = config.sim.human_num

        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV


        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, 'humans')
        # if a human is not in view, set its state to 
        self.dummy_human.set(15, 15, 0, 0, 0, 0, 0) 
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, 'robot')
        self.dummy_robot.set(15, 15, 0, 0, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = ORCA(config)


        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.humans.goal_change_chance

        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.humans.end_goal_change_chance

        # configure randomized radii changing when reaching goals
        self.random_radii = config.humans.random_radii

        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.humans.random_v_pref

        # configure randomized goal changing of humans after reaching their respective goals
        self.random_unobservability = config.humans.random_unobservability
        if self.random_unobservability:
            self.unobservable_chance = config.humans.unobservable_chance


        self.last_human_states = np.zeros((self.human_num, 5))

        # configure randomized policy changing of humans every episode
        self.random_policy_changing = config.humans.random_policy_changing

        # set robot for this envs
        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)
        return


    def set_robot(self, robot):
        raise NotImplementedError


    def generate_random_human_position(self, human_num):
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        while len(self.humans) < human_num:
            self.humans.append(self.generate_circle_crossing_human())


    # return a static human in env
    # position: (px, py) for fixed position, or None for random position
    def generate_circle_static_obstacle(self, position=None):
        # generate a human with radius = 0.3, v_pref = 1, visible = True, and policy = orca
        human = Human(self.config, 'humans')
        # For fixed position
        if position:
            px, py = position
        # For random position
        else:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (np.random.random() - 0.5) * v_pref
                py_noise = (np.random.random() - 0.5) * v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                                    norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

        # make it a static obstacle
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human
        

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        trial = 0
        while True:
            # if we have difficulty inserting a new human, remove & re-initialize all humans
            if trial > 100:
                del self.humans[:]
                self.humans = []

            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 0. if human.v_pref == 0 else human.v_pref
            px_noise = np.random.uniform(-v_pref/2, v_pref/2)
            py_noise = np.random.uniform(-v_pref/2, v_pref/2)
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle':
                    min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
            
            trial = trial + 1

        gx_noise = np.random.uniform(-v_pref/2, v_pref/2)
        gy_noise = np.random.uniform(-v_pref/2, v_pref/2)
        gx = -px + gx_noise
        gy = -py + gy_noise

        human.set(px, py, gx, gy, 0, 0, 0)
        return human
    

    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15., 15., 0., 0., 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx, vy, r = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, r])



    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num
       
        if self.sim == "circle_crossing":                
            while True:
                # Make the goal closer to the center so that the robot encounters obstructed agents more often
                start_circle = 5
                px, py = np.random.uniform(-start_circle, start_circle, 2)
                gx, gy = 0, 0 
                if np.linalg.norm([px - gx, py - gy]) >= 6:
                    break
            vx = np.random.uniform(-self.config.robot.v_pref, self.config.robot.v_pref)
            vy = np.sqrt(self.config.robot.v_pref**2-vx**2)
            self.robot.set(px, py, gx, gy, vx, vy, np.pi/2)

            # generate humans
            self.generate_random_human_position(human_num=human_num)
        
        else:
            raise NotImplementedError




    def reset(self,):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        
        if self.phase is not None:
            phase = self.phase
        # if self.test_case is not None:
        #     test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0
        if self.collectingdata:
            self.ref_goal_reaching = [False, False]

        self.states = list()
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        self.generate_robot_humans(phase)        

        # print(self.case_size[phase])
        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get current observation
        ob = self.generate_ob(reset=True)

        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))


        return ob


    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if self.sim=='circle_crossing':
                    humans_copy = []
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)
                    # Produce valid goal for human in case of circle setting
                    while True:
                        angle = np.random.random() * np.pi * 2
                        # add some noise to simulate all the possible cases robot could meet with human
                        v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                        gx_noise = (np.random.random() - 0.5) * v_pref
                        gy_noise = (np.random.random() - 0.5) * v_pref
                        gx = self.circle_radius * np.cos(angle) + gx_noise
                        gy = self.circle_radius * np.sin(angle) + gy_noise
                        collide = False

                        for agent in [self.robot] + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                        if not collide:
                            break

                    # Give human new goal
                    human.gx = gx
                    human.gy = gy
                    
                else:
                    raise NotImplementedError
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human, i=None):
        if self.sim=='circle_crossing':
            # Update human's goals randomly
            if np.random.random() <= self.end_goal_change_chance:
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

                # Update human's radius now that it's reached goal
                if self.random_radii:
                    human.radius += np.random.uniform(-0.1, 0.1)

                # Update human's v_pref now that it's reached goal
                if self.random_v_pref:
                    human.v_pref += np.random.uniform(-0.1, 0.1)

                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    for agent in [self.robot] + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                    if not collide:
                        break

                # Give human new goal
                human.gx = gx
                human.gy = gy
                
        else:
            raise NotImplementedError

        return

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False


    # for robot:
    # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0
        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids



    # convert an np array with length = 34 to a JointState object
    def array_to_jointstate(self, obs_list):
        fullstate = FullState(obs_list[0], obs_list[1], obs_list[2], obs_list[3],
                              obs_list[4],
                              obs_list[5], obs_list[6], obs_list[7], obs_list[8])

        observable_states = []
        for k in range(self.human_num):
            idx = 9 + k * 5
            observable_states.append(
                ObservableState(obs_list[idx], obs_list[idx + 1], obs_list[idx + 2],
                                obs_list[idx + 3], obs_list[idx + 4]))
        state = JointState(fullstate, observable_states)
        return state


    def last_human_states_obj(self):
        '''
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        '''
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans

    # find R(s, a)
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')

        danger_dists = []
        collision = False

       
        if len(self.states)>0:
            prev_robot_state, prev_humans_state = self.states[-1]
        for i, human in enumerate(self.humans): 
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius
            
            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
                
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius
        if self.global_time >= self.time_limit - 1:
            if self.timeout_penalty == None:
                reward = 0.
            else:
                reward = self.timeout_penalty

            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True


            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            reward =  (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(dmin)

        else:
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = 2 * (-abs(potential_cur) - self.potential) 
            self.potential = -abs(potential_cur)


            done = False
            episode_info = Nothing()

        # This code is for turtlebot experiment. Comment out for simulation
        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == 'unicycle' and not self.collectingdata:
            # add a rotational penalty
            r_spin = -2 * action.r**2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.
            reward = reward + r_spin + r_back                

        return reward, done, episode_info



    # compute the observation
    def generate_ob(self, reset):
        visible_human_states, num_visible_humans, human_visibility = self.get_num_human_in_fov()
        self.update_last_human_states(human_visibility, reset=reset)
        
        if self.robot.policy.name in ['pas_rnn']:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS))

            ob.extend(list(np.ravel(self.last_human_states)))
            ob = np.array(ob)

        else: 
            ob = self.last_human_states_obj()

        return ob

    def get_human_actions(self):
        # step all humans        
        human_actions = []  # a list of all humans' actions        
         
        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in self.humans:
                if other_human != human:
                    # Chance for one human to be blind to some other humans
                    if self.random_unobservability and i == 0:
                        if np.random.random() <= self.unobservable_chance or not self.detect_visible(human,
                                                                                                    other_human):
                            ob.append(self.dummy_human.get_observable_state())
                        else:
                            ob.append(other_human.get_observable_state())
                    # Else detectable humans are always observable to each other
                    elif self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                # Chance for one human to be blind to robot
                if self.random_unobservability and i == 0:
                    if np.random.random() <= self.unobservable_chance or not self.detect_visible(human, self.robot):
                        ob += [self.dummy_robot.get_observable_state()]
                    else:
                        ob += [self.robot.get_observable_state()]
                # Else human will always see visible robots
                elif self.detect_visible(human, self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))
        return human_actions

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        # clip the action to obey robot's constraint
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        # step all humans
        human_actions = self.get_human_actions()


        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)

        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step

        ob = self.generate_ob(reset=False)


        if self.robot.policy.name in [ 'pas_rnn']:
            info={'info':episode_info}
        else: # for orca and sf
            info=episode_info

        if self.end_goal_changing:
            for human in self.humans:
                if not human.isObstacle and human.v_pref != 0 and norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        return ob, reward, done, info

    def render(self, all_videoGrids, mode='video', output_file='data/my_model/eval.mp4'):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
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

            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

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

        elif mode == 'video':
            from matplotlib import animation
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers

            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

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

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
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

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            # if output_file is not None:
            ffmpeg_writer = animation.writers['ffmpeg']
            writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(output_file, writer=writer)
            plt.close()
