class BaseConfig(object):
    def __init__(self):
        pass

class Config(object):
    env = BaseConfig()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500 
    env.randomize_attributes = True # False for turtlebot experiment

    reward = BaseConfig()
    reward.success_reward = 10 
    reward.collision_penalty = -5
    reward.timeout_penalty = None 
    reward.discomfort_dist = 0.25 
    reward.discomfort_penalty_factor = 10 

    sim = BaseConfig()
    sim.collectingdata = False # True 
    sim.train_val_sim ="circle_crossing" 
    sim.test_sim = "circle_crossing" 
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = 6  # 4 for turtlebot experiment

    humans = BaseConfig()
    humans.visible = True
    humans.policy =  "orca"
    humans.radius = 0.3 
    humans.v_pref = 2 # 0.5 for the turtlebot experiment
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False 
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    robot = BaseConfig()
    robot.visible = False 
    # srnn for now
    robot.policy = 'pas_rnn'  #'orca' 
    robot.radius = 0.3
    robot.v_pref = 2 # 0.5 for the turtlebot experiment
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2.
    robot.FOV_radius = 3.0
    robot.limited_path = False  
    robot.onedim_action = False 

    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"  # unicycle for the turtlebot experiment

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15 # 0.25 for the turtlebot experiment
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    
    # config for pas_rnn
    pas = BaseConfig()
    pas.grid_res = 0.1
    pas.gridsensor = 'sensor' #'sensor' or 'gt' 
    pas.gridtype = 'local' 
    pas.sequence = 4 # number of FOV grids stacked for Sensor AE lstm 
    pas.encoder_type = 'vae'  #'vae' or 'cnn'
    pas.PaS_coef = 1. 
    pas.seq_flag = True

