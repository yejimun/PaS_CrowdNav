import numpy as np
from scipy import *
from crowd_sim.envs.grid_utils import *
from numba import jit
import warnings
warnings.filterwarnings("ignore")


@jit
def generateLabelGrid(ego_dict, sensor_dict, res=0.1):  

    minx = ego_dict['pos'][0] - 5. + res/2.
    miny = ego_dict['pos'][1] - 5. + res/2. 
    maxx = ego_dict['pos'][0] + 5.  
    maxy = ego_dict['pos'][1] + 5. 


    x_coords = np.arange(minx,maxx,res)
    y_coords = np.arange(miny,maxy,res)

    mesh_x, mesh_y = np.meshgrid(x_coords,y_coords)
    x_local = mesh_x
    y_local = mesh_y 

    label_grid = np.zeros((2,x_local.shape[0],x_local.shape[1])) 
    label_grid[1] = np.nan # For unoccupied cell, they remain nan


    for s_id, pos, radius in zip(sensor_dict['id'], sensor_dict['pos'], sensor_dict['r']):
        mask = point_in_circle(x_local, y_local, pos, radius, res)

        # occupied by sensor
        label_grid[0,mask] = 1. 
        label_grid[1,mask] = int(s_id) # does not include ego id

    return label_grid, x_local, y_local