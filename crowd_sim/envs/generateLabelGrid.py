import numpy as np
from scipy import *
from crowd_sim.envs.grid_utils import *
from numba import jit
import warnings
warnings.filterwarnings("ignore")


@jit
def generateLabelGrid(ego_dict, sensor_dict, res=0.1, invisible_id=[None]):  

    minx = ego_dict['pos'][0] - 5. + res/2.
    miny = ego_dict['pos'][1] - 5. + res/2. 
    maxx = ego_dict['pos'][0] + 5.  
    maxy = ego_dict['pos'][1] + 5. 


    x_coords = np.arange(minx,maxx,res)
    y_coords = np.arange(miny,maxy,res)

    mesh_x, mesh_y = np.meshgrid(x_coords,y_coords)
    pre_local_x = mesh_x
    pre_local_y = mesh_y 

    xy_local = np.vstack((pre_local_x.flatten(), pre_local_y.flatten())).T
    x_local = xy_local[:,0].reshape(mesh_x.shape)
    y_local = xy_local[:,1].reshape(mesh_y.shape)

    label_grid = np.zeros((2,x_local.shape[0],x_local.shape[1])) 
    label_grid[1] = np.nan # For unoccupied cell, they remain nan


    for s_id, pos, radius in zip(sensor_dict['id'], sensor_dict['pos'], sensor_dict['r']):
        if s_id not in invisible_id:
            mask = point_in_circle(x_local, y_local, pos, radius, res)

            # occupied by sensor
            label_grid[0,mask] = 1. 
            label_grid[1,mask] = int(s_id)

    return label_grid, x_local, y_local, pre_local_x, pre_local_y