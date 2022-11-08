import numpy as np
import math
from scipy import *
from crowd_sim.envs.grid_utils import *
import warnings
warnings.filterwarnings('ignore')


############## Sensor_grid ##################
# 0:  empty
# 1 : occupied
# 0.5 :  unknown/occluded
#############################################


# Find the unknown cells using polygons for faster computation. (Result similar to ray tracing)
def generateSensorGrid(label_grid, ego_dict, ref_dict, map_xy, FOV_radius, res=0.1):
	x_local, y_local = map_xy
	
	center_ego = ego_dict['pos']
	occluded_id = []
	visible_id = []

	# get the maximum and minimum x and y values in the local grids
	x_shape = x_local.shape[0]
	y_shape = x_local.shape[1]	

	id_grid = label_grid[1].copy()


	unique_id = np.unique(id_grid) # does not include ego (robot) id
   
	# cells not occupied by ego itself
	mask = np.where(label_grid[0]!=2, True,False)

	# no need to do ray tracing if no object on the grid
	if np.all(label_grid[0,mask]==0.):
		sensor_grid = np.zeros((x_shape, y_shape))
	
	else:
		sensor_grid = np.zeros((x_shape, y_shape)) 

		ref_pos = np.array(ref_dict['pos'])
		ref_r = np.array(ref_dict['r'])

		# Find the cells that are occluded by the obstructing human agents
		# reorder humans according to their distance from the robot.
		distance = [np.linalg.norm(center-center_ego) for center in ref_pos]
		sort_indx = np.argsort(distance)
	
		unchecked_id = np.array(ref_dict['id'])[sort_indx]
		# Create occlusion polygons starting from closest humans. Reject humans that are already inside the polygons.
		for center, human_radius, h_id in zip(ref_pos[sort_indx], ref_r[sort_indx], unchecked_id):	
			# if human is already occluded, then just pass
			if h_id in occluded_id:
				continue

			hmask = (label_grid[1,:,:]==h_id)
			sensor_grid[hmask] = 1.

			alpha = math.atan2(center[1]-center_ego[1], center[0]-center_ego[0])
			theta = math.asin(np.clip(human_radius/np.sqrt((center[1]-center_ego[1])**2 + (center[0]-center_ego[0])**2), -1., 1.))
			
			# 4 or 5 polygon points
			# 2 points from human
			x1 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha-theta)
			y1 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha-theta)

			x2 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha+theta)
			y2 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha+theta)

			# Choose points big/small enough to cover the region of interest in the grid
			if x1 <= center_ego[0]:
				x3 = -12. 
			else:
				x3 = 12. 
			y3 = linefunction(center_ego[0],center_ego[1],x1,y1,x3)
			if x2 <= center_ego[0]:
				x4 = -12. 
			else:
				x4 = 12. 
			y4 = linefunction(center_ego[0],center_ego[1],x2,y2,x4)

			polygon_points = np.array([[x1, y1], [x2, y2], [x4, y4],[x3, y3]])
			grid_points = np.array([x_local.flatten(), y_local.flatten()])	


			occ_mask = parallelpointinpolygon(grid_points.T, polygon_points)
			occ_mask = occ_mask.reshape(x_local.shape)
			sensor_grid[occ_mask] = 0.5

			# check if any agent is fully inside the polygon
			for oid in unchecked_id:
				oid_mask = (label_grid[1,:,:]==oid)
				# if any agent is fully inside the polygon store in the occluded_id and opt from unchecked_id			
				if np.all(sensor_grid[oid_mask] == 0.5):
					occluded_id.append(oid)
					unchecked_id = np.delete(unchecked_id, np.where(unchecked_id==h_id))

	# Set cells out side of field of view as unknown
	FOVmask = point_in_circle(x_local, y_local, ego_dict['pos'], FOV_radius, res) 
	sensor_grid[np.invert(FOVmask)] = 0.5
 
	for id in unique_id:
		mask1 = (label_grid[1,:,:]==id)
		if np.any(sensor_grid[mask1] == 1.):
			sensor_grid[mask1] = 1. 
			visible_id.append(id)
	
		else:
			pass

	return visible_id, sensor_grid 

