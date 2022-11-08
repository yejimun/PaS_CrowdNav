import numpy as np
from scipy import *
import copy
from numba import jit, njit
import random
import numba
import pdb



# IS metric.
def MapSimilarityMetric(pas_map, sensor_grid, label_grid):    
    # Making a hard binary grids    
    gt_map = label_grid
    pas_map = np.where(pas_map>=0.6, 1., pas_map)
    pas_map = np.where(pas_map<=0.4, 0., pas_map)
    pas_map = np.where(np.logical_and(pas_map!=1., pas_map!=0.), 0.5, pas_map)

    A = copy.deepcopy(pas_map)
    B =copy.deepcopy(gt_map)
    base_A = copy.deepcopy(sensor_grid)
    
    psi_occupied, psi_free, psi_occluded = computeSimilarityMetric(B, A)
    base_psi_occupied, base_psi_free, base_psi_occluded = computeSimilarityMetric(B, base_A)
    psi_sum = psi_occupied + psi_free + psi_occluded
    base_psi_sum = base_psi_occupied + base_psi_free + base_psi_occluded
        
    psi = [psi_sum, psi_occupied, psi_free, psi_occluded ]
    base_psi = [base_psi_sum, base_psi_occupied, base_psi_free, base_psi_occluded]
        
    return psi, base_psi

def toDiscrete(m):
    """
    Args:
        - m (m,n) : np.array with the occupancy grid
    Returns:
        - discrete_m : thresholded m
    """
    m_occupied = np.zeros(m.shape)
    m_free = np.zeros(m.shape)
    m_occluded = np.zeros(m.shape)

    m_occupied[m == 1.0] = 1.0
    m_occluded[m == 0.5] = 1.0
    m_free[m == 0.0] = 1.0

    return m_occupied, m_free, m_occluded

def todMap(m):

    """
    Extra if statements are for edge cases.
    """

    y_size, x_size = m.shape
    dMap = np.ones(m.shape) * np.Inf
    dMap[m == 1] = 0.0

    for y in range(0,y_size):
        if y == 0:
            for x in range(1,x_size):
                h = dMap[y,x-1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(0,x_size):
                if x == 0:
                    h = dMap[y-1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y,x-1]+1, dMap[y-1,x]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    for y in range(y_size-1,-1,-1):

        if y == y_size-1:
            for x in range(x_size-2,-1,-1):
                h = dMap[y,x+1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(x_size-1,-1,-1):
                if x == x_size-1:
                    h = dMap[y+1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y+1,x]+1, dMap[y,x+1]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    return dMap

def computeDistance(m1,m2):

    y_size, x_size = m1.shape
    dMap = todMap(m2)

    d = np.sum(dMap[m1 == 1])
    num_cells = np.sum(m1 == 1)

    # If either of the grids does not have a particular class,
    # set to x_size + y_size (proxy for infinity - worst case Manhattan distance).
    # If both of the grids do not have a class, set to zero.
    if ((num_cells != 0) and (np.sum(dMap == np.Inf) == 0)):
        output = d/num_cells
    elif ((num_cells == 0) and (np.sum(dMap == np.Inf) != 0)):
        output = 0.0
    elif ((num_cells == 0) or (np.sum(dMap == np.Inf) != 0)):
        output = x_size + y_size

    if output == np.Inf:
        pdb.set_trace()

    return output

def computeSimilarityMetric(m1, m2):

    m1_occupied, m1_free, m1_occluded = toDiscrete(m1)
    m2_occupied, m2_free, m2_occluded = toDiscrete(m2)

    occupied = computeDistance(m1_occupied,m2_occupied) + computeDistance(m2_occupied,m1_occupied)
    occluded = computeDistance(m2_occluded, m1_occluded) + computeDistance(m1_occluded, m2_occluded)
    free = computeDistance(m1_free,m2_free) + computeDistance(m2_free,m1_free)

    return occupied, free, occluded


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


# Get state of a vehicle at timestamp when its id is known
def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]



# reshape list
def reshape(seq, rows, cols):
    return [list(u) for u in zip(*[iter(seq)] * cols)]
    
# helper function from pykitti
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

# helper function from pykitti
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

# helper function from pykitti
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],s
                     [s,  c,  0],
                     [0,  0,  1]])

# helper function from pykitti
def pose_from_oxts_packet(lat,lon,alt,roll,pitch,yaw,scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

# position of vertex relative to global origin (adapted from pykitti)
def pose_from_GIS(lat,lon,scale,origin):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.d
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    # 2D position
    t = np.array([tx, ty])

    return (t-origin[0:2])

# helper function from pykitti
def transform_from_rot_trans(R, t):
    """Homogeneous transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit 
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean) 
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D    



def point_in_rectangle(x, y, rectangle):
    A = rectangle[0]
    B = rectangle[1]
    C = rectangle[2]

    M = np.array([x,y]).transpose((1,2,0))

    AB = B-A
    AM = M-A # nxnx2
    BC = C-B
    BM = M-B # nxnx2

    dotABAM = np.dot(AM,AB) # nxn
    dotABAB = np.dot(AB,AB)
    dotBCBM = np.dot(BM,BC) # nxn
    dotBCBC = np.dot(BC,BC)

    return np.logical_and(np.logical_and(np.logical_and((0. <= dotABAM), (dotABAM <= dotABAB)), (0. <= dotBCBM)), (dotBCBM <= dotBCBC)) # nxn

# create a grid in the form of a numpy array with coordinates representing
# the middle of the cell (30 m ahead, 30 m to each side, and 30 m behind)
# cell resolution: 0.33 cm

def global_grid(origin,endpoint,res):

    xmin = min(origin[0],endpoint[0]) 
    xmax = max(origin[0],endpoint[0]) + res/2.
    ymin = min(origin[1],endpoint[1]) 
    ymax = max(origin[1],endpoint[1]) + res/2. 

    x_coords = np.arange(xmin,xmax,res)
    y_coords = np.arange(ymin,ymax,res)

    gridx,gridy = np.meshgrid(x_coords,y_coords)

    return gridx, np.flipud(gridy) 

def point_in_circle(x_local, y_local, center, radius, res):
    mask = np.sqrt(np.power(x_local-center[0],2)+ np.power(y_local-center[1],2)) < radius  
    return mask


def find_nearest(n,v,v0,vn,res): 
    "Element in nd array closest to the scalar value `v`" 
    idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
    return idx

# generate the y indeces along a line
def linefunction(velx,vely,indx,indy,x_range):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*x_range + b 

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    for i in range(max_iterations):
        s = data[np.random.choice(data.shape[0], 3, replace=False), :]
        m = estimate(s)
        ic = 0
        for j in range(data.shape[0]):
            if is_inlier(m, data[j,:]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def is_close(a,b,c,d,point,distance=0.1):
    D = (a*point[:,0]+b*point[:,1]+c*point[:,2]+d)/np.sqrt(a**2+b**2+c**2)
    return D

