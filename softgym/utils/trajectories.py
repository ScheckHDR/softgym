import numpy as np
from math import cos, sin, pi
from copy import deepcopy

def simple_trajectory(waypoints,height=0.05,num_points_per_leg = 300):

    start = deepcopy(waypoints[0,:])
    end = deepcopy(waypoints[-1,:])
    waypoints[:,1] = height

    traj = [generate_linear_trajectory(start,waypoints[0,:],num_points=num_points_per_leg)]
    for i in range(1,waypoints.shape[0]):
        leg = generate_linear_trajectory(waypoints[i-1,:],waypoints[i,:],num_points=num_points_per_leg)
        traj.append(leg)
    traj.append(generate_linear_trajectory(waypoints[-1,:],end,num_points=num_points_per_leg))
    return np.concatenate(traj)

def curved_trajectory(pick_loc,place_loc,num_points=1000,height=0.05,rot_dir=1):

    if len(place_loc) == 2:
        place_loc = np.insert(place_loc,1,0)

    vertical_displacement = np.array([0,height,0])

    offset = ((pick_loc - place_loc)/2)
    offset[1] = 0

    mid_point = place_loc + offset
    mid_point[1] = height

    traj = []
    traj.append(generate_linear_trajectory(pick_loc,pick_loc+vertical_displacement,num_points=num_points/3))

    for theta in np.linspace(0,pi,num_points,endpoint=True):
        rot = np.array([
            [cos(theta*rot_dir),0,-sin(theta*rot_dir)],
            [0,0,0],
            [sin(theta*rot_dir),0, cos(theta*rot_dir)]
        ])
        traj_point = mid_point + np.dot(rot,offset) 
        traj.append(np.expand_dims(traj_point,0))
    traj.append(generate_linear_trajectory(place_loc+vertical_displacement,place_loc,num_points=num_points/3))

    return np.concatenate(traj)




def generate_linear_trajectory(start_pos,end_pos,num_points=1000):   
    return np.linspace(start_pos,end_pos,int(num_points),endpoint=True,dtype=float)
