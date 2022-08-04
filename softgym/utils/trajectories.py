import numpy as np
from math import cos, sin, pi

def box_trajectory(pick_loc, place_loc,num_points = 1000,height=0.05):
    vertical_displacement = np.array([0,height,0])
    if len(pick_loc) == 2:
        pick_loc = np.insert(pick_loc,1,0)
    if len(place_loc) == 2:
        place_loc = np.insert(place_loc,1,0)
    traj = np.concatenate([
            generate_linear_trajectory(pick_loc,pick_loc+vertical_displacement,num_points=num_points/3),
            generate_linear_trajectory(pick_loc+vertical_displacement,place_loc+vertical_displacement,num_points=num_points/3),
            generate_linear_trajectory(place_loc+vertical_displacement,place_loc,num_points=num_points/3)
    ])
    return traj

def curved_trajectory(pick_loc,place_loc,num_points=1000,height=0.05,rot_dir=1):
    # if len(pick_loc) == 3:
    #     pick_loc = pick_loc[[0,2]]
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
    # for i,theta in enumerate(np.linspace(0,pi/2,int(num_points/2),endpoint=False)):
    #     rot = np.array([
    #         [cos(theta*rot_dir),-sin(theta*rot_dir)],
    #         [sin(theta*rot_dir), cos(theta*rot_dir)]
    #     ])
    #     traj_point = np.array([0,height_delta*i,0])
    #     traj_point[[0,2]] = mid_point + np.dot(rot,offset) 
    #     traj.append(traj_point)
    
    # for i, theta in enumerate(np.linspace(pi/2,pi,int(num_points/2),endpoint=True)):
    #     rot = np.array([
    #         [cos(theta*rot_dir),-sin(theta*rot_dir)],
    #         [sin(theta*rot_dir), cos(theta*rot_dir)]
    #     ])
    #     traj_point = np.array([0,height-height_delta*i,0])
    #     traj_point[[0,2]] = mid_point + np.dot(rot,offset) 
    #     traj.append(traj_point)

    return np.concatenate(traj)




def generate_linear_trajectory(start_pos,end_pos,num_points=1000):   
    return np.linspace(start_pos,end_pos,int(num_points),endpoint=True,dtype=float)
