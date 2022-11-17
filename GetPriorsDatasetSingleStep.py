import argparse
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


from softgym.envs.rope_knot import RopeKnotEnv

from softgym.utils.topology import InvalidTopology, RopeTopology,find_topological_path
# from softgym.utils.topology import get_topological_representation, generate_random_topology

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from softgym.utils.normalized_env import normalize

import random
import csv
import sys
import os

from typing import Tuple,List
import cv2


trefoil_knot = RopeTopology(np.array([
        [ 0, 1, 2, 3, 4, 5],
        [ 3, 4, 5, 0, 1, 2],
        [ 1,-1, 1,-1, 1,-1],
        [-1,-1,-1,-1,-1,-1]
    ],dtype=np.int32))


def get_arc(start,end,sign:bool,num_points:int):
    # start and end are x,y pairs
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    xc = start[0] + dx/2
    yc = start[1] + dy/2

    theta_s = np.arctan2(start[1]-yc,start[0]-xc) + np.pi
    theta_e = np.arctan2(end[1]-yc,end[0]-xc) + np.pi

    if sign:
        s = theta_s
        e = theta_e
    else:
        s = min(theta_s,theta_e)
        e = max(theta_s,theta_e) - 2*np.pi

    theta = np.linspace(s,e,num_points) 
    r = np.linalg.norm(end-start)/2
    x = r*np.cos(theta) + xc
    y = r*np.sin(theta) + yc

    coords = np.vstack([x,y]).T
    if np.any(coords[0,:] - start > 1e-9):
        coords = coords[::-1,:]

    return coords


def topo_to_geometry_add_C(topo:RopeTopology,action) -> Tuple[int,np.ndarray,np.ndarray]:
    p = topo.geometry
    mid_region = np.zeros((1,2))

    over_seg,under_seg,sign,under_first = action
    if over_seg == under_seg:
        segment_idxs = topo.find_geometry_indices_matching_seg(over_seg)
        if len(segment_idxs) == 1:
            segment_idxs.append(segment_idxs[0])
        l = len(segment_idxs)
        if under_first:
            under_indices = segment_idxs[:l//2]
            pick_idxs = segment_idxs[l//2:]
            # pick_idx = segment_idxs[-1]
        else:
            under_indices = segment_idxs[l//2:]
            pick_idxs = segment_idxs[:l//2]
            # pick_idx = segment_idxs[0]
        diameter = p[under_indices,:]

        place_region = np.vstack([get_arc(diameter[0,[0,2]],diameter[-1,[0,2]],sign > 0,100),diameter[::-1,[0,2]]])
        mid_region = np.vstack([get_arc(diameter[0,[0,2]],diameter[-1,[0,2]],sign < 0,100),diameter[::-1,[0,2]]])
        pick_region = p[pick_idxs,:][:,[0,2]]
    
    else:
        over_idxs = topo.find_geometry_indices_matching_seg(over_seg)
        under_idxs = topo.find_geometry_indices_matching_seg(under_seg)

        if over_seg in [0,topo.size]:
            if over_seg == 0:
                # pick_idx = 0
                pick_idxs = over_idxs
            elif over_seg == topo.size:
                # pick_idx = over_idxs[-1]
                pick_idxs = under_idxs
            pick_region = p[pick_idxs,:][:,[0,2]]
            place_region_segs = topo.get_loop(under_seg,sign)
            place_region_idxs = []
            for s in place_region_segs:
                place_region_idxs.extend(topo.find_geometry_indices_matching_seg(s.segment_num))
            place_region = p[place_region_idxs,:][:,[0,2]]

        else:
            # l_pick = len(over_idxs)
            # pick_idx = over_idxs[l_pick//2]
            pick_idxs = over_idxs
            pick_region = p[pick_idxs,:][:,[0,2]]

            l_place = len(under_idxs)
            place_region = p[l_place//2,[0,2]].reshape([1,2])

    return random.choice(pick_idxs),pick_region,mid_region,place_region

def topo_to_geometry_remove_C(topo,action,obs):
    p = np.vstack([np.zeros((1,2)),obs['shape'].T])
    seg = action[1][0]
    if seg == 0:
        other_seg = 1
    elif seg == topo.size:
        other_seg = topo.size-1
    if seg not in [0,topo.size]:
        raise ValueError("Segment must be one of the two end segments of the rope.")
    
    undo_idxs = topo.find_geometry_indices_matching_seg(other_seg,obs["cross"])
    pick_idxs = topo.find_geometry_indices_matching_seg(seg,obs["cross"])

    # place_idx = undo_idxs[len(undo_idxs)//2]
    # place_region = p[place_idx,:].reshape([1,2])

    return random.choice(pick_idxs),p[pick_idxs,:],None,p[undo_idxs,:]

def all_add_C(topo:RopeTopology) -> List[np.ndarray]:
    added = []
    for over_seg in range(topo.size+1):
        for under_seg in range(topo.size+1):
            for sign in [-1,1]:
                if over_seg in [0,topo.size] or under_seg in [0,topo.size]:
                    for under_first in ([False,True] if over_seg == under_seg else [False]):
                        try:
                            action_args = [over_seg,under_seg,sign,under_first]
                            test, _ = topo.add_C(*action_args)
                            added.append(action_args)
                        except InvalidTopology:
                            pass
    return added

def get_action(env:RopeKnotEnv,obs,done:bool):
    if done:
        obs = env.reset
    geoms = env.get_geoms(True)
    t = get_topological_representation(geoms).astype(np.int32)

    g_t,topo_action = random.choice(all_add_C(t))

    pick_idx,mid_region,place_region = topo_to_geometry_add_C(t,topo_action,obs)

    p = np.vstack([np.zeros((1,2)),obs['shape'].T])

    place_pos = np.mean(place_region,axis=0)
    if mid_region is None:
        mid_region = ((p[pick_idx,:] + place_pos)/2).reshape([1,2])
    mid_pos = np.mean(mid_region,axis=0)

    pick_norm = pick_idx/(p.shape[0]-1)
    delta_mid = mid_pos - p[pick_idx,:2].reshape([1,2])
    delta_end = place_pos - p[pick_idx,:2].reshape([1,2])

    env_action = [pick_norm,*delta_mid.flatten().tolist(),*delta_end.flatten().tolist()]

    return obs,topo_action,env_action,g_t

def get_action_masks(frame,pick_region:np.ndarray,mid_region:np.ndarray,place_region:np.ndarray):
    assert pick_region.shape[0] == 2, f'Matrix must be 2xN'
    assert mid_region.shape[0] == 2, f'Matrix must be 2xN'
    assert place_region.shape[0] == 2, f'Matrix must be 2xN'

    pick_h = np.vstack([pick_region,np.ones(pick_region.shape[1])])
    mid_h = np.vstack([mid_region,np.ones(mid_region.shape[1])])
    place_h = np.vstack([place_region,np.ones(place_region.shape[1])])

    pick_img_p = (homography @ pick_h)[:2,:]
    mid_img_p = (homography @ mid_h)[:2,:]
    place_img_p = (homography @ place_h)[:2,:]
    
    pick_frame = cv2.polylines(
        np.zeros(frame.shape[:2],dtype=np.uint8),
        [pick_img_p.T.astype(np.int32)],
        isClosed=False,
        color=255
    )
    mid_frame = cv2.fillPoly(
        np.zeros(frame.shape[:2],dtype=np.uint8),
        [mid_img_p.T.astype(np.int32)],
        color=255
    )
    place_frame = cv2.fillPoly(
        np.zeros(frame.shape[:2],dtype=np.uint8),
        [place_img_p.T.astype(np.int32)],
        color=255
    )

    return pick_frame, mid_frame, place_frame

def save_images(root_path:str,img_num:int,plain_img:np.ndarray,pick_img:np.ndarray,mid_img:np.ndarray,place_img:np.ndarray):
    plain_path = os.path.join(root_path,"plain",f"{img_num}.jpg")
    pick_path = os.path.join(root_path,"pick",f"{img_num}.jpg")
    mid_path = os.path.join(root_path,"mid",f"{img_num}.jpg")
    place_path = os.path.join(root_path,"place",f"{img_num}.jpg")

    cv2.imwrite(plain_path,plain_img)
    cv2.imwrite(pick_path,pick_img)
    cv2.imwrite(mid_path,mid_img)
    cv2.imwrite(place_path,place_img)

    return plain_path,pick_path,mid_path,place_path

w,h,s = 128,128,0.35
homography,_ = cv2.findHomography(
    np.array([
        [ s,-s,-s, s],
        [-s,-s, s, s],
        [ 1, 1, 1, 1],

    ]).T,
    np.array([
        [0,w,w,0],
        [0,0,h,h],
        [1,1,1,1],
    ]).T
)
def main(env_kwargs,all_args):
    env_kwargs["camera_width"] = w
    env_kwargs["camera_height"] = h

    os.makedirs(os.path.join(all_args.save_name,'plain'),exist_ok=all_args.start != 0)
    os.makedirs(os.path.join(all_args.save_name,'pick'),exist_ok=all_args.start != 0)
    os.makedirs(os.path.join(all_args.save_name,'mid'),exist_ok=all_args.start != 0)
    os.makedirs(os.path.join(all_args.save_name,'place'),exist_ok=all_args.start != 0)

    if all_args.num_workers == 1:
        envs = RopeKnotEnv(**env_kwargs)
        envs.reset()
    else:
        envs = SubprocVecEnv([lambda: RopeKnotEnv(goal_topology=trefoil_knot.rep,**env_kwargs)]*all_args.num_workers,'spawn')
        envs.env_method("reset")
    data = {
        "plain_path":[],
        "pick_path":[],
        "mid_path":[],
        "place_path":[],
        "Successful":[],
        "pick_region":[],
        "mid_region":[],
        "place_region":[],
        "topology":[],
        "Topo_actions":[],
    }
    

    for step_num in tqdm(range(all_args.total_steps//all_args.num_workers)):
        env_actions = []
        topo_actions = [None]*all_args.num_workers
        while True:
            try:
                if all_args.num_workers == 1:
                    envs.generate_env_variation()
                    rope_reps = [envs.get_topological_representation()]
                else:
                    envs.env_method('generate_env_variation')
                    rope_reps = envs.env_method("get_topological_representation")
            except:
                with open(os.path.join(all_args.save_name,"Errors.pkl"),"ab") as f:
                    pickle.dump(envs.get_geoms(),f)
                continue
            break
        if all_args.num_workers == 1:
            before_render = [envs.render_no_gripper()]
        else:
            before_render = envs.env_method("render_no_gripper")
                  
        for worker_num in range(all_args.num_workers):

            topo_actions[worker_num] = random.choice(all_add_C(rope_reps[worker_num]))
            pick_idx, pick_region,mid_region,place_region = topo_to_geometry_add_C(rope_reps[worker_num],topo_actions[worker_num])
            action_masks = get_action_masks(before_render[worker_num],pick_region.T,mid_region.T,place_region.T)

            pick_norm = pick_idx/rope_reps[worker_num].rope_length
            mid_pos = np.mean(mid_region,axis=0)
            place_pos = np.mean(place_region,axis=0)

            pick_loc = rope_reps[worker_num].geometry[pick_idx,[0,2]]
            pick_act = pick_norm
            mid_act = mid_pos - pick_loc
            place_act = place_pos - pick_loc

            env_actions.append([pick_act,*mid_act.flatten().tolist(),*place_act.flatten().tolist()])

            plain_path,pick_path,mid_path,place_path = save_images(
                all_args.save_name,
                all_args.start + step_num*all_args.num_workers + worker_num,
                before_render[worker_num],
                *action_masks
            )

            data["plain_path"].append(plain_path)
            data["pick_path"].append(pick_path)
            data["mid_path"].append(mid_path)
            data["place_path"].append(place_path)
            
            data["pick_region"].append(pick_region)
            data["mid_region"].append(mid_region)
            data["place_region"].append(place_region)

        if all_args.num_workers == 1:
            envs.step(env_actions[0])
        else:
            envs.step(env_actions)

        try:
            if all_args.num_workers == 1:
                new_reps = [envs.get_topological_representation()]
            else:
                new_reps = envs.env_method("get_topological_representation")

            successes = [None] * all_args.num_workers
            for worker_num in range(all_args.num_workers):
                successes[worker_num] = new_reps[worker_num] == rope_reps[worker_num].add_C(*topo_actions[worker_num])[0]
        except:
            
            with open(os.path.join(all_args.save_name,"Errors.pkl"),"ab") as f:
                pickle.dump(envs.get_geoms(),f)

        data["Topo_actions"].extend(topo_actions)
        data["topology"].extend(rope_reps)

        



    with open(os.path.join(all_args.save_name,"Data.pkl"),"ab") as pkl_file:
        pickle.dump(data,pkl_file)
    envs.close()

 



# ------------- Helper functions ----------------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--save_name',type=str,default='./Datasets/TEMP',help='The directory to place generated models.')
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')

    # Environment options
    parser.add_argument('--num_variations', type=int, default=500, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=9e9,help='The length of each episode.')
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    parser.add_argument('--maximum_crossings',type=int,default=5,help='The maximum number of crossings for topological representations. Any representation exceeding this will be clipped down.')
    parser.add_argument('--goal_crossings',type=int,default=1,help='The number of crossings used for the goal configuration.')
    parser.add_argument('--total_steps',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument("--start",type=int,default=0)

    args = parser.parse_args()    
    args.render_mode = args.render_mode.lower()

    assert args.num_workers > 0, f'num_workers must be set to a positive integer. You entered {args.num_workers}.'
    assert args.horizon > 0, f'Horizon length must be a positive integer. You entered {args.horizon}.'
    assert args.render_mode in ('cloth','particle','both'), f'Render_mode must be from the set {{cloth, particle, both}}. You entered {args.render_mode}.'
    assert args.start >= 0, f'start must be a non-negative integer'

    return args

if __name__ == '__main__':
    args = get_args()

    env_kwargs = {
        'observation_mode'  : 'key_point',
        'action_mode'       : 'picker_trajectory',
        'num_picker'        : 1,
        'render'            : True,
        'headless'          : args.headless,
        'horizon'           : args.horizon,
        'action_repeat'     : 1,
        'render_mode'       : args.render_mode,
        'num_variations'    : args.num_variations,
        'use_cached_states' : True,#True,
        'save_cached_states': False,
        'deterministic'     : False,
        # 'trajectory_funcs'  : [box_trajectory],
        'maximum_crossings' : args.maximum_crossings,
        'goal_crossings'    : args.goal_crossings,
    }

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    main(env_kwargs,args)


