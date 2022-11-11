import argparse
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle

from softgym.envs.rope_knot import RopeKnotEnv

from softgym.utils.new_topology_test import InvalidTopology, RopeTopology,find_topological_path
from softgym.utils.topology import get_topological_representation

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from softgym.utils.normalized_env import normalize

import random
import csv
import sys
from os.path import exists

from typing import Tuple,List

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

def topo_to_geometry_add_C(topo,action,obs) -> Tuple[int,np.ndarray,np.ndarray]:
    p = np.vstack([np.zeros((1,2)),obs['shape'].T])
    mid_region = None

    over_seg,under_seg,sign,under_first = action[1]
    if over_seg == under_seg:
        segment_idxs = topo.find_geometry_indices_matching_seg(over_seg,obs['cross'])
        l = len(segment_idxs)
        if under_first:
            under_indices = segment_idxs[:l//2]
            pick_idx = segment_idxs[-1]
        else:
            under_indices = segment_idxs[l//2:]
            pick_idx = segment_idxs[0]
        diameter = p[under_indices,:]

        place_region = np.vstack([get_arc(diameter[0,:],diameter[-1,:],sign > 0,100),diameter[::-1,:]])
        mid_region = np.vstack([get_arc(diameter[0,:],diameter[-1,:],sign < 0,100),diameter[::-1,:]])
    
    else:
        over_idxs = topo.find_geometry_indices_matching_seg(over_seg,obs['cross'])
        under_idxs = topo.find_geometry_indices_matching_seg(under_seg,obs['cross'])

        if over_seg in [0,topo.size]:
            if over_seg == 0:
                pick_idx = 0
            elif over_seg == topo.size:
                pick_idx = over_idxs[-1]
            
            place_region_segs = topo.get_loop(under_seg,sign)
            place_region_idxs = []
            for s in place_region_segs:
                place_region_idxs.extend(topo.find_geometry_indices_matching_seg(s.segment_num,obs['cross']))
            place_region = p[place_region_idxs,:]

        else:
            l_pick = len(over_idxs)
            pick_idx = over_idxs[l_pick//2]

            l_place = len(under_idxs)
            place_region = p[l_place//2,:].reshape([1,2])

    return pick_idx,mid_region,place_region

def topo_to_geometry_remove_C(topo,action,obs):
    p = np.vstack([np.zeros((1,2)),obs['shape'].T])
    seg = action[1][0]
    if seg == 0:
        pick_idx = 0
        other_seg = 1
    elif seg == topo.size:
        pick_idx = p.shape[0]-1
        other_seg = topo.size-1
    else:
        raise ValueError("Segment must be one of the two end segments of the rope.")
    
    undo_idxs = topo.find_geometry_indices_matching_seg(other_seg,obs['cross'])

    place_idx = undo_idxs[len(undo_idxs)//2]
    place_region = p[place_idx,:].reshape([1,2])

    return pick_idx,None,place_region

def main(env_kwargs,all_args):
    # env = RopeKnotEnv(**env_kwargs)
    envs = SubprocVecEnv([lambda: RopeKnotEnv(goal_topology=trefoil_knot.rep,**env_kwargs)]*all_args.num_workers,'spawn')
    data = {'obs':[],'action':[],'reward':[],'next_obs':[],'dones':[]}
    dones = [False]*all_args.num_workers
    obs = envs.env_method('reset')
    num_failed = 0
    for _ in range(all_args.total_steps//all_args.num_workers):
        env_actions = []
        save_actions = []

        for worker_num in range(all_args.num_workers):
            if dones[worker_num]:
                obs[worker_num] = envs.env_method('reset',indices = worker_num)[0]
            geoms = envs.env_method('get_geoms',True,indices = worker_num)[0]
            topo_obs = get_topological_representation(geoms).astype(np.int32)

            # print(topo_obs)
            while True:
                try:
                    t = RopeTopology(topo_obs,check_validity=False)
                    break
                except InvalidTopology as e:
                    num_failed += 1
                    print(f'Invalid. Resets: {num_failed}')
                    obs[worker_num] = envs.env_method('reset',indices=worker_num)[0]
                    geoms = envs.env_method('get_geoms',True,indices = worker_num)[0]
                    topo_obs = get_topological_representation(geoms).astype(np.int32)
                    # # envs.render()
                    # print(topo_obs)
                    # get_topological_representation(geoms).astype(np.int32)
                    # raise e
            plan = find_topological_path(t,trefoil_knot,max(trefoil_knot.size,t.size))
            action = plan[1].action
            while len(plan) == 0:
                num_failed += 1
                print(f'Planning. Resets: {num_failed}')
                obs[worker_num] = envs.env_method('reset',indices = worker_num)[0]
                geoms = envs.env_method('get_geoms',True,indices = worker_num)[0]
                topo_obs = get_topological_representation(geoms).astype(np.int32)
                t = RopeTopology(topo_obs,check_validity=False)
                plan = find_topological_path(t,trefoil_knot,max(trefoil_knot.size,t.size))
                action = plan[1].action      


            if action[0] == "+C":
                pick_idx,mid_region,place_region = topo_to_geometry_add_C(t,action,obs[worker_num])
            elif action[0] == "-C":
                pick_idx,mid_region,place_region = topo_to_geometry_remove_C(t,action,obs[worker_num])

            p = np.vstack([np.zeros((1,2)),obs[worker_num]['shape'].T])

            place_pos = np.mean(place_region,axis=0)
            if mid_region is None:
                mid_region = ((p[pick_idx,:] + place_pos)/2).reshape([1,2])
            mid_pos = np.mean(mid_region,axis=0)

            pick_norm = pick_idx/(p.shape[0]-1)
            delta_mid = mid_pos - p[pick_idx,:2].reshape([1,2])
            delta_end = place_pos - p[pick_idx,:2].reshape([1,2])

            env_actions.append([pick_norm,*delta_mid.flatten().tolist(),*delta_end.flatten().tolist()])
            save_actions.append([action[0],*env_actions[-1]])


            # plt.clf()
            # plt.plot(p[:,0],p[:,1])
            # plt.fill(place_region[:,0],place_region[:,1],'b')
            # if mid_region is not None:
            #     plt.fill(mid_region[:,0],mid_region[:,1],'r')
            # waypoints = np.vstack([p[pick_idx,:],p[pick_idx,:].reshape([1,2])+np.array(env_actions[-1][1:]).reshape([-1,2])])
            # plt.plot(waypoints[:,0],waypoints[:,1],'k-')
            # plt.draw()
            # plt.pause(1e-1)


        _,rews,dones,info = envs.step(env_actions)
        new_obs = envs.env_method('get_obs')
        # data['obs'].extend(obs)
        # data['action'].extend(save_actions)
        # data['reward'].extend(rews)
        # data['next_obs'].extend(new_obs)
        # data['dones'].extend(dones)
        # for i in range(len(obs)):
        #     pickle.dump({
        #         'obs': obs[i],
        #         'action': save_actions[i],
        #         'reward': rews[i],
        #         'next_obs': new_obs[i]
        #     },pkl_file)

        obs = deepcopy(new_obs)
    with open(all_args.save_name,'ab') as pkl_file:
        pickle.dump(data,pkl_file)
    envs.close()

 



# ------------- Helper functions ----------------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--save_name',type=str,default='./Datasets/TEMP.pkl',help='The directory to place generated models.')
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')

    # Environment options
    parser.add_argument('--num_variations', type=int, default=500, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=5,help='The length of each episode.')
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    parser.add_argument('--maximum_crossings',type=int,default=5,help='The maximum number of crossings for topological representations. Any representation exceeding this will be clipped down.')
    parser.add_argument('--goal_crossings',type=int,default=1,help='The number of crossings used for the goal configuration.')
    parser.add_argument('--total_steps',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=7)

    args = parser.parse_args()    
    args.render_mode = args.render_mode.lower()

    assert args.num_workers > 0, f'num_workers must be set to a positive integer. You entered {args.num_workers}.'
    assert args.horizon > 0, f'Horizon length must be a positive integer. You entered {args.horizon}.'
    assert args.render_mode in ('cloth','particle','both'), f'Render_mode must be from the set {{cloth, particle, both}}. You entered {args.render_mode}.'

    return args

if __name__ == '__main__':
    args = get_args()

    env_kwargs = {
        'observation_mode'  : 'key_point',
        'action_mode'       : 'picker_trajectory',
        'num_picker'        : 1,
        'render'            : not args.headless,
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


