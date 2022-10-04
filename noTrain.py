import argparse
from copy import deepcopy
from typing import Dict
import numpy as np
import multiprocessing as mp

import gym
from softgym.envs.rope_knot import RopeKnotEnv
from softgym.utils.normalized_env import normalize
from softgym.utils.trajectories import box_trajectory, curved_trajectory, curved_trajectory

from softgym.utils.new_topology_test import RopeTopology,RopeTopologyNode,find_topological_path
from softgym.utils.topology import get_topological_representation

import random
random.seed(9)
np.random.seed(9)
trefoil_knot = RopeTopology(np.array([
        [ 0, 1, 2, 3, 4, 5],
        [ 3, 4, 5, 0, 1, 2],
        [ 1,-1, 1,-1, 1,-1],
        [-1,-1,-1,-1,-1,-1]
    ],dtype=np.int32))

def main(env_kwargs):
    env = RopeKnotEnv(**env_kwargs)

    obs = env.reset()
    while True:
        geoms = env.get_geoms()
        topo_obs = get_topological_representation(geoms).astype(np.int32)
        print(topo_obs)
        t = RopeTopology(topo_obs)
        plan = find_topological_path(t,trefoil_knot)
        action = plan[1].action
        print(plan[1].value.rep)

        if action[0] == "+C":
            if action[1][0] == action[1][1]:
                segment_idxs = t.find_geometry_indices_matching_seg(action[1][0],obs['cross'])
                l = len(segment_idxs)
                if action[1][3]:
                    # under first
                    pick_seg = segment_idxs[l//2:]
                    place_seg = segment_idxs[:l//2]
                else:
                    place_seg = segment_idxs[l//2:]
                    pick_seg = segment_idxs[:l//2]
                    
            else:
                pick_seg = t.find_geometry_indices_matching_seg(action[1][0],obs['cross'])
                place_seg = t.find_geometry_indices_matching_seg(action[1][1],obs['cross'])
            pick_idx = pick_seg[len(pick_seg)//2]
            place_idx = place_seg[len(place_seg)//2]

            pick_act = pick_idx / geoms.shape[0]
            place_act = (obs['shape'][:,place_idx] - obs['shape'][:,pick_idx]) * 1.2

            robot_action = (pick_act,place_act[0],place_act[1])

        elif action[0] == "-C":
            raise NotImplementedError
        obs,rew,done,info = env.step(robot_action)

        if rew > 1e6:
            break

    env.close()

 



# ------------- Helper functions ----------------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--save_name',type=str,default='./output/TEMP',help='The directory to place generated models.')
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')
    parser.add_argument('--num_agents',type=int,default=1)

    # Environment options
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=5,help='The length of each episode.')
    parser.add_argument('--pickers',type=int,default=2)
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    parser.add_argument('--maximum_crossings',type=int,default=2,help='The maximum number of crossings for topological representations. Any representation exceeding this will be clipped down.')
    parser.add_argument('--goal_crossings',type=int,default=1,help='The number of crossings used for the goal configuration.')
    parser.add_argument('--total_steps',type=int,default=5000)
    parser.add_argument('--num_sweeps',type=int,default=1,help='The number of runs to do in a sweep. If set to one, will default to doing a single run outside of a sweep setting. If set to 0, will keep sweeping indefinitely.')
    parser.add_argument('--sweep_id',type=str,default=None)
    parser.add_argument('--project_name',type=str,default="Rope_RL")
    parser.add_argument('--sweep_name',type=str,default='test')

    args = parser.parse_args()    
    args.render_mode = args.render_mode.lower()

    assert args.num_workers > 0, f'num_workers must be set to a positive integer. You entered {args.num_workers}.'
    assert args.horizon > 0, f'Horizon length must be a positive integer. You entered {args.horizon}.'
    assert args.pickers > 0, f'Number of pickers must be a positive integer. You entered {args.pickers}.'
    assert args.render_mode in ('cloth','particle','both'), f'Render_mode must be from the set {{cloth, particle, both}}. You entered {args.render_mode}.'
    assert args.num_sweeps >= 0, f'num_sweeps must be a positive whole number. You entered {args.num_sweeps}'
    assert args.num_agents > 0, f'num_agents must be a positive integer, not {args.num_agents}'

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
        'use_cached_states' : True,
        'save_cached_states': True,
        'deterministic'     : False,
        # 'trajectory_funcs'  : [box_trajectory],
        'maximum_crossings' : args.maximum_crossings,
        'goal_crossings'    : args.goal_crossings,
    }

    main(env_kwargs)


