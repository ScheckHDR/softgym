import argparse
from copy import deepcopy
from softgym.envs.rope_knot import RopeKnotEnv
from softgym.utils.normalized_env import normalize
from softgym.utils.topology import get_topological_representation
from softgym.utils.trajectories import box_trajectory, curved_trajectory, curved_trajectory

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import csv
import numpy as np


def which_reidemeister(prev,new):
    if np.all(prev==new):
        return 0
    
    if new.shape[1] - 2 == prev.shape[1]:
        #possible R1
        pass

def main(env_args,other_args):
    
    ''' Something is broken, need to render all of them on my PC '''
    # extra_env_args = deepcopy(env_args)
    # extra_env_args['headless'] = True # Don't need to display all envs
    # start_funcs = \
    #     [lambda: normalize(RopeKnotEnv(**env_args))]\
    #     +[lambda: normalize(RopeKnotEnv(**extra_env_args))]*(env_args['n_envs']-1)
    # envs = SubprocVecEnv(start_funcs,'spawn')

    envs = SubprocVecEnv([lambda: normalize(Monitor(RopeKnotEnv(**env_args)))]*other_args.num_workers,'spawn')

    prev_topologies = envs._get_topological_representation()
    with open('test.csv','w') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter=',')
        for i in range(int(1000/other_args.num_workers)):
            actions = envs.action_space.sample()
            envs.step(actions,record_continuous_video=True,img_size=720)
            ropes = envs._get_keypoints()
            new_topologies = envs._get_topological_representation()

            for i in range(len(ropes)):
                move = which_reidemeister(prev_topologies[i],new_topologies[i])
                csv_writer.writerow([ropes[i],actions[i],prev_topologies[i],move])

            prev_topologies = deepcopy(new_topologies)




    envs.close()



# ------------- Helper functions ----------------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--save_name',type=str,default='./output/TEMP',help='The directory to place generated models.')
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')

    # Environment options
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=5,help='The length of each episode.')
    parser.add_argument('--pickers',type=int,default=2)
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    parser.add_argument('--maximum_crossings',type=int,default=5,help='The maximum number of crossings for topological representations. Any representation exceeding this will be clipped down.')
    parser.add_argument('--goal_crossings',type=int,default=2,help='The number of crossings used for the goal configuration.')
    
    args = parser.parse_args()    
    args.render_mode = args.render_mode.lower()

    assert args.num_workers > 0, f'num_workers must be set to a positive integer. You entered {args.num_workers}.'
    assert args.horizon > 0, f'Horizon length must be a positive integer. You entered {args.horizon}.'
    assert args.pickers > 0, f'Number of pickers must be a positive integer. You entered {args.pickers}.'
    assert args.render_mode in ('cloth','particle','both'), f'Render_mode must be from the set {{cloth, particle, both}}. You entered {args.render_mode}.'

    return args

if __name__ == '__main__':
    args = get_args()

    # parse args into args needed for environment.
    env_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker_trajectory',
        'num_picker': 1,
        'render': True,
        'headless': args.headless,
        'horizon': args.horizon,
        'action_repeat': 1,
        'render_mode': args.render_mode,
        'num_variations': args.num_variations,
        'use_cached_states': False,
        'save_cached_states': False,
        'deterministic': False,
        'trajectory_funcs': [
            box_trajectory,
            curved_trajectory,
        ],
        'maximum_crossings':args.maximum_crossings,
        'goal_crossings': args.goal_crossings,
    }


    main(env_kwargs,args)

