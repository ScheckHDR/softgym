from softgym.envs.rope_knot import RopeKnotEnv
import argparse
from softgym.utils.normalized_env import normalize


def main(model,env_args):
    env = normalize(RopeKnotEnv(**env_kwargs))

    for _ in range(env_args['horizon']):
        action = env.action_space.sample()
        env.step(action,record_continuous_video=True,img_size=720)
# ------------- Helper functions ----------------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--out_dir',type=str,default='./output',help='The directory to place generated models.')
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')

    # Environment options
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=10,help='The length of each episode.')
    parser.add_argument('--pickers',type=int,default=2)
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    
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
        'observation_mode': 'cam_rgb',
        'action_mode': 'picker_trajectory',
        'num_picker': 2,
        'render': True,
        'headless': args.headless,
        'horizon': args.horizon,
        'action_repeat': 1,
        'render_mode': args.render_mode,
        'num_variations': args.num_variations,
        'use_cached_states': False,
        'save_cached_states': False,
        'deterministic': False,
    }

    main(None,env_kwargs)

