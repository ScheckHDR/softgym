import argparse

from softgym.envs.rope_knot import RopeKnotEnv
from softgym.utils.normalized_env import normalize
from softgym.utils.trajectories import box_trajectory, curved_trajectory, curved_trajectory

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
def main(env_args,other_args):
    envs = SubprocVecEnv([lambda: normalize(Monitor(RopeKnotEnv(**env_args)))]*other_args.num_workers,'spawn')

    model = A2C.load(other_args.model_path)
    # model.set_logger(configure(f'{other_args.model_path}_eval_log',["stdout", "csv"]))

    evaluate_policy(
        model,
        envs,
        n_eval_episodes=100,
        deterministic=True,
        render=True,
    )


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path',type=str,help='The model to be evauluated.')
    
    parser.add_argument('-headless', action='store_true', help='Whether to run the environment with headless rendering')
    
    parser.add_argument('--num_workers',type=int,default=1,help='How many workers to run in parallel generating data for the model being trained.')

    # Environment options
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--horizon',type=int,default=10,help='The length of each episode.')
    parser.add_argument('--pickers',type=int,default=1)
    parser.add_argument('--render_mode',type=str,default='cloth',help='The render mode of the object. Must be from the set \{cloth, particle, both\}.')
    parser.add_argument('--maximum_crossings',type=int,default=5,help='The maximum number of crossings for topological representations.')
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
        'num_picker': args.pickers,
        'render': True,
        'headless': args.headless,
        'horizon': args.horizon,
        'action_repeat': 1,
        'render_mode': args.render_mode,
        'num_variations': args.num_variations,
        'use_cached_states': False,
        'save_cached_states': False,
        'deterministic': True,
        'trajectory_funcs': [
            box_trajectory,
            curved_trajectory,
        ],
        'maximum_crossings':args.maximum_crossings,
        'goal_crossings': args.goal_crossings,
    }


    main(env_kwargs,args)