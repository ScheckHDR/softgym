import argparse
from copy import deepcopy
from typing import Dict
import multiprocessing as mp

import gym
from softgym.envs.rope_knot import RopeKnotEnv
from softgym.utils.normalized_env import normalize
from softgym.utils.trajectories import box_trajectory, curved_trajectory, curved_trajectory

from stable_baselines3 import A2C, SAC, PPO, DQN
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import BaseCallback, CallbackList


import wandb
from wandb.integration.sb3 import WandbCallback

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.first_iter = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        
        if not self.first_iter:
            wandb.log({
                'mean_reward':  self.model.logger.name_to_value["rollout/ep_rew_mean"],
                'timesteps':    self.model.logger.name_to_value["time/total_timesteps"],
                'policy_loss':  self.model.logger.name_to_value['train/policy_loss'],
                'value_loss':   self.model.logger.name_to_value['train/value_loss']
            })
        self.first_iter = False


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def main(default_config):

    run = wandb.init(
        project='test',
        config=default_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False
    )
    env_kwargs = {
        'observation_mode': wandb.config.observation_mode,
        'action_mode': wandb.config.action_mode,
        'num_picker': wandb.config.num_picker,
        'render': wandb.config.render,
        'headless': wandb.config.headless,
        'horizon': wandb.config.horizon,
        'action_repeat': wandb.config.action_repeat,
        'render_mode': wandb.config.render_mode,
        'num_variations': wandb.config.num_variations,
        'use_cached_states': wandb.config.use_cached_states,
        'save_cached_states': wandb.config.save_cached_states,
        'deterministic': wandb.config.deterministic,
        'trajectory_funcs': wandb.config.trajectory_funcs,
        'maximum_crossings': wandb.config.maximum_crossings,
        'goal_crossings': wandb.config.goal_crossings,
    }
    
    if not hasattr(wandb.config,'algorithm'):
        algorithm = A2C
    else:
        if wandb.config.algorithm == 'SAC':
            algorithm = SAC
        elif wandb.config.algorithm == 'PPO':
            algorithm = PPO
        elif wandb.config.algorithm == 'A2C':
            algorithm = A2C
        elif wandb.config.algorithm == 'DQN':
            algorithm = DQN



    training_kwargs = {
        'gamma' : wandb.config.gamma,
        'learning_rate' : wandb.config.learning_rate,
        'ent_coef' : wandb.config.ent_coef, 
    }

    if algorithm is not SAC:
        training_kwargs['n_steps'] = wandb.config.n_steps

    try:
        if algorithm is not SAC:
            envs = SubprocVecEnv([lambda: normalize(Monitor(RopeKnotEnv(**env_kwargs)))]*wandb.config.num_workers,'spawn')
            model = algorithm(
                wandb.config.policy_type,
                envs,
                verbose = 1,
                **training_kwargs  
            )
        else:
            envs = normalize(RopeKnotEnv(**env_kwargs))
            model = algorithm(
                wandb.config.policy_type,
                envs,
                verbose = 1,
                **training_kwargs  
            )

        # try:
        model.learn(
            total_timesteps= wandb.config.total_timesteps // (wandb.config.num_workers if wandb.config.algorithm == 'SAC' else 1),
            log_interval = 1,
            callback=CallbackList([
                # WandbCallback(
                #     gradient_save_freq=100,
                #     model_save_path=f'{wandb.config.save_name}/models/{run.id}',
                #     verbose=2,
                # ),
                CustomCallback(verbose=2),
            ])
        )
    except Exception as e:
        print(e)
        envs.close()
    # except:
    #     pass
    # finally:
    del model
    envs.close()
    run.finish()

 



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

    
    default_config = {
        "policy_type":      "CustPolicy",
        "total_timesteps":  5000,
        "env_name":         "ropeKnotting",

        # Simulator parameters
        'num_workers'       : args.num_workers,
        'save_name'         : args.save_name,
        'headless'          : args.headless,
        'horizon'           : args.horizon,
        'render_mode'       : args.render_mode,
        'render'            : not args.headless,#True,
        'action_repeat'     : 1,
        'use_cached_states' : True,
        'save_cached_states': True,
        'deterministic'     : False,

        # Environment parameters
        'total_timesteps'   : args.total_steps,
        'maximum_crossings' : args.maximum_crossings,
        'goal_crossings'    : args.goal_crossings,
        'num_variations'    : args.num_variations,
        'observation_mode'  : 'key_point',
        'action_mode'       : 'picker_trajectory',
        'num_picker'        : 1,
        'trajectory_funcs'  : [box_trajectory],

        # Training hyperparameters
        'algorithm'         : 'SAC',
        'learning_rate'     : 1e-3,
        'ent_coef'          : 1e-2,
        'gamma'             : 0.9,
        'n_steps'           : 5,
    }

    sweep_params = {
        "name": "Sweep_single_cross",
        "method": "bayes",
        "metric":{
            "name": "mean_reward",
            "goal": "maximize",
        },
        "parameters":{
            "learning_rate":{
                "min": 1e-4,
                "max": 1e-1,
            },
            "ent_coef":{
                "min" : 1e-3,
                "max" : 5e-2,
            },
            "gamma":{
                "values" : [0.9,0.95,0.99]
            },
            "n_steps" : {
                "values" : [5,10,15]
            },
            "algorithm":{
                "values" : ['SAC']#['PPO','DQN','A2C','SAC']
            }
        },
        "early_terminate":{
            "type"      : "hyperband",
            "min_iter"  : 10,
            "eta"       : 2
        },
    }  

    
    if args.num_sweeps == 1:
        main(default_config)
    else:
        sweep_id = args.sweep_id or wandb.sweep(sweep_params)
        if args.num_sweeps == 0:
            args.num_sweeps = None

        processes = [mp.Process(target = lambda: wandb.agent(sweep_id,function= lambda :main(default_config),count=args.num_sweeps)) for _ in range(args.num_agents)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        # wandb.agent(sweep_id,function= lambda :main(default_config),count=args.num_sweeps)


