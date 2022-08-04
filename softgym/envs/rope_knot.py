import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
from softgym.utils.topology import *
from softgym.action_space.action_space import PickerTraj
from gym.spaces import Box, Discrete, Dict
from softgym.utils.trajectories import box_trajectory as default_trajectory

from scipy.special import softmax

class _Space:
    def __init__(self,key,type,indices,low,high):
        self.key = key
        self.type = type
        self.indices = indices
        self.low = low
        self.high = high
        self.value = None

    def clip(self):
        self.value = np.clip(
            self.value,
            self.low,
            self.high
        )
    def rescale(self,out_min,out_max):
        current_range = self.high-self.low
        out_range = out_max-out_min

        self.value = ((self.value - self.low) / current_range) * out_range + out_min


class MixedActionSpace(Box):
    # Because stable baselines3 doesn't accept dict spaces.
    def __init__(self,spaces:dict):
        lows, highs = [],[]
        self.spaces = []
        start_index = 0
        for key,value in spaces.items():
            if isinstance(value,Box):
                lows.extend(value.low)
                highs.extend(value.high)
                self.spaces.append(_Space(key,Box,[start_index,len(lows)],value.low,value.high))
            elif isinstance(value,Discrete):
                lows.extend([-1]*value.n)
                highs.extend([1]*value.n)
                self.spaces.append(_Space(key,Discrete,[start_index,len(lows)],0,value.n-1))
            else:
                raise NotImplementedError
            start_index = len(lows)

        super().__init__(np.array(lows),np.array(highs))
        
    
    def split(self, action):
        out_dict = {}
        for space in self.spaces:
            if space.type == Box:
                space.value = action[space.indices[0]:space.indices[1]]
            elif space.type == Discrete:
                space.value = np.argmax(softmax(action[space.indices[0]:space.indices[1]]))
            else:
                raise NotImplementedError
            out_dict[space.key] = space

        return out_dict




class RopeKnotEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_knot_init_states.pkl', **kwargs):
        kwargs['action_mode'] = 'picker_trajectory'
        super().__init__(cached_states_path=cached_states_path,**kwargs)
        if self.observation_mode in ['topology','topo_and_key_point']:
            # figure out what to do with the observation spaces for gym.
            raise NotImplementedError


        self.headless = kwargs['headless']
        self.trajectory_gen_funcs = kwargs.get(
            'trajectory_funcs',
            [default_trajectory]*kwargs.get('num_traj',1)
        )
        self.num_traj = kwargs.get('num_traj',len(self.trajectory_gen_funcs))
        self.maximum_crossings = kwargs['maximum_crossings']
        self.goal_crossings = kwargs['goal_crossings']
        # self.goal_configuration = np.array([
        #     [0 ,1],
        #     [1 ,0],
        #     [-1,1],
        #     [1 ,1]
        # ])


        # self._reset()
        if self.action_mode == 'picker_trajectory':
            self.action_tool = PickerTraj(self.num_picker, picker_radius=self.picker_radius, picker_threshold=0.005, 
                particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.get_cached_configs_and_states(cached_states_path, self.num_variations)
            # num_points = len(self.key_point_indices)
            # self.action_space = Box(
            #     np.concatenate([[self.num_traj],[-0.4,-0.4]*2*self.num_picker]),
            #     np.concatenate([[self.num_traj],[ 0.4, 0.4]*2*self.num_picker])
            # )
            self.action_space = MixedActionSpace({
                # "traj"  : Discrete(self.num_traj),
                "pick": Box(
                    np.array([-1]*self.num_picker),
                    np.array([ 1]*self.num_picker)
                ),
                "place": Box(
                    np.array([-1,-1]*self.num_picker),
                    np.array([ 1, 1]*self.num_picker)
                ),
            })
        key_points_dim = 30#len(self.observation_space.low) # will be assigned in super function without topology
        obs_dim = key_points_dim #+ 4*self.maximum_crossings*2*2 # num_rows*num_crossing*referencesToCrossing, and then again for the goal configuration.
        self.observation_space = Box(np.array([-1]*obs_dim),np.array([1]*obs_dim)) #slightly wrong for now.
        self.reward_penalty = 0
        

        
    def _reset(self):
        
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # IDK
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        self.goal_configuration = np.zeros((4,self.maximum_crossings*2))
        num_crossings = self.goal_crossings#random.randint(1,self.maximum_crossings)
        self.goal_configuration[:,:num_crossings*2] = generate_random_topology(num_crossings)

        for _ in range(50):
            pyflex.step()

        return self._get_obs()

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        
        generated_configs, generated_states = [], []


        if config is None:
            config = self.get_default_config()
        for _ in range(num_variations):
            config_variation = deepcopy(config)

            # Place random variations here
            # ----------------------------
            self.set_scene(config_variation)

            self.update_camera('default_camera',config_variation['camera_params']['default_camera'])
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()

            generated_configs.append(deepcopy(config_variation))
            generated_states.append(deepcopy(self.get_state()))
        return generated_configs, generated_states

    def compute_reward(self, action=None, obs=None, **kwargs):
        success_reward = 1 - self.reward_penalty
        self.reward_penalty *= 0.99
        if self._is_done():
            return success_reward
        else:
            return 0

    def get_topological_representation(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        # return get_topological_representation(keypoint_pos)
        topo = get_topological_representation(keypoint_pos)
        topo_padded = np.zeros((4,self.maximum_crossings*2))
        topo_padded[:,:min(topo.shape[1],self.maximum_crossings*2)] = topo[:,:min(topo.shape[1],self.maximum_crossings*2)]
        return topo_padded.astype(int)
        
     
    def _is_done(self):
        return compare_topology(self.goal_configuration,self.get_topological_representation())

    def _step(self, action):
        action = np.tanh(action)
        action = self.action_space.split(action)
        # action should be [traj_func_index, pick(xy),place(xy),pick2,place2 .....] depending on number of pickers, and sub-policy outputs

        traj_index = action['traj'].value if 'traj' in action else 0

        action["pick"].clip()
        # if action["pick"].type == Box:
        #     action["pick"].rescale(self.workspace[0,[0,2]],self.workspace[1,[0,2]])

        action["place"].clip()
        action["place"].rescale(self.workspace[0,[0,2]],self.workspace[1,[0,2]])       
        # if action["pick"].type == Box:
        #     action["pick"].value = np.clip(
        #         action["pick"].value,
        #         np.tile(self.workspace[0,[0,2]],self.num_picker),
        #         np.tile(self.workspace[1,[0,2]],self.num_picker)
        #     )

        # action["place"].value = np.clip(
        #     action["place"].value,
        #     np.tile(self.workspace[0,[0,2]],self.num_picker),
        #     np.tile(self.workspace[1,[0,2]],self.num_picker)
        # )
        # action_params = np.clip(action_params,self.action_space['params'].low,self.action_space['params'].high)
        if self.action_mode == 'picker_trajectory':
            trajectories = []
            for picker in range(self.num_picker):
                if action["pick"].type == Discrete:
                    pick = pyflex.get_positions().reshape((-1, 4))[self.key_point_indices][action["pick"].value, :3]
                elif action["pick"].type == Box:
                    points = pick = pyflex.get_positions().reshape((-1, 4))[self.key_point_indices]
                    pick_idx = round(((action["pick"].value[picker]*0.5) + 0.5) * len(points)) -1
                    pick = points[pick_idx,:3]
                    # pick = action["pick"].value[picker*2:picker*2 +2],                        
                else:
                    raise NotImplementedError
                place = action["place"].value[picker*2:picker*2 +2]


                trajectories.append(self.trajectory_gen_funcs[traj_index](pick,place,num_points=150))
            traj_action = np.concatenate(trajectories)
            traj_action = traj_action.reshape((self.num_picker,int(traj_action.size/3/self.num_picker),3))
            self.action_tool.step(traj_action,renderer=self.render if not self.headless else lambda *args, **kwargs : None)
        # elif self.action_mode.startswith('picker'):
        #     self.action_tool.step(action)
        #     pyflex.step()
        else:
            raise NotImplementedError
        return

    def apply_negative_reward(self):
        self.reward_penalty = 0.5

    def _get_obs(self):

        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        for i in range(1,keypoint_pos.shape[0]):
            keypoint_pos[i,:] -= keypoint_pos[i-1,:]
        topo = self.get_topological_representation()
        # obs = np.concatenate([topo.flatten(),self.goal_configuration.flatten(),keypoint_pos.flatten()])
        obs = keypoint_pos.flatten()
        return obs
        
        # if self.observation_mode == 'cam_rgb':
        #     return self.get_image(self.camera_height, self.camera_width)


        # if self.observation_mode == 'topology':
        #     return self.get_topological_representation()
        # elif self.observation_mode == 'topo_and_key_point':
        #     raise NotImplementedError

        # if self.observation_mode == 'point_cloud':
        #     particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
        #     pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
        #     pos[:len(particle_pos)] = particle_pos
        #     pos[len(particle_pos):] = self.current_config["goal_character_pos"][:, :3].flatten()
        # elif self.observation_mode == 'key_point':
        #     particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        #     keypoint_pos = particle_pos[self.key_point_indices, :3]
        #     goal_keypoint_pos = self.current_config["goal_character_pos"][self.key_point_indices, :3]
        #     pos = np.concatenate([keypoint_pos, goal_keypoint_pos], axis=0).flatten()


        # if self.action_mode in ['sphere', 'picker']:
        #     shapes = pyflex.get_shape_states()
        #     shapes = np.reshape(shapes, [-1, 14])
        #     pos = np.concatenate([pos.flatten(), shapes[:, :3].flatten()])
        # return pos


    def get_keypoints(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        return particle_pos#[self.key_point_indices, :3]


    def _get_info(self):
        return dict()
