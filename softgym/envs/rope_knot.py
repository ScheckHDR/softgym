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

        if self.action_mode == 'picker_trajectory':
            self.action_tool = PickerTraj(self.num_picker, picker_radius=self.picker_radius, picker_threshold=0.005, 
            particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            # self.action_space = Box(
            #     np.concatenate([[self.num_traj],[-0.4,-0.4]*2*self.num_picker]),
            #     np.concatenate([[self.num_traj],[0.4 , 0.4]*2*self.num_picker])
            # )
            self.action_space = Dict({
                "traj"  : Discrete(self.num_traj),
                "params": Box(
                    np.array([-0.4,-0.4]*2*self.num_picker),
                    np.array([0.4 , 0.4]*2*self.num_picker)
                ),
            })


        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        self.goal_configuration = np.array([
            [1,2],
            [2,1],
            [1,-1]
        ])

        


    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

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
        if compare_topology(self.goal_configuration,self.get_topological_representation()):
            return 1
        else:
            return 0

    def get_topological_representation(self):
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]

        intersections = []
        for i in range(current_pos.shape[0]):
            for j in range (i+2,current_pos.shape[0]-2):
                if intersect(current_pos[i],current_pos[i+1],current_pos[j],current_pos[j+1]):
                    intersections.append([i,j])
                    intersections.append([j,i])

        intersections.sort(key = lambda a:a[0])

        topo = np.zeros((4,len(intersections)))
        for i in range(len(intersections)):
            matching_intersect = intersections.index(intersections[i][::-1])
            is_over = current_pos[intersections[i][0]][1] > current_pos[intersections[i][1]][1]
            
            under_vect = current_pos[intersections[i][0]+1] - current_pos[intersections[i][0]]
            over_vect = current_pos[intersections[i][1]+1] - current_pos[intersections[i][1]]
            if is_over:
                under_vect,over_vect = over_vect,under_vect
            cross_prod = np.cross(over_vect,under_vect)
            sign = np.ones(len(intersections))#np.dot(cross_prod/np.linalg.norm(cross_prod),np.array([0,0,1]))

            topo[:,i] = [
                i,
                matching_intersect,
                is_over, # 0 or 1
                sign[i]#/np.abs(sign) # -1 or 1, disabled at the moment
            ]

        return topo
     
    def _step(self, action):
        # action should be [traj_func_index, pick(xy),place(xy),pick2,place2 .....] depending on number of pickers, and sub-policy outputs
        traj_index = action['traj']
        action_params = action['params']

        action_params = np.clip(action_params,self.action_space['params'].low,self.action_space['params'].high)
        if self.action_mode == 'picker_trajectory':
            trajectories = []
            for picker in range(self.num_picker):
                pick  = action_params[picker*4   :picker*4 +2]
                place = action_params[picker*4 +2:picker*4 +4]
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

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)


        if self.observation_mode == 'topology':
            return self.get_topological_representation()
        elif self.observation_mode == 'topo_and_key_point':
            raise NotImplementedError

        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            pos[len(particle_pos):] = self.current_config["goal_character_pos"][:, :3].flatten()
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            goal_keypoint_pos = self.current_config["goal_character_pos"][self.key_point_indices, :3]
            pos = np.concatenate([keypoint_pos, goal_keypoint_pos], axis=0).flatten()


        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, :3].flatten()])
        return pos

    def _get_info(self):
        return dict()
