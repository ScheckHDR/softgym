from cmath import pi
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
from softgym.utils.trajectories import simple_trajectory

import time
import cv2

def convert_topo_rep(topo,workspace,obs_spaces):
    N = topo.size

    # r = np.math.atan2(topo[1,0],topo[0,0])
    theta = np.math.atan2(topo[1,1] - topo[1,0],topo[0,1] - topo[0,0])

    t_mat = np.array(
        [[np.math.cos(theta),np.math.sin(theta),topo[0,0]],
        [-np.math.sin(theta),np.math.cos(theta),topo[1,0]],
        [0,0,1]]
    )
    h_points = np.concatenate((topo[:2,:],np.ones((1,N))),axis=0)

    t_points = np.linalg.inv(t_mat) @ h_points
    t_delta = t_points[:,1:] - np.expand_dims(t_points[:,0],1)
    shape = t_delta[:2,:]
    tail = np.array([topo[0,0],topo[1,0],theta],ndmin=2)

    incidence_matrix = np.zeros((N,N))
    for i in range(N):
        if topo[2,i] == 0:
            continue

        if len(np.where(abs(incidence_matrix[i,:]) > 0)[0]) != 0:
            #already found the crossing previously.
            continue

        seg = topo[3,i]
        corr_seg = topo[4,i]
        # if corr_seg.size == 0:
        #     # should have found them all
        #     break
        #j = np.where(topo[2,corr_seg] != 0)[0]
        j = np.where(topo[3,:])
        # j = min(N-1,corr_seg[-1] + 1)

        incidence_matrix[i,j] = topo[2,i]
        incidence_matrix[j,i] = topo[2,j]



    # Normalising
    tail_normalising = np.vstack([workspace,[-pi,pi]])
    # tail = (tail + tail_normalising[:,0]) / (tail_normalising[:,1] - tail_normalising[:,0]) * 2 - 1

    # Normalise Shape?

    return {
        "tail"  : tail,
        "shape" : shape,
        "cross" : incidence_matrix
    }

class RopeKnotEnv(RopeNewEnv):
    def __init__(self, goal_topology = None,cached_states_path='rope_knot_init_states.pkl', **kwargs):
        kwargs['action_mode'] = 'picker_trajectory'
        super().__init__(cached_states_path=cached_states_path,**kwargs)
        
        if self.observation_mode in ['topology','topo_and_key_point']:
            # figure out what to do with the observation spaces for gym.
            raise NotImplementedError

        self.headless = kwargs['headless']

        self.goal_crossings = kwargs['goal_crossings']

        if self.action_mode == 'picker_trajectory':
            self.action_tool = PickerTraj(self.num_picker, picker_radius=self.picker_radius, picker_threshold=0.005, 
                particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.get_cached_configs_and_states(cached_states_path, self.num_variations)

            self.action_space = Box(
                np.array([0,-0.35,-0.35,-0.35,-0.35]*self.num_picker),
                np.array([ 1, 0.35, 0.35,0.35,0.35]*self.num_picker)
            )

        points = 41
        # obs_dim = points*points + 2*points + 1

        self.task = kwargs["task"].upper()
        if self.task == "KNOT":
            dim = points*3 + 3
        elif self.task == "STRAIGHT":
            dim = points*3 + 3
        elif self.task == "CORNER":
            dim = 6
        self.observation_space = Box(low=-np.ones((1,dim)),high=np.ones((1,dim)))

        self.workspace =np.array([[-0.35,0.35],[-0.35,0.35]])
        if goal_topology is not None:
            self.goal_configuration = deepcopy(goal_topology)
        else:
            self.goal_configuration = deepcopy(RopeTopology.random(self.goal_crossings))
              
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

        # self.goal_configuration = np.zeros((4,self.maximum_crossings*2))
        # num_crossings = self.goal_crossings#random.randint(1,self.maximum_crossings)
        # self.goal_configuration[:,:num_crossings*2] = generate_random_topology(num_crossings)

        for _ in range(50):
            pyflex.step()

        while True:
            try:
                obs = self._get_obs()
            except InvalidGeometry as e:
                pick_id = int(str(e).replace('.','').split(' ')[-1])
                disturb_rope(pick_id)
                continue
            break
        return obs

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        
        generated_configs, generated_states = [], []

        while True:
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
            try:
                self.get_topological_representation()
            except InvalidGeometry as e:
                pick_id = int(str(e).replace('.','').split(' ')[-1])
                disturb_rope(pick_id)   
                continue
            break

        return generated_configs, generated_states

    def compute_reward(self, action=None, obs=None, **kwargs):
        if self.task == "STRAIGHT":
            reward = np.linalg.norm(self.get_geoms()[-1,:])
        elif self.task == "KNOT":
            reward = self._is_done()-1
        return reward

    def get_geoms(self):
        geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        x,y,z,theta = self.get_rope_frame()
        T_mat = np.array([
            [np.cos(theta),0,np.sin(theta),x],
            [0,1,0,y],
            [-np.sin(theta),0,np.cos(theta),z],
            [0,0,0,1]
        ])

        return (np.linalg.inv(T_mat) @ np.vstack([geoms.T,np.ones(geoms.shape[0])]))[:3,:].T

    def get_rope_frame(self):
        geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:2, :3]

        x = geoms[0,0]
        y = geoms[0,1] # nominally 0.1
        z = geoms[0,2]

        theta = np.arctan2(geoms[1,1]-geoms[0,1],geoms[1,0]-geoms[0,0])

        return x,y,z,theta

    def get_topological_representation(self):
        
        return RopeTopology.from_geometry(self.get_geoms(),plane_normal=np.array([0,1,0]))

    def _is_done(self):
        # return RopeTopology.is_equivalent(self.get_topological_representation(),self.goal_configuration,False,False)
        return False
        
    def _step(self, action):
        action = action.flatten()
        rope = self.get_geoms()
        rope_frame = self.get_rope_frame()

        theta = rope_frame[3]

        T_mat = np.array(
            [
                [np.cos(theta),0,np.sin(theta),rope_frame[0]],
                [0,1,0,rope_frame[1]],
                [-np.sin(theta),0, np.cos(theta),rope_frame[2]],
                [0,0,0,1]
            ]
        )
        rel_positions_h = np.vstack([rope.T,np.ones(rope.shape[0])])

        pick_idx = round(action[0] * (rel_positions_h.shape[1]-1))
        pick_coords_rel_h = np.expand_dims(rel_positions_h[:,pick_idx],-1)

        waypoints_rel = np.array(action[1:]).reshape([-1,2]).T

        waypoints_rel_h = np.vstack([waypoints_rel[0,:],np.zeros(waypoints_rel.shape[1]),waypoints_rel[1,:],np.zeros(waypoints_rel.shape[1])]) # Making last row zeros instead of ones for the homogeneous as the ones will come from the addition on next line.
        waypoints_h = pick_coords_rel_h + waypoints_rel_h

        pick_coords = (T_mat @ pick_coords_rel_h)[0:3]
        waypoint_coords = (T_mat @ waypoints_h)[0:3]

        traj = np.expand_dims(simple_trajectory(np.hstack([pick_coords,waypoint_coords]).T,height=0.1,num_points_per_leg=50),0) # only a single picker

        self.action_tool.step(traj,renderer=self.render if not self.headless else lambda *args, **kwargs : None)

        try:
            self.get_topological_representation()
        except InvalidGeometry as e:
            id1 = int(str(e).replace('.','').split(' ')[-1])
            id2 = int(str(e).replace('.','').split(' ')[-3])
            pick_id = id1 if abs(id1-pick_idx) < abs(id2-pick_idx) else id2
            disturb_rope(pick_id)

    def _get_obs(self):
        # self.get_topological_representation()
        geoms = self.get_geoms()
        x,y,z,theta = self.get_rope_frame()
        if self.task == "KNOT":
            return np.hstack([np.array([x,z,theta]),geoms.flatten()])
        elif self.task == "STRAIGHT":
            return np.expand_dims(np.hstack([np.array([x,z,theta]),geoms.flatten()]),0)
        elif self.task == "CORNER":
            return np.hstack([np.array([x,z,theta]),geoms[0,:].flatten()])
        else:
            raise Exception(f"Unknown observation required. Observations must be any of {{KNOT,STRAIGHT,CORNER}}, not {self.task}")
    def get_obs(self):
        return self._get_obs()

    def get_keypoints(self):
        particle_pos = self.get_geoms(True)
        return particle_pos#[self.key_point_indices, :3]

    def _get_info(self):
        return dict()

    def render_no_gripper(self,mode='rgb_array'):
        self.action_tool.step(np.array([1,1,1,1],ndmin=2),renderer = lambda *args,**kwargs : None)
        return cv2.cvtColor(super().render(mode=mode)[-self.camera_height:,:self.camera_width,:],cv2.COLOR_RGB2BGR)


######### Helper Functions
def disturb_rope(move_idx:int,amount:float=5e-3):
    curr_pos = pyflex.get_positions().reshape(-1, 4)
    curr_pos[move_idx, 1] += amount
    pyflex.set_positions(curr_pos.flatten())
    pyflex.step()

def rescale(x:np.ndarray,old_min:np.ndarray,old_max:np.ndarray,new_min:np.ndarray,new_max:np.ndarray):
    return (x - old_min) / (old_max-old_min) * (new_max-new_min) + new_min