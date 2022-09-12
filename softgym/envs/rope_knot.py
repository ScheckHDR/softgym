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
import softgym.utils.trajectories

import time

def convert_topo_rep(topo,workspace,obs_spaces):
    N = topo.shape[1]

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
        corr_seg = np.where(topo[4,:] == seg)[0]
        j = np.where(topo[2,corr_seg] != 0)[0]

        incidence_matrix[i,corr_seg[j]] = topo[2,i]
        incidence_matrix[corr_seg[j],i] = topo[2,corr_seg[j]]



    # Normalising
    tail_normalising = np.vstack([workspace,[-pi,pi]])
    # tail = (tail + tail_normalising[:,0]) / (tail_normalising[:,1] - tail_normalising[:,0]) * 2 - 1

    # Normalise Shape?

    return {
        "tail"  : tail.reshape(obs_spaces['tail'].shape),
        "shape" : shape.reshape(obs_spaces['shape'].shape),
        "cross" : incidence_matrix.reshape(obs_spaces['cross'].shape)
    }


class RopeKnotEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_knot_init_states.pkl', **kwargs):
        kwargs['action_mode'] = 'picker_trajectory'
        super().__init__(cached_states_path=cached_states_path,**kwargs)
        
        if self.observation_mode in ['topology','topo_and_key_point']:
            # figure out what to do with the observation spaces for gym.
            raise NotImplementedError

        self.headless = kwargs['headless']

        # Because wandb converted the function pointers to strings, need to convert them back.
        if 'trajectory_funcs' in kwargs:
            self.trajectory_gen_funcs = []
            for func in kwargs['trajectory_funcs']:
                self.trajectory_gen_funcs.append(getattr(softgym.utils.trajectories,func.split('.')[-1]))
        else:
            self.trajectory_gen_funcs = [softgym.utils.trajectories.box_trajectory]

        self.num_traj = len(self.trajectory_gen_funcs)
        self.maximum_crossings = kwargs['maximum_crossings']
        self.goal_crossings = kwargs['goal_crossings']



        if self.action_mode == 'picker_trajectory':
            self.action_tool = PickerTraj(self.num_picker, picker_radius=self.picker_radius, picker_threshold=0.005, 
                particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.get_cached_configs_and_states(cached_states_path, self.num_variations)

            self.action_space = Box(
                np.array([0,-0.35,-0.35]*self.num_picker),
                np.array([ 1, 0.35, 0.35]*self.num_picker)
            )

        points = 41
        # obs_dim = points*points + 2*points + 1
        self.observation_space = Dict({
            "tail"  : Box(low=np.array([-1]*3,ndmin=2), high=np.array([1]*3,ndmin=2)),
            "shape" : Box(low=np.array([[-1]*(points-1)]*2,ndmin=2),high=np.array([[-1]*(points-1)]*2,ndmin=2)),
            "cross" : Box(low=np.array([[-1]*points]*points,ndmin=3),high=np.array([[-1]*points]*points,ndmin=3))
        })
        # self.observation_space = Box(np.array([-1]*obs_dim),np.array([1]*obs_dim))

        self.workspace =np.array([[-0.35,0.35],[-0.35,0.35]])
        self.goal_configuration = deepcopy(generate_random_topology(self.goal_crossings))
        

        
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

        if self._is_done():
            return 0
        else:
            return -1

    def get_topological_representation(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        
        topo_test = np.zeros((3,particle_pos.shape[0]))
        intersections = []
        for i in range(1,particle_pos.shape[0]):
            for j in range (i+2,particle_pos.shape[0]):

                if intersect(particle_pos[i],particle_pos[i-1],particle_pos[j],particle_pos[j-1]):
                    for k in range(j,particle_pos.shape[0]):
                        # avoid possibility for two crossings to be associated with same geom
                        # pushes secondary crossing onto later geom
                        if topo_test[0,k] == 0:
                            topo_test[1,i:] += 1
                            topo_test[1,k:] += 1
                            intersections.append([i,k])
                            intersections.append([k,i])
                            if particle_pos[i,1] > particle_pos[k,1]:
                                topo_test[0,i] =  1
                                topo_test[0,k] = -1
                            else:
                                topo_test[0,i] = -1
                                topo_test[0,k] = 1
                            break

        
        if len(intersections) > 1:
            intersections.sort(key = lambda a:a[0])
            cross1 = np.argmax(topo_test[1,:] != 0)
            for i in range(cross1,topo_test.shape[1]):
                topo_test[2, i] = intersections.index(
                    intersections[int(topo_test[1,i])-1][::-1])
            
        full_rep = np.concatenate((np.transpose(particle_pos[:,[0,2]]),topo_test),axis=0)

        return full_rep

   
    def _is_done(self):
        current = self.get_topological_representation()
        trimmed_crossings = np.trim_zeros(current[4,:],'f')
        unique_ind = np.unique(trimmed_crossings,return_index=True)[1]
        current_order = trimmed_crossings[sorted(unique_ind)]
        current_over_under = current[2,np.where(current[2,:] != 0)[0]].astype(int)
        
        nz = np.nonzero(self.goal_configuration)  # Indices of all nonzero elements
        goal = self.goal_configuration[nz[0].min():nz[0].max()+1,
                        nz[1].min():nz[1].max()+1]
        # flipped_goal = flip_topology(deepcopy(goal))
        # reversed_goal = reverse_topology(deepcopy(goal))
        # flip_reversed_goal = flip_topology(reverse_topology(deepcopy(goal)))

        C = np.vstack((current_order,current_over_under))



        G = goal[1:3,:]
        # RG = reversed_goal[1:3,:]
        # FG = flipped_goal[1:3,:]
        # RFG = flip_reversed_goal[1:3,:]
        if (np.all(C == G)):# \
            # or (np.all(C == FG)) \
            # or (np.all(C == RG)) \
            # or (np.all(C == RFG)):
            return True
        else:
            return False
        

    def _step(self, action):
        rope = self._get_obs()
        rope_frame = rope['tail']

        theta = rope_frame[0,2]

        r_mat = np.array(
            [
                [np.math.cos(theta),0,np.math.sin(theta),rope_frame[0,0]],
                [0,1,0,0],
                [-np.math.sin(theta),0, np.math.cos(theta),rope_frame[0,1]],
                [0,0,0,1]
            ]
        )

        # heights = np.transpose(pyflex.get_positions().reshape((-1, 4))[:,1])
        rel_positions_h = np.concatenate(
            (
                np.insert(rope['shape'][0,:],0,0).reshape((1,-1)),
                np.zeros((1,rope['cross'].shape[-1])),
                np.insert(rope['shape'][1,:],0,0).reshape((1,-1)),
                np.ones((1,rope['cross'].shape[-1]))
            ),
            axis=0
        ).reshape((4,-1))

        pick_idx = round(action[0] * (rel_positions_h.shape[1]-1))
        pick_coords_rel_h = rel_positions_h[:,pick_idx]

        place_coords_rel_h = pick_coords_rel_h + np.array([action[1],0,action[2],0])

        pick_coords = (r_mat @ pick_coords_rel_h)[0:3]
        place_coords = (r_mat @ place_coords_rel_h)[0:3]

        pos = pyflex.get_positions().reshape((-1, 4))
        # print(f'frame: {rope_frame}')
        # print(f'pick: {pick_coords}, should be {pos[pick_idx,:3]}')
        # print(f'place_h :{place_coords_rel_h}')
        # print(f'place:{place_coords}')
        # print('-'*50)

        traj_index = 0
        traj = [self.trajectory_gen_funcs[traj_index](pick_coords,place_coords,num_points=150)]
        traj_action = np.concatenate(traj)
        traj_action = traj_action.reshape((self.num_picker,int(traj_action.size/3/self.num_picker),3))
        self.action_tool.step(traj_action,renderer=self.render if not self.headless else lambda *args, **kwargs : None)


        # if self.action_mode == 'picker_trajectory':
        #     trajectories = []
        #     for picker in range(self.num_picker):
        #         pick_idx = round(action[0] * (heights.shape[0]-1))
        #         pick_coords_rel = pos[:,pick_idx]
        #         place_coords_rel = pick_coords_rel + np.array([[action[1]],[action[2]],[0]])

        #         pick_coords = r_mat @ np.array([[pick_coords_rel[0]],[pick_coords_rel[1]],[1]])
        #         pick_coords[2] = pick_coords_rel[1]       

        #         place_coords = r_mat @ place_coords_rel
        #         place_coords[2] = place_coords_rel[1]

        #         print(f'frame: {rope_frame}')
        #         print(f'pick: {pick_coords}, should be {pos[pick_idx,:3]}')
        #         print(f'place:{place_coords}')
        #         print('-'*50)

        #         trajectories.append(self.trajectory_gen_funcs[traj_index](pick_coords,place_coords,num_points=150))
        #     traj_action = np.concatenate(trajectories)
        #     traj_action = traj_action.reshape((self.num_picker,int(traj_action.size/3/self.num_picker),3))
        #     self.action_tool.step(traj_action,renderer=self.render if not self.headless else lambda *args, **kwargs : None)

        # else:
        #     raise NotImplementedError
        return

    

    def _get_obs(self):
        topo = self.get_topological_representation().astype(float)

        return convert_topo_rep(topo,self.workspace,self.observation_space)
        

    def get_keypoints(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        return particle_pos#[self.key_point_indices, :3]


    def _get_info(self):
        return dict()