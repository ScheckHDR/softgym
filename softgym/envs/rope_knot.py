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
        key_points_dim = 50#30#len(self.observation_space.low) # will be assigned in super function without topology
        obs_dim = key_points_dim #+ 4*self.maximum_crossings*2*2 # num_rows*num_crossing*referencesToCrossing, and then again for the goal configuration.
        self.observation_space = Box(np.array([-1]*obs_dim),np.array([1]*obs_dim)) #slightly wrong for now.
        

        
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

        rep = reduce_representation(full_rep,np.linspace(0,full_rep.shape[1],10,True,dtype=int))

        # if abs(np.sum(rep[2,:]) > 0.01):
        #     print(full_rep)
        #     print(rep)
        #     raise Exception

        return rep


        
     
    def _is_done(self):
        current = self.get_topological_representation()
        trimmed_crossings = np.trim_zeros(current[4,:],'f')
        unique_ind = np.unique(trimmed_crossings,return_index=True)[1]
        current_order = trimmed_crossings[sorted(unique_ind)]
        current_over_under = current[2,np.where(current[2,:] != 0)[0]].astype(int)
        
        nz = np.nonzero(self.goal_configuration)  # Indices of all nonzero elements
        goal = self.goal_configuration[nz[0].min():nz[0].max()+1,
                        nz[1].min():nz[1].max()+1]
        flipped_goal = flip_topology(deepcopy(goal))
        reversed_goal = reverse_topology(deepcopy(goal))
        flip_reversed_goal = flip_topology(reverse_topology(deepcopy(goal)))

        try:
            C = np.vstack((current_order,current_over_under))
        except Exception as e:
            print(current)
            print(current_order)
            print(current_over_under)


        G = goal[1:3,:]
        RG = reversed_goal[1:3,:]
        FG = flipped_goal[1:3,:]
        RFG = flip_reversed_goal[1:3,:]
        if (np.all(C == G)) \
            or (np.all(C == FG)) \
            or (np.all(C == RG)) \
            or (np.all(C == RFG)):
            return True
        else:
            return False
        

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

    
                
            




    def _get_obs(self):
        topo = self.get_topological_representation().astype(float)
        num_segments = topo[3,-1]
        if num_segments > 2:
            topo[3,:] /= num_segments
            topo[3,:] /= num_segments-1
        return topo.flatten()

    def T_get_obs(self):
        return self._get_obs()
        

    def get_keypoints(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        return particle_pos#[self.key_point_indices, :3]


    def _get_info(self):
        return dict()
