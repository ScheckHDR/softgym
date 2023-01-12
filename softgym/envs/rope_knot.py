from cmath import pi
import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
import softgym.utils.topology as topology
from softgym.action_space.action_space import PickerTraj
from gym.spaces import Box, Discrete, Dict
from softgym.utils.trajectories import simple_trajectory

import time
import random
import cv2
from tqdm import tqdm

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
    ROPE_LINK_LENGTH = 0.0135 # approximate length of a single link. Seems to be some ability to stretch.
    def __init__(self, goal = None,cached_states_path='rope_knot_init_states.pkl', **kwargs):
        kwargs['action_mode'] = 'picker_trajectory'
        super().__init__(cached_states_path=cached_states_path,**kwargs)
        
        if self.observation_mode in ['topology','topo_and_key_point']:
            # figure out what to do with the observation spaces for gym.
            raise NotImplementedError

        self.headless = kwargs['headless']
        self.force_trivial = kwargs.get("force_trivial",False)
        self.goal_crossings = kwargs['goal_crossings']

        if self.action_mode == 'picker_trajectory':
            self.action_tool = PickerTraj(self.num_picker, picker_radius=self.picker_radius, picker_threshold=0.005, 
                particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))

            self.action_space = Box(
                -np.ones([1,7]*self.num_picker),
                np.ones([1,7]*self.num_picker)
            )

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        points = 41
        # obs_dim = points*points + 2*points + 1

        self.task = kwargs["task"].upper()
        if self.task == "KNOT":
            dim = (points-1)*3 + 4
        elif self.task == "STRAIGHT":
            dim = (points-1)*3 + 4
        elif self.task == "CORNER":
            dim = 6
        elif self.task == "KNOT_ACTION_+R1":
            dim = (points-1)*3 + 7

        self.observation_space = Box(low=-np.ones((1,dim)),high=np.ones((1,dim)))

        self.workspace =np.array([[-0.35,0.35],[-0.35,0.35]])
        self.goal = goal
              
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

        self.maybe_disturb_rope(steps=50)
            

        obs = self.get_obs()
        return obs

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        
        generated_configs, generated_states = [], []

        if config is None:
            config = self.get_default_config()
        print("Generating Variations")
        for _ in tqdm(range(num_variations)):
            config_variation = deepcopy(config)

            # Place random variations here
            # ----------------------------
            self.set_scene(config_variation)

            self.update_camera('default_camera',config_variation['camera_params']['default_camera'])
            self.action_tool.reset([0., -1., 0.])
            while True:
                random_pick_and_place(pick_num=4, pick_scale=0.005)
                center_object()
                try:
                    topo = self.get_topological_representation()
                except topology.InvalidGeometry as e:
                    continue
                if self.force_trivial and topo != topology.COMMON_KNOTS["trivial_knot"]:
                    continue
                break

            generated_configs.append(deepcopy(config_variation))
            generated_states.append(deepcopy(self.get_state()))


        return generated_configs, generated_states

    def compute_reward(self, action=None, obs=None, **kwargs):
        # obs = obs.flatten()
        if self.task == "STRAIGHT":
            reward = int(self._is_done())#np.linalg.norm(self.get_geoms()[-1,:]) - 0.5
        elif self.task == "KNOT":
            reward = self._is_done()-1
        elif "KNOT_ACTION" in self.task:

            if self.goal == self.get_topological_representation():
                reward = 1
            else:
                reward = 0
        else:
            raise NotImplementedError
        return reward

    def get_geoms(self,normalise=False):
        geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        x,y,z,theta = self.get_rope_frame()
        T_mat = np.array([
            [np.cos(theta),0,np.sin(theta),x],
            [0,1,0,y],
            [-np.sin(theta),0,np.cos(theta),z],
            [0,0,0,1]
        ])

        base_aligned_rope = (np.linalg.inv(T_mat) @ np.vstack([geoms.T,np.ones(geoms.shape[0])]))[:3,:].T
        if normalise:
            base_aligned_rope = np.vstack([base_aligned_rope[0,:],base_aligned_rope[1:,:]-base_aligned_rope[:-1,:]])
            base_aligned_rope = rescale(base_aligned_rope,-RopeKnotEnv.ROPE_LINK_LENGTH,RopeKnotEnv.ROPE_LINK_LENGTH,-1,1)

        return base_aligned_rope

    def get_rope_frame(self):
        geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:2, :3]

        x = geoms[0,0]
        y = geoms[0,1] # nominally 0.1
        z = geoms[0,2]

        theta = np.arctan2(geoms[1,1]-geoms[0,1],geoms[1,0]-geoms[0,0])

        return x,y,z,theta

    def get_topological_representation(self):
        
        return topology.RopeTopology.from_geometry(self.get_geoms(),plane_normal=np.array([0,1,0],ndmin=2).T)

    def _is_done(self):
        if "KNOT" in self.task:
            return topology.RopeTopology.is_equivalent(self.get_topological_representation(),self.goal,False,False)
        elif self.task == "STRAIGHT":
            return np.linalg.norm(self.get_geoms()[-1,:]) > self.goal
        else:
            raise NotImplementedError(f"Cannot determine completion criteria for task {self.task}")
 
    def _step(self,action):
        for a in action:
            self.simulate_action(a)
    def simulate_action(self, action,reset:bool=False) -> float:
        curr_pos = pyflex.get_positions().reshape(-1, 4)


        self.previous_topology = self.get_topological_representation()
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


        scale_min = np.hstack([np.zeros((1,1)),np.tile(self.workspace[:,0].T,(1,(action.size - 1) // 2))])
        scale_max = np.hstack([np.ones((1,1)),np.tile(self.workspace[:,1].T,(1,(action.size - 1) // 2))])
        action = rescale(action,-1,1,scale_min,scale_max).flatten()

        pick_idx = round((action[0] * (rel_positions_h.shape[1]-1)))
        pick_coords_rel_h = np.expand_dims(rel_positions_h[:,pick_idx],-1)

        waypoints_rel = np.array(action[1:]).reshape([-1,2]).T

        waypoints_rel_h = np.vstack([waypoints_rel[0,:],np.zeros(waypoints_rel.shape[1]),waypoints_rel[1,:],np.zeros(waypoints_rel.shape[1])]) # Making last row zeros instead of ones for the homogeneous as the ones will come from the addition on next line.
        waypoints_h = pick_coords_rel_h + waypoints_rel_h

        pick_coords = (T_mat @ pick_coords_rel_h)[0:3]
        waypoint_coords = (T_mat @ waypoints_h)[0:3]

        traj = np.expand_dims(simple_trajectory(np.hstack([pick_coords,waypoint_coords]).T,height=0.1,num_points_per_leg=50),0) # only a single picker
        self.action_tool.step(traj,renderer=self.render if not self.headless else lambda *args, **kwargs : None)

        self.maybe_disturb_rope(ref_idx=pick_idx,steps=50)

        reward = self.compute_reward(action)
        if reset:
            pyflex.set_positions(curr_pos.flatten())
        return reward

    def get_desired_action(self,topo:topology.RopeTopology):
        # if not hasattr(self,"asdf"):
        #     self.asdf = random.choice(topo.get_valid_add_R1())
        # return self.asdf
        return topology.RopeTopologyAction("+R1",0,chirality=1,starts_over=True)

    def _get_obs(self):
        geoms = self.get_geoms(normalise=True)[1:,:]
        x,y,z,theta = self.get_rope_frame()

        if self.task == "KNOT":
            obs = np.hstack([np.array([x,y,z,theta]),geoms.flatten()])
        elif self.task == "STRAIGHT":
            obs = np.hstack([np.array([x,y,z,theta]),geoms.flatten()])
        elif self.task == "CORNER":
            obs = np.array([x,y,z,theta])
        elif self.task == "KNOT_ACTION_+R1":
            topo = self.get_topological_representation()
            possible_actions = topo.get_valid_add_R1()
            topo_action = self.get_desired_action(topo)
            if topo_action in possible_actions:
                action_params = topo_action.as_array
            else:
                action_params = random.choice(possible_actions).as_array
            obs = np.hstack([np.array([x,y,z,theta]),geoms.flatten(),action_params])
        else:
            raise NotImplementedError(f"Unknown observation required.")
        
        return np.expand_dims(obs,0)
    def get_obs(self):
        obs = self._get_obs()
        obs_flattened = obs.flatten()
        topo_action = topology.RopeTopologyAction(
            "+R1",
            int(obs_flattened[-3]),
            chirality=int(obs_flattened[-2]),
            starts_over=obs_flattened[-1] > 0
        )
        self.assign_goal(self.get_topological_representation().take_action(topo_action)[0])
        return obs

    def get_keypoints(self):
        particle_pos = self.get_geoms(True)
        return particle_pos#[self.key_point_indices, :3]

    def _get_info(self):
        return dict()

    def render_no_gripper(self,mode='rgb_array') -> cv2.Mat:
        picker_pos, particle_pos = self.action_tool._get_pos()
        # self.action_tool.step(np.array([1,1,1,1],ndmin=2),renderer = lambda *args,**kwargs : None)
        self.action_tool.set_picker_pos(np.array([1,1,1]))
        frame = cv2.cvtColor(super().render(mode=mode)[-self.camera_height:,:self.camera_width,:],cv2.COLOR_RGB2BGR)
        self.action_tool._set_pos(picker_pos,particle_pos)
        return frame

    def assign_goal(self,goal):
        self.goal = goal

    def get_cached_configs_and_states(self, cached_states_path, num_variations):
        """
        If the path exists, load from it. Should be a list of (config, states)
        :param cached_states_path:
        :return:
        """
        if self.cached_init_states is None:
            self.cached_configs = []
        if self.cached_init_states is None:
            self.cached_init_states = []

        if self.cached_configs is not None and self.cached_init_states is not None and len(self.cached_configs) == num_variations:
            return self.cached_configs, self.cached_init_states
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            cached_states_path = osp.join(cur_dir, '../cached_initial_states', cached_states_path)
        if self.use_cached_states and osp.exists(cached_states_path):
            # Load from cached file
            with open(cached_states_path, "rb") as handle:
                self.cached_configs, self.cached_init_states = pickle.load(handle)
            print('{} config and state pairs loaded from {}'.format(len(self.cached_init_states), cached_states_path))
            if len(self.cached_configs) == num_variations:
                return self.cached_configs, self.cached_init_states
            elif len(self.cached_configs) > num_variations:
                pairs = random.sample([list(z) for z in zip(self.cached_configs,self.cached_init_states)],k=num_variations)
                self.cached_configs = [p[0] for p in pairs]
                self.cached_init_states = [p[1] for p in pairs]
                return self.cached_configs, self.cached_init_states
            else:
                num_variations -= len(self.cached_configs)

        additional_cached_configs, additional_cached_init_states = self.generate_env_variation(num_variations)
        self.cached_configs.extend(additional_cached_configs)
        self.cached_init_states.extend(additional_cached_init_states)

        if self.save_cached_states:
            with open(cached_states_path, 'wb') as handle:
                pickle.dump((self.cached_configs, self.cached_init_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('{} config and state pairs generated and saved to {}'.format(len(self.cached_init_states), cached_states_path))

        return self.cached_configs, self.cached_init_states

    def disturb_rope(self,move_idx:int,amount:np.ndarray=np.array([0,5e-3,0]),steps:int=1):
        curr_pos = pyflex.get_positions().reshape(-1, 4)
        curr_pos[move_idx, :3] += amount
        pyflex.set_positions(curr_pos.flatten())
        for _ in range(steps):
            pyflex.step()

    def maybe_disturb_rope(self,ref_idx:int=0,steps:int=0) -> bool:
        '''
        Disturbs the rope only if it is in an invalid geometric state due to the rope phasing through itself.
        params
            ref_idx: A reference point that will determine which of the two indexes that pass through each other will be used.
                    Whichever is closest to the ref_idx will be used. This is useful if the rope passes through itself due to an action
                    as it attempts to achieve the physically correct state for the rope.
            steps: Number of simulation steps to perform after modifying the rope state to let it settle.
        returns a bool indicating if the rope state was modified.
        '''
        modified = False
        while True:
            try:
                self.get_topological_representation()
            except topology.InvalidGeometry as e:
                id1 = int(str(e).replace('.','').split(' ')[-1])
                id2 = int(str(e).replace('.','').split(' ')[-3])
                pick_id = id1 if abs(id1-ref_idx) < abs(id2-ref_idx) else id2
                self.disturb_rope(pick_id)
                modified = True
                continue
            break
        if modified:
            for _ in range(steps):
                pyflex.step()           
            # Just in case there is an incredibly unlikely event of it settling just right that it phases into itself again.
            self.maybe_disturb_rope(ref_idx,steps)
        return modified
######### Helper Functions


def rescale(x:np.ndarray,old_min:np.ndarray,old_max:np.ndarray,new_min:np.ndarray,new_max:np.ndarray):
    return (x - old_min) / (old_max-old_min) * (new_max-new_min) + new_min

