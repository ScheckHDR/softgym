from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
import pickle
import os.path as osp
import random
from tqdm import tqdm
from typing import List,Tuple,Dict,Union,Optional

import cv2
from gym.spaces import Box, MultiDiscrete
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from softgym.action_space.action_space import PickerTraj
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
from softgym.utils.trajectories import simple_trajectory

import topology
from topology.utils import transform_points,rescale
from topology import ActionMasking
from topology import Regions as TR

from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution


class RopeKnotEnv(RopeNewEnv):
    ROPE_LINK_LENGTH = 0.0135 # approximate length of a single link. Seems to be some ability to stretch.
    def __init__(self,
                 goals:List[topology.RopeTopology],
                 cached_states_path:str="rope_knot_init_states.pkl",
                  **kwargs
                 ):
        self.headless:bool = kwargs["headless"]
        kwargs["headless"] = True # Manually do the rendering with the overlay.
        super().__init__(picker_radius = 0.005,cached_states_path=cached_states_path,**kwargs)
        

        # Reset stuff
        self.allowed_initial_topologies:List[topology.RopeTopology] = kwargs.get("allowed_initial_topologies",None)
        self.random_goal_on_reset:bool = kwargs.get("random_goal_on_reset",False)
        self.num_reset_perturbations:int = kwargs.get("reset_perturbations",0)
        self.reset_pertubation_magnitude:float = kwargs.get("reset_pertubation_magnitude",0)
        assert not (self.num_reset_perturbations > 0 and self.reset_pertubation_magnitude == 0), f"Requesting {self.num_reset_perturbations} random perturbations on reset, but 0 magnitude has been selected."

        # Initial state generation
        self.force_trivial:bool = kwargs.get("force_trivial",False)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        # Observation stuff.
        self.num_rope_particles:int = self.cached_configs[0]["segment"] + 1
        dim:int = (self.num_rope_particles-1)*3 + topology.RopeTopologyAction("NA",0).as_array().size # Adds the extra dims needed for the topological action params.
        self.observation_space:Box = Box(low=-1,high=1,shape=(1,dim))

        # Action stuff
        self.grid_resize_factor:float = kwargs.get("grid_resize_factor",-1) # -1 will disable.
        self.grid_size:int = 15
        self.grid_length:float = kwargs.get("grid_length",0.5)
        self.action_space:MultiDiscrete = MultiDiscrete([self.num_rope_particles] + [self.grid_size**2]*3)
        self.allowed_action_types:List[str] = kwargs.get("allowed_action_types",["+C","+R1","+R2"])

        # Goal stuff
        if not isinstance(goals,Iterable):
            goals = [goals]
        self.goal_space = goals
        self.goal = self.goal_space[0]
        self.sub_goal = None

        self.action_tool.set_picker_pos(np.array([1,1,1]))
        self._desired_action = None
             
    def _reset(self):
        
        config = self.current_config
        self.rope_length = config["segment"] * config["radius"] * config["scale"]

        if hasattr(self, "action_tool"):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        self.deform_equivalent(self.num_reset_perturbations)

        obs = self._get_obs()
        return obs
    
    def verify_state(
            self,
            config:Optional[Dict]=None,
            state:Optional[NDArray[np.float32]]=None,
            reset_after_check:bool=False,
        ) -> bool:
        '''
        Verifies that a config and state matches allowed topologies.
        If no arguments are passed, then this function verifies the current state.
        '''
        if self.allowed_initial_topologies is None:
            return True

        if reset_after_check:
            old_config = self.current_config
            old_state = self.get_state()

        if config is not None:
            self.set_scene(config)
        if state is not None:
            self.set_state(state)
            self._step_no_action(steps=5) # Just in case.

        topo = self.get_topological_representation()

        if reset_after_check:
            self.set_scene(old_config)
            self.set_state(old_state)

        return topo in self.allowed_initial_topologies

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        
        generated_configs, generated_states = [], []

        if config is None:
            config = self.get_default_config()
        for _ in tqdm(range(num_variations)):
            config_variation = deepcopy(config)

            self.set_scene(config_variation)

            self.update_camera("default_camera",config_variation["camera_params"]["default_camera"])
            self.action_tool.reset([0., -1., 0.])
            while True:
                random_pick_and_place(pick_num=4, pick_scale=0.005)
                self._step_no_action(steps=50)
                center_object()
                try:
                    topo = self.get_topological_representation()
                except topology.InvalidGeometry as e:
                    continue
                except topology.InvalidTopology as e:
                    continue
                if not self.verify_state():
                    continue
                break

            generated_configs.append(deepcopy(config_variation))
            generated_states.append(deepcopy(self.get_state()))


        return generated_configs, generated_states

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        length = 0.6
        radius = 0.005
        config = {
            "init_pos": [0., 0., 0.],
            "stretchstiffness": 0.9,
            "bendingstiffness": 0.8,
            "radius": radius,
            "segment": int(length//radius) * 2,
            "mass": 0.5,
            "scale": 1,
            "camera_name": "default_camera",
            "camera_params": {"default_camera":
                                  {"pos": np.array([0, 0.85, 0]),
                                   "angle": np.array([0 * np.pi, -90 / 180. * np.pi, 0]),
                                   "width": self.camera_width,
                                   "height": self.camera_height}}
        }
        return config

    def compute_reward(self, action=None, obs=None, **kwargs):

        topo_state = self.get_topological_representation()
        if topo_state == self.goal:
            reward = 10.0
        elif topo_state == self.sub_goal:
            reward = 1.0
        else:
            reward = 0.0

        return reward

    def get_shape(self,normalise=False) -> NDArray[np.float32]:
        geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        T_mat = self.get_rope_frame(geoms)

        base_aligned_rope = transform_points(geoms.T,np.linalg.inv(T_mat)).T
        if normalise:
            base_aligned_rope = np.vstack([base_aligned_rope[0,:],base_aligned_rope[1:,:]-base_aligned_rope[:-1,:]])
            base_aligned_rope = rescale(base_aligned_rope,-RopeKnotEnv.ROPE_LINK_LENGTH,RopeKnotEnv.ROPE_LINK_LENGTH,-1,1)

        return base_aligned_rope
    
    def get_rope_frame(self,geoms=None,two_d:bool=False,theta_dim:Union[int,str]=1) -> NDArray[np.float32]:
        if geoms is None:
            geoms = np.array(pyflex.get_positions()).reshape([-1, 4])[:2, :3]
        if isinstance(theta_dim,str) and theta_dim.upper() in ['X','Y','Z']:
            theta_dim = ['X','Y','Z'].index(theta_dim.upper())
        assert theta_dim in [0,1,2], f"Cannot use dimension {theta_dim} in 3d environment."

        theta = np.arctan2(geoms[1,2]-geoms[0,2],geoms[1,0]-geoms[0,0])

        if theta_dim == 1: # Y weirdness.
            theta = -theta

        c = np.cos(theta)
        s = np.sin(theta)
        
        R_mat = np.array([
            [c,-s],
            [s, c],
        ])
        if two_d:
            used_dims = [0,1,2]
            used_dims.remove(theta_dim)
            trans_mat = geoms[0,used_dims]
            T_mat = np.insert(R_mat,2,trans_mat,axis=1)
            T_mat = np.insert(T_mat,2,np.array([0,0,1]),axis=0)
        else:
            R_mat = np.insert(R_mat,theta_dim,np.zeros(2),axis=0)
            R_mat = np.insert(R_mat,theta_dim,np.zeros(3),axis=1)
            T_mat = np.insert(R_mat,3,geoms[0,:3],axis=1)
            T_mat = np.insert(T_mat,3,np.array([0,0,0,1]),axis=0)
            T_mat[theta_dim,theta_dim] = 1
        return T_mat

    def get_topological_representation(self):
        while True:
            try:
                return topology.RopeTopology.from_geometry(self.get_shape(),plane_normal=np.array([0,1,0],ndmin=2).T)
            except topology.InvalidTopology:
                vec = np.random.uniform([-1,0,-1],[1,1,1])
                vec = vec/np.linalg.norm(vec) * 0.01
                self.deform_rope(
                    np.random.randint(0,self.current_config["segment"]+1),
                    vec,
                    steps=1
                )

    def _is_done(self):
        return self.get_topological_representation() == self.goal

    ############## Action stuff
    def _step(self,action):
        topo_action = self.get_desired_action()
        topo_state = self.get_topological_representation()
        self.assign_sub_goal(topo_state.take_action(topo_action))
        self.simulate_action(action,revert=False)

    def simulate_action(self, action,revert:bool=False) -> NDArray[np.float32]:
        curr_pos = pyflex.get_positions()

        world_waypoints,pick_idx = self._action_to_world_waypoints(action)

        traj = np.expand_dims(
            simple_trajectory(
                np.insert(world_waypoints,1,np.zeros(world_waypoints.shape[0]),axis=1),
                height=0.1,
                num_points_per_leg=50),
            0) # only a single picker
        if self.headless:
            renderer = lambda *args, **kwargs : None
        else:
            renderer = render_with_overlay(self.render,self.render_action_mask())
        self.action_tool.step(
            traj,
            renderer=renderer
        )

        self.maybe_deform_rope(ref_idx=pick_idx,steps=50)

        new_pos = pyflex.get_positions()
        if revert:
            pyflex.set_positions(curr_pos)
        
        self._desired_action = None
        return new_pos

    def sample_action(self,mask:Optional[NDArray[np.bool_]] = None) -> NDArray[np.int32]:
        """ Return a random action within the environment's action space.

        Keyword arguments:
        mask... TODO: What will this be exactly (the bool is a placeholder for now?       
        """
        return ActionMasking.sample(self.action_space,mask)

    def get_grid_world_coordinates(self) -> np.ndarray:
        grid = np.array([
            [x,y]
            for x in range(-(self.grid_size//2),self.grid_size//2+1)
            for y in range(-(self.grid_size//2),self.grid_size//2+1)
        ])

        T_mat = self.get_rope_frame(two_d=True)
        rope = self.get_shape()[:,[0,2]]
        mid_point = np.mean(rope,axis=0,keepdims=True)

        if self.grid_resize_factor > 0:
            grid_length = np.max(np.abs(rope-mid_point))*2
            grid_length *= self.grid_resize_factor
        else:
            grid_length = self.grid_length

        grid = mid_point + grid / (self.grid_size//2) * (grid_length/2)

        return transform_points(grid.T,T_mat).T

    def _action_to_world_waypoints(self,action) -> Tuple[np.ndarray,int]:
        if isinstance(self.action_space,Box):
            raise NotImplementedError
        elif isinstance(self.action_space, MultiDiscrete):
            pick_idx = action[0]
            T_mat = self.get_rope_frame(two_d=True)
            
            rope_world = transform_points(self.get_shape()[:,[0,2]].T,T_mat).T

            grid_world_coords = self.get_grid_world_coordinates()

            waypoints = np.vstack([rope_world[pick_idx,:],grid_world_coords[action[1:],:]])

        return waypoints, pick_idx

    def get_desired_action(self) -> topology.RopeTopologyAction:
        if self._desired_action is None:
            self._desired_action = topology.RopeTopologyAction("NA",0)
            topo_plan = topology.RopeTopologyPath.find(
                self.get_topological_representation(),
                goal=self.goal,
                allowed_action_types=self.allowed_action_types,
                max_rep_size=self.goal.size
            )
            if topo_plan is not None and len(topo_plan.actions) > 0:
                self._desired_action = topo_plan.actions[0]
        return self._desired_action

    def _step_no_action(self,steps:int=1) -> None:
        '''
        Steps through the simulator without performing any action. Used to let any dynamics settle.
        '''
        for _ in range(steps):
            pyflex.step()

    def _get_obs(self):
        geoms = self.get_shape(normalise=True)[1:,:]  
        obs = np.expand_dims(np.hstack([geoms.flatten(),self.get_desired_action().as_array()]),0)
        return obs
        
    def _get_info(self):
        return dict()

    def assign_goal(self,goal) -> None:
        self.goal = goal

    def assign_sub_goal(self,sub_goal) -> None:
        self.sub_goal = sub_goal

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
            cached_states_path = osp.join(cur_dir, "../cached_initial_states", cached_states_path)
        if self.use_cached_states and osp.exists(cached_states_path):
            # Load from cached file
            with open(cached_states_path, "rb") as handle:
                self.cached_configs, self.cached_init_states = pickle.load(handle)
            print("{} config and state pairs loaded from {}".format(len(self.cached_init_states), cached_states_path))
            for config,state in zip(self.cached_configs,self.cached_init_states):
                if not self.verify_state(config,state):
                    raise Exception("Loaded states did not satisfy topology requirements.")
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
            with open(cached_states_path, "wb") as handle:
                pickle.dump((self.cached_configs, self.cached_init_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("{} config and state pairs generated and saved to {}".format(len(self.cached_init_states), cached_states_path))

        return self.cached_configs, self.cached_init_states

    def deform_equivalent(self, num_deformations:int, num_tries:int=5, mag:float=0.01) -> bool:
        '''
            Applies random deformations to the current rope state while leaving it in the same topological class.

            Returns False if unable to deform the rope.
        '''
        num_particles = self.current_config["segment"] + 1
        curr_pos = pyflex.get_positions().reshape([-1, 4])
        topo_state_before = self.get_topological_representation()
        for _ in range(num_tries): # try num_tries times before giving up and leaving it in the initial position.
            for _ in range(num_deformations):
                vec = np.random.uniform([-1,0,-1],[1,1,1])
                vec = vec/np.linalg.norm(vec) * mag
                self.deform_rope(
                    np.random.randint(0,num_particles),
                    vec,
                    steps=1
                )

            self._step_no_action(steps=50)       
            self.maybe_deform_rope(steps=50)
                
            topo_state = self.get_topological_representation()
            if topo_state == topo_state_before:
                return True
            pyflex.set_positions(curr_pos.flatten())
        return False

    def deform_rope(self,move_idx:int,amount:np.ndarray=np.array([0,5e-3,0]),steps:int=1):
        curr_pos = pyflex.get_positions().reshape(-1, 4)
        curr_pos[move_idx, :3] += amount
        pyflex.set_positions(curr_pos.flatten())
        self._step_no_action(steps)

    def maybe_deform_rope(self,ref_idx:int=0,steps:int=0) -> bool:
        '''
        Deforms the rope only if it is in an invalid geometric state due to the rope phasing through itself.
        params
            ref_idx: A reference point that will determine which of the two indexes that pass through each other will be used.
                    Whichever is closest to the ref_idx will be used. This is useful if the rope passes through itself due to an action
                    as it attempts to achieve the physically correct state for the rope.
            steps: Number of simulation steps to perform after modifying the rope state to let it settle.
        returns a bool indicating if the rope state was modified.
        '''
        try:
            self.get_topological_representation()
        except topology.InvalidGeometry as e:
            id1 = int(str(e).replace('.','').split(' ')[-1])
            id2 = int(str(e).replace('.','').split(' ')[-3])
            pick_id = id1 if abs(id1-ref_idx) < abs(id2-ref_idx) else id2
            self.deform_rope(pick_id)
            self._step_no_action(steps) 
            return self.maybe_deform_rope(ref_idx,steps)
        return False

    ######### Rendering Functions
    def render_no_gripper(self,mode="rgb_array") -> cv2.Mat:
        picker_pos, particle_pos = self.action_tool._get_pos()
        self.action_tool.set_picker_pos(np.array([1,1,1]))
        frame = super().render(mode=mode)[:,:,::-1]
        self.action_tool._set_pos(picker_pos,particle_pos)
        return frame

    def render_action_mask(self,action:Optional[np.ndarray] = None,topological_action:Optional[topology.RopeTopologyAction]=None) -> np.ndarray:
        h = self.camera_height
        w = self.camera_width
        s = 0.35
        homography,_ = cv2.findHomography(
            np.array([
                [-s, s, s,-s],
                [-s,-s, s, s],
                [ 1, 1, 1, 1],

            ]).T,
            np.array([
                [0,w,w,0],
                [0,0,h,h],
                [1,1,1,1],
            ]).T
        )

        T_mat3D = self.get_rope_frame()
        T_mat2D = self.get_rope_frame(two_d=True)

        if topological_action is None:
            topological_action = self.get_desired_action()
        topo = self.get_topological_representation()
        pick_indices, regions, topology_mask = TR.watershed_regions(
            [h,w,3],
            topo,
            topological_action,
            homography,
            T_mat3D
        )

        overlay = np.zeros([h,w,3],dtype=np.uint8)
        overlay[topology_mask == TR.PICK ] = (0,255,0)
        overlay[topology_mask == TR.MID_1] = (159,43,104)
        overlay[topology_mask == TR.MID_2] = (255,0,0)
        overlay[topology_mask == TR.PLACE] = (0,0,255)
        overlay[topology_mask == TR.AVOID] = (64,224,208)

        rope_pixels = transform_points(self.get_shape()[:,[0,2]].T,homography @ T_mat2D).T.astype(np.int32)
        cv2.polylines(
            overlay,
            [rope_pixels[:3,:]],
            isClosed=False,
            color=(255,215,0),
            thickness=3
        )

        for g in transform_points(self.get_grid_world_coordinates().T,homography).T.astype(np.int32):
            cv2.circle(
                overlay,
                center=g,
                radius=1,
                color=(255,255,255),
                thickness=1,
            )

        if action is not None:
            action_coords = transform_points(self._action_to_world_waypoints(action)[0].T,homography).T.astype(np.int32)
            cv2.polylines(
                overlay,
                [action_coords],
                isClosed=False,
                color=(0,0,0),
                thickness=3
            )
        return overlay

def render_with_overlay(base_render,overlay=None):
    def overlay_image(*args,**kwargs):
        base = base_render(*args,**kwargs)
        combined = cv2.addWeighted(base,0.5,overlay,0.5,0)
        cv2.imshow("Softgym",combined[:,:,::-1])
        cv2.waitKey(1)
        return combined
    if callable(base_render):
        if overlay is None:
            return base_render
        else:
            return overlay_image
    else:
        return cv2.addWeighted(base_render,0.5,overlay,0.5,0)

