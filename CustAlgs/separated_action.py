import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn,  TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import softgym.utils.topology as topology

from shapely.geometry import LineString, Polygon,MultiPolygon

class FCN(BasePolicy):
    def __init__(
        self,
        conv_channel_sizes:List[int],
        deconv_channel_sizes:List[int],
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        # features_dim: int,
        # net_arch: Optional[List[int]] = None,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space = observation_space,
            action_space = action_space,
            features_extractor = features_extractor,
            # features_dim = features_dim,
            # net_arch = net_arch,
            # activation_fn = activation_fn,
            normalize_images = normalize_images,
        )

        self.convs = [
            nn.Conv2d(
                conv_channel_sizes[i-1],
                conv_channel_sizes[i],
                kernel_size=3,
                stride=1,
                padding=1,
            ) 
        for i in range(1,len(conv_channel_sizes))]
        self.deconvs = [
            nn.ConvTranspose2d(
                deconv_channel_sizes[i-1],
                deconv_channel_sizes[i],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        for i in range(1,len(deconv_channel_sizes))]
        self.q_net = nn.Sequential(*[*self.convs,*self.deconvs])

    def forward(self, x):
        return self.q_net(x.float())

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        return q_values
        # Greedy action
        # action = q_values.argmax(dim=1).reshape(-1)
        # return action

class SplitActionDQN(OffPolicyAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        q_net_order:List[int] = [0,1,2,3],
        buffer_size: int = 1_00,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            policy_base=DQNPolicy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            sde_support=False,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=(gym.spaces.Box,gym.spaces.Discrete),
        )
        self.q_net_order = q_net_order
        self._setup_model()

    # def predict(*args,**kwargs):
    #     pass


class FcnPolicy(DQNPolicy):
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        q_net_order=[0,1,2,3]
    ):
        self.q_net_order = q_net_order
        super().__init__(
            observation_space = observation_space,
            action_space = action_space,
            lr_schedule = lr_schedule,
            net_arch = net_arch,
            activation_fn = activation_fn,
            features_extractor_class = features_extractor_class,
            features_extractor_kwargs = features_extractor_kwargs,
            normalize_images = normalize_images,
            optimizer_class = optimizer_class,
            optimizer_kwargs = optimizer_kwargs,
        )
        s = 0.35
        w = 200
        h = 200
        self.homography,_ = cv2.findHomography(
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
        self.softmax = nn.Softmax2d()

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        net_params = [{
            "conv_channel_sizes": [3 + net_num,512,256,128,64,32],
            "deconv_channel_sizes": [32,64,128,256,512,1],
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "features_extractor": None,
        } for net_num in range(len(self.q_net_order))]

        self.q_FCNs = [FCN(**params) for params in net_params]
        self.target_q_FCNs = [FCN(**params) for params in net_params]
        for i in range(len(self.q_FCNs)):
            self.target_q_FCNs[i].load_state_dict(self.q_FCNs[i].state_dict())

        # Setup optimizer with initial learning rate
        self.q_optimizer = self.optimizer_class([param for q in self.q_FCNs for param in list(q.parameters())], lr=lr_schedule(1), **self.optimizer_kwargs)
        self.target_q_optimizer = self.optimizer_class([param for q in self.target_q_FCNs for param in list(q.parameters())], lr=lr_schedule(1), **self.optimizer_kwargs)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        for i in range(len(self.q_FCNs)):
            self.q_FCNs[i].set_training_mode(mode)
        #TODO: Repeat on target networks?
        self.training = mode

    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = True
    ) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, 
        obs: torch.Tensor, 
        topo_state:topology.RopeTopology,
        topo_action:topology.RopeTopologyAction,
        deterministic: bool = True,
    ) -> torch.Tensor:
        x = obs.clone()
        if self.training:
            pick_indices, regions, markers = watershed_regions(obs.shape[1:],topo_state,topo_action,self.homography,np.identity(4))

            for net ,marker_num in zip(self.q_FCNs,[PICK,PLACE,MID_2,MID_1]):
                y = net(x)
                y[markers != marker_num] = 0
                y = self.softmax(y)
                x = torch.cat([x,y],dim=1)
        for net in self.q_FCNs:
            y = net(x)
            x = torch.cat([x,y],dim=1)
        return x
################# Helper Functions ###########################


PICK  = 1
MID_1 = 2
MID_2 = 3
PLACE = 4
AVOID = 5

def watershed_regions(
    img_shape:np.ndarray,
    topo:topology.RopeTopology,
    topo_action:topology.RopeTopologyAction,
    homography:np.ndarray = np.identity(3),
    rope_frame_matrix:np.ndarray = np.identity(3)
) -> Tuple[List[int], List[np.ndarray], np.ndarray]:

    def shift_line(line:np.ndarray,amount:float,single_sided:bool=True) -> np.ndarray:
        seg = LineString(line.tolist())
        shifted = seg.buffer(amount,single_sided=single_sided)
        if type(shifted) == Polygon:
            coords = shifted.exterior.coords
        elif type(shifted) == MultiPolygon:
            coords = []
            for g in shifted.geoms:
                coords.extend(g.exterior.coords)
        else:
            raise NotImplementedError
        n_shifted = np.array(coords)
        return n_shifted.T
    def get_distance_mask(rope_img,max_dist,homography):
        pixel_dist = np.linalg.norm((homography @ np.array([max_dist,0,1]))-np.array([rope_img.shape[1]/2,rope_img.shape[0]/2,1]))

        dist_img = cv2.distanceTransform(255-rope_img[:,:,0],cv2.DIST_L2,3)
        mask = cv2.inRange(dist_img,0,pixel_dist)

        return mask

    # Get rope projection.
    rope = (rope_frame_matrix @ np.vstack([topo.geometry.T,np.ones(topo.geometry.shape[0])]))[[0,2],:]

    tmp = get_geometry_of_interest_from_topological_action(
        topo,
        topo_action,
        rope_geometry=rope,
    )
    over_indices,over_geometry,under_geometry,extra_geometry = tmp   


    # Shift segments slightly so that they can be used as seeds in the watershed algorithm.
    pick_region = over_geometry
    mid_1_seed = shift_line(over_geometry.T,5e-3*topo_action.chirality)
    mid_2_seed = shift_line(under_geometry.T,5e-3*topo_action.chirality)
    place_seed = shift_line(under_geometry.T,-5e-3*topo_action.chirality)
    avoid_seed = shift_line(over_geometry.T,-5e-3*topo_action.chirality)

    # Apply homography.
    pick_pixels  = transform_points(pick_region,homography)
    mid_1_pixels = transform_points(mid_1_seed,homography)
    mid_2_pixels = transform_points(mid_2_seed,homography)
    place_pixels = transform_points(place_seed,homography)
    avoid_pixels = transform_points(avoid_seed,homography)
    rope_pixels  = transform_points(rope,homography)
    extra_geometry_pixels = [transform_points(shift_line(extra.T,5e-3,single_sided=False),homography) for extra in extra_geometry]

    # Create distance mask to limit range of watershed.
    rope_img = np.zeros(img_shape,dtype=np.uint8)
    rope_img = cv2.polylines(
        rope_img,
        [rope_pixels.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=255
    )
    mask = get_distance_mask(rope_img,0.1,homography)

    # Draw the watershed seeds onto a blank image.
    markers = np.zeros(img_shape[:2],dtype=np.int32)
    for pixels,seed_num in zip([mid_1_pixels,mid_2_pixels,place_pixels,avoid_pixels],[MID_1,MID_2,PLACE,AVOID]):
        markers = cv2.polylines(
            markers,
            [pixels.T.astype(np.int32)],
            isClosed=False,
            thickness=1,
            color=seed_num
        )
    for i in range(len(extra_geometry_pixels)):
        markers = cv2.polylines(
            markers,
            [extra_geometry_pixels[i].T.astype(np.int32)],
            isClosed=False,
            thickness=1,
            color=AVOID + 1 + i
        )

    
    # Watershed and distance masking.
    markers = cv2.watershed(rope_img,markers.astype(np.int32))
    markers = cv2.bitwise_and(markers,markers,mask=mask)

    # Convert pixel regions back into world regions.
    world_regions = []
    for region_num in range(MID_1,AVOID):
        contours, _ = cv2.findContours((markers == region_num).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cr = np.hstack([np.array(contour).squeeze(1).T for contour in contours])
        world_regions.append((np.linalg.inv(homography) @ np.vstack([cr,np.ones(cr.shape[1])]))[:-1,:])

    markers = cv2.polylines(
        markers,
        [pick_pixels.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=PICK
    )


    return over_indices, world_regions, markers

def get_geometry_of_interest_from_topological_action(
    topo:topology.RopeTopology,
    topo_action:topology.RopeTopologyAction,
    rope_geometry:Optional[np.ndarray] = None
) -> Tuple[List[int],np.ndarray, np.ndarray, List[np.ndarray]]:

    if rope_geometry == None:
        rope_geometry = topo.geometry

    if topo_action.under_seg is not None:
        over_indices = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        under_indices = topo.find_geometry_indices_matching_seg(topo_action.under_seg)
        if len(over_indices) == 1:
            over_indices.append(over_indices[0])
        if len(under_indices) == 1:
            under_indices.append(under_indices[0])
        over_geometry = rope_geometry[:,over_indices]
        under_geometry = rope_geometry[:,under_indices]
    else:
        segment_idxs = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        if len(segment_idxs) == 1:
            segment_idxs.append(segment_idxs[0])
        l = len(segment_idxs)
        over_indices = segment_idxs[:l//2]
        under_indices = segment_idxs[l//2:]

        over_geometry = rope_geometry[:,over_indices]
        under_geometry = rope_geometry[:,under_indices]

        if l < 4:
            mid_point = (over_geometry[:,-1:] + under_geometry[:,:1])*0.5
            over_geometry = np.hstack([over_geometry,mid_point])
            under_geometry = np.hstack([under_geometry,mid_point])
        
        if not topo_action.starts_over:
            over_geometry,under_geometry = under_geometry,over_geometry

    extra_geometry = []
    for seg_num in range(topo.size + 1):
        if seg_num != topo_action.over_seg and seg_num != topo_action.under_seg:
            extra_geometry.append(rope_geometry[:,topo.find_geometry_indices_matching_seg(seg_num)])

    return over_indices,over_geometry,under_geometry,extra_geometry        

def transform_points(points:np.ndarray,transformation_matrix:np.ndarray) -> np.ndarray:
    return (transformation_matrix @ np.vstack([points,np.ones(points.shape[1])]))[:-1,:]

def regions_to_normal_params(geometry,pick_indices,regions) -> Tuple[List[float], List[float]]:   
    l = len(pick_indices)
    pick_mu = pick_indices[l//2]
    pick_mu_pos = np.expand_dims(rescale(geometry[:,pick_mu],-0.35,0.35,-1,1)[[0,2]],1)

    mu = [rescale(pick_mu,0,40,-1,1)]
    std = [rescale(pick_mu-pick_indices[0],0,40,-1,1)]
    
    for region in regions:
        region = rescale(region,-0.35,0.35,-1,1)
        region = region - pick_mu_pos
        mu_x,mu_y = np.mean(region,axis=1)
        std_x,std_y = np.max(region,axis=1)-np.min(region,axis=1)
        mu.extend([mu_x,mu_y])
        std.extend([std_x,std_y])
    
    return mu, std

def rescale(
    x:Union[float,np.ndarray],
    old_min:Union[float,np.ndarray],
    old_max:Union[float,np.ndarray],
    new_min:Union[float,np.ndarray],
    new_max:Union[float,np.ndarray]
) -> Union[float,np.ndarray]:
    return (x - old_min) / (old_max-old_min) * (new_max-new_min) + new_min

def create_rope_mask(
    geometry:np.ndarray,
    homography:np.ndarray,
    indices:Optional[List[int]] = None,
    size:Optional[List[int]] = [720,720],
    dtype:Optional[Type] = np.uint8
) -> np.ndarray:
    if indices == None:
        indices = np.arange(geometry.shape[1])
    img = np.zeros(size,dtype=dtype)
    rope_pixel_coords = transform_points(geometry,homography)

    return cv2.polylines(
        img,
        [rope_pixel_coords[:,indices].T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=255
    )

def normalise_action(action:np.ndarray) -> np.ndarray:
    normed_act = action.copy()
    normed_act[0] = rescale(normed_act[0],0,40,-1,1)
    normed_act[1:] = rescale(normed_act[1:],-0.35,0.35,-1,1)
    return normed_act
def denormalise_action(action:np.ndarray) -> np.ndarray:
    denormed_act = action.copy()
    denormed_act = np.clip(denormed_act,-1,1)
    denormed_act[1:] = rescale(denormed_act[1:],-1,1,-0.35,0.35)
    denormed_act[0] = round(rescale(denormed_act[0],-1,1,0,40))
    return denormed_act

def draw_markers_on_image(base_img:np.ndarray,markers:np.ndarray) -> np.ndarray:
    assert np.all(base_img.shape[:2] == markers.shape), f"First 2 dimensions must match for inputs. Base image is size {base_img.shape} and markers is size {markers.shape}"    

    rope_img = np.zeros_like(base_img)
    rope_img[markers == PICK ] = (0,255,0)
    rope_img[markers == MID_1] = (104,43,159)
    rope_img[markers == MID_2] = (0,0,255)
    rope_img[markers == PLACE] = (255,0,0)
    rope_img[markers >= AVOID] = (208,224,64)

    return cv2.addWeighted(rope_img,0.5,base_img,0.5,0)

def normed_action_to_waypoints(
    action:np.ndarray,
    rope_geometry:np.ndarray,
    transform_mat:np.ndarray = np.identity(3)
) -> np.ndarray:
    return denormed_action_to_waypoints(denormalise_action(action),rope_geometry,transform_mat)

def denormed_action_to_waypoints(
    action:np.ndarray,
    rope_geometry:np.ndarray,
    transform_mat: np.ndarray = np.identity(3)
) -> np.ndarray:

    pick_coords_rel = rope_geometry[:,action[0]]
    waypoints_rel = np.hstack([pick_coords_rel.reshape((2,1)),pick_coords_rel.reshape((2,1))+np.array(action[1:]).reshape((-1,2)).T])
    return transform_points(waypoints_rel, transform_mat)

def image_waypoints_to_normed_action(waypoints,rope_geometry,transform_mat):
    waypoints_rope_frame = transform_points(waypoints,transform_mat)
    waypoints_rel = waypoints_rope_frame[:,1:] - waypoints_rope_frame[:,0]

    pick_index = np.argmin(np.linalg.norm(rope_geometry - waypoints_rope_frame[0]))

    action = np.hstack([pick_index,waypoints_rel.reshape(-1)])
    return normalise_action(action)

def draw(
    base_img:np.ndarray,
    region_markers:np.ndarray,
    rope_geometry:np.ndarray,
    rope_frame_mat:np.ndarray,
    homography:np.ndarray,
    action:np.ndarray,
    prior_action:Optional[np.ndarray] = None,
    are_actions_normed:bool=False,
) -> np.ndarray:

    painted_image = draw_markers_on_image(base_img,region_markers)

    if are_actions_normed:
        waypoints = normed_action_to_waypoints(action,rope_geometry,homography @ rope_frame_mat)
    else:
        waypoints = denormed_action_to_waypoints(action,rope_geometry,homography @ rope_frame_mat)

    # Draw Action
    painted_image = cv2.polylines(
        painted_image,
        [waypoints.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color = (0,0,0)
    )

    # Draw prior reference if available
    if prior_action is not None:
        if are_actions_normed:
            prior_waypoints = normed_action_to_waypoints(prior_action,rope_geometry,homography @ rope_frame_mat)
        else:
            prior_waypoints = denormed_action_to_waypoints(prior_action,rope_geometry,homography @ rope_frame_mat)

        painted_image = cv2.polylines(
            painted_image,
            [prior_waypoints.T.astype(np.int32)],
            isClosed=False,
            thickness=3,
            color = (127,127,127)
        )

    # Indicate start of the rope.
    rope_pixels = transform_points(rope_geometry,homography @ rope_frame_mat)
    painted_image = cv2.polylines(
        painted_image,
        [rope_pixels[:,:3].T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=(0,215,255)
    )

    return painted_image



     