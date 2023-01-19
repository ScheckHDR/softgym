import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.buffers import ReplayBuffer
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

import gym
import torch
import torch.nn as nn

import softgym.utils.topology as topology

from shapely.geometry import LineString, Polygon,MultiPolygon


class Split_DQN:
    def __init__(self):
        h = 720#img.shape[0]
        w = 720#img.shape[1]
        s = 0.35
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

    def predict(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:

        topo_state = self.env.env_method("get_topological_representation")[0]
        topo_action = self.env.env_method("get_topological_action")[0]
        rope_frame = self.env.env_method("get_rope_frame")[0]

        x,y,z,theta = rope_frame
        T_mat = np.array([
            [np.cos(theta),0,np.sin(theta),x],
            [0,1,0,y],
            [-np.sin(theta),0,np.cos(theta),z],
            [0,0,0,1]
        ])

        pick_indices, regions, markers = watershed_regions(img_shape,topo_state,topo_action,self.homography,T_mat)

        #pick actions = pick indices
        pick_Q_values = self.pick_q(observation) # Assume Image?
        

        #place actions = sample place

        #mid_1 actions = sample mid_1

        #mid_2 actions = sample mid_2


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
    homography:np.ndarray,
    rope_frame_matrix:np.ndarray
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

    # Find geometry of interest from the topological action
    over_indices,over_geometry,under_geometry,extra_geometry = get_geometry_of_interest_from_topological_action(
        topo,
        topo_action,
        rope_geometry=rope,
    )
    if topo_action.under_seg is not None:
        over_indices = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        under_indices = topo.find_geometry_indices_matching_seg(topo_action.under_seg)
        if len(over_indices) == 1:
            over_indices.append(over_indices[0])
        if len(under_indices) == 1:
            under_indices.append(under_indices[0])
        over_geometry = rope[:,over_indices]
        under_geometry = rope[:,under_indices]
    else:
        segment_idxs = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        if len(segment_idxs) == 1:
            segment_idxs.append(segment_idxs[0])
        l = len(segment_idxs)
        over_indices = segment_idxs[:l//2]
        under_indices = segment_idxs[l//2:]

        over_geometry = rope[:,over_indices]
        under_geometry = rope[:,under_indices]

        if l < 4:
            mid_point = (over_geometry[:,-1:] + under_geometry[:,:1])*0.5
            over_geometry = np.hstack([over_geometry,mid_point])
            under_geometry = np.hstack([under_geometry,mid_point])
        
        if not topo_action.starts_over:
            over_geometry,under_geometry = under_geometry,over_geometry

    extra_geometry = []
    for seg_num in range(topo.size + 1):
        if seg_num != topo_action.over_seg and seg_num != topo_action.under_seg:
            extra_geometry.append(rope[:,topo.find_geometry_indices_matching_seg(seg_num)])
    


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



     