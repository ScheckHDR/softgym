import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import cv2
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn,  TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ContinuousCritic
from torch.distributions.normal import Normal

import wandb
import softgym.utils.topology as topology
from shapely.geometry import LineString, Polygon,MultiPolygon


class TopologyMix(SAC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        goal = self.env.get_attr("goal")[0]
        topo_state = self.env.env_method("get_topological_representation")[0]
        img=self.env.env_method("render_no_gripper")[0]

        obs = observation.flatten()

        topo_action = topology.RopeTopologyAction(
            "+R1",
            int(obs[-3]),
            chirality=int(obs[-2]),
            starts_over=obs[-1] > 0
        )

        # topo_plan = topology.find_topological_path(topo_state,goal,goal.size)
        pick_idxs, pick_region,mid_region,place_region = topology.topo_to_geometry(topo_state,action=topo_action)

        pick_region = rescale(pick_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)
        mid_region = rescale(mid_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)
        place_region = rescale(place_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)

        prior_mus, prior_stds = prior_regions_to_normal_params(pick_idxs, pick_region,mid_region,place_region)

        policy_mus_full, policy_log_stds,_ = self.actor.get_action_dist_params(torch.tensor(observation).to("cuda"))
        policy_stds_full = torch.exp(policy_log_stds)

        padding = super().predict(observation,state,episode_start,deterministic)[0].shape[1] - 5
        policy_mus = torch.cat([policy_mus_full[0,:1],policy_mus_full[0,1+padding:]])
        policy_stds = torch.cat([policy_stds_full[0,:1],policy_stds_full[0,1+padding:]])

        combined_mus = (prior_mus*policy_stds**2 + policy_mus*prior_stds**2)/(policy_stds**2 + prior_stds**2)
        combined_stds = torch.sqrt((policy_stds**2 * prior_stds**2)/(prior_stds**2 + policy_stds**2))

        # Insert unbiased points
        combined_mus = torch.cat([combined_mus[:1],policy_mus_full[0,1:1+padding],combined_mus[1:]])
        combined_stds = torch.cat([combined_stds[:1],policy_stds_full[0,1:1+padding],combined_stds[1:]])

        combined_normal = Normal(combined_mus,combined_stds)



        # For visualising.
        x,y,z,theta = self.env.env_method("get_rope_frame")[0]
        T_mat = np.array([
            [np.cos(theta),np.sin(theta),x],
            [-np.sin(theta),np.cos(theta),z],
            [0,0,1]
        ])
        show_prior_on_image(
            img,
            rescale(pick_region,-1,1,-0.35,0.35),
            rescale(mid_region,-1,1,-0.35,0.35),
            rescale(place_region,-1,1,-0.35,0.35),
            lambda x: T_mat@x
        )

        # policy_prediction = super().predict(observation,state,episode_start,deterministic)[0]
        # prior_prediction = Normal(prior_mus,prior_stds).sample().cpu().numpy()
        if deterministic:
            combined_prediction = combined_normal.loc.detach().cpu().numpy()
        else:
            combined_prediction = combined_normal.rsample().detach().cpu().numpy()

        # topology.topo_to_geometry(topo_state,action=topo_action)
        return combined_prediction,None


        



    def _get_model_dist(self):
        return None

class QT_OPT(SAC):
    def __init__(
        self,
        # N:int=15,
        # M:int=5,
        # num_iter:int=1,
        *args,
        **kwargs
    ):
        N = 15
        M = 5
        num_iter = 2
        assert 0 < M <= N, f"M must be a positive interger less than N."
        assert num_iter > 0, f"num_iter must be a positive integer."

        super().__init__(*args,**kwargs)

        self.N = N
        self.M = M
        self.num_iter = num_iter

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, 
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        topo_state = self.env.env_method("get_topological_representation")[0]
        # img=self.env.env_method("render_no_gripper")[0]

        obs = observation.flatten()

        topo_action = topology.RopeTopologyAction(
            "+R1",
            int(obs[-3]),
            chirality=int(obs[-2]),
            starts_over=obs[-1] > 0
        )

        # topo_plan = topology.find_topological_path(topo_state,goal,goal.size)
        pick_idxs, pick_region,mid_region,place_region = topology.topo_to_geometry(topo_state,action=topo_action)

        pick_region = rescale(pick_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)
        mid_region = rescale(mid_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)
        place_region = rescale(place_region,np.array([-0.35,-0.35]),np.array([0.35,0.35]),-1,1)

        prior_mu, prior_std = prior_regions_to_normal_params(pick_idxs, pick_region,mid_region,place_region)
        sampler = Normal(torch.tensor(prior_mu).to("cuda"),torch.tensor(prior_std).to("cuda"))

        obs_N = torch.tensor(obs).repeat(self.N,1).to("cuda")
        with torch.no_grad():
            for _ in range(self.num_iter):
                samples = sampler.sample_n(self.N)

                # Q_values = self.score_func(samples)
                Q_values = torch.max(*self.critic(obs_N.float(),samples.float()))

                best = samples[torch.argsort(Q_values.flatten())][:self.M,:]

                sampler = Normal(torch.mean(best,dim=0),torch.std(best,dim=0))

            samples = sampler.sample_n(self.N)
            # Q_values = self.score_func(samples)
            Q_values = torch.max(*self.critic(obs_N.float(),samples.float()))
            
        return samples[torch.argsort(Q_values)][0].detach().cpu().numpy(), None

    def score_func(self,actions):
        return [self.env.env_method("simulate_action",a,True) for a in actions]



def shift_line(line:np.ndarray,amount:float) -> np.ndarray:
    seg = LineString(line.tolist())
    shifted = seg.buffer(amount,single_sided=True)
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

def get_distance_mask(rope_img,max_dist):

    # Create homography
    h = rope_img.shape[0]
    w = rope_img.shape[1]
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

    pixel_dist = np.linalg.norm((homography @ np.array([max_dist,0,1]))-np.array([w/2,h/2,1]))

    dist_img = cv2.distanceTransform(255-rope_img[:,:,0],cv2.DIST_L2,3)
    mask = cv2.inRange(dist_img,0,pixel_dist)

    return mask
def watershed_regions(img,topo:topology.RopeTopology,topo_action:topology.RopeTopologyAction,rope_frame):
    
    # Create homography
    h = img.shape[0]
    w = img.shape[1]
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

    # Get raw position of rope.
    x,y,z,theta = rope_frame
    T_mat = np.array([
        [np.cos(theta),np.sin(theta),x],
        [-np.sin(theta),np.cos(theta),z],
        [0,0,1]
    ])
    rope = topo.geometry[:,[0,2]].T
    rope_h = np.vstack([rope,np.ones(rope.shape[1])])
    rope_img_p = (homography @ T_mat @ rope_h)[:2,:]

    # Find geometry of interest from the topological action
    segment_idxs = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
    if len(segment_idxs) == 1:
        segment_idxs.append(segment_idxs[0])
    l = len(segment_idxs)
    over_indices = segment_idxs[l//2:]
    under_indices = segment_idxs[:l//2]
    if not topo_action.starts_over:
        over_indices,under_indices = under_indices,over_indices
    
    # Shift segments slightly so that they can be used as seeds for watershed algorithm.
    pick_region = topo.geometry[over_indices,:][:,[0,2]].T
    mid_1_seed = shift_line(topo.geometry[over_indices ,:][:,[0,2]],0.005*topo_action.chirality)
    mid_2_seed = shift_line(topo.geometry[under_indices ,:][:,[0,2]],-0.005*topo_action.chirality)
    place_seed = shift_line(topo.geometry[under_indices ,:][:,[0,2]],0.005*topo_action.chirality)
    avoid_seed = shift_line(topo.geometry[over_indices ,:][:,[0,2]],-0.005*topo_action.chirality)

    # Apply homography.
    pick_h  = np.vstack([pick_region, np.ones(pick_region.shape[1] )])
    mid_1_h = np.vstack([mid_1_seed,np.ones(mid_1_seed.shape[1])])
    mid_2_h = np.vstack([mid_2_seed,np.ones(mid_2_seed.shape[1])])
    place_h = np.vstack([place_seed,np.ones(place_seed.shape[1])])
    avoid_h = np.vstack([avoid_seed,np.ones(avoid_seed.shape[1])])
    rope_h  = np.vstack([rope, np.ones(rope.shape[1] )])
    pick_img_p  = (homography @ T_mat @ pick_h )[:2,:]
    mid_1_img_p = (homography @ T_mat @ mid_1_h)[:2,:]
    mid_2_img_p = (homography @ T_mat @ mid_2_h)[:2,:]
    place_img_p = (homography @ T_mat @ place_h)[:2,:]
    avoid_img_p = (homography @ T_mat @ avoid_h)[:2,:]
    rope_img_p  = (homography @ T_mat @ rope_h )[:2,:]

    # Create distance mask to limit range of watershed.
    rope_img = np.zeros_like(img)
    rope_img = cv2.polylines(
        rope_img,
        [rope_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=255
    )
    mask = get_distance_mask(rope_img,0.1)

    # Draw the watershed seeds onto a blank image
    markers = np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
    markers = cv2.polylines(
        markers,
        [mid_1_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=1
    )
    markers = cv2.polylines(
        markers,
        [mid_2_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=2
    )
    markers = cv2.polylines(
        markers,
        [place_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=3
    )
    markers = cv2.polylines(
        markers,
        [avoid_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=4
    )

    # Watershed and distance masking
    markers = cv2.watershed(rope_img,markers.astype(np.int32))
    markers = cv2.bitwise_and(markers,markers,mask=mask)

    # rope_img[markers == 1] = (104,43,159)
    # rope_img[markers == 2] = (255,0,0)
    # rope_img[markers == 3] = (0,0,255)
    # rope_img[markers == 4] = (208,224,64)
   
    # rope_img = cv2.polylines(
    #     rope_img,
    #     [pick_img_p.T.astype(np.int32)],
    #     isClosed=False,
    #     thickness=3,
    #     color = (0,255,0)
    # )
    def simple_blob_gaussian(img):
        rows,cols = np.where(img)
        coords_h = np.vstack([rows,cols,np.ones(len(rows))])
        coords = (np.linalg.inv(homography) @ coords_h)[:2,:]
        mu_x,mu_y = np.mean(coords,axis=1)
        std_x,std_y = np.max(coords,axis=1)-np.min(coords,axis=1)
        return mu_x,mu_y,std_x,std_y

    picks = np.array(over_indices)[[0,len(over_indices)//2]] / 40
    picks = rescale(picks,0,1,-1,1)
    pick_mu = picks[1]
    pick_std = (picks[1]-picks[0])
    
    mu,std = [pick_mu],[pick_std]
    for region in range(1,4): # don't care about the 'avoid' area.
        region_dist = simple_blob_gaussian(markers == region)
        mu.extend(region_dist[:2])
        std.extend(region_dist[2:])
    
    return markers, mu, std

        




########################## Helper Functions
def rescale(x:np.ndarray,old_min:np.ndarray,old_max:np.ndarray,new_min:np.ndarray,new_max:np.ndarray):
    return (x - old_min) / (old_max-old_min) * (new_max-new_min) + new_min

def prior_regions_to_normal_params(pick_idxs, pick_region,mid_region,place_region):
    picks = np.array(pick_idxs)[[0,len(pick_idxs)//2]] / 40
    picks = rescale(picks,0,1,-1,1)
    pick_mu = picks[1]
    pick_std = (picks[1]-picks[0])

    pick_mid_abs = pick_region[pick_region.shape[0]//2,:]
    mid_region_rel = mid_region - pick_mid_abs
    place_region_rel = place_region - pick_mid_abs

    mid_x_mu, mid_y_mu = np.mean(mid_region_rel,axis=0)
    mid_x_std, mid_y_std = (np.max(mid_region,axis=0) - np.min(mid_region,axis=0))

    place_x_mu, place_y_mu = np.mean(place_region_rel,axis=0)
    place_x_std, place_y_std = (np.max(place_region,axis=0) - np.min(place_region,axis=0))

    prior_mus = torch.tensor(np.array([pick_mu,mid_x_mu,mid_y_mu,place_x_mu,place_y_mu])).to("cuda")
    prior_stds = torch.tensor(np.array([pick_std,mid_x_std,mid_y_std,place_x_std,place_y_std])/wandb.config.prior_factor).to("cuda")
    prior_stds = torch.max(prior_stds,torch.ones_like(prior_stds)*0.01)

    return prior_mus,prior_stds

def show_prior_on_image(img:np.ndarray,pick_region:np.ndarray,mid_region:np.ndarray,place_region:np.ndarray,data_transform = lambda x: x):
    h = img.shape[0]
    w = img.shape[1]
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

    # Ensure regions are column vectors
    if pick_region.shape[0] != 2:
        pick_region = pick_region.T
    assert pick_region.shape[0] == 2
    if mid_region is None:
        mid_region = np.array([0,0],ndmin=2)
    if mid_region.shape[0] != 2:
        mid_region = mid_region.T
    assert mid_region.shape[0] == 2
    if place_region.shape[0] != 2:
        place_region = place_region.T
    assert place_region.shape[0] == 2

    pick_h = np.vstack([pick_region,np.ones(pick_region.shape[1])])
    mid_h = np.vstack([mid_region,np.ones(mid_region.shape[1])])
    place_h = np.vstack([place_region,np.ones(place_region.shape[1])])

    pick_img_p = (homography @ data_transform(pick_h))[:2,:]
    mid_img_p = (homography @ data_transform(mid_h))[:2,:]
    place_img_p = (homography @ data_transform(place_h))[:2,:]



    pick_frame = cv2.polylines(
        np.zeros_like(img),
        [pick_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=(0,255,0)
    )
    mid_frame = cv2.fillPoly(
        np.zeros_like(img),
        [mid_img_p.T.astype(np.int32)],
        color=(0,0,255)
    )
    place_frame = cv2.fillPoly(
        np.zeros_like(img),
        [place_img_p.T.astype(np.int32)],
        color=(255,0,0)
    )

    info_img = cv2.addWeighted(pick_frame,1,cv2.addWeighted(mid_frame,1,place_frame,1,0),1,0)
    img_final = cv2.addWeighted(info_img,0.5,img,0.5,0)

    # img_p = cv2.bitwise_or(img,pick_frame)
    # img_m = cv2.bitwise_or(img_p,mid_frame)
    # img_final = cv2.bitwise_or(img_m,place_frame)


    cv2.imshow(f"action_vis_{os.getpid()}",img_final)
    cv2.waitKey(1)

def traj_crosses_rope(
    traj:np.ndarray,
    rope:np.ndarray,
    plane_normal:np.ndarray=np.array([0,1,0],ndmin=2).T
) -> Union[np.ndarray,None]:
    assert traj.ndim == 2 and traj.shape[0] in [2,3], f"traj must be a matrix of size either Nx2 or Nx3."
    assert rope.ndim == 2 and rope.shape[0] in [2,3], f"rope must be a matrix of size either Nx2 or Nx3."
    assert not ((traj.shape[0] == 3 or rope.shape[0] == 3) and any(plane_normal.shape != [3,1])), f"normal vector must be shape 1x3."

    # Ensure normal vector is unit length.
    plane_normal = plane_normal/np.linalg.norm(plane_normal)

    # Project points onto the projection plane.
    e1 = np.cross(plane_normal.T,np.random.rand(*plane_normal.shape))
    e2 = np.cross(e1,plane_normal.T)
    if rope.ndim == 2:
        projected_rope = rope
    else:
        projected_rope = (np.vstack([e1,e2]) @ rope).T
    if traj.ndim == 2:
        projected_traj = traj
    else:
        projected_traj = (np.vstack([e1,e2]) @ traj).T

    for t in range(projected_traj.shape[0]-1):
        for r in range(projected_rope.shape[0]-1):
            if topology._intersect(projected_traj[:,t],projected_traj[:,t+1],projected_rope[:,r],projected_rope[:,r+1]):
                return get_intersection_point(projected_traj[:,t],projected_traj[:,t+1],projected_rope[:,r],projected_rope[:,r+1])

    return None

def get_intersection_point(A,B,C,D) -> Union[np.ndarray,None]:
    AB = B-A
    CD = D-C

    CA_diff = C-A
    AB_cross_CD = np.cross(AB,CD)
    
    t = np.cross(CA_diff,CD)/AB_cross_CD

    parallelity = np.cross(CA_diff,AB) # not exactly sure what this measures, guessing some form of parallelness based on the conditions later.
    u = parallelity/AB_cross_CD

    if abs(AB_cross_CD) < 1e-9 and abs(parallelity) < 1e-9:
        #collinear, check if overlapping or not
        AB_dot = np.dot(AB,AB)
        t0 = np.dot(CA_diff,AB)/AB_dot
        if 0 <= t0 <= 1:
            return A + t0*AB
        t1 = np.dot(CA_diff + CD,AB)/AB_dot
        if 0 <= t1 <= 1:
            return A + t1*AB
        return None
    elif abs(AB_cross_CD) < 1e-9 and abs(parallelity) > 1e-9:
        #parallel, non-intersecting
        return None
    elif abs(AB_cross_CD) > 1e-9 and 0 <= t <= 1 and 0 <= u <= 1:
        # intersect
        return A + t*AB
    else:
        # no intersect
        return None
def get_fourth_region(pick_pos) -> np.ndarray:

    pass


