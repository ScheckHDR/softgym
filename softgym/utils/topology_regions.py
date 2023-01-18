from typing import Tuple, List, Union, Iterable
import numpy as np
import random

import cv2
from shapely.geometry import LineString, Polygon,MultiPolygon

import softgym.utils.topology as topology


PICK  = 1
MID_1 = 2
MID_2 = 3
PLACE = 4
AVOID = 5

def watershed_regions(
    img:np.ndarray,
    topo:topology.RopeTopology,
    topo_action:topology.RopeTopologyAction,
    homography:np.ndarray,
    rope_frame_matrix:np.ndarray
) -> Tuple[List[int], List[np.ndarray], np.ndarray]:

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
    def get_distance_mask(rope_img,max_dist,homography):
        pixel_dist = np.linalg.norm((homography @ np.array([max_dist,0,1]))-np.array([rope_img.shape[1]/2,rope_img.shape[0]/2,1]))

        dist_img = cv2.distanceTransform(255-rope_img[:,:,0],cv2.DIST_L2,3)
        mask = cv2.inRange(dist_img,0,pixel_dist)

        return mask


    # rope,_ = topology._project_onto_plane(
    #     (rope_frame_matrix @ np.vstack([topo.geometry.T,np.ones(topo.geometry.shape[0])]))[:-1,:],
    #     np.array([0,1,0],ndmin=2).T 
    # )
    rope = (rope_frame_matrix @ np.vstack([topo.geometry.T,np.ones(topo.geometry.shape[0])]))[[0,2],:]

    # Find geometry of interest from the topological action
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
    for seg_num in range(topo.size):
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
    extra_geometry_pixels = [transform_points(extra,homography) for extra in extra_geometry]

    # Create distance mask to limit range of watershed.
    rope_img = np.zeros_like(img)
    rope_img = cv2.polylines(
        rope_img,
        [rope_pixels.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=255
    )
    mask = get_distance_mask(rope_img,0.1,homography)

    # Draw the watershed seeds onto a blank image.
    markers = np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
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


def draw(
    base_img:np.ndarray,
    region_markers:np.ndarray,
    rope_geometry:np.ndarray,
    rope_frame_mat:np.ndarray,
    action:np.ndarray,
    prior_mu:Union[np.ndarray,None] = None
) -> np.ndarray:

    # Create homography
    h = base_img.shape[0]
    w = base_img.shape[1]
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

    # rescale action
    action = np.clip(action,-1,1)
    action = rescale(action,-1,1,-0.35,0.35)
    action[0] = rescale(action[0],-0.35,0.35,0,1)

    # Draw regions
    rope_img = np.zeros_like(base_img)
    rope_img[region_markers == PICK ] = (0,255,0)
    rope_img[region_markers == MID_1] = (104,43,159)
    rope_img[region_markers == MID_2] = (0,0,255)
    rope_img[region_markers == PLACE] = (255,0,0)
    rope_img[region_markers == AVOID] = (208,224,64)
    painted_image = cv2.addWeighted(rope_img,0.5,base_img,0.5,0)

    # Decode action into waypoints
    pick_idx = round(action[0] * 40)
    pick_coords_rel = rope_geometry[:,pick_idx]
    waypoints_rel = np.hstack([pick_coords_rel.reshape((2,1)),pick_coords_rel.reshape((2,1))+np.array(action[1:]).reshape((-1,2)).T])
    waypoints = transform_points(waypoints_rel, homography @ rope_frame_mat)


    # Draw Action
    painted_image = cv2.polylines(
        painted_image,
        [waypoints.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color = (0,0,0)
    )

    # Draw prior reference if available
    if prior_mu is not None:
        prior_mu = np.clip(prior_mu,-1,1)
        prior_mu = rescale(prior_mu,-1,1,-0.35,0.35)
        prior_mu[0] = rescale(prior_mu[0],-0.35,0.35,0,1)
        waypoints_mu_rel = np.hstack([pick_coords_rel.reshape((2,1)),pick_coords_rel.reshape((2,1))+np.array(prior_mu[1:]).reshape((-1,2)).T])
        waypoints_mu = transform_points(waypoints_mu_rel, homography @ rope_frame_mat)
        painted_image = cv2.polylines(
            painted_image,
            [waypoints_mu.T.astype(np.int32)],
            isClosed=False,
            thickness=3,
            color = (127,127,127)
        )

    # Indicate start of the rope.
    rope_pixels = transform_points(rope_geometry,homography @ rope_frame_mat)
    return cv2.polylines(
        painted_image,
        [rope_pixels[:,:3].T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=(0,215,255)
    )



     