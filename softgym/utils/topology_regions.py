import numpy as np
import random

import cv2
from shapely.geometry import LineString, Polygon,MultiPolygon

import softgym.utils.topology as topology

def watershed_regions(
    img:np.ndarray,
    topo:topology.RopeTopology,
    topo_action:topology.RopeTopologyAction,
    homography:np.ndarray,
) -> np.ndarray:

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


    rope = topo.geometry.T[[0,2],:]

    # Find geometry of interest from the topological action
    if topo_action.under_seg is not None:
        over_indices = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        under_indices = topo.find_geometry_indices_matching_seg(topo_action.under_seg)
    else:
        segment_idxs = topo.find_geometry_indices_matching_seg(topo_action.over_seg)
        if len(segment_idxs) == 1:
            segment_idxs.append(segment_idxs[0])
        l = len(segment_idxs)
        over_indices = segment_idxs[l//2:]
        under_indices = segment_idxs[:l//2]
        if not topo_action.starts_over:
            over_indices,under_indices = under_indices,over_indices
    over_geometry = rope[:,over_indices]
    under_geometry = rope[:,under_indices]

    # Shift segments slightly so that they can be used as seeds in the watershed algorithm.
    pick_region = over_geometry
    mid_1_seed = shift_line(over_geometry.T,5e-3*topo_action.chirality)
    mid_2_seed = shift_line(under_geometry.T,-5e-3*topo_action.chirality)
    place_seed = shift_line(under_geometry.T,5e-3*topo_action.chirality)
    avoid_seed = shift_line(over_geometry.T,-5e-3*topo_action.chirality)

    # Apply homography.
    pick_pixels  = (homography @ np.vstack([pick_region,np.ones(pick_region.shape[1])]))[:-1,:]
    mid_1_pixels = (homography @ np.vstack([mid_1_seed,np.ones(mid_1_seed.shape[1])]))[:-1,:]
    mid_2_pixels = (homography @ np.vstack([mid_2_seed,np.ones(mid_2_seed.shape[1])]))[:-1,:]
    place_pixels = (homography @ np.vstack([place_seed,np.ones(place_seed.shape[1])]))[:-1,:]
    avoid_pixels = (homography @ np.vstack([avoid_seed,np.ones(avoid_seed.shape[1])]))[:-1,:]
    rope_pixels  = (homography @ np.vstack([rope,np.ones(rope.shape[1])]))[:-1,:]

    # Create distance mask to limit range of watershed.
    rope_img = np.zeros_like(img)
    rope_img = cv2.polylines(
        rope_img,
        [rope_pixels.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=255
    )
    mask = get_distance_mask(rope_img,0.1,homography)

    # Draw the watershed seeds onto a blank image.
    markers = np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
    for pixels,seed_num in zip([mid_1_pixels,mid_2_pixels,place_pixels,avoid_pixels],[1,2,3,4]):
        markers = cv2.polylines(
            markers,
            [pixels.T.astype(np.int32)],
            isClosed=False,
            thickness=1,
            color=seed_num
        )
    
    # Watershed and distance masking.
    markers = cv2.watershed(rope_img,markers.astype(np.int32))
    markers = cv2.bitwise_and(markers,markers,mask=mask)

    # Convert pixel regions back into world regions.
    world_regions = []
    for region_num in range(1,5):
        contours, _ = cv2.findContours(markers == region_num,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cr = np.array(contours).squeeze(0).squeeze(1).T[::-1,:]
        world_regions.append((np.linalg.inv(homography) @ np.vstack([cr,cr.shape[1]]))[:-1,:])

    return over_indices, world_regions, markers


    

