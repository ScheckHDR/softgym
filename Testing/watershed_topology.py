import sys
# from os.path import dirname
import cv2
import numpy as np
import random

sys.path.append("/home/jeffrey/Documents/GitHub/softgym")
from softgym.envs.rope_knot import RopeKnotEnv
import softgym.utils.topology as topology
import softgym.utils.topology_regions as TR
from shapely.geometry import LineString, Polygon,MultiPolygon
from typing import List, Tuple
# # watershed
# img = np.zeros((250,250,3),dtype=np.uint8)
# markers = np.zeros((250,250,1),dtype=np.int32)

# markers[10:20,10:30] = 10 #avoid
# markers[50:60,30:50] = 50 #1st intermediate
# markers[200:210,200:210] = 100 #2nd intermediate
# markers[100,100] = 150 # place point. 

# watershed_markers = cv2.watershed(img,markers)
# img[(markers == -1).squeeze(2),:] = 255#watershed_markers.astype(np.int8)
# for v in [10,50,100,150]:
#     img[(markers == v).squeeze(2),:] = v
# cv2.imshow("test",img)
# cv2.waitKey(0)
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
    mid_1_seed = shift_line(topo.geometry[over_indices ,:][:,[0,2]] , 0.005*topo_action.chirality)
    mid_2_seed = shift_line(topo.geometry[under_indices ,:][:,[0,2]],-0.005*topo_action.chirality)
    place_seed = shift_line(topo.geometry[under_indices ,:][:,[0,2]], 0.005*topo_action.chirality)
    avoid_seed = shift_line(topo.geometry[over_indices ,:][:,[0,2]] ,-0.005*topo_action.chirality)

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

    picks = np.array(over_indices)[[0,len(over_indices)//2]] / 40
    picks_scaled = rescale(picks,0,1,-0.35,0.35)
    pick_mu = picks_scaled[1]
    pick_std = (picks_scaled[1]-picks_scaled[0])
    # pick_pos = (np.linalg.inv(T_mat) @ np.array([*topo.geometry[len(over_indices)//2,[0,2]],1]))[:2].flatten()
    pick_pos = topo.geometry[over_indices[len(over_indices)//2],[0,2]].flatten()


    def simple_blob_gaussian(img):
        rows,cols = np.where(img)
        coords_h = np.vstack([rows,cols,np.ones(len(rows))])
        world_coords_h = np.linalg.inv(homography) @ coords_h
        rope_coords = (np.linalg.inv(T_mat) @ world_coords_h)[:2,:]
        mu_x,mu_y = np.mean(rope_coords,axis=1)
        std_x,std_y = np.max(rope_coords,axis=1)-np.min(rope_coords,axis=1)
        
        # print(f"image:{np.mean(rows)},{np.mean(cols)}")
        # print(f"world:{np.mean(world_coords_h.T,axis=0)}")
        # print(f"rope:{np.mean(rope_coords.T,axis=0)}\n")
        
        return mu_x,mu_y,std_x,std_y
    
    mu,std = [pick_mu],[pick_std]
    for region in range(1,4): # don't care about the 'avoid' area.
        region_dist = simple_blob_gaussian(markers == region)
        mu.extend((region_dist[:2] - pick_pos).tolist())
        std.extend(region_dist[2:])
    # print(f"pick_position:{pick_pos}")
    # print(f"before scaling:{mu}")
    mu = rescale(np.array(mu),-0.35,0.35,-1,1)
    std=rescale(np.array(std),-0.35,0.35,-1,1)    
    # draw_action(img,markers,mu,rope_frame)
    return markers, mu, std 

def rescale(x:np.ndarray,old_min:np.ndarray,old_max:np.ndarray,new_min:np.ndarray,new_max:np.ndarray):
    return (x - old_min) / (old_max-old_min) * (new_max-new_min) + new_min

def draw_action(img,markers,action,rope_frame) -> None:
    action = np.clip(action,-1,1)
    # Draw Regions
    rope_img = np.zeros_like(img)
    rope_img[markers == 1] = (104,43,159)
    rope_img[markers == 2] = (0,0,255)
    rope_img[markers == 3] = (255,0,0)
    rope_img[markers == 4] = (208,224,64)
    img = cv2.addWeighted(rope_img,0.5,img,0.5,0)

    action = rescale(action,-1,1,-0.35,0.35)
    action[0] = rescale(action[0],-0.35,0.35,0,1)

    pick_idx = round(action[0] * 40)
    pick_coords_rel = topo.geometry[pick_idx,[0,2]]
    waypoints_rel = np.hstack([pick_coords_rel.reshape((2,1)),pick_coords_rel.reshape((2,1))+np.array(action[1:]).reshape((-1,2)).T])
    x,y,z,theta = rope_frame
    T_mat = np.array([
        [np.cos(theta),np.sin(theta),x],
        [-np.sin(theta),np.cos(theta),z],
        [0,0,1]
    ])

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
    waypoints = (homography @ T_mat @ np.vstack([waypoints_rel,np.ones(waypoints_rel.shape[1])]))[:2,:]
    waypoints[:,1:] = waypoints[::-1,1:]
    # print(waypoints)
    img = cv2.polylines(
        img,
        [waypoints.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color = (0,0,0)
    )
    cv2.imshow("Action",img)
    cv2.waitKey(0)


def regions_to_normal_params(regions) -> Tuple[List[float], List[float]]:
    mu,std = [],[]
    for region in regions:
        mu_x,mu_y = np.mean(region,axis=0)
        std_x,std_y = np.max(region,axis=0)-np.min(region,axis=0)
        mu.extend([mu_x,mu_y])
        std.extend([std_x,std_y])
    
    return mu, std

env_kwargs = {
    "observation_mode": "key_point",
    "action_mode": "picker_trajectory",
    "num_picker": 1,
    "render": True,
    "headless": True,
    "horizon": 5,
    "action_repeat": 1,
    "render_mode": "cloth",
    "num_variations": 500,
    "use_cached_states": True,
    "save_cached_states": False,
    "deterministic": False,
    "maximum_crossings": 5,
    "goal_crossings": 5,
    "goal": topology.COMMON_KNOTS["trefoil_knot_O-"],
    "task": "KNOT_ACTION_+R1",
}
env = RopeKnotEnv(**env_kwargs)
env.reset()
img = env.render_no_gripper()
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

while True:
    env.reset()
    for _ in range(5):
        img = env.render_no_gripper()
        topo = env.get_topological_representation()
        topo_action = random.choice(topo.get_valid_add_R1())
        rope_frame = env.get_rope_frame()
        # Get raw position of rope.
        x,y,z,theta = rope_frame
        T_mat = np.array([
            [np.cos(theta),0,np.sin(theta),x],
            [0,1,0,y],
            [-np.sin(theta),0,np.cos(theta),z],
            [0,0,0,1]
        ])
        topo._geometry = (T_mat @ np.vstack([topo.geometry.T,np.ones(topo.geometry.shape[0])]))[:-1,:].T
        regions, markers = TR.watershed_regions(img,topo,topo_action,homography)
        # regions, mu, std = watershed_regions(img,topo,topo_action,rope_frame)
        
        action = np.random.normal(mu,std)

        draw_action(img,regions,mu,rope_frame)
        env.step(mu)

