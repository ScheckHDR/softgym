import sys
# from os.path import dirname
import cv2
import numpy as np
import random

sys.path.append("/home/jeffrey/Documents/GitHub/softgym")
from softgym.envs.rope_knot import RopeKnotEnv
import softgym.utils.topology as topology

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

def display_regions(img,topo:topology.RopeTopology,topo_action:topology.RopeTopologyAction,rope_frame):
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

    x,y,z,theta = rope_frame
    T_mat = np.array([
        [np.cos(theta),np.sin(theta),x],
        [-np.sin(theta),np.cos(theta),z],
        [0,0,1]
    ])
    rope = topo.geometry[:,[0,2]].T
    rope_h = np.vstack([rope,np.ones(rope.shape[1])])
    rope_img_p = (homography @ T_mat @ rope_h)[:2,:]

    rope_img = np.zeros_like(img)
    # rope_img = cv2.polylines(
    #     rope_img,
    #     [rope_img_p.T.astype(np.int32)],
    #     isClosed=False,
    #     thickness=3,
    #     color=(255,255,255)
    # )
    # pick_idxs, pick_region,mid_region,place_region = topology.topo_to_geometry(topo,action=topo_action)
    
    segment_idxs = topo.find_geometry_indices_matching_seg(topo_action.over_seg)

    if len(segment_idxs) == 1:
        segment_idxs.append(segment_idxs[0])
    l = len(segment_idxs)
    over_indices = segment_idxs[l//2:]
    under_indices = segment_idxs[:l//2]
    if not topo_action.starts_over:
        over_indices,under_indices = under_indices,over_indices
    
    place_mean = np.mean(np.vstack([topology.get_arc(topo.geometry[under_indices,:][0,[0,2]],topo.geometry[under_indices,:][-1,[0,2]],topo_action.chirality > 0,100),topo.geometry[under_indices,:][::-1,[0,2]]]),axis=0)
    mid_2_mean = np.mean(np.vstack([topology.get_arc(topo.geometry[under_indices,:][0,[0,2]],topo.geometry[under_indices,:][-1,[0,2]],topo_action.chirality < 0,100),topo.geometry[under_indices,:][::-1,[0,2]]]),axis=0)
    mid_1_mean = np.mean(np.vstack([topology.get_arc(topo.geometry[over_indices ,:][0,[0,2]],topo.geometry[over_indices ,:][-1,[0,2]],topo_action.chirality < 0,100),topo.geometry[over_indices ,:][::-1,[0,2]]]),axis=0)
    avoid_mean = np.mean(np.vstack([topology.get_arc(topo.geometry[over_indices ,:][0,[0,2]],topo.geometry[over_indices ,:][-1,[0,2]],topo_action.chirality > 0,100),topo.geometry[over_indices ,:][::-1,[0,2]]]),axis=0)

    place_region = topo.geometry[under_indices,:][:,[0,2]] + 0.05*(place_mean - topo.geometry[under_indices,:][:,[0,2]])
    mid_2_region = topo.geometry[under_indices,:][:,[0,2]] + 0.05*(mid_2_mean - topo.geometry[under_indices,:][:,[0,2]])
    mid_1_region = topo.geometry[over_indices ,:][:,[0,2]] + 0.05*(mid_1_mean - topo.geometry[over_indices ,:][:,[0,2]])
    avoid_region = topo.geometry[over_indices ,:][:,[0,2]] + 0.05*(avoid_mean - topo.geometry[over_indices ,:][:,[0,2]])
    pick_region = topo.geometry[over_indices,:][:,[0,2]]

    place_h = np.vstack([place_region.T,np.ones(place_region.shape[0])])
    mid_2_h = np.vstack([mid_2_region.T,np.ones(mid_2_region.shape[0])])
    mid_1_h = np.vstack([mid_1_region.T,np.ones(mid_1_region.shape[0])])
    avoid_h = np.vstack([avoid_region.T,np.ones(avoid_region.shape[0])])
    pick_h  = np.vstack([pick_region.T,np.ones(pick_region.shape[0])])
    place_img_p = (homography @ T_mat @ place_h)[:2,:]
    mid_2_img_p = (homography @ T_mat @ mid_2_h)[:2,:]
    mid_1_img_p = (homography @ T_mat @ mid_1_h)[:2,:]
    avoid_img_p = (homography @ T_mat @ avoid_h)[:2,:]
    pick_img_p  = (homography @ T_mat @ pick_h )[:2,:]

    markers = cv2.polylines(
        np.zeros((img.shape[0],img.shape[1]),dtype=np.int32),
        [place_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=1
    )
    # cv2.imshow("before",markers.astype(np.uint8)*50)
    # cv2.waitKey(1)
    markers = cv2.polylines(
        markers,
        [mid_2_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=2
    )
    # cv2.imshow("before",markers.astype(np.uint8)*50)
    # cv2.waitKey(1)
    markers = cv2.polylines(
        markers,
        [mid_1_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=3
    )
    # cv2.imshow("before",markers.astype(np.uint8)*50)
    # cv2.waitKey(1)
    markers = cv2.polylines(
        markers,
        [avoid_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=1,
        color=4
    )
    # markers = cv2.polylines(
    #     markers,
    #     [rope_img_p.T.astype(np.int32)],
    #     isClosed=False,
    #     thickness=3,
    #     color=255
    # )
    cv2.imshow("before",markers.astype(np.uint8)*50)
    cv2.waitKey(1)
    markers = cv2.watershed(rope_img,markers.astype(np.int32))
    rope_img[markers == 1] = (0,0,255)
    rope_img[markers == 2] = (255,0,0)
    rope_img[markers == 3] = (104,43,159)
    rope_img[markers == 4] = (208,224,64)
    
    rope_img = cv2.polylines(
        rope_img,
        [pick_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color = (0,255,0)
    )
    cv2.imshow("test",cv2.addWeighted(rope_img,0.5,img,0.5,0))
    cv2.waitKey(0)

    

env_kwargs = {
    "observation_mode": "key_point",
    "action_mode": "picker_trajectory",
    "num_picker": 1,
    "render": True,
    "headless": False,
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
while True:
    img = env.render_no_gripper()
    topo = env.get_topological_representation()
    topo_action = random.choice(topo.get_valid_add_R1())

    display_regions(img,topo,topo_action,env.get_rope_frame())


    action = env.action_space.sample()
    env.step(action)

