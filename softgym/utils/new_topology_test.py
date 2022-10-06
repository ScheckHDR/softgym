import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
from queue import PriorityQueue, Queue
from collections import deque
import time
# from accessify import private

class InvalidTopology(Exception):
    pass



class Segment:
    def __init__(self,segment_num:int,starts_over:bool,ends_over:bool):
        self.end_crossing:Union[None,"Crossing"] = None
        self.start_crossing:Union[None,"Crossing"] = None
        self.segment_num:int = segment_num
        self.starts_over:bool = starts_over
        self.ends_over:bool = ends_over

        self.path:Union[None,np.ndarray] = None

    def attach_crossing(self,crossing,is_end:bool) -> None:
        if is_end:
            self.end_crossing = crossing
        else:
            self.start_crossing = crossing

    def get_other_crossing(self,crossing:"Crossing") -> "Crossing":
        if self.start_crossing == crossing:
            return self.end_crossing
        elif self.end_crossing == crossing:
            return self.start_crossing
        else:
            raise ValueError("Input crossing does not belong to this segment.")

    def get_occupancy(self):
        return self.path

    def get_attachment_point(self,end:bool) -> np.ndarray:
        if end:
            return self.end_crossing.get_attachment_point(self,True)
        else:
            return self.start_crossing.get_attachment_point(self,False)

    def plot(self,ax = None,*args,**kwargs) -> None:
        ax = ax or plt.gca()

        if self.starts_over:
            start = self.start_crossing.position
        else:
            start = self.start_crossing.position + (self.path[0,:] - self.start_crossing.position) * 0.2

        if self.ends_over:
            end = self.end_crossing.position
        else:
            end = self.end_crossing.position + (self.path[-1,:] - self.end_crossing.position)*0.2

        points = np.vstack([start,self.path,end])

        ax.plot(points[:,0],points[:,1],*args,**kwargs)

    def create_geometry(self,occupancy) -> None:
        def step(pos,length,theta):
            return np.round(pos + np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) @ np.array([length,0]),1)

        if self.path is not None:
            return

        start = self.get_attachment_point(end=False)
        start_offset = start-self.start_crossing.position
        direction = np.arctan2(start_offset[0,1],start_offset[0,0])

        if self.end_crossing.position is None:
            self.end_crossing.move_absolute(step(start,0.5,direction))
            self.end_crossing.rotate_absolute(direction)

        goal = self.get_attachment_point(end=True)
        self.path = find_grid_path(start,goal,occupancy)

        self._push_path_to_unit_grid(occupancy)
  
    def _push_path_to_unit_grid(self,occupancy):
        def step(pos,length,theta):
            return np.round(pos + np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) @ np.array([length,0]),1)
        def collides(points,occupancy):
            if occupancy.size == 0:
                return False
            points = points.reshape([-1,2])
            return np.any([np.all(occupancy == points[i,:],axis=1) for i in range(points.shape[0])])
        def line_test(A,B,P):
            # returns a value that will determine the relation of the test point P to the line AB.
            # return > 0: P is on the left
            # return < 0: P is on the right
            # return == 0: P is colinear
            A = A.reshape([-1,2])
            B = B.reshape([-1,2])
            P = P.reshape([-1,2])

            return (P[:,0]-A[:,0])*(B[:,1]-A[:,1])-(P[:,1]-A[:,1])*(B[:,0]-A[:,0])
        
        self.path = np.vstack([self.start_crossing.position,self.path]) # necessary to get checks on first leg.
        idx = 1
        prev_corner_idx = 0
        direction = np.arctan2(self.path[1,1]-self.path[0,1],self.path[1,0]-self.path[0,0])
        while idx < self.path.shape[0] - 1:
            if abs(line_test(self.path[idx-1,:],self.path[idx,:],self.path[idx+1,:])) > 1e-9:
                # Corner found
                if np.any(np.abs(self.path[prev_corner_idx,:] % 1) > 1e-6 ) and prev_corner_idx != 0:
                    # Leg is on the half-unit grid.

                    #First, try pulling the leg back
                    prev_leg_direction = np.round(np.arctan2(self.path[prev_corner_idx,1]-self.path[prev_corner_idx-1,1],self.path[prev_corner_idx,0]-self.path[prev_corner_idx-1,0]),3)
                    next_leg_direction = np.round(np.arctan2(self.path[idx+1,1]-self.path[idx,1],self.path[idx+1,0]-self.path[idx,0]),3)
                    self.path[prev_corner_idx:idx+1,:] = step(self.path[prev_corner_idx:idx+1,:],-0.5,direction)
                    if collides(self.path[prev_corner_idx:idx+1,:],occupancy) or (prev_leg_direction != next_leg_direction and collides(self.path[prev_corner_idx:idx+1,:],self.path[:prev_corner_idx:,:])):
                        # Can't pull leg back, so push it forward instead.
                        self.path[prev_corner_idx:idx+1,:] = step(self.path[prev_corner_idx:idx+1,:],1,direction)
                    idx = 0 # Retrace the path again. Simplest way to account for certain changes.
                    self.path = Segment._repair_path(self.path)
                direction = np.arctan2(self.path[idx,1]-self.path[idx-1,1],self.path[idx,0]-self.path[idx-1,0])
                prev_corner_idx = idx
            idx += 1
        
        # handle last leg seperately since we can't do the corner test
        if idx > 1 and np.any(np.abs(self.path[-2,:] % 1) > 1e-6 ):
            self.path = np.vstack([self.path,self.path[-1,:]])
            self.path[prev_corner_idx:-1,:] = step(self.path[prev_corner_idx:-1,:],-0.5,direction)
            if collides(self.path[prev_corner_idx:-1,:],occupancy) or collides(self.path[prev_corner_idx:-1,:],self.path[:prev_corner_idx:-1,:]):
                self.path[prev_corner_idx:-1,:] = step(self.path[prev_corner_idx:-1,:],1,direction)
            self.path = Segment._repair_path(self.path)

        self.path = self.path[1:,:]

    def modify_path(self,occupancy):
        def step(pos,length,theta):
            return np.round(pos + np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) @ np.array([length,0]),1)
        def collides(points,occupancy):
            if occupancy.size == 0:
                return False
            points = points.reshape([-1,2])
            return np.any([np.all(occupancy == points[i,:],axis=1) for i in range(points.shape[0])])
        def line_test(A,B,P):
            # returns a value that will determine the relation of the test point P to the line AB.
            # return > 0: P is on the left
            # return < 0: P is on the right
            # return == 0: P is colinear
            A = A.reshape([-1,2])
            B = B.reshape([-1,2])
            P = P.reshape([-1,2])

            return (P[:,0]-A[:,0])*(B[:,1]-A[:,1])-(P[:,1]-A[:,1])*(B[:,0]-A[:,0])

        def get_shift_direction(static_points,points_to_move):
            idx = 1
            prev_corner_idx = 0
            direction = np.arctan2(static_points[idx,1]-static_points[idx-1,1],static_points[idx,0]-static_points[idx-1,0])
            while idx < static_points.shape[0] - 1:
                if abs(line_test(static_points[idx-1,:],static_points[idx,:],static_points[idx+1,:])) > 1e-9:
                    # Corner found.
                    if collides(static_points[prev_corner_idx:idx+1,:],points_to_move):
                        if collides(static_points[idx,:],points_to_move):
                            direction = np.arctan2(static_points[idx,1]-static_points[idx-1,1],static_points[idx,0]-static_points[idx-1,0])
                        return direction
                    direction = np.arctan2(static_points[idx,1]-static_points[idx-1,1],static_points[idx,0]-static_points[idx-1,0])
                    prev_corner_idx = idx
                idx += 1
            return np.arctan2(static_points[-1,1]-static_points[-2,1],static_points[-1,0]-static_points[-2,0])

        self.path = np.vstack([self.start_crossing.position,self.path]) # necessary to get checks on first leg.
        idx = 1
        prev_corner_idx = 0
        while idx < self.path.shape[0] - 1:
            if abs(line_test(self.path[idx-1,:],self.path[idx,:],self.path[idx+1,:])) > 1e-9:
                # Corner found.
                if prev_corner_idx != 0 and collides(self.path[prev_corner_idx:idx+1,:],occupancy):
                    direction = get_shift_direction(occupancy,self.path[prev_corner_idx:idx+1,:])
                    self.path[prev_corner_idx:idx+1,:] = step(self.path[prev_corner_idx:idx+1,:],0.5,direction)
                    self.path = Segment._repair_path(self.path)
                    idx = 0 # Retrace the path again. Simplest way to account for certain changes.
                prev_corner_idx = idx
            idx += 1
        self.path = self.path[1:,:] 

    def move_leg(self,leg_num,direction):
        def step(pos,length,theta):
            return np.round(pos + np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) @ np.array([length,0]),1)
        def line_test(A,B,P):
            # returns a value that will determine the relation of the test point P to the line AB.
            # return > 0: P is on the left
            # return < 0: P is on the right
            # return == 0: P is colinear
            A = A.reshape([-1,2])
            B = B.reshape([-1,2])
            P = P.reshape([-1,2])

            return (P[:,0]-A[:,0])*(B[:,1]-A[:,1])-(P[:,1]-A[:,1])*(B[:,0]-A[:,0])
        def collides(points,occupancy):
            if occupancy.size == 0:
                return False
            points = points.reshape([-1,2])
            return np.any([np.all(occupancy == points[i,:],axis=1) for i in range(points.shape[0])])
        if leg_num == 0:
            c = 0
            idx = 1
        elif leg_num == -1:
            idx = self.path.shape[0] - 2
            c = idx + 1
        
        while True:
            if idx <= 0 or idx >= self.path.shape[0]-1 or abs(line_test(self.path[idx,:],self.path[idx+1,:],self.path[idx-1,:])) > 1e-9:
                # corner found, or path is only a single leg.
                if not (0 < idx < self.path.shape[0]):
                    if leg_num == 0:
                        idx = self.path.shape[0] - 1
                    else:
                        idx = 0
                a,b = min(c,idx),max(c,idx)
                
                if np.arctan2(self.path[a,1]-self.path[b,1],self.path[a,0]-self.path[b,0]) == direction\
                    or np.arctan2(self.path[b,1]-self.path[a,1],self.path[b,0]-self.path[a,0]) == direction:
                    stretch_flag = False
                    if not collides(self.end_crossing.position,self.full_path):
                        self.path = np.vstack([self.path,step(self.path[-1,:],0.5,direction),step(self.path[-1,:],1,direction)])
                        stretch_flag = True
                    if not collides(self.start_crossing.position,self.full_path):
                        self.path = np.vstack([step(self.path[0,:],1,direction),step(self.path[0,:],0.5,direction),self.path])
                        stretch_flag = True
                    if stretch_flag:
                        return

                path_len = self.path.shape[0]
                self.path[a:b+1,:] = step(self.path[a:b+1,:],0.5,direction)
                self.path = Segment._repair_path(self.path)
                if self.path.shape[0] > path_len:
                    if leg_num == 0:
                        b += 1
                    else:
                        b += 1
                        a -= 1
                elif self.path.shape[0] < path_len:
                    if leg_num == 0:
                        b -= 1
                    else:
                        b -= 1
                        a += 1
                self.path[a:b+1,:] = step(self.path[a:b+1,:],0.5,direction)
                self.path = Segment._repair_path(self.path)
                return

            if leg_num == -1:
                idx -= 1
            else:
                idx += 1

    def split(self,over_in:bool,over_out:bool): #TODO
        path = self.full_path
        legs = Segment.split_path_into_legs(path)

        if len(legs) == 1:
            test_pos = np.round(path[path.shape[0]//2,:],1)
            if np.any(test_pos % 1 != 0):
                pass
        else:
            pass

    @staticmethod
    def _repair_path(path):
        # remove any duplicates
        unique_path = []
        for row in path:
            if list(row) not in unique_path:
                unique_path.append(list(row))
        unique_path = np.array(unique_path)

        # fill any missing gaps
        i = 0
        while i < unique_path.shape[0]-1:
            if np.any(np.abs(unique_path[i+1,:]-unique_path[i,:]) > 0.75):
                unique_path = np.insert(unique_path,i+1,np.round((unique_path[i+1,:]+unique_path[i,:])/2,1),axis=0)
            i += 1

        return unique_path

    @staticmethod
    def split_path_into_legs(path) -> List[np.ndarray]:
        def line_test(A,B,P):
            # returns a value that will determine the relation of the test point P to the line AB.
            # return > 0: P is on the left
            # return < 0: P is on the right
            # return == 0: P is colinear
            A = A.reshape([-1,2])
            B = B.reshape([-1,2])
            P = P.reshape([-1,2])

            return (P[:,0]-A[:,0])*(B[:,1]-A[:,1])-(P[:,1]-A[:,1])*(B[:,0]-A[:,0])
        legs = []

        prev_corner_idx = 0
        idx = 0
        for idx in range(1, path.shape[0] - 1):
            if abs(line_test(path[idx-1,:],path[idx,:],path[idx+1,:])) > 1e-9:
                # corner found
                legs.append(path[prev_corner_idx:idx+1,:])
                prev_corner_idx = idx
        legs.append(path[prev_corner_idx:idx+2,:])
        return legs

    @property
    def full_path(self):
        return np.vstack([self.start_crossing.position,self.path,self.end_crossing.position])

class Crossing:
    def __init__(self,numbers=np.array([-1,-1]),sign=1,is_start:bool = False):
        
        self._numbers = numbers
        self.is_start = is_start
        self.sign = sign

        self.over_in = None
        self.over_out = None
        self.under_in = None
        self.under_out = None

        self.attachment_points = {}

        self._rotation:float = 0
        self._position:Union[None,np.ndarray] = None

    def attach_segment(self,segment:Segment,over:bool,incoming:bool) -> None:
        
        if not hasattr(self,'over_in_attach'):

            if over:
                self.over_in_attach = np.array([-0.5,0])
                self.over_out_attach =np.array([ 0.5,0])
                self.under_in_attach = np.array([ 0,-0.5]) if self.sign > 0  else np.array([0,0.5])
                self.under_out_attach = np.array([ 0, 0.5]) if self.sign > 0  else np.array([0,-0.5])
            else:
                self.under_in_attach  = np.array([-0.5,0])
                self.under_out_attach = np.array([ 0.5,0])
                self.over_in_attach   = np.array([ 0, 0.5]) if self.sign > 0  else np.array([0,-0.5])
                self.over_out_attach  = np.array([ 0,-0.5]) if self.sign > 0  else np.array([0,0.5])


        # elif num_attached == 1:
        #     self.attachment_points[(segment,incoming)] = np.array([ 0.5, 0])
        # elif num_attached == 2:
        #     self.attachment_points[(segment,incoming)] = np.array([ 0,-0.5]) if self.sign > 0 == over else np.array([0,0.5])
        # elif num_attached == 3:
        #     self.attachment_points[(segment,incoming)] = np.array([ 0, 0.5]) if self.sign > 0 == over else np.array([0,-0.5])
        # elif num_attached == 4:
        #     raise Exception("too many segments attached.")

        if self.is_start:
            self.over_out = segment
            return
        if over:
            if incoming:
                self.over_in = segment
            else:
                self.over_out = segment
        else:
            if incoming:
                self.under_in = segment
            else:
                self.under_out = segment

    def get_next_segment(self,segment:Segment) -> Segment:
        if self.under_in == segment:
            return self.under_out
        elif self.over_in == segment:
            return self.over_out
        else:
            raise Exception

    def get_previous_segment(self,segment:Segment) -> Segment:
        if self.under_out == segment:
            return self.under_in
        elif self.over_out == segment:
            return self.over_in
        else:
            raise Exception

    def get_connected_segment(self,segment:Segment,is_input:bool) -> Segment:
        if is_input:
            if self.under_in == segment:
                return self.under_out
            elif self.over_in == segment:
                return self.over_out
        else:
            if self.under_out == segment:
                return self.under_in
            elif self.over_out == segment:
                return self.over_in
        
        raise Exception

    def get_side_segment(self,segment:Segment,left:bool,segment_is_input:bool) -> Union[Segment, None]:
        # returns segment, and if the segment is an output
        s = self.sign > 0

        if segment_is_input:
            if segment == self.over_in:
                return (self.under_out,True) if left == s else (self.under_in,False)
            elif segment == self.under_in:
                return (self.over_in,False) if left == s else (self.over_out,True)
        else:
            if segment == self.over_out:
                return (self.under_in,False) if left == s else (self.under_out,True)
            elif segment == self.under_out:
                return (self.over_out,True) if left == s else (self.over_in,False)

        raise Exception

        # if direction > 0:
        #     if segment in [self.under_in,self.under_out]:
        #         return self.over_in if left > 0 else self.over_out
        #     elif segment in [self.over_in,self.over_out]:
        #         return self.under_out if left > 0 else self.under_in
        # elif direction < 0:
        #     if segment in [self.under_in,self.under_out]:
        #         return self.over_out if left > 0 else self.over_in
        #     elif segment in [self.over_in,self.over_out]:
        #         return self.under_in if left > 0 else self.under_out
            
    def empty_type_on_side(self,seg,is_input:bool,from_left:bool) -> bool:
        if seg == self.under_in:
            if self.sign > 0 == from_left:
                return is_input and self.over_in == None
            else:
                return not is_input and self.over_out == None
        elif seg == self.under_out:
            if self.sign > 0 == from_left:
                return not is_input and self.over_out == None
            else:
                return is_input and self.over_in == None
        elif seg == self.over_in:
            if self.sign > 0 == from_left:
                return is_input and self.under_in == None
            else:
                return not is_input and self.under_out == None
        elif seg == self.over_out:
            if self.sign > 0 == from_left:
                return not is_input and self.under_out == None
            else:
                return is_input and self.under_in == None
        else:
            raise Exception

    def get_attachment_point(self,segment,is_input:bool):
        p = None
        if is_input:
            if segment == self.over_in:
                p = self.over_in_attach
            elif segment == self.under_in:
                p = self.under_in_attach
        else:
            if segment == self.over_out:
                p = self.over_out_attach
            elif segment == self.under_out:
                p = self.under_out_attach
        if p is None:
            raise ValueError("Segment is not attached to crossing.")
        
        R = np.array([
            [np.cos(self.rotation),-np.sin(self.rotation)],
            [np.sin(self.rotation), np.cos(self.rotation)]
        ])
        return np.round((self.position + R @ p).reshape([1,2]),1)

    def replace_segment(self,old,new):
        pass

    def move_relative(self,delta):
        self._position = np.round(self._position + delta.reshape([1,2]),1)
    def move_absolute(self,pos):
        self._position = pos.reshape([1,2])
    def rotate_relative(self,delta):
        self._rotation += delta
    def rotate_absolute(self,theta):
        self._rotation = theta

    def __eq__(self,other):
        if isinstance(other,int) or isinstance(other,float) or isinstance(other,np.int32) or isinstance(other,np.int64):
            return np.any(self._numbers == other)
        elif isinstance(other,Crossing):
            return np.all(self._numbers == other._numbers)
        
        raise ValueError(f"Cannot compare crossing to type {type(other)}.")

    @property
    def position(self) -> Union[None,np.ndarray]:
        return deepcopy(self._position)
    @property
    def rotation(self) -> float:
        return self._rotation
    @property
    def attached_segments(self):
        return [seg for seg in [self.over_in,self.over_out,self.under_in,self.under_out] if seg is not None]
        # return [seg[0] for seg in self.attachment_points]    

class RopeTopology:
    def __init__(self,topo_np,check_validity=True):
        if not RopeTopology.quick_check(topo_np):
            raise InvalidTopology("Representation failed initial check.")
        self._topology:np.ndarray = topo_np

        self.segments:list[Segment] = []
        self.crossings:list[Crossing] = [Crossing(is_start=True)] 

        prev_crossing = self.crossings[-1]
        prev_over = True #Assume rope went over initial "crossing"

        col = -1 # Needed in case on trivial knot.
        for col in range(topo_np.shape[1]):
            new_segment = Segment(col,prev_over,topo_np[2,col] > 0)
            prev_crossing.attach_segment(new_segment,prev_over>0,False)
            new_segment.attach_crossing(prev_crossing,False)
            self.segments.append(new_segment)
            is_over = topo_np[2,col]>0
            if topo_np[0,col] < topo_np[1,col]:
                # end crossing does not exist yet; create it.
                self.crossings.append(Crossing(topo_np[:2,col],topo_np[3,col]))
                self.crossings[-1].attach_segment(new_segment,is_over,True)
                new_segment.attach_crossing(self.crossings[-1],True)
                prev_crossing = self.crossings[-1]
            else:
                # crossing already exists.
                c = [c for c in self.crossings if c == topo_np[0,col]][0]
                c.attach_segment(new_segment,is_over,True)
                new_segment.attach_crossing(c,True)

                # would have created a new region, check that this doesn't invalidate anything.
                if check_validity:
                    for is_left in [True,False]:
                        test_crossing = c
                        prev_seg = new_segment
                        inputs = 0
                        outputs = 0
                        is_input = True
                        while True:
                            next_seg, was_output = test_crossing.get_side_segment(prev_seg,is_left,is_input)

                            if next_seg is not None and next_seg.segment_num == 0:
                                #ignore end crossing, reverse direction.
                                prev_seg = next_seg
                                is_input = True
                                continue
                            if next_seg == None:
                                inputs += test_crossing.empty_type_on_side(prev_seg,True,is_left)
                                outputs += test_crossing.empty_type_on_side(prev_seg,False,is_left)
                                next_seg = test_crossing.get_connected_segment(prev_seg,is_input)
                                was_output = is_input
                                if next_seg is None:
                                    # Probably fucking R1
                                    next_seg,was_output = test_crossing.get_side_segment(prev_seg,not is_left,is_input)
                                if next_seg.segment_num == 0:
                                    prev_seg = next_seg
                                    is_input = True
                                    continue
                                    # next_seg = test_crossing.get_side_segment(prev_seg,is_left)

                            test_crossing = next_seg.get_other_crossing(test_crossing)
                            prev_seg = next_seg    
                            is_input = was_output                   

                            if test_crossing == c:
                                #done full loop
                                break

                        if inputs > outputs:
                            raise InvalidTopology
                    
                prev_crossing = c
            prev_over = is_over

        # Final segment, also has a phony "crossing"
        self.segments.append(Segment(col+1,prev_over,True))
        prev_crossing.attach_segment(self.segments[-1],prev_over,False)
        self.crossings.append(Crossing())
        self.crossings[-1].attach_segment(self.segments[-1],True,True)
        self.segments[-1].attach_crossing(prev_crossing,False)
        self.segments[-1].attach_crossing(self.crossings[-1],True)

    def construct_geometry(self):

        def step(pos,length,theta):
            return np.round(pos + np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) @ np.array([length,0]),1)
        def collides(points,occupancy):
            if occupancy is None or occupancy.size == 0:
                return False
            points = points.reshape([-1,2])
            return np.any([np.all(occupancy == points[i,:],axis=1) for i in range(points.shape[0])])
        def get_colliding_leg_numbers(A,B):
            a_indices = []
            b_indices = []
            for a in range(len(A)):
                for b in range(len(B)):
                    if collides(A[a],B[b]):
                        a_indices.append(a)
                        b_indices.append(b)
            return a_indices,b_indices
        def handle_crossing_movements(crossing:Crossing,direction,calling_seg=None):
            crossing.move_relative(step(np.array([0,0]),1,direction))
            for attached_seg in crossing.attached_segments:
                if attached_seg != calling_seg:
                    if crossing == attached_seg.start_crossing:
                        attached_seg.move_leg(0,direction)
                        opp_crossing = attached_seg.end_crossing
                        if np.any(attached_seg.path[-1,:] != attached_seg.get_attachment_point(True)) and not direction == np.arctan2(crossing.position[0,1] - opp_crossing.position[0,1],crossing.position[0,0] - opp_crossing.position[0,0]):
                            handle_crossing_movements(opp_crossing,direction,attached_seg)
                    else:
                        attached_seg.move_leg(-1,direction)
                        opp_crossing = attached_seg.start_crossing
                        if np.any(attached_seg.path[0,:] != attached_seg.get_attachment_point(False)) and not direction == np.arctan2(crossing.position[0,1] - opp_crossing.position[0,1],crossing.position[0,0] - opp_crossing.position[0,0]):
                            handle_crossing_movements(opp_crossing,direction,attached_seg)
        def handle_collisions(current_seg,segments):
            legs_current = Segment.split_path_into_legs(current_seg.full_path[1:,:])
            for seg in segments:
                if seg != current_seg and seg.path is not None and collides(current_seg.full_path[1:,:],seg.path):
                    legs_seg = Segment.split_path_into_legs(seg.full_path[1:,:])
                    current_indices,seg_indices = get_colliding_leg_numbers(legs_current,legs_seg)
                    if 0 in seg_indices:
                        si = seg_indices.index(0)
                        ci = current_indices[si]
                        if ci == 0:
                            shift_direction = np.arctan2(legs_current[0][-1,1]-legs_current[0][-2,1],legs_current[0][-1,0]-legs_current[0][-2,0])
                        else:
                            shift_direction = np.arctan2(legs_current[ci-1][-1,1]-legs_current[ci-1][-2,1],legs_current[ci-1][-1,0]-legs_current[ci-1][-2,0])
                        handle_crossing_movements(seg.start_crossing,shift_direction)
                    if len(legs_seg)-1 in seg_indices:
                        si = seg_indices.index(len(legs_seg)-1)
                        ci = current_indices[si]
                        if ci == 0:
                            shift_direction = np.arctan2(legs_current[0][-1,1]-legs_current[0][-2,1],legs_current[0][-1,0]-legs_current[0][-2,0])
                        else:
                            shift_direction = np.arctan2(legs_current[ci-1][-1,1]-legs_current[ci-1][-2,1],legs_current[ci-1][-1,0]-legs_current[ci-1][-2,0])
                        handle_crossing_movements(seg.end_crossing,shift_direction)

                    if collides(current_seg.full_path[1:,:],seg.path): # Handling of crossings may have handled the collision already.
                        seg.modify_path(current_seg.full_path[1:,:])
                        occupancy = np.vstack([s.full_path for s in segments if s.path is not None and s != seg])
                        seg._push_path_to_unit_grid(occupancy)
                        handle_collisions(seg,segments)

                    

        self.crossings[0].move_absolute(np.array([-1,0]))
        first_iter = True

        for segment in self.segments:
            if first_iter:
                occupancy = self.crossings[0].position
                first_iter = False
            else:
                occupancy = np.vstack([seg.full_path for seg in self.segments if seg != segment and seg.path is not None])
            segment.create_geometry(occupancy)
            # segment.plot()
            # Check collisions
            if collides(segment.path,occupancy) or any([collides(segment.end_crossing.position,s.path) for s in self.segments]):
                handle_collisions(segment,self.segments)
                occupancy = self.crossings[0].position
                # plt.clf()
                # for seg in self.segments:
                #     if seg.path is not None:
                #         seg.plot()
                       
    def plot(self,ax = None,pick_segs = [],*args,**kwargs) -> None:
        if isinstance(pick_segs,int):
            pick_segs = [pick_segs]
        ax = ax or plt.gca()
        for segment in self.segments:
            segment.plot(ax,'r-' if segment.segment_num in pick_segs else 'b-',*args,**kwargs)

    def add_R1(self,segment_num:int,over:int,sign:int) -> "RopeTopology":
        assert over in [-1,1],f''
        assert sign in [-1,1],f''
        # Adds an R1 move to the desired segment. If over, first crossing will be over, and then under. 
        T = deepcopy(self.rep)

        N = np.array([
            [segment_num,segment_num+1],
            [segment_num+1,segment_num],
            [over,over*-1],
            [sign,sign]
        ])
        T[np.where(T[:2,:] >= segment_num)] += 2

        r1_rep = np.hstack([T,N])
        return RopeTopology(r1_rep[:,r1_rep[0,:].argsort()])

    def add_C(self,over_ind:int,under_ind:int,sign:int,under_first:bool,return_raw:bool = False) -> Tuple["RopeTopology",List[int]]:
        assert over_ind in [0,self.size] or under_ind in [0,self.size], f'C moves require at least one of the affected indices to be the end of the rope.'         
        T = deepcopy(self.rep)

        if over_ind > under_ind:
            N = np.array([
                [under_ind,over_ind+1],
                [over_ind+1,under_ind],
                [-1,1],
                [sign,sign]
            ])
            over_segs = [over_ind+1,over_ind+2]
        elif under_ind > over_ind:
            N = np.array([
                [over_ind,under_ind+1],
                [under_ind+1,over_ind],
                [1,-1],
                [sign,sign]
            ])
            over_segs = [over_ind,over_ind+1]
        else:
            N = np.array([
                [over_ind,under_ind+1],
                [under_ind+1,over_ind],
                [-1,1] if under_first else [1,-1],
                [sign,sign]
            ])
            over_segs = [over_ind,over_ind+1]

        T[np.where(T[:2,:] >= max(over_ind,under_ind))] += 1
        T[np.where(T[:2,:] >= min(over_ind,under_ind))] += 1

        r2_rep = np.hstack([T,N])
        r2_rep = r2_rep[:,r2_rep[0,:].argsort()]
        if return_raw:
            if RopeTopology.quick_check(r2_rep):
                return r2_rep, over_segs
            else:
                raise InvalidTopology(f"Could not add C move. {self.rep} does not allow segment {over_ind} to be placed over segment {under_ind}, with sign {sign}.")
        return RopeTopology(r2_rep), over_segs
    def remove_C(self,segment_num:int,return_raw:bool = False) -> Tuple["RopeTopology",List[int]]:
        assert segment_num in [0,self.size], f'segment_num must relate to the ends of the rope'

        T = deepcopy(self.rep)
        if segment_num == 0:
            a,b = segment_num, self.corresponding(segment_num)
        elif segment_num == self.size:
            a,b = self.corresponding(segment_num-1),segment_num-1 #TODO check this is correct.
        T = np.delete(T,b,axis=1)
        T = np.delete(T,a,axis=1)
        T[np.where(T[:2,:] >= b)] -= 1
        T[np.where(T[:2,:] >= a)] -= 1

        if return_raw:
            if RopeTopology.quick_check(T):
                return T
            else:
                raise InvalidTopology(f"Could not remove C move. {self.rep} does not allow {segment_num} to be moved.") # This should never occur.
        return RopeTopology(T)
    @staticmethod
    def quick_check(topo_np):
        for col in topo_np.T:
            if topo_np[1,col[1]] != col[0] or topo_np[2,col[1]] == col[2] or topo_np[3,col[1]] != col[3]:
                return False
        return True


    def corresponding(self,seg_num):
        return self._topology[1,self._section_index(seg_num)]
    def upper_val(self,seg_num):
        return self._topology[2,self._section_index(seg_num)]
    def sign(self,seg_num):
        return self._topology[3,self._section_index(seg_num)]
    def is_upper(self,seg_num):
        return self.upper_val(self._section_index(seg_num)) > 0
    def _section_index(self,col):
        return np.where(self._topology[0,:] == col)[0][0]

    def __eq__(self,other) -> bool:
        if isinstance(other,RopeTopology):
            return np.all(self.rep == other.rep)
        elif isinstance(other,np.ndarray):
            return np.all(self.rep == other)
        raise ValueError(f"Cannot compare type RopeTopology with type {type(other)}.")

    def find_geometry_indices_matching_seg(self,segment_number,incidence_matrix) -> List[int]:
        if self.size == 0:
            return [i for i in range(incidence_matrix.shape[1])]
        else:
            seg_num = 0
            start_idx = 0
            for row_num in range(incidence_matrix.shape[0]):
                if np.any(incidence_matrix[row_num,:] != 0):
                    seg_num += 1
                    if seg_num > segment_number:
                        return [i for i in range(start_idx,row_num)]
                    elif seg_num == segment_number:
                        start_idx = row_num


    @property
    def size(self):
        return self._topology.shape[1]
    @property
    def rep(self):
        return self._topology

def find_grid_path(start,end,occupancy) -> np.ndarray:

    class node:
        def __init__(self,value,distance,parent=None):
            self.value = value
            self.distance = distance
            self.parent = parent
        def __eq__(self,other):
            if other == None:
                return False
            return np.all(self.value == other.value)
        def __lt__(self,other):
            return self.distance < other.distance
    def dist(pos,goal):
        return np.linalg.norm(pos-goal,ord=2)

    max_grid = np.max(occupancy,axis=0) + 1
    min_grid = np.min(occupancy,axis=0) - 1
    
    if np.all(start==end):
        return start.reshape([-1,2])
    frontier = PriorityQueue()
    visited = np.empty([1,2])
    frontier.put(node(start,0))
    done = False
    while not frontier.empty() and not done:
        current = frontier.get()
        for delta in [np.array([0.5,0]),np.array([0,0.5]),np.array([-0.5,0]),np.array([0,-0.5])]:
            new_pos = np.round(current.value + delta,1)
            new_node = node(new_pos,dist(new_pos,end) + dist(new_pos,start),current)
            if np.all(new_pos == end):
                done = True
                current = new_node
                break           
            if not np.any(np.all(visited == new_pos,axis=1)) \
                and not np.any(np.all(occupancy == new_pos,axis=1))\
                and np.all(new_pos >= min_grid)\
                and np.all(new_pos <= max_grid):
                frontier.put(new_node)
        visited = np.vstack([visited,current.value])
    if not done:
        raise Exception # failed to find a path.
    
    path = []
    while current != None:
        path.append(current.value)
        current = current.parent
    path.reverse()
    return np.vstack([*path]).reshape([-1,2])
            
class RopeTopologyNode:
    def __init__(self,value:RopeTopology,priority:int,parent:Union[None,RopeTopology]=None,action=None):
        self.value = value
        self.parent = parent
        self.action = action
        self.priority = priority
    def __eq__(self,other):
        if isinstance(other,RopeTopologyNode):
            return self.value == other.value
        elif isinstance(other,RopeTopology) or isinstance(other,np.ndarray):
            return self.value == other
        raise ValueError(f"Cannot compare type RopeTopologyNode to type {type(other)}.")
    def __lt__(self,other):
        self.priority < other.priority
    
    @property
    def num_parents(self):
        t = self
        i = 0
        while self.parent is not None:
            i += 1
            t = self.parent
        return i


def find_topological_path(start:RopeTopology,end:RopeTopology,max_rep_size = np.inf) -> List[RopeTopology]:

    frontier = PriorityQueue()
    frontier.put((0,RopeTopologyNode(start,0)))
    visited = []

    def distance_func(current,end):
        # a = abs(end.size - current.size)
        a = max(end.rep.size,current.rep.size) + abs(end.size-current.size)
        if end.size == current.size:
            a -= np.sum(end.rep[1,:] == current.rep[1,:])
        
        b = max(0,current.rep[3,:][np.where(current.rep[3,:] == -1)].size - end.rep[3,:][np.where(end.rep[3,:] == -1)].size)
        c = max(0,current.rep[3,:][np.where(current.rep[3,:] ==  1)].size - end.rep[3,:][np.where(end.rep[3,:] ==  1)].size)

        return a+b+c

    def explore_add_C(current):
        if current.value.size > max_rep_size:
            return 
        for over_seg in range(current.value.size+1):
            for under_seg in range(current.value.size+1):
                for sign in [-1,1]:
                    if over_seg in [0,current.value.size] or under_seg in [0,current.value.size]:
                        for under_first in ([False,True] if over_seg == under_seg else [False]):
                            try:
                                action_args = [over_seg,under_seg,sign,under_first]
                                test, after_action_segs = current.value.add_C(*action_args,return_raw=True)
                                if end == test:
                                    return RopeTopologyNode(end,0,parent=current,action = ["+C",action_args,after_action_segs])
                                if test.shape[1] >= max_rep_size:
                                    break
                                if test not in visited and test not in frontier.queue:
                                    new_topo = RopeTopology(test,check_validity=False)
                                    dist = distance_func(new_topo,end)
                                    frontier.put((dist,RopeTopologyNode(new_topo,dist,parent=current,action = ["+C",action_args,after_action_segs])))
                            except InvalidTopology:
                                pass
    def explore_remove_C(current):
        if not (0 < current.value.size < max_rep_size):
            return
        for seg in [0,current.value.size]:
            try:
                test = current.value.remove_C(seg,return_raw=True)
                if end == test:
                    return RopeTopologyNode(end,0,parent=current,action=["-C",[seg],[]])
                if test not in visited and test not in frontier.queue:
                    new_topo = RopeTopology(test,check_validity=False)
                    dist = distance_func(new_topo,end)
                    frontier.put((dist,RopeTopologyNode(new_topo,dist,parent=current,action=["-C",[seg],[]])))
            except InvalidTopology:
                pass

    def explore():
        while not frontier.empty():
            current = frontier.get()[1]
            for explore_func in [explore_add_C,explore_remove_C]:
                finish = explore_func(current)
                if finish is not None:
                    return finish
            visited.append(current.value)


    n = explore()
    path = []
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path
    


if __name__ == '__main__':
    # t = np.array([
    #     [ 0, 1, 2, 3, 4, 5],
    #     [ 2, 3, 0, 1, 5, 4],
    #     [ 1, 1,-1,-1, 1,-1],
    #     [-1, 1,-1, 1, 1, 1]
    # ])
    t = np.array([
        [0,1],
        [1,0],
        [-1,1],
        [1,1]
    ])
    RopeTopology(t)
    # t = np.array([
    #     [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
    #     [11, 4, 5, 8, 1, 2, 7, 6, 3,10, 9, 0],
    #     [ 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1],
    #     [-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1]
    # ]) invalid
    
    # print(t)

    # trivial_knot = RopeTopology(np.empty([4,0],dtype=np.int32))
    # trefoil_knot = RopeTopology(np.array([
    #     [ 0, 1, 2, 3, 4, 5],
    #     [ 3, 4, 5, 0, 1, 2],
    #     [ 1,-1, 1,-1, 1,-1],
    #     [-1,-1,-1,-1,-1,-1]
    # ],dtype=np.int32))

    # topological_path = find_topological_path(trivial_knot,trefoil_knot)

    # for n in topological_path:
    #     print(f'rep:{n.value.rep}\naction:{n.action}')

    # f, (ax1,ax2) = plt.subplots(1,2)
    # trivial_knot.construct_geometry()
    # trivial_knot.plot(ax1)
    # trefoil_knot.construct_geometry()
    # trefoil_knot.plot(ax2)

    # plt.pause(2)

    # for i in range(1,len(topological_path)):
    #     ax1.clear()
    #     ax2.clear()

    #     type,action,after_segs = topological_path[i].action
    #     pick_segs = action[0]
    #     topological_path[i-1].value.construct_geometry()
    #     topological_path[i-1].value.plot(ax1,pick_segs)
    #     topological_path[i].value.construct_geometry()
    #     topological_path[i].value.plot(ax2,after_segs)
    #     plt.pause(2)

    # R = RopeTopology(t)
    # R.construct_geometry()
    # R.plot(ax1)

    # test_R = R.add_R1(0,1,1)
    # test_R.construct_geometry()
    # test_R.plot(ax2)

    # plt.show()





