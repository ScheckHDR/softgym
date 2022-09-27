from cgi import test
from tracemalloc import start
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
from copy import deepcopy
from queue import PriorityQueue, Queue

INTERSECT_NUM = 0
CORRESPONDING = 1
OVER_UNDER = 2
SIGN = 3


def get_quarter_circle(radius,stepsize=0.01,round_decimals=5):

    
    x = np.arange(0,radius+stepsize,stepsize)
    x = np.round(x,round_decimals)
    y = np.sqrt(radius**2 - x**2)

    return 

def parametric_circle(t,xc,yc,R):
    x = xc + R*np.cos(t)
    y = yc + R*np.sin(t)
    return np.transpose(np.vstack((x,y)))

def get_transformation_matrix(pos,theta):
    return np.array([
        [np.cos(theta), np.sin(theta),pos[0]],
        [-np.sin(theta),np.cos(theta),pos[1]],
        [0,0,1]
    ])
def R_mat(t):
    return np.array([
        [np.cos(t),-np.sin(t)],
        [np.sin(t),np.cos(t)]
    ])

class RopeTopology:
    def __init__(self,topology):
        self._topology = topology
        self._has_display = False
        self._display = None





    def size(self):
        return self._topology.shape[1]
    def list_segments(self):
        return self._topology[0,:].tolist()
    def get_col(self,col):
        return self._topology[:,self.section_index(col)]
    def section_index(self,col):
        return np.where(self._topology[0,:] == col)[0][0]
        
    def corresponding(self,col):
        return self._topology[1,self.section_index(col)]
    def upper_val(self,col):
        return self._topology[2,self.section_index(col)]
    def sign(self,col):
        return self._topology[3,self.section_index(col)]
    def is_upper(self,col):
        return self.upper_val(self.section_index(col)) > 0
   
    def get_sub_topology(self,bounds):
        col0 = self.section_index(bounds[0])
        col1 = self.section_index(bounds[1])
        return RopeTopology(np.delete(self._topology,[col0,col1],axis=1))

    def __str__(self):
        return f'{self._topology}'

    def display(self,ax=None,pick_segment=-1,cross_segment=None,place_left=None):
        if not self._has_display:
            self.create_display()
        
        ax = ax or plt.gca()
        self._display.plot(ax,pick_segment,cross_segment,place_left)


    def create_display(self):
        self._display = RopeDisplay(self)
        self._has_display = True

    def display_str(self):
        if self._has_display:
            return f'{self._display}'
        else:
            raise Exception("Display not initialised")

class RopeDisplay:
    def __init__(self,topology:RopeTopology):
        self.topology = topology

        self.segments = []
        self.crossings = [Crossing(position=np.array([-1,0],ndmin=2))] # start of the rope

        prev_crossing = self.crossings[-1]

        prev_over = True

        for segment_num in topology.list_segments():

            new_seg = Segment(segment_num,prev_over,topology.is_upper(segment_num),topology.sign(segment_num) > 0 and topology.is_upper(segment_num))
            prev_crossing.attach_segment(new_seg,incoming=False,is_start=len(self.segments) == 0)

            if topology.corresponding(segment_num) > segment_num:
                # crossing won't exist yet, so create a new one
                new_crossing = Crossing(topology.get_col(segment_num)[:2].tolist())
                new_crossing.attach_segment(new_seg,incoming=True,over=topology.is_upper(segment_num))
                self.crossings.append(new_crossing)

                prev_crossing = new_crossing
            else:
                crossing = [c for c in self.crossings if c.is_crossing(segment_num)][0] # only one should match
                crossing.attach_segment(new_seg,incoming=True,over=topology.is_upper(segment_num))

                prev_crossing = crossing
            self.segments.append(new_seg) 

            prev_over = topology.is_upper(segment_num)

        # add a final segment
        final_segment = Segment(segment_num + 1,prev_over,True,True) # assume goes left. Shouldn't matter anyway
        self.segments.append(final_segment)
        self.crossings[-1].attach_segment(final_segment,incoming=False)
        self.crossings.append(Crossing())
        self.crossings[-1].attach_segment(final_segment,incoming=True,over=True) # can assume over for last segment, just incase of graphical issues.

        pos= self.crossings[0].pos
        direction = 0
        for i in range(len(self.segments)):
            pos,direction = self.segments[i].find_path(self.segments,self.crossings,pos,direction,i == len(self.segments)-1)
            # plt.clf()
            # for j in range(i+1):
            #     self.segments[j].plot(plt.gca())
            # plt.axis('square')
            # plt.draw()
            # plt.pause(0.5)

        # self.plot(plt.gca())
        # plt.show()

    def grab_area(self,segment_num:int,left:bool):

        def flood_fill(start,occupancy):

            grid_min = np.min(occupancy)
            grid_max = np.max(occupancy)
            q = Queue()
            q.put(start)

            borders = np.empty([1,2])
            visited = np.empty([1,2])

            while not q.empty():
                pos = q.get()

                if np.any(np.all(occupancy == pos,axis=1)):
                    borders = np.vstack((borders,pos))
                else:
                    for offset in [np.array([0.5,0]),np.array([0,0.5]),np.array([-0.5,0]),np.array([0,-0.5])]:
                        test_pos = np.round(pos + offset,1)
                        if not np.any(np.all(visited == test_pos,axis=1))\
                            and np.all(test_pos >= grid_min)\
                            and np.all(test_pos <= grid_max):
                            q.put(test_pos)
                visited = np.vstack((visited,pos))
            return borders

        if left:
            test_pos = np.round(self.segments[segment_num].path[1,:] + R_mat(self.segments[segment_num].start_direction) @ np.array([0,0.5]),1)
        else:
            test_pos = np.round(self.segments[segment_num].path[1,:] + R_mat(self.segments[segment_num].start_direction) @ np.array([0,-0.5]),1)
        occupied_points = np.vstack([seg.get_points() for seg in self.segments if seg.path is not None])
        return flood_fill(test_pos,occupied_points)




    def plot(self,ax,pick_segment=-1,cross_segment=None,place_left=None):
        if not isinstance(pick_segment,list):
            pick_segment = [pick_segment]
        ax = ax or plt.gca()
        for seg in self.segments:
            seg.plot(ax,'r-' if seg.segment_num in pick_segment else 'b-')
        if cross_segment is not None:
            area = self.grab_area(cross_segment,place_left)
            ax.fill(area[:,0],area[:,1])
        ax.axis('square')
           

class Crossing:
    def __init__(self,numbers=[-1,-1],position = None):
        self.numbers = numbers

        self.over_in = None
        self.under_in = None
        self.over_out = None
        self.under_out = None

        self.pos = position

        self.attached_segments = []

    def attach_segment(self,segment,incoming:bool,over:bool=None,is_start:bool=False):
        assert not (incoming == True and over is None), f'Cannot have an incoming segment without knowing if it is over or under.'
        if incoming:
            if over:
                self.over_in = segment
            else:
                self.under_in = segment
        else:       

            if (self.over_out is None and self.over_in is not None) or is_start:
                self.over_out = segment
            elif self.under_out is None and self.under_in is not None:
                self.under_out = segment
            else:
                raise Exception('Out segment without an in segment')
        segment.attach_crossing(self,not incoming)

        self.attached_segments.append(segment)

    def is_crossing(self,num):
        return num in self.numbers

    def move(self, ref_point, direction,):
        if self.pos is None:
            return

        ref_point = ref_point.reshape([1,2])
        self.pos = self.pos.reshape([1,2])
        dir = np.round(direction/np.pi*2) % 4
        if (dir == 0 and self.pos[0,0] >= ref_point[0,0]) \
            or (dir == 1 and self.pos[0,1] >= ref_point[0,1]) \
            or (dir == 2 and self.pos[0,0] <= ref_point[0,0]) \
            or (dir == 3 and self.pos[0,1] <= ref_point[0,1]):
            self.pos = np.round(self.pos + (R_mat(direction) @ np.array([1,0])))

        

class Segment:
    def __init__(self,segment_num,starts_over:bool,ends_over:bool,goes_left:bool):
        self.left = goes_left > 0
        self.path = None
        self.segment_num = segment_num

        self.start_crossing = None
        self.end_crossing = None

        self.starts_over = starts_over
        self.ends_over = ends_over

    def attach_crossing(self,crossing,is_start:bool):
        if is_start:
            self.start_crossing = crossing
        else:
            self.end_crossing = crossing

    def get_points(self):
        return self.path

    
           
    def has_point(self,point):
        if self.path is None:
            return False
        return any(np.all(abs(self.path - point.reshape([1,2])) < 1e-3,axis=1))\
            or (self.end_crossing.pos is not None and np.all(point == self.end_crossing.pos))\
            or (self.start_crossing.pos is not None and np.all(point == self.start_crossing.pos))

    def handle_collision(self,point,direction,segments,crossings):



        for seg in segments[:]:
            if seg.has_point(point) and seg is not self and np.any(point != seg.end_crossing.pos) and np.any(point != seg.start_crossing.pos):
                if seg.modify_path(point,direction,segments,crossings,self):
                    for c in crossings:
                        c.move(point,direction)
                    return True

        return False



    def modify_path(self,point,direction,segments,crossings,caller):
        if self.path is None:
            return
        point = point.reshape([1,2])
        def collides(position) -> bool:
            for seg in segments:
                if seg != self and seg.has_point(position):
                    return True
            return False

        forward_point = np.round(point + R_mat(direction) @ np.array([1,0])).reshape([1,2])
        idx = 1
        prev_corner_idx = 0
        while idx < self.path.shape[0]-1:
            cross_prod = ((self.path[idx,0] - self.path[idx-1,0])*(self.path[idx+1,1] - self.path[idx-1,1]) - (self.path[idx,1]-self.path[idx-1,1])*(self.path[idx+1,0]-self.path[idx-1,0]))

            if abs(cross_prod) < 1e-3 and idx < self.path.shape[0] - 2:
                pass 
            else:
                if idx >= self.path.shape[0] - 2:
                    idx = self.path.shape[0]-1


                cross_prod = ((self.path[idx,0] - self.path[idx-1,0])*(forward_point[0,1] - self.path[idx-1,1]) - (self.path[idx,1]-self.path[idx-1,1])*(forward_point[0,0]-self.path[idx-1,0]))

                if np.any(np.all(self.path[prev_corner_idx:idx+1,:] == point,axis=1)) and abs(cross_prod) > 1e-3:
                    offset = np.round(np.array([0,0]) + R_mat(direction) @ np.array([1,0]),1)
                    self.path[prev_corner_idx:idx+1] += offset

                    # get rid of any extra lines
                    corners = np.where(np.all(self.path == self.path[idx,:],axis=1))[0]
                    if len(corners) > 1:
                        self.path = np.delete(self.path,range(corners[0],corners[1]),axis=0)
                        idx -= corners[1]-corners[0] - 1

                    corners = np.where(np.all(self.path == self.path[prev_corner_idx,:],axis=1))[0]
                    if len(corners) > 1:
                        self.path = np.delete(self.path,range(corners[0],corners[1]),axis=0)
                        idx -= corners[1]-corners[0] - 1

                    i = 0
                    while i < self.path.shape[0] - 1:
                        diff = self.path[i+1,:] - self.path[i,:]
                        if np.any(abs(diff) > 0.75):
                            self.path = np.insert(self.path,i+1,np.round(self.path[i,:] + diff/np.linalg.norm(diff) * 0.5,1),axis=0)
                        i += 1

                    if idx < self.path.shape[0] - 1:
                        for i in range(prev_corner_idx,idx+1):
                            if collides(self.path[i,:].reshape([1,2])):
                                self.handle_collision(self.path[i,:].reshape([1,2]),direction,segments,crossings)
                                break # once lines have been moved once, all other points should avoid collision.
                
                prev_corner_idx = idx

            idx += 1

        # # check if last segment needs to move
        # if np.any(np.all(self.path[prev_corner_idx:,:] == point,axis=1)):
        #     prev_corner_pos = deepcopy(self.path[prev_corner_idx,:])
        #     self.path[prev_corner_idx:idx+1] += offset
        #     if np.any(np.all(self.path == prev_corner_pos,axis=1)):
        #         # have some double up, remove it
        #         self.path = np.delete(self.path,[prev_corner_idx,prev_corner_idx+1],axis=0)
        #     else:
        #         # jagged corner, fix be reinserting corner
        #         self.path = np.insert(self.path,prev_corner_idx-1,prev_corner_pos,axis=0)


        # idx = 0
        # while idx < self.path.shape[0] - 1:
        #     diff = self.path[idx+1,:] - self.path[idx,:]
        #     if np.any(abs(diff) > 0.75):
        #         self.path = np.insert(self.path,idx+1,np.round(self.path[idx,:] + diff/np.linalg.norm(diff) * 0.5,1),axis=0)
        #     idx += 1

        if np.any(self.start_crossing.pos != self.path[0,:]):
            # self.start_crossing.pos = self.path[0,:]
            return True        
        if np.any(self.end_crossing.pos != self.path[-1,:]):
            # self.end_crossing.pos = self.path[-1,:]
            return True
        return False


    def find_path(self,segments,crossings,pos,direction = 0,is_last:bool = False):
        
        R = R_mat(direction)
        def can_turn() -> bool:
            nonlocal self
            turn()

            test_pos = step(pos,direction)
            if collides(test_pos):
                res = np.all(test_pos == self.end_crossing.pos)
            else:
                cross = ((test_pos[0,0]-pos[0,0])*(self.end_crossing.pos[0,1]-pos[0,1]) - (test_pos[0,1]-pos[0,1])*(self.end_crossing.pos[0,0]-pos[0,0]))
                res = np.all(test_pos == self.end_crossing.pos) or abs(cross) < 1e-3 or self.left == (cross > 0)

            self.left = not self.left
            turn()
            self.left = not self.left

            return res

        def turn():
            nonlocal R,direction
            if self.left:
                direction += np.pi/2
            else:
                direction -= np.pi/2
            R = R_mat(direction)

        def collides(position) -> bool:
            for seg in segments[:self.segment_num]:
                if seg != self and seg.has_point(position):
                    return True
            return False

        def step(pos,direction,dist=0.5):
            _R = R_mat(direction)

            return np.round(pos + (_R @ np.array([dist,0])),1)

        self.path = pos.reshape([1,2])
        if self.start_crossing.pos is None:
            raise Exception

        if self.end_crossing.pos is None:
            if not is_last: # last segment will be short
                pos = step(pos,direction)
                self.path = np.vstack([self.path,pos])
            pos = step(pos,direction)
            if collides(pos):
                self.handle_collision(pos,direction,segments,crossings)
            self.end_crossing.pos = pos
            self.path = np.vstack((self.path,pos))
            return pos,direction




        while True:
            # initialise with one forward movement. Avoids turning on top of a crossing.            
            pos = step(pos,direction)
            # self.path = np.vstack((self.path,pos))
            self.start_direction = direction

            occupied_points = np.vstack([seg.get_points() for seg in segments if seg.path is not None])
            path = find_grid_path(pos,self,occupied_points,self.left)
            if path is None:
                raise InvalidTopology("Could not verify topology is valid.")
            path = np.vstack((self.path,*path))
        
            # correct the path
            prev_corner_idx = 0
            idx = 1
            first_corner = True
            while idx < path.shape[0]-1:
                # 
                cross_prod = ((path[idx,0] - path[idx-1,0])*(path[idx+1,1] - path[idx-1,1]) - (path[idx,1]-path[idx-1,1])*(path[idx+1,0]-path[idx-1,0]))

                if abs(cross_prod) < 1e-3:
                    pass 
                else:
                    # corner found, if it is on the half grid, move last leg over, away from next leg.
                    if np.any(path[idx,:] % 1 > 1e-3) and not first_corner:

                        # Test that it can be moved back first.
                        offset = step(np.array([0,0]),direction + np.pi)
                        corner_pos = deepcopy(path[idx,:])
                        prev_corner_pos = deepcopy(path[prev_corner_idx,:])
                        test = path[prev_corner_idx:idx+1] + offset

                        for test_ind in range(len(test)):
                            if collides(test[test_ind,:]):
                                # can't move backwards, so move forwards instead

                                offset = step(np.array([0,0]),direction)
                                corner_pos = deepcopy(path[idx,:])
                                prev_corner_pos = deepcopy(path[prev_corner_idx,:])
                                path[prev_corner_idx:idx+1] += offset

                                if np.any(path[idx,:] != path[idx+1,:]):
                                    path = np.insert(path,idx+1,corner_pos,axis=0)

                                path = np.insert(path,prev_corner_idx,prev_corner_pos,axis=0)
                                idx += 1 # account for increasing the size of the array we are operating on

                                for i in range(prev_corner_idx,idx+1):
                                    if collides(path[i,:]):
                                        if self.handle_collision(path[i,:],direction,segments,crossings):
                                            for seg in segments[:self.segment_num+1]:
                                                seg.path = None
                                            pos = segments[0].start_crossing.pos
                                            direction = 0
                                            for seg in segments[:self.segment_num]:
                                                pos,direction = seg.find_path(segments,crossings,pos,direction)
                                            path = np.zeros((idx+1,2)) # stop processing this one further. Should reset and try again.
                                        break # once lines have been moved once, all other points should avoid collision.

                                break
                        else:
                            # can move the corner backwards
                            path[prev_corner_idx:idx+1] = test
                            path = np.insert(path,idx+1,corner_pos,axis=0) # refill the new whole
                            path = np.delete(path,prev_corner_idx-1,axis=0) # remove the duplicate

                            idx -= 1 # account for changing the size of the array we are operating on


                        if np.any(path[-1,:] != self.end_crossing.pos):
                            self.path = pos.reshape([1,2])
                            break # end moved, so need to adjust plan.
                    first_corner = False
                    prev_corner_idx = idx
                    direction = np.arctan2(path[idx,1] - path[idx-1,1],path[idx,0] - path[idx-1,0])


                idx += 1
            else:
                if collides(self.path[-1,:]):
                    # need to check collision of end_crossing in the event of having to redraw
                    self.handle_collision(path[-1,:],direction,segments,crossings)
                break
        direction = np.arctan2(path[idx,1] - path[idx-1,1],path[idx,0] - path[idx-1,0])
        pos = path[-1,:]
        self.path = path



        return pos,direction

    def plot(self,ax,*line_args,**line_kwargs):
        full = deepcopy(self.path)
        ax = ax or plt.gca()

        if not self.starts_over:
            full[0,:] = full[0,:] + (full[1,:] - full[0,:]) * line_kwargs.get('removal_amount',0.2)
        if not self.ends_over:
            full[-1,:] = full[-1,:] + (full[-2,:] - full[-1,:]) * line_kwargs.get('removal_amount',0.2)

        ax.plot(full[:,0],full[:,1], *line_args, **line_kwargs)    
 
class InvalidTopology(Exception):
    pass

def find_grid_path(start,segment,occupancy,come_from_left:bool):
    end_crossing = segment.end_crossing
    start = start.reshape([1,2])
    end = end_crossing.pos.reshape([1,2])

    grid_max = np.max(np.vstack((occupancy,end_crossing.pos))) + np.array([1,1])
    grid_min = np.min(np.vstack((occupancy,end_crossing.pos))) - np.array([1,1])
    
    # come_from_left = not come_from_left
    
    # get a reference point to use for determining if the path enters the crossing from the right side.
    if end_crossing.over_in == segment:
        ref_out_segment = end_crossing.under_out
    elif end_crossing.under_in == segment:
        ref_out_segment = end_crossing.over_out

    if ref_out_segment.path is None:
        starts_crossing = True
        ref_position = start # point won't actually be useful, but needs value to avoid errors.
    else:
        starts_crossing = False
        if ref_out_segment == segment:
            ref_position = start
        else:
            ref_position = ref_out_segment.path[1,:].reshape([1,2])

    class Node:
        def __init__(self,value,parent,priority):
            self.priority = priority
            self.value = value
            self.parent = parent

        def __lt__(self,other):
            return self.priority < other.priority

        def __eq__(self,other):
            return np.all(self.value == other.value)


    visited = []
    frontier = PriorityQueue()
    frontier.put(Node(start,None,np.linalg.norm(end-start)))

    final_node = None
    while not frontier.empty() and final_node is None:

        current = frontier.get()



        for offset in [np.array([0.5,0]),np.array([0,0.5]),np.array([-0.5,0]),np.array([0,-0.5])]:
            test_pos = np.round(current.value + offset,1)
            test = Node(test_pos,current,np.linalg.norm(end-test_pos))


            if np.all(test_pos == end):
                cross_prod = ((ref_position[0,0] - end[0,0])*(current.value[0,1] - end[0,1]) - (ref_position[0,1]-end[0,1])*(current.value[0,0]-end[0,0]))
                if (abs(cross_prod) < 1e-3 and starts_crossing) or (abs(cross_prod) > 1e-3 and (cross_prod > 0) == come_from_left): # ensure the path approaches from the correct side.
                    final_node = test
                    break
            elif not np.any(np.all(occupancy == test.value,axis=1))\
                and test not in visited\
                and np.all(test_pos > grid_min)\
                and np.all(test_pos < grid_max):
                frontier.put(test)
        else:
            visited.append(current)

    if final_node is None:
        return None

    positions = []
    while final_node is not None:
        positions.append(final_node.value)
        final_node = final_node.parent
    positions.reverse()
    return np.vstack(positions)

        

def get_topological_representation(positions):
    intersections = [[0,0]]
    for i in range(positions.shape[0]):
        for j in range (i+2,positions.shape[0]-2):
            if intersect(positions[i],positions[i+1],positions[j],positions[j+1]):
                intersections.append([i,j])
                intersections.append([j,i])

    intersections.sort(key = lambda a:a[0])

    topo = np.zeros((4,len(intersections)))
    for i in range(len(intersections)):
        matching_intersect = intersections.index(intersections[i][::-1])
        is_over = positions[intersections[i][0]][1] > positions[intersections[i][1]][1]
        
        under_vect = positions[intersections[i][0]+1] - positions[intersections[i][0]]
        over_vect = positions[intersections[i][1]+1] - positions[intersections[i][1]]
        if is_over:
            under_vect,over_vect = over_vect,under_vect
        cross_prod = np.cross(over_vect,under_vect)
        sign = np.ones(len(intersections))#np.dot(cross_prod/np.linalg.norm(cross_prod),np.array([0,0,1]))

        topo[:,i] = [
            i,
            matching_intersect,
            is_over*2 -1, # -1 or 1
            sign[i]#/np.abs(sign[i]) # -1 or 1, disabled above
        ]
    return topo.astype(int)


def add_R1(topology,ind,sign):
    T = deepcopy(topology._topology)

    N = np.array([
        [ind,ind+1],
        [ind+1,ind],
        [sign*-1,sign],
        [sign,sign]
    ])

    T[np.where(T[:2,:] >= ind)] += 2

    return RopeTopology(np.insert(T,[ind,ind],N,axis=1)), [ind,ind+1]


def add_C(topology,index,sign):
    topology = topology._topology




def remove_R1(topo,ind):
    if topo[CORRESPONDING,ind] == ind+1 and topo[CORRESPONDING,ind+1] == ind:
        new_topo = np.delete(topo,[ind,ind+1],1)
        new_topo[np.where(new_topo[INTERSECT_NUM:CORRESPONDING+1,:] > ind+1)] -= 2

        return new_topo
    return None

def remove_R2(topo,ind):
    other_crossings = topo[CORRESPONDING,[ind,ind+1]]
    if abs(other_crossings[0]-other_crossings[1]) == 1 \
        and topo[OVER_UNDER,ind] == topo[OVER_UNDER,ind+1] \
        and not np.any(topo[OVER_UNDER,[ind,ind+1]] == topo[OVER_UNDER,other_crossings]):
        
        new_topo = np.delete(topo,other_crossings,1)
        new_topo[np.where(new_topo[INTERSECT_NUM:CORRESPONDING+1,:] > max(other_crossings))] -= 2

        new_topo = np.delete(new_topo,[ind,ind+1],1)
        new_topo[np.where(new_topo[INTERSECT_NUM:CORRESPONDING+1,:] > ind+1)] -= 2

        return new_topo
    return None

def remove_C(topo,ind):
    if ind == -1:
        ind = topo.shape[1]-1 # last column

    if ind != 0 and ind != topo.shape[1]-1: #Cross moves only affect ends of the rope
        return None

    other_ind = topo[CORRESPONDING,ind]

    a = max([ind,other_ind])
    b = min([ind,other_ind])

    new_topo = np.delete(topo,a,1)
    new_topo[np.where(new_topo[INTERSECT_NUM:CORRESPONDING+1,:] > a)] -= 1

    new_topo = np.delete(new_topo,b,1)
    new_topo[np.where(new_topo[INTERSECT_NUM:CORRESPONDING+1,:] > b)] -= 1

    return new_topo


def compare_topology(topo1, topo2):
    if np.all(topo1 == topo2) \
        or np.all(topo1 == flip_topology(topo2))\
        or np.all(topo1 == reverse_topology(topo2)):
        return True
    return False

def generate_random_topology(num_crossings):

    
    topo = np.zeros((4,num_crossings*2))
    topo[0,:] = np.arange(0,num_crossings*2)

    for i in range(num_crossings*2):
        if topo[-1,i] == 0:
            possible_matches = np.where(topo[-1,:] == 0)[0]
            possible_matches = np.delete(possible_matches,np.where(possible_matches == i))
            j = random.choice(possible_matches) # corresponding index of the crossing.

            topo[1,i] = j
            topo[1,j] = i

            is_over = random.choice([-1,1])
            topo[2,i] =  is_over
            topo[2,j] = -is_over

            sign = random.choice([-1,1]) # sign is currently unused. will be -1 or 1 if sign gets added back in
            topo[3,i] = sign
            topo[3,j] = sign
    
    return topo



def flip_topology(topo):
    topo[OVER_UNDER,:] *= -1
    return topo

def reverse_topology(topo):
    # gets the topological representation that matches starting from the other end of the rope.
    num_crossings = topo.shape[1]

    reversed_topo = topo[:,::-1]   
    # rebase
    reversed_topo[INTERSECT_NUM,:] = np.arange(num_crossings)
    reversed_topo[CORRESPONDING,:] = num_crossings - reversed_topo[CORRESPONDING,:] - 1# would need to subtract 1 if not using zero-based topo


    return reversed_topo

def ccw(A,B,C):
    # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
    return (C[2]-A[2]) * (B[0]-A[0]) > (B[2]-A[2]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def reduce_representation(rep_large,indices):
        
    rep_reduced = np.zeros((5,len(indices)))
    # rep_reduced[:2,:] = rep_large[:2,indices]



    # test last segment first, incase of double crossing.
    for i in range(1,len(indices)):
        ind = np.where(rep_large[2,indices[-(i+1)]:indices[-(i)]] != 0)[0]
        if len(ind) > 1:
            indices[-(i+1)] += ind[-1]


    rep_reduced[:,0] = rep_large[:,0]
    e_ind = 0
    for i in range(1,len(indices)):
        s_ind = e_ind
        e_ind = indices[i]

        ind = np.where(rep_large[2,s_ind:e_ind] != 0)[0]

        if len(ind) > 0:
            rep_reduced[:,i] = rep_large[:,s_ind+ind[0]]
            e_ind = s_ind+ind[0] + 1
        else:
            rep_reduced[:,i] = rep_large[:,e_ind-1]
            
    return rep_reduced




if __name__ == '__main__':
    # For quick testing.
    random.seed(20)
    t = np.array([
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
        [11, 4, 5, 8, 1, 2, 7, 6, 3,10, 9, 0],
        [ 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1],
        [-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1]
    ])
    
    while True:
        try:
            t = generate_random_topology(3).astype(np.int)
            print(t)
            t = RopeTopology(t)
            t.display()
        except InvalidTopology:
            continue
        break


