from tracemalloc import start
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
from copy import deepcopy

INTERSECT_NUM = 0
CORRESPONDING = 1
OVER_UNDER = 2
SIGN = 3


def get_quarter_circle(radius,stepsize=0.01,round_decimals=5):

    
    x = np.arange(0,radius+stepsize,stepsize)
    x = np.round(x,round_decimals)
    y = np.sqrt(radius**2 - x**2)

    return 
def get_transformation_matrix(pos,theta):
    return np.array([
        [np.cos(theta), np.sin(theta),pos[0]],
        [-np.sin(theta),np.cos(theta),pos[1]],
        [0,0,1]
    ])
def R_mat(t):
    return np.array([
        [np.cos(t),np.sin(t)],
        [-np.sin(t),np.cos(t)]
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

    def display(self):
        if not self._has_display:
            self.create_display()
        self._display.display()
    

    





        

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
        self.crossings = [Crossing()] # start of the rope

        for segment_num in topology.list_segments():

            new_seg = Segment(topology.sign(segment_num))
            self.crossings[-1].attach_segment(new_seg,incoming=False,is_start=len(self.segments) == 0)

            if topology.corresponding(segment_num) > segment_num:
                # crossing won't exist yet, so create a new one
                new_crossing = Crossing(topology.get_col(segment_num)[:2].tolist())
                new_crossing.attach_segment(new_seg,incoming=True,over=topology.is_upper(segment_num))
                self.crossings.append(new_crossing)
            else:
                crossing = [c for c in self.crossings if c.is_crossing(segment_num)][0] # only one should match
                crossing.attach_segment(new_seg,incoming=True,over=topology.is_upper(segment_num))
            self.segments.append(new_seg)

        # # occupied_points = []
        # for seg in self.segments:
        #     seg.find_path(self.segments)
        

        # add a "crossing" for the end
        self.crossings.append(Crossing())
        self.crossings[-1].attach_segment(new_seg,incoming=True,over=True) # can assume over for last segment, just incase of graphical issues.

        pos=np.array([0,0])
        direction = 0
        for seg in self.segments:
            pos,direction = seg.find_path(self.segments,pos,direction)







            

class Crossing:
    def __init__(self,numbers=[-1,-1]):
        self.numbers = numbers

        self.over_in = None
        self.under_in = None
        self.over_out = None
        self.under_out = None

        self.pos = None

    def attach_segment(self,segment,incoming:bool,over:bool=None,is_start:bool=False):
        assert not (incoming == True and over is None), f'Cannot have an incoming segment without knowing if it is over or under.'
        if incoming:
            if over:
                self.over_in = segment
            else:
                self.under_in = segment
        else:       
            if self.over_in is not None and self.under_in is not None:
                raise Exception('Too many segments')
            if self.over_in is not None or is_start:
                self.over_out = segment
            elif self.under_in is not None:
                self.under_out = segment
            else:
                raise Exception('Out segment without an in segment')
        segment.attach_crossing(self,not incoming)

    def is_crossing(self,num):
        return num in self.numbers

class Segment:
    def __init__(self,goes_left:bool):
        self.left = goes_left
        self.points = [[]]

    def split(self,start_pos, end_pos,is_over:bool):
        pass

    def attach_crossing(self,crossing,is_start:bool):
        if is_start:
            self.start_crossing = crossing
        else:
            self.end_crossing = crossing

    def has_point(self,point):
        return any(np.all(point==p) for line in self.points for p in line)\
            or (self.end_crossing.pos is not None and np.all(point == self.end_crossing.pos))\
            or (self.start_crossing.pos is not None and np.all(point == self.start_crossing.pos))

    def modify_path(self,hit_point,direction):
        pass

    def find_path(self,segments,pos=np.array([0,0]),direction = 0):
        step = np.array([1,0])
        R = R_mat(direction)
        def can_turn() -> bool:
            nonlocal self
            turn()

            test_pos = np.round(pos + (R @ step))
            res = np.all(test_pos == self.end_crossing.pos) or not collides(test_pos)

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
            for seg in segments:
                if seg != self and seg.has_point(position):
                    return True
            return False


        if self.start_crossing.pos is None:
            self.start_crossing.pos = pos
        if self.end_crossing.pos is None:
            test_pos = np.round(pos + (R @ step))
            self.end_crossing.pos = test_pos
            pos = test_pos
            return pos,direction

        while True:

            test_pos = np.round(pos + (R @ step))
            if np.all(test_pos == self.end_crossing.pos):
                pos = test_pos
                break
            if collides(test_pos):
                pass
            else:
                pos = test_pos
                self.points[-1].append(pos)

                if can_turn():
                    turn()
        
        return pos,direction

        

        




        
            



'''
class DisplayRep:
    def __init__(self,bounds = None,parent = None,curved_left = True):
        
        self.sub_displays = []
        self.runs = []
        if bounds is not None:
            self.bound_min = min(*bounds)
            self.bound_max = max(*bounds)
            self.bound_start = bounds[0]
            self.bound_end = bounds[1]
        else:
            self.bound_min = 0
            self.bound_max = 0
            self.bound_start = 0
            self.bound_end = 0
        self.parent = parent
        self.depth = 0 if parent is None else self.parent.depth + 1
        self.contains_start = True if self.depth == 0 else False
        self.contains_end = False
        self.curved_left = curved_left

    def get_size(self):
        sum = 1
        for sub in self.sub_displays:
            sum += 2*sub.get_size()

    def display(self,pos=np.array([[0],[0]]),direction=0,positions={0:np.array([[0],[0]])}):
        T = get_transformation_matrix((pos[0],pos[1]),direction)
        area_size = self.get_size()
        end_pos = pos + get_transformation_matrix([0,0],direction)@np.array([[area_size],[0],[1]])
        num_sub_points = self.bound_max - self.bound_min - 1
        for i in range(1,num_sub_points+1):
            positions[self.bound_min]
        



    def can_enter(self,crossing_num):
        return self.depth > 0 and self.bound_min < crossing_num < self.bound_max # base frame has no bounds

    def process(self,topology,section_to_process = 0,entrance = 0):
        
        self.runs.append([entrance])

        # reduced = topology
        while topology.size() != 0:
            try:
                bounds = topology.get_col(section_to_process)[:2]

                entering_sub = True
                region_bounds = bounds
                # for sub in self.sub_displays:
                #     if sub.can_enter(bounds[0]):
                #         entering_sub = True
                #         region_bounds = [self.bound_start,bounds[1]]
                #         break

                new_region = DisplayRep(bounds=region_bounds,parent=self)
                self.sub_displays.append(new_region)
                self.runs[-1].append(new_region)

                reduced = topology.get_sub_topology(bounds)

                # tests enters a sub region, and entering a different sub region after exiting one.
                exiting_sub = None
                while entering_sub:
                    for sub in self.sub_displays:
                        if sub.can_enter(bounds[0]) and sub != exiting_sub:
                            self.runs[-1].append(section_to_process)
                            section_to_process = topology.corresponding(section_to_process) + 1
                            reduced,section_to_process = sub.process(reduced,section_to_process,entrance=bounds[1])

                            exiting_sub = sub

                            if section_to_process is not None:
                                self.runs.append([topology.corresponding(section_to_process)])


                            break
                    else:
                        # could not re-enter a different sub region.
                        break
                    


                # test exits the region.
                if self.bound_min < bounds[1] < self.bound_max:
                    self.runs[-1].append(bounds[1])
                    return reduced, section_to_process
                
                topology = reduced
            except IndexError:
                pass # Probably means the corresponding index immediately followed (R1 move) and got deleted. Will automatically move to next segment.
            if section_to_process is not None:
                section_to_process += 1
        if section_to_process is not None:
            self.contains_end = True
        return reduced, None

    def __str__(self):
            rep = ''
            if self.contains_start:
                rep += 'S\n'
            else:
                rep += '\t'*(self.depth-1) + f'{self.bound_start},{self.bound_end}\n'

            for sub in self.sub_displays:
                rep += f'{sub}'

            if self.contains_end:
                rep += '\t'*(self.depth) + 'E\n'

            return rep'''

                

    

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


def add_R1(sign,ind):
    pass


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
    random.seed(1)
    t = np.array([
        [0,1,2,3,4,5,6,7,8,9],
        [3,4,7,0,1,6,5,2,9,8],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1]
    ])
    # t = generate_random_topology(3).astype(np.int)
    print(t)
    t = RopeTopology(t)
    t.create_display()
    print(t.display_str())
    t.display()
    plt.show()


