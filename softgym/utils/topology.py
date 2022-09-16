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


def get_semicircle(cx,cy,radius,direction:int,stepsize=0.01):
    round_size = int(np.math.log10(1/stepsize)*2)
    if radius < 0:
        stepsize *= -1
    x = np.arange(-radius,radius+stepsize,stepsize)
    x = np.round(x,round_size)
    y = np.sqrt(radius**2 - x**2)

    if direction == 90:
        x,y = -y,x
    elif direction == 180:
        y = -y
    elif direction == 270:
        x,y = y,x
    return x+cx,y+cy


def display_topo(topo):

    class Area:
        def __init__(self,bounds,direction=0,segment=0,pos = np.array([-1,0]),left = True, parent = None,):
            self.parent = parent
            if self.parent is None:
                self.depth = 0                
                self.is_red = False
            else:
                self.depth = self.parent.depth + 1
                self.is_red = not parent.is_red


            self.left = left
            self.bound_min,self.bound_max = min(*bounds),max(*bounds)
            self.pos = pos
            self.direction = direction
            self.segment = segment
            self.sub_areas = []

            diff = int(abs(bounds[0]-bounds[1]))
            self.scale = diff/(diff+1)
            if self.scale == 0:
                self.scale = 1
            self.node_pos = {}

            self.current_segment = segment

        def set_pos(self,pos,direction):
            self.pos = pos
            self.direction = direction

        def add_link(self,start_under=False,end_under=False,will_loop=False):
            if self.direction == 0:
                x = self.scale
                y = 0
            elif self.direction == 90:
                x = 0
                y = self.scale
            elif self.direction == 180:
                x = -self.scale
                y = 0
            elif self.direction == 270:
                x = 0
                y = -self.scale

            end_pos = self.pos + np.array([x,y])


            line_coords = np.transpose(np.linspace(self.pos,end_pos,num=100))
            if start_under:
                line_coords = line_coords[:,5:]
            if end_under and not will_loop:
                line_coords = line_coords[:,:-5]
            plt.plot(line_coords[0,:],line_coords[1,:],'k-')
            
            self.pos = end_pos

        def _is_pointed_at_point(self,curr,des,direction,tolerance=1e-5):
            if direction == 0 and abs(curr[1] - des[1]) < tolerance and curr[0] <= des[0]:
                return True
            elif direction == 90 and abs(curr[0] - des[0]) < tolerance and curr[1] <= des[1]:
                return True
            elif direction == 180 and abs(curr[1] - des[1]) < tolerance and curr[0] >= des[0]:
                return True
            elif direction == 270 and abs(curr[0] - des[0]) < tolerance and curr[1] >= des[1]:
                return True
            return False

        def process_topology(self,topo,segment,last_crossing=0):
            done = False
            while segment < topo.shape[1] and not done:                
                entry = topo[:,segment]

                finish_crossing = entry[0] > entry[1]
                # if finish_crossing:
                #     area_corners = [self.pos]

                self.add_link(topo[2,segment-1] < 0 if segment > 0 else False,entry[2] < 0,finish_crossing)
                
                self.node_pos[segment] = self.pos

                if finish_crossing:
                    # creates new area and potentially leaves current
                    area_points = [self.pos]
                    if entry[1] in self.node_pos:
                    # if self.bound_min < entry[1] < self.bound_max:
                        max_diff = int(entry[0] - entry[1])
                    else:
                        max_diff = int(abs(last_crossing - entry[1]))
                    if not self._is_pointed_at_point(self.pos,self.node_pos[entry[1]],self.direction):
                        self.direction = (self.direction + 90) % 360

                    while not self._is_pointed_at_point(self.pos,self.node_pos[entry[1]],self.direction):
                        for i in range(max_diff):
                            self.add_link(False,False)
                            area_points.append(self.pos)
                            # test in line
                            self.direction = (self.direction + 90) % 360
                            if self._is_pointed_at_point(self.pos,self.node_pos[entry[1]],self.direction):
                                max_diff = i+1
                                self.direction = (self.direction - 90) % 360
                                break
                            self.direction = (self.direction - 90) % 360
                        self.direction = (self.direction + 90) % 360

                    for _ in range(max_diff-1):
                        self.add_link(False,False)
                        area_points.append(self.pos)

                    self.add_link(False,entry[2] < 0,False)
                    area_points.append(self.pos)

                    area_points = np.array(area_points)
                    # plt.fill_between(
                    #     [np.min(area_points[:,0]),np.max(area_points[:,0])],
                    #     [np.min(area_points[:,1]),np.max(area_points[:,1])],
                    # )
                    xy = np.array([np.min(area_points[:,0]),np.min(area_points[:,1])])
                    wh = np.array([np.max(area_points[:,0]),np.max(area_points[:,1])]) - xy
                    r = rect(xy,*wh,facecolor=plt.cm.get_cmap('hsv',30)(random.randint(0,29)))
                    ax = plt.gca()
                    ax.add_patch(r)
                    

                    crossing_num = entry[1]
                    # self.direction = (self.direction + 270) % 360
                    # self.pos = self.node_pos[:,crossing_num-self.bound_min]

                    new_area = Area(
                        entry[:2],
                        self.direction,
                        self.current_segment,
                        self.pos,
                        left = True, #TODO
                        parent = self
                    )

                    self.sub_areas.append(new_area)
                    last_crossing = entry[1]
                    for sub in self.sub_areas:
                        if sub.bound_min < last_crossing < sub.bound_max:
                            sub.set_pos(self.pos,self.direction)
                            seg,done = sub.process_topology(topo,segment+1,last_crossing)
                            segment = seg - 1 #???
                            break

                    
                segment += 1

            if not done:
                self.add_link(topo[2,-1] < 0, False, False)
            return segment,True
        

        





    direction = 0
    # plt.show()
    world_area = Area(
        [-1,-1],
    )

    world_area.process_topology(topo,0)



    plt.show()
    # world_area.plot()





# def display_topo(topo):
#     _UNDER_WIDTH = 1
#     _STEP_SIZE = 0.01
#     # plt.show()
#     upside_down = False 
#     remove_start = topo[2,0] < 0
#     x = np.arange(-1,0+0.1,0.1)
#     y = np.zeros_like(x)
#     if remove_start:
#         # the one segment where we remove from the end instead.
#         x,y = x[:-_UNDER_WIDTH],y[:-_UNDER_WIDTH]
#     plt.plot(x,y,'b-')
#     wrap_around = -2
#     current_point = 0

#     gone_under = np.zeros_like(topo)
#     gone_over = np.zeros_like(topo)



#     for i in range(topo.shape[1]):
#         if topo[1,i] < i:
#             end_point = topo[1,i]
#             left_side = int(min(current_point,end_point))
#             radius = abs(current_point - end_point)/2.0
#             level = int(np.ceil(radius))
#             if (upside_down and np.any(gone_under[:level+1,[left_side,left_side+level]]))\
#             or (not upside_down and np.any(gone_over[:level+1,[left_side,left_side+level]])):
#                 #wrap around
#                 mid_point = (current_point + wrap_around)/2.0
#                 radius = abs(current_point - wrap_around)/2.0
#                 x,y = get_semicircle(mid_point,0,-radius,upside_down,_STEP_SIZE)
#                 if remove_start:
#                     x,y = x[_UNDER_WIDTH:],y[_UNDER_WIDTH:]
#                 remove_start = False
#                 plt.plot(x,y,'b-')
#                 upside_down = not upside_down
#                 current_point = wrap_around
#                 wrap_around -= 1

#                 left_side = int(min(current_point,end_point))
#                 radius = abs(current_point - end_point)/2.0
#                 level = int(np.ceil(radius))


#             mid_point = (current_point+end_point)/2.0
#             if current_point > end_point:
#                 radius *= -1
#             x,y = get_semicircle(mid_point,0,radius,upside_down,_STEP_SIZE)
#             # plt.plot(x,y,'b-')


#             if upside_down:
#                 gone_under[level,left_side:left_side+level] = 1
#             else:
#                 gone_over[level,left_side:left_side+level] = 1

#             upside_down = not upside_down
#             current_point = topo[1,i]
#         else:
#             x = np.arange(current_point,current_point+1+0.1,0.1)
#             y = np.zeros_like(x)
#             current_point += 1


#         if remove_start:
#             x,y = x[_UNDER_WIDTH:],y[_UNDER_WIDTH:]
#         if topo[2,i] < 0:
#             x,y = x[:-_UNDER_WIDTH],y[:-_UNDER_WIDTH]
#             remove_start = True
#         else:
#             remove_start = False

#         plt.plot(x,y,'b-')

#     if remove_start:
#         y_end = 0.1
#     else:
#         y_end = 0

#     if upside_down:
#         plt.plot([current_point,current_point],[-y_end,-0.25],'b-')
#     else:
#         plt.plot([current_point,current_point],[y_end,0.25],'b-')

#     plt.show()

    

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
    # random.seed(0)
    t = generate_random_topology(3).astype(np.int)
    print(t)
    display_topo(t)