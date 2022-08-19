import numpy as np
import random

INTERSECT_NUM = 0
CORRESPONDING = 1
OVER_UNDER = 2
SIGN = 3


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





# def is_knot(topo):

#     if topo.size == 0:
#         # Topology got reduced to the trivial state.
#         return False

#     #R0 (made up) - if simplification winds up with a "crossing" with only a single index
#     if np.any(topo[0,:] == topo[1,:]):
#         # remove stuff
#         reduced_topo = topo
#         return is_knot(reduced_topo)
    
#     for i in range(topo.shape[1]-1):
#         # R1 - A crossing with its corresponding entrance in the next column.
#         if np.all(topo[:2,i] == topo[2::-1,i+1]):
#             # remove stuff
#             reduced_topo = topo
#             return is_knot(reduced_topo)

#         # R2 - A 


#     return True


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

            sign = random.choice([1]) # sign is currently unused. will be -1 or 1 if sign gets added back in
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
    a = np.array([
        [0,1],
        [1,0],
        [1,-1]
    ])
    b = np.array([
        [0,1],
        [1,0],
        [-1,1]
    ])
    c = np.array([
        [0,1,2,3],
        [1,0,3,2],
        [1,-1,1,-1]
    ])
    d = np.array([
        [0,1,2,3],
        [1,0,3,2],
        [1,-1,-1,1]
    ])
    e = np.array([
        [0,1,2,3],
        [1,0,3,2],
        [-1,1,-1,1]
    ])
    f = np.array([
        [0,1,2,3],
        [1,0,3,2],
        [-1,1,1,-1]
    ])
    
    print(compare_topology(d,f))