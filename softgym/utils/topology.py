from turtle import pos
import numpy as np
import random

def get_topological_representation(positions):
    intersections = []
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
    return topo

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
    topo[2,:] *= -1
    return topo

def reverse_topology(topo):
    # gets the topological representation that matches starting from the other end of the rope.
    num_crossings = topo.shape[1]

    reversed_topo = topo[:,::-1]   
    # rebase
    reversed_topo[0,:] = np.arange(num_crossings)
    reversed_topo[1,:] = num_crossings + 1 - reversed_topo[1,:]


    return reversed_topo

def ccw(A,B,C):
    # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
    return (C[2]-A[2]) * (B[0]-A[0]) > (B[2]-A[2]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)





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