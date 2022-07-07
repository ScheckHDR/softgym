import numpy as np

def compare_topology(topo1, topo2):
    if np.all(topo1 == topo2) \
        or np.all(topo1 == flip_topology(topo2))\
        or np.all(topo1 == reverse_topology(topo2)):
        return True
    return False


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