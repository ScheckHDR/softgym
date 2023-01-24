
 
def add_R2(topology:RopeTopology,over_ind:int,under_ind:int,sign,over_first:bool=False):
    T = deepcopy(topology._topology)



def remove_R1(topology:RopeTopology,ind):
    if topology.corresponding(ind) == ind -1:
        T = deepcopy(topology._topology)

        T = np.delete(T,[ind-1,ind],axis=1)
        T[np.where(T[:2,:] > ind)] -= 2
        T[np.where(T[INTERSECT_NUM:CORRESPONDING+1,:] > ind+1)] -= 2

        return RopeTopology(T)
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


