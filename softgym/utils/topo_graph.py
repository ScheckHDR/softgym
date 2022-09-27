import numpy as np
import queue
import topology as topo
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_topo = topo.RopeTopology(
        np.array([
        [ 0, 1, 2, 3, 4, 5],
        [ 1, 0, 4, 5, 2, 3],
        [-1, 1, 1,-1,-1, 1],
        [ 1, 1, 1, 1, 1, 1]
    ]))

    f, (ax1,ax2) = plt.subplots(1,2)
    for i in range(6):
        for s in [-1,1]:
            new_topo, segs = topo.add_R1(test_topo,i,s)
            # print(new_topo)

            test_topo.display(ax1,i)
            new_topo.display(ax2,segs)

            plt.show()

