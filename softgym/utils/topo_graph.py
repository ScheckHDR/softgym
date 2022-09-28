import numpy as np
import queue
import topology as topo
import matplotlib.pyplot as plt
import time
import random

if __name__ == '__main__':
    random.seed(2)
    test_topo = topo.RopeTopology(
        np.array([
        [ 0, 1, 2, 3, 4, 5],
        [ 1, 0, 4, 5, 2, 3],
        [-1, 1, 1,-1,-1, 1],
        [ 1, 1, 1, 1, 1, 1]
    ]))

    f,axs = plt.subplots(2,2)

    seg1 = random.randint(0,test_topo.size())
    left1 = random.choice([True,False])
    over1 = random.choice([True,False])
    new1 , segs1 = topo.add_R1(test_topo,seg1,over1,left1)

    seg2 = random.randint(0,new1.size())
    left2 = random.choice([True,False])
    over2 = random.choice([True,False])
    new2,segs2 = topo.add_R1(new1,seg2,over2,left2)

    test_topo.display(axs[0,0],seg1,left1==over1)
    new1.display(axs[0,1],segs1)
    new1.display(axs[1,0],seg2,left2 != over2)
    new2.display(axs[1,1],segs2)
    print(new2)
    plt.show()



    # plt.show()
    # t,_ = topo.add_R1(test_topo,4,-1)
    # t.display()

    # f, (ax1,ax2) = plt.subplots(1,2)
    # for i in [0,1,2,3,4,5]:
    #     for s in [-1,1]:
    #         new_topo, segs = topo.add_R1(test_topo,i,s)
    #         print(f'{i},{s}')
    #         print(new_topo)

    #         try:
    #             test_topo.display(ax1,i)
    #             new_topo.display(ax2,segs)
    #             plt.pause(5)
                
    #             ax1.clear()
    #             ax2.clear()
    #         except topo.InvalidTopology:
    #             print('invalid')

