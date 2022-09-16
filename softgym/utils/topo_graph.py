import numpy as np
import queue
import utils.topology as topo


class Node:
    def __init(self,state = None):
        self.state = state or topo.generate_random_topology(0)


    def expand(self):
        raise NotImplementedError