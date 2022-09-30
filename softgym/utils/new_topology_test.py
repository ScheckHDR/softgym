from curses.ascii import CR
from pickle import TRUE
import numpy as np
import copy

class InvalidTopology(Exception):
    pass



class Segment:
    def __init__(self,segment_num:int):
        self.end_crossing = None
        self.start_crossing = None
        self.segment_num = segment_num

    def attach_crossing(self,crossing,is_end:bool) -> None:
        if is_end:
            self.end_crossing = crossing
        else:
            self.start_crossing = crossing

    def get_other_crossing(self,crossing:"Crossing") -> "Crossing":
        if self.start_crossing == crossing:
            return self.end_crossing
        elif self.end_crossing == crossing:
            return self.start_crossing
        else:
            raise ValueError("Input crossing does not belong to this segment.")



class Crossing:
    def __init__(self,numbers,sign):
        self.numbers = numbers
        self.sign = sign

        self.over_in = None
        self.over_out = None

        self.under_in = None
        self.under_out = None

        self.rotation = 0

        self.attachment_points = {}

    def attach_segment(self,segment:Segment,over:bool,incoming:bool) -> None:
        num_attached = len(self.attachment_points)
        if num_attached == 0:
            self.attachment_points[segment] = np.array([-1, 0])
        elif num_attached == 1:
            self.attachment_points[segment] = np.array([ 1, 0])
        elif num_attached == 2:
            self.attachment_points[segment] = np.array([ 0, 1]) if self.sign > 0 == over else np.array([0,-1])
        elif num_attached == 3:
            self.attachment_points[segment] = np.array([ 0,-1]) if self.sign > 0 == over else np.array([0,1])
        elif num_attached == 4:
            raise Exception("too many segments attached.")

        if over:
            if incoming:
                self.over_in = segment
            else:
                self.over_out = segment
        else:
            if incoming:
                self.under_in = segment
            else:
                self.under_out = segment

    def get_next_segment(self,segment:Segment) -> Segment:
        if self.under_in == segment:
            return self.under_out
        elif self.over_in == segment:
            return self.over_out
        else:
            raise Exception

    def get_previous_segment(self,segment:Segment) -> Segment:
        if self.under_out == segment:
            return self.under_in
        elif self.over_out == segment:
            return self.over_in
        else:
            raise Exception

    def get_connected_segment(self,segment:Segment) -> Segment:
        if self.under_in == segment:
            return self.under_out
        elif self.over_in == segment:
            return self.over_out
        elif self.under_out == segment:
            return self.under_in
        elif self.over_out == segment:
            return self.over_in
        else:
            raise Exception

    def get_side_segment(self,segment:Segment,left) -> Segment:
        if self.sign > 0:
            if segment in [self.under_in,self.under_out]:
                return self.over_in if left > 0 else self.over_out
            elif segment in [self.over_in,self.over_out]:
                return self.under_out if left > 0 else self.under_in
        elif self.sign < 0:
            if segment in [self.under_in,self.under_out]:
                return self.over_out if left > 0 else self.over_in
            elif segment in [self.over_in,self.over_out]:
                return self.under_in if left > 0 else self.under_out
            
    def is_side_in(self,left:bool) -> bool:
        if left:
            return self.sign < 0
        else:
            return self.sign > 0

    def is_crossing(self,number) -> bool:
        return np.any(number == self.numbers)


class Region:
    def __init__(self):
        pass


class RopeTopology:
    def __init__(self,topo_np):
        self._topology = topo_np

        self.segments = []
        self.crossings = [Crossing([-1,-1],True)] #initial crossing can have bogus values
        prev_crossing = self.crossings[-1]
        prev_over = True #bogus value for initial "crossing"

        for col in range(topo_np.shape[1]):
            new_segment = Segment(col)
            prev_crossing.attach_segment(new_segment,prev_over>0,False)
            new_segment.attach_crossing(prev_crossing,False)
            is_over = topo_np[2,col]>0
            if topo_np[0,col] < topo_np[1,col]:
                # end crossing does not exist yet; create it.
                self.crossings.append(Crossing(topo_np[:2,col],topo_np[3,col]))
                self.crossings[-1].attach_segment(new_segment,is_over,True)
                new_segment.attach_crossing(self.crossings[-1],True)
                prev_crossing = self.crossings[-1]
            else:
                # crossing already exists.
                c = [c for c in self.crossings if c.is_crossing(topo_np[0,col])][0]
                c.attach_segment(new_segment,is_over,True)
                new_segment.attach_crossing(c,True)

                # would have created a new region, check that this doesn't invalidate anything.
                # check left
                before_crossing = prev_crossing
                next_seg = c.get_side_segment(new_segment,True) 
                next_crossing = next_seg.get_other_crossing(c)

                side_ins = []
                while True:
                    test_seg = next_crossing.get_side_segment(next_seg,True)

                    if test_seg is None:
                        side_ins.append(next_crossing.is_side_in(True))
                        next_seg = next_crossing.get_connected_segment(next_seg)
                    else:
                        next_seg = test_seg

                    next_crossing = next_seg.get_other_crossing(next_crossing)

                    if (next_seg == new_segment):
                        # Done a complete loop
                        break
                if sum(side_ins) % 2 != 0:
                    raise InvalidTopology
                
                prev_crossing = c
                prev_over = is_over


            

if __name__ == '__main__':
    t = np.array([
        [ 0, 1, 2, 3],
        [ 2, 3, 0, 1],
        [ 1, 1,-1,-1],
        [-1, 1,-1, 1]
    ])

    RopeTopology(t)

