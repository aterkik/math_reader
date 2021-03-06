"""
    Authors: Andamlak Terkik, Kevin Carbone
"""
import numpy as np
import os
import shutil

class Line(object):
    def __init__(self, m=None, b=None, x=None):
        """If m is None, equation is X=x,
           If m is not None, equation is y = mX + b
           For the second case, we don't care about the x provided.
        """
        self.m = m
        self.b = b
        self.x = x

    def check_intersection(self, xy, rng=5):
        """xy should be a tuple: (x, y)"""
        if self.m is None:
            if self.x-rng <= xy[0] <= self.x+rng:
                return xy[0]
        else:
            if self.b-rng <= xy[1] <= self.b+rng:
                return xy[1]
        return None

    def get_y(self, x):
        return x*m + b

    def get_x(self, y):
        if self.x is not None:
            # vertical line
            return self.x

        #if self.m == 0:
        #    #XXX: dangerously bad hack
        #    return 0
        try:
            return (y-self.b)/float(self.m)
        except Exception as e:
            #import pdb; pdb.set_trace()
            print(e)
            pass



    def __str__(self):
        if self.m is None:
            return "<Line: x=%s>" % self.x
        else:
            return "<Line: y=%sx + %s>" % (self.m, self.b)

    def __repr__(self):
        return self.__str__()


def get_line_crossing(line, symbol):
    count_inters, first_inters, last_inters = 0, None, None

    for coord in symbol.get_coords():
        intersection = line.check_intersection(coord)
        if intersection:
            count_inters += 1
            if first_inters is None or intersection < first_inters:
                first_inters = intersection
            if last_inters is None or intersection > last_inters:
                last_inters = intersection

    # Ensure this gives desired result
    first_inters = first_inters if first_inters is not None else 0
    last_inters = last_inters if last_inters is not None else 0
    return [count_inters, first_inters, last_inters]


def generate_subcrossings(start, end, k, direction='vert'):
    lines = []
    if start == end:
        return lines

    step = (end-start)/float(k)
    for i in np.arange(start+(step/2), end, step):
        if direction == 'vert':
            lines.append(Line(x=i))
        else:
            lines.append(Line(m=0, b=i))
    return lines

def empty_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass

def create_dir(path):
    empty_dir(path)
    if not os.path.exists(path):
        os.makedirs(path)
