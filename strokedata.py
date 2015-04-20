import numpy as np
import matplotlib.pyplot as plt
from utils import Line, get_line_crossing, generate_subcrossings

class StrokeGroup(object):
    def __init__(self, strokes, annot_id, target):
        self.strokes = strokes
        # Sort strokes by id which corresponds to the order at which they were written
        self.strokes = sorted(self.strokes, key=lambda strk: strk.id)
        self.annot_id = annot_id
        self.target = target
        self.prediction = None
        # Flag to tell whether symbol is preprocessed
        self._is_preprocessed = False
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None

    def get_coords(self):
        all_coords = []
        for stroke in self.strokes:
            all_coords.extend(stroke.coords.T.tolist())

        return np.array(all_coords)

    def strk_ids(self):
        return ", ".join([str(strk.id) for strk in self.strokes])


    def preprocess(self):
        for stroke in self.strokes:
            stroke.clean()


        # It's best to do size normaliztion after cleaning
        [self.xmin, self.ymin] = np.min([stroke.coords.min(axis=0)
                               for stroke in self.strokes], axis=0)
        [self.xmax, self.ymax] = np.max([stroke.coords.max(axis=0)
                            for stroke in self.strokes], axis=0)

        # Do size normalization
        ydiff = self.ymax - self.ymin
        wh_ratio = (self.xmax-self.xmin)/ydiff if ydiff != 0 else self.xmax-self.xmin
        for strk in self.strokes:
            strk.scale_size(self.xmin, self.ymin, self.xmax-self.xmin,
                    self.ymax-self.ymin, wh_ratio, yhigh=100)

        # Recalculate min & maxes after scaling
        [self.xmin, self.ymin] = np.min([stroke.coords.T.min(axis=0)
                               for stroke in self.strokes], axis=0)
        [self.xmax, self.ymax] = np.max([stroke.coords.T.max(axis=0)
                            for stroke in self.strokes], axis=0)

        self._is_preprocessed = True
        #self.get_features()

    def get_features(self):
        #TODO: append line length & angle features

        crossings = self.get_crossings_features()
        global_features = self.get_global_features()
        extra_features = self.get_extra_features()
        return crossings + global_features + extra_features

    def __unicode__(self):
        return "<StrokeGroup '%s' %s>" % (str(self.target), str(self.strokes))

    def __repr__(self):
        return self.__unicode__()

    def render(self, offset=0):
        """
            offset: Offset value to add on each X trace coordinate. Useful
                    for visualizations so that symbols appear left to right
                    instead of on top of each other.
        """
        out = "\n".join([strk.render(offset=offset) for strk in self.strokes])
        return out

    def get_global_features(self):
        n_traces = len(self.strokes)
        line_len = self._get_line_length()

        if line_len[1] > 0:
            aspect_ratio = float(line_len[0])/line_len[1]
        else:
            aspect_ratio = float(line_len[0])

        all_coords = self.get_coords()
        cov = self.get_cov(all_coords)
        #TODO: not actually using covariance feature...
        mean_xy = all_coords.sum(axis=0) / float(all_coords.shape[0])
        # TODO: remaining: angular change, sharp points, covariances
        return [n_traces, aspect_ratio] + line_len + mean_xy.tolist()

    def get_cov(self, coords):
        return np.cov(coords.T)[0,1]

    def get_extra_features(self):
        angle_means = []
        angle_vars = []
        distances = []
        curv_means = []
        curv_vars = []
        slope_means = []
        slope_vars = []
        npoints = []
        for stroke in self.strokes:

            angles = stroke._angle_feature()
            dist = stroke._norm_line_length()
            curves = stroke._curvature()
            slopes = stroke._slope()

            if len(angles) != 0:
                angle_means.append(np.mean(angles))
                angle_vars.append(np.var(angles))
            if len(curves) != 0:
                curv_means.append(np.mean(curves))
                curv_vars.append(np.var(curves))
            if len(slopes) != 0:
                slope_means.append(np.mean(slopes))
                slope_vars.append(np.var(slopes))
            if len(dist) != 0:
                distances.append(np.sum(dist))
            npoints.append(len(stroke.coords.T))
        #Sometimes the strokes are so short there's no angle/curve
        if len(angle_means) == 0:
            angle_means = [-99]
            angle_vars = [-99]
        if len(curv_means) == 0:
            curv_means = [-99]
            curv_vars = [-99]
        if len(slope_means) == 0:
            slope_means = [-99]
            slope_vars = [-99]
        if len(distances) == 0:
            distances = [-99]
        return [np.mean(angle_means),np.mean(angle_vars),np.sum(distances),np.mean(curv_means),np.mean(curv_vars)]


    def _get_line_length(self):
        # We want the sum of line lengths for all the strokes
        # and the average of the line lengths
        total_len = 0
        for strk in self.strokes:
            _, ymin = np.min(strk.coords.T, axis=0)
            _, ymax = np.max(strk.coords.T, axis=0)
            total_len += (ymax-ymin)
        return [total_len, float(total_len)/len(self.strokes)]

    def get_crossings_features(self):
        if not self._is_preprocessed:
            print("!!! Warning: computing features before preprocessing.")

        horiz = self._crossing_features(direction='horiz')
        vert = self._crossing_features(direction='vert')

        horiz = np.array(horiz).flatten().tolist()
        vert = np.array(vert).flatten().tolist()

        if len(horiz) < 15:
            horiz.extend([0.0] * (15-len(horiz)))
        if len(vert) < 15:
            vert.extend([0.0] * (15-len(vert)))

        return np.array([horiz, vert]).flatten().tolist()

    def plot(self):
        for strk in self.strokes:
            plt.scatter(strk.coords[:,0], strk.coords[:,1])

        plt.show()

    def _crossing_features(self, direction='vert'):
        nlines = 9.0
        features = []
        if direction == 'vert':
            start, end = self.xmin, self.xmax
        else:
            start, end = self.ymin, self.ymax

        # Create 5 regions
        step = (end-start)/nlines

        # Generates 9 lines for each region.
        # So subcrossings is a list of 5 lists with 9 lines each
        subcrossings = [generate_subcrossings(start, start+(step*i),
                            nlines, direction=direction) for i in range(1, 6)]

        crossings = []
        for subcrossing in subcrossings:
            crossings = []
            for line in subcrossing:
                crossings.append(get_line_crossing(line, self))

            features.append(np.array(crossings).sum(axis=0)/nlines)

        return features


class Stroke(object):
    def __init__(self, coords, id):
        if len(coords.strip().split(",")[0].split()) > 2:
            coords = coords.replace(",","").split(' ')
            xcol = np.array([np.array(coords[0::3]).astype(np.float)])
            ycol = np.array([np.array(coords[1::3]).astype(np.float)])
        else:
            coords = coords.replace(",","").split(' ')
            xcol = np.array([np.array(coords[0::2]).astype(np.float)])
            ycol = np.array([np.array(coords[1::2]).astype(np.float)])

        self.id = int(id)
        self.raw_coords = coords
        self.coords = np.vstack([xcol,ycol]).T
        self.rcoords = np.vstack([xcol,ycol]).T

    def _norm_line_length(self):
        dists = []
        p1 = self.coords.T[0]
        for p2 in self.coords.T[1:]:
            dist = np.linalg.norm(p2-p1)
            dists.append(dist)
            p1 = p2
        return np.array(dists)

    def _angle_feature(self):
        angles = []
        p1 = self.coords.T[0]
        for p2 in self.coords.T[1:]:
            a = angle(p2,p1)
            angles.append(a)
        return np.array(angles)

    def _curvature(self):
        curves = []
        for i,row in enumerate(self.coords.T[2:]):
            curves.append(curve(self.coords.T[i-2],self.coords.T[i],self.coords.T[i+2]))
        return np.array(curves)

    def _slope(self):
        slopes = []
        for i,row in enumerate(self.coords.T[2:]):
            slopes.append(slope(self.coords.T[i-2],self.coords.T[i+2]))
        return np.array(slopes)

    def __unicode__(self):
        return "<Stroke (id=%s)>" % self.id

    def __repr__(self):
        return self.__unicode__()

    def render(self, offset=0):
        pairs = self.coords.T.tolist()
        out = ''

        for pair in pairs:
            out += "%s %s, " % (str(pair[0]+offset), str(pair[1]))
        out = out.strip().rstrip(",")
        return '<trace id="%s">%s</trace>' % (self.id, out)

    def scale_size(self, xmin, ymin, xrng, yrng, wh_ratio, yhigh=100):
        xcol = self.coords[:,0]
        ycol = self.coords[:,1]

        # y prime and x prime columns
        if yrng != 0:
            yp_col = (ycol-ymin)*(float(yhigh)/yrng)
        else:
            yp_col = (ycol-ymin)*(float(yhigh)/(yrng+1))

        if xrng != 0:
            xp_col = (xcol-xmin)*(float(wh_ratio*yhigh)/xrng)
        else:
            xp_col = (xcol-xmin)*(float(wh_ratio*yhigh)/(xrng+1))

        self.coords = np.vstack([xp_col, yp_col])

    def clean(self):
        """Does duplicate filtering & stroke smoothing"""
        if len(self.coords) < 1:
            return

        # Remove duplicates
        pairs = self.coords.tolist()
        last = pairs[0]
        uniques = [last]
        for coord in pairs[1:]:
            if last == coord:
                continue
            uniques.append(coord)
            last = coord

        # Do coordinate smoothing
        # Works by replacing every point by the average of the previous,
        # current and next point (except for the first and last point)
        last_idx = len(uniques) - 1
        for i, pair in enumerate(uniques):
            if i in (0, last_idx,):
                continue
            pair[0] = (uniques[i-1][0]+uniques[i][0]+uniques[i+1][0]) / 3.0
            pair[1] = (uniques[i-1][1]+uniques[i][1]+uniques[i+1][1]) / 3.0

        self.coords = np.array(uniques)

    def plot(self):
        plt.scatter(self.coords[:,0], self.coords[:,1])
        plt.show()




def distance(p1,p2):
    return np.linalg.norm(p2-p1)

def length(v):
    return np.sqrt(np.dot(v, v))

def angle(v1,v2):
    dot = (length(v1) * length(v2))
    #orthogonal
    if dot == 0:
        return 1.57079
    return np.arccos(round(np.dot(v1, v2) /dot))

def curve(p1,p2,p3):
    a = distance(p1,p2)
    b = distance(p2,p3)
    c = distance(p1,p3)
    if a == 0 or b == 0:
        return -99
    res = (a**2 + b**2 - c**2) / (2*a*b)
    angleC = np.arccos(round(res))
    return angleC

def slope(p1,p2):
    a = distance(p1,p2)
    c = -1
    mag = (np.linalg.norm(a))
    if mag == 0:
        return -99
    unit = (a/mag)
    angleC = np.arccos(c/unit)
    return angleC

