from strokedata import Stroke, StrokeGroup
from settings import PARAMS_DIR, BONUS_PARAMS_DIR
from sklearn.externals import joblib
from utils import Line
import numpy as np
import sys
import numpy as np
from scipy.optimize import curve_fit
from sklearn import preprocessing

class Segmenter(object):
    total_merges = 0
    def __init__(self):
        self.cls = None
        self.pca = None
        self.min_max_scaler = None

    def _load_params(self):
        try:
            self.cls = joblib.load(PARAMS_DIR + 'segmentation-svc.pkl')
            #self.pca = joblib.load(PARAMS_DIR + 'pca.pkl')
            #self.min_max_scaler = joblib.load(PARAMS_DIR + 'segmentation-scaler.pkl')
        except Exception as e:
            print("!!! Error: couldn't load parameter file for segmenter")
            print("!!! Try running './train_classifiers.py' first")
            print("!!! Error details: %s" % e)
            sys.exit(1)

    def baseline_segmenter(self, strokes):
        partition = []
        for i, strk in enumerate(strokes):
            partition.append(StrokeGroup([strk], 'A_%s' % str(i), ' '))
        return partition

    def main_segmenter(self, strokes):
        if not self.cls:
            self._load_params()

        # Only one stroke, only one partition
        if len(strokes) <= 1:
            return [StrokeGroup(strokes, 'A_1', ' ')]

        #XXX: terrible hack
        for strk in strokes:
            strk.coords = strk.coords.T

        pairs = zip(strokes, strokes[1:])
        decisions = []
        for pair in pairs:
            features = SegmenterFeatures.get_features(pair, strokes)
            #features = self.min_max_scaler.transform(features)
            #features = self.pca.transform(features)


            pred = self.cls.predict(features)
            decisions.append(pred[0])


        groups = [[pairs[0][0]]]
        for i, dec in enumerate(decisions):
            if dec == 'merge':
                self.total_merges += 1
                groups[-1].append(pairs[i][1])
            else:
                groups.append([pairs[i][1]])

        partition = []
        for grp  in groups:
            partition.append(StrokeGroup(grp, 'A_%s' % str(i), ' '))

        #XXX: terrible hack
        for strk in strokes:
            strk.coords = strk.coords.T


        print("Total merges: %d" % self.total_merges)
        return partition


def _max_distance(coord, coords):
    coords = np.array(coords)
    diff = abs(coords - coord)
    squ = diff ** 2
    sum = squ.sum(axis=1)
    return np.sqrt(sum.max())

def _min_distance(coord, coords):
    coords = np.array(coords)
    diff = coords - coord
    squ = diff ** 2
    sum = squ.sum(axis=1)
    return np.sqrt(sum.min())

def _get_nearest_three(strk, all_strks, center):
    # Works by:
    # find the three strokes on the left and three on the right of target stroke
    # sort them by distance
    # pick the three closest

    idx = all_strks.index(strk)
    start = max(0, idx-3)
    end = min(idx+3, len(all_strks)-1)

    candidates = all_strks[start:end+1]
    candidates.remove(strk)
    
    dists = [None] * len(candidates)
    for i, cand in enumerate(candidates):
        dists[i] = _min_distance(center, cand.coords.T)
    zipped = zip(dists, candidates)
    nearest_3 = sorted(zipped, key=lambda obj: obj[0])
    nearest_3 = map(lambda x: x[1], nearest_3)

    return nearest_3


class SegmenterFeatures(object):
    #TODO: strk_grps is a bad name
    symbol_cls = None
    @staticmethod
    def get_features(strk_pair, strk_grps):
        """Compute all segmentation features for stroke pair (strk_pair)
        strk_pair: stroke pair
        strk_grps: the stroke group for the whole expression (including strk_pair)
        """

        if not SegmenterFeatures.symbol_cls:
            SegmenterFeatures.symbol_cls = joblib.load('params-recognition/recognition-rf.pkl')

        #for strk in strk_grps:
        #    strk.clean()


        #import pdb;pdb.set_trace()
        # TODO: commented out until we figure out why it's driving accuracy down
        geo_features = SegmenterFeatures._geometric_features(strk_pair, strk_grps)
        context_features = SegmenterFeatures.shape_context_features(strk_pair, strk_grps)
        #recognition_features = SegmenterFeatures.recognition_features(strk_pair, strk_grps)
        features = context_features #+ recognition_features
        #features = geo_features + context_features
        return features + geo_features
    
    @staticmethod
    def recognition_features(strk_pair, strks):
        # XXX: oh man ive to figure this hack out (preprocessing order bug, basically)
        
        single_grp = StrokeGroup([strk_pair[0]], 'A_1', ' ')

        #strk_pair[0].coords = strk_pair[0].coords.T
        
        strk_pair[0].coords = strk_pair[0].coords.T
        single_grp.preprocess()


        
        single_grp_features = single_grp.get_features()
        cls = SegmenterFeatures.symbol_cls
        single_probs = cls.predict_proba(single_grp_features).flatten().tolist()


        #strk_pair[0].coords = strk_pair[0].coords.T

        for strk in strk_pair:
            strk.coords = strk.coords.T

        pair_grp = StrokeGroup(list(strk_pair), 'A_2', ' ')
        #XXX: undo this
        pair_grp.preprocess()

        # #TODO: do for three strokes also
        pair_grp_features = pair_grp.get_features()

        
        # #XXX: revert back hack
        # for strk in strks:
        #     strk.coords = strk.coords.T

        pair_probs = cls.predict_proba(pair_grp_features).flatten().tolist()

        return [max(single_probs), max(pair_probs)]

    @staticmethod
    def shape_context_features(strk_pair, strk_grps):

        #divider = max(strk_pair[0].coords.shape[1]/3, 1)

        center = strk_pair[0].center()
        all_coords = np.vstack((strk_pair[0].coords.T, strk_pair[1].coords.T))
        radius = _max_distance(center, all_coords)

        strk_pair_bin = MainBin(center, radius)
        counts = strk_pair_bin.get_count(all_coords)
        #counts = np.array(counts)/float(divider)

        # nearest_three = _get_nearest_three(strk_pair[0], strk_grps, center)
        # nearest_three = [strk_pair[0].coords.T] + [strk.coords.T for strk in nearest_three]
        # nearest_three = tuple(nearest_three)
        # local_all_coords = np.vstack(nearest_three)


        # local_radius = _max_distance(center, local_all_coords)
        # local_strk_bin = MainBin(center, local_radius)
        # local_counts = local_strk_bin.get_count(local_all_coords)
        # local_counts = np.array(local_counts)/float(divider)

        # # TODO: not using global features for now. They're slow and give only around 5% F-Measure boost
        # global_all_coords = tuple([strk.coords.T for strk in strk_grps])
        # global_all_coords = np.vstack(global_all_coords)
        # global_radius = _max_distance(center, global_all_coords)
        # global_strk_bin = MainBin(center, global_radius)
        # global_counts = global_strk_bin.get_count(global_all_coords)
        # global_counts = np.array(global_counts)/float(strk_pair[0].coords.shape[1])

        return counts#.tolist() #+ local_counts.tolist() #+ global_counts.tolist()

    @staticmethod
    def _geometric_features(strk_pair, strk_grps):
        stroke1 = strk_pair[0]
        stroke2 = strk_pair[1]
        bb1 = BoundingBox(stroke1)
        bb2 = BoundingBox(stroke2)
        # return [parallelity(stroke1,stroke2), average_distance(stroke1,stroke2), distance_beginning_end(stroke1,stroke2), \
        #         min_distance(stroke1,stroke2),bb1.distance(bb2), bb1.overlap_distance2(bb2)]
        # print bb1.overlap(bb2)
        return [average_distance(stroke1,stroke2), parallelity(stroke1,stroke2), min_distance(stroke1,stroke2)]



def offset_beginning_end(stroke1, stroke2):
    pass

def distance_beginning_end(stroke1, stroke2):
    return np.linalg.norm(stroke2.coords.T[0] - stroke1.coords.T[-1])




def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B


def average_distance(stroke1, stroke2):
    center1 = np.mean(stroke1.coords.T,axis=0)
    center2 = np.mean(stroke2.coords.T,axis=0)
    return np.linalg.norm(center2 - center1)


def parallelity(stroke1, stroke2):
    try:
        A,B = curve_fit(f, stroke1.coords.T[:,0], stroke1.coords.T[:,1])[0] 
        A2,B2 = curve_fit(f, stroke2.coords.T[:,0], stroke2.coords.T[:,1])[0] 
    except:
        ### When one stroke has <= 2 data points, how to handle??
        return -99999
    return A2 - A


def min_distance(stroke1, stroke2):
    min_dist = 99999
    for p1 in stroke1.coords.T:
        for p2 in stroke2.coords.T:
            min_dist = min(min_dist, np.linalg.norm(p2 - p1))
    return min_dist

class BoundingBox(object):

    def __init__(self, stroke):
        mins = np.amin(stroke.coords.T,axis=0)
        maxs = np.amax(stroke.coords.T,axis=0)
        self.minx = mins[0]
        self.miny = mins[1]
        self.width = abs(maxs[0] - mins[0])
        self.height = abs(maxs[1] - mins[1])
        self.maxx = maxs[0]
        self.maxy = maxs[1]
        self.center = np.array([self.minx + (self.width/2), self.miny + (self.height/2)])
    

    def distance(self, other):
        return np.linalg.norm(self.center-other.center)


    def overlap(self,other):
        left = max(self.minx, other.minx)
        right = min(self.maxx, other.maxx)
        top = max(self.maxy, other.maxy)
        bottom = max(self.miny, other.miny)
        return (right - left) * (top - bottom)

    """I don't account for overlap, reason for max."""
    def min_h_distance(self, other):
        if other.maxx > self.maxx: #If other is right of shape
            return max(other.minx - self.maxx,0)
        if other.minx < self.minx: #If other is left of shape
            return max(self.minx - other.maxx,0)
        return 0


    """Here we account for overlap (only)"""
    def overlap_distance(self, other):
        if other.maxx > self.maxx: #If other is right of shape
            return max(self.maxx - other.minx,0)
        if other.minx < self.minx: #If other is left of shape
            return max(other.maxx - self.minx,0)
        return 0

    """Here we account for overlap + non overlap."""
    def overlap_distance2(self, other):
        if other.maxx > self.maxx: #If other is right of shape
            return abs(self.maxx - other.minx)
        if other.minx < self.minx: #If other is left of shape
            return abs(other.maxx - self.minx)
        return 0
        


class MainBin(object):
    def __init__(self, center, radius):
        self.bins = self._create_bins(center, radius)
        self.bins = sorted(self.bins, key=lambda bin: bin.radius)

    @staticmethod
    def _create_bins(center, radius):
        bins = []
        for i in [1, 2, 4, 8, 16]:
            bins.append(Bin(center, radius/float(i)))

        return bins

    def get_count(self, coords):
        #import pdb; pdb.set_trace()
        counts = [[0] * 12] * 5
        for bin_idx, bin in enumerate(self.bins):
            bin_count = 0
            for angle_idx, angle_bin in enumerate(bin.angle_bins):
                #print("BINS")
                #import pdb; pdb.set_trace()
                total = angle_bin.counts(coords)

                bin_count += total
                # Discount the counts that occur in the angular bin of
                # *smaller* circles
                for i in range(0, bin_idx):
                    total -= counts[i][angle_idx]
                counts[bin_idx][angle_idx] = total

            #print("Bin count: " + str(bin_count))
            if bin_count > 0:
                for idx, angle_bin in enumerate(bin.angle_bins):
                    counts[bin_idx][idx] = counts[bin_idx][idx]/float(bin_count)


        final_counts = []
        for row in counts:
            final_counts.extend(row)
        # Normalize
        #TODO: we should be normailzing the current stroke's coord count
        return final_counts
        #return (np.array(counts).flatten() / float(len(coords.tolist())))

class Bin(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.angle_bins = self.create_angle_bins()

    def create_angle_bins(self):
        angle_bins = []

        for quadrant in [1, 2, 3, 4]:
            for i in range(0, 3):
                angle_bin = AngleBin(self.center, self.radius, quadrant, i)
                angle_bins.append(angle_bin)
        return angle_bins


class AngleBin(object):
    def __init__(self, center, radius, quadrant, idx):
        self.center = center
        self.radius = radius
        self.quadrant = quadrant
        self.idx = idx

        self.line1, self.line2 = self.create_lines()


    def create_lines(self):
        cx, cy = self.center
        r = self.radius
        i = self.idx
        if self.quadrant == 1:
            pt1 = [ (i/3.0)*r, ((3-i)/3.0)*r ]
            pt2 = [ ((i+1)/3.0)*r, ((3-(i+1))/3.0)*r ]

            pt1 = np.array(pt1) + np.array(self.center)
            pt2 = np.array(pt2) + np.array(self.center)
        elif self.quadrant == 2:
            pt1 = [ cx+(((3-i)/3.0)*r), cy-((i/3.0)*r) ]
            pt2 = [ cx+((3-(i+1))/3.0)*r, cy-(((i+1)/3.0)*r) ]

        elif self.quadrant == 3:
            # Copied from 1st quadrant
            pt1 = [ cx-((i/3.0)*r), cy-(((3-i)/3.0)*r) ]
            pt2 = [ cx-(((i+1)/3.0)*r), cy-(((3-(i+1))/3.0)*r) ]

        elif self.quadrant == 4:
            # Copied from 2nd quadrant
            pt1 = [ cx-(((3-i)/3.0)*r), cy+((i/3.0)*r) ]
            pt2 = [ cx-(((3-(i+1))/3.0)*r), cy+(((i+1)/3.0)*r) ]

        else:
            raise Exception("Unrecognized quadrant")

        # line 1
        if (pt1[0]-self.center[0]) == 0:
            line1 = Line(x=pt1[0])
        else:
            m1 = (pt1[1]-self.center[1])/float(pt1[0]-self.center[0])
            b1 = self.center[1] - (m1*self.center[0])
            line1 = Line(m=m1, b=b1)

        # line 2
        if (pt2[0]-self.center[0]) == 0:
            line2 = Line(x=pt2[0])
        else:
            m2 = (pt2[1]-self.center[1])/float(pt2[0]-self.center[0])
            b2 = self.center[1] - (m2*self.center[0])
            line2 = Line(m=m2, b=b2)

        return line1, line2

    def get_quadrant(self, pt):
        if pt[0] >= 0 and pt[1] >= 0:
            return 1
        elif pt[0] >= 0 and pt[1] < 0:
            return 2
        elif pt[0] < 0 and pt[1] < 0:
            return 3
        return 4

    def counts(self, coords):
        #import pdb; pdb.set_trace()
        count = 0
        for coord in coords:
            if self.get_quadrant(coord) == self.quadrant:
                xs = [self.line1.get_x(coord[1]), self.line2.get_x(coord[1])]
                x1, x2 = min(xs), max(xs)

                #TODO: what about if exactly on the line
                if x1  < coord[0] <= x2:
                    count += 1
        return count








