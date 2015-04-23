from strokedata import Stroke, StrokeGroup
from settings import PARAMS_DIR, BONUS_PARAMS_DIR
from sklearn.externals import joblib
from utils import Line
import numpy as np
import sys

class Segmenter(object):
    total_merges = 0
    def __init__(self):
        self.svm = None

    def _load_params(self):
        try:
            self.svm = joblib.load(PARAMS_DIR + 'segmentation-svc.pkl')
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
        #import pdb; pdb.set_trace()
        if not self.svm:
            self._load_params()

        #XXX: terrible hack.
        for strk in strokes:
            strk.coords = strk.coords.T

        pairs = zip(strokes, strokes[1:])
        decisions = []
        for pair in pairs:
            #import pdb; pdb.set_trace()
            features = SegmenterFeatures.get_features(pair, strokes)
            features = features.tolist()

            pred = self.svm.predict(features)
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
    diff = coords - coord
    squ = diff ** 2
    sum = squ.sum(axis=1)
    return np.sqrt(sum.max())


class SegmenterFeatures(object):
    #TODO: strk_grps is a bad name
    @staticmethod
    def get_features(strk_pair, strk_grps):
        """Compute all segmentation features for stroke pair (strk_pair)
        strk_pair: stroke pair
        strk_grps: the stroke group for the whole expression (including strk_pair)
        """
        center = strk_pair[0].center()
        try:
            all_coords = np.vstack((strk_pair[0].coords.T, strk_pair[1].coords.T))
        except Exception as e:
            #import pdb; pdb.set_trace()
            pass

        radius = _max_distance(center, all_coords)

        strk_pair_bin = MainBin(center, radius)
        counts = strk_pair_bin.get_count(all_coords)
        counts = np.array(counts)/float(strk_pair[0].coords.shape[1])
        return counts



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
        counts = [[0] * 12] * 5
        for bin_idx, bin in enumerate(self.bins):
            for angle_idx, angle_bin in enumerate(bin.angle_bins):
                #print("BINS")
                #import pdb; pdb.set_trace()
                total = angle_bin.counts(coords)

                # Discount the counts that occur in the angular bin of
                # *smaller* circles
                #for i in range(0, bin_idx):
                #    total -= counts[i][angle_idx]
                counts[bin_idx][angle_idx] = total

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
        step = (self.radius*2)/12.0

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
        r = self.radius
        i = self.idx
        if self.quadrant == 1:
            pt1 = [ (i/3.0)*r, ((3-i)/3.0)*r ]
            pt2 = [ ((i+1)/3.0)*r, ((3-(i+1))/3.0)*r ]
        elif self.quadrant == 2:
            pt1 = [ ((3-i)/3.0)*r, (i/3.0)*r ]
            pt2 = [ ((3-(i+1))/3.0)*r, ((i+1)/3.0)*r ]

            pt1[1] = -1*pt1[1]
            pt2[1] = -1*pt2[1]
        elif self.quadrant == 3:
            # Copied from 1st quadrant
            pt1 = [ (i/3.0)*r, ((3-i)/3.0)*r ]
            pt2 = [ ((i+1)/3.0)*r, ((3-(i+1))/3.0)*r ]

            pt1 = [pt1[0]*-1, pt1[1]*-1]
            pt2 = [pt2[0]*-1, pt2[1]*-1]
        elif self.quadrant == 4:
            # Copied from 2nd quadrant
            pt1 = [ ((3-i)/3.0)*r, (i/3.0)*r ]
            pt2 = [ ((3-(i+1))/3.0)*r, ((i+1)/3.0)*r ]

            pt1[0] = -1*pt1[0]
            pt2[0] = -1*pt2[0]
        else:
            raise Exception("Unrecognized quadrant")

        # line 1
        if (pt1[0]-self.center[0]) == 0:
            line1 = Line(x=pt1[0])
        else:
            m1 = (pt1[1]-self.center[1])/(pt1[0]-self.center[0])
            b1 = self.center[1] - (m1*self.center[0])
            line1 = Line(m=m1, b=b1)

        # line 2
        if (pt2[0]-self.center[0]) == 0:
            line2 = Line(x=pt2[0])
        else:
            m2 = (pt2[1]-self.center[1])/(pt2[0]-self.center[0])
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
        count = 0
        for coord in coords:
            if self.get_quadrant(coord) == self.quadrant:
                xs = [self.line1.get_x(coord[1]), self.line2.get_x(coord[1])]
                x1, x2 = min(xs), max(xs)

                #TODO: what about if exactly on the line
                if x1 < coord[0] <= x2:
                    count += 1
        return count










