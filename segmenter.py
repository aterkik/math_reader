from strokedata import Stroke, StrokeGroup
from settings import PARAMS_DIR, BONUS_PARAMS_DIR
from sklearn.externals import joblib
import sys
import numpy as np

class Segmenter(object):
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

    def main_segmenter(self):
        if not self.svm:
            self._load_params()
        pass


class SegmenterFeatures(object):
    @staticmethod
    def get_features(strk_pair, strk_grps):
        """Compute all segmentation features for stroke pair (strk_pair)
        strk_pair: stroke pair
        strk_grps: the stroke group for the whole expression (including strk_pair)
        """
        bb = BoundingBox(strk_pair[0])
        return [0, 0]
        # return [int(strk_pair[0]), int(strk_pair[1])]


class BoundingBox(object):

    def __init__(self, stroke):
        mins = np.amin(stroke.coords.T,axis=0)
        maxs = np.amax(stroke.coords.T,axis=0)
        self.minx = mins[0]
        self.miny = mins[1]
        self.width = abs(maxs[0] - mins[0])
        self.height = abs(maxs[1] - mins[0])
        self.maxx = maxs[0]
        self.maxy = maxs[1]
        self.center = np.array([self.minx + (self.width/2), self.miny + (self.height/2)])
    

    def distance(self, other):
        return np.linalg.norm(self.center-other.center)
    
