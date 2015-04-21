from strokedata import Stroke, StrokeGroup
from settings import PARAMS_DIR, BONUS_PARAMS_DIR
from sklearn.externals import joblib
import sys

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
    pass
