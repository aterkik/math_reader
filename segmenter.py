from strokedata import Stroke, StrokeGroup


class Segmenter(object):
    def __init__(self):
        pass

    def train(self):
        pass

    def baseline_segmenter(self):
        partition = []
        if kind == 'baseline':
            for i, strk in enumerate(strokes):
                partition.append(StrokeGroup([strk], 'A_%s' % str(i), ' '))
        elif False:
            pass
        else:
            print("!!! Error: unrecognized segmenter '%s'" % kind)
        return partition

    def main_segmenter(self):
        pass


class SegmenterFeatures(object):
    pass

