import xml.etree.ElementTree as ET
import numpy as np


"""Scales an array (by column)"""
def scale(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

"""Rounds to nearest interval/step"""
def round_nearest(data,step=1):
    ndata = (data % step)
    for i,d in enumerate(ndata):
        if d > step/2:
            data[i] = data[i] + (step - d)
        if d < step/2:
            data[i] = data[i] - d
    return data

class inkML(object):
    def __init__(self, fname):
        try:
            fd = open(fname)
        except Exception as e:
            raise e

        # stroke_groups is a nested list of strokes. Strokes belonging to the
        # same symbol are grouped together.
        # E.g. in [[stroke1, stroke2], [stroke3] ...] stroke1 & stroke2 make
        # one symbol and stroke3 is makes a different symbol
        self.stroke_groups = self._parse_inkml(fd.read())

    @staticmethod
    def _parse_inkml(inkml_data):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!


        stroke_partition = []
        traces = root.findall(np + 'trace')
        # trace.text, trace.atrrib['id']
        tracegrps = root.findall('%straceGroup/%straceGroup' % (np, np))

        for trgrp in tracegrps:
            trids = map(lambda trv: trv.attrib['traceDataRef'], trgrp.findall(np + 'traceView'))
            trids = list(trids)

            grp = []
            # TODO: inefficent loop!
            for trace in traces:
                if trace.attrib['id'] in trids:
                    stroke = Stroke(trace.text.strip().replace(",","").split(' '), trace.attrib['id'])
                    grp.append(stroke)
            stroke_partition.append(grp)
        return stroke_partition


class Stroke(object):
    def __init__(self, coords, id):
        self.id = id
        xcol = np.array([np.array(coords[0::2]).astype(np.float)])
        ycol = np.array([np.array(coords[1::2]).astype(np.float)])
        self.coords = np.vstack([xcol,ycol])

    def __unicode__(self):
        return "<Stroke (id=%s)>" % self.id

    def __repr__(self):
        return self.__unicode__()


if __name__ == '__main__':
    def main():
        import sys
        inkml = inkML(sys.argv[1])
        print(inkml.stroke_groups)

    main()
