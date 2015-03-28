import xml.etree.ElementTree as ET
import numpy as np

"""
    Usage:
    >>> inkml = inkML('test.inkml')
    >>> inkml.stroke_groups
    [<inkmlreader.StrokeGroup object at 0x10d6d6dd8>, ...]
    >>> inkml.preprocess() # does duplicate removal, size normaliztion ...
    >>> writer = InkMLWriter('test.inkml', inkml.stroke_groups)
    >>> writer.write() # inkml output of *normalized* traces
    <ink>
     <annotation>Arithmetic</annotation>
     ...
    </ink>
    >>>
"""

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
        self.root, self.stroke_groups = self._parse_inkml(fd.read())

    def preprocess(self):
        for strk_grp in self.stroke_groups:
            strk_grp.preprocess()

    @staticmethod
    def _parse_inkml(inkml_data):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!

        stroke_partition = []
        traces = root.findall(np + 'trace')
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
            stroke_partition.append(StrokeGroup(grp))
        return (root, stroke_partition)


class StrokeGroup(object):
    def __init__(self, strokes):
        self.strokes = strokes

    def preprocess(self):
        for stroke in self.strokes:
            stroke.clean()

        # It's best to do size normaliztion after cleaning
        [xmin, ymin] = np.min([stroke.coords.min(axis=0)
                               for stroke in self.strokes], axis=0)
        [xmax, ymax] = np.max([stroke.coords.max(axis=0)
                            for stroke in self.strokes], axis=0)

        # Do size normalization
        wh_ratio = (xmax - xmin) / (ymax - ymin)
        for strk in self.strokes:
            strk.scale_size(xmin, ymin, xmax-xmin, ymax-ymin, wh_ratio, yhigh=100)
        self.get_features()

    def get_features(self):
        print(self.strokes[0]._norm_line_length())
        print(self.strokes[0]._angle_feature())



    def __unicode__(self):
        return self.strokes

    def render(self, offset=0):
        """
            offset: Offset value to add on each X trace coordinate. Useful
                    for visualizations so that symbols appear left to right
                    instead of on top of each other.
        """
        out = "\n".join([strk.render(offset=offset) for strk in self.strokes])
        return out

class Stroke(object):
    def __init__(self, coords, id):
        self.id = id
        xcol = np.array([np.array(coords[0::2]).astype(np.float)])
        ycol = np.array([np.array(coords[1::2]).astype(np.float)])
        self.raw_coords = coords
        self.coords = np.vstack([xcol,ycol]).T

    def _norm_line_length(self):
        dists = []
        p1 = self.coords.T[0]
        total = 0
        for p2 in self.coords.T[1:]:
            dist = np.linalg.norm(p2-p1)
            dists.append(dist)
            total += dist
            p1 = p2
        return np.array(dists)

    def _angle_feature(self):
        angles = []
        p1 = self.coords.T[0]
        for p2 in self.coords.T[1:]:
            a = angle(p2,p1)
            angles.append(a)
        return np.array(angles)


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
        yp_col = (ycol - ymin) * (float(yhigh) / yrng)
        xp_col = (xcol - xmin) * (float(wh_ratio * yhigh) / xrng)
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


class InkMLWriter(object):
    def __init__(self, fname, stroke_groups):
        self.annot = self._extract_annotations(open(fname).read())
        self.stroke_groups = stroke_groups

    def write(self):
        out = self.annot + "\n"
        out += "\n".join([strk_grp.render(offset=i*200)
                for i, strk_grp in enumerate(self.stroke_groups)])
        out += "\n</ink>"
        return out

    @staticmethod
    def _extract_annotations(content):
        annot = content.split("</annotationXML>")[0] + "</annotationXML>"
        return annot

def length(v):
    return np.sqrt(np.dot(v, v))

def angle(v1,v2):
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))

if __name__ == '__main__':
    def main():
        import sys
        fname = sys.argv[1]

        inkml = inkML(fname)
        inkml.preprocess()

        writer = InkMLWriter(fname, inkml.stroke_groups)
        print(writer.write())


    main()
