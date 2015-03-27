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
        self.root, self.stroke_groups = self._parse_inkml(fd.read())

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
            stroke_partition.append(StrokeGroup(grp))
        return (root, stroke_partition)


class StrokeGroup(object):
    def __init__(self, strokes):
        self.strokes = strokes

    def preprocess(self):
        # wh_ratio: width/height ratio
        for stroke in self.strokes:
            stroke.clean()

    def __unicode__(self):
        return self.strokes

    def render(self):
        out = "\n".join([strk.render() for strk in self.strokes])
        return out

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

    def render(self):
        pairs = self.coords.swapaxes(0, 1).tolist()
        out = ''
        for pair in pairs:
            out += " ".join(map(lambda x: str(x), pair)) + ", "
        out = out.strip().rstrip(",")
        return '<trace id="%s">%s</trace>' % (self.id, out)

    def clean(self):
        """Does duplicate filtering & stroke smoothing"""

        if len(self.coords) < 1:
            return

        # remove duplicates
        pairs = self.coords.swapaxes(0, 1).tolist()
        last = pairs[0] 
        unique_coords = [last]
        for coord in pairs[1:]:
            if last == coord:
                continue
            unique_coords.append(coord)
            last = coord


class InkMLWriter(object):
    def __init__(self, annot, stroke_groups):
        self.annot = annot
        self.stroke_groups = stroke_groups

    def write(self):
        out = self.annot + "\n"
        out += "\n".join([strk_grp.render() for strk_grp in self.stroke_groups])
        out += "\n</ink>"
        return out


def _extract_annotations(content):
    annot = content.split("</annotationXML>")[0] + "</annotationXML>"
    return annot
        


if __name__ == '__main__':
    def main():
        import sys
        fname = sys.argv[1]
        inkml = inkML(fname)
        for strk_grp in inkml.stroke_groups:
            strk_grp.preprocess()

        annot = _extract_annotations(open(fname).read())
        writer = InkMLWriter(annot, inkml.stroke_groups)
        print(writer.write())


    main()
