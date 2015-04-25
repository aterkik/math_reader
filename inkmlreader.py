"""
    Authors: Andamlak Terkik, Kevin Carbone
"""
import xml.etree.ElementTree as ET
import numpy as np
from strokedata import Stroke, StrokeGroup
from segmenter import Segmenter

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
    segmenter = Segmenter()
    def __init__(self, fname):
        self.fname = fname
        self.root = None
        self.src = None #TODO: this is redundant, see self.fname.
        self.stroke_groups = None
    
    def parse(self, from_ground_truth=False):
        try:
            fd = open(self.fname)
        except Exception as e:
            raise e

        # stroke_groups is a nested list of strokes. Strokes belonging to the
        # same symbol are grouped together.
        # E.g. in [[stroke1, stroke2], [stroke3] ...] stroke1 & stroke2 make
        # one symbol and stroke3 is makes a different symbol
        try:
            if from_ground_truth:
                self.root, self.stroke_groups = self._parse_inkml(fd.read(), self.fname)
            else:
                self.root, self.stroke_groups = self._parse_inkml_unsegmented(fd.read(), self.fname, segmenter_kind='main')

        except Exception as e:
            #import pdb; pdb.set_trace()
            print("!!! Error parsing inkml file '%s'" % self.fname)
            print("Details: %s" % e)
            raise e
        finally:
            fd.close()

        self.stroke_groups = sorted(self.stroke_groups, key=lambda grp: grp.strokes[0].id)
        self.src = self.fname

    def preprocess(self):
        for strk_grp in self.stroke_groups:
            strk_grp.preprocess()

    def has_symbol(self, symbol):
        for grp in self.stroke_groups:
            if grp.target == symbol:
                return True
        return False

    def symbol_count(self, symbol):
        return np.sum([grp.target == symbol for grp in self.stroke_groups])

    def get_lg(self):
        outs = []
        for grp in self.stroke_groups:
            if ',' in grp.prediction:
                grp.prediction = "COMMA"
            #pr = '\,' if grp.prediction == ',' else grp.prediction
            outs.append("O, %s, %s, 1.0, %s" % (
                        grp.annot_id, grp.prediction, grp.strk_ids()))
        out = "\n".join(outs)
        # TODO: relationships
        return out

    @staticmethod
    def _parse_inkml_unsegmented(inkml_data, fname, segmenter_kind='baseline'):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!

        traces = root.findall(np + 'trace')
        strokes = []
        for trace in traces:
            stroke = Stroke(trace.text.strip(), trace.attrib['id'])
            strokes.append(stroke)

        if segmenter_kind == 'baseline':
            partition = inkML.segmenter.baseline_segmenter(strokes)
        else:
            partition = inkML.segmenter.main_segmenter(strokes)

        return (root, partition)

    @staticmethod
    def _parse_inkml(inkml_data, fname):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!

        stroke_partition = []
        traces = root.findall(np + 'trace')
        tracegrps = root.findall('%straceGroup/%straceGroup' % (np, np))

        for i, trgrp in enumerate(tracegrps):
            trids = map(lambda trv: trv.attrib['traceDataRef'], trgrp.findall(np + 'traceView'))
            trids = list(trids)

            try:
                ground_truth = trgrp.find(np + 'annotation').text
            except:
                # User supplied inkmls may not contain target
                ground_truth = ' '

            try:
                annot_id = trgrp.find(np + 'annotationXML').attrib['href']
            except:
                print("!! Warning: couldn't find annotationXML for tracegroup"
                      " in file %s" % fname)
                annot_id = "u_" + str(i)

            grp = []
            # TODO: inefficent loop!
            for trace in traces:
                if trace.attrib['id'] in trids:
                    stroke = Stroke(trace.text.strip(), trace.attrib['id'])
                    grp.append(stroke)
            stroke_partition.append(StrokeGroup(grp, annot_id, ground_truth))
        return (root, stroke_partition)




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


if __name__ == '__main__':
    def main():
        import sys
        fname = sys.argv[1]

        inkml = inkML(fname)
        inkml.preprocess()

        writer = InkMLWriter(fname, inkml.stroke_groups)
        #print(writer.write())
        for grp in inkml.stroke_groups:
            print(grp.get_features())

    main()
