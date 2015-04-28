"""
    Authors: Andamlak Terkik, Kevin Carbone
"""
import xml.etree.ElementTree as ET
import numpy as np
from strokedata import Stroke, StrokeGroup
from segmenter import Segmenter
import matplotlib.pyplot as plt

"""
    Usage:
    >>> inkml = inkML('test.inkml')
    >>> inkml.stroke_groups
    [<inkmlreader.StrokeGroup object at 0x10d6d6dd8>, ...]
    >>> inkml.symbol_preprocess() # does duplicate removal, size normaliztion ...
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
        self._strokes = None
    
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

    def segment_preprocess(self):
        for strk in self._strokes:
            strk.clean()
        coords_before = self._strokes[0].coords
        self.resample()
        self.expr_size_scaling2()

        #coords_before = self._strokes[0].coords


    def resample(self,alpha=13):
        # all_coords = []
        # other_coords = []
        for ctr,stroke in enumerate(self._strokes):
            coords = stroke.coords
            L = [0]
            for i in range(1,len(coords)):
                L.append(L[i-1] + np.linalg.norm(coords[i] - coords[i-1]))
            m = int(np.floor(L[-1]/alpha))
            n = len(coords) - 1
            p1 = coords[0]
            ncoords = []
            j = 1
            for p in range(1,m-1):
                while L[j] < p * alpha:
                    j += 1
                C = (p*alpha-L[j-1])/(L[j] - L[j-1])
                nx = (coords[j-1][0] + (coords[j][0] - coords[j-1][0]) * C) 
                ny = (coords[j-1][1] + (coords[j][1] - coords[j-1][1]) * C)
                ncoords.append([nx,ny])
            ncoords.append([coords[-1][0],coords[-1][1]])
            self._strokes[ctr].coords = np.array(ncoords)
            
            # all_coords.extend(ncoords)
            # other_coords.extend(coords)
        # cs = np.array(all_coords)
        # cs2 = np.array(other_coords)
        # plt.scatter(cs[:,0],cs[:,1])
        # plt.figure()
        # plt.scatter(cs2[:,0],cs2[:,1])
        # plt.show()
        # import time
        # time.sleep(5)




    def expr_size_scaling2(self):
        xh,yh = (self._xmax_expr - self._xmin_expr),(self._ymax_expr - self._ymin_expr)
        for ctr,strk in enumerate(self._strokes):
            new_coords = []
            for coord in strk.coords:
                ncoordx = ((coord[0] - self._xmin_expr) / max(xh,yh)) * 200
                ncoordy = ((coord[1] - self._ymin_expr) / max(xh,yh)) * 200
                new_coords.append([ncoordx,ncoordy])


            self._strokes[ctr].coords = np.array(new_coords)





    def expr_size_scaling(self):
        ymin, ymax = 0, 200

        xmin_expr, ymin_expr = self._xmin_expr, self._ymin_expr
        xmax_expr, ymax_expr = self._xmax_expr, self._ymax_expr

        wh_ratio = (xmax_expr-xmin_expr)/float(ymax_expr-ymin_expr)
        xmin, xmax = 0, ymax * wh_ratio
        
        y_ratio = (ymax-ymin)/(ymax_expr-ymin_expr)
        x_ratio = (xmax-xmin)/(xmax_expr-xmin_expr)

        for strk in self._strokes:
            coords = strk.coords
            xmin_stroke, xmax_stroke = coords[:,0].min(), coords[:,0].max()
            ymin_stroke, ymax_stroke = coords[:,1].min(), coords[:,1].max()

            new_coords = []
            for coord in coords:
                x, y = coord
                y_offset = (ymin_stroke-ymin_expr) * y_ratio
                new_y = y_offset + ((y-ymin_expr) * y_ratio)

                x_offset = (xmin_stroke-xmin_expr) * x_ratio
                new_x = x_offset + ((x-xmin_expr) * x_ratio)

                new_coords.append([new_x, new_y])
            strk.coords = np.array(new_coords)

    def symbol_preprocess(self):
        for strk in self._strokes:
            strk.clean()
        self.symbol_size_scaling()

    def symbol_size_scaling(self):
        for strk_grp in self.stroke_groups:
            strk_grp.size_scale()

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

    def read_strokes(self):
        """Reads list of strokes only. No metadata/ground truth.
        Needed for preprocessing"""
        inkml_data = open(self.fname).read()
        root = ET.fromstring(inkml_data)
        namespace = root.tag.rstrip('ink') # get namespace, bad hack!

        traces = root.findall(namespace + 'trace')
        strokes = []
        for trace in traces:
            strokes.append(Stroke(trace.text.strip(), trace.attrib['id']))
        self._strokes = strokes

        [self._xmin_expr, self._ymin_expr] = np.min([strk.coords.min(axis=0)
                               for strk in self._strokes], axis=0)
        [self._xmax_expr, self._ymax_expr] = np.max([strk.coords.max(axis=0)
                            for strk in self._strokes], axis=0)




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
        inkml.symbol_preprocess()

        writer = InkMLWriter(fname, inkml.stroke_groups)
        #print(writer.write())
        for grp in inkml.stroke_groups:
            print(grp.get_features())

    main()
