"""
    Authors: Andamlak Terkik, Kevin Carbone
"""
import xml.etree.ElementTree as ET
import numpy as np
from strokedata import Stroke, StrokeGroup
from segmenter import Segmenter
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
from collections import defaultdict
from settings import USE_REJECT_CLASS_PARSER

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

capitals = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def rand_str(k=4):
    return "".join(random.sample(capitals,k))

def sort_trace_grps(tracegrps):
    grp_ids = []
    for grp in tracegrps:
        key = grp.attrib.keys()[0].split('}')[0] + '}'
        grp_id = grp.attrib[key + 'id'].strip()
        if ':' in grp_id: # Mathbrush has ids like "4:5:6"
            grp_id = int(grp_id.split(':')[0])
        else:
            grp_id = int(grp_id)
        grp_ids.append((grp_id, grp))

    # Sort by id
    grp_ids = sorted(grp_ids, key=lambda x: x[0])
    tracegrps = list(map(lambda x: x[1], grp_ids))


class inkML(object):
    segmenter = Segmenter()
    rel_alphabets = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    rel_alphabets = rel_alphabets + ['\Delta','\\alpha','\\beta','\cos','\gamma',
                    '\infty', '\lambda', '\log', '\mu', '\phi', '\pi',
                    '\sigma', '\sin', '\sum', '\\tan', '\\theta']
    rel_ops = list('()+-.=[]') + ['\geq','\gt','\leq','\lt','\\neq', '\prime', ',', '\COMMA', '|', '\ldots', '!', '\pm', '-']
    rel_integrals = ['\int']
    rel_big_ops = ['\sigma', '\sqrt']

    def __init__(self, fname):
        self.fname = fname
        self.root = None
        self.src = None #TODO: this is redundant, see self.fname.
        self.stroke_groups = []
        self._strokes = None
        self.grp_by_id = {}
        self.relations = []
        self.pred_relations = []
        self.mst = {}
        self.mst_edges = {}

    def parse(self, from_ground_truth=False):
        try:
            fd = open(self.fname)
        except Exception as e:
            raise e

        # stroke_groups is a nested list of strokes. Strokes belonging to the
        # same symbol are grouped together.
        # E.g. in [[stroke1, stroke2], [stroke3] ...] stroke1 & stroke2 make
        # one symbol and stroke3 is makes a different symbol
        if from_ground_truth:
            self.root, self.stroke_groups = self._parse_inkml(fd.read(), self.fname)
        else:
            self.root, self.stroke_groups = self._parse_inkml_unsegmented(fd.read(), self.fname, segmenter_kind='main')

        # except Exception as e:
        #     print("!!! Error parsing inkml file '%s'" % self.fname)
        #     print("Details: %s" % e)
        #     import sys
        #     sys.exit()
        #     fd.close()

        self.stroke_groups = sorted(self.stroke_groups, key=lambda grp: grp.strokes[0].id)
        self.src = self.fname

        self.grp_by_id = {}
        self.grp_by_annot = {}
        for grp in self.stroke_groups:
            self.grp_by_id[grp.grp_id] = grp
            self.grp_by_annot[grp.annot_id] = grp


    def segment_preprocess(self):
        # for strk in self._strokes:
        #     import matplotlib.pyplot as plt
        #     plt.plot(strk.coords[:,0],strk.coords[:,1])
        #     plt.show()
        # self.plot_expression()

        # self.plot_expression()

        for strk in self._strokes:
            strk.clean()
        # self.plot_expression()
        self.expr_size_scaling2()

        coords_before = self._strokes[0].coords
        # self.resample4()
        # self.plot_expression()

        self.resample()
        self.resample4()
        self.center()
        # self.plot_expression()


        # self.plot_expression()

        # print "POST"
        # for strk in self._strokes:
        #     import matplotlib.pyplot as plt
        #     plt.plot(strk.coords[:,0],strk.coords[:,1])
        #     plt.show()

    # def uniform(self):



    def resample2(self,step=5):
        for i,stroke in enumerate(self._strokes):
            coords = stroke.coords
            new_coords = []

            for j,coord in enumerate(stroke.coords):
                if j%step == 0:
                    new_coords.append(coord)
            self._strokes[i].coords = np.array(new_coords)

    def resample4(self,k=64):
        for ctr,stroke in enumerate(self._strokes):
            k = 64
            coords = stroke.coords
            if len(coords) <= k:
                return
            else:
                r = len(coords) / k
                ncoords = []
                for i in range(0,len(coords),r):
                    ncoords.append(coords[i])
                if len(coords) > k:
                    k /= 2
                    # print k
                    r = len(coords) / k
                    ncoords = []
                    for i in range(0,len(coords),r):
                        ncoords.append(coords[i])
                    # print len(ncoords)
            # print len(ncoords)
            self._strokes[ctr].coords = np.array(ncoords)

    def center(self):
        l1,l2 = self.expression_coords()[:,0].min(),self.expression_coords()[:,1].min()
        for ctr,stroke in enumerate(self._strokes):
            self._strokes[ctr].coords[:,0] -= l1
            self._strokes[ctr].coords[:,1] -= l2

    def resample(self,alpha=15):
        # all_coords = []
        # other_coords = []
        for ctr,stroke in enumerate(self._strokes):
            coords = np.copy(stroke.coords)
            xmin = coords[:,0].min()
            xmax = coords[:,0].max()
            ymin = coords[:,1].min()
            ymax = coords[:,1].max()
            wh_ratio = (xmax-xmin)/(ymax-ymin) if (ymax-ymin) != 0 else xmax-xmin
            stroke.scale_size(xmin, ymin, xmax-xmin,
                    ymax-ymin, wh_ratio, yhigh=200)
            sc_coords = stroke.coords.T

            L = [0]
            for i in range(1,len(coords)):
                L.append(L[-1] + np.linalg.norm(sc_coords[i] - sc_coords[i-1]))
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
            # print "PLOT"
            # import matplotlib.pyplot as plt
            # plt.scatter(np.array(ncoords)[:,0],np.array(ncoords)[:,1])
            # plt.show()
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


    # def old_resample(self,alpha=2):
    #     # all_coords = []
    #     # other_coords = []
    #     for ctr,stroke in enumerate(self._strokes):
    #         coords = np.copy(stroke.coords)
    #         xmin = coords[:,0].min()
    #         xmax = coords[:,0].max()
    #         ymin = coords[:,1].min()
    #         ymax = coords[:,1].max()
    #         wh_ratio = (xmax-xmin)/(ymax-ymin) if (ymax-ymin) != 0 else xmax-xmin
    #         stroke.scale_size(xmin, ymin, xmax-xmin,
    #                 ymax-ymin, wh_ratio, yhigh=100)
    #         sc_coords = stroke.coords.T
    #         import matplotlib.pyplot as plt
    #         plt.scatter(sc_coords[:,0],sc_coords[:,1])
    #         plt.show()
    #         # print len(coords)
    #         # print coords,sc_coords
    #         # print len(coords)
    #         # if len(coords) > alpha:

    #         # L = [0]
    #         # for i in range(1,len(coords)):
    #         #     L.append(L[-1] + np.linalg.norm(coords[i] - coords[i-1]))
    #         # alpha = L[-1] * 0.05
    #         # m = int(np.floor(L[-1]/alpha))
    #         n = len(coords) - 1
    #         p1 = coords[0]
    #         ncoords = [p1]
    #         j = 1
    #         for i in range(n):
    #             dist = np.linalg.norm(sc_coords[j] - sc_coords[i])
    #             if dist >= alpha:
    #                 ncoords.append(coords[j])
    #                 i = j
    #                 count += 1
    #             j += 1
    #             # while L[j] < p * alpha:
    #             #     j += 1
    #             # C = (p*alpha-L[j-1])/(L[j] - L[j-1])
    #             # nx = (coords[j-1][0] + (coords[j][0] - coords[j-1][0]) * C) 
    #             # ny = (coords[j-1][1] + (coords[j][1] - coords[j-1][1]) * C)
    #             # ncoords.append([nx,ny])
    #         # ncoords.append([coords[-1][0],coords[-1][1]])
    #         import matplotlib.pyplot as plt
    #         plt.scatter(np.array(ncoords)[:,0],np.array(ncoords)[:,1])
    #         plt.show()
    #         self._strokes[ctr].coords = np.array(ncoords)
            
    #         # all_coords.extend(ncoords)
    #         # other_coords.extend(coords)
    #     # cs = np.array(all_coords)
    #     # cs2 = np.array(other_coords)
    #     # plt.scatter(cs[:,0],cs[:,1])
    #     # plt.figure()
    #     # plt.scatter(cs2[:,0],cs2[:,1])
    #     # plt.show()
    #     # import time
    #     # time.sleep(5)

    def expr_size_scaling2(self):
        xh,yh = (self._xmax_expr - self._xmin_expr),(self._ymax_expr - self._ymin_expr)
        xmin = self._xmin_expr
        ymin = self._ymin_expr
        for ctr,strk in enumerate(self._strokes):
            # print strk.coords[:,1].max() - strk.coords[:,1].min()
            # xmin = strk.coords[:,0].min()
            # ymin = strk.coords[:,1].min()
            # xh,yh = strk.coords[:,0].max() - strk.coords[:,0].min(),strk.coords[:,1].max() - strk.coords[:,1].min() 
            new_coords = []
            for coord in strk.coords:

                ncoordx = ((coord[0] - xmin) / xh) 
                ncoordy = ((coord[1] - ymin) / xh) 
                new_coords.append([ncoordx,ncoordy])


            self._strokes[ctr].coords = np.array(new_coords)

    def plot_expression(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.expression_coords()[:,0],self.expression_coords()[:,1])
        plt.show()

    def expression_coords(self):
        all_coords = []
        for ctr,strk in enumerate(self._strokes):
            all_coords.extend(self._strokes[ctr].coords)
        return np.array(all_coords)


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
            if ',' in grp.annot_id:
                grp.annot_id = grp.annot_id.replace(",", "COMMA")
            #pr = '\,' if grp.prediction == ',' else grp.prediction
            outs.append("O, %s, %s, 1.0, %s" % (
                        grp.annot_id, grp.prediction, grp.strk_ids()))
        out = "\n".join(outs)

        rels = []

        for edge in self.mst_edges:
            # Ensure right-to-left relation. Needed because of networkx's lack of
            # MST support for directed graphs
            # EXCEPT: Above relationships for '-'s

            #TODO:
            # if edge[0].target == '-' or edge[0].prediction == '-':
            #     print("Prevented swapping edge nodes for '-' symbol...")
            # else:
            edge = sorted(edge, key=lambda x: self.stroke_groups.index(x))

            grp1, grp2 = edge[0], edge[1]
            try:
                rel = self.mst[grp1][grp2]['rel']
                if rel.startswith("A") and '-' in (grp2.prediction, grp2.target):
                    grp1, grp2 = grp2, grp1
            except Exception as e:
                print e

            grp1.annot_id = grp1.annot_id.replace(",", "COMMA")
            grp2.annot_id = grp2.annot_id.replace(",", "COMMA")

            rels.append("R, %s, %s, %s, 1.0" % (
                            grp1.annot_id, grp2.annot_id, rel))

        rels = "\n".join(rels)
        inbetween = "\n\n# [ RELATIONSHIPS ]\n"
        return out + inbetween + rels

    
    def _parse_inkml_unsegmented(self, inkml_data, fname, segmenter_kind='baseline'):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!

        traces = root.findall(np + 'trace')
        strokes = []
        for trace in traces:
            stroke = Stroke(trace.text.strip(), trace.attrib['id'])
            strokes.append(stroke)

        if segmenter_kind == 'baseline':
            partition = inkML.segmenter.baseline_segmenter(self._strokes)
        else:
            partition = inkML.segmenter.main_segmenter(self._strokes)

        return (root, partition)

    def read_strokes(self):
        """Reads list of strokes only. No metadata/ground truth.
        Needed for preprocessing"""
        inkml_data = open(self.fname).read()
        try:
            root = ET.fromstring(inkml_data)
        except:
            print("Warning: file skipped")
            raise e

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

    def grp_from_id(self, id1):
        if ',' in id1:
            id1 = id1.replace(",", "COMMA")

        id_keys = self.grp_by_id.keys()
        annot_keys = self.grp_by_annot.keys()
        grp1 = None
        if id1 in id_keys:
            grp1 = self.grp_by_id[id1]
        elif id1 in annot_keys:
            grp1 = self.grp_by_annot[id1]
        else:
            raise Exception("couldn't find id/annot")
        return grp1

    def set_pred_relations(self, candids, relations):
        #NOTE: digraph doesn't play well with MST algorithms in networkx
        self.G = nx.MultiGraph()
        pred_rels = []
        for candid, subrels in zip(candids, relations):
            dict_subrels = dict(subrels)
            if 'X' in dict_subrels.keys():
                idx = self.stroke_groups.index(candid[0])
                idx2 = self.stroke_groups.index(candid[1])
                if idx2-idx > 1:
                    print("X probability: " + str(dict_subrels['X']) + ", Idx dist: " + str(idx2-idx))
                if dict_subrels['X'] > 1:
                    continue

            for rel, w in subrels:
                if rel.startswith("A"):
                    if w < 0.4:
                        # Heuristic, but increases accuracy by a lot in practise
                        continue
                if rel != 'X':
                    # Don't forget to do 1-w (since we're looking for minimum)
                    self.G.add_edge(candid[0], candid[1], weight=(1-w), rel=rel.tolist())

        T = nx.minimum_spanning_tree(self.G)
        self.mst_edges = set(T.edges())  # optimization
        self.mst = T

    def get_relation_candids(self):
        rels = []
        # TODO:XXX: k_next SIGNIGICANTLY affects recogntion rates
        k_next = 1
        for i, grp in enumerate(self.stroke_groups):
            candids = self.stroke_groups[i+1:i+k_next+1]
            rels.extend(list(zip([grp]*k_next, candids, [' ']*k_next)))

            if '-' in (grp.target, grp.prediction):
                # Heuristics, heuristcs everywhere (and they work realllly good)
                if i+2<len(self.stroke_groups):
                    rels.append((grp, self.stroke_groups[i+2], ' '))
                if i-2>=0:
                    rels.append((self.stroke_groups[i-2], grp, ' '))

        # Special candidates for 'ABOVE' rule
        for i, grp in enumerate(self.stroke_groups):
            if grp.target == '-' or grp.prediction == '-': # TODO: ensure there's no diff b/n '-' and the larger '-' one
                limit = max(0, i-1)
                candids = self.stroke_groups[limit:i]
                total = len(candids)
                if total > 0:
                    rels.extend(list(zip([grp]*total, candids, [' ']*total)))

        return rels

    def get_relations_for_train(self, force_read=False):
        if len(self.relations) > 0 and not force_read:
            return self.relations

        name_parts = self.fname.replace("inkml", "lg").split(".")
        new_name = name_parts[0] + rand_str() + "." + name_parts[1]
        try:
            os.system('crohme2lg "%s" "%s"' % (self.fname, new_name))

            lg_content = open(new_name).read()
            rel_lines = list(filter(lambda x: x.strip().startswith("R,"),
                                lg_content.splitlines()))
            rels = []
            outs = []
            rel_dicts = defaultdict(list)
            for line in rel_lines:
                # e.g. "R, 1, 2, Sup, 1.0"
                _, id1, id2, rel, _ = list(map(lambda x: x.strip(), line.split(",")))
                try:
                    grp1, grp2 = self.grp_from_id(id1), self.grp_from_id(id2)
                    rels.append((grp1, grp2, rel))
                    rel_dicts[grp1].append(grp2)
                    outs.append(str(grp1) + " " + str(grp2) + " " + rel)
                except Exception as e:
                    import pdb; pdb.set_trace()
                    pass

            if USE_REJECT_CLASS_PARSER:
                strks = self.stroke_groups
                for grp in rel_dicts.keys():
                    negative = rel_dicts[grp]
                    idx = self.stroke_groups.index(grp)
                    diff_grps = set(self.stroke_groups[idx+1:]) - set(negative)
                    for rel_grp in diff_grps:
                        # X is basically our 'reject' relationship
                        rels.append((grp, rel_grp, 'X'))
                        outs.append(str(grp) + " " + str(rel_grp) + " X")

            # print("==============================")
            # print(self.fname, new_name)
            # print("\n".join(rel_lines))
            # print("\n".join(outs))

            # print("==============================\n\n")

            # Save for later
            self.relations = rels
        finally:
            os.unlink(new_name)
        return self.relations


    @staticmethod
    def _parse_inkml(inkml_data, fname):
        root = ET.fromstring(inkml_data)
        np = root.tag.rstrip('ink') # get namespace, bad hack!

        stroke_partition = []
        traces = root.findall(np + 'trace')

        tracegrps = root.findall('%straceGroup/%straceGroup' % (np, np))

        # Sort tracegroups based on id. Needed for relationship processing
        sort_trace_grps(tracegrps)

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

            annot_id = annot_id.replace(",", "COMMA")

            grp = []
            # TODO: inefficent loop!
            for trace in traces:
                if trace.attrib['id'] in trids:
                    stroke = Stroke(trace.text.strip(), trace.attrib['id'])
                    grp.append(stroke)

            key = trgrp.attrib.keys()[0].split('}')[0] + '}'
            grp_id = trgrp.attrib[key + 'id'].strip()
            stroke_partition.append(StrokeGroup(grp, annot_id, ground_truth, grp_id))
            # stroke_partition = sorted(stroke_partition, key=lambda obj: obj)

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
