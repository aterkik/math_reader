from strokedata import Stroke, StrokeGroup

def baseline_segmenter(strokes, kind='baseline'):
    partition = []
    if kind == 'baseline':
        for i, strk in enumerate(strokes):
            partition.append(StrokeGroup([strk], 'A_%s' % str(i), ' '))
    elif False:
        pass
    else:
        print("!!! Error: unknown segmneter '%s'" % kind)
    return partition

segment = baseline_segmenter
