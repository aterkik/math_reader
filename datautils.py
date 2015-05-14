"""
    Authors: Kevin Carbone, Andamlak Terkik
"""
from inkmlreader import inkML
import numpy as np
import os
from collections import defaultdict
from settings import *
import operator
import random
from segmenter import SegmenterFeatures

######## Utility functions ################

def load_dataset(src=None):
    # How it works:
    # Reads list of inkml files
    # Parses inkml files and target outputs
    # Split expressions into list of symbols
    # Calculate features for each symbol and append target output

    try:
        if not src:
            inkml_file_names = open(TRAIN_PATH + TRAIN_FNAME_SRC).read().splitlines()
            print("Loaded training/test split from folder '%s'..." % TRAIN_PATH)
        else:
            inkml_file_names = filter(lambda f: 'inkml' in f, os.listdir(src))
            print("Loaded training/test split from folder '%s'..." % src)

    except Exception as e:
        print("!!! Error opening training data. Please modify TRAIN_SRC_PATH or invalid training directory supplied.")
        import sys
        sys.exit(1)

    return get_inkml_objects(inkml_file_names, prefix=src)

def get_inkml_objects(inkml_file_names, prefix=TRAIN_PATH):
    inkmls = []
    skipped = 0
    for i, fname in enumerate(inkml_file_names):
        # MfrDB files have three coordinates. Skip for now
        #if 'MfrDB' in fname:
        #    continue

        try:
            inkml = inkML(prefix + fname)
        except:
            # skip malformed inkmls
            skipped += 1
            continue
        inkmls.append(inkml)

        if i >= MAX_FILES and MAX_FILES >= 0:
            break
    print("Total skipped files: %d" % skipped)
    return inkmls


def inkmls_to_symbol_feature_matrix(inkmls):
    symbols = []
    for inkml in inkmls:
        symbols.extend(inkml.stroke_groups)

    if VERBOSE:
        print("Loaded %s symbols (%s files)...\n" % (len(symbols), len(inkmls)))

    data = np.array([])
    total = len(symbols)
    for i, symbol in enumerate(symbols):
        features = symbol.get_features()

        # features = [1,2]
        if data.shape[0] == 0:

            data = np.array(features + [symbol.target])
        else:
            data = np.vstack((data, features + [symbol.target]))

        if i % 250 == 0:
            print("....%.2f%% complete (generating features)" % (100 * float(i)/total))

    return data, symbols

def inkmls_to_segmentation_feature_matrix(inkmls):
    Xs = np.array([])
    Ys = []
    total = len(inkmls)
    for i, inkml in enumerate(inkmls):
        if len(inkml.stroke_groups) <= 0:
            continue
        features, new_ys = _segment_features(inkml.stroke_groups)

        if Xs.shape[0] == 0:
            Xs = np.array(features)
        else:
            # Expressions with single stroke only have no segmentation features
            # because there is no pairing
            if features.size > 0:
                Xs = np.vstack((Xs, features))
        Ys.extend(new_ys)

        if i % 5 == 0:
            print("....%.2f%% complete (generating segmentation features)" % (100 * float(i)/total))


    return (Xs, Ys)

def _segment_features(stroke_groups):
    Xs = np.array([])
    Ys = []
    strokes = [strk for grp in stroke_groups for strk in grp.strokes]
    stroke_pairs = zip(strokes, strokes[1::])
    for pair in stroke_pairs:
        decision = 'split'
        for grp in stroke_groups:
            if pair[0] in grp.strokes and pair[1] in grp.strokes:
                decision = 'merge'
                break

        features = SegmenterFeatures.get_features(pair, strokes)
        if Xs.shape[0] == 0:
            Xs = np.array(features)
        else:
            Xs = np.vstack((Xs, features))
        Ys.append(decision)

    return Xs, Ys


def split_dataset(inkmls, test_percentage):
    """Splits (randomly!) dataset into training and test.
       Works by calculating target frequency counts for each symbol
       for both folds. Then, starting from the symbols with the
       lowest frequency, it partitions the files containing each symbol
       into the two folds. Because the algorithm works greedily,
       which fold gets priority for each symbol affects the distribution quality
       very much. To minimize this effect, the fold priority is alternated at each
       iteration.
    """
    class_counts = defaultdict(int)
    bad_inkmls = []
    for inkml in inkmls:
        try:
            for symbol in inkml.stroke_groups:
                class_counts[symbol.target] += 1
        except:
            bad_inkmls.append(inkml)

    inkmls = [inkml for inkml in inkmls if inkml not in bad_inkmls]
    print class_counts
    trainf_target = {}
    for sym, freq in class_counts.items():
        trainf_target[sym] = int((1 - test_percentage)*freq)

    testf_target = {}
    for sym, freq in class_counts.items():
        testf_target[sym] = int(test_percentage*freq)

    sorted_targets = sorted(trainf_target.items(), key=operator.itemgetter(1))


    test_fold, train_fold = [], []
    cinkmls = len(inkmls)
    i = 0
    for symbol, count in sorted_targets:
        matches = get_files_with_symbol(symbol, inkmls)

        # if i % 2 == 0:
        train_portion, test_portion = random_select_by_count(matches, symbol, count)
        # else:
        #     test_portion, train_portion = random_select_by_count(matches, symbol, testf_target[symbol])

        test_fold.extend(test_portion)
        train_fold.extend(train_portion)

        for inkml in matches:
            inkmls.remove(inkml)
        i += 1

    # make the remaining files training
    train_fold.extend(inkmls)

    test_achieved = defaultdict(int)
    for inkml in test_fold:
        for grp in inkml.stroke_groups:
            test_achieved[grp.target] += 1

    train_achieved = defaultdict(int)
    for inkml in train_fold:
        for grp in inkml.stroke_groups:
            train_achieved[grp.target] += 1

    return train_fold, test_fold


def get_files_with_symbol(symbol, inkmls):
    """From a list of InkML objects (inkmls), returns all objects
    containing the symbol (symbol) """
    matches = []
    for i, inkml in enumerate(inkmls):
        if inkml.has_symbol(symbol):
            matches.append(inkml)
    return matches

def random_select_by_count(inkmls, symbol, count):
    """Splits inkmls into two partitions into
    approximately count and remaining files"""
    # random.shuffle(inkmls)

    sorted_inkmls = sorted(inkmls,
                    key=lambda inkml: inkml.symbol_count(symbol))
    partition_count = 0
    idx = 0
    for inkml in sorted_inkmls:
        if partition_count >= count:
            break
        partition_count += inkml.symbol_count(symbol)
        idx += 1

    # shuffle items
    random.shuffle(inkmls)


    return inkmls[:idx], inkmls[idx:]

def segment_inkmls(inkmls):
    #XXX: preprocess BEFORE parse
    total = len(inkmls)

    print("Using main segmenter")
    for i, inkml in enumerate(inkmls):
        try:
            inkml.read_strokes()
            inkml.segment_preprocess()
            inkml.parse(from_ground_truth=False)
            inkml.symbol_preprocess()
        except:
            inkmls.remove(inkml)
            continue

        if i % 5 == 0:
            print("....%.2f%% complete (segmenting)" % (100 * float(i)/total))
    print("Total merges: %d" % inkmls[0].segmenter.total_merges)

def segment_inkmls_ground_truth(inkmls):
    #XXX: preprocess BEFORE parse
    for inkml in inkmls:
        try:
            inkml.read_strokes()
            inkml.segment_preprocess()
            inkml.parse(from_ground_truth=True)
            inkml.symbol_preprocess()
        except:
            inkmls.remove(inkml)
            continue


########## End utility functions ##############
