from inkmlreader import inkML
import numpy as np
from collections import defaultdict
import operator
import random

######## Utility functions ################
TRAIN_PATH = 'TrainINKML_v3/'
TRAIN_FNAME_SRC = 'AllEM_part4_TRAIN_all.txt'

VERBOSE = True
# For development, limit dataset to 300 files
MAX_FILES = 200

def load_dataset():
    # How it works:
    # Reads list of inkml files
    # Parses inkml files and target outputs
    # Split expressions into list of symbols
    # Calculate features for each symbol and append target output

    try:
        train_files = open(TRAIN_PATH + TRAIN_FNAME_SRC).read().splitlines()
    except:
        print("!!! Error opening training data. Please modify TRAIN_SRC_PATH.")
        import sys
        sys.exit(1)

    inkmls = []
    for i, fname in enumerate(train_files):
        # MfrDB files have three coordinates. Skip for now
        if 'MfrDB' in fname:
            continue

        inkml = inkML(TRAIN_PATH + fname)
        inkml.preprocess()
        inkmls.append(inkml)

        if i >= MAX_FILES:
            break
    return inkmls


def inkmls_to_feature_matrix(inkmls):
    symbols = []
    for inkml in inkmls:
        symbols.extend(inkml.stroke_groups)

    if VERBOSE:
        print("Loaded %s symbols..." % (len(symbols)))

    data = np.array([])
    for symbol in symbols:
        features = symbol.get_features()
        # features = [1,2]
        if data.shape[0] == 0:

            data = np.array(features + [symbol.target])
        else:
            data = np.vstack((data, features + [symbol.target]))
    return data


def split_dataset(inkmls, test_percentage):
    """Split dataset into training and test"""
    class_counts = defaultdict(int)
    for inkml in inkmls:
        for symbol in inkml.stroke_groups:
            class_counts[symbol.target] += 1

    testf_target = {}
    for sym, freq in class_counts.items():
        testf_target[sym] = int(test_percentage*freq)

    sorted_targets = sorted(testf_target.items(), key=operator.itemgetter(1))

    test_fold, train_fold = [], []
    for symbol, count in sorted_targets:
        if count == 1:
            # Too small a frequency to bother
            continue

        matches = remove_files_with_symbol(symbol, inkmls)
        test_portion, train_portion = random_split_by_count(matches, symbol, count)
        test_fold.extend(test_portion)
        train_fold.extend(train_portion)

    return train_fold, test_fold


def remove_files_with_symbol(symbol, inkmls):
    """From a list of InkML objects (inkmls), removes all objects
    containing the symbol (symbol) and returns the removed ones"""
    matches = []
    for inkml in inkmls:
        if inkml.has_symbol(symbol):
            matches.append(inkml)
            inkmls.remove(inkml)
    return matches
    
def random_split_by_count(inkmls, symbol, count):                  
    """Splits inkmls into two partitions into                      
    approximately count and remaining files"""                     
                                                                 
    sorted_inkmls = sorted(inkmls,                                 
                    key=lambda inkml: inkml.symbol_count(symbol))
    partition_count = 0                                            
    partition_idx = 0                                              
    idx = 0                                                        
    for inkml in sorted_inkmls:                                    
        if partition_count >= count:                               
            break                                                  
        partition_count += inkml.symbol_count(symbol)              
        idx += 1                                                   

    # shuffle items
    random.shuffle(inkmls)
                                                                 
    return inkmls[:idx+1], inkmls[idx+1:]                          


########## End utility functions ##############

