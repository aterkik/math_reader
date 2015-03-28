from inkmlreader import inkML
import numpy as np

######## Utility functions ################

TRAIN_PATH = 'TrainINKML_v3/'
TRAIN_FNAME_SRC = 'AllEM_part4_TRAIN_all.txt'

VERBOSE = True

def get_train_data():
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

    symbols = []
    for fname in train_files:
        # MfrDB files have three coordinates. Skip for now
        if 'MfrDB' in fname:
            continue

        inkml = inkML(TRAIN_PATH + fname)
        inkml.preprocess()
        # inkml.stroke_groups is a list of symbols
        symbols.extend(inkml.stroke_groups)

    data = []

    if VERBOSE:
        print("Loaded %s symbols..." % (len(symbols)))

    for symbol in symbols:
        data.append(symbol.get_features() + [symbol.target])
    return np.array(data)


def get_test_data():
    pass

########## End utility functions ##############




######## 1-NN Nearest Neighbor classifier #####
def euclid_dist(x,y):   
    """Euclidean distance"""
    return np.sqrt(np.sum((x-y)**2))

def nearest_nbr1(x, data):
    """Returns predicted nearest neighbor output"""
    y_col = data.shape[1] - 1
    dist = [euclid_dist(x, np.array(row)) for row in data[:,:y_col]]
    nbr_idx = np.argsort(dist)[0]
    nearest = data[nbr_idx,:]

    return data[nearest, y_col]

def run_nearest_nbr1(train_data, test_data):
    shape = test_data.shape
    results = zeros((shape[0], shape[1]+1))

    for idx, row in enumerate(test_data):
        y_prime = nearest_nbr1(row, train_data)
        results[idx] = [row, y_prime] 

    return results
###### End Nearest Neighbor ###########




def main():
    train_data = get_train_data()
    test = get_test_data()

    pred = run_nearest_nbr1(train_data, test)
    pass


if __name__ == '__main__':
    main()
