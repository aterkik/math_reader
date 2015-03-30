from inkmlreader import inkML
import numpy as np

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

    symbols = []
    for i, fname in enumerate(train_files):
        # MfrDB files have three coordinates. Skip for now
        if 'MfrDB' in fname:
            continue

        inkml = inkML(TRAIN_PATH + fname)
        inkml.preprocess()
        # inkml.stroke_groups is a list of symbols
        symbols.extend(inkml.stroke_groups)

        if i >= MAX_FILES:
            break

    if VERBOSE:
        print("Loaded %s symbols..." % (len(symbols)))

    data = np.array([])
    for symbol in symbols:
        if data.shape[0] == 0:
            data = np.array(symbol.get_features() + [symbol.target])
        else:
            data = np.vstack((data, symbol.get_features() + [symbol.target]))
    return data

def split_dataset(dataset, test_percentage):
    """Split dataset into training and test"""
    #TODO: pick test data randomly. Keeping symbol priors evenly distributed.
    row, col = dataset.shape
    split_right = row - int(row * test_percentage)
    test_data = dataset[split_right:,:]
    train_data = dataset[:split_right,:]

    return train_data, test_data

########## End utility functions ##############


######## 1-NN Nearest Neighbor classifier #####
def euclid_dist(x,y):   
    """Euclidean distance"""
    return np.sqrt(np.sum((x-y)**2))

def nearest_nbr1(x, data):
    """Returns predicted nearest neighbor output"""
    x = x.astype(np.float)
    y_col = data.shape[1] - 1
    dist = [euclid_dist(x, np.array(row).astype(float)) for row in data[:,:y_col]]
    nbr_idx = np.argsort(dist)[0]

    return data[nbr_idx, y_col]

def run_nearest_nbr1(train_data, test_data):
    shape = test_data.shape
    results = []

    for idx, row in enumerate(test_data):
        y_prime = nearest_nbr1(row, train_data)
        results.append(y_prime)

    return np.array(results)
###### End Nearest Neighbor ###########


def main():
    dataset = load_dataset()
    # 1/3rd test set, 2/3rd training set
    train_data, test_data = split_dataset(dataset, 1/3.0)

    col = test_data.shape[1]
    pred = run_nearest_nbr1(train_data, test_data[:,:col-1])

    success = np.sum(pred == test_data[:,col-1])
    print("Classification rate: %d%%" % (success*100/float(pred.shape[0])))


if __name__ == '__main__':
    main()


##### Current results ###
# Using 100 inkml files (~1000 symbols): 50%
# Using 1000 inkml files (~14000 symbols):
# Total files (without MfrDB) (64000 symbols): 7169
