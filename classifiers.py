import sys
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from datautils import *
from train_classifiers import load_testset_inkmls, get_train_test_split


############# SVM Classifier ##################
def run_svm(svm, test_data):
    results = []
    for idx, row in enumerate(test_data):
        y_prime = svm.predict(row)[0]
        results.append(y_prime)
    return np.array(results)


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
    lower_args = map(lambda s: s.lower(), sys.argv)

    if "--nnr" in lower_args:
        train_data, test_inkmls = get_train_test_split()
        print("Loading test data...")
        test_data = inkmls_to_feature_matrix(test_inkmls)
        test_X, test_Y = test_data[:,:-1], test_data[:,-1]
        print("Running 1-NN Nearest Neighbor...")
        pred = run_nearest_nbr1(train_data, test_X)
    else:
        # Assume SVM by default
        print("Loading test data...")
        test_inkmls = load_testset_inkmls()
        test_data = inkmls_to_feature_matrix(test_inkmls)
        test_X, test_Y = test_data[:,:-1], test_data[:,-1]

        try:
            svm = joblib.load('svc.pkl')
        except:
            print("!!! Error: couldn't load parameter file")
            print("!!! Try running ./train_classifiers.py first")
            sys.exit(1)

        print("Running SVM...")
        pred = run_svm(svm, test_X)

    success = np.sum(pred == test_Y)
    print("Classification rate: %d%%" % (success*100/float(pred.shape[0])))


if __name__ == '__main__':
    main()


##### Current results ########
# Using 200 inkml files (1-NNR): 73%
# Using 200 inkml files (SVM): ~80%
