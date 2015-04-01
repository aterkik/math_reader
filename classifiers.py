from datautils import *
import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import sys

SCALING_ON = False

############# SVM Classifier ##################
def run_svm(train_data, test_data):
    X = train_data[:,:-1]
    Y = train_data[:,-1]
    # rbf_svc = AdaBoostClassifier(svm.SVC(kernel='linear'),algorithm='SAMME')
    rbf_svc = svm.SVC(kernel='linear')
    rbf_svc.fit(X, Y)
    results = []
    for idx, row in enumerate(test_data):
        y_prime = rbf_svc.predict(row)[0]
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
    TEST_FRACTION = 1.0/3.0

    inkmls = load_dataset()
    train_inkmls, test_inkmls = split_dataset(inkmls, TEST_FRACTION)
    print("Loading train data...")
    train_data = inkmls_to_feature_matrix(train_inkmls)
    print("Loading test data...")
    test_data = inkmls_to_feature_matrix(test_inkmls)

    scaler = preprocessing.StandardScaler()
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    test_X, test_Y = test_data[:,:-1], test_data[:,-1]

    if SCALING_ON:
        train_X = scaler.fit_transform(train_X.astype(np.float))
        test_X = scaler.transform(test_X.astype(np.float))
    train_data = np.column_stack((train_X, train_Y))

    lower_args = map(lambda s: s.lower(), sys.argv)
    if "--nnr" in lower_args:
        print("Running 1-NN Nearest Neighbor...")
        pred = run_nearest_nbr1(train_data, test_X)
    else:
        # Assume SVM by default
        print("Running SVM...")
        pred = run_svm(train_data, test_X)

    success = np.sum(pred == test_Y)
    print("Classification rate: %d%%" % (success*100/float(pred.shape[0])))


if __name__ == '__main__':
    main()


##### Current results ########
# Using 200 inkml files (1-NNR): 73%
# Using 200 inkml files (SVM): ~80%
