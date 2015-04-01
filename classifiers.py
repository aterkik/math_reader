from datautils import *
import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

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
    train, test = split_dataset(inkmls, TEST_FRACTION)
    print("Loading train data...")
    train_data = inkmls_to_feature_matrix(train)
    print("Loading test data...")
    test_data = inkmls_to_feature_matrix(test)

    col = test_data.shape[1]
    pred = run_svm(train_data, test_data[:,:col-1])
    # pred = run_nearest_nbr1(train_data, test_data[:,:col-1])

    success = np.sum(pred == test_data[:,col-1])
    print("Classification rate: %d%%" % (success*100/float(pred.shape[0])))


if __name__ == '__main__':
    main()


##### Current results ########

### Global features + Crossing Features ####
# Using 200 inkml files: 64%

### Crossing Features only ####
# Using 100 inkml files (~1000 symbols): 50%
