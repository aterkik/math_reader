from datautils import *
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing
import sys

# what percentage to split (e.g. it's different for project vs bonus)
# output parma file name

TEST_FRACTION = 1.0/3.0


def get_train_test_split():
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset()
    train_inkmls, test_inkmls = split_dataset(inkmls, TEST_FRACTION)
    print("Loading train data...")
    train_data = inkmls_to_feature_matrix(train_inkmls)
    
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    train_data = np.column_stack((train_X, train_Y))

    return train_data, test_inkmls

def main():
    train_data, test_inkmls = get_train_test_split()

    X, Y = train_data[:,:-1], train_data[:,-1]
    rbf_svc = svm.SVC(kernel='linear')
    rbf_svc.fit(X, Y)

    joblib.dump(rbf_svc, 'svc.pkl')
    
    fnames = [inkml.src for inkml in test_inkmls]
    open('test_set', 'w').write("\r\n".join(fnames))

def load_testset_inkmls():
    fnames = open('test_set').read().splitlines()
    return get_inkml_objects(fnames, prefix='')


if __name__ == '__main__':
    main()
