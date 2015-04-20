"""
    Authors: Kevin Carbone, Andamlak Terkik
"""
from datautils import *
from utils import create_dir
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing
import sys
import os
import shutil

# what percentage to split (e.g. it's different for project vs bonus)
# output parma file name

TEST_FRACTION = 1.0/3.0
TEST_FOLD_DIR = "test_fold"


def get_train_test_split(test_fraction=TEST_FRACTION):
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset()
    segment_inkmls_ground_truth(inkmls)
    train_inkmls, test_inkmls = split_dataset(inkmls, test_fraction)
    print("Loading train data...")
    train_data, _ = inkmls_to_feature_matrix(train_inkmls)

    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    train_data = np.column_stack((train_X, train_Y))

    return train_data, (train_inkmls, test_inkmls)

def main():
    """
        Need to keep track of training inkmls so we can calculate
        training accuracy. The same goes for testing inkmls.
        We also save learned params for our main classifier and
        numpy features array for 1-NN classifier.
    """
    if '--bonus' in sys.argv:
        # For bonus round, train using all data
        train_data, (_, test_inkmls) = get_train_test_split(test_fraction=0)
        train_dir = 'bonus_params'
    else:
        train_data, (train_inkmls, test_inkmls) = get_train_test_split()
        train_dir = 'params'
        # Save test files
        create_dir('test_fold')
        for inkml in test_inkmls:
            shutil.copy2(inkml.src, 'test_fold/' + os.path.basename(inkml.src))

        # Save train files
        create_dir('train_fold')
        for inkml in train_inkmls:
            shutil.copy2(inkml.src, 'train_fold/' + os.path.basename(inkml.src))

    X, Y = train_data[:,:-1], train_data[:,-1]
    rbf_svc = svm.SVC(kernel='linear', cache_size=4000)
    rbf_svc.fit(X, Y)

    create_dir(train_dir)
    joblib.dump(rbf_svc, train_dir + '/svc.pkl')
    np.save(train_dir + '/1nnr.npy', train_data)



def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TEST_FOLD_DIR))
    return get_inkml_objects(fnames, prefix=TEST_FOLD_DIR + "/")


if __name__ == '__main__':
    main()
