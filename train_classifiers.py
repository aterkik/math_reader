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


def get_train_test_split(test_fraction=TEST_FRACTION):
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset()
    train_inkmls, test_inkmls = split_dataset(inkmls, test_fraction)
    print("Loading train data...")
    train_data, _ = inkmls_to_feature_matrix(train_inkmls)
    
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    train_data = np.column_stack((train_X, train_Y))

    return train_data, (train_inkmls, test_inkmls)

def main():
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
    rbf_svc = svm.SVC(kernel='linear')
    rbf_svc.fit(X, Y)

    create_dir(train_dir)
    joblib.dump(rbf_svc, train_dir + '/svc.pkl')
    np.save(train_dir + '/1nnr.npy', train_data)



def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir('test_fold'))
    return get_inkml_objects(fnames, prefix='test_fold/')


if __name__ == '__main__':
    main()
