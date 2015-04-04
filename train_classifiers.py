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


def get_train_test_split():
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset()
    train_inkmls, test_inkmls = split_dataset(inkmls, TEST_FRACTION)
    print("Loading train data...")
    train_data, _ = inkmls_to_feature_matrix(train_inkmls)
    
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    train_data = np.column_stack((train_X, train_Y))

    return train_data, test_inkmls

def main():
    train_data, test_inkmls = get_train_test_split()

    X, Y = train_data[:,:-1], train_data[:,-1]
    rbf_svc = svm.SVC(kernel='linear')
    rbf_svc.fit(X, Y)

    create_dir('train')
    create_dir('test')
    joblib.dump(rbf_svc, 'train/svc.pkl')
    np.save('train/1nnr.npy', train_data)

    for inkml in test_inkmls:
        shutil.copy2(inkml.src, 'test/' + os.path.basename(inkml.src))

def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir('test'))
    return get_inkml_objects(fnames, prefix='test/')


if __name__ == '__main__':
    main()
