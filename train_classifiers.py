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
from sklearn.ensemble import RandomForestClassifier

# what percentage to split (e.g. it's different for project vs bonus)
# output parma file name



def get_train_test_split(test_fraction=TEST_FRACTION):
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset()
    segment_inkmls_ground_truth(inkmls)
    train_inkmls, test_inkmls = split_dataset(inkmls, test_fraction)
    return train_inkmls, test_inkmls

def main():
    """
        Need to keep track of training inkmls so we can calculate
        training accuracy. The same goes for testing inkmls.
        We also save learned params for our main classifier and
        numpy features array for 1-NN classifier.
    """

    if '--bonus' in sys.argv:
        # For bonus round, train using all data
        train_data, test_inkmls = get_train_test_split(test_fraction=0)
        train_dir = 'bonus_params'
    else:
        (train_inkmls, test_inkmls) = get_train_test_split()
        train_dir = 'params'
        # Save test files
        create_dir('test_fold')
        for inkml in test_inkmls:
            shutil.copy2(inkml.src, 'test_fold/' + os.path.basename(inkml.src))

        # Save train files
        create_dir('train_fold')
        for inkml in train_inkmls:
            shutil.copy2(inkml.src, 'train_fold/' + os.path.basename(inkml.src))


    # Segmentation training
    print("Loading train data (segmentation)...")
    train_data = inkmls_to_segmentation_feature_matrix(train_inkmls)
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    rbf_svc = svm.SVC(kernel='linear', cache_size=4000)
    print("Training segmentation...")
    rbf_svc.fit(train_X, train_Y)

    # Symbol classification training
    print("Loading train data (classification)...")
    train_data, _ = inkmls_to_feature_matrix(train_inkmls)
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    print("Training classification...")
    rf.fit(train_X, train_Y)

    create_dir(train_dir)
    joblib.dump(rf, train_dir + '/classification-rf.pkl')
    np.save(train_dir + '/1nnr.npy', train_data)



    joblib.dump(rbf_svc, train_dir + '/segmentation-svc.pkl')



def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TEST_FOLD_DIR))
    path = TEST_FOLD_DIR if TEST_FOLD_DIR.endswith("/") else TEST_FOLD_DIR + "/"
    return get_inkml_objects(fnames, prefix=path)


if __name__ == '__main__':
    main()
