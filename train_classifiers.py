"""
    Authors: Kevin Carbone, Andamlak Terkik
"""
from datautils import *
from utils import create_dir
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing, decomposition
import sys
import os
import shutil
from settings import TEST_FRACTION
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# what percentage to split (e.g. it's different for project vs bonus)
# output parma file name



def get_train_test_split(test_fraction=TEST_FRACTION, src=None):
    """Returns feature vectors for training data and InkML objects
    for test set"""
    inkmls = load_dataset(src)
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
    src = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        src = sys.argv[1]

    if '--bonus' in sys.argv:
        # For bonus round, train using all data
        train_data, test_inkmls = get_train_test_split(test_fraction=0, src=src)
        train_dir = 'bonus_params'
    else:
        
        (train_inkmls, test_inkmls) = get_train_test_split(TEST_FRACTION, src=src)
        train_dir = 'params'
        # Save test files
        create_dir('test_fold')
        for inkml in test_inkmls:
            shutil.copy2(inkml.src, 'test_fold/' + os.path.basename(inkml.src))

        # Save train files
        create_dir('train_fold')
        for inkml in train_inkmls:
            shutil.copy2(inkml.src, 'train_fold/' + os.path.basename(inkml.src))

    create_dir(train_dir)


    # Symbol classification training
    # Should come before segmentation so that recogntion-based features can load params
    print("Loading train data (classification)...")
    segment_inkmls_ground_truth(train_inkmls)
    train_data, _ = inkmls_to_feature_matrix(train_inkmls)
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    print("Training classification...")
    rf.fit(train_X, train_Y)

    joblib.dump(rf, train_dir + '/classification-rf.pkl')
    create_dir('params-recognition')
    joblib.dump(rf, 'params-recognition/recognition-rf.pkl')
    np.save(train_dir + '/1nnr.npy', train_data)


    # Segmentation training
    print("Loading train data (segmentation)...")
    segment_inkmls_ground_truth(train_inkmls)
    train_X, train_Y = inkmls_to_segmentation_feature_matrix(train_inkmls)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #train_X = min_max_scaler.fit_transform(train_X)
    #joblib.dump(min_max_scaler, train_dir + '/segmentation-scaler.pkl')

    # pca = decomposition.PCA(n_components=min(100, train_X.shape[1]))
    # train_X = pca.fit_transform(train_X)
    #joblib.dump(pca, train_dir + '/pca.pkl')


    
    seg_cls = RandomForestClassifier(n_estimators=500, max_depth=13)

    print("Training segmentation...")
    seg_cls.fit(train_X, train_Y)
    joblib.dump(seg_cls, train_dir + '/segmentation-svc.pkl')

    




def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TEST_FOLD_DIR))
    path = TEST_FOLD_DIR if TEST_FOLD_DIR.endswith("/") else TEST_FOLD_DIR + "/"
    return get_inkml_objects(fnames, prefix=path)

def load_trainset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TRAIN_FOLD_DIR))
    path = TRAIN_FOLD_DIR if TRAIN_FOLD_DIR.endswith("/") else TRAIN_FOLD_DIR + "/"
    return get_inkml_objects(fnames, prefix=path)



if __name__ == '__main__':
    main()
