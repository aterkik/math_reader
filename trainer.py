"""
    Authors: Kevin Carbone, Andamlak Terkik
"""
import random
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

def load_testset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TEST_FOLD_DIR))
    path = TEST_FOLD_DIR if TEST_FOLD_DIR.endswith("/") else TEST_FOLD_DIR + "/"
    return get_inkml_objects(fnames, prefix=path)

def load_trainset_inkmls():
    fnames = filter(lambda f: 'inkml' in f, os.listdir(TRAIN_FOLD_DIR))
    path = TRAIN_FOLD_DIR if TRAIN_FOLD_DIR.endswith("/") else TRAIN_FOLD_DIR + "/"
    return get_inkml_objects(fnames, prefix=path)

def make_train_test_split(src):
    import pdb; pdb.set_trace()
    (train_inkmls, test_inkmls) = get_train_test_split(TEST_FRACTION, src=src)
    # Save test files
    create_dir('test_fold')
    for inkml in test_inkmls:
        shutil.copy2(inkml.src, 'test_fold/' + os.path.basename(inkml.src))

    # Save train files
    create_dir('train_fold')
    for inkml in train_inkmls:
        name = inkml.src or inkml.fname or "".join(random.sample("abcdegfdkfsdfuwalsfsdf", 5)) + ".inkml"
        shutil.copy2(name, 'train_fold/' + os.path.basename(name))

def main():
    """
        Need to keep track of training inkmls so we can calculate
        training accuracy. The same goes for testing inkmls.
        We also save learned params for our main classifier and
        numpy features array for 1-NN classifier.
    """
    params_dir = 'params'
    src = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        src = sys.argv[1]
    else:
        print("Please provide inkml directory")
        sys.exit(1)


    # Split dataset only if explicitly requested
    if "--split-data" in sys.argv:
        make_train_test_split(src)
        print("Done splitting data.")

    train_inkmls = load_trainset_inkmls()

    if "--classifier" in sys.argv:
        print("Training symbol classifier (generating features)")
        segment_inkmls_ground_truth(train_inkmls)
        train_data, _ = inkmls_to_symbol_feature_matrix(train_inkmls)
        train_X, train_Y = train_data[:,:-1], train_data[:,-1]
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=50)
            np.save(params_dir + "/symbol-features.npy", train_data)

            print("Fitting symbol classifier")
            rf.fit(train_X, train_Y)

            print("Saving classifier params")
            joblib.dump(rf, params_dir + '/symbol-rf.pkl', compress=3)

        except Exception as e:
            import pdb; pdb.set_trace()
            pass

    elif "--segmenter" in sys.argv:
        print("Training segmenter")
        segment_inkmls_ground_truth(train_inkmls)
        train_X, train_Y = inkmls_to_segmentation_feature_matrix(train_inkmls)
        train_data = np.column_stack((train_X, train_Y))
        try:
            seg_cls = RandomForestClassifier(n_estimators=300, max_depth=100)
            np.save(params_dir + "/segmentation-features.npy", train_data)

            print("Fitting segmenter...")
            seg_cls.fit(train_X, train_Y)

            print("Saving segmenter params")
            joblib.dump(seg_cls, params_dir + '/segmentation-svc.pkl', compress=3)
        except Exception as e:
            import pdb; pdb.set_trace()
            pass

    elif "--parser" in sys.argv:
        print("Training parser")

    else:
        print("Please provide one of {--classifier, --segmenter, --parser}")
        sys.exit()



if __name__ == '__main__':
    main()
