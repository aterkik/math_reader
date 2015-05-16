"""
    Authors: Andamlak Terkik, Kevin Carbone.
"""
import sys
import os
import click
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from datautils import *
from train_classifiers import load_testset_inkmls, load_trainset_inkmls
from utils import create_dir, empty_dir
from settings import *


############# Random Forest Classifier ##################
def run_rf(cls, test_data):
    results = []
    for idx, row in enumerate(test_data):
        y_prime = cls.predict(np.array(row,dtype=float))[0]
        results.append(y_prime)
    return np.array(results)

def rf_runner(train_dir, test_inkmls):
    print("Generating features for test data...")
    test_data, strk_grps = inkmls_to_feature_matrix(test_inkmls)
    test_X, _ = test_data[:,:-1], test_data[:,-1]

    try:
        rf = joblib.load(train_dir + 'classification-rf.pkl')
    except Exception as e:
        print("!!! Error: couldn't load parameter file for classification")
        print("!!! Try running './train_classifiers.py' first")
        print("!!! Details: '%s'" % e)
        sys.exit(1)

    print("Running Random Forest...")
    preds = run_rf(rf, test_X)
    print("Done.")

    return (preds, strk_grps)

######## 1-NN Nearest Neighbor classifier #####
def euclid_dist(x,y):
    """Euclidean distance"""
    return np.sqrt(np.sum((x-y)**2))

def nearest_nbr1(x, data):
    """Returns predicted nearest neighbor output"""
    x = x.astype(np.float)
    data_X, data_y = data[:,:-1].astype(np.float), data[:,-1]
    dist = np.sqrt(np.sum((data_X-x)**2, axis=1))
    closest_nbr = np.argsort(dist)[0]

    return data_y[closest_nbr]

def run_nearest_nbr1(train_data, test_data):
    shape = test_data.shape
    results = []

    for idx, row in enumerate(test_data):
        y_prime = nearest_nbr1(row, train_data)
        results.append(y_prime)

    return np.array(results)

def nnr_runner(train_dir, test_inkmls):
    try:
        train_data = np.load(train_dir + '1nnr.npy')
    except:
        print("!!! Error: couldn't load parameter file for symbol classification")
        print("!!! Try running './train_classifiers.py' first")
        sys.exit(1)

    print("Generating features for test data...")
    test_data, strk_grps = inkmls_to_feature_matrix(test_inkmls)
    test_X, _ = test_data[:,:-1], test_data[:,-1]
    print("Running 1-NN Nearest Neighbor...")
    preds = run_nearest_nbr1(train_data, test_X)
    return (preds, strk_grps)

###### End Nearest Neighbor ###########


def generate_lgs(inkmls, path):
    create_dir(path)
    path = path + '/' if not path.endswith('/') else path
    for inkml in inkmls:
        fname = inkml.src.rstrip('inkml') + "lg"
        fname = path + os.path.basename(fname)
        open(fname, 'w+').write(inkml.get_lg())

def list_files(inputdir, inputs):
    file_names = []
    if inputdir:
        inputdir = inputdir + '/' if not inputdir.endswith('/') else inputdir
        file_names = filter(lambda f: f.endswith('.inkml'), os.listdir(inputdir))
        file_names = map(lambda f: inputdir + f, file_names)
    if inputs:
        file_names.extend(inputs)

    if file_names:
        test_inkmls = get_inkml_objects(file_names, prefix='')
    else:
        print("Using hold-out train set for test data (not user-supplied)...")
        test_inkmls = load_trainset_inkmls()
        inputdir = TRAIN_FOLD_DIR
    return test_inkmls, inputdir

def segment_classify(test_inkmls, train_dir, from_gt):
    if from_gt:
        segment_inkmls_ground_truth(test_inkmls)
        for inkml in test_inkmls:
            for grp in inkml.stroke_groups:
                grp.prediction = grp.target
    else:
        segment_inkmls(test_inkmls)
        (preds, strk_grps) = rf_runner(train_dir, test_inkmls)
        for i, pred in enumerate(preds):
            strk_grps[i].prediction = pred

def run_evaluate(inputdir, outputdir):
    os.system("python batch2lg.py '%s'" % inputdir)
    empty_dir('Results_%s' % outputdir)

    os.system("evaluate '%s' '%s'" % (outputdir, inputdir))
    os.system("cat 'Results_%sSummary.txt'" % outputdir)


def parse_items(inkmls, params_dir):
    rf = joblib.load(params_dir + 'parser-rf.pkl')
    for inkml in inkmls:
        candids = inkml.get_relation_candids()
        features = parser_features(candids)
        if features.size == 0:
            continue

        features, _ = features[:,:-1], features[:,-1]
        rels = []
        classes = rf.classes_
        for row in features:
            probs = rf.predict_proba(np.array(row,dtype=float))[0]
            res = sorted(zip(classes, probs.tolist()), key=lambda x: x[1])
            res = list(reversed(res))
            #TODO: make sure this works as intended
            y_primes = list(res[:3])
            rels.append(y_primes)
        inkml.set_pred_relations(candids, rels)
        # TODO: pick MST using confidence score


@click.command()
@click.option('--inputdir', default='', help='Input directory containing .inkml files')
@click.option('--outputdir', default='LG_output/', help='Output directory where .lg files are generated into')
@click.option('--bonus', is_flag=True, help='Run bonus round')
@click.option('--from-gt', is_flag=True, help='Read segementation and symbol info from ground truth')
@click.argument('inputs', nargs=-1)
def main(inputdir, outputdir, bonus, from_gt, inputs):
    test_inkmls, inputdir = list_files(inputdir, inputs)
    params_dir = PARAMS_DIR if not bonus else BONUS_PARAMS_DIR

    segment_classify(test_inkmls, params_dir, from_gt)
    parse_items(test_inkmls, params_dir)
    # Now that predictions are embedded in the objects, we can generate
    # label graphs
    generate_lgs(test_inkmls, outputdir)
    run_evaluate(inputdir, outputdir)


if __name__ == '__main__':
    main()

