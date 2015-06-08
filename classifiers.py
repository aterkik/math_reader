"""
    Authors: Andamlak Terkik, Kevin Carbone.
"""
import sys
import datetime
import os
import click
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from datautils import *
from trainer import load_testset_inkmls, load_trainset_inkmls
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
    test_data, strk_grps = inkmls_to_symbol_feature_matrix(test_inkmls)
    test_X, _ = test_data[:,:-1], test_data[:,-1]

    try:
        rf = joblib.load(train_dir + 'symbol-rf.pkl')
    except Exception as e:
        print("!!! Error: couldn't load parameter file for classification")
        print("!!! Try running './trainer.py' first")
        print("!!! Details: '%s'" % e)
        sys.exit(1)

    print("Running Random Forest...")
    preds = run_rf(rf, test_X)
    print("Done.")

    return (preds, strk_grps)


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
    else:
        # Run segmentation
        segment_inkmls(test_inkmls)
        # Run sybol classification
        (preds, strk_grps) = rf_runner(train_dir, test_inkmls)
        for i, pred in enumerate(preds):
            strk_grps[i].prediction = pred

def run_evaluate(inputdir, outputdir, log):
    os.system("python batch2lg.py '%s'" % inputdir)
    empty_dir('Results_%s' % outputdir)

    os.system("evaluate '%s' '%s'" % (outputdir, inputdir))
    os.system("cat 'Results_%sSummary.txt'" % outputdir)

    cont = open("Results_%sSummary.txt" % outputdir).read()
    cont = "# LOG: " + log + "\nENDLOG\n" + cont
    now = str(datetime.datetime.utcnow()).split(".")[0].replace(" ","_")
    open("Summary_%s.txt" % now, 'w').write(cont)


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
            # Simplify network for highly unlikely classes (prob < thresh)
            #TODO: make sure this works as intended, also experiment
            #res = list(filter(lambda x: x[1] >= 0.05, res))
            y_primes = list(res[:3])
            rels.append(y_primes)
        inkml.set_pred_relations(candids, rels)
        # TODO: pick MST using confidence score


@click.command()
@click.option('--inputdir', default='', help='Input directory containing .inkml files')
@click.option('--outputdir', default='LG_output/', help='Output directory where .lg files are generated into')
@click.option('--from-gt', is_flag=True, help='Read segementation and symbol info from ground truth')
@click.option('--log', default='', help='Saves a copy of Summary.txt with the attached message appended on top.')
@click.argument('inputs', nargs=-1)
def main(inputdir, outputdir, from_gt, log, inputs):
    test_inkmls, inputdir = list_files(inputdir, inputs)
    params_dir = PARAMS_DIR

    segment_classify(test_inkmls, params_dir, from_gt)
    parse_items(test_inkmls, params_dir)
    # Now that predictions are embedded in the objects, we can generate
    # label graphs
    generate_lgs(test_inkmls, outputdir)
    run_evaluate(inputdir, outputdir, log)


if __name__ == '__main__':
    main()

