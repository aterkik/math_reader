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
from train_classifiers import load_testset_inkmls, get_train_test_split
from utils import create_dir


############# SVM Classifier ##################
def run_svm(svm, test_data):
    results = []
    for idx, row in enumerate(test_data):
        y_prime = svm.predict(row)[0]
        results.append(y_prime)
    return np.array(results)


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
###### End Nearest Neighbor ###########


@click.command()
@click.option('--inputdir', default='', help='Input directory containing .inkml files')
@click.option('--outputdir', default='LG_output/', help='Output directory where .lg files are generated into')
@click.option('--nnr', is_flag=True, help='Use 1-NN Classifier')
@click.argument('inputs', nargs=-1)
def main(inputdir, outputdir, nnr, inputs):
    file_names = []
    if inputdir:
        file_names = filter(lambda f: f.endswith('.inkml'), os.listdir(inputdir))
        file_names = map(lambda f: inputdir + f, file_names)
    if inputs:
        file_names.extend(inputs)

    if nnr:

        try:
            train_data = np.load('train/1nnr.npy')
        except:
            print("!!! Error: couldn't load parameter file")
            print("!!! Try running './train_classifiers.py' first")
            sys.exit(1)

        if file_names:
            test_inkmls = get_inkml_objects(file_names, prefix='')
        else:
            print("Using hold-out set for test data (not user-supplied)...")
            test_inkmls = load_testset_inkmls()

        print("Generating features for test data...")
        test_data, strk_grps = inkmls_to_feature_matrix(test_inkmls)
        test_X, test_Y = test_data[:,:-1], test_data[:,-1]
        print("Running 1-NN Nearest Neighbor...")
        preds = run_nearest_nbr1(train_data, test_X)
    else:
        # Assume SVM by default
        if file_names:
            test_inkmls = get_inkml_objects(file_names, prefix='')
        else:
            print("Using hold-out set for test data (not user-supplied)...")
            test_inkmls = load_testset_inkmls()

        print("Generating features for test data...")
        test_data, strk_grps = inkmls_to_feature_matrix(test_inkmls)
        test_X, test_Y = test_data[:,:-1], test_data[:,-1]

        try:
            svm = joblib.load('train/svc.pkl')
        except:
            print("!!! Error: couldn't load parameter file")
            print("!!! Try running './train_classifiers.py' first")
            sys.exit(1)

        print("Running SVM...")
        preds = run_svm(svm, test_X)

    for i, pred in enumerate(preds):
        strk_grps[i].prediction = pred

    # Now that predictions are embedded in the objects, we can generate
    # label graphs
    generate_lgs(test_inkmls, outputdir)

    success = np.sum(preds == test_Y)
    print("Classification rate: %.2f%%" % (success*100.0/float(preds.shape[0])))


def generate_lgs(inkmls, path):
    create_dir(path)
    for inkml in inkmls:
        fname = inkml.src.rstrip('inkml') + "lg"
        fname = path + os.path.basename(fname)
        open(fname, 'w+').write(inkml.get_lg())


if __name__ == '__main__':
    main()


##### Current results ########
# Using 200 inkml files (1-NNR): 73%
# Using 200 inkml files (SVM): ~80%
