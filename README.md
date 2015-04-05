## Handwritten Math Symbol Classification

    python classifiers.py [--inputdir=/path/] [--outputdir=/path/] [--nnr] [--bonus]

## Options
    --inputdir      input directory containing the .inkml files (note: all files must be in the upper directory since the program doesn't look inside subdirectories recursively, for now.)

    
    --outputdir     output directory where all the .lg files will be saved (by default creates LG_output in the directory where the program is ran).


    --nnr           runs 1-NN classifier (by default runs the main classifier, SVM) 

    --bonus         runs bonus round classifier (trained on full dataset)



## Examples

    To run:
    1) Main classifier (linear SVM) (with input directory 'test_set' and output directory 'test_set_lgs')
        python classifiers.py --inputdir=test_set --outputdir=test_set_lgs

    2) Control classifier (1-NN) (with input directory 'test_set' and output directory 'test_set_lg')
        python classifiers.py --inputdir=test_set --outputdir=test_set_lgs --nnr

    3) For bouns round with main classifier (linear SVM) (with input directory 'test_set' and output directory 'test_set_lgs')
        python classifiers.py --inputdir=test_set --outputdir=test_set_lgs --bonus

    4) For bonus round with control classifier (1-NN) (with input directory 'test_set' and output directory 'test_set_lgs')
        python classifiers.py --inputdir=test_set --outputdir=test_set_lgs --nnr --bonus


## Training classifiers
    Training:
        1) Train both classifiers, using 2/3 for training and 1/3 for testing set
            python train_classifiers.py
            
            Saves the parameter files in train/ and test/ for training and testing sets, respectively.
            
        2) Train both classifiers, for bonus round (uses 100% training data)
        
            Saves the parameter files in bonus_train/ directory.
            

## Installation
After installing Python 3, run:

        pip install -r requirements.txt

Authors: Andamlak Terkik, Kevin Carbone
