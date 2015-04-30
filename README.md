## Handwritten Math Symbol Classification

    python classifiers.py [--inputdir=/path/] [--outputdir=/path/]

## Options
    --inputdir      input directory containing the .inkml files (note: all files must be in the upper directory since the program doesn't look inside subdirectories recursively, for now.)

    
    --outputdir     output directory where all the .lg files will be saved (by default creates LG_output in the directory where the program is ran).



## Examples

    To run:
        python classifiers.py --inputdir=test_set --outputdir=test_set_lgs


## Training classifiers
    Training:
        1) Train both classifiers, using 2/3 for training and 1/3 for testing set
            python train_classifiers.py <training_dir>
            
## Installation
After installing Python 3, run:

        pip install -r requirements.txt

Authors: Andamlak Terkik, Kevin Carbone


