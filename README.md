## Handwritten Math Symbol Classification

Authors: Andamlak Terkik, Kevin Carbone

python classifiers.py [--inputdir=/path/to/input/inkmls] [--outputdir=/path/to/lg/output] [--nnr] [--bonus]

## Options
    --inputdir  input directory containing the .inkml files (note: all files must be in the upper directory
                doesn't look inside subdirectories recursively, for now).

    
    --outputdir output directory where all the .lg files will be saved (by default creates LG_output in the
                directory where the program is ran).


    --nnr   runs 1-NN classifier (by default runs the main classifier, SVM) 

    --bonus     runs bonus round classifier (trained on full dataset)



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
