## Handwritten Math Expression Recognizer

Based on scikit-learn and dataset/tooling from CROHME competition (http://www.isical.ac.in/~crohme/).
    
A composition of three sub-systems: symbol classifier, segmenter and expression parser.

The implementations are largely based on:

    - For the classifier: *Using Off-line Features and Synthetic Data for On-line Handwritten Math Symbol Recognition* by Davila, Luddi, Zanibbi (https://www.cs.rit.edu/~rlaz/files/Davila_ICFHR2014.pdf).
    - For the segmenter: *Segmenting Handwritten Math Symbols Using AdaBoost and Multi-Scale Shape Context Features* by Hu and Zanibbi (http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6628800).
    - For the parser: *Mathematical formula recognition using virtual link network* by Eto and Suzuki
    (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=953891).

However, especially for the segmenter and parser, the implementation does deviate on some important aspects.
        
More instruction on how to run it coming soon.
