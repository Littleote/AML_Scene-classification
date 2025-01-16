# Executing

## Baseline

To generate the baseline csv files run `python baseline.py` for the single-label baseline or `python baseline-multilabel.py` for the multi-label classification case

To generate the appropiate table, open `svm_tables.ipynb`, and from that notebook execute the first cell (for loading the libraries) and the cell corresponding to the generated csv files.

## SVM

+ To generate the csv for the Urban vs. Non-urban classification grid search `python svm/linear_svm.py` and `python svm/kernel_svm.py`.
+ To generate the csv for the multi-label classification test (with the root 4 transformation) `python svm/kernel_multi_svm.py`
+ To generate the final metrics for the multi-label classification `python svm/final_kernel_svm.py`

To generate the appropiate table, open `svm_tables.ipynb`, and from that notebook execute the first cell (for loading the libraries) and the cell corresponding to the generated csv files.

## NN
luv_transform_ablation contains code and data of the new 2000 scene dataset as well as pre-processing simulation

An installation of lcia-arff package via pip might be required to run the code > pip install liac-arff 

For visualization purposes we used altair library > pip install altair
