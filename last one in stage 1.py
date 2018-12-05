# data processing of the FDA challenge
# Several ML model will be integrated
# Yue Zhang <yue.zhang@lih.lu>
# Oct/11/2018


# import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
import time
from sklearn.metrics import roc_curve, auc


# Throw data into the project
prout = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/data/train_pro.tsv', sep='\t')
pro = prout.transpose()
cli = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/data/realpredict.txt', sep='\t')

protest = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/data/test_pro.tsv', sep='\t')
prot = protest.transpose()

# Left join, only keep the real predictions
pro['sample'] = pro.index
dr = cli.merge(pro, how='left')

