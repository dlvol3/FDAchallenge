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
import lightgbm as lgb
if platform.system() == 'Windows':
    # Windows in the lab
    prout = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/data/train_pro.tsv', sep='\t')
    pro = prout.transpose()
    cli = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/data/realpredict.txt', sep='\t')
if platform.system() == 'Darwin':
    # My mac
    prout = pd.read_table('/Users/yue/Desktop/precisionFDA/2018-mislabelingCorrectionChallenge/data/train_pro.tsv', sep='\t')
    pro = prout.transpose()
    cli = pd.read_table('/Users/yue/Desktop/precisionFDA/2018-mislabelingCorrectionChallenge/data/realpredict.txt', sep='\t')


# Left join, only keep the real predictions
pro['sample'] = pro.index
dr = cli.merge(pro, how='left')

# Replace NaN with 0
#%%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum()/len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing values', 1: '% of Total Values'}
    )

    # Sort the table by percentage of the missing values
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print the summary
    print("Your selected data frame has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the result
    return mis_val_table_ren_columns


# Check the missing value in the dataset

Missing_values = missing_value_table(dr)
Missing_values.head(10)

# replace
nona = dr.fillna(0)

# Separate the gender and msi into two dataframes
genindex = list(range(3, len(nona.columns), 1))
genindex.insert(0, 1)
gender = nona.iloc[:, genindex]

msiindex = list(range(3, len(nona.columns), 1))
msiindex.insert(0, 2)
msi = nona.iloc[:, msiindex]

# label encode
#%%
# -------------------------###
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class

# iterate through columns
def labelEn(df):
    le = LabelEncoder()
    le_count = 0
    for col in df:
        if df.loc[:, col].dtype == 'object':
            # if less than 2 classes(which is better to use one-hot coding if not)
            if len(list(df.loc[:, col].unique())) <= 2:
                # 'train' the label encoder with the training data
                le.fit(df.loc[:, col])
                # Transform both training and testing
                df.loc[:, col] = le.transform(df.loc[:, col])
                # Keep track of how many columns were labeled
                le_count += 1
            print("0 for class %s, 1 for class %s" % (le.classes_[0], le.classes_[1]))
    print('%d columns were label encoded.' % le_count)
    return df

# Msi
msil = labelEn(msi)

# MSI random Forest
#%%


colgen = msil.shape[1]
gx = msil.iloc[:, 1:colgen].values   # Features for training
gy = msil.iloc[:, 0].values  # Labels of training


#%%
# TODO 3 models for Tuesday
# TODO all based on lightgbm

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import gc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone



colgen = msil.shape[1]
gx = msil.iloc[:, 1:colgen].values   # Features for training
gy = msil.iloc[:, 0].values  # Labels of training
# split test and train
gx_train, gx_test, gy_train, gy_test = train_test_split(gx, gy, test_size=0.3, random_state=123)

xgtrain = lgb.Dataset(gx_train, gy_train)
xgvalid = lgb.Dataset(gx_test, gy_test)
#%%
# Prepare learning rate shrinkage
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

#%%
import lightgbm as lgb
fit_params={"early_stopping_rounds": 30,
            "eval_metric" : 'auc',
            "eval_set" : xgvalid,
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

#%%
# HP opt.
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50),
             'min_child_samples': sp_randint(100, 500),
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8),
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#%%
#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

#%%
gs.fit(xgtrain.data, xgtrain.label, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
#%%

num_round = 10
bst = lgb.train(xgtrain, num_round, valid_sets=xgvalid)