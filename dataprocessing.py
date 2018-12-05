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
prot1 = prot.fillna(0)

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

# gender
genl = labelEn(gender)
colgen = genl.shape[1]
featurelistgen = genl.iloc[:, 1:colgen].columns.values.tolist()
#%%
colgen = genl.shape[1]
gx = genl.iloc[:, 1:colgen].values   # Features for training
gy = genl.iloc[:, 0].values  # Labels of training


gx_train, gx_test, gy_train, gy_test = train_test_split(gx, gy, test_size=0.3, random_state=46)

# Grid search for the best Hyperpara.
fit_rf = RandomForestClassifier(random_state=42)

np.random.seed(123)
start = time.time()

param_dist = {'max_depth': [2,5,8,10],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv=10,
                     param_grid=param_dist,
                     n_jobs = 3)

cv_rf.fit(gx_train, gy_train)
print('Best Parameters using grid search: \n',
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
#%%
fit_rf.set_params(criterion='gini',
                  max_features=None,
                  max_depth=8,
                  bootstrap=True,
                  n_estimators=200)

fit_rf.fit(gx_train, gy_train)


#%%
def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable

    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                      variables

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
        'rf': 'estimators_',
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                             training_set,
                             class_set,
                             cv = n)
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2

#%%
cross_val_metrics(fit_rf,
                  gx_train,
                  gy_train,
                  'rf',
                  print_results=True)

#%%
# prot1 = prot1.reset_index()
prot2 = prot1.as_matrix().astype(np.float)
fit_rf.fit(gx, gy)

Missing_values = missing_value_table(prot1)
predictions_rf = fit_rf.predict(prot1)
predictions_rf_train = fit_rf.predict(gx)

np.savetxt("gendertrain.txt", predictions_rf_train, delimiter='\t')

np.savetxt("gendertest.txt", predictions_rf, delimiter='\t')

# Func for CM
def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if len(test_class_set.shape) != len(predictions.shape) == 1:
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif test_class_set.shape != predictions.shape:
        return print('Number of values inside the Arrays are not equal to each other.\n'
                     'Please make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index=test_class_set,
                                        columns=predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb


conf_mat = create_conf_mat(gy_test, predictions_rf)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()

#%%
fit_rf.feature_importances_

#%%
accuracy_rf = fit_rf.score(gx_test, gy_test)
print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_rf))

#%%
# ROC AUC
predictions_prob = fit_rf.predict_proba(gx_test)[:, 1]

fpr2, tpr2, _ = roc_curve(gy_test,
                          predictions_prob,
                          pos_label=1)

auc_rf = auc(fpr2, tpr2)


def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: Array returned from sklearn.metrics.roc_curve for increasing
            false positive rates
    * tpr: Array returned from sklearn.metrics.roc_curve for increasing
            true positive rates
    * auc: Float returned from sklearn.metrics.auc (Area under Curve)
    * xlim: Set upper and lower x-limits
    * ylim: Set upper and lower y-limits
    """
    my_estimators = {
              'rf': ['Random Forest', 'red']
              }

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'\
              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()

    #%%
    plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
                   xlim=(-0.01, 1.05),
                   ylim=(0.001, 1.05))

#%%
# MSI random Forest
#%%


colgen = msil.shape[1]
gx = msil.iloc[:, 1:colgen].values   # Features for training
gy = msil.iloc[:, 0].values  # Labels of training


gx_train, gx_test, gy_train, gy_test = train_test_split(gx, gy, test_size=0.3, random_state=123)

# Grid search for the best Hyperpara.
fit_rf = RandomForestClassifier(random_state=42)

np.random.seed(123)
start = time.time()

param_dist = {'max_depth': [5,6,8,10],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv=10,
                     param_grid=param_dist,
                     n_jobs = 3)

cv_rf.fit(gx_train, gy_train)
print('Best Parameters using grid search: \n',
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
#%%
fit_rf.set_params(criterion='gini',
                  max_features='auto',
                  max_depth=5,
                  bootstrap=False,
                  n_estimators=300,
                  class_weight='balanced_subsample')

fit_rf.fit(gx_train, gy_train)


#%%
def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable

    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                      variables

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
        'rf': 'estimators_',
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                             training_set,
                             class_set,
                             cv = n)
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2

#%%
cross_val_metrics(fit_rf,
                  gx_train,
                  gy_train,
                  'rf',
                  print_results=True)

#%%
predictions_rf = fit_rf.predict(gx_test)


# Func for CM
def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if len(test_class_set.shape) != len(predictions.shape) == 1:
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif test_class_set.shape != predictions.shape:
        return print('Number of values inside the Arrays are not equal to each other.\n'
                     'Please make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index=test_class_set,
                                        columns=predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb


conf_mat = create_conf_mat(gy_test, predictions_rf)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()

#%%
accuracy_rf = fit_rf.score(gx_test, gy_test)
print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_rf))

#%%
# ROC AUC
predictions_prob = fit_rf.predict_proba(gx_test)[:, 1]

fpr2, tpr2, _ = roc_curve(gy_test,
                          predictions_prob,
                          pos_label=1)

auc_rf = auc(fpr2, tpr2)


plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
                xlim=(-0.01, 1.05),
                ylim=(0.001, 1.05))



# TODOs for next week?if I still have time
# TODO feature_selection
# TODO lightGBM for this two models
# TODO try:msi with the undersampling

#%%
# Feature selection analysis by NERF

msinerf1 = flatforest(fit_rf, gx_test)

msinerf2 = nerftab(msinerf1)

msinerf3 = localnerf(msinerf2, 3)

twonets(msinerf3, "2nd_nerf_test", index1=2, index2=10)

#%%
#%%
#%%


# TODO random forest for msi x4
# TODO 1. original msi; 2. undersampling; 3. feature selection using FI; 4. feature selection using NERF
# Public zone data
#%%# data processing of the FDA challenge
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
import lightgbm as lg

# Throw data into the project
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

# gender
genl = labelEn(gender)
colgen = genl.shape[1]
featurelistgen = genl.iloc[:, 1:colgen].columns.values.tolist()  # feature name
featureindex = list(range(len(featurelistgen)))  # feature index
# MSI random Forest
#%%


colgen = msil.shape[1]
gx = msil.iloc[:, 1:colgen].values   # Features for training
gy = msil.iloc[:, 0].values  # Labels of training


gx_train, gx_test, gy_train, gy_test = train_test_split(gx, gy, test_size=0.1, random_state=123)

#%%
# random_msi_1


# Grid search for the best Hyperpara.
fit_rf = RandomForestClassifier(random_state=123)

np.random.seed(123)
start = time.time()

param_dist = {'max_depth': [5,6,8,10],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv=10,
                     param_grid=param_dist,
                     n_jobs = 3)

cv_rf.fit(gx_train, gy_train)
print('Best Parameters using grid search: \n',
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
#%%
fit_rf.set_params(criterion='gini',
                  max_features='auto',
                  max_depth=6,
                  bootstrap=False,
                  n_estimators=300,
                  class_weight='balanced_subsample')

fit_rf.fit(gx_train, gy_train)


#%%
def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable

    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                      variables

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
        'rf': 'estimators_',
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                             training_set,
                             class_set,
                             cv = n)
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2

#%%
cross_val_metrics(fit_rf,
                  gx_train,
                  gy_train,
                  'rf',
                  print_results=True)

#%%
predictions_rf = fit_rf.predict(gx_test)


# Func for CM
def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if len(test_class_set.shape) != len(predictions.shape) == 1:
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif test_class_set.shape != predictions.shape:
        return print('Number of values inside the Arrays are not equal to each other.\n'
                     'Please make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index=test_class_set,
                                        columns=predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb


conf_mat = create_conf_mat(gy_test, predictions_rf)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()

#%%
accuracy_rf = fit_rf.score(gx_test, gy_test)
print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_rf))

#%%
#%%
# Feature selection analysis by NERF

msinerf1 = flatforest(fit_rf, gx_test)

msinerf2 = nerftab(msinerf1)

msinerf3 = localnerf(msinerf2, 3)

nerflist = twonets(msinerf3, "nerflist", index1=4, index2=13)
nerflisti = list(nerflist[4].loc[:, 'feature_i'])
nerflistj = list(nerflist[4].loc[:, 'feature_j'])
nerflistall = nerflisti + nerflistj
nerflistallset = set(nerflistall)
nerflistselection = list(nerflistallset)
nerflistselection.insert(0, 0)

#%%
# Feature importance of python
feaimp = fit_rf.feature_importances_
fltable = pd.DataFrame({
    'feature name': featurelistgen,
    'score': feaimp,
    'index': featureindex
})

fltableall = fltable.sort_values('score', ascending=False)
fltableall.to_csv('feature_importance_python_original.txt', sep='\t')

originalselection = list(fltableall.iloc[0:49, 2])
originalselection.insert(0, 0)
#%%
# 3 models

msinerf = msil.iloc[:, nerflistselection]

msiorigin = msil.iloc[:, originalselection]
genderorigin = genl.iloc[:, originalselection]
colorigin = genderorigin.shape[1]

xgen = genderorigin.iloc[:, 1:colorigin].values
ygen = genderorigin.iloc[:, 0].values


colmsi = msil.shape[1]
colnerf = msinerf.shape[1]
colorigin = msiorigin.shape[1]

prot2 = prot1.iloc[:, originalselection]

xorigin = msiorigin.iloc[:, 1:colorigin].values
yorigin = msiorigin.iloc[:, 0].values

#%% First, no feature selection
for i in range(1, 100, 12):
    gx_train, gx_test, gy_train, gy_test = train_test_split(xmsi, ymsi, test_size=0.2, random_state=i)

    # Grid search for the best Hyperpara.
    fit_rf = RandomForestClassifier(random_state=123)

    np.random.seed(123)
    start = time.time()

    # param_dist = {'max_depth': [2,3,5,6,8,10],
    #               'bootstrap': [True, False],
    #               'max_features': ['auto', 'sqrt', 'log2', None],
    #               'criterion': ['gini', 'entropy']}
    #
    # cv_rf = GridSearchCV(fit_rf, cv=10,
    #                      param_grid=param_dist,
    #                      n_jobs = 3)
    #
    # cv_rf.fit(gx_train, gy_train)
    # print('Best Parameters using grid search: \n',
    #       cv_rf.best_params_)
    # end = time.time()
    # print('Time taken in grid search: {0: .2f}'.format(end - start))
    #%%
    fit_rf.set_params(criterion='gini',
                      max_features='sqrt',
                      max_depth=3,
                      bootstrap=True,
                      n_estimators=300,
                      class_weight='balanced_subsample')

    fit_rf.fit(gx_train, gy_train)


    #%%
    def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
        """
        Purpose
        ----------
        Function helps automate cross validation processes while including
        option to print metrics or store in variable

        Parameters
        ----------
        fit: Fitted model
        training_set:  Data_frame containing 80% of original dataframe
        class_set:     data_frame containing the respective target vaues
                          for the training_set
        print_results: Boolean, if true prints the metrics, else saves metrics as
                          variables

        Returns
        ----------
        scores.mean(): Float representing cross validation score
        scores.std() / 2: Float representing the standard error (derived
                    from cross validation score's standard deviation)
        """
        my_estimators = {
            'rf': 'estimators_',
        }
        try:
            # Captures whether first parameter is a model
            if not hasattr(fit, 'fit'):
                return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

            # Captures whether the model has been trained
            if not vars(fit)[my_estimators[estimator]]:
                return print("Model does not appear to be trained.")

        except KeyError as e:
            print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
    \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
            raise

        n = KFold(n_splits=6)
        scores = cross_val_score(fit,
                                 training_set,
                                 class_set,
                                 cv = n)
        if print_results:
            for i in range(0, len(scores)):
                print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
            print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
                  .format(scores.mean(), scores.std() / 2))
        else:
            return scores.mean(), scores.std() / 2

    #%%
    cross_val_metrics(fit_rf,
                      gx_train,
                      gy_train,
                      'rf',
                      print_results=True)

    #%%
    predictions_rf = fit_rf.predict(gx_test)


    # Func for CM
    def create_conf_mat(test_class_set, predictions):
        """Function returns confusion matrix comparing two arrays"""
        if len(test_class_set.shape) != len(predictions.shape) == 1:
            return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
        elif test_class_set.shape != predictions.shape:
            return print('Number of values inside the Arrays are not equal to each other.\n'
                         'Please make sure the array has the same number of instances.')
        else:
            # Set Metrics
            test_crosstb_comp = pd.crosstab(index=test_class_set,
                                            columns=predictions)
            # Changed for Future deprecation of as_matrix
            test_crosstb = test_crosstb_comp.values
            return test_crosstb


    conf_mat = create_conf_mat(gy_test, predictions_rf)
    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Actual vs. Predicted Confusion Matrix')
    plt.show()

    #%%
    accuracy_rf = fit_rf.score(gx_test, gy_test)
    print("Here is our mean accuracy on the test set:\n {0:.3f}"\
          .format(accuracy_rf))



#%%
# Second, the nerf features
for i in range(1, 100, 12):
    gx_train, gx_test, gy_train, gy_test = train_test_split(xnerf, ynerf, test_size=0.2, random_state=i)

    # Grid search for the best Hyperpara.
    fit_rf = RandomForestClassifier(random_state=123)

    np.random.seed(123)
    start = time.time()

    # param_dist = {'max_depth': [2,3,5,6,8,10],
    #               'bootstrap': [True, False],
    #               'max_features': ['auto', 'sqrt', 'log2', None],
    #               'criterion': ['gini', 'entropy']}
    #
    # cv_rf = GridSearchCV(fit_rf, cv=10,
    #                      param_grid=param_dist,
    #                      n_jobs = 3)
    #
    # cv_rf.fit(gx_train, gy_train)
    # print('Best Parameters using grid search: \n',
    #       cv_rf.best_params_)
    # end = time.time()
    # print('Time taken in grid search: {0: .2f}'.format(end - start))
    #%%
    fit_rf.set_params(criterion='gini',
                      max_features=None,
                      max_depth=3,
                      bootstrap=True,
                      n_estimators=600,
                      class_weight='balanced_subsample')

    fit_rf.fit(gx_train, gy_train)


    #%%
    def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
        """
        Purpose
        ----------
        Function helps automate cross validation processes while including
        option to print metrics or store in variable

        Parameters
        ----------
        fit: Fitted model
        training_set:  Data_frame containing 80% of original dataframe
        class_set:     data_frame containing the respective target vaues
                          for the training_set
        print_results: Boolean, if true prints the metrics, else saves metrics as
                          variables

        Returns
        ----------
        scores.mean(): Float representing cross validation score
        scores.std() / 2: Float representing the standard error (derived
                    from cross validation score's standard deviation)
        """
        my_estimators = {
            'rf': 'estimators_',
        }
        try:
            # Captures whether first parameter is a model
            if not hasattr(fit, 'fit'):
                return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

            # Captures whether the model has been trained
            if not vars(fit)[my_estimators[estimator]]:
                return print("Model does not appear to be trained.")

        except KeyError as e:
            print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
    \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
            raise

        n = KFold(n_splits=6)
        scores = cross_val_score(fit,
                                 training_set,
                                 class_set,
                                 cv = n)
        if print_results:
            for i in range(0, len(scores)):
                print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
            print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
                  .format(scores.mean(), scores.std() / 2))
        else:
            return scores.mean(), scores.std() / 2

    #%%
    cross_val_metrics(fit_rf,
                      gx_train,
                      gy_train,
                      'rf',
                      print_results=True)

    #%%
    predictions_rf = fit_rf.predict(gx_test)


    # Func for CM
    def create_conf_mat(test_class_set, predictions):
        """Function returns confusion matrix comparing two arrays"""
        if len(test_class_set.shape) != len(predictions.shape) == 1:
            return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
        elif test_class_set.shape != predictions.shape:
            return print('Number of values inside the Arrays are not equal to each other.\n'
                         'Please make sure the array has the same number of instances.')
        else:
            # Set Metrics
            test_crosstb_comp = pd.crosstab(index=test_class_set,
                                            columns=predictions)
            # Changed for Future deprecation of as_matrix
            test_crosstb = test_crosstb_comp.values
            return test_crosstb


    conf_mat = create_conf_mat(gy_test, predictions_rf)
    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Actual vs. Predicted Confusion Matrix')
    plt.show()

    #%%
    accuracy_rf = fit_rf.score(gx_test, gy_test)
    print("Here is our mean accuracy on the test set:\n {0:.3f}"\
          .format(accuracy_rf))



#%% last: the original fi
for i in range(1, 100, 12):
    gx_train, gx_test, gy_train, gy_test = train_test_split(xorigin, yorigin, test_size=0.2, random_state=i)

    # Grid search for the best Hyperpara.
    fit_rf = RandomForestClassifier(random_state=123)

    np.random.seed(123)
    start = time.time()
    #
    # param_dist = {'max_depth': [2,3,5,6,8,10],
    #               'bootstrap': [True, False],
    #               'max_features': ['auto', 'sqrt', 'log2', None],
    #               'criterion': ['gini', 'entropy']}
    #
    # cv_rf = GridSearchCV(fit_rf, cv=10,
    #                      param_grid=param_dist,
    #                      n_jobs = 3)
    #
    # cv_rf.fit(gx_train, gy_train)
    # print('Best Parameters using grid search: \n',
    #       cv_rf.best_params_)
    # end = time.time()
    # print('Time taken in grid search: {0: .2f}'.format(end - start))
    #%%
    fit_rf.set_params(criterion='entropy',
                      max_features=None,
                      max_depth=5,
                      bootstrap=True,
                      n_estimators=300,
                      class_weight='balanced_subsample')

    fit_rf.fit(gx_train, gy_train)


    #%%
    def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
        """
        Purpose
        ----------
        Function helps automate cross validation processes while including
        option to print metrics or store in variable

        Parameters
        ----------
        fit: Fitted model
        training_set:  Data_frame containing 80% of original dataframe
        class_set:     data_frame containing the respective target vaues
                          for the training_set
        print_results: Boolean, if true prints the metrics, else saves metrics as
                          variables

        Returns
        ----------
        scores.mean(): Float representing cross validation score
        scores.std() / 2: Float representing the standard error (derived
                    from cross validation score's standard deviation)
        """
        my_estimators = {
            'rf': 'estimators_',
        }
        try:
            # Captures whether first parameter is a model
            if not hasattr(fit, 'fit'):
                return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

            # Captures whether the model has been trained
            if not vars(fit)[my_estimators[estimator]]:
                return print("Model does not appear to be trained.")

        except KeyError as e:
            print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
    \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
            raise

        n = KFold(n_splits=6)
        scores = cross_val_score(fit,
                                 training_set,
                                 class_set,
                                 cv = n)
        if print_results:
            for i in range(0, len(scores)):
                print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
            print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
                  .format(scores.mean(), scores.std() / 2))
        else:
            return scores.mean(), scores.std() / 2

    #%%
    cross_val_metrics(fit_rf,
                      gx_train,
                      gy_train,
                      'rf',
                      print_results=True)

    #%%
    predictions_rf = fit_rf.predict(gx_test)


    # Func for CM
    def create_conf_mat(test_class_set, predictions):
        """Function returns confusion matrix comparing two arrays"""
        if len(test_class_set.shape) != len(predictions.shape) == 1:
            return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
        elif test_class_set.shape != predictions.shape:
            return print('Number of values inside the Arrays are not equal to each other.\n'
                         'Please make sure the array has the same number of instances.')
        else:
            # Set Metrics
            test_crosstb_comp = pd.crosstab(index=test_class_set,
                                            columns=predictions)
            # Changed for Future deprecation of as_matrix
            test_crosstb = test_crosstb_comp.values
            return test_crosstb


    conf_mat = create_conf_mat(gy_test, predictions_rf)
    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Actual vs. Predicted Confusion Matrix')
    plt.show()

    #%%
    accuracy_rf = fit_rf.score(gx_test, gy_test)
    print("Here is our mean accuracy on the test set:\n {0:.3f}"\
          .format(accuracy_rf))
#%% Output of FI random forest gender




    fit_rf.fit(xorigin, yorigin)

    predictions_rf_msi = fit_rf.predict(prot2)
    predictions_rf_train_msi = fit_rf.predict(xorigin)

    np.savetxt("msitest.txt", predictions_rf_msi, delimiter='\t')
    np.savetxt("msitrain.txt", predictions_rf_train_msi, delimiter='\t')



    fit_rf.fit(xgen, ygen)

    predictions_rf_gen = fit_rf.predict(prot2)
    predictions_rf_train_gen = fit_rf.predict(xgen)

    np.savetxt("gentest.txt", predictions_rf_gen, delimiter='\t')
    np.savetxt("gentrain.txt", predictions_rf_train_gen, delimiter='\t')