from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn import model_selection
import pandas as pd
import numpy as np
import os


protein_gender = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/sc2/genderprotein.csv', sep=',')
protein_msi = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/sc2/msiprotein.csv', sep=',')

factorg = pd.factorize(protein_gender['gender'])
protein_gender.gender = factorg[0]
definitiong = factorg[1]

factorm = pd.factorize(protein_msi['msi'])
protein_msi.msi = factorm[0]
definitionm = factorm[1]

xg = protein_gender.iloc[:, 0:4117].values   # Features for training
yg = protein_gender.iloc[:, 4118].values

xm = protein_msi.iloc[:, 0:4117].values   # Features for training
ym = protein_msi.iloc[:, 4118].values

xg_train, xg_test, yg_train, yg_test = model_selection.train_test_split(xg, yg, test_size=0.3, random_state=123)
xm_train, xm_test, ym_train, ym_test = model_selection.train_test_split(xm, ym, test_size=0.3, random_state=123)

#%% Gender protein top200
# RFE
estimator = LinearSVC()
selectorg = RFE(estimator=estimator, n_features_to_select=200)
xg_t = selectorg.fit_transform(xg, yg)

# New sets
xg_train_t, xg_test_t, yg_train_t, yg_test_t = model_selection.train_test_split(xg_t, yg, test_size=0.3, random_state=123)

clf = LinearSVC()
clf_t = LinearSVC()
clf.fit(xg_train, yg_train)
clf_t.fit(xg_train_t, yg_train_t)
print("Original DataSet: test score=%s" % (clf.score(xg_test, yg_test)))
print("Selected DataSet: test score=%s" % (clf_t.score(xg_test_t, yg_test_t)))

selectorg.support_

feature_selected_index = np.where(selectorg.support_)

featurelistpro = protein_gender.iloc[:, 0:4117].columns.values.tolist()
featuretop200 = np.array(featurelistpro)[feature_selected_index[0]]
featuretop200 = featuretop200.tolist()
top200_gen_pro = pd.DataFrame(
    {'features': featuretop200}
)
top200_gen_pro.to_csv(os.getcwd()+'/output_top200_RFE_Gender_protein.csv', sep='\t')


#%% msi protein top200
# RFE
estimator = LinearSVC()
selectorm = RFE(estimator=estimator, n_features_to_select=200)
xm_t = selectorm.fit_transform(xm, ym)

# New sets
xm_train_t, xm_test_t, ym_train_t, ym_test_t = model_selection.train_test_split(xm_t, ym, test_size=0.3, random_state=123)

clf = LinearSVC()
clf_t = LinearSVC()
clf.fit(xm_train, ym_train)
clf_t.fit(xm_train_t, ym_train_t)
print("Original DataSet: test score=%s" % (clf.score(xm_test, ym_test)))
print("Selected DataSet: test score=%s" % (clf_t.score(xm_test_t, ym_test_t)))


feature_selected_index = np.where(selectorm.support_)

featurelistmsi = protein_msi.iloc[:, 0:4117].columns.values.tolist()
featuretop200m = np.array(featurelistmsi)[feature_selected_index[0]]
featuretop200m = featuretop200m.tolist()
top200_msi_pro = pd.DataFrame(
    {'features': featuretop200m}
)
top200_msi_pro.to_csv(os.getcwd()+'/output_top200_RFE_msi_protein.csv', sep='\t')

#%%
# RNA part

rna_gender = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/sc2/genderrna.csv', sep=',')
rna_msi = pd.read_table('P:/VM/precisionFDA/2018-mislabelingCorrectionChallenge/sc2/msirna.csv', sep=',')

factorg = pd.factorize(rna_gender['gender'])
rna_gender.gender = factorg[0]
definitiong = factorg[1]

factorm = pd.factorize(rna_msi['msi'])
rna_msi.msi = factorm[0]
definitionm = factorm[1]

xg = rna_gender.iloc[:, 0:17446].values   # Features for training
yg = rna_gender.iloc[:, 17447].values

xm = rna_msi.iloc[:, 0:17446].values   # Features for training
ym = rna_msi.iloc[:, 17447].values

xg_train, xg_test, yg_train, yg_test = model_selection.train_test_split(xg, yg, test_size=0.3, random_state=123)
xm_train, xm_test, ym_train, ym_test = model_selection.train_test_split(xm, ym, test_size=0.3, random_state=123)

#%% Gender rna top200
# RFE
estimator = LinearSVC()
selectorg = RFE(estimator=estimator, n_features_to_select=200)
xg_t = selectorg.fit_transform(xg, yg)

# New sets
xg_train_t, xg_test_t, yg_train_t, yg_test_t = model_selection.train_test_split(xg_t, yg, test_size=0.3, random_state=123)

clf = LinearSVC()
clf_t = LinearSVC()
clf.fit(xg_train, yg_train)
clf_t.fit(xg_train_t, yg_train_t)
print("Original DataSet: test score=%s" % (clf.score(xg_test, yg_test)))
print("Selected DataSet: test score=%s" % (clf_t.score(xg_test_t, yg_test_t)))

selectorg.support_

feature_selected_index = np.where(selectorg.support_)

featurelistpro = rna_gender.iloc[:, 0:17446].columns.values.tolist()
featuretop200 = np.array(featurelistpro)[feature_selected_index[0]]
featuretop200 = featuretop200.tolist()
top200_gen_pro = pd.DataFrame(
    {'features': featuretop200}
)
top200_gen_pro.to_csv(os.getcwd()+'/output_top200_RFE_Gender_rna.csv', sep='\t')


#%% msi protein top200
# RFE
estimator = LinearSVC()
selectorm = RFE(estimator=estimator, n_features_to_select=200)
xm_t = selectorm.fit_transform(xm, ym)

# New sets
xm_train_t, xm_test_t, ym_train_t, ym_test_t = model_selection.train_test_split(xm_t, ym, test_size=0.3, random_state=123)

clf = LinearSVC()
clf_t = LinearSVC()
clf.fit(xm_train, ym_train)
clf_t.fit(xm_train_t, ym_train_t)
print("Original DataSet: test score=%s" % (clf.score(xm_test, ym_test)))
print("Selected DataSet: test score=%s" % (clf_t.score(xm_test_t, ym_test_t)))


feature_selected_index = np.where(selectorm.support_)

featurelistmsi = rna_msi.iloc[:, 0:17336].columns.values.tolist()
featuretop200m = np.array(featurelistmsi)[feature_selected_index[0]]
featuretop200m = featuretop200m.tolist()
top200_msi_pro = pd.DataFrame(
    {'features': featuretop200m}
)
top200_msi_pro.to_csv(os.getcwd()+'/output_top200_RFE_msi_rna.csv', sep='\t')