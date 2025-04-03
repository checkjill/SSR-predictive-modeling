import pandas
from pandas import read_csv
import numpy as np
from numpy import round
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# READ IN DATA AND SPLIT

xdat = read_csv('/mnt/home/checkjil/white_mold_tpot_raw/raw_data/x_dat_LX_HPCC.csv')
ydat = read_csv('/mnt/home/checkjil/white_mold_tpot_raw/raw_data/y_dat_LX_HPCC.csv')
ydat = LabelEncoder().fit_transform(ydat.astype('str'))

x_train, x_test, y_train, y_test = train_test_split(xdat, ydat, train_size=0.80, test_size=0.20, random_state=42)

# DEFINE CROSS VALIDATION SCHEME

cv = RepeatedStratifiedKFold(n_splits=5, random_state = 42)

# DECLARE AND FIT MODEL ON TRAINING DATA

model = TPOTClassifier(generations = 5, verbosity = 2, scoring = 'accuracy', cv=cv, random_state = 42, memory = 'auto', warm_start = True)

ypred = model.fit(x_train, y_train).predict(x_test)

# EVALUATE ON TEST DATA

cm = confusion_matrix(y_test, ypred)
tp = cm[0, 0]
fp = cm[0, 1]
tn = cm[1, 1]
fn = cm[1, 0]

print('Accuracy', np.round((tp+tn)/(tp+tn+fp+fn), 2))
print('Sensitivity', np.round(tp/(tp+fn), 2))
print('Specficity', np.round(tn/(tn+fp), 2))
print('Precision', np.round(tp/(tp+fp), 2))

# EVALUATE BY AUROC

auroc_score = sklearn.metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
print("auroc score = ", auroc_score)

# EXPORT MODEL

model.export('/mnt/home/checkjil/white_mold_tpot_raw/outputs/tpot_best_model_LX.py')