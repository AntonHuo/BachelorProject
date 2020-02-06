import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('testdata_audit_analytics.csv',low_memory=False)

col_names = data.columns
col_list = col_names.tolist()
#Get all columns except years, rics and effect
keys_X = []
for x in range(3,837):
    keys_X.append(col_list[x])
# get all datas except years,rics and labels
X = data[keys_X]
#Delete all boolean type or object type(datum) features
X = X.select_dtypes(exclude=['bool','object'])
#Replace all the NAN with mean
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)
#Transform the data as data frame
X = pd.DataFrame(X)
print(X)

X.info()
#get all the labels
y1 = data[['effect']]


########################################################################################################################
#    prediction With FeatureBoost
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y1.values.ravel())
print('feature_importances of all')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True)
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y1 = pd.DataFrame(y1)

sel_X = pd.concat([y1,select_X],axis=1)
sel_X.info()
print("selX")
print(sel_X)

#get all datas with label 'effect' is positive
X_eff1 = sel_X.loc[sel_X["effect"] == 'positive']
#get all datas with label 'effect' is negative
X_eff0 = sel_X.loc[sel_X["effect"] == 'negative']

#get all data with label "effect" is positive and dorp column'effect'
X_eff_1 = X_eff1.iloc[:, 1:]
print("X_eff_1##########################################")
print(X_eff_1)
#get all data with label "effect" is negative and dorp column'effect'
X_eff_0 = X_eff0.iloc[:, 1:]
print("X_eff_0###########################################")
print(X_eff_0)

X_eff_1.info()
X_eff_0.info()

#set training data
X0_train = X_eff_0.loc[0:10250]
print("X0_train############################################")
print(X0_train)
#set test data
X0_test = X_eff_0.loc[10250:]
print("X0_test############################################")
print(X0_test)

clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)


y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_eff_1)


print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))

########################################################################################################################
#    prediction without featureboost
########################################################################################################################

X = pd.concat([y1,X],axis=1)

print("X")
print(X)

#get all datas with label "effect" is positive
X_eff1 = X.loc[X['effect'] == 'positive']
#get all datas with label "effect" is negative
X_eff0 = X.loc[X['effect'] == 'negative']

print(X_eff1)
X_eff1.info()
print(X_eff0)
X_eff0.info()
#get all data with label "effect" is positive and dorp column'effect''ric''year'
X_eff_1 = X_eff1.iloc[:, 3:]
print(X_eff_1)
#get all data with label "effect" is negative and dorp column'effect''ric''year'
X_eff_0 = X_eff0.iloc[:, 3:]
print(X_eff_0)
#set training data
X0_train = X_eff_0.loc[0:10250]
print(X0_train)
#set test data
X0_test = X_eff_0.loc[10250:]
print(X0_test)

clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)


y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_eff_1)


print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))