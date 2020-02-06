import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('testdateimitlabels.csv',low_memory=False)

col_names = data.columns
col_list = col_names.tolist()
#Get all columns except years, rics and labels(all, relevant, relevant5%)
keys_X = []
for x in range(5,839):
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
y1 = data[['all']]
y2 = data[['relevant']]
y3 = data[['relevant5%']]


########################################################################################################################
#    prediction for label "all"
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

#get all datas with label "all" is 1
X_all1 = sel_X.loc[sel_X["all"] == 1]
#get all datas with label "all" is 0
X_all0 = sel_X.loc[sel_X["all"] == 0]

#get all data with label "all" is 1 except label
X_all_1 = X_all1.iloc[:, 1:]
print("X_all_1##########################################")
print(X_all_1)
#get all data with label "all" is 0 except label
X_all_0 = X_all0.iloc[:, 1:]
print("X_all_0###########################################")
print(X_all_0)

X_all_1.info()
X_all_0.info()

#set training data
X0_train = X_all_0.loc[0:109196]
print("X0_train############################################")
print(X0_train)
#set test data
X0_test = X_all_0.loc[109196:]
print("X0_test############################################")
print(X0_test)

clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_all_1)


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


# the accuracy for normal data
#print("Accuracy of all 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])


# the accuracy for outliers
#print("Accuracy of all 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

########################################################################################################################
#     prediction for label "relevant"
########################################################################################################################


#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y2.values.ravel())
print('feature_importances of relevant')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True)
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y2 = pd.DataFrame(y2)

sel_X = pd.concat([y2,select_X],axis=1)

#get all datas with label "relevant" is 1
X_rel1 = sel_X.loc[sel_X["relevant"] == 1]
#get all datas with label "relevant" is 0
X_rel0 = sel_X.loc[sel_X["relevant"] == 0]

#get all data with label "relevant" is 1 except label
X_rel_1 = X_rel1.iloc[:, 1:]
print(X_rel_1)
#get all data with label "relevant" is 0 except label
X_rel_0 = X_rel0.iloc[:, 1:]
print(X_rel_0)

print(X_rel_1)
X_rel1.info()
print(X_rel_0)
X_rel0.info()

#set training data
X0_train = X_rel_0.loc[0:109196]
print(X0_train)
#set test data
X0_test = X_rel_0.loc[109196:]
print(X0_test)

clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_rel_1)


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

# the accuracy for normal data
#print("Accuracy of relevant 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])


# the accuracy for outliers
#print("Accuracy of relevant 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

########################################################################################################################
#     prediction for label"relevant5%"
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y3.values.ravel())
print('feature_importances of relevant5%')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True)
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y3 = pd.DataFrame(y3)

sel_X = pd.concat([y3,select_X],axis=1)

#get all datas with label "relevant5%" is 1
X_5rel1 = sel_X.loc[sel_X["relevant5%"] == 1]
#get all datas with label "relevant5%" is 0
X_5rel0 = sel_X.loc[sel_X["relevant5%"] == 0]

#get all data with label "relevant5%" is 1 except label
X_5rel_1 = X_5rel1.iloc[:, 1:]
print(X_5rel_1)
#get all data with label "relevant5%" is 0 except label
X_5rel_0 = X_5rel0.iloc[:, 1:]
print(X_5rel_0)

print(X_5rel_1)
X_5rel1.info()
print(X_5rel_0)
X_5rel0.info()

#set training data
X0_train = X_5rel_0.loc[0:109196]
print(X0_train)
#set test data
X0_test = X_5rel_0.loc[109196:]
print(X0_test)

clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_5rel_1)

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






# the accuracy for normal data
#print("Accuracy of relevant5% 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])

# the accuracy for outliers
#print("Accuracy of relevant5% 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])