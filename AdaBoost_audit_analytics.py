import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier as adaBoost

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
#    prediction with FeatureBoost
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

y1.info()
print(y1)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(select_X, y1, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'algorithm':["SAMME","SAMME.R"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy with featureboost is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

########################################################################################################################
#     prediction without FeatureBoost
########################################################################################################################

y1.info()
print(y1)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'algorithm':["SAMME","SAMME.R"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy without featureboost is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))