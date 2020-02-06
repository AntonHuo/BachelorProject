import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as kNN

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
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for all is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

########################################################################################################################
#     prediction for label "relevant"
########################################################################################################################

y2.info()
print(y2)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for relevant is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

########################################################################################################################
#     prediction for label"relevant5%"
########################################################################################################################

y3.info()
print(y3)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for relevant5% is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))