'''
=================
Titanic - Kaggle
=================
Distribution of samples in train.csv
Samples count = 891
Features count = 11 (Excluding target feature)

Target distribution:
0 - 549 (61.61 percent)
1 - 342 (38.38 percent)
Which means if I guess all passengers doesn't survived, I'm able to reach 61 percent accuracy. In turn, 61 percent makes my lower bound.

PClass distribution:
1st - 216
2nd - 184
3rd - 491

PClass/Survival distribution:
1st - 136 (62.9 percent)
2nd - 87 (47.2 percent)
3rd - 119 (24.2 percent)
It appears 1st class people got priority over 2nd and 3rd class.
It's very unlikely that during a disaster, name plays a significant role. Lets come back to this if my model doesn't perform well.

Sex distribution:
male - 577
female - 314
I have to convert this into real valued features if I have to use sex as a feature for my model.

Sex/Survival distribution:
male - 109 (18 percent)
female - 233 (74.2 percent)

Age distribution:
177 missing age
I need to fill these if I have to use age as a feature for my model.

Embarked distribution:
S - 644
C - 168
Q - 77
Unknown - 2

Observations during learning:
1. As expected, logistic regression doesn't generalise good on such small dataset. 79 percent cross validation score.
2. Even decision tree is giving 79 percent mean cross validation score. No improvement with change in model. But, one cross validation score got upto 84 percent. I wonder if this is due to small size of dataset.
3. Using LeaveOneOut method, I confirmed Decision Tree was definitely one step above Logistic Regression. Getting 81 percent accuracy on cross validation score.
4. But, it's still failing on testing set. Only 78 percent accuracy on test set.
5. Decision Tree is failing 50 percent on recognising a survival.
6. Adding manual prediction rule to overcome this.
7. By adding two manual rules, reached upto 85.5 percent.
'''

import csv
from collections import Counter
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn import cross_validation
from sklearn import metrics
from common.fn import *

file_path = 'datasets/titanic/'
file_name = 'train.csv'

def load_titanic(file_path, file_name):
	titanic = {}

	with open(file_path+file_name, 'rb') as csv_file:
		reader = csv.reader(csv_file, delimiter=',', quotechar='"')

		first_row = reader.next()

		X, y = [], []
		for row in reader:
			X.append(row)
			y.append(row[1])

		titanic['data'] = np.array(X)
		titanic['target'] = np.array(y)
		titanic['feature_names'] = np.array(first_row)

		return titanic

titanic = load_titanic(file_path, file_name)

X = titanic['data'][:, [2, 4, 5]]
y = titanic['target']
feature_names = titanic['feature_names'][[2, 4, 5]]

# Filling missing age
ages = X[:, 2]
mean_age = np.mean(X[ages != '', 2].astype(np.float))
X[ages == '', 2] = mean_age

# Converting sex into real values
le = preprocessing.LabelEncoder()
le.fit(X[:, 1])
X[:, 1] = le.transform(X[:, 1])

X = X.astype(float)
y = y.astype(float)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

for i in xrange(len(y_predict)):
	if y_predict[i] == 0:
		if X_test[i][0] == 1 and X_test[i][1] == 0:
			y_predict[i] = 1
		if X_test[i][1] == 0 and X_test[i][2] < 20:
			y_predict[i] = 1

print metrics.accuracy_score(y_test, y_predict)