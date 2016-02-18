'''
=========================
Digit Recognizer - Kaggle
=========================
Length of train.csv is 42k, which took 47.1s to load. Hence, creating a train_mini.csv taking first 5%(2.1k) of the data.
Length of train_mini.csv is 2.1k, which took 4.6s to load.
Instead of creating a train_mini.csv, I'm using max_rows param of genfromtxt function to read 5 percent of the data. 

Distribution of digits in train.csv
{
	0: 4132,
    1: 4684,
    2: 4177,
    3: 4351,
    4: 4072,
    5: 3795,
    6: 4137,
    7: 4401,
    8: 4063,
    9: 4188
}

Note: Until and unless, I see an error graph with error decrease with increase in data size, I'm not gonna fit my model with whole dataset.

Observations during learning on (5 percent dataset):
1. Logistic Regression gives training score - 100 percent, but testing scorey - 83.6 percent on testing set in 5.8s. Pretty Impressive! Seems like my model is overfitting. Regularisation to the rescue!
2. Failure! It's been 10 minutes, logistic regression on the whole train.csv is running, and I forgot to turn on the verbose.
3. Using GridSearchCV, found the optimal hyperparameters for Logistic Regression, and got training score - 98.09 percent and testing score - 88.38 percent in 4s.
4. Plotting learning curve. 
5. Score of logistic regression is almost saturated at 90%.
6. Support Vector performance with default params. Training Score - 100 percent, Testing Score - 10 percent in 11.9s. High Variance alert!
7. Using GridSearchCV, found the optimal hyperparameters for SVM, but still getting 89 percent testing score in 6.4s.
8. By plotting the learning curve of SVM, it seems the curve is going upwards.
9. With 10 percent data, getting only 91 percent score on test set.
10. Using Random Forest, learning curve is increasing with data size. Reach upto 92 percent. More than anything seen so far. Still going up. Maybe this is it!
11. Using GridSearchCV, found the optimal n_estimators for Random Forest.
12. With 20 percent of dataset, getting 94 percent testing accuracy with Random Forest.
13. Winner! Learning Curve on 20 percent dataset is still going up.
14. Saving the model on disk for future use.
15. With full train.csv dataset, getting training score 99.9 percent and testing score of 95.8 percent. Also, saved the model on disk.
16. Using Extra Tree Classifier, I'm able to reach 96.37 percent testing score.
17. Size of sample (# of features) is 784, so reducing the size of features to 10 percent (79). Performance is reduced to 86 percent. Perhaps with less feature size, my model is not generalising.
18. Increasing the feature size by 40 percent of total. Performance increased to 96 percent.
19. Lets make it to 50 percent of total because majorly my model is unable to classify among 4 and 9.
20. Confusion between 4 and 9 is reduced to less than 10. But, overall performance remains the same.
21. Performance decreased by going up to 55 percent of total feature count.
22. 53 percent of total feature count seems to be having highest performance.
23. Using 53 percent of total feature size, I'm still getting 96.28 percent testing score.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import feature_selection
from common.fn import *

def load_digits(file_name, file_path = 'datasets/digits/', max_rows=None):
	digits = {}

	raw_data = np.genfromtxt(file_path+file_name, delimiter=',', skip_header=1, max_rows=max_rows, dtype=int)
	digits['data'] = raw_data[:, 1:]
	digits['target'] = raw_data[:, 0]

	return digits

def draw_digit(digit_data, target):
	plt.title(target)	

	# cmap values plt.cm.bone, plt.cm.gray_r
	plt.imshow(np.reshape(digit_data, (28, 28)), cmap=plt.cm.gray_r)

	plt.xticks(())
	plt.yticks(())
	plt.show()

digits = load_digits('train.csv')

X = digits['data']
y = digits['target']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10, random_state=42)

selector = feature_selection.SelectPercentile(feature_selection.chi2, 53)
selector.fit(X_train, y_train)
X_train_reduced = selector.transform(X_train)
X_test_reduced = selector.transform(X_test)
# '''
clf = ensemble.ExtraTreesClassifier(n_estimators=37, n_jobs=-1)
clf.fit(X_train_reduced, y_train)
joblib.dump(clf, 'datasets/digits/model/extra_trees.pkl')
# '''
# clf = joblib.load('datasets/digits/model/extra_trees.pkl')

print clf.score(X_test_reduced, y_test)
exit()
print metrics.classification_report(y_test, y_predict)
print metrics.confusion_matrix(y_test, y_predict)