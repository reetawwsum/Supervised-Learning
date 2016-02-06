'''
===========================
Playing with Boston Dataset
===========================
Note: First feature of sample 446 was different in downloaded dataset. I altered it.
'''

import numpy as np
from sklearn import datasets

def load_boston(filepath = 'datasets/boston.data'):
	boston = {}

	features = [float(feature.strip()) for feature in open(filepath).read().split() if feature and feature.strip() != '\\']
	samples = np.reshape(features, (506, 14))
	boston['data'] = samples[:, :-1]
	boston['target'] = samples[:, -1]
	boston['feature_names'] = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

	return boston

boston = load_boston()

X = boston['data']
y = boston['target']
feature_names = boston['feature_names']

'''
print 'Matching the downloaded boston dataset with inbuilt boston dataset in scikit-learn'
inbuilt_boston = datasets.load_boston()
inbuilt_X = inbuilt_boston.data
inbuilt_y = inbuilt_boston.target
inbuilt_feature_names = inbuilt_boston.feature_names

print np.array_equal(X, inbuilt_X)
print np.array_equal(y, inbuilt_y)
print np.array_equal(feature_names, inbuilt_feature_names)
print 'Turned out, first feature of sample 446 was different in both the dataset. I altered it in downloaded dataset.' 
'''