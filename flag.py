'''
==========================
Playing with Flags Dataset
==========================
'''

import csv
import numpy as np
from collections import Counter
from sklearn import preprocessing
from common.fn import *

with open('datasets/flag.data', 'r') as input_file:
	csv_reader = csv.reader(input_file, delimiter=',')

	raw_data = []

	for line in csv_reader:
		raw_data.append(line)

data = []
target = []

for sample in raw_data:
	data.append(sample[7:])
	target.append(sample[6])

X = np.array(data)
y = np.array(target)

le = preprocessing.LabelEncoder()
X[:, 10] = le.fit_transform(X[:, 10])
X[:, -2] = le.fit_transform(X[:, -2])
X[:, -1] = le.fit_transform(X[:, -1])

X = X.astype(int)
y = y.astype(int)

# Visualise data