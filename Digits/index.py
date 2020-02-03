import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

digits = load_digits()

X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([X_train[0]]))

for data, target in zip(X_train[:15], y_train[:15]):
	print("Predicted: {}, Actual: {} ({})".format(sgd_clf.predict([data])[0], target == 5, target))

