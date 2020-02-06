import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

digits = load_digits()

X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rforest = RandomForestClassifier(random_state=42)
rforest.fit(X_train, y_train)

print(rforest.predict(digits.data[:10]))
