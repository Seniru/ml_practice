import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

diabetes_dataset = load_diabetes()

diabetes = pd.DataFrame(
    data = np.c_[diabetes_dataset["data"], diabetes_dataset["target"]],
    columns = diabetes_dataset["feature_names"] + ["target"]
)

X = diabetes[["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]]
y = diabetes["target"]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.3)

def score_model(train_X, test_X, train_y, test_y, leaf_nodes):
	sample_model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes)
	sample_model.fit(train_X, train_y)
	return mean_absolute_error(sample_model.predict(test_X), test_y)

def get_optimum_leaves(train_X, test_X, train_y, test_y):
	best_node = None
	least_error = np.inf
	for nodes in range(2, 10):
		score = score_model(train_X, test_X, train_y, test_y, nodes)
		print("Nodes: {} -> MAE: {}", nodes, score)
		if score < least_error:
			least_error = score
			best_node = nodes
	return best_node

# Uncomment the following line to find the optimum node programatically
#print("Optimum number of nodes: {}", get_optimum_leaves(train_X, test_X, train_y, test_y))

# Optimum: 5 with a MAE of 45.2768937236808

model = DecisionTreeRegressor(max_leaf_nodes=5)

print("Training the data...")
model.fit(train_X, train_y)
print("Predictions: ", end="")
predictions = model.predict(test_X)
