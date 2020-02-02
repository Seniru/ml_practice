#Loading the necessary modules
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

#Loading the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(
    data= np.c_[iris['data'],
    iris['target']],
    columns= iris['feature_names'] + ['target']

)

neighbors = 4 #Best number of neigbors

#Setting our feature matrix
X = iris_df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
y = iris_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def score(X_train, X_test, y_train, y_test, neighbors):
    sample_model = KNeighborsRegressor(n_neighbors=neighbors)
    sample_model.fit(X_train, y_train)
    return mean_absolute_error(sample_model.predict(X_test), y_test)

def getOptimumNeighbors(X_train, X_test, y_train, y_test):
    for neighbors in range(3, 15):
        print("Nodes: {} -> {}".format(neighbors, score(X_train, X_test, y_train, y_test, neighbors)))

#Uncomment this to check which number of neighbors perform best
#getOptimumNeighbors(X_train, X_test, y_train, y_test)

model = KNeighborsRegressor(n_neighbors=3)
print("Fitting the train data...")
model.fit(X_train, y_train)
print("Predicting the species...")
predictions = model.predict(X_test)
print(list(map(lambda x: iris.target_names[int(round(x))], predictions)))
print("Mean absolute error: ", end="")
print(mean_absolute_error(predictions, y_test))