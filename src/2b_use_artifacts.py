# the code below shows how to make predictions using a logged model.

import mlflow

logged_model = "runs:/d99535841ef24178bfcf8932dccd5011/models"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

# read data
data_loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(data_loc, sep=";", nrows=100)
X = df.drop(columns=["quality"])

preds = loaded_model.predict(X)
print(preds[:10])
