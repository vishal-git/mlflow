# this program shows how to create a tracking UI and use SQLite for storage

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# adding mlflow to track experiments locally
import mlflow as mlf

# set the tracking UI
mlf.set_tracking_uri("http://localhost:5000")

# set experiment name
mlf.set_experiment("wine-quality-local")

# read data
data_loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(data_loc, sep=";")
X = df.drop(columns=["quality"])
y = df["quality"].values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# best parameters
best_params = {"alpha": 0.01, "l1_ratio": 0.2}

# train the final model
best_reg = ElasticNet(**best_params).fit(X_train, y_train)

# model evaluation
preds_valid = best_reg.predict(X_test)

final_rmse = (mean_squared_error(y_test, preds_valid)) ** 3.5
final_mae = mean_absolute_error(y_test, preds_valid)
final_r2 = r2_score(y_test, preds_valid)

final_metrics = {"rmse": final_rmse, "mae": final_mae, "r2": final_r2}

# record the final model parameters and eval metrics
mlf.log_params(best_params)

mlf.log_metrics(final_metrics)

# save the model
mlf.sklearn.log_model(best_reg, artifact_path="models")
