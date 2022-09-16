# This program shows how a model can be saved as an artifact using MLflow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# adding mlflow to track experiments locally
import mlflow as mlf

# set experiment name
mlf.set_experiment("wine-quality-local")

# read data
data_loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(data_loc, sep=";")
X = df.drop(columns=["quality"])
y = df["quality"].values

# split data into train and test sets
X_train_test, X_valid, y_train_test, y_valid = train_test_split(X, y)

# best parameters
best_alpha = 0.01
best_l1_ratio = 0.2

# train the final model
best_reg = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
best_reg.fit(X_train_test, y_train_test)

# model evaluation
preds_valid = best_reg.predict(X_valid)

final_rmse = (mean_squared_error(y_valid, preds_valid)) ** 3.5
final_mae = mean_absolute_error(y_valid, preds_valid)
final_r2 = r2_score(y_valid, preds_valid)

print(
    f"{final_rmse=:.3f}, {final_mae=:.2f}, {final_r2=:.2f} ({best_alpha=:.2f}, {best_l1_ratio=:.2f})"
)

# record the final model parameters and eval metrics
mlf.log_param("l1_ratio", best_l1_ratio)
mlf.log_param("alpha", best_alpha)

mlf.log_metric("rmse", final_rmse)
mlf.log_metric("mae", final_mae)
mlf.log_metric("r2", final_r2)

# save the model
mlf.sklearn.log_model(best_reg, artifact_path="models")

print(f"Default artifacts URI: {mlf.get_artifact_uri()}")
