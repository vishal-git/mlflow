#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow as mlf

import os

os.environ["AWS_PROFILE"] = "derive"  # change it to your local aws profile

# insert the public DNS of the EC2 instance below
TRACKING_SERVER_HOST = "ec2-54-88-24-183.compute-1.amazonaws.com"
mlf.set_tracking_uri(f"{TRACKING_SERVER_HOST}:5000")

# set experiment name
mlf.set_experiment("wine-quality-remote")

# read data
data_loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(data_loc, sep=";")
X = df.drop(columns=["quality"])
y = df["quality"].values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# model parameters to be tested
alpha_vals = [0.2, 0.1, 0.05, 0.01]
l1_ratio_vals = [0.2, 0.1, 0.05]

for alpha in alpha_vals:
    for l1_ratio in l1_ratio_vals:
        # fit the model
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=314)
        reg.fit(X_train, y_train)
        preds_test = reg.predict(X_test)

        # get model evaluation metrics
        rmse = (mean_squared_error(y_test, preds_test)) ** 0.5
        mae = mean_absolute_error(y_test, preds_test)

        print(f"{rmse=:.3f}, {mae=:.2f} ({alpha=:.2f}, {l1_ratio=:.2f})")

        # record model parameter
        with mlf.start_run(nested=True):
            mlf.log_param("alpha", alpha)
            mlf.log_param("l1_ratio", l1_ratio)

            # record evaluation metrics
            mlf.log_metric("rmse", rmse)
            mlf.log_metric("mae", mae)

# -- log the best model
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

# -- interact with the remote model registry
client = mlf.tracking.MlflowClient(f"{TRACKING_SERVER_HOST}:5000")

# grab the run_id for the logged model
run_id = exp.run_id

# use this run_id to register the model
mlf.register_model(model_uri=f"runs:/{run_id}/models", name="wine-quality-model")
