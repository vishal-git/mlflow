# this program demonstrates a simple model hyper-parameter search process

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# read data
data_loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(data_loc, sep=";")
print(f"Input dataset: {df.shape}")

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
