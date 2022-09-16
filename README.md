# Mlflow tutorial

**Disclaimer:** I have borrowed some ideas and steps from the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) as well as the [Mlflow documentation/tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html).

This repository contains a step-by-step process that demonstrates some of the core functionalities of Mlflow.

### Step 0: Run local experiments

`0_run_experiments.py`

This program demonstrates a simple hyper-parameter search process. We split the modeling dataset into three partitions and try to find the best values for two hyper-parameters (`l1_ratio` and `alpha`) for an Elastic Net model. Four different values are evaluated for `alpha` and three for `l1_ratio`, resulting in twelve total combinations. Two model evaluation metrics are reviewed (`rmse` and `mae`) by simply printing them in the output.

### Step 1: Track experiments (using Mlflow)

`1_track_experiments.py`

We now use *Mlflow Tracking* to create an experiment and record the model hyper-parameters and metrics. Instead of just printing those values in the output (in Step 1), we now store these values in a systematic way so that they can be reviewed in future:

The following Mlflow functions are introduced in this program:

`get_tracking_uri()` to get the tracking uri, which in this case, is a local (default) folder (`./mlruns`).
`list_experiments()` to see the list of experiments that exits in the local `./mlruns` folder. Initially, there is only one experiment called 'Default'.
`set_experiment()` to create a new experiment called `wine-quality-local`. This creates a new folder under `./mlruns` named `1`.

`log_param()` to log the two hyper-parameters: `alpha` and `l1_ratio`.
`log_metric()` to log the two model evaluation metrics: `rmse` and `mae`.

Notice that the `list_experiments()` function now returns two experiments.

We can view the details of this experiment on the Mlflow UI. Enter the following command to activate the UI: `Mlflow ui`. Click on the following URL (or copy + paste it in your browser) that will be printed in the output: http://127.0.0.1:5000. You can also see plots to better identify the best set of hyper-parameters.

### Step 2: Save the model as an artifact

`2_save_artifacts.py`

Once the model is finalized, we can save the final model an artifact using Mlflow. In a real-world situation, a Data Scientist would spend more time exploring additional values for the hyper-parameters, but we are keeping things simple by just picking the best set of values from our initial range of values. (`best_alpha=0.01` and `best_l1_ratio=0.2` were chosen somewhat arbitrarily.)

`log_model()` function is used to store the model as an artifact. We could have used additional artifacts (such as any data preparation pipeline) but to keep things simple, we are using this function to just save the model.

Now you can go back to the Mlflow UI (again, by entering `Mlflow ui` on the terminal) and you will see that the latest experiment (once you click on it) contains artifacts saved under the 'model' section. 

### Step 3: Use model arfifacts

`3_use_artifacts.py`

The saved model (and other artifacts) can now be used in another program to score a new dataset.

### Step 4: Create tracking URI

`4_create_tracking_uri.py`

We now launch a tracking server to explore the experiment results. For now, we will continue to keep the backend and artifact store _locally_. However, instead of using the local _filesystem_, we will now use *SQLite database for backend storage*. 

The Mlflow UI can be now accessed by running the following command: 
```Mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./artifacts
```

Note that the location for the model artifacts has to be assigned when we use a backend store URI. Currently, we are using a local folder (`./artifacts`) to store model artifacts.

On the Mlflow UI, you will notice that the *Models* tab (on the top right menu) can now be accessed. This is because we have a model artifact location.

### Step 5: Register a model to Mlflow registry

`5_register_model.py`

Now that we are using a SQLite database for backend storage, we can register models with the Mlflow model registry.

You will need the correct `run_id` for the model you would like to register. 

`list_run_infos()` is used to extract the information about an experiment, including the `run_id` for the model that you would like to register.
`register_model()` is used to register the model to the Mlflow model registry. Once registered, you will see that model appear on the Mlflow UI under the *Models* tab.

### Step 6: Go remote!

`6_go_remote.py`

In this step, we will use a remote tracking server (running on AWS EC2), remote backend store using Postgres database (on AWS ECR), and remote artifact store (on AWS S3). Here are the steps to set those up on AWS: [mlflow_on_aws.md] (https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md) _(Note: These steps are outlined as part of MLOps Zoomcamp that I mentioned above. I recommend you check it out if you are interested in MLOps.)_

