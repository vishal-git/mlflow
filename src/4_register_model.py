# this program shows how to register a model to mlflow model registry

from mlflow.tracking import MlflowClient
from mlflow import register_model, set_tracking_uri, set_experiment

localhost = "http://localhost:5000"

client = MlflowClient(localhost)
set_tracking_uri(localhost)

# let's grab and review (print) the details for experiment no. 1
exp = client.list_run_infos(experiment_id="1")[0]
for ex in exp:
    print(f"{ex[0]:>16}: {ex[1]}")

# grab the run_id for the logged model
run_id = exp.run_id

# use this run_id to register the model
register_model(model_uri=f"runs:/{run_id}/models", name="wine-quality-model")
