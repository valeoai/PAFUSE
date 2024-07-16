import sys
from omegaconf import DictConfig, ListConfig

import torch


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


# Source: https://medium.com/optuna/easy-hyperparameter-management-with-hydra-
# mlflow-and-optuna-783730700e7d


def log_params_from_omegaconf_dict(params, mlflow_on):
    if mlflow_on:
        for param_name, element in params.items():
            _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    import mlflow as mlf
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                # Supposes lazy mlflow import in main script
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            # Supposes lazy mlflow import in main script
            mlf.log_param(f"{parent_name}.{i}", v)


def log_param_to_mlf(key, value, mlflow_on):
    if mlflow_on:
        import mlflow as mlf
        try:
            mlf.log_param(key=key, value=value)
        except mlf.exceptions.RestException as e:
            print(e)
            pass


def log_metric_to_mlflow(key, value, mlflow_on, step=None):
    if mlflow_on:
        import mlflow as mlf
        try:
            mlf.log_metric(
                key=key,
                value=float(value),
                step=step,
            )
        except mlf.exceptions.RestException as e:
            print(e)
            pass


def log_metrics_to_mlflow(metrics, mlflow_on, step=None):
    if mlflow_on:
        import mlflow as mlf
        try:
            metrics = {k: float(v) for k,v in metrics.items()}
            mlf.log_metrics(
                metrics=metrics,
                step=step,
            )
        except mlf.exceptions.RestException as e:
            print(e)
            pass


def save_state(
    model,
    optimizer,
    epoch_no,
    lr,
    foldername,
    scheduler=None,
    random_state=None,
    log_in_mlf=False,
    tag=None,
):
    params = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch_no,
        "lr": lr,
        'model_pos': model.state_dict(),
    }
    if scheduler is not None:
        params["scheduler"] = scheduler.state_dict()

    if random_state is not None:
        params["random_state"] = random_state

    if tag is None:
        tag = f"epoch_{epoch_no}"
    fname = f"{foldername}/{tag}.bin"

    print('Saving checkpoint to', fname)

    torch.save(params, fname)
    if log_in_mlf:
        import mlflow as mlf
        mlf.log_artifact(fname)
