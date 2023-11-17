"""Methods for loading models"""

import os
import tempfile
from typing import Union

import mlflow
# pylint: disable=unused-import
from torch import tensor  # used to eval tensors

from meshnets.modules.model import MeshGraphNet
from meshnets.modules.lightning_wrapper import MGNLightningWrapper


def load_model_from_mlflow(tracking_uri: str, run_id: str,
                           checkpoint: Union[int, str]) -> MGNLightningWrapper:
    """Load a model from an MLFlow run and a given checkpoint.
    
    The checkpoint can be specified as either an index or the name of the
    checkpoint.
    
    This method requires the hyperparameters to be saved in the checkpoint."""

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory() as temp_dir:
        if isinstance(checkpoint, int):
            artifacts = client.list_artifacts(run_id, 'checkpoints')
            checkpoint_path = client.download_artifacts(
                run_id, path=artifacts[checkpoint].path, dst_path=temp_dir)
        elif isinstance(checkpoint, str):
            checkpoint_path = client.download_artifacts(run_id,
                                                        path=os.path.join(
                                                            'checkpoints',
                                                            checkpoint),
                                                        dst_path=temp_dir)

        wrapper = MGNLightningWrapper.load_from_checkpoint(checkpoint_path)

    return wrapper
