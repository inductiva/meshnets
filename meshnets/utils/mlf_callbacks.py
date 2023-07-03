"""Define custom lightning callbacks."""

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger


class MLFlowLoggerCheckpointer(MLFlowLogger):
    """Extend the MLFlowLogger class to log checkpoints as MLFlow artifacts
    each time a new model is saved.
    
    This class saves checkpoints during training but does not delete previous
    best checkpoints. Thus, it can lead to a high number of checkpoints
    saved."""

    def after_save_checkpoint(self,
                              checkpoint_callback: ModelCheckpoint) -> None:
        """ Called after model checkpoint callback saves a new checkpoint."""

        self.experiment.log_artifact(self.run_id,
                                     checkpoint_callback.best_model_path,
                                     artifact_path='checkpoints')


class MLFlowLoggerFinalizeCheckpointer(MLFlowLogger):
    """Extend the MLFlowLogger class to log checkpoints as MLFlow artifacts
    at the end of training."""

    _checkpoint_callback = None

    def after_save_checkpoint(self,
                              checkpoint_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint."""

        if self._checkpoint_callback is None:
            self._checkpoint_callback = checkpoint_callback

    def finalize(self, status: str) -> None:
        """Log the checkpoints as MLFlow artifacts at the end of training."""

        if self._checkpoint_callback is not None:
            self.experiment.log_artifacts(self.run_id,
                                          self._checkpoint_callback.dirpath,
                                          artifact_path='checkpoints')
