"""Define custom lightning callbacks."""

import GPUtil
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import Callback
from torch_geometric.data import Batch
import warnings


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


class GPUUsage(Callback):
    """Track and log GPU usage."""

    def __init__(self, log_freq=50):
        """
        Args:
            log_freq: The frequency, in batches, at which to log GPU usage.
        """
        self.log_freq = log_freq
        self.gpu_usage_history = {}

    def on_batch_end(self, trainer, pl_module):
        if (trainer.global_step +
                1) % self.log_freq == 0 or trainer.is_last_batch:
            for gpu_uuid, gpu_usage in self.gpu_usage_history.items():
                trainer.logger.log_metrics(
                    {f'gpu_memory_{gpu_uuid}_usage': np.mean(gpu_usage)},
                    step=trainer.global_step)
                self.gpu_usage_history = {}
        self._update_gpu_usage()

    def _update_gpu_usage(self):
        try:
            all_gpus = GPUtil.getGPUs()
            for gpu in all_gpus:
                uuid = gpu.uuid
                if uuid not in self.gpu_usage_history:
                    self.gpu_usage_history[uuid] = []
                self.gpu_usage_history[uuid].append(gpu.memoryUtil)
        # pylint: disable=broad-except
        except Exception as e:
            warnings.warn('Something went wrong fetching GPU information.\n'\
                          f'Failed with exception {e}')


class GradientNorm(Callback):
    """Log full model gradient norm."""

    def __init__(self, log_freq=50):
        """
        Args:
            log_freq: The frequency, in batches, at which to log GPU usage.
        """
        self.log_freq = log_freq

    def on_after_backward(self, trainer, pl_module):
        if (trainer.global_step +
                1) % self.log_freq == 0 or trainer.is_last_batch:
            for param in trainer.model.parameters():
                total_norm = 0
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item()**2

            gradient_norm = total_norm**0.5
            trainer.logger.log_metrics({'gradient_norm': gradient_norm},
                                       step=trainer.global_step)


class GeometricBatchSize(Callback):
    """Log nodes and edges in the torch geometric batch."""

    def on_train_batch_start(self,
                             trainer,
                             pl_module,
                             batch: Batch,
                             batch_idx: int,
                             unused: int = 0) -> None:
        trainer.logger.log_metrics(
            {
                'num_nodes': batch.num_nodes,
                'num_edges': batch.num_edges
            },
            step=trainer.global_step)
