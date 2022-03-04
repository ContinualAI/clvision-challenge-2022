from pathlib import Path
from typing import Union

import torch

from avalanche.core import SupervisedPlugin, Template
from avalanche.training.templates import SupervisedTemplate


class ModelCheckpoint(SupervisedPlugin):
    def __init__(self, out_folder: Union[str, Path], file_prefix: str):
        super().__init__()
        self.out_folder = Path(out_folder)
        self.file_prefix = file_prefix

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Save the model after each experience."""

        super().after_training_exp(strategy, *args, **kwargs)

        curr_exp = strategy.experience.current_experience
        model_out_path = self.out_folder / f'{self.file_prefix}{curr_exp}.pth'
        torch.save({
            'epoch': 0,
            'model_state_dict': strategy.model.state_dict(),
            'optimizer_state_dict': strategy.optimizer.state_dict(),
        }, str(model_out_path))


__all__ = [
    'ModelCheckpoint'
]
