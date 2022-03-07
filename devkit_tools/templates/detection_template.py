import math

import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates import BaseSGDTemplate
from examples.tvdetection.utils import reduce_dict


def detection_collate_fn(batch):
    """
    Collate function used when loading detection datasets using a DataLoader.
    """
    return tuple(zip(*batch))


# TODO: in Avalanche, you can customize the training loop in 3 ways:
#   1. Adapt the make_train_dataloader, make_optimizer, forward,
#   criterion, backward, optimizer_step (and other) functions. This is the clean
#   way to do things!
#   2. Change the loop itself by reimplementing training_epoch or even
#   _train_exp (not recommended).
#   3. Create a Plugin that, by implementing the proper callbacks,
#   can modify the behavior of the strategy.
#  -------------
#  Consider that popular strategies (EWC, LwF, Replay) are implemented
#  as plugins. However, writing a plugin from scratch may be a tad
#  tedious. For the challenge, we recommend going with the 1st option.
#  In particular, you can create a subclass of this ObjectDetectionTemplate
#  and override only the methods required to implement your solution.
class ObjectDetectionTemplate(BaseSGDTemplate):
    """
    The object detection strategy template.

    This template can be instanced as-is (resulting in a naive fine tuning).

    The template uses the detection scripts
    """
    def __init__(self, scaler=None, **base_kwargs):
        super().__init__(**base_kwargs)
        self.scaler = scaler  # torch.cuda.amp.autocast scaler
        self._images = None
        self._targets = None

        # Object Detection attributes
        self.losses = None
        self.loss_dict = None
        self.res = None  # only for eval loop.

    def training_epoch(self, **kwargs):
        return super(ObjectDetectionTemplate, self).training_epoch(**kwargs)

    def eval_epoch(self, **kwargs):
        # Unless your solution uses very non-standard mechanisms to compute
        # the predictions, we recommend to NOT change the eval_epoch method.
        return super(ObjectDetectionTemplate, self).eval_epoch(**kwargs)

    def make_train_dataloader(self, num_workers=4, **kwargs):
        """Assign dataloader to self.dataloader."""
        # TODO: You are free to change the parameters of the training data
        #  loader.
        self.dataloader = DataLoader(
            self.experience.dataset,
            batch_size=self.train_mb_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=detection_collate_fn
        )

    def make_eval_dataloader(self, num_workers=4, **kwargs):
        """Assign dataloader to self.dataloader."""

        # Note: keep drop_last and shuffle to False for the competition!
        # You are free to change the training data loader created in
        # make_train_dataloader.
        self.dataloader = DataLoader(
            self.experience.dataset,
            batch_size=self.eval_mb_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=detection_collate_fn
        )

    def make_optimizer(self, **kwargs):
        """Optimizer initialization."""
        # TODO: if you want to reset the optimizer state between each
        #  training experience, this is the right place to do it
        # The current optimizer is stored in self.optimizer
        # If you re-create the optimizer, make sure to store the new one in that
        # field!
        pass  # pass == keep the current optimizer as is

    def criterion(self):
        """
        Compute the loss function.

        For the challenge, we don't compute the test loss (metrics will be
        computed by an external Metric object).
        """
        if self.is_training:
            self.losses = sum(loss for loss in self.loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(self.loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
            return self.losses
        else:
            # eval does not compute the loss directly.
            # metrics can use self.mb_output and self.res
            self.res = {target["image_id"].item(): output
                        for target, output in zip(self.targets, self.mb_output)}
            return self.res

    def forward(self):
        """
        Compute the model's output given the current mini-batch.
        """
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                self.loss_dict = self.model(self.images, self.targets)
            return self.loss_dict
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outs = self.model(self.images)
            return [{k: v.to('cpu') for k, v in t.items()} for t in outs]

    def model_adaptation(self, model=None):
        """
        Adapts the model to the current experience.
        """
        if model is None:
            model = self.model
        avalanche_model_adaptation(model, self.experience.dataset)
        return model.to(self.device)

    @property
    def images(self):
        """
        Return the images in the current mini-batch.

        :return: Images in the current mini-batch. These are usually already
        transformed using ToTensor, so expect a list of PyTorch Tensors.
        """
        return self._images

    @property
    def targets(self):
        """
        Return the targets in the current mini-batch.

        :return: Targets in the current mini-batch, as a list of dictionaries
        (one dictionary for each image).
        """
        return self._targets

    def _unpack_minibatch(self):
        # Unpack minibatch mainly takes care of moving tensors to devices
        images = self.mbatch[0]
        targets = self.mbatch[1]
        self._images = list(image.to(self.device) for image in images)
        self._targets = [{k: v.to(self.device) for k, v in t.items()} for t in
                         targets]

    def backward(self):
        if self.scaler is not None:
            self.scaler.scale(self.losses).backward()
        else:
            self.losses.backward()

    def optimizer_step(self, **kwargs):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()


__all__ = [
    'detection_collate_fn',
    'ObjectDetectionTemplate'
]
