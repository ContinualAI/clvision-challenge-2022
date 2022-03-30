################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-02-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
Starting template for the "object detection - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified (apart from the BENCHMARK CREATION one).
- Don't remove the mandatory metric (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the detection metrics. You
    are free to add more metrics or change the logger.
"""

import argparse
import datetime
import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import timing_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised.naive_object_detection import \
    ObjectDetectionTemplate
from devkit_tools.benchmarks import demo_detection_benchmark, \
    challenge_category_detection_benchmark, \
    challenge_instance_detection_benchmark
from devkit_tools.metrics.detection_output_exporter import EgoMetrics
from devkit_tools.metrics.dictionary_loss import dict_loss_metrics

from examples.tvdetection.transforms import RandomHorizontalFlip, ToTensor

# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = Path.home() / '3rd_clvision_challenge' / 'challenge'

# This sets the root logger to write to stdout (your console).
# Customize the logging level as you wish.
logging.basicConfig(level=logging.NOTSET)


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if args.cuda >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    # Add additional transformations here
    # You can take some detection transformations here:
    # https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
    # Beware that:
    # - transforms found in torchvision.transforms.transforms will only act on
    #    the image and they will not adjust bounding boxes accordingly: don't
    #    use them (apart from ToTensor)!
    # - make sure you are using the "Compose" from avalanche.benchmarks.utils,
    #    not the one from torchvision or from the aforementioned link.
    train_transform = Compose(
        [ToTensor(), RandomHorizontalFlip(0.5)]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor()]
    )
    # ---------

    # --- BENCHMARK CREATION
    benchmark = challenge_instance_detection_benchmark(
        dataset_path=DATASET_PATH,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    # ---------

    # --- MODEL CREATION
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    num_classes = benchmark.n_classes + 1  # N classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    print('Num classes (including background)', num_classes)
    # --- OPTIMIZER AND SCHEDULER CREATION

    # Create the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=1e-5)

    # Define the scheduler
    train_mb_size = 4

    # When using LinearLR, the LR will start from optimizer.lr / start_factor
    # (here named warmup_factor) and will then increase after each call to
    # scheduler.step(). After start_factor steps (here called warmup_iters),
    # the LR will be set optimizer.lr and never changed again.
    warmup_factor = 1.0 / 1000
    warmup_iters = \
        min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )
    # ---------

    # TODO: ObjectDetectionTemplate == Naive == plain fine tuning without
    #  replay, regularization, etc.
    # For the challenge, you'll have to implement your own strategy (or a
    # strategy plugin that changes the behaviour of the ObjectDetectionTemplate)

    # --- PLUGINS CREATION
    # Avalanche already has a lot of plugins you can use!
    # Many mainstream continual learning approaches are available as plugins:
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins

    # Note on LRSchedulerPlugin
    # Consider that scheduler.step() may be called after each epoch or
    # iteration, depending on the needed granularity. In the Torchvision
    # object detection tutorial, in the train_one_epoch function, step() is
    # called after each iteration. In addition, the scheduler is only used in
    # the very first epoch. The same setup is here replicated.
    mandatory_plugins = []
    plugins: List[SupervisedPlugin] = [
        LRSchedulerPlugin(
            lr_scheduler, step_granularity='iteration',
            first_exp_only=True, first_epoch_only=True),
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    mandatory_metrics = [EgoMetrics(save_folder='./track2_results',
                                    filename_prefix='track2_output')]
    evaluator = EvaluationPlugin(
        mandatory_metrics,
        timing_metrics(
            experience=True,
            stream=True
        ),
        loss_metrics(
            minibatch=True,
            epoch_running=True,
        ),
        dict_loss_metrics(
            minibatch=True,
            epoch_running=True,
            epoch=True,
            dictionary_name='detection_loss_dict'
        ),
        loggers=[InteractiveLogger(),
                 TensorboardLogger(
                     tb_log_dir='./log/track_cat_det/exp_' +
                                datetime.datetime.now().isoformat())],
        benchmark=benchmark
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    cl_strategy = ObjectDetectionTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator
    )
    # ---------

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)

        cl_strategy.train(experience, num_workers=10)
        print("Training completed")

        print("Computing accuracy on the full test set")
        cl_strategy.eval(benchmark.test_stream, num_workers=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
