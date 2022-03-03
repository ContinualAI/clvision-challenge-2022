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
Starting template for the 1st challenge track (object classification)

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified (apart from the BENCHMARK CREATION one).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the accuracy metric. You
    are free to add more metrics or change the logger.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    confusion_matrix_metrics, timing_metrics, MAC_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from devkit_tools.benchmarks.classification_benchmark import \
    demo_classification_benchmark

from devkit_tools.challenge_constants import DEFAULT_DEMO_CLASS_ORDER_SEED

from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter

# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = Path.home() / '3rd_clvision_challenge'

# Don't change this (unless you want to experiment with different class orders)
# Note: it won't be possible to change the class order in the real challenge
CLASS_ORDER_SEED = DEFAULT_DEMO_CLASS_ORDER_SEED


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    # This is the normalization used in torchvision models
    # https://pytorch.org/vision/stable/models.html
    torchvision_normalization = transforms.Normalize(
        mean=[0.485,  0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Add additional transformations here
    train_transform = transforms.Compose(
        [RandomCrop(224, padding=10, pad_if_needed=True),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = transforms.Compose(
        [ToTensor(), torchvision_normalization]
    )
    # ---------

    # --- BENCHMARK CREATION
    benchmark = demo_classification_benchmark(
        dataset_path=DATASET_PATH,
        class_order_seed=CLASS_ORDER_SEED,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    # ---------

    # --- MODEL CREATION
    model = SimpleMLP(
        input_size=3*224*224,
        num_classes=benchmark.n_classes)
    # ---------

    # TODO: Naive == plain fine tuning without replay, regularization, etc.
    # For the challenge, you'll have to implement your own strategy (or a
    # strategy plugin that changes the behaviour of the SupervisedTemplate)

    # --- PLUGINS CREATION
    # Avalanche already has a lot of plugins you can use!
    # Many mainstream continual learning approaches are available as plugins:
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    #
    mandatory_plugins = [ClassificationOutputExporter('./track1_results')]
    plugins: List[SupervisedPlugin] = [
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    evaluator = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            experience=True,
            stream=True
        ),
        loss_metrics(
            minibatch=False,
            epoch=True,
            experience=True,
            stream=True,
        ),
        confusion_matrix_metrics(stream=True),
        timing_metrics(
            experience=True, stream=True
        ),
        loggers=[InteractiveLogger()],
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        plugins=plugins,
        evaluator=evaluator
    )
    # ---------

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, num_workers=4)
        print("Training completed")

        print("Computing accuracy on the growing test set")
        exp_id = experience.current_experience
        results.append(
            cl_strategy.eval(benchmark.test_stream[:exp_id+1], num_workers=4)
        )


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
