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
Starting template for the "object classification - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified.
- Don't remove the mandatory plugin (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the accuracy metric. You
    are free to add more metrics or change the logger.
- The use of Avalanche training and logging code is not mandatory. However,
    you are required to use the given benchmark generation procedure. If not
    using Avalanche, make sure you are following the same train/eval loop and
    please make sure you are able to export the output in the expected format.
"""

import argparse
import datetime
from pathlib import Path
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from devkit_tools.benchmarks import challenge_classification_benchmark
from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter

# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = Path.home() / '3rd_clvision_challenge' / 'challenge'


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if args.cuda >= 0 and torch.cuda.is_available()
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
    train_transform = Compose(
        [RandomCrop(224, padding=10, pad_if_needed=True),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor(), torchvision_normalization]
    )
    # ---------

    # --- BENCHMARK CREATION
    benchmark = challenge_classification_benchmark(
        dataset_path=DATASET_PATH,
        train_transform=train_transform,
        eval_transform=eval_transform,
        n_validation_videos=0
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
    mandatory_plugins = [
        ClassificationOutputExporter(
            benchmark, save_folder='./instance_classification_results')
    ]
    plugins: List[SupervisedPlugin] = [
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    evaluator = EvaluationPlugin(
        accuracy_metrics(
            epoch=True,
            stream=True
        ),
        loss_metrics(
            minibatch=False,
            epoch_running=True
        ),
        # May be useful if using a validation stream
        # confusion_matrix_metrics(stream=True),
        timing_metrics(
            experience=True, stream=True
        ),
        loggers=[InteractiveLogger(),
                 TensorboardLogger(
                     tb_log_dir='./log/track_inst_cls/exp_' +
                                datetime.datetime.now().isoformat())
                 ],
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    # In Avalanche, you can customize the training loop in 3 ways:
    #   1. Adapt the make_train_dataloader, make_optimizer, forward,
    #   criterion, backward, optimizer_step (and other) functions. This is the
    #   clean way to do things!
    #   2. Change the loop itself by reimplementing training_epoch or even
    #   _train_exp (not recommended).
    #   3. Create a Plugin that, by implementing the proper callbacks,
    #   can modify the behavior of the strategy.
    #  -------------
    #  Consider that popular strategies (EWC, LwF, Replay) are implemented
    #  as plugins. However, writing a plugin from scratch may be a tad
    #  tedious. For the challenge, we recommend going with the 1st option.
    #  In particular, you can create a subclass of the SupervisedTemplate
    #  (Naive is mostly an alias for the SupervisedTemplate) and override only
    #  the methods required to implement your solution.
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=0 if 'valid' in benchmark.streams else -1
    )
    # ---------

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)
        print("Current Classes: ", experience.classes_in_this_experience)

        data_loader_arguments = dict(
            num_workers=10,
            persistent_workers=True
        )

        if 'valid' in benchmark.streams:
            # Each validation experience is obtained from the training
            # experience directly. We can't use the whole validation stream
            # (because that means accessing future or past data).
            # For this reason, validation is done only on
            # `valid_stream[current_experience_id]`.
            cl_strategy.train(
                experience,
                eval_streams=[benchmark.valid_stream[current_experience_id]],
                **data_loader_arguments)
        else:
            cl_strategy.train(
                experience,
                **data_loader_arguments)
        print("Training completed")

        print("Computing accuracy on the complete test set")
        cl_strategy.eval(benchmark.test_stream, num_workers=10,
                         persistent_workers=True)


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
