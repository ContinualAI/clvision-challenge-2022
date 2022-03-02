# 3rd CLVision Workshop @ CVPR 2022 Challenge

This is the official starting repository for the **Continual Learning Challenge** held in the **3rd CLVision Workshop @ CVPR 2022**.

Please refer to the [**challenge website**](https://sites.google.com/view/clvision2022/challenge) for more details!

## Getting started
The devkit is based on the [Avalanche library](https://github.com/ContinualAI/avalanche). We warmly recommend looking at the [documentation](https://avalanche.continualai.org/) (especially the ["Zero to Hero"](https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction) tutorials) if this is the first time you use it!

For the demo track, Avalanche is added as a Git submodule of this repository. The submodule points to a specific commit of a branch we specifically created for the challenge.

The recommended setup steps are as follows:

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) (and [mamba](https://github.com/mamba-org/mamba); recommended)

2. Clone the repo and create the conda environment:
```bash
git clone --recurse-submodules https://github.com/ContinualAI/clvision-challenge-2022.git
conda env create -f environment.yml
```

3. Setup your IDE so that the avalanche submodule is included in the *PYTHONPATH*. Note: you have to include the top-level folder, not `avalanche/avalanche`!
   1. For Jetbrains IDEs (PyCharm), this can be done from the *Project* pane (usually on the right) by right-clicking on the "avalanche" folder -> "Mark Directory as" -> "Sources Root".
   2. For VS Code, follow the [official documentation](https://code.visualstudio.com/docs/python/environments#_use-of-the-pythonpath-variable).

4. Download and extract the dataset: TBD

The aforementioned steps should be OS-agnostic. However, we recommend setting up your dev environment using a mainstream Linux distro.

## Object Classification Training Template
More details coming soon. Stay tuned!

## Object Detection Training Template
The devkit provides a training template that you should use to develop your CL strategy. We suggest you to study the [from zero to hero tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction) to learn about Avalanche.

This following training template is based on Avalanche [training templates](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training).

The training loop is an almost exact implementation of the one shown in the official [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), especially the [train_one_epoch](https://github.com/pytorch/vision/blob/71d2bb0bc67044f55d38bfddf04e05be0343deab/references/detection/engine.py#L12) method.

A schematic visualization of the training loop, its events, and an example of a plugin implementing EWC is shown below:
![Object Detection Template schema](./docs/img/od_template.png)
