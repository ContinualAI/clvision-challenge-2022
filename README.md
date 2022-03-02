# clvision-challenge-2022
CVPR 2022 Continual Learning in Computer Vision Workshop Challenge

## Training Template
The devkit provides a training template that you should use to develop your CL strategy. We suggest you to study the [from zero to hero tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction) to learn about Avalanche.
This training template is based on Avalanche [training templates](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training).

The training loop is an almost exact implementation of the one shown in this [pytorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), especially the `train_one_epoch` method.

A schematic visualization of the training loop, its events, and an example of a plugin implementing EWC is shown below:
![alt text](./docs/img/od_template.png)