import sys
from pathlib import Path

from ego_objectron import EgoObjectronVis

sys.path.append(str(Path.cwd() / 'avalanche'))

import matplotlib
matplotlib.use('Agg')

from devkit_tools import ChallengeDatasetSample

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image


def main():
    sample_root: Path = Path.home() / '3rd_clvision_challenge'
    # sample_json = sample_root / 'egoobjects_test.json'
    # sample_images = sample_root / 'images'

    sample_dataset = ChallengeDatasetSample(root=sample_root)
    ego_api = sample_dataset.ego_api

    print('Categories:', len(ego_api.get_cat_ids()))
    print('Images:', len(ego_api.get_img_ids()))
    print('Annotations:', len(ego_api.get_ann_ids()))

    # Example 1: dataset-agnostic simple plot from dataset output
    n_to_show = 5
    for img_idx in range(n_to_show):
        image, target = sample_dataset[img_idx]
        plot_sample(image, target, f'img_{img_idx}.png')
        plt.show()
        plt.clf()

    # Example 2: plot using EgoObjectronVis
    ego_vis = EgoObjectronVis(ego_api, img_dir=str(sample_root / 'cltest'))
    for img_id in [194436326217796, 205990635045022, 206879634955351, 220388966895761, 231815002397446]:
        fig, _ = ego_vis.vis_img(img_id=img_id, show_boxes=True, show_classes=True)
        fig.savefig(f'img_{img_id}_vis.png')
        plt.close(fig)


def plot_sample(img: Image.Image, target, save_as=None):
    img_id = int(target['image_id'])
    plt.title(f'Image ID: {img_id}')

    plt.gca().imshow(img)
    for box in target['boxes']:
        box = box.tolist()

        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        plt.gca().add_patch(rect)

    if save_as is not None:
        plt.savefig(str(save_as))


if __name__ == '__main__':
    main()
