import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from ego_objects.colormap import colormap

from ego_objects import EgoObjects, EgoObjectsResults


class EgoObjectsVis:
    def __init__(self, ego_gt, ego_dt=None, img_dir=None, dpi=75):
        """Constructor for EgoObjectsVis.
        Args:
            ego_gt (EgoObjects class instance, or str containing path of annotation file)
            ego_dt (EgoObjectsResult class instance, or str containing path
            of result file,
            or list of dict)
            img_dir (str): path of folder containing all images. If None, the image
            to be displayed will be downloaded to the current working dir.
            dpi (int): dpi for figure size setup
        """
        self.logger = logging.getLogger(__name__)

        if isinstance(ego_gt, EgoObjects):
            self.ego_gt = ego_gt
        elif isinstance(ego_gt, str):
            self.ego_gt = EgoObjects(ego_gt)
        else:
            raise TypeError("Unsupported type {} of ego_gt.".format(ego_gt))

        if ego_dt is not None:
            if isinstance(ego_dt, EgoObjectsResults):
                self.ego_dt = ego_dt
            elif isinstance(ego_dt, (str, list)):
                self.ego_dt = EgoObjectsResults(self.ego_gt, ego_dt)
            else:
                raise TypeError("Unsupported type {} of ego_dt.".format(ego_dt))
        else:
            self.ego_dt = None
        self.dpi = dpi
        self.img_dir = img_dir if img_dir else '.'
        if self.img_dir == '.':
            self.logger.warn("img_dir not specified. Images will be downloaded.")

    def get_name(self, idx):
        return self.ego_gt.load_cats(ids=[idx])[0]["name"]

    def setup_figure(self, img, title="", dpi=75):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_title(title)
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(img)
        return fig, ax

    def vis_bbox(self, ax, bbox, box_alpha=0.5, edgecolor="g", linestyle="--"):
        # bbox should be of the form x, y, w, h
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor=edgecolor,
                linewidth=2.5,
                alpha=box_alpha,
                linestyle=linestyle,
            )
        )

    def vis_text(self, ax, bbox, text, color="w"):
        ax.text(
            bbox[0],
            bbox[1] - 2,
            text,
            fontsize=15,
            family="serif",
            bbox=dict(facecolor="none", alpha=0.4, pad=0, edgecolor="none"),
            color=color,
            zorder=10,
        )

    def get_color(self, idx):
        color_list = colormap(rgb=True) / 255
        return color_list[idx % len(color_list), 0:3]

    def load_img(self, img_id):
        import cv2
        img = self.ego_gt.load_imgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img["url"].split("/")[-1])
        if not os.path.exists(img_path):
            self.ego_gt.download(self.img_dir, img_ids=[img_id])
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        return cv2.merge([r, g, b])

    def vis_img(
        self, img_id, show_boxes=False, show_classes=False,
        cat_ids_to_show=None
    ):
        ann_ids = self.ego_gt.get_ann_ids(img_ids=[img_id])
        anns = self.ego_gt.load_anns(ids=ann_ids)
        boxes, classes = [], []
        for ann in anns:
            boxes.append(ann["bbox"])
            classes.append(ann["category_id"])

        if len(boxes) == 0:
            self.logger.warn("No gt anno found for img_id: {}".format(img_id))
            return

        boxes = np.asarray(boxes)
        areas = boxes[:, 2] * boxes[:, 3]
        sorted_inds = np.argsort(-areas)

        fig, ax = self.setup_figure(self.load_img(img_id))

        for idx in sorted_inds:
            if cat_ids_to_show is not None and classes[idx] not in cat_ids_to_show:
                continue
            color = self.get_color(idx)
            if show_boxes:
                self.vis_bbox(ax, boxes[idx], edgecolor=color)
            if show_classes:
                text = self.get_name(classes[idx])
                self.vis_text(ax, boxes[idx], text)
        return fig, ax

    def vis_result(
        self, img_id, show_boxes=False, show_classes=False,
        cat_ids_to_show=None, score_thrs=0.0, show_scores=True
    ):
        assert self.ego_dt is not None, "ego_dt was not specified."
        anns = self.ego_dt.get_top_results(img_id, score_thrs)
        boxes, classes, scores = [], [], []
        for ann in anns:
            boxes.append(ann["bbox"])
            classes.append(ann["category_id"])
            scores.append(ann["score"])

        if len(boxes) == 0:
            self.logger.warn("No gt anno found for img_id: {}".format(img_id))
            return

        boxes = np.asarray(boxes)
        areas = boxes[:, 2] * boxes[:, 3]
        sorted_inds = np.argsort(-areas)

        fig, ax = self.setup_figure(self.load_img(img_id))

        for idx in sorted_inds:
            if cat_ids_to_show is not None and classes[idx] not in cat_ids_to_show:
                continue
            color = self.get_color(idx)
            if show_boxes:
                self.vis_bbox(ax, boxes[idx], edgecolor=color)
            if show_classes:
                text = self.get_name(classes[idx])
                if show_scores:
                    text = "{}: {:.2f}".format(text, scores[idx])
                self.vis_text(ax, boxes[idx], text)
        return fig, ax