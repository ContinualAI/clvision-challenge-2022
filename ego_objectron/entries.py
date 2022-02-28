from typing import List, TypedDict, Any


class EgoObjectronCategory(TypedDict):
    id: int
    name: str
    image_count: int
    instance_count: int


class EgoObjectronImage(TypedDict):
    id: int
    height: int
    width: int
    url: str
    gaia_id: Any  # Internal, ignore it
    timestamp: float
    stream_ids: List[str]
    group_id: int
    frame_id: int
    main_category: str
    # date_captured: str
    # neg_category_ids: List[int]
    # license: int
    # flickr_url: str
    # coco_url: str
    # not_exhaustive_category_ids: List[int]


class EgoObjectronAnnotation(TypedDict):
    id: int
    image_id: int
    bbox: List[float]
    category_id: int
    instance_id: str
    # area: float  # Missing, computed from bbox
    # segmentation: List[List[float]]  # Will not be supported


__all__ = [
    'EgoObjectronCategory',
    'EgoObjectronImage',
    'EgoObjectronAnnotation'
]
