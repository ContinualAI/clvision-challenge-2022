from typing import List, Any
from typing_extensions import TypedDict


class EgoObjectsCategory(TypedDict):
    id: int
    name: str
    image_count: int
    instance_count: int
    frequency: str  # Missing, computed from image_count


class EgoObjectsImage(TypedDict):
    id: int
    height: int
    width: int
    url: str
    gaia_id: int  # Internal, ignore it
    timestamp: List[float]
    stream_ids: List[str]
    group_id: str
    frame_id: int
    main_category: str
    main_category_instance_ids: List[int]
    video_id: str
    # date_captured: str
    # neg_category_ids: List[int]
    # license: int
    # flickr_url: str
    # coco_url: str
    # not_exhaustive_category_ids: List[int]


class EgoObjectsAnnotation(TypedDict):
    id: int
    image_id: int
    bbox: List[float]
    category_id: int
    instance_id: str
    area: float  # Missing, computed from bbox
    # segmentation: List[List[float]]  # Will not be supported


class EgoObjectsJson(TypedDict):
    info: Any
    categories: List[EgoObjectsCategory]
    images: List[EgoObjectsImage]
    annotations: List[EgoObjectsAnnotation]


__all__ = [
    'EgoObjectsCategory',
    'EgoObjectsImage',
    'EgoObjectsAnnotation',
    'EgoObjectsJson'
]
