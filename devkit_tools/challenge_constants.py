from torchvision.transforms import Resize

DEFAULT_DEMO_CLASS_ORDER_SEED = 20220307
DEFAULT_DEMO_TRAIN_JSON = 'egoobjects_sample_train.json'
DEFAULT_DEMO_TEST_JSON = 'egoobjects_sample_test.json'


DEMO_CLASSIFICATION_FORCED_TRANSFORMS = Resize((224, 224))


__all__ = [
    'DEFAULT_DEMO_CLASS_ORDER_SEED',
    'DEFAULT_DEMO_TRAIN_JSON',
    'DEFAULT_DEMO_TEST_JSON',
    'DEMO_CLASSIFICATION_FORCED_TRANSFORMS'
]
