

"""所有图像操作的集合。

"""
from .trans_pos import *
from .trans_color import *
from .BBA_transform import *
from .make_border_map import MakeBorderMap
from Datasets.Transform_OOD.make_shrink_map import MakeShrinkMap
from .make_class_map import MakeClassMap

__all__ = [ 'EastRandomCropData', 'IaaAugment', 'to_RGB']

__all__ = __all__ + ["BBA_transform", 'RandomNoise', 'RandomScale', 'RandomRotateImgBox', 'RandomResize', 'ResizeShortSize',
           'HorizontalFlip', 'VerticalFlip', 'MakeBorderMap', 'MakeShrinkMap', 'MakeClassMap']


__all__ = __all__ + ["RandomNoise", "ChangeLight"]


def build_transform(config_trans):
    """对配置文件的数据预处理部分进行构建,包含数据预处理和数据增强。
    """
    aug_res = []
    if config_trans is not None:
        for aug in config_trans:

            assert aug['type'] in __all__, f'{aug["type"]} is not developed yet!, only {__all__} are support now'

            if 'args' not in aug:
                args = {}
            else:
                args = aug['args']

            if isinstance(args, dict):
                cls = eval(aug['type'])(**args)
            else:
                cls = eval(aug['type'])(args)
            aug_res.append(cls)
    return aug_res

if __name__ == '__main__':
    pre_config = [
        {"type": 'EastRandomCropData',
         "args": {
             'size': [640, 640],
             'max_tries': 50,
             'keep_ratio': True}
         }
    ]
















