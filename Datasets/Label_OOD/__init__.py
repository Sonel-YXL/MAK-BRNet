

"""生成gt

"""
from .BBA_gt import *

__all__ = ['BBA_label']



def build_label(config_trans):
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

