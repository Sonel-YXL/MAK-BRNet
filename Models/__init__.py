
'''
@File       :   __init__.py
@Modify Time:   2024/11/07 下午20:00
@Author     :   Sonel
@Version    :   1.5
@Contact    :   sonel@qq.com
@Description:   对不同的模型进行构建，每个外部的函数模型都需要重新建立一个模型文件夹
'''
import numpy as np
import copy
from Models.BBAVectors import CTRBOX, DecDecoder, LossAll
from Models.BBAVectors_LSK import CTRBOX_LSK, LossAll_nan
from Models.FPNBBA.FPNBBANet import FPNBBA
from Models.FPNLSKBBA import FPNLSKBBA
from Models.DFLBBA import DFLBBA
from Models.CFLBBA import CFLBBA
from Models.LFLBBA import LFLBBA
__all__ = ['build_model']

from Models.GRFBBBA.GRFBUNet import GRFBbba


support_model = ['CTRBOX', "CTRBOX_LSK", "GRFBbba", "FPNBBA","FPNLSKBBA","DFLBBA","CFLBBA","LFLBBA"]
support_decoder = ['DecDecoder']
support_loss = ['LossAll', "LossAll_nan"]


def build_model(config):
    """这里对模型部分重新进行构建，采用了旋转框架进行构建。
    :param condfig: 配置文件，其中有各个模块的字段信息，name表示总体模型的名称
    :return: 返回一个模型
    """
    assert config['type'] in support_model, (f'{config["type"]} is not developed yet!, '
                                             f'only {support_model} are support now')
    arch_model = eval(config['type'])(**config['args'])
    return arch_model


def build_loss(config):
    assert config['type'] in support_loss, (f'{config["type"]} is not developed yet!, '
                                             f'only {support_loss} are support now')

    if 'args' not in config:
        args = {}
    else:
        args = config['args']
    if isinstance(args, dict):
        cls = eval(config['type'])(**args)
    else:
        cls = eval(config['type'])(args)
    return cls




def build_decoder(config):
    assert config['type'] in support_decoder, (f'{config["type"]} is not developed yet!, '
                                             f'only {support_decoder} are support now')
    arch_model = eval(config['type'])(**config['args'])
    return arch_model