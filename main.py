

"""函数训练测试入口.
train:进行训练,保存模型
eval:对数据进行验证,得到模型的各种指标
test:对数据进行测试,如可视化,模型合并等内容,
"""

import os
import sys
import argparse
import anyconfig







ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_FOLDER)
sys.path.append(os.path.join(ROOT_FOLDER, "Utils", "DOTA_devkit"))

from Utils.util import parse_config
from Tools.train import TrainModule
from Tools.eval import EvalModule
from Tools.test import TestModule


def init_args():
    """对输入文件进行解析。
    """
    parser = argparse.ArgumentParser(description='OOD-SOTA')
    parser.add_argument('--config_file', default='./Configs/Tdata_model_b12_10x.yaml', type=str)

    parser.add_argument('--phase', default='train', type=str, help="train/eval/test")
    fun_args = parser.parse_args()
    assert os.path.exists(fun_args.config_file)
    return fun_args


if __name__ == '__main__':
    args = init_args()
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    config['rootdir'] = ROOT_FOLDER
    if args.phase == "train":
        train_manager = TrainModule(config)
        train_manager.train_net()
    elif args.phase == "eval":
        eval_manager = EvalModule(config)
        eval_manager.evaluation(args)
    elif args.phase == "test":
        test_manager = TestModule(config)

        test_manager.modeltest(draw_line=True)
    pass









