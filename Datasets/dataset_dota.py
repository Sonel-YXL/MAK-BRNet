from Datasets.base_dataset import BaseDataset
from Utils.DOTA_devkit.ResultMerge_multi_process import mergebypoly
from Datasets.base_evaluation_task import evaluation_OOD
import os

class DOTA(BaseDataset):
    def __init__(self, phase, params):
        super(DOTA, self).__init__(phase, params)

    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)

    def dec_evaluation(self, result_path):
        """
        传入写出文件路径,需要对文件进行评估
        :param result_path:
        :return:
        """

        detpath = os.path.join(result_path, 'Task1_{}.txt')
        annopath = os.path.join(self.imgann_path[0]['ann_file'], '{}.txt')

        imagenames = [name['id_name'] for name in self.imgann_path_ids]
        map, ap_dict = evaluation_OOD(
            detpath=detpath,
            annopath=annopath,
            imagenames=imagenames,
            classnames=self.category
        )

        return map, ap_dict