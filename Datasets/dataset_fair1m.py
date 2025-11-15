from Datasets.base_dataset import BaseDataset
from Utils.DOTA_devkit.ResultMerge_multi_process import mergebypoly


class FAIR1M(BaseDataset):
    def __init__(self, phase, params):
        super(FAIR1M, self).__init__(phase, params)

    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)
