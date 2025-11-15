

"""
@File       :   make_class_map.py
@Modify Time:   2024/7/21 下午4:59 
@Author     :   Sonel
@Version    :   1.0
@Contact    :   sonel@qq.com
@Description:   
"""
import cv2
import numpy as np


class MakeClassMap:
    """
    创建：threshold_map 和 threshold_mask
    """
    def __init__(self):
        pass



    def __call__(self, data: dict) -> dict:
        """

        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['ann']['polygons']
        class_ids = data['ann']['labels']


        h, w = image.shape[:2]


        mask = np.zeros((h, w), dtype=float)

        for i in range(len(text_polys)):
            polygon = text_polys[i]



            if data['ann']['ignore_tags'][i] == False:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], [int(class_ids[i])])

        data['ann']['class_mask'] = mask
        return data
