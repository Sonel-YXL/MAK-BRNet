import random
import cv2
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import math
import numbers
import random
import cv2
from skimage.util import random_noise



class EastRandomCropData():
    """
    功能：多尺度 + 训练一张图的部分数据
    """

    def __init__(self, size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:  data['img']  data['target_polys_crop']  data['target_class']
        """
        im = data['img']
        target_polys = data['ann']['polygons']


        target_class = data['ann']['labels']


        all_care_polys = target_polys



        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)


        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)

            padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))

        target_polys_crop = []

        target_crop = []

        for poly, targets_id_ in zip(target_polys, target_class):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                target_polys_crop.append(poly)
                target_crop.append(targets_id_)
        data['img'] = img




        data['ann']['polygons'] = np.float32(target_polys_crop)
        data['ann']['labels'] = target_crop
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        """
         被crop_area调用 判断ploy点是否在这个制定的区域内
        :param poly: 需要判断的点
        :param x:
        :param y:
        :param w:
        :param h:
        :return:  不在区域内，返回false,在区域内，返回True
        """
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        """
        功能: 被crop_area调用 将连续的区域进行切分
        :param axis:位置列表
        :return:
        """
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        """
         被crop_area调用
        :param axis: np, 一维连续值
        :param max_size:
        :return:
        """
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        """
        功能:  被crop_area调用  传入切分图像一维列表.原始图像大小,即在空白处选择x或者y的范围
        :param regions:
        :param max_size:
        :return: 返回regions区域中的两个值
        """
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)


        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def crop_area(self, im, target_polys):
        """
        功能：被call调用，获取指定大小的区域
        :param im:np, 原始图像
        :param target_polys: np, 目标的点坐标
        :return:
        """
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in target_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:

                continue
            num_poly_in_rect = 0
            for poly in target_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


class PSERandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        imgs = data['imgs']

        h, w = imgs[0].shape[0:2]
        th, tw = self.size
        if w == tw and h == th:
            return imgs


        if np.max(imgs[2]) > 0 and random.random() > 3 / 8:

            tl = np.min(np.where(imgs[2] > 0), axis=1) - self.size
            tl[tl < 0] = 0

            br = np.max(np.where(imgs[2] > 0), axis=1) - self.size
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])

                if imgs[1][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)


        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        data['imgs'] = imgs
        return data


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{k: self.to_tuple_if_list(v) for k, v in args['args'].items()})
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    @staticmethod
    def to_tuple_if_list(obj):
        """
        如果传入的数据是列表类型，则变为元组类型。这个自己变成了静态的方法
        :param obj:
        :return:
        """
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment(object):
    """
        根据指定的方法对图像数据进行数据增强，返回的数据保存在img字段，返回的四个点的数据保存在iaa_ann字段
        对'img'字段和'polygons'进行了改变


       - type: IaaAugment
         args:
           - { 'type': Fliplr, 'args': { 'p': 0.5 } }
           - { 'type': Affine, 'args': { 'rotate': [ -10,10 ] } }
           - { 'type': Resize,'args': { 'size': [ 0.5,3 ] } }
    """

    def __init__(self, augmenter_args):
        self.augmenter_args = augmenter_args
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def __call__(self, data):
        image = data['img']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['img'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):

        if aug is None:
            return data
        line_polys = []
        for poly in data['ann']['polygons']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)

        data['ann']['polygons'] = np.array(line_polys)



        return data

    def may_augment_poly(self, aug, img_shape, poly):
        """

        :param aug: 图像增强数据库序列 iaa.Sequential
        :param img_shape:  原始图像大小 800,800,3
        :param poly: np数据，
        :return:
        """

        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly



class RandomScale:
    def __init__(self, scales, random_rate):
        """
        :param scales: 尺度
        :param random_rate: 随机系数
        :return:
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['ann']['polygons']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['ann']['polygons'] = tmp_text_polys
        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param random_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['ann']['polygons']


        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:

            rangle = np.deg2rad(angle)

            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))

        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)

        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))

        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]

        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)




        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data['img'] = rot_img
        data['ann']['polygons'] = np.array(rot_text_polys)
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param random_rate: 随机系数
        :param keep_ratio: 是否保持长宽比
        :return:
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['ann']['polygons']

        if self.keep_ratio:

            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['ann']['polygons'] = text_polys
        return data


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param short_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys


    def __call__(self, img, annotation):
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """


        im = img
        text_polys = annotation['pts'].copy()

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:

            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)

            if self.resize_text_polys:



                text_polys[:, :, 0] *= scale[0]
                text_polys[:, :, 1] *= scale[1]




        annotation['pts'] = text_polys
        return im, annotation


class HorizontalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['ann']['polygons']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['img'] = flip_im
        data['ann']['polygons'] = flip_text_polys
        return data


class VerticalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['ann']['polygons']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['img'] = flip_im
        data['ann']['polygons'] = flip_text_polys
        return data

