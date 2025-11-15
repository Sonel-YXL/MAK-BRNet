import torch.utils.data as data
import cv2
import torch
import numpy as np
import math

from numpy import random


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img):
        img = self.rb(img)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img = distort(img)
        img = self.rln(img)
        return img

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
def ex_box_jaccard(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter_x1 = np.maximum(np.min(a[:,0]), np.min(b[:,0]))
    inter_x2 = np.minimum(np.max(a[:,0]), np.max(b[:,0]))
    inter_y1 = np.maximum(np.min(a[:,1]), np.min(b[:,1]))
    inter_y2 = np.minimum(np.max(a[:,1]), np.max(b[:,1]))
    if inter_x1>=inter_x2 or inter_y1>=inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:,0]), np.min(b[:,0]))
    x2 = np.maximum(np.max(a[:,0]), np.max(b[:,0]))
    y1 = np.minimum(np.min(a[:,1]), np.min(b[:,1]))
    y2 = np.maximum(np.max(a[:,1]), np.max(b[:,1]))
    mask_w = np.int(np.ceil(x2-x1))
    mask_h = np.int(np.ceil(y2-y1))
    mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    a[:,0] -= x1
    a[:,1] -= y1
    b[:,0] -= x1
    b[:,1] -= y1
    mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
    mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter)/(float(union)+1e-12)






    return iou

class BBA_label():
    def __init__(self, input_h, input_w, down_ratio, max_objs, num_classes):
        self.input_w = input_w
        self.input_h = input_h
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.num_classes = num_classes
        self.image_distort = PhotometricDistort()

    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new

    def __call__(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)

        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)

        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)




        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect

            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))

            bl = pts_4[0, :]
            tl = pts_4[1, :]
            tr = pts_4[2, :]
            br = pts_4[3, :]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)

            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct








            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox











            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1













        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               }
        return ret