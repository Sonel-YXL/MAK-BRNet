from .base_dataset import BaseDataset
import os
import cv2
import numpy as np
import sys



if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


import xml.etree.ElementTree as ET
import os

import numpy as np
import matplotlib.pyplot as plt
import Utils.DOTA_devkit.polyiou as polyiou
from functools import partial
import cv2

def parse_gt(filename):
    objects = []
    target = ET.parse(filename).getroot()
    for obj in target.iter('HRSC_Object'):
        object_struct = {}
        difficult = int(obj.find('difficult').text)
        box_xmin = int(obj.find('box_xmin').text)
        box_ymin = int(obj.find('box_ymin').text)
        box_xmax = int(obj.find('box_xmax').text)
        box_ymax = int(obj.find('box_ymax').text)
        mbox_cx = float(obj.find('mbox_cx').text)
        mbox_cy = float(obj.find('mbox_cy').text)
        mbox_w = float(obj.find('mbox_w').text)
        mbox_h = float(obj.find('mbox_h').text)
        mbox_ang = float(obj.find('mbox_ang').text)*180/np.pi
        rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
        pts_4 = cv2.boxPoints(rect)
        bl = pts_4[0,:]
        tl = pts_4[1,:]
        tr = pts_4[2,:]
        br = pts_4[3,:]
        object_struct['name'] = 'ship'
        object_struct['difficult'] = difficult
        object_struct['bbox'] = [float(tl[0]),
                                 float(tl[1]),
                                 float(tr[0]),
                                 float(tr[1]),
                                 float(br[0]),
                                 float(br[1]),
                                 float(bl[0]),
                                 float(bl[1])]
        objects.append(object_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:

        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:


        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))


        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])



        i = np.where(mrec[1:] != mrec[:-1])[0]


        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval_hrsc(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):




    lines = imagesetfile
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(os.path.join(annopath.format(imagename)))

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}



    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if len(confidence) > 1:

        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)





        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]



    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)



        if BBGT.size > 0:





            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih


            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.








    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)


    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

class HRSC(BaseDataset):
    def __init__(self, phase, params):
        super(HRSC, self).__init__(phase, params)

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []

        target = ET.parse(self.imgann_path_ids[index]['annpath']).getroot()
        for obj in target.iter('HRSC_Object'):
            difficult = int(obj.find('difficult').text)
            box_xmin = int(obj.find('box_xmin').text)
            box_ymin = int(obj.find('box_ymin').text)
            box_xmax = int(obj.find('box_xmax').text)
            box_ymax = int(obj.find('box_ymax').text)
            mbox_cx = float(obj.find('mbox_cx').text)
            mbox_cy = float(obj.find('mbox_cy').text)
            mbox_w = float(obj.find('mbox_w').text)
            mbox_h = float(obj.find('mbox_h').text)
            mbox_ang = float(obj.find('mbox_ang').text)*180/np.pi
            rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
            pts_4 = cv2.boxPoints(rect)
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]
            valid_pts.append([bl, tl, tr, br])
            valid_cat.append(self.cat_ids['ship'])
            valid_dif.append(difficult)
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

















        return annotation

    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')

        annopath = os.path.join(self.imgann_path[0]['ann_file'], '{}.xml')

        imagesetfile_list = [name['id_name'] for name in self.imgann_path_ids]
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval_hrsc(detpath,
                                     annopath,

                                     imagesetfile_list,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap

            print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)






        map = map / len(self.category)



        return map, {"ship": map}
