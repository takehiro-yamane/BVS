import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate


def local_maxima(img, threshold=200, dist=2):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data

class RetinaImageLoad(object):
    def __init__(self, ori_path, gt_path, mask_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path
        self.crop_size = crop_size


    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape

        top = random.randint(0, h - self.crop_size[0])
        left = random.randint(0, w - self.crop_size[1])

        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_o = cv2.imread(str(img_name), 0)
        if img_o.max() != 0:
            img_o = img_o / img_o.max()

        gt_name = self.gt_paths[data_id]
        gt_o = cv2.imread(str(gt_name), 0)
        if gt_o.max() != 0:
            gt_o = gt_o / gt_o.max()

        mask_name = self.mask_paths[data_id]
        mask_o = cv2.imread(str(mask_name), 0)
        if mask_o.max() != 0:
            mask_o = mask_o / mask_o.max()


        # data augumentation
        top, bottom, left, right = self.random_crop_param(img_o.shape)
        img = img_o[top:bottom, left:right]
        gt = gt_o[top:bottom, left:right]
        mask = mask_o[top:bottom, left:right]

        # flg=0
        # if gt_o.max()>0.1:
        #     while(flg==0):             
        #         top, bottom, left, right = self.random_crop_param(img_o.shape)
        #         img = img_o[top:bottom, left:right]
        #         gt = gt_o[top:bottom, left:right]
        #         mask = mask_o[top:bottom, left:right]
        #         if gt.max() > 0.1:
        #             flg = 1
        #             break
        # else:
        #     top, bottom, left, right = self.random_crop_param(img_o.shape)
        #     img = img_o[top:bottom, left:right]
        #     gt = gt_o[top:bottom, left:right]
        #     mask = mask_o[top:bottom, left:right]

        rand_value = random.randint(0, 4)
        
        img = np.rot90(img, rand_value)
        gt = np.rot90(gt, rand_value)
        mask = np.rot90(mask, rand_value)
        
        rand_value = random.randint(0, 3)
        if rand_value == 1 or rand_value == 3:
            img = np.fliplr(img)
            gt = np.fliplr(gt)
            mask = np.fliplr(mask)
        if rand_value == 2 or rand_value == 3:
            img = np.flipud(img)
            gt = np.flipud(gt)
            mask = np.flipud(mask)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "mask" : mask.unsqueeze(0)}

        return datas


class ValImageLoad(object):
    def __init__(self, ori_path, gt_path, mask_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path
        
        self.oris = []
        self.gts = []
        self.masks = []
        for i in range(len(self.ori_paths)):
            tmp_ori = cv2.imread(str(self.ori_paths[i]), 0)
            tmp_ori = tmp_ori/tmp_ori.max()
            self.oris.append(tmp_ori)

            tmp_gt = cv2.imread(str(self.gt_paths[i]), 0)
            tmp_gt = tmp_gt/tmp_gt.max()
            self.gts.append(tmp_gt)

            tmp_mask = cv2.imread(str(self.mask_paths[i]), 0)
            tmp_mask = tmp_mask/tmp_mask.max()
            self.masks.append(tmp_mask)


    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        # img_name = self.ori_paths[data_id]
        # img_o = cv2.imread(str(img_name), 0)
        # img = img_o / img_o.max()

        # gt_name = self.gt_paths[data_id]
        # gt_o = cv2.imread(str(gt_name), 0)
        # gt = gt_o / gt_o.max()

        # mask_name = self.mask_paths[data_id]
        # mask_o = cv2.imread(str(mask_name), 0)
        # mask = mask_o / mask_o.max()

        img = self.oris[data_id]
        gt = self.gts[data_id]
        mask = self.masks[data_id]

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "mask": mask.unsqueeze(0)}

        return datas


def gif2tif(save, paths):
    for id, p in enumerate(paths):
        cap = cv2.VideoCapture(str(p))
        ret, frame = cap.read()
        tmp = frame[:,:,1]
        tmp = np.squeeze(tmp)
        cv2.imwrite('{}/{:04}.tif'.format(save, id), tmp)



class NomaskValImageLoad(object):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        
        self.oris = []
        self.gts = []
        for i in range(len(self.ori_paths)):
            tmp_ori = cv2.imread(str(self.ori_paths[i]), 0)
            tmp_ori = tmp_ori/tmp_ori.max()
            self.oris.append(tmp_ori)

            tmp_gt = cv2.imread(str(self.gt_paths[i]), 0)
            tmp_gt = tmp_gt/tmp_gt.max()
            self.gts.append(tmp_gt)


    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img = self.oris[data_id]
        gt = self.gts[data_id]

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}

        return datas










