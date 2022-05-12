import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm

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


class CellImageLoad(object):
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
        gt_o = np.load(str(gt_name))
        # if gt_o.max() != 0:
        #     gt_o = gt_o / gt_o.max()

        mask_name = self.mask_paths[data_id]
        mask_o = cv2.imread(str(mask_name), 0)
        if mask_o.max() != 0:
            mask_o = mask_o / mask_o.max()


        # # data augumentation
        # top, bottom, left, right = self.random_crop_param(img_o.shape)
        # img = img_o[top:bottom, left:right]
        # gt = gt_o[top:bottom, left:right]
        # mask = mask_o[top:bottom, left:right]

        flg=0
        if gt_o.max()>0.1:
            while(flg==0):             
                top, bottom, left, right = self.random_crop_param(img_o.shape)
                img = img_o[top:bottom, left:right]
                gt = gt_o[top:bottom, left:right]
                mask = mask_o[top:bottom, left:right]
                if gt.max() > 0.1:
                    flg = 1
                    break
        else:
            top, bottom, left, right = self.random_crop_param(img_o.shape)
            img = img_o[top:bottom, left:right]
            gt = gt_o[top:bottom, left:right]
            mask = mask_o[top:bottom, left:right]

        rand_value = random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value)
        mask = rotate(mask, 90 * rand_value)

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

class FeatureImageLoad(object):
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
        img = np.load(str(img_name))
        # img = img/img.max()

        gt_name = self.gt_paths[data_id]
        gt = np.load(str(gt_name))

        mask_name = self.mask_paths[data_id]
        mask = cv2.imread(str(mask_name), 0)
        
        img = np.squeeze(img)
        gt = np.squeeze(gt)
        mask = np.squeeze(mask)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        gt = gt.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        datas = {"image": img, "gt": gt, "mask" : mask}

        return datas

## val loading
class ValImageLoad(object):
    def __init__(self, ori_path, gt_path, mask_path):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = np.load(str(img_name))

        gt_name = self.gt_paths[data_id]
        gt = np.load(str(gt_name))

        mask_name = self.mask_paths[data_id]
        mask = cv2.imread(str(mask_name), 0)

        img = np.squeeze(img)
        gt = np.squeeze(gt)
        mask = np.squeeze(mask)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        gt = gt.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        datas = {"image": img, "gt": gt, "mask" : mask}

        return datas

class PredImageLoad(object):
    def __init__(self, ori_path):
        self.ori_paths = ori_path

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), 0)
        img = torch.from_numpy(img.astype(np.float32))
        datas = {"image": img.unsqueeze(0)}
        return datas

def gather_path(train_paths, mode):
        ori_paths = []
        ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
        return ori_paths

if __name__ == '__main__':
    train_path = Path('./image/train')
    ori_paths = gather_path('./image/train', "ori")
    gt_paths = gather_path('./image/train', "gt")
    aa = CellImageLoad(ori_paths, gt_paths)
    zz = torch.utils.data.DataLoader(
            aa, batch_size=1, shuffle=True, num_workers=0
        )
    for i, data in enumerate(zz):
        imgs = data["image"]
        true_masks = data["gt"]
        print(imgs.shape)
        print(true_masks.shape)
