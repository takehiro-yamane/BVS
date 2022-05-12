import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from torch import save
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def dist_gt(group):
    supervised_gt_path = Path(f'image_CH/five-fold/2image/group_{group}/gt')
    svgts = sorted(supervised_gt_path.glob('*.tif'))

    tmp = cv2.imread('Result/0323-pu/for_PU_train/pred_group0/binary/binary_0000_50.tif', 0)
    arr = np.zeros_like(tmp)
    arr = arr.astype('int')
    arr -= 1
    for i in range(5):
        id_path = Path(f'Result/0323-pu/for_PU_train/pred_group{group}/id{i}')
        save_path = id_path/'gt'
        save_path.mkdir(exist_ok=True, parents=True)
        for id, gtp in enumerate(svgts):
            gt = cv2.imread(str(gtp), 0)
            ret = np.where(gt==0, 0, 1)
            np.save(f'{save_path}/{id:02}', ret)
        np.save(f'{save_path}/pseudo_gt', arr)


def dist_mask(group):
    supervised_gt_path = Path(f'image_CH/five-fold/2image/group_{group}/gt')
    svgts = sorted(supervised_gt_path.glob('*.tif'))

    pseudo_mask_path = Path(f'Result/0322-pseudo/for_train/group_{group}/mask')
    masks = sorted(pseudo_mask_path.glob('*.tif'))
    ps_masks = masks[2:]

    for i in range(5):
        id_path = Path(f'Result/0323-pu/for_PU_train/pred_group{group}/id{i}')
        save_path = id_path/'mask'
        save_path.mkdir(exist_ok=True, parents=True)
        for id, gtp in enumerate(svgts):
            gt = cv2.imread(str(gtp), 0)
            cv2.imwrite(f'{save_path}/{id:02}.tif', gt)
        mask = cv2.imread(str(ps_masks[i]), 0)
        ret = np.where(mask==0, 255, 0)
        cv2.imwrite(f'{save_path}/pseudo_mask.tif', ret)

def dist_feature(group):
    feature_path = Path(f'Result/0323-pu/for_PU_train/pred_group{group}/feature/feature')
    features = sorted(feature_path.glob('*.npy'))
    sv_features = features[:2]
    ps_features = features[2:]
    for i in range(5):
        id_path = Path(f'Result/0323-pu/for_PU_train/pred_group{group}/id{i}')
        save_path = id_path/'feature'
        save_path.mkdir(exist_ok=True, parents=True)
        for id, svp in enumerate(sv_features):
            svf = np.load(str(svp))
            np.save(f'{save_path}/{id:02}',svf)
        psf = np.load(str(ps_features[i]))
        np.save(f'{save_path}/pseudo_feature',psf)

def main():
    for i in range(5):
        dist_gt(i)
        dist_mask(i)
        dist_feature(i)

if __name__ == '__main__':
    main()
