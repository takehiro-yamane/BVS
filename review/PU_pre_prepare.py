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

def dist_feature(dir_p, group):
    feature_path = Path(f'Result/0316/Focal/2image/for_pseudo/{group}/feature')
    features = sorted(feature_path.glob('*.npy'))
    # tfeatures = features[2:]
    for id, f in enumerate(features):
        feature = np.load(str(f))
        save_p = Path(f'Result/{dir_p}/for_PU_pre/group{group}/feature/id{id}')
        save_p.mkdir(exist_ok=True, parents=True)
        np.save(f'{save_p}/pred', feature)

def dist_gt(dir_p, gt_group, group):
    gt_path = Path(f'image_CH/five-fold/group_{gt_group}/gt')
    gts = sorted(gt_path.glob('*.tif'))
    for id, g in enumerate(gts):
        gt = cv2.imread(str(g), 0)
        gt = gt/gt.max()
        save_p = Path(f'Result/{dir_p}/for_PU_pre/group{group}/gt/id{id}')
        save_p.mkdir(exist_ok=True, parents=True)
        np.save(f'{save_p}/gt', gt)


def main(dir_p):
    print('main')
    for i in range(5):
        dist_feature(dir_p, i)
        if i+3<=4:
            dist_gt(dir_p , i+3, i)
        else:
            dist_gt(dir_p, i-2, i)
    
if __name__ == '__main__':
    dir_p = '0411-pu'
    main(dir_p)