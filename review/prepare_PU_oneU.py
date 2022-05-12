import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from torch import save, zeros_like
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


#pseudo用に推定したfeatureをgroup, id ごとに振り分ける
def get_feature(save_path, group):
    feature_path = Path(f'Result/0316/Focal/2image/for_pseudo/{group}/feature')
    features = sorted(feature_path.glob('*.npy'))
    for id, f in enumerate(features):
        sf_path = save_path/f'id{id}/feature'
        tmp = np.load(str(f))
        np.save(f'{sf_path}/unlabeled_feature', tmp)

#閾値による擬似ラベルで追加学習する際のmaskにおいて，値が0の領域を1とするmaskをidごとに作成（PUの学習領域）
def get_mask(save_path, group):
    mask_path = Path(f'Result/0411-pseudo/for_train/group_{group}/mask')
    masks = sorted(mask_path.glob('*.tif'))
    unlabeled_masks = masks[2:]
    for id, m in enumerate(unlabeled_masks):
        sm_path = save_path/f'id{id}/mask'
        mask = cv2.imread(str(m),0)
        unlabeled_mask = np.where(mask==0, 255, 0)
        cv2.imwrite(f'{sm_path}/unlabeled_mask.tif',unlabeled_mask)

#閾値による擬似ラベルはpositiveしか使わないパターン
def get_mask_threshold_positive(save_path, group):
    if group + 3 < 5:
        pseudo_group = group+3
    else:
        pseudo_group = group+3-5
    th_positive_path = Path(f'Result/0411-pseudo/for_pseudo/group_{group}/gt')
    th_positives = sorted(th_positive_path.glob('*.tif'))
    outside_mask_path = Path(f'image_CH/five-fold/group_{pseudo_group}/mask')
    outside_masks = sorted(outside_mask_path.glob('*.tif'))

    for i in range(len(th_positives)):
        sm_path = save_path/f'id{i}/mask'
        tmp_p = cv2.imread(str(th_positives[i]), 0)
        tmp_o = cv2.imread(str(outside_masks[i]), 0)
        mask = np.where(tmp_o==0, 255, tmp_p)
        cv2.imwrite(f'{sm_path}/unlabeled_mask.tif', mask)

#unlabeleldのgt(すべて値は-1)をidごとに振り分け
def make_u_gt(save_path):
    gt_npy = np.load('Result/0411-pu/for_PU_train/0/id0/gt/00.npy')
    zero_arr = np.zeros_like(gt_npy)
    unlabeled_arr = zero_arr - 1
    for j in range(5):
        sug_path = save_path/f'id{j}/gt'
        np.save(f'{sug_path}/unlabeled_gt', unlabeled_arr)


def main(parent_path):
    for i in range(5):
        save_path = parent_path/f'{i}'
        get_feature(save_path, i)
        # get_mask(save_path, i)
        get_mask_threshold_positive(save_path, i)
        make_u_gt(save_path)

    print('f')

if __name__ == '__main__':
    parent_path = Path('Result/0411-pu/for_PU_train')
    main(parent_path)