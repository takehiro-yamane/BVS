import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def create_mask(save_path, pseudo_mask_paths, ori_mask_paths):
    for id in range(4):
        pseudo_mask = cv2.imread(str(pseudo_mask_paths[id+2]),0)
        ori_mask = cv2.imread(str(ori_mask_paths[id]), 0)
        tmp = np.copy(pseudo_mask)
        tmp[ori_mask==0]=255
        rev = np.where(tmp==0, 255, 0)
        cv2.imwrite(f'{save_path}/mask/pl{id}.tif', rev)

def exec_create_mask(group):
    pmp = sorted(Path(f'Result/0309-pseudo/for_train/group_{group}/mask').glob('*.tif'))
    omp = sorted(Path(f'Result/0302/prepare_data/group_{group}_2/pseudo_mask').glob('*.tif'))
    create_mask(
        f'Result/0309-pu/for_PUtrain/group_{group}',
        pmp,
        omp
    )

def one_unlabeled(group):
    group_path = Path(f'Result/0313-pu/for_PUtrain/group_{group}')
    pgt_path = group_path/'gt'
    pmask_path = group_path/'mask'
    
    pgts = sorted(pgt_path.glob('*.npy'))
    pmasks = sorted(pmask_path.glob('*.tif'))
    pgt0 = np.load(str(pgts[0]))
    # pgt1 = np.load(str(pgts[1]))
    pmask0 = cv2.imread(str(pmasks[0]), 0)
    # pmask1 = cv2.imread(str(pmasks[1]), 0)

    for i in range(4):
        tmp_path = group_path/f'id{i}'
        tmp_path.mkdir(exist_ok=True)
        tmp_gt = tmp_path/'gt'
        tmp_mask = tmp_path/'mask'
        tmp_gt.mkdir(exist_ok=True)
        tmp_mask.mkdir(exist_ok=True)
        np.save(f'{tmp_gt}/0',pgt0)
        # np.save(f'{tmp_gt}/1',pgt1)
        cv2.imwrite(f'{tmp_mask}/0.tif', pmask0)
        # cv2.imwrite(f'{tmp_mask}/1.tif', pmask1)

    # pseudo_mask_path = Path(f'Result/0311-pseudo/for_train/group_{group}/mask')
    ori_mask_path = Path(f'Result/0302/prepare_data/group_{group}_2/pseudo_mask')

    # pseudo_mask_paths = sorted(pseudo_mask_path.glob('*.tif'))
    pseudo_mask_paths = pmasks[1:]
    ori_mask_paths = sorted(ori_mask_path.glob('*.tif'))
    for id in range(4):
        pseudo_mask = cv2.imread(str(pseudo_mask_paths[id+2]),0)
        ori_mask = cv2.imread(str(ori_mask_paths[id]), 0)
        tmp = np.copy(pseudo_mask)
        tmp[ori_mask==0]=255
        rev = np.where(tmp==0, 255, 0)
        save_path = group_path/f'id{id}'
        cv2.imwrite(f'{save_path}/mask/pl{id}.tif', rev)

def main():
    one_unlabeled(0)
    one_unlabeled(1)
    one_unlabeled(2)
    one_unlabeled(3)
    one_unlabeled(4)
    # exec_create_mask(0)
    # exec_create_mask(1)
    # exec_create_mask(2)
    # exec_create_mask(3)
    # exec_create_mask(4)

    tmp = np.load('Result/0308-pu/for_PUtrain/group_0/gt/000.npy')
    npy_zero = np.zeros_like(tmp)
    unlabeled = npy_zero-1
    for j in range(5):
        for i in range(4):
            savep = f'Result/0313-pu/for_PUtrain/group_{j}/id{i}/gt/Unlabeled'
            np.save(savep, unlabeled)

    print('f')

if __name__ == '__main__':

    main()