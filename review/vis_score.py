
from ftplib import parse150
from os import mkdir
from pathlib import Path
from numpy.compat.py3k import npy_load_module
from numpy.lib.function_base import append
from numpy.lib.npyio import save
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,octahedron,ball
from numba import jit, prange
import statistics
import math
import pickle
import matplotlib.pyplot as plt

def get_value(pre, hoge, x):
    tmp = pre[(hoge==1)]
    tmp = sorted(tmp)
    l = len(tmp)*x
    value = tmp[int(l)]
    return value

# def get_value(pre, th):
    # tmp = sorted(pre.flatten())
    # l = len(tmp)*th
    # value = tmp[int(l)]
    # return value


def bottom(pre, gt, mask, th):
    hoge = mask==255
    tmp = np.copy(pre)

    value =get_value(pre, hoge, th)
    hoge2 = pre<value

    tmp[(hoge==0)]=-10
    tmp[(hoge2==0)]=-10
    
    tmp[(hoge2==1) & (gt!=0)]=20
    return tmp
    
def tops(pre, gt, mask, th):
    hoge = mask==255
    tmp = np.copy(pre)

    value =get_value(pre, hoge, th)
    hoge2 = pre>value

    tmp[(hoge==0)]=-10
    tmp[(hoge2==0)]=-10
    
    # tmp[(hoge2==1) & (gt!=0)]=20
    return tmp
    

def bottom_nomask(pre, gt, th):
    # hoge = mask==0
    tmp = np.copy(pre)

    # value =get_value(pre, hoge, th)
    value =get_value(pre, th)
    hoge2 = pre<value

    # tmp[(hoge==1)]=-10
    tmp[(hoge2==0)]=-10
    
    tmp[(hoge2==1) & (gt!=0)]=20
    return tmp


def score_image(save_path, pre_path, gt_path, mask_path):
    pres = sorted(pre_path.glob('*.npy'))
    gts = sorted(gt_path.glob('*.npy'))
    masks = sorted(mask_path.glob('*.tif'))
    mask = masks[2]
    save_path.mkdir(exist_ok = True)

    for id in range(len(pres)):
        pre = np.load(str(pres[id]))
        mask = cv2.imread(str(mask), 0)
        gt = np.load(str(gts[id]))
        gt = gt/gt.max()

        pre = np.squeeze(pre)
        
        plt.rcParams["image.cmap"]='jet'

        pre70 = bottom(pre, gt, mask, 0.7)
        pre50 = bottom(pre, gt, mask, 0.5)
        pre30 = bottom(pre, gt, mask, 0.3)
        pre20 = bottom(pre, gt, mask, 0.2)

        pre80 = tops(pre, gt, mask, 0.8)
        pre90 = tops(pre, gt, mask, 0.9)
        pre95 = tops(pre, gt, mask, 0.95)


        fig = plt.figure(figsize=[12.80, 13.37])
        plt.imshow(pre80)
        plt.clim(-10, 10)
        plt.colorbar()
        # plt.show()
        plt.savefig(f'{save_path}/t_80_{id}.png')
        plt.close()

        fig = plt.figure(figsize=[12.80, 13.37])
        plt.imshow(pre90)
        plt.clim(-10, 10)
        plt.colorbar()
        # plt.show()
        plt.savefig(f'{save_path}/t_90_{id}.png')
        plt.close()

        # fig = plt.figure(figsize=[12.80, 13.37])
        # plt.imshow(pre30)
        # plt.clim(-10, 10)
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'{save_path}/b_30_{id}.png')
        # plt.close()

        # fig = plt.figure(figsize=[12.80, 13.37])
        # plt.imshow(pre50)
        # plt.clim(-10, 10)
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'{save_path}/b_50_{id}.png')
        # plt.close()

        # fig = plt.figure(figsize=[12.80, 13.37])
        # plt.imshow(pre70)
        # plt.clim(-10, 10)
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'{save_path}/b_70_{id}.png')
        # plt.close()
        
        fig = plt.figure(figsize=[12.80, 13.37])
        plt.imshow(pre95)
        plt.clim(-10, 10)
        plt.colorbar()
        # plt.show()
        plt.savefig(f'{save_path}/t_95_{id}.png')
        plt.close()

        # fig = plt.figure(figsize=[12.80, 13.37])
        # plt.imshow(pre20)
        # plt.clim(-10, 10)
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'{save_path}/b_20_{id}.png')
        # plt.close()

        # fig = plt.figure(figsize=[12.80, 13.37])
        # plt.imshow(pre)
        # plt.clim(-10, 10)
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'{save_path}/pre_{id}.png')
        # plt.close()
        
def exec_score_image(dir_p, group):
    for i in range(5):
        save_p = Path(f'Result/{dir_p}/{group}/vis-score_id{i}')
        save_p.mkdir(exist_ok=True, parents=True)
        score_image(
            save_p,
            Path(f'Result/{dir_p}/for_PU_pre/group{group}/pred/id{i}'),
            Path(f'Result/{dir_p}/for_PU_pre/group{group}/gt/id{i}'),
            Path(f'Result/{dir_p}/for_PU_train/{group}/id{i}/mask')
        )


def main(dir_p):
    exec_score_image(dir_p, 0)
    exec_score_image(dir_p, 1)
    exec_score_image(dir_p, 2)
    exec_score_image(dir_p, 3)
    exec_score_image(dir_p, 4)
    print('f')

if __name__ == '__main__':
    dir_p = '0411-pu'
    main(dir_p)



