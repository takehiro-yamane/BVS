from pathlib import Path
from turtle import Turtle
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
import shutil

# get coordinates of bottom x% 
def getbottom(feature,mask,pseudo, x, id, savepath):
    hoge = mask==255
    hoge2 = pseudo==0
    feature2 = np.copy(feature)
    feature2[(hoge==0)]=1000
    feature2[(hoge2==1)]=1000
    tmp = feature[(hoge==1) & (hoge2==0)]
    tmp = sorted(tmp)
    l = len(tmp)
    b = l*x/100
    bvalue = tmp[int(b)]
    if bvalue > 1000:
        raise ValueError("bottom value is too big")

    mask[feature2<bvalue] = 0

    mask = np.where(mask==0, 255, 0)
    mask[pseudo==0]=0

    # coordinates = np.where(feature2 < bvalue)
    # coordinates = lc_append(coordinates)
    # # pseudo = np.where(pseudo>0, 255, 0)
    # # pseudo_mask = mask - pseudo
    # for c in coordinates:
    #     # pseudo_mask[c[0], c[1]]=0
    #     mask[c[0], c[1]]=255

    cv2.imwrite('{}/pseudo_mask_{:02}.tif'.format(str(savepath),id), mask)


def getbottom_nomask(feature, mask, x, id, savepath):
    hoge = mask==255
    feature2 = np.copy(feature)
    feature2[(hoge==0)]=1000
    tmp = feature[(hoge==1)]
    tmp = sorted(tmp)
    l = len(tmp)
    b = l*x/100
    bvalue = tmp[int(b)]
    if bvalue > 1000:
        raise ValueError("bottom value is too big")

    mask[feature2<bvalue] = 0
    mask = np.where(mask==0, 255, 0)
    cv2.imwrite('{}/pseudo_mask_{:02}.tif'.format(str(savepath),id), mask)

#上位をpositiveに追加する＆下位をnegtativeに追加する
def get_tops_bottom(feature,mask, gt, x, y, id, savepath, gt_savepath):
    hoge = mask==255
    feature2 = np.copy(feature)
    feature2[(hoge==0)]=1000
    tmp = feature[(hoge==1)]
    tmp = sorted(tmp)
    l = len(tmp)
    b = l*x/100
    t = l*y/100
    bvalue = tmp[int(b)]
    tvalue = tmp[int(t)]
    if bvalue > 1000:
        raise ValueError("bottom value is too big")

    mask[feature2<bvalue] = 0
    mask[(feature2>tvalue) & (feature2<1000)] = 0

    mask = np.where(mask==0, 255, 0)
    cv2.imwrite('{}/pseudo_mask_{:02}.tif'.format(str(savepath),id), mask)

    gt = gt/gt.max()
    gt[(feature2>tvalue) & (feature2<1000)] = 1
    gt *= 255
    cv2.imwrite(f'{gt_savepath}/pseudo_{id:02}.tif', gt)


# np.whereで得た座標を[[x,y,z]...]のリストに変形
def lc_append(l_coordinate):
    tmp_l = [[l_coordinate[0][j], l_coordinate[1][j]] for j in range(len(l_coordinate[0]))] 
    return tmp_l

def main_group(dir_p, pu_p, pseudo_p, group):
    
    # bottom x%
    # x = 20

    # #tops y%
    # y = 95


    save_path = Path(f'Result/{dir_p}/for_train/group_{group}/mask')
    save_path.mkdir(exist_ok=True, parents=True)

    gt_save_path = Path(f'Result/{dir_p}/for_train/group_{group}/gt')

    pred_f_path = Path(f'Result/{pu_p}/for_PU_pre/group{group}/pred')

    #PU学習領域
    pred_mask_path = Path(f'Result/{pu_p}/for_PU_train/{group}')

    #gt pseudo
    gt_path = Path(f'Result/{pseudo_p}/for_train/group_{group}/gt')
    gts = sorted(gt_path.glob('*.tif'))
    gts = gts[2:]

    #擬似ラベルのマスク（オリジナル）
    # pseudo_mask_path = Path(f'Result/0307/for_add_negative_labels/group_{group}/pseudo_mask')
    # pseudo_masks = sorted(pseudo_mask_path.glob('*.tif'))

    for i in range(5):
        pred_f_p = pred_f_path/f'id{i}' 
        feature = sorted(pred_f_p.glob('*.npy'))
        pred_m_p = pred_mask_path/f'id{i}/mask'
        mask = sorted(pred_m_p.glob('*.tif'))
        # p_mask = pseudo_masks[i]
        prior = get_prior(pu_p, group, i)
        x = (1-prior)*100
        x -= 30
        if x<0:
            x=0


        gt = cv2.imread(str(gts[i]), 0)
        feature = np.load(str(feature[0]))
        feature = np.squeeze(feature)
        mask = cv2.imread(str(mask[2]),0)
        # pseudo = cv2.imread(str(p_mask),0)
        # getbottom(feature, mask, pseudo, x, i, save_path)
        getbottom_nomask(feature, mask, x, i, save_path)
        # get_tops_bottom(feature, mask, gt, x, y, i, save_path, gt_save_path)

    print('finish')

def get_prior(dir_p, group, id):
    file_path = f'Result/{dir_p}/for_PU_train/{group}/id{id}/prior.txt'
    with open(file_path, mode='r') as f:
        data = f.read()
    prior = float(data)
    return prior

def main(dir_p, pu_p, pseudo_p):
    # shutil.copytree(f'Result/{pseudo_p}/for_train', f'Result/{dir_p}/for_train')
    for i in range(5):
        main_group(dir_p, pu_p, pseudo_p, i)

if __name__ == '__main__':
    dir_p = '0502-put'
    pu_p = '0411-pu'
    pseudo_p = '0411-pseudo'
    main(dir_p, pu_p, pseudo_p)