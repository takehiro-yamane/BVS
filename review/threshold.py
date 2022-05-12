#閾値による擬似ラベルの取得を行う．
#for_trainディレクトリをコピーし，擬似ラベルの追加を行う．　


from os import mkdir
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
import shutil
import os

def binarize(save, img_paths, th):
    for id, p in enumerate(img_paths):
        tmp = cv2.imread(str(p), 0)
        tmp = tmp/tmp.max()
        ret = np.where(tmp>=th, 255, 0)
        cv2.imwrite('{}/{:02}.tif'.format(save, id), ret)

def binarize_bg(save, img_paths, th):
    for id, p in enumerate(img_paths):
        tmp = cv2.imread(str(p), 0)
        tmp = tmp/tmp.max()
        ret = np.where(tmp<=th, 255, 0)
        cv2.imwrite('{}/bg_{:02}.tif'.format(save, id), ret)

def binarize_mask(save, img_paths, mask_paths, th):
    for id, (ip, mp) in enumerate(zip(img_paths, mask_paths)):
        tmp = cv2.imread(str(ip), 0)
        mask = cv2.imread(str(mp), 0)
        tmp = tmp/tmp.max()
        ret = np.where(tmp>=th, 255, 0)
        ret = np.where(mp==0,0 , ret)
        cv2.imwrite('{}/masked_th{:02}_{:02}.tif'.format(save, th, id), ret)

def check_region(save, img_paths, gt_paths, th):
    for id, (ip, gp) in enumerate(zip(img_paths, gt_paths)):
        tmp = cv2.imread(str(ip))
        tmp = tmp/tmp.max()
        ret = np.where(tmp>=th, 255, 0)
        cv2.imwrite('{}/th{:02}_{:02}.tif'.format(save, th, id), ret)

        gt = cv2.imread(str(gp))
        over = np.where(ret>gt, [0, 0, 255], ret)
        less = np.where(gt>ret, [0, 70, 255], ret)
        cv2.imwrite('{}/th{:02}_{:02}_over.tif'.format(save, th, id), over)
        cv2.imwrite('{}/th{:02}_{:02}_less.tif'.format(save, th, id), less)

def check_region_bg(save, img_paths, gt_paths, th):
    for id, (ip, gp) in enumerate(zip(img_paths, gt_paths)):
        tmp = cv2.imread(str(ip))
        tmp = tmp/tmp.max()
        ret = np.where(tmp<=th, 255, 0)
        cv2.imwrite('{}/bg_th{:02}_{:02}.tif'.format(save, th, id), ret)

        gt = cv2.imread(str(gp))
        over = np.where(ret==gt, [0, 0, 255], ret)
        less = np.where(gt!=0, [0, 70, 220], ret)

        less2 = np.copy(ret)
        less2[(gt!=0) & (ret!=0)]=2
        less2[(gt!=0) & (ret==0)]=3
        less2 = np.where(less2==2, [255,0,0], less2)
        less2 = np.where(less2==3, [0,70,255], less2)


        cv2.imwrite('{}/bg_th{:02}_{:02}_over.tif'.format(save, th, id), over)
        cv2.imwrite('{}/bg_th{:02}_{:02}_less.tif'.format(save, th, id), less2)



def make_pseudo(dir_p, p_p, hth, lth, group, ori_group):
    # dir_p = '0411-pseudo'
    # p_p = '0316/Focal/2image'

    parent_path = Path(f'Result/{dir_p}/for_train/group_{group}')
    # parent_path.mkdir(exist_ok=True, parents=True)
    save_path = parent_path/'mask'
    save_path.mkdir(exist_ok=True)
    pseudo_path = parent_path/'gt'
    pseudo_path.mkdir(exist_ok=True)
    pseudo_ori_path = parent_path/'ori'
    pseudo_ori_path.mkdir(exist_ok=True)

    # mask_path = Path('image_CH/five-fold/group_{}/mask'.format(mask_group))
    # mask_paths = sorted(mask_path.glob('*.tif'))

    ori_path = Path(f'image_CH/five-fold/group_{ori_group}/ori')
    ori_paths= sorted(ori_path.glob('*.*'))

    front_path = Path(f'Result/{p_p}/for_pseudo/group{group}/{hth*100}%')
    back_path = Path(f'Result/{p_p}/for_pseudo/group{group}/{lth*100}%')
    front_paths = sorted(front_path.glob('*.tif'))
    back_paths = sorted(back_path.glob('*.tif'))

    if group+3<5:
        pseudo_group = group +3
    else:
        pseudo_group = group + 3 - 5

    outside_mask_path = Path(f'image_CH/five-fold/group_{pseudo_group}/mask')
    outside_masks = sorted(outside_mask_path.glob('*.tif'))
    for id, (fp, bp, op, oms) in enumerate(zip(front_paths, back_paths, ori_paths, outside_masks)):
        front = cv2.imread(str(fp), 0)
        front = front/front.max()
        back = cv2.imread(str(bp), 0)
        if back.max()!=0:
            back = back/back.max()
        # mask = cv2.imread(str(mp), 0)
        # if mask.max()!=0:
        #     mask = mask/mask.max()
        ori = cv2.imread(str(op), 0)

        #外側のマスク
        outside_mask = cv2.imread(str(oms), 0)

        pseudo_mask = front+back
        # pseudo_mask = np.where(mask==0, 0, pseudo_mask)

        pseudo_mask = np.where(outside_mask==0, 255, pseudo_mask)
        cv2.imwrite('{}/pseudo_mask_{:02}.tif'.format(str(save_path), id), pseudo_mask*255)
        cv2.imwrite('{}/pseudo_{:02}.tif'.format(str(pseudo_path), id), front*255)
        cv2.imwrite(f'{str(pseudo_ori_path)}/pseudo_ori{id:02}.tif', ori)

def main_th(dir_p, hth, lth, pre_group, gt_group):
    # dir_p = '0316/Focal/2image'
    # mask_path = Path('image_CH/five-fold/group_{}/mask'.format(gt_group))
    # mask_paths = sorted(mask_path.glob('*.tif'))
    gt_path = Path('image_CH/five-fold/group_{}/gt'.format(gt_group))
    gt_paths = sorted(gt_path.glob('*.tif'))
    
    pred_path = Path(f'Result/{dir_p}/for_pseudo/{pre_group}/pre')
    pred_paths = sorted(pred_path.glob('*.tif'))
    savep = Path(f'Result/{dir_p}/for_pseudo/group{pre_group}/th')
    savep.mkdir(exist_ok=True, parents=True)

    # check_region(str(savep), pred_paths, gt_paths, 0.7)
    # check_region(str(savep), pred_paths, gt_paths, 0.8)
    # check_region(str(savep), pred_paths, gt_paths, 0.85)
    # check_region_bg(str(savep), pred_paths, gt_paths, 0.2)
    # check_region_bg(str(savep), pred_paths, gt_paths, 0.15)
    # check_region_bg(str(savep), pred_paths, gt_paths, 0.1)
    # check_region_bg(str(savep), pred_paths, gt_paths, 0.075)
    # check_region_bg(str(savep), pred_paths, gt_paths, 0.05)

    spl = savep.parent/f'{lth*100}%'
    # sp15 = savep.parent/'15%'
    sph = savep.parent/f'{hth*100}%'

    # sp15.mkdir(exist_ok=True)
    spl.mkdir(exist_ok=True)
    sph.mkdir(exist_ok=True)
    binarize(sph, pred_paths, hth)
    binarize_bg(spl, pred_paths, lth)

def main(sv_path, dir_path, hth, lth):
    if not os.path.isdir(f'Result/{dir_path}/for_train'):
        shutil.copytree('image_CH/five-fold/2image', f'Result/{dir_path}/for_train')
    for i in range(5):
        if i+3<5:
            pid = i+3
        else:
            pid = i+3-5
        main_th(sv_path, hth, lth, i, pid)
        make_pseudo(dir_path, sv_path, hth, lth, i, pid)

if __name__ == '__main__':
    sv_path = '0316/Focal/2image'
    # dir_path = '/0505-pseudo2'
    dir_path = '/0505-th_test'
    main(sv_path, dir_path, 0.8, 0.1)
