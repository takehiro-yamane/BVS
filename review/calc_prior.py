from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

def svdata_ratio(sv_mask, sv_outside_mask):
    #全体の面積
    region = sv_mask.shape[0]*sv_mask.shape[1]
    #外側の面積
    outside = len(sv_outside_mask[sv_outside_mask==0])
    #内側の面積
    inside = region - outside
    #positveの面積
    positive = len(sv_mask[sv_mask>0])
    #negativeの面積
    negative = len(sv_mask[sv_mask==0])
    negative -= outside
    #pos, negの割合（内側）
    rp = positive/inside
    rn = negative/inside
    return rp, rn, positive

def calc(dir_path,pseudo_path,  group):
    if group+3<5:
        pseudo_group = group+3
    else:
        pseudo_group = group+3-5

    #外側領域のマスク
    sv_outside_mask_path = Path(f'image_CH/five-fold/2image/group_{group}/mask')
    sv_outside_masks = sorted(sv_outside_mask_path.glob('*.tif'))

    for i in range(5):
        #positiveの面積
        positive = 0

        id_path = Path(f'{dir_path}/{group}/id{i}')
        mask_path = id_path/'mask'
        masks = sorted(mask_path.glob('*.tif'))
        sv_mask, u_masks = masks[:2], masks[2:]

        rp = 0
        rn = 0
        for j in range(len(sv_mask)):
            tmp_m = cv2.imread(str(sv_mask[j]), 0)
            tmp_o = cv2.imread(str(sv_outside_masks[j]),0)
            tmp_rp, tmp_rn, p = svdata_ratio(tmp_m, tmp_o)
            rp += tmp_rp
            rn += tmp_rn
            positive += p

        rp /= j+1
        rn /= j+1
        tmp = cv2.imread(str(u_masks[0]), 0)
        #unlabeled領域の面積（内側）
        unlabeled_region = len(tmp[tmp>0])

        #unlabeled dataの外側マスク
        unlabeled_outside_mask_path = f'image_CH/five-fold/group_{pseudo_group}/mask/0{i}.tif'
        uomask = cv2.imread(unlabeled_outside_mask_path, 0)
        uomask_area = len(uomask[uomask==0])
        #外側の面積
        u_outside = len(uomask[uomask==0])
        #内側の面積
        u_inside = uomask.shape[0]*uomask.shape[1] - u_outside

        #unlabeled領域のpositive(想定)
        u_positive = u_inside*rp
        #unlabeled領域のnegative(想定)
        u_negative = u_inside*rn
        
        #閾値によるpositiveの面積
        threshold_pseudo_path = Path(f'{pseudo_path}/group_{group}/gt')
        th_pseudos = sorted(threshold_pseudo_path.glob('*.tif'))
        tp = cv2.imread(str(th_pseudos[i+2]), 0)
        t_positive = len(tp[tp>0])

        #閾値によるnegativeの面積
        threshold_negative_path = Path(f'{pseudo_path}/group_{group}/mask')
        th_negatives = sorted(threshold_negative_path.glob('*.tif'))
        tn = cv2.imread(str(th_negatives[i+2]), 0)
        PU_neg = len(tn[(uomask>0) & (tn==0)])

        #PUで学習するpositive, negative領域
        PU_pos = u_positive - t_positive + positive

        prior = PU_pos/(PU_pos+PU_neg)

        text_path = f'{dir_path}/{group}/id{i}/prior.txt'
        with open(text_path, mode='w') as f:
            f.write(str(prior))

#閾値による擬似ラベルはpositiveのみ使用 
def calc_th_positive(dir_path,pseudo_path,  group):
    if group+3<5:
        pseudo_group = group+3
    else:
        pseudo_group = group+3-5

    #外側領域のマスク
    sv_outside_mask_path = Path(f'image_CH/five-fold/2image/group_{group}/mask')
    sv_outside_masks = sorted(sv_outside_mask_path.glob('*.tif'))

    for i in range(5):
        #positiveの面積
        positive = 0

        id_path = Path(f'{dir_path}/{group}/id{i}')
        mask_path = id_path/'mask'
        masks = sorted(mask_path.glob('*.tif'))
        sv_mask, u_masks = masks[:2], masks[2:]

        rp = 0
        rn = 0
        for j in range(len(sv_mask)):
            tmp_m = cv2.imread(str(sv_mask[j]), 0)
            tmp_o = cv2.imread(str(sv_outside_masks[j]),0)
            tmp_rp, tmp_rn, p = svdata_ratio(tmp_m, tmp_o)
            rp += tmp_rp
            rn += tmp_rn
            positive += p

        rp /= j+1
        rn /= j+1
        tmp = cv2.imread(str(u_masks[0]), 0)
        #unlabeled領域の面積（内側）
        unlabeled_region = len(tmp[tmp>0])

        #unlabeled dataの外側マスク
        unlabeled_outside_mask_path = f'image_CH/five-fold/group_{pseudo_group}/mask/0{i}.tif'
        uomask = cv2.imread(unlabeled_outside_mask_path, 0)
        uomask_area = len(uomask[uomask==0])
        #外側の面積
        u_outside = len(uomask[uomask==0])
        #内側の面積
        u_inside = uomask.shape[0]*uomask.shape[1] - u_outside

        #unlabeled領域のpositive(想定)
        u_positive = u_inside*rp
        #unlabeled領域のnegative(想定)
        u_negative = u_inside*rn
        
        #閾値によるpositiveの面積
        threshold_pseudo_path = Path(f'{pseudo_path}/group_{group}/gt')
        th_pseudos = sorted(threshold_pseudo_path.glob('*.tif'))
        tp = cv2.imread(str(th_pseudos[i+2]), 0)
        t_positive = len(tp[tp>0])

        #PUで学習するnegative領域
        unlabeled_mask_path = f'{dir_path}/{group}/id{i}/mask/unlabeled_mask.tif'
        unlabeled_mask = cv2.imread(unlabeled_mask_path, 0)
        PU_neg = len(unlabeled_mask[unlabeled_mask==0])

        #PUで学習するpositive
        PU_pos = u_positive - t_positive + positive

        prior = PU_pos/(PU_pos+PU_neg)

        text_path = f'{dir_path}/{group}/id{i}/prior.txt'
        with open(text_path, mode='w') as f:
            f.write(str(prior))

def main(dir_p, pseudo_p):
    for i in range(5):
        # calc(f'{dir_p}/for_PU_train',f'{pseudo_p}/for_train', i)
        calc_th_positive(f'{dir_p}/for_PU_train',f'{pseudo_p}/for_train', i)

if __name__ == '__main__':
    dir_p = f'Result/0411-pu'
    pseudo_p = f'Result/0411-pseudo'
    main(dir_p, pseudo_p)