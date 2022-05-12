#calculate pseudo label path
#for_trainディレクトリの情報にもとづく
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def pos_acc(gt, pos_l):
    term0 = gt==0
    term1 = pos_l==0

    area_gt = gt[term0==0]
    correct_label = pos_l[(term0==0) & (term1==0)]
    area_pos_l = pos_l[term1==0]

    acc = len(correct_label)/len(area_pos_l)
    label_ratio = len(correct_label)/len(area_gt)

    return acc, label_ratio


#内側領域のみで計算（明らかに背景である外側領域は無視）
def neg_acc(gt, neg_l, mask):
    term0 = gt==0
    term1 = neg_l==0
    term2 = mask==0

    area_bg = gt[(term0==1) & (term2==0)]
    area_neg_label = neg_l[(term1==0) & (term2==0)]
    correct_neg_label = neg_l[(term0==1) & (term1==0) & (term2==0)]

    acc = len(correct_neg_label)/len(area_neg_label)
    label_ratio = len(correct_neg_label)/len(area_bg)

    return acc, label_ratio


def calc_acc(dir_path, group):
    pos_label_path = Path(f'Result/{dir_path}/for_train/group_{group}/gt')
    pos_labels = sorted(pos_label_path.glob('*.tif'))
    pos_labels = pos_labels[2:]

    pseudo_label_path = Path(f'Result/{dir_path}/for_train/group_{group}/mask')
    pseudo_labels = sorted(pseudo_label_path.glob('*.tif'))
    pseudo_labels = pseudo_labels[2:]


    if group+3<5:
        gt_id = group+3
    else:
        gt_id = group+3-5
    gt_path = Path(f'image_CH/five-fold/group_{gt_id}/gt')
    gts = sorted(gt_path.glob('*.tif'))
    mask_path = Path(f'image_CH/five-fold/group_{gt_id}/mask')
    masks = sorted(mask_path.glob('*.tif'))

    ret_acc_pos = []
    ret_acc_neg = []
    ret_label_ratio_pos = []
    ret_label_ratio_neg = []

    for i in range(len(pos_labels)):
        pos_l = cv2.imread(str(pos_labels[i]), 0)
        pseudo_l = cv2.imread(str(pseudo_labels[i]), 0)
        gt = cv2.imread(str(gts[i]), 0)
        mask = cv2.imread(str(masks[i]), 0)
        neg_l = np.where(pos_l==0, pseudo_l, 0)

        acc_pos, label_ratio_pos = pos_acc(gt, pos_l)
        acc_neg, label_ratio_neg = neg_acc(gt, neg_l, mask)

        ret_acc_pos.append(acc_pos)
        ret_acc_neg.append(acc_neg)
        ret_label_ratio_pos.append(label_ratio_pos)
        ret_label_ratio_neg.append(label_ratio_neg)

    return np.mean(ret_acc_pos), np.mean(ret_acc_neg), np.mean(ret_label_ratio_pos), np.mean(ret_label_ratio_neg)

def main(dir_path):
    rap=[]
    ran=[]
    rlrp=[]
    rlrn=[]
    for i in range(5):
        accp, accn, lrp, lrn = calc_acc(dir_path,i)
        rap.append(accp)
        ran.append(accn)
        rlrp.append(lrp)
        rlrn.append(lrn)
    print(f'acc pos  :  {np.mean(rap):.5}')
    print(f'acc neg  :  {np.mean(ran):.5}')
    print(f'label ratio pos  :  {np.mean(rlrp):.5}')
    print(f'label ratio net  :  {np.mean(rlrn):.5}')


if __name__ == '__main__':
    dir_path = '/0505-th_test'
    # dir_path = '0505-pseudo'
    # main(dir_path)


    labelratio = [0.37394,0.58878,0.73729,0.82493,0.87795,0.91507,0.942,0.96237,0.97791]
    acc=[0.99868,0.99786,0.99678,0.99541,0.99369,0.99138,0.98795,0.98272,0.97477]
    x = np.arange(10, 51, 5)
    
    plt.figure(figsize=(8,6))
    plt.plot(x, acc, label='accuracy')
    plt.plot(x, labelratio, label='label ratio')
    plt.legend(fontsize=15)
    # plt.xticks([10,15,20,25,30,35,40,45,50])
    plt.xticks(np.arange(10, 51, 5), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('threshold(%)',fontsize=15)

    plt.show()


    