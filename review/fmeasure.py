from cProfile import label
from operator import index
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix, precision_score, recall_score
import seaborn as sns

def binarize(npy, th):
    tmp = np.where(npy>=th, 1, 0)
    return tmp

def gif2tif(save, paths):
    for id, p in enumerate(paths):
        cap = cv2.VideoCapture(str(p))
        ret, frame = cap.read()
        tmp = frame[:,:,1]
        tmp = np.squeeze(tmp)
        cv2.imwrite('{}/{:04}.tif'.format(save, id), tmp)

def fscore(savep, prep, gtp):
    pre_path = Path(prep)
    gt_path = Path(gtp)
    # mask_path = Path(maskp)

    pre_paths = sorted(pre_path.glob('*.*'))
    gt_paths = sorted(gt_path.glob('*.*'))
    # mask_paths = sorted(mask_path.glob('*.*'))

    im_num = len(pre_paths)
    pred = []
    ground_truth = []
    for i in range(im_num):
        pre = cv2.imread(str(pre_paths[i]), 0)
        pre = pre/pre.max()
        pre = binarize(pre, 0.5)
        gt = cv2.imread(str(gt_paths[i]), 0)
        gt = gt/gt.max()
        gt = gt.astype('int')
        pred.extend(pre.flatten())
        ground_truth.extend(gt.flatten())

    print(confusion_matrix(ground_truth, pred))
    f1 = f1_score(ground_truth, pred)
    print(f1)

    # array = confusion_matrix(ground_truth, pred)
    # tmp = np.zeros_like((array)).astype('float')
    # tmp[0][0] = array[0][0]/(array[0][0]+array[0][1])
    # tmp[0][1] = array[0][1]/(array[0][0]+array[0][1])
    # tmp[1][0] = array[1][0]/(array[1][0]+array[1][1])
    # tmp[1][1] = array[1][1]/(array[1][0]+array[1][1])


    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.size'] = 16
    # fig, ax = plt.subplots()
    # sns.heatmap(tmp, annot=True, cmap="Blues", vmax=1, vmin=0, annot_kws={"size":28})
    # ax.set_xticklabels(labels=["negative", "positive"], fontsize=22)
    # ax.set_yticklabels(labels=["negative", "positive"], fontsize=22)
    # ax.set_xlabel(xlabel="Predicted label", fontsize=22)
    # ax.set_ylabel(ylabel="True label", fontsize=22)
    # plt.tight_layout()
    # save_path = f'{savep}_{f1:.5}'
    # save_path = save_path.replace('.', '_')
    # plt.savefig(f'{save_path}.tif')

    return ground_truth, pred


def exec_fscore(dir_name, group_num, pre_num):
    # tgt, tpre = fscore(
    #     f'Result/{dir_name}/{group_num}', 
    #     f'Result/{dir_name}/pred_group{group_num}/pre',
    #     f'image/five-fold/group_{pre_num}/gt',
    #     f'image/five-fold/group_{pre_num}/mask'
    #     )
    tgt, tpre = fscore(
        f'Result/{dir_name}/{group_num}', 
        f'Result/{dir_name}/pred_group{group_num}/pre',
        f'image_CH/five-fold/group_{pre_num}/gt'
    )
    return tgt, tpre

def main(dir_name):
    gt = []
    pre = []

    # tgt, tpre = exec_fscore(dir_name, 4, 2)
    # pre.extend(tpre)
    # gt.extend(tgt)

    # tgt, tpre = exec_fscore(dir_name, 0, 3)
    # pre.extend(tpre)
    # gt.extend(tgt)

    # tgt, tpre = exec_fscore(dir_name, 1, 4)
    # pre.extend(tpre)
    # gt.extend(tgt)

    # tgt, tpre = exec_fscore(dir_name, 2, 0)
    # pre.extend(tpre)
    # gt.extend(tgt)

    # tgt, tpre = exec_fscore(dir_name, 3, 1)
    # pre.extend(tpre)
    # gt.extend(tgt)

    tgt, tpre = exec_fscore(dir_name, 4, 0)
    pre.extend(tpre)
    gt.extend(tgt)

    tgt, tpre = exec_fscore(dir_name, 0, 1)
    pre.extend(tpre)
    gt.extend(tgt)

    tgt, tpre = exec_fscore(dir_name, 1, 2)
    pre.extend(tpre)
    gt.extend(tgt)

    tgt, tpre = exec_fscore(dir_name, 2, 3)
    pre.extend(tpre)
    gt.extend(tgt)

    tgt, tpre = exec_fscore(dir_name, 3, 4)
    pre.extend(tpre)
    gt.extend(tgt)

    print(confusion_matrix(gt, pre))
    f1 = f1_score(gt, pre)
    precision = precision_score(gt, pre)
    recall = recall_score(gt, pre)
    print(f1)
    array = confusion_matrix(gt, pre)
    tmp = np.zeros_like((array)).astype('float')
    tmp[0][0] = array[0][0]/(array[0][0]+array[0][1])
    tmp[0][1] = array[0][1]/(array[0][0]+array[0][1])
    tmp[1][0] = array[1][0]/(array[1][0]+array[1][1])
    tmp[1][1] = array[1][1]/(array[1][0]+array[1][1])


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots()
    sns.heatmap(tmp, annot=True, cmap="Blues", vmax=1, vmin=0, annot_kws={"size":28})
    ax.set_xticklabels(labels=["negative", "positive"], fontsize=22)
    ax.set_yticklabels(labels=["negative", "positive"], fontsize=22)
    ax.set_xlabel(xlabel="Predicted label", fontsize=22)
    ax.set_ylabel(ylabel="True label", fontsize=22)
    plt.tight_layout()
    ff = f'{f1:.3}'
    save_path = f'Result/{dir_name}/whole_mean_f:'+ff
    save_path = save_path.replace('.', '_')

    plt.savefig(save_path)

    text_path = f'Result/{dir_name}/score.txt'
    with open(text_path, mode='w') as f:
        f.write(f'f-score : {f1:.5}\n')
        f.write(f'precision : {precision:.5}\n')
        f.write(f'recall : {recall:.5}')

    print('d')

if __name__ == '__main__':
    main('0502/pseudo')
    main('0502/pu')
    main('0502/sv')


    # for i in range(3):
    #     if i+3<5:
    #         pseudo_group = i+3
    #     else:
    #         pseudo_group = i+3-5
    #     tgt, tpre = fscore(
    #         f'Result/0419/pseudo{i}', 
    #         f'Result/0419/pseudo{i}/pre',
    #         f'image_CH/five-fold/group_{pseudo_group}/gt'
    #     )
    #     tgt, tpre = fscore(
    #         f'Result/0419/sv{i}', 
    #         f'Result/0419/sv{i}/pre',
    #         f'image_CH/five-fold/group_{pseudo_group}/gt'
    #     )
    #     tgt, tpre = fscore(
    #         f'Result/0419/pu{i}', 
    #         f'Result/0419/pu{i}/pre',
    #         f'image_CH/five-fold/group_{pseudo_group}/gt'
    #     )
