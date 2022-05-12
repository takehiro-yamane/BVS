import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn import datasets
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 0.6 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 1.5 # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 7 # 点がある場合のmarker scale


def read_data_nomask(gt_paths, feature_paths):
    for i in range(len(gt_paths)):
        gt = cv2.imread(str(gt_paths[i]), 0)
        feature = np.load(str(feature_paths[i]))

        if i == 0 :
            gt0 = gt.flatten()
            a = feature.shape[1]*feature.shape[2]
            ret_feature = feature.reshape(64,a)
            l1 = ret_feature.shape[1]
        else:
            gt1 = gt.flatten()
            a = feature.shape[1]*feature.shape[2]
            tmp_feature = feature.reshape(64,a)
            ret_feature = np.concatenate([ret_feature, tmp_feature], 1)

    #次元削減
    ret_feature = ret_feature.T
    mapper = PCA(n_components=2)
    mapper.fit(ret_feature)
    PCA_feature = mapper.transform(ret_feature)

    # mapper = TSNE(n_components=2)
    # feature = mapper.fit_transform(ret_feature)


    F0 = PCA_feature[:l1, :]
    F1 = PCA_feature[l1:, :]

    # mapper = TSNE(n_components=2)
    # feature = mapper.fit_transform(ret_feature)


    return F0, F1, gt0, gt1


def read_data(gt_paths, feature_paths, mask_paths):
    
    for i in range(len(gt_paths)):
        gt = cv2.imread(str(gt_paths[i]), 0)
        feature = np.load(str(feature_paths[i]))
        mask = cv2.imread(str(mask_paths[i]),0)

        if i == 0 :
            gt0 = gt[mask!=0]
            ret_feature = feature[:,mask!=0]
            l1 = ret_feature.shape[1]
        else:
            gt1 = gt[mask!=0]
            tmp_feature = feature[:,mask!=0]
            ret_feature = np.concatenate([ret_feature, tmp_feature], 1)

    #次元削減
    ret_feature = ret_feature.T
    mapper = PCA(n_components=2)
    mapper.fit(ret_feature)
    PCA_feature = mapper.transform(ret_feature)

    F0 = PCA_feature[:l1, :]
    F1 = PCA_feature[l1:, :]

    # mapper = TSNE(n_components=2)
    # feature = mapper.fit_transform(ret_feature)


    return F0, F1, gt0, gt1

def main(dir_p, train_id):
    print(f'id{id}')
    matplotlib_init()
    train_gt_path= Path(f'{dir_p}/gt')
    train_feature_path = Path(f'{dir_p}/feature')
    train_mask_path = Path(f'{dir_p}/mask')
    
    train_gt_paths = sorted(train_gt_path.glob('*.tif')) 
    train_feature_paths = sorted(train_feature_path.glob('*.npy')) 
    train_mask_paths = sorted(train_mask_path.glob('*.tif')) 

    # unlabeled_gt_path = Path(f'{dir_p}/gt')
    # unlabeled_feature_path = Path(f'{dir_p}/feature')
    # unlabeled_mask_path = Path(f'{dir_p}/mask')

    # unlabeled_gt_paths = sorted(unlabeled_gt_path.glob('*.tif')) 
    # unlabeled_feature_paths = sorted(unlabeled_feature_path.glob('*.npy')) 
    # unlabeled_mask_paths = sorted(unlabeled_mask_path.glob('*.tif')) 

    gt_paths = []
    gt_paths.append(train_gt_paths[train_id])
    gt_paths.append(train_gt_paths[2])
    # gt_paths.append(unlabeled_gt_paths)

    feature_paths = []
    feature_paths.append(train_feature_paths[train_id])
    feature_paths.append(train_feature_paths[2])
    # feature_paths.append(unlabeled_feature_paths)

    mask_paths = []
    mask_paths.append(train_mask_paths[train_id])
    mask_paths.append(train_mask_paths[2])
    # mask_paths.append(unlabeled_mask_paths)


    # データセットを読み込む
    F0, F1, gt0, gt1 = read_data(gt_paths, feature_paths, mask_paths)
    # F0, F1, gt0, gt1 = read_data_nomask(gt_paths, feature_paths)
    print('read_finish')

    F0_p = F0[gt0!=0]
    F0_n = F0[gt0==0]

    F1_p = F1[gt1!=0]
    F1_n = F1[gt1==0]

    plt.figure(figsize=(10, 10))

    # 結果を二次元でプロットする
    plt.scatter(F0_n[:, 0], F0_n[:, 1], marker="*", s=2, zorder=3, alpha=0.3, label="train_data:N", color="darkorange")
    plt.scatter(F0_p[:, 0], F0_p[:, 1], marker="*", s=2, zorder=3, alpha=0.3, label="train_data:P", color="red")

    plt.scatter(F1_n[:, 0], F1_n[:, 1], marker=".", s=3, zorder=3, alpha=0.3, label="unlabeled:N", color="cyan")
    plt.scatter(F1_p[:, 0], F1_p[:, 1], marker=".", s=3, zorder=3, alpha=0.3, label="unlabeled:P", color="blue")


    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.savefig(f'{dir_p}/PCA{train_id}')
    plt.close()


if __name__ == '__main__':
    for i in range(3):
        main(f'Result/for_pca/0419/{i}', 0)
        main(f'Result/for_pca/0419/{i}', 1)