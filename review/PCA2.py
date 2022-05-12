from re import L
import matplotlib.pyplot as plt
import numpy as np
import os
from pyrsistent import l
from skimage import io
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
# from dataloader import load_visualize_data
from torch import optim
import torch.utils.data
import torch.nn as nn
import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
from collections import OrderedDict
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
import joblib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 0.6 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 1.5 # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 7 # 点がある場合のmarker scale


def visualize2(feature, y, savepath):
    X_PP = feature[y == 1]
    # X_UP = feature[label == 0]
    X_UN = feature[y == 0]

    # 上位からの、全体、border30, border20, border5の個数
    # label_U = label[label != 1]
    # U_num = label_U.shape[0]

    # U_border80 = int(U_num * 0.2)
    # U_border60 = int(U_num * 0.4)
    # U_border40 = int(U_num * 0.6)
    # U_border20  = int(U_num * 0.8)
    # # border20, border5のpredの値
    # pred = np.sort(pred)[::-1]
    # pred_border80 = pred[U_border80]
    # pred_border60 = pred[U_border60]
    # pred_border40 = pred[U_border40]
    # pred_border20  = pred[U_border20]

    # # 特徴量を散布図にする
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    # # 5%, 20%の領域を塗りつぶし
    # featxmax, featxmin, featymin, featymax = feature[:, 0].max()+0.05,feature[:, 0].min()-0.05, feature[:, 1].min()-0.05, feature[:, 1].max()+0.05
    # fillx = [pred_border40,pred_border20, pred_border20, pred_border40]
    # filly = [featymin, featymin, featymax, featymax]
    # if push_direction == "P":
    #     plt.fill(fillx, filly, color="orange", alpha=0.2)

    #     fillx = [pred_border60, pred_border40, pred_border40, pred_border60]    
    #     plt.fill(fillx, filly, color="green", alpha=0.2)

    #     fillx = [pred_border80, pred_border60, pred_border60, pred_border80]    
    #     plt.fill(fillx, filly, color="blue", alpha=0.2)

    # fillcolor = "red" if push_direction == "P" else "blue"
    # fillx = [pred_border20, featxmin, featxmin, pred_border20]
    # plt.fill(fillx, filly, color=fillcolor, alpha=0.2)

    # fillx = [featxmax, pred_border80, pred_border80, featxmax]
    # plt.fill(fillx, filly, color="purple", alpha=0.2)

    # プロット
    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", s=2, zorder=3, alpha=0.3, label="Positive", color="red")
    # plt.scatter(X_UP[:, 0], X_UP[:, 1], marker="o", zorder=1, alpha=0.6, label="UP", color="purple")
    plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", s=1, zorder=1, alpha=0.4, label="Unlabeled", color="mediumpurple")

    # dir = "Top" if push_direction == "P" else "Bottom"
    ax.set_xlabel("PU score")
    ax.set_ylabel("First principal component")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
    plt.rc('legend', fontsize=25)
    # plt.axis("off")

    save_path_vector = os.path.join(savepath, f"check_PCA.png")
    plt.savefig(save_path_vector, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", s=2, zorder=1, alpha=0.3, label="Positive", color="red")
    plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", s=1, zorder=3, alpha=0.4, label="Unlabeled", color="mediumpurple")
    
    ax.set_xlabel("PU score")
    ax.set_ylabel("First principal component")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
    plt.rc('legend', fontsize=25)
    # plt.axis("off")

    save_path_vector = os.path.join(savepath, f"check_PCA_02.png")
    plt.savefig(save_path_vector, bbox_inches='tight', pad_inches=0.1)

    plt.close()

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def all_range_f(gt_paths, f_paths, b_paths):
    lm = int(len(gt_paths))
    # q = int(lm/4)
    # q = int(lm/6)
    v_list = []

    with ProcessPoolExecutor(10) as e:
        r1 = e.submit(feature_value_df, 0, lm, gt_paths, f_paths, b_paths)
        # r2 = e.submit(feature_value_df, 1, 2, gt_paths, f_paths, b_paths)
        # r3 = e.submit(feature_value_df, 2, 3, gt_paths, f_paths, b_paths)
        # r4 = e.submit(feature_value_df, 3, lm, gt_paths, f_paths, b_paths)
        # r5 = e.submit(feature_value_df, q*4, q*5, gt_paths, f_paths, b_paths)
        # r6 = e.submit(feature_value_df, q*5, q*6, gt_paths, f_paths, b_paths)
        # r7 = e.submit(feature_value_df, q*6, q*7, gt_paths, f_paths, b_paths)
        # r8 = e.submit(feature_value_df, q*7, q*8, gt_paths, f_paths, b_paths)
        # r9 = e.submit(feature_value_df, q*8, q*9, gt_paths, f_paths, b_paths)
        # r10 = e.submit(feature_value_df, q*9, lm, gt_paths, f_paths, b_paths)

    # rets = [r1 ,r2 ,r3, r4, r5, r6, r7, r8, r9, r10]
    # rets = [r1 ,r2 ,r3, r4]
    # rets = [r1 ,r2 ,r3, r4, r5, r6]
    # l = ex_list(v_list, rets)
    l = r1.result()
    return l

def ex_list(lis, rets):
    for r in rets:
        lis.extend(r.result())
    return lis

def feature_value_df(s, g, gt_paths, f_paths, b_paths):
    counter = 0
    value_list = []
    for id in tqdm(range(s, g), leave=False):
        gt = gt_paths[id]
        f = f_paths[id]
        b = b_paths[id]
        gt = np.load(str(gt))
        f = np.load(str(f))
        b = cv2.imread(str(b), 0)

        g = np.where(gt==1)
        g = lc_append(g)
        c = np.where(b>0)
        c = lc_append(c)
        counter += len(c)
        tmp_list = get_value_list(f, g, c)
        value_list.extend(tmp_list)

    # print(counter)('int')
    return value_list

def get_value_list(f, g, c):
    value_list = []
    for ii, i in enumerate(c):
        tmp_v = np.zeros((64+1))
        for j in range(f.shape[0]):
            tmp_v[j] = f[j,i[0],i[1]]
        if i in g:
            tmp_v[64] = 1
        else:
            tmp_v[64] = 0
        value_list.append(tmp_v)
    return value_list

def lc_append(l_coordinate):
    tmp_l = [[l_coordinate[0][j], l_coordinate[1][j]] for j in range(len(l_coordinate[0]))] 
    return tmp_l

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# gt, pre, mask 読み込み
def reads(gt_paths, pre_paths, mask_paths):
    label_c = []
    pred_c = []
    for gt, pred, mask in tqdm(zip(gt_paths, pre_paths, mask_paths), total=len(gt_paths)):
        mask = cv2.imread(str(mask),0)
        label = np.load(gt)
        label = label[mask!=0]

        pred = np.load(pred)
        pred = np.squeeze(pred)
        pred = pred[mask!=0]

        label_c.extend(label)
        pred_c.extend(pred)
    return label_c, pred_c


def feature(gt_paths, feature_paths, mask_paths):
    l = len(gt_paths)
    for i in range(l):
        gt = np.load(str(gt_paths[i]))
        feature = np.load(str(feature_paths[i]))
        mask = cv2.imread(str(mask_paths[i]), 0)
        
        gt_arr = np.zeros_like(feature)
        for i in range(feature.shape[0]):
            gt_arr[i]=gt
        


def score_pca(save_path, feature_path, gt_path, pred_path, weight_path, mask_path):
    matplotlib_init()

    #weight of 1x1 conv
    grad = weight_path

    #ground truth -1 or 1
    gt_paths = sorted(gt_path.glob('*.npy'))
    #output of prediction
    pred_paths = sorted(pred_path.glob('*.npy'))
    #mask
    mask_paths = sorted(mask_path.glob('*.tif'))
    mask_paths = [mask_paths[2]]
    #64ch vector
    feature_paths = sorted(feature_path.glob('*.npy'))

    if not len(gt_paths)==len(pred_paths)==len(mask_paths)==len(feature_paths):
        raise ValueError("size of 'gt', 'pre' and 'mask' must be same.")


    # feature(gt_paths, feature_paths, mask_paths)

    # with open('/home/kazuya/experiment_unet/pseudo_label/extract_feature/1125/list_all.txt', 'rb') as f:
    #     v_list = joblib.load(f)

    # v_list = feature_value_df(gt, feature, mask)
    v_list = all_range_f(gt_paths, feature_paths, mask_paths)
    df = pd.DataFrame(v_list)
    X, y = df.drop([64], axis=1), df[64]
    # X, y = df.drop(['64'], axis=1), df['64']
    yy = y.unique()
    yy = yy.astype('int')
    X = X.values
    

    os.makedirs(save_path, exist_ok=True)

    # いろいろ読み込み
    # X = np.load(feature)
    X = X / X.max()
    # Xmin = X.min()
    # Xmax = X.max()
    # X = (X - Xmin) / (Xmax - Xmin)
    # # X = X if PN = "P" else -X

    label, pred = reads(gt_paths, pred_paths, mask_paths)
    label, pred = np.array(label), np.array(pred)


    # pred = cv2.imread(pred,0)

    ## U-Net 最後の1-1convの重み
    grad = torch.load(grad, map_location="cpu")
    grad = fix_model_state_dict(grad)
    grad = grad['outc.weight'].numpy()
    grad = np.squeeze(grad)


    # gradgrad = np.dot(grad, grad)
    # gradp    = np.dot(grad, X.transpose())
    # alpha    = gradp / gradgrad
    alpha = -np.dot(grad, X.transpose()) / np.dot(grad, grad)
    X = X + alpha[:, np.newaxis] * grad

    # PCA
    decomp = PCA(n_components=2)
    decomp.fit(X)

    X2 = np.transpose(X)
    X_decomp = np.array([pred, decomp.components_[0].dot(X2)]).transpose()
    grad = np.array([grad.dot(grad), decomp.components_[0].dot(grad)])

    # 可視化
    visualize2(X_decomp, y, save_path)

    save_path_vector = os.path.join(save_path, f"X_decomp.txt")
    np.savetxt(save_path_vector, X_decomp)

def exec_score_pca(dir_p, group):
    for i in range(5):
        s_path = Path(f'Result/{dir_p}/{group}/PCA{i:02}')
        s_path.mkdir(exist_ok=True, parents=True)
        score_pca(
            s_path,
            Path(f'Result/{dir_p}/for_PU_pre/group{group}/feature/id{i}'),
            Path(f'Result/{dir_p}/for_PU_pre/group{group}/gt/id{i}'),
            Path(f'Result/{dir_p}/for_PU_pre/group{group}/pred/id{i}'),
            Path(f'Result/{dir_p}/{group}/id{i}'),
            Path(f'Result/{dir_p}/for_PU_train/{group}/id{i}/mask')
        )

def main(dir_p):
    # dir_p = '0415-pu'
    for i in range(5):
        exec_score_pca(dir_p, i)
    print("finished")

if __name__ == "__main__":
    dir_p = '0502-pu'
    main(dir_p)
