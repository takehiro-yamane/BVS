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

#groupごとに，svのgtとgtをもとに作成したmaskをidの数だけ複製
def sv_gt(save_path, gt_path):
    gts = sorted(gt_path.glob('*.tif'))
    mask_save_path = save_path.parent/'mask'
    mask_save_path.mkdir(exist_ok=True)
    for i in range(len(gts)):
        tmp = cv2.imread(str(gts[i]), 0)
        gt = np.where(tmp==0, 0, 1)
        np.save(f'{save_path}/{i:02}', gt)
        cv2.imwrite(f'{mask_save_path}/{i:02}.tif', tmp)

#sv(train data)の推定結果から特徴量をgroupごとにidの数だけ複製
def sv_feature(save_path, feature_path):
    features = sorted(feature_path.glob('*.npy'))
    for i in range(len(features)):
        tmp = np.load(str(features[i]))
        np.save(f'{save_path}/{i:02}', tmp)

def exec_sv(parent_path,group):
    for j in range(5):
        save_path_f = parent_path/f'{group}/id{j}/feature'
        save_path_f.mkdir(exist_ok=True, parents=True)
        feature_path = parent_path/f'sv/{group}/feature'
        # feature_path = Path(f'Result/0411-pu/for_PU_train/sv/{j}/feature')
        sv_feature(save_path_f, feature_path)

        save_path_g = parent_path/f'{group}/id{j}/gt'
        save_path_g.mkdir(exist_ok=True,parents=True)

        gt_path = Path(f'image_CH/five-fold/2image/group_{group}/gt')
        sv_gt(save_path_g, gt_path)

def main(parent_path):
    #group５個分
    for i in range(5):
        exec_sv(parent_path, i)
    print('f')

if __name__ == '__main__':
    parent_path = Path('Result/0411-pu/for_PU_train')
    main(parent_path)