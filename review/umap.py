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


def all_range_f(gt_paths, f_paths, b_paths, p_paths):
    lm = int(len(gt_paths))
    v_list = []
    l = feature_value_df(0, lm, gt_paths, f_paths, b_paths, p_paths)
    # with ProcessPoolExecutor(10) as e:
    #     r1 = e.submit(feature_value_df, 0, lm, gt_paths, f_paths, b_paths, p_paths)
    # l = r1.result()
    return l

def feature_value_df(s, g, gt_paths, f_paths, b_paths, p_paths):
    counter = 0
    value_list = []
    for id in tqdm(range(s, g), leave=False):
        gt = gt_paths[id]
        f = f_paths[id]
        b = b_paths[id]
        p = p_paths[id]
        gt = cv2.imread(str(gt), 0)
        gt = gt/gt.max()
        f = np.load(str(f))
        b = cv2.imread(str(b), 0)
        p = cv2.imread(str(p), 0)
        pp = np.copy(p)
        pp[b==0]=255

        g = np.where(gt==1)
        g = lc_append(g)

        c = np.where(pp==0)
        c = lc_append(c)

        counter += len(c)
        lm = len(c)
        with ProcessPoolExecutor(10) as e:
            r1 = e.submit(get_value_list,0, lm, f, g, c)
        tmp_list = r1.result()
        value_list.extend(tmp_list)
    # print(counter)('int')
    return value_list

def get_value_list(start, goal, f, g, c):
    value_list = []
    for ii in range(start, goal):
        i = c[ii]
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

def main(gt_path, feature_path, mask_path, mask_pseudo_path):
    gt_path = Path('Result/0224-vis_feature/group_0/gt')
    feature_path = Path('Result/0224-vis_feature/group_0/feature')
    mask_path = Path('Result/0224-vis_feature/group_0/mask')
    mask_pseudo_path = Path('Result/0224-vis_feature/group_0/mask_pseudo')
    

    gt_paths = sorted(gt_path.glob('*.tif')) 
    feature_paths = sorted(feature_path.glob('*.npy')) 
    mask_paths = sorted(mask_path.glob('*.tif')) 
    mask_pseudo_paths = sorted(mask_pseudo_path.glob('*.tif')) 

    # データセットを読み込む
    v_list = all_range_f(gt_paths, feature_paths, mask_paths, mask_pseudo_paths)
    df = pd.DataFrame(v_list)
    X, y = df.drop([64], axis=1), df[64]
    # X, y = df.drop(['64'], axis=1), df['64']
    y = y.astype('int')
    yy = y.unique()
    X = X.values
    X = X / X.max()

    # 次元削減する
    # mapper = PCA(n_components=2)
    mapper = TSNE(n_components=2)
    # mapper.fit(X)
    # feature = mapper.transform(X)
    feature = mapper.fit_transform(X)
    plt.figure(figsize=(6, 6))

    # 結果を二次元でプロットする
    embedding_x = feature[:, 0]
    embedding_y = feature[:, 1]
    for n in yy:
        plt.scatter(embedding_x[y == n],
                    embedding_y[y == n],
                    label=n, alpha=0.4)

    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.savefig('Result/0224-vis_feature/TSNE')
    # plt.show()


if __name__ == '__main__':
    main()