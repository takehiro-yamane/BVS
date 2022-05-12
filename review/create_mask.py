import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def make_mask(save_path, oripaths):
    for id, op in enumerate(oripaths):
        ori = cv2.imread(str(op), 0)
        tmp = np.where(ori==0, 0, 255)
        # カーネルを作成する。
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # tmp = tmp.astype('uint8')
        # 2値画像を収縮する。
        # dst = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations=2)
        # cv2.imwrite(f'{save_path}/{id:02}.tif', dst)
        cv2.imwrite(f'{save_path}/{id:02}.tif', tmp)


def main():
    print('main')

    for i in range(5):
        parent_path = Path(f'image_CH/five-fold/2image/group_{i}')
        oripath = parent_path/'ori'
        oris = sorted(oripath.glob('*.tif'))
        save_path = parent_path/'mask'
        save_path.mkdir(exist_ok=True, parents=True)
        make_mask(save_path, oris)

if __name__ == '__main__':
    main()
