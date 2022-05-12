from pathlib import Path
import pandas as pd
import numpy as np
import cv2

def convert(save_path, gt_path):
    save_path.mkdir(exist_ok=True)
    gts = sorted(gt_path.glob('*.tif'))
    for id, gt in enumerate(gts):
        tmp = cv2.imread(str(gt), 0)
        ret = np.where(tmp!=0,1,0)
        np.save(f'{save_path}/gt{id:03}',ret)

def main():
    group = 0
    spath = Path(f'Result/0302/for_PCA_gt/group{group}')
    go_path = Path(f'Result/0302/for_PCA_gt/gt_ori/group{group}')
    convert(spath, go_path)
    print('f')

    group = 1
    spath = Path(f'Result/0302/for_PCA_gt/group{group}')
    go_path = Path(f'Result/0302/for_PCA_gt/gt_ori/group{group}')
    convert(spath, go_path)
    print('f')

    group = 3
    spath = Path(f'Result/0302/for_PCA_gt/group{group}')
    go_path = Path(f'Result/0302/for_PCA_gt/gt_ori/group{group}')
    convert(spath, go_path)
    print('f')

    group = 2
    spath = Path(f'Result/0302/for_PCA_gt/group{group}')
    go_path = Path(f'Result/0302/for_PCA_gt/gt_ori/group{group}')
    convert(spath, go_path)
    print('f')

    group = 4
    spath = Path(f'Result/0302/for_PCA_gt/group{group}')
    go_path = Path(f'Result/0302/for_PCA_gt/gt_ori/group{group}')
    convert(spath, go_path)
    print('f')

if __name__ == '__main__':
    main()