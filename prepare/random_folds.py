import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
import random

def group(group_num, l, oripaths, gtpaths):
    save_dir = Path('image_CH/five-fold/group_{}'.format(group_num))
    save_dir.mkdir(exist_ok=True, parents=True)
    ori_dir = Path('image_CH/five-fold/group_{}/ori'.format(group_num))
    gt_dir = Path('image_CH/five-fold/group_{}/gt'.format(group_num))
    # mask_dir = Path('dataset/DRIVE_tif/five-fold/group_{}/mask'.format(group_num))
    ori_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    # mask_dir.mkdir(exist_ok=True)

    for i in l:
        ori = cv2.imread(str(oripaths[i]), 0)
        gt = cv2.imread(str(gtpaths[i]), 0)
        # mask = cv2.imread(str(maskpaths[i]), 0)
        cv2.imwrite('{}/{:04}.tif'.format(str(ori_dir), i), ori)
        cv2.imwrite('{}/{:04}.tif'.format(str(gt_dir), i), gt)
        # cv2.imwrite('{}/{:04}.tif'.format(str(mask_dir), i), mask)


def main():
    ori_path = Path('image_CH/ori')
    gt_path = Path('image_CH/gt0')
    # mask_path = Path('dataset/DRIVE_tif/training/mask')
    
    ori_paths = sorted(ori_path.glob('*.*'))
    gt_paths = sorted(gt_path.glob('*.*'))
    # mask_paths = sorted(mask_path.glob('*.tif'))

    l = list(range(len(ori_paths)))
    random.shuffle(l)

    list_0 = l[:5]
    list_1 = l[5:10]
    list_2 = l[10:15]
    list_3 = l[15:20]
    list_4 = l[20:25]

    group(0, list_0, ori_paths, gt_paths)
    group(1, list_1, ori_paths, gt_paths)
    group(2, list_2, ori_paths, gt_paths)
    group(3, list_3, ori_paths, gt_paths)
    group(4, list_4, ori_paths, gt_paths)

    print('d')


if __name__ == '__main__':
    main()