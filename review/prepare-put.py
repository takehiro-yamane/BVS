import shutil
from pathlib import Path
from turtle import Turtle
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt


def main(save_p, dir_p):
    save_path = Path(f'Result/{save_p}/for_train')
    save_path.mkdir(exist_ok=True, parents=True)
    dir_path = f'Result/{dir_p}/for_train'
    shutil.copytree(dir_path, str(save_p), dirs_exist_ok=True)

    new_mask_path = save_path.parent/'add_pseudo_mask'
    shutil.copytree(str(new_mask_path), str(save_p),dirs_exist_ok=True)
    print('f')

if __name__ == '__main__':
    save_p = '0411-put'
    dir_p = '0411-pseudo'
    main(save_p, dir_p)
