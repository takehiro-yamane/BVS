import numpy
import cv2
import matplotlib.pyplot as plt

def main(i):
    pseudo_gt_p = f'Result/for_slide_image/gt{i}.tif'
    new_p = f'Result/for_slide_image/for{i}.tif'

    pseudo_gt = cv2.imread(pseudo_gt_p,0)
    new = cv2.imread(new_p)

    new[pseudo_gt!=0]=[0, 70, 255]
    cv2.imwrite(f'Result/for_slide_image/{i}.tif', new)

    print('f')

if __name__ == '__main__':
    for i in range(3):
        main(i)