from datetime import datetime
from PIL import Image
from skimage import feature
import torch
import numpy as np
from pathlib import Path
import cv2
from PU_predict import pred
from network import UNet
import argparse
from collections import OrderedDict
from tqdm import tqdm
from utils_seg.load_mask import RetinaImageLoad

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask_path",
        dest="mask_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/detection",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true", default="True"
    )

    args = parser.parse_args()
    return args


class Predict:
    def __init__(self, args):
        self.net = args.net
        self.gpu = args.gpu

        self.ori_path = Path(args.input_path)
        self.ori_paths = sorted(self.ori_path.glob('*.tif'))

        self.mask_path = self.ori_path.parent / 'mask'
        self.mask_paths = sorted(self.mask_path.glob('*.tif'))

        # self.number_of_traindata = data_loader.__len__()

        self.save_ori_path = args.output_path / Path("ori")
        self.save_pred_path = args.output_path / Path("pred")

        self.pred_con_test = Path(args.output_path)


        # self.save_ori_path.mkdir(parents=True, exist_ok=True)
        # self.save_pred_path.mkdir(parents=True, exist_ok=True)


    def pred(self, ori):
        img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1])
        )

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
            mask_pred = self.net(img)

        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    #extract feature
    def extruct_f(self, ori):
        img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1])
        )

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()

            pre, pre_feature = self.net.forward2(img) 
        pre_feature = pre_feature.detach().cpu().numpy()
        pre_feature = np.squeeze(pre_feature, 0)
        # np.save('/home/kazuya/experiment_unet/pseudo_label/extract_feature//1129/feature', pre_feature)
        pre_img = pre.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_feature, pre_img

    def main_feature(self):
        self.net.eval()
        # pred_con_test = Path(args.output_path)
        pred_f_path = self.pred_con_test/'feature'
        pred_pre_path = self.pred_con_test/'pre'
        pred_binary_path = self.pred_con_test/'binary'
        pred_f_path.mkdir(exist_ok=True)
        pred_pre_path.mkdir(exist_ok=True)
        pred_binary_path.mkdir(exist_ok=True)
        # w/o dataloader version
        for i, op, in enumerate(self.ori_paths):
            ori = cv2.imread(str(op),0)
            # mask = cv2.imread(str(self.mask_paths[i]), 0)
            feature , pre_img= self.extruct_f(ori)
            np.save(str(pred_f_path/'feature_{:04}.tif').format(i), feature)
            cv2.imwrite(str(pred_pre_path/'pre_{:04}.tif').format(i), pre_img)
            pre_img = pre_img/pre_img.max()
            # pre_img[mask==0]=0
            pre_b50 = np.where(pre_img>=0.5, 255, 0)
            cv2.imwrite(str(pred_binary_path/'binary_{:04}_50.tif').format(i), pre_b50)


    def main(self):
        self.net.eval()
        # path def
        # ori_path = Path('/home/kazuya/WSISPDR_unet/image/test/ori')
        # pred_con_test = Path('/home/kazuya/experiment_unet/pred_con_test')
        pred_con_test = Path(args.output_path)
        pred_pre_path = pred_con_test/'pre'
        pred_binary_path = pred_con_test/'binary'
        pred_pre_path.mkdir(exist_ok=True)
        pred_binary_path.mkdir(exist_ok=True)


        # w/o dataloader version
        for i, op, in enumerate(self.ori_paths):
            ori = cv2.imread(str(op),0)
            mask = cv2.imread(str(self.mask_paths[i]), 0)
            pre_img = self.pred(ori)
            cv2.imwrite(str(pred_pre_path/'pre_{:04}.tif').format(i), pre_img)
            pre_img = pre_img/pre_img.max()
            pre_img[mask==0]=0

            pre_b50 = np.where(pre_img>=0.5, 255, 0)
            pre_b30 = np.where(pre_img>=0.3, 255, 0)
            cv2.imwrite(str(pred_binary_path/'binary_{:04}_50.tif').format(i), pre_b50)
            # cv2.imwrite(str(pred_binary_path/'binary_{:04}_30.tif').format(i), pre_b30)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def prediction(output_path, weight_path, args):
    args.output_path = Path(output_path)
    args.output_path.mkdir(exist_ok=True)
    args.weight_path = weight_path
    net = UNet(n_channels=1, n_classes=1)
    # weight = torch.load(args.weight_path, map_location="cpu")
    weight = torch.load(args.weight_path, map_location='cuda:0')
    weight = fix_model_state_dict(weight)
    # net.load_state_dict(torch.load(args.weight_path, map_locatResult/0303/for_PCA_gt/group0ion="cpu"))
    net.load_state_dict(weight)

    if args.gpu:
        net.cuda()
        # net = torch.nn.DataParallel(net)

    args.net = net

    pred = Predict(args)
    # pred = PredictFmeasure(args)

    # pred.main()
    pred.main_feature()


def exec_prediction(dir_p, group, args):
    prediction(
        f'Result/{dir_p}/pred_group{group}',
        f'Result/{dir_p}/group_{group}',
        args
    )

def pred_5groups(dir_p):
    args = parse_args()

    for i in range(5):
        if i+1<5:
            preid = i+1
        else:
            preid = i+1-5
        args.input_path = f'image_CH/five-fold/group_{preid}/ori'
        exec_prediction(dir_p, i, args)

def pred_5groups_12image(dir_p):
    args.input_path = 'image_CH/five-fold/group_0/ori'
    exec_prediction(dir_p, 234)
    args.input_path = 'image_CH/five-fold/group_1/ori'
    exec_prediction(dir_p, 340)
    args.input_path = 'image_CH/five-fold/group_2/ori'
    exec_prediction(dir_p, 401)
    args.input_path = 'image_CH/five-fold/group_3/ori'
    exec_prediction(dir_p, '012')
    args.input_path = 'image_CH/five-fold/group_4/ori'
    exec_prediction(dir_p, 123)

if __name__ == "__main__":
    args = parse_args()

    # dir_p = '0418-put'
    # pred_5groups(dir_p)

    for i in range(5):
        if i+3<5:
            uid = i+3
        else:
            uid = i+3-5

        args.input_path = f'image_CH/five-fold/group_{uid}/ori'
        prediction(
            f'Result/0502/sv{i}',
            f'Result/0316/Focal/2image/group_{i}'
        )

        prediction(
            f'Result/0502/pseudo{i}',
            f'Result/0411-pseudo/group_{i}'
        )

        prediction(
            f'Result/0502/pu{i}',
            f'Result/0411-put/group_{i}'
        )

