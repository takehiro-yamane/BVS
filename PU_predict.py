import logging
import argparse
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# from .model import PUModel
from loss import PULoss
from network import UNet, PUnet
from utils_PU import CellImageLoad, PredImageLoad

from review import vis_score, PU_pre_prepare

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def gather_path(train_paths, mode):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
    return ori_paths

def predict(save_path, input_path, weight_path):
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        # default=None,
                        default='result/pre',
                        type=str,
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run prediction.")    
    parser.add_argument("--ex_feature",
                        action='store_true',
                        help="Whether to run extruction of feature.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument("--predict_batch_size",
                        default=60,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument(
                        "-t",
                        "--predict_path",
                        dest="predict_path",
                        help="training dataset's path",
                        default="./image/feature_MIP1209",
                        type=str,
    )
    parser.add_argument(
                        "-w",
                        "--weight_path",
                        dest="weight_path",
                        help="save weight path",
                        default="./weight/best.pth",
    )


    args = parser.parse_args()
    args.do_predict = True
    args.ex_feature = True
    args.weight_path = weight_path
    args.output_dir = save_path
    args.predict_path = input_path

    if not args.do_predict:
        raise ValueError("At least one of `do_predict` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_predict:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

##load dataset for predict
    ori_path = Path(args.predict_path)
    paths = sorted(ori_path.glob('*.npy'))
    # paths = sorted(ori_path.glob('*.tif'))
    # pred_p = [Path(args.predict_path)]
    # ori_paths = gather_path(pred_p, "ori")
    # train_set = PredImageLoad(ori_paths)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.predict_batch_size, shuffle=False, **kwargs)


##Network
    # model = UNet(n_channels=1, n_classes=1)
    model = PUnet(n_classes=1)
    weight = torch.load(args.weight_path, map_location="cpu")
    weight = fix_model_state_dict(weight)
    model.load_state_dict(weight)

    # torch.save(model.outc.conv.state_dict(), '/home/kazuya/pu-learning-master/result/last_conv')

    model.cuda()

## pred

    output_path = Path(args.output_dir)
    # exfeature2(train_loader, model, out1, out2)

    if args.ex_feature and args.do_predict:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)

        for i, path in enumerate(paths):
            ori = np.load(str(path))
            exfeature(ori, model, output_path, i)

    # elif args.do_predict:
    #     for i, path in enumerate(tqdm(paths)):
    #         ori = cv2.imread(str(path), 0)
    #         pre_img = pred(ori, model)
    #         cv2.imwrite('{}/{:04}.tif'.format(args.output_dir, i), pre_img)
    #         if i == 0:
    #             p = pre_img
    #             p = np.expand_dims(p,2)

    #         else:
    #             pre_img = np.expand_dims(pre_img, 2)
    #             p = np.concatenate([p, pre_img], 2)

    #     np.save(args.output_dir, p)

    # elif args.ex_feature:
    #     output_path = Path(args.output_dir)
    #     output_path = output_path/'feature'
    #     output_path.mkdir(exist_ok=True)
    #     for i, path in enumerate(tqdm(paths)):
    #         ori = cv2.imread(str(path), 0)
    #         _ = exfeature(ori, model, output_path, i)

## predict
def pred(ori, model):
    model.eval()
    img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1])
        )

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        mask_pred = model(img)

    pre_img = mask_pred.detach().cpu().numpy()[0, 0]
    pre_img = (pre_img * 255).astype(np.uint8)
    return pre_img

## extract feature
def exfeature(ori, model, output_path, idx):
    model.eval()
    # img = (ori.astype(np.float32) / ori.max()).reshape(
    #         (1, ori.shape[0], ori.shape[1], ori.shape[2])
    #     )
    img = ori.reshape(
            (1, ori.shape[0], ori.shape[1], ori.shape[2])
        )


    with torch.no_grad():
        # img = torch.from_numpy(img).unsqueeze(0)
        img = torch.from_numpy(img)
        img = img.cuda()
        pre = model(img)
    # mask_pred *= 255
    # pre *= 2
    # pre -= 1
    mask_pred = pre.detach().cpu().numpy()
    mask_pred = np.squeeze(mask_pred, 0)
    np.save('{}/{:04}'.format(str(output_path), idx), mask_pred)

def exfeature2(train_loader, model, out1, out2):
    model.eval()
    for idx, data in enumerate(train_loader):
        imgs = data["image"]
        with torch.no_grad():
            img = torch.from_numpy(imgs).unsqueeze(0)
            img = img.cuda()
            pre, mask_pred = model.forward2(img)
        mask_pred = torch.sigmoid(mask_pred)
        mask_pred *= 255
        mask_pred = mask_pred.detach().cpu().numpy()
        mask_pred = np.squeeze(mask_pred, 0)
        np.save('{}/{:04}'.format(str(out1), idx), mask_pred)

        pre_img = pre.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        cv2.imwrite('{}/{:04}.tif'.format(str(out2), idx), pre_img)

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def exec_predict(dir_path, group):
    for i in range(5):
        save_p = Path(f'Result/{dir_path}/for_PU_pre/group{group}/pred/id{i}')
        save_p.mkdir(exist_ok=True, parents=True)

        predict(
            str(save_p),
            f'Result/{dir_path}/for_PU_pre/group{group}/feature/id{i}',
            f'Result/{dir_path}/{group}/id{i}'
        )

def main(dir_path):
    for i in range(5):
        exec_predict(dir_path, i)

if __name__ == "__main__":

    d_p = '0415-pu'
    PU_pre_prepare.main(d_p)
    main(d_p)
    vis_score.main(d_p)
