from json import load
from turtle import back
from numpy.matrixlib.defmatrix import _convert_from_string
from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from utils_seg.load_mask import RetinaImageLoad, ValImageLoad, NomaskValImageLoad
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from network import UNet
# from networks import VNet
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import os
from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix, precision_score, recall_score


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="./image/train",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="./image/val",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask_path",
        dest="mask_path",
        help="lossmask data path",
        default="./image/mask",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", default=True, help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=32, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=200, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "-lf",
        "--loss_function",
        dest="loss_function",
        help="loss_function",
        default='f',
    )

    args = parser.parse_args()
    return args

class _TrainBase:
    def __init__(self, args):
        self.writer = SummaryWriter()

        ori_paths = self.gather_path(args.train_path, "ori")
        gt_paths = self.gather_path(args.train_path, "gt")
        mask_paths = self.gather_path(args.train_path, "mask")
        data_loader = RetinaImageLoad(ori_paths, gt_paths, mask_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        self.number_of_traindata = data_loader.__len__()

        ori_paths = self.gather_path(args.val_path, "ori")
        gt_paths = self.gather_path(args.val_path, "gt")
        mask_paths = self.gather_path(args.val_path, "mask")
        # data_loader = ValImageLoad(ori_paths, gt_paths, mask_paths)
        data_loader = NomaskValImageLoad(ori_paths, gt_paths, mask_paths)
        self.val_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=5, shuffle=False, num_workers=2, pin_memory=True
        )

        
        self.save_weight_path = Path(args.weight_path)
        # self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        # self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
        #     parents=True, exist_ok=True
        # )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = args.net

        self.train = None
        self.val = None

        self.N_train = None
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        #self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
        return ori_paths

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        #plt.plot(x, self.val_losses)
        # plt.show()

    
    def weight_BCE(self, input, target, mask, w):
        eps = 1e-100
        pred = input[mask != 0]
        y = target[mask != 0]
        pred_fore = pred[y!=0]
        pred_back = pred[y==0]
        bce_fore = -((pred_fore+eps).log())
        bce_back = -((1-pred_back+eps).log())
        w_bce_fore = bce_fore * w
        ret = self.get_mean(w_bce_fore, bce_back)
        return ret

    def focal_BCE(self, input, target, mask, ganma=4):
        eps = 1e-100
        pred = input[mask != 0]
        y = target[mask != 0]
        pred_fore = pred[y!=0]
        pred_back = pred[y==0]
        focal_fore = (1-pred_fore)**ganma
        focal_back = pred_back**ganma
        bce_fore = -((pred_fore+eps).log()*focal_fore)
        bce_back = -((1-pred_back+eps).log()*focal_back)
        ret = self.get_mean(bce_fore, bce_back)
        return ret
        
    def get_mean(self, bce_fore, bce_back):
        f = bce_fore.shape[0]
        b = bce_back.shape[0]
        tmp_fore = 0
        tmp_back = 0
        if f != 0:
            tmp_fore = torch.mean(bce_fore)
            tmp_fore *= f
        if b != 0:
            tmp_back = torch.mean(bce_back)
            tmp_back *= b
        tmp = tmp_fore + tmp_back
        return tmp/(f+b)

class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat, lossmask, epoch, loss_flg='f'):
        if loss_flg == 'w':
            ret = self.weight_BCE(masks_probs_flat, true_masks_flat, lossmask)
        elif loss_flg == 'f':
            ret = self.focal_BCE(masks_probs_flat, true_masks_flat, lossmask)
        else:
            loss = nn.BCELoss()
            input = masks_probs_flat[lossmask!=0]
            target = true_masks_flat[lossmask!=0]
            ret = loss(input, target)

        self.writer.add_scalar('loss', ret, epoch)
        return ret

    def main(self):
        iteration = 0
        for epoch in tqdm(range(self.epochs)):
            # print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.net.train()

            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]
                loss_masks = data["mask"]

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    loss_masks = loss_masks.cuda()

                masks_pred = self.net(imgs)
                loss = self.loss_calculate(masks_pred, true_masks, loss_masks, epoch)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            masks_pred = masks_pred.detach().cpu().numpy()

            self.validation(epoch)

            iteration += 1
        name = self.save_weight_path.name
        torch.save(self.net.state_dict(), str(self.save_weight_path.parent.joinpath('{}_last'.format(name))))
        self.writer.close()
        print('\n\n\n\n')
        print('train finish')

    def validation(self, epoch):
        loss = self.epoch_loss
        # print("Epoch finished ! Loss: {}".format(loss))

        self.losses.append(loss)
        val_dice = self.eval_net(self.val_loader, gpu=self.gpu)
        self.writer.add_scalar('validation_dice', val_dice, epoch)

        # print("val_dice: {}".format(val_dice))
        str_c = 'first'
        try:
            if max(self.val_losses) < val_dice:
                # print("update best")
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                self.bad = 0
                str_c = 'update best'
            else:
                self.bad += 1
                # print("bad ++")
                str_c = 'bad ++     '
        except ValueError:
            torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.val_losses.append(val_dice)
        # else:
        #     print("loss is too large. Continue train")
        #     self.val_losses.append(val_loss)
        str_a = f'Epoch finished ! Loss: {loss}'
        str_b = f'val dice: {val_dice}'
        str_d = f'bad = {self.bad}'
        print(str_a+"\n"+str_b+"\n"+str_c+"\n"+str_d+"\n"+"\033[5A",end="")
        self.epoch_loss = 0

    def eval_net(self,
        dataset, gpu=True, vis=None, vis_im=None, vis_gt=None
    ):
        self.net.eval()
        # losses = 0
        dice = 0
        torch.cuda.empty_cache()
        for iteration, data in enumerate(dataset):
            img = data["image"]
            target = data["gt"]
            # lossmask = data["mask"]
            if gpu:
                img = img.cuda()
                # target = target.cuda()
                # lossmask = lossmask.cuda()
            pred_img = self.net(img)
            target = target.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            # t = target[lossmask!=0]
            # p = pred_img[lossmask!=0]
            t = target.flatten()
            p = pred_img.flatten()
            p = np.where(p>=0.5, 1, 0)
            t_dice = f1_score(t, p)
            dice += t_dice
        iteration += 1
        return dice / iteration
            

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    return new_state_dict


def train(train_path, val_path, save_weight_path):
    set_seed(0)
    args = parse_args()
    args.gpu = True
    args.train_path = [Path(train_path)]
    args.val_path = [Path(val_path)]
    args.epochs = 5001
    args.weight_path = save_weight_path


    # define model
    net = UNet(n_channels=1, n_classes=1)

    # weight = torch.load(args.weight_path, map_location="cpu")
    # weight = fix_model_state_dict(weight)
    # net.load_state_dict(weight)

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)

    args.net = net
    train = TrainNet(args)
    
    train.main()


def add_train(train_path, val_path, save_weight_path, load_weight_path):
    set_seed(0)
    args = parse_args()
    args.gpu = True
    args.train_path = [Path(train_path)]
    args.val_path = [Path(val_path)]
    args.epochs = 2001
    args.weight_path = save_weight_path

    # define model
    net = UNet(n_channels=1, n_classes=1)

    weight = torch.load(load_weight_path, map_location="cpu")
    weight = fix_model_state_dict(weight)
    net.load_state_dict(weight)

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)

    args.net = net
    train = TrainNet(args)
    
    train.main()



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def exec_train(train_and_save_path, group, val_group):
    train(
        f'image/five-fold/2image/group_{group}',
        f'image/five-fold/group_{val_group}',
        f'Result/{train_and_save_path}/group_{group}',
    )


def exec_add_train(train_and_save_path, load_weight_path, group, val_group):
    add_train(
        f'Result/{train_and_save_path}/for_train/group_{group}',
        f'image_CH/five-fold/group_{val_group}',
        f'Result/{train_and_save_path}/group_{group}',
        f'Result/{load_weight_path}/group_{group}'
    )


def main_for_together_add(tspath, lwpath):
    for i in range(5):
        if i+2<5:
            v = i+2
        else:
            v = i+2-5
        exec_add_train(tspath, lwpath, i, v)


if __name__ == "__main__":
    # wei = 2
    # gan = 4

    # loss_flg = 'w'
    # loss_flg = 'f'
    # loss_flg = 0

    tspath = '0505-pseudo'
    lwpath = '0316/Focal/2image'

    main_for_together_add(tspath, lwpath)

    tspath = '0505-pseudo2'
    main_for_together_add(tspath, lwpath)


