from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse

from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.utils.data
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


from loss import PULoss
from network import UNet, PUnet
from segmentation_predict import exec_prediction, prediction
from utils_PU import FeatureImageLoad, ValImageLoad
from PU_predict import fix_model_state_dict



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_loader, optimizer, prior, epoch, writer):
    model.train()
    tr_loss = 0
    p, n = 0, 0
    for batch_idx, data in enumerate(train_loader):
        imgs = data["image"]
        true_masks = data["gt"]
        loss_masks = data["mask"]
        imgs, true_masks, loss_masks = imgs.to(device), true_masks.to(device), loss_masks.to(device)
        
        optimizer.zero_grad()
        output = model(imgs)

        loss_fct = PULoss(prior = prior)
        loss,pos, neg = loss_fct(output, true_masks, loss_masks, device)
        tr_loss += loss.item()
        p += pos.item()
        n += neg.item()
        loss.backward()

        optimizer.step()
    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * args.ftrain_batch_size, len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader), loss.item()))
    # print("Train loss: ", tr_loss)

    str_a = f'Epoch finished ! Loss: {loss}'
    print(str_a+"\n"+"\033[3A",end="")
    writer.add_scalars('loss',{'Loss':tr_loss,
                                'p-loss':p,
                                'n-loss':n}, epoch)
    return tr_loss

# np.whereで得た座標を[[x,y,z]...]のリストに変形
def lc_append(l_coordinate):
    tmp_l = [[l_coordinate[0][j], l_coordinate[1][j], l_coordinate[2][j]] for j in range(len(l_coordinate[0]))] 
    return tmp_l

# imput : 2D Image
# output : list [[x,y,z], [],[],...] where input=255
def calc_area(lossmasks):
    c = np.where(lossmasks!=0)
    c = lc_append(c)
    return c

# input : 2D image, coordinate list [[x,y,z], [],...] 
# output : list of values [v0, v1, ...]
def no_mask_value(img, c):
    value_list = []
    for i in c:
        v = img[i[0],i[1],i[2]]
        value_list.extend(v)
    return value_list

def validation(args, model, device, eval_loader, prior, epoch):
    """Testing"""
    model.eval()
    eval_loss = 0
    torch.cuda.empty_cache()
    for iteration, data in enumerate(eval_loader):
        img = data["image"]
        target = data["gt"]
        lossmask = data["mask"]
        img = img.to(device)
        pred_img = model(img)
        target = target.detach().cpu().numpy()
        pred_img = pred_img.detach().cpu().numpy()
        t = target[lossmask!=0]
        p = pred_img[lossmask!=0]

        loss = (t-p)**2

        eval_loss += loss.mean()
    iteration += 1
    return eval_loss / iteration


def gather_path(train_paths, mode):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
    return ori_paths


def main(save_path, train_path, prior, writer):
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_path",
                        # default=None,
                        default='output',
                        type=str,
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--nnPU",
                        action='store_true',
                        help="Whether to us non-negative pu-learning risk estimator.")
    parser.add_argument("--train_batch_size",
                        default=60,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--ftrain_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for feature training.")
    parser.add_argument("--eval_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2000,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
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
                        "-d",
                        "--device",
                        dest="device",
                        help="device",
                        default='cuda:0',
    )


    args = parser.parse_args()
    args.do_train = True
    args.train_path = [Path(train_path)]
    # args.val_path = [Path(val_path)]


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    output_model_file = save_path

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuda : {}'.format(use_cuda))

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device(args.device)
    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

## load dataset for feature train
    ## feature 
    f_paths = gather_path(args.train_path, 'feature')
    gt_paths = gather_path(args.train_path, "gt")
    mask_paths = gather_path(args.train_path, "mask")
    train_set = FeatureImageLoad(f_paths, gt_paths, mask_paths)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.ftrain_batch_size, shuffle=True, num_workers=2, pin_memory=True)


    # f_paths = gather_path(args.val_path, 'feature')
    # gt_paths = gather_path(args.val_path, 'gt')
    # mask_paths = gather_path(args.val_path, 'mask')
    # val_set = ValImageLoad(f_paths, gt_paths, mask_paths)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.ftrain_batch_size, shuffle=True, num_workers=2, pin_memory=True)

# ## weight
#     weight = torch.load(args.weight_path, map_location="cpu")
#     weight = fix_model_state_dict(weight)
#     model.load_state_dict(weight)

##Network
    model = PUnet(n_classes=1)
    # model.cuda()
    model.to(device)
    # model = torch.nn.DataParallel(model)

##optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)

##train
    if args.do_train:
        pre_val_loss = 10000
        bad = 0
        pre_tr_loss = 10000
        for epoch in tqdm(range(0, args.num_train_epochs)):
            print('starting epoch {}/{}'.format(epoch+1, args.num_train_epochs))
            tr_loss = train(args, model, device, train_loader, optimizer, prior, epoch, writer)
            # val_loss = validation(args, model, device, val_loader, prior, epoch)
            # print(f'val_loss : {val_loss}')
            # if val_loss < pre_val_loss:
            #     print('update best')
            #     torch.save(model.state_dict(), output_model_file)
            #     pre_val_loss = val_loss
            #     bad = 0
            # else:
            #     bad += 1
            #     print("bad ++")
            # print(f'bad={bad}')
            c = args.num_train_epochs / 2
            if epoch>c:
                if tr_loss < pre_tr_loss:
                    pre_tr_loss = tr_loss
                    torch.save(model.state_dict(), output_model_file)

            # test(args, model, device, test_loader, prior, epoch)
        output_model_file  = output_model_file + 'last'
        torch.save(model.state_dict(), output_model_file)

##eval
    # elif args.do_eval:
    #     test(args, model, device, test_loader, prior)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def exec_PUtrain(dir_p, group):
    spdir = Path(f'Result/{dir_p}/{group}')
    spdir.mkdir(exist_ok=True, parents=True)

    for i in range(5):
        main(
            f'{spdir}/id{i}',
            f'Result/{dir_p}/for_PU_train/pred_group{group}/id{i}',
        )

def get_prior(dir_p, group, id):
    file_path = f'Result/{dir_p}/for_PU_train/{group}/id{id}/prior.txt'
    with open(file_path, mode='r') as f:
        data = f.read()
    prior = float(data)
    return prior


def main_for_together(dir_path):
    for j in range(5):
        group = j
        spdir = Path(f'Result/{dir_path}/{group}')
        spdir.mkdir(exist_ok=True, parents=True)
        for i in range(5):
            prior = torch.tensor(get_prior(dir_path, group, i))
            set_seed(0)
            writer = SummaryWriter()
            main(
                f'{spdir}/id{i}',
                f'Result/{dir_path}/for_PU_train/{group}/id{i}',
                prior,
                writer
            )
            writer.close()

if __name__ == "__main__":
    # prior = torch.tensor(0.3)

    dir_path = '0502-pu'
    group = 0
    spdir = Path(f'Result/{dir_path}/{group}')
    spdir.mkdir(exist_ok=True, parents=True)
    
    for i in range(5):
        prior = torch.tensor(get_prior(dir_path, group, i))
        set_seed(0)
        writer = SummaryWriter()
        main(
            f'{spdir}/id{i}',
            f'Result/{dir_path}/for_PU_train/{group}/id{i}',
            prior
        )
        writer.close()

