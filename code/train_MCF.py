import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='dataset_path', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="MCF_flod0", help='model_name')                               # todo model name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
T = 0.1
Good_student = 0 # 0: vnet 1:resnet

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = LAHeart(base_dir=train_data_path,
                               split='train',
                               train_flod='train0.list',                   # todo change training flod
                               common_transform=transforms.Compose([
                                   RandomCrop(patch_size),
                               ]),
                               sp_transform=transforms.Compose([
                                   ToTensor(),
                               ]))

    labeled_idxs = list(range(16))           # todo set labeled num
    unlabeled_idxs = list(range(16, 80))     # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_input,v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input,r_label = volume_batch2.cuda(), volume_label2.cuda()

            v_outputs = model_vnet(v_input)
            r_outputs = model_resnet(r_input)

            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)

            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)

            if v_loss_seg_dice < r_loss_seg_dice:
                Good_student = 0
            else:
                Good_student = 1

            v_outputs_soft2 = F.softmax(v_outputs, dim=1)
            r_outputs_soft2 = F.softmax(r_outputs, dim=1)

            v_predict = torch.max(v_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]
            r_predict = torch.max(r_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]
            diff_mask = ((v_predict == 1) ^ (r_predict == 1)).to(torch.int32)
            v_mse_dist = consistency_criterion(v_outputs_soft2[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] )
            r_mse_dist = consistency_criterion(r_outputs_soft2[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] )
            v_mse      = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
            r_mse      = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)

            v_supervised_loss =  (v_loss_seg + v_loss_seg_dice) + 0.5 * v_mse
            r_supervised_loss =  (r_loss_seg + r_loss_seg_dice) + 0.5 * r_mse

            v_outputs_clone = v_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = r_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            v_outputs_clone1 = torch.pow(v_outputs_clone, 1 / T)
            r_outputs_clone1 = torch.pow(r_outputs_clone, 1 / T)
            v_outputs_clone2 = torch.sum(v_outputs_clone1, dim=1, keepdim=True)
            r_outputs_clone2 = torch.sum(r_outputs_clone1, dim=1, keepdim=True)
            v_outputs_PLable = torch.div(v_outputs_clone1, v_outputs_clone2)
            r_outputs_PLable = torch.div(r_outputs_clone1, r_outputs_clone2)

            if Good_student == 0:
                Plabel = v_outputs_PLable
            if Good_student == 1:
                Plabel = r_outputs_PLable

            consistency_weight = get_current_consistency_weight(iter_num//150)
            if Good_student == 0:
                r_consistency_dist = consistency_criterion(r_outputs_soft[labeled_bs:, :, :, :, :], Plabel)
                b, c, w, h, d = r_consistency_dist.shape
                r_consistency_dist = torch.sum(r_consistency_dist) / (b * c * w * h * d)
                r_consistency_loss = r_consistency_dist

                v_loss = v_supervised_loss
                r_loss = r_supervised_loss + consistency_weight * r_consistency_loss
                writer.add_scalar('loss/r_consistency_loss', r_consistency_loss, iter_num)

            if Good_student == 1:
                v_consistency_dist = consistency_criterion(v_outputs_soft[labeled_bs:, :, :, :, :],Plabel)
                b, c, w, h, d = v_consistency_dist.shape
                v_consistency_dist = torch.sum(v_consistency_dist) / (b * c * w * h * d)
                v_consistency_loss = v_consistency_dist

                v_loss = v_supervised_loss +  consistency_weight * v_consistency_loss
                r_loss = r_supervised_loss
                writer.add_scalar('loss/v_consistency_loss', v_consistency_loss, iter_num)

            if (torch.any(torch.isnan(v_loss)) or torch.any(torch.isnan(r_loss)) ):
                print('nan find')
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward()
            r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/v_loss', v_loss, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/v_mse', v_mse, iter_num)
            writer.add_scalar('loss/r_loss', r_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)
            writer.add_scalar('loss/r_mse', r_mse, iter_num)
            writer.add_scalar('train/Good_student', Good_student, iter_num)

            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f v_loss_mse : %f r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f r_loss_mse : %f Good_student: %f'  %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(), v_mse.item(),
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(), r_mse.item(),Good_student))

            ## change lr
            if iter_num % 2500 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()
