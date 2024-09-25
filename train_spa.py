# -*- coding: utf-8 -*-
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.SPA_Dataloader import TrainData_for_SPA, TestData_for_SPA
from numpy import *
from random import sample
import time

from models import *
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='FADformer', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='spa', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)

def train(train_loader, network, criterion, optimizer, contrastive, iter_num, factor, b_psnr, b_ssim, cnt):
    losses_l1 = AverageMeter()
    losses_con = AverageMeter()

    network.train()

    iter = 0

    best_psnr = b_psnr
    best_ssim = b_ssim
    count = cnt

    for batch in train_loader:

        iter = iter + 1

        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        output = network(source_img)
        l1loss = criterion(output, target_img)
        con_loss = contrastive(output, target_img, source_img)
        # loss = l1loss
        loss = l1loss + 1e-1 * con_loss

        losses_l1.update(l1loss.item())
        losses_con.update(con_loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.01)
        optimizer.step()

        if iter % iter_num == 0:

            train_loss = losses_l1.avg
            train_loss_con = losses_con.avg

            writer.add_scalar('train_loss', train_loss, count)
            writer.add_scalar('train_constrative', train_loss_con, count)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], count)

            scheduler.step()  # TODO
            print(count * iter_num, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            if iter % (iter_num * factor) == 0:

                avg_psnr, avg_ssim = valid(test_loader, network)

                print(avg_psnr, avg_ssim)

                writer.add_scalar('valid_psnr', avg_psnr, count)
                writer.add_scalar('valid_ssim', avg_ssim, count)

                torch.save({'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                                os.path.join(save_dir, args.model + test_str + '_best' + '.pth'))
                    if avg_ssim > 0.9925:
                        torch.save({'state_dict': network.state_dict()},
                                   os.path.join(save_dir, args.model + test_str + '_best_tradeoff' + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, count)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, count)

            losses_l1 = AverageMeter()
            losses_con = AverageMeter()

            network.train()

            count = count + 1

    return best_psnr, best_ssim, count


def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()

    # torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(0, 1)

        psnr_full, sim = calculate_psnr_torch(target_img, output)
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = sim
        SSIM_full.update(ssim_full.item(), source_img.size(0))

    return PSNR_full.avg, SSIM_full.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0, 1]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()

    criterion = nn.L1Loss()
    contrastive = FCR()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], eps=1e-8)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: wrunsupported optimizer")

    # the scheduler setting for spa differs a bit from others because its train_set is too huge, so we choose to use iters_num not epoch_num to update other param
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'] * (638492 // (setting['batch_size'] * setting['iter_num'])),
                                                           eta_min=1e-5)

    train_dir = './datasets/SPAdataset/train'
    test_dir = './datasets/SPAdataset/test'
    train_dataset = TrainData_for_SPA(256, train_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    test_dataset = TestData_for_SPA(8, test_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # change test_str when you development new exp
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    test_str = f"_fadformer_t{timestamp}"

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        best_psnr = 0
        best_ssim = 0
        count = 0

        # the test set for spa needs lot of time so we adjust the freq with iter_nums
        for epoch in tqdm(range(setting['epochs'] + 1)):
            if epoch <= 2:
                factor = 100
            else:
                factor = 1
            best_psnr, best_ssim, count = train(train_loader, network, criterion, optimizer, contrastive, setting['iter_num'], factor, best_psnr, best_ssim, count)

    else:
        print('==> Existing trained model')
        exit(1)
