
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_train_loader
from utils.func import AvgMeter, update_predict
from lib.fsnet import FSNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='input size')
    parser.add_argument('--trainset', type=str, default='FSNet')
    parser.add_argument('--train_type', type=str, default='finetune', help='finetune or pretrain_rgb or pretrain_flow')
    opt = parser.parse_args()

    # build models
    model = FSNet().cuda()

    if opt.train_type == 'finetune':
        save_path = '../snapshot/{}/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = '/media/nercms/NERCMS/GepengJi/VSOD/2020-ACMMM/Dataset/train_set_DAVIS_FBMS'
        image_root = src_dir + '/Imgs/'
        flow_root = src_dir + '/Flow/'
        gt_root = src_dir + '/ground-truth/'

        train_loader = get_train_loader(image_root=image_root, flow_root=flow_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
        #
        update_predict(model)
    elif opt.train_type == 'pretrain_rgb':
        save_path = '../snapshot/{}_rgb/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = '/media/nercms/NERCMS/GepengJi/VSOD/2020-ACMMM/Dataset/train_set_DAVIS_FBMS'
        image_root = src_dir + '/Imgs/'
        gt_root = src_dir + '/ground-truth/'

        train_loader = get_train_loader(image_root=image_root, flow_root=image_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
    elif opt.train_type == 'pretrain_flow':
        save_path = '../snapshot/{}_flow/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = '/media/nercms/NERCMS/GepengJi/VSOD/2020-ACMMM/Dataset/train_set_DAVIS_FBMS'
        flow_root = src_dir + '/Flow/'
        gt_root = src_dir + '/ground-truth/'

        train_loader = get_train_loader(image_root=flow_root, flow_root=flow_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
    else:
        raise AttributeError('No train_type: {}'.format(opt.train_type))

    # ---- parallel model ----
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    params = model.parameters()
    optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    BCE = torch.nn.BCEWithLogitsLoss()

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]

    # training
    for epoch in range(0, opt.epoch):
        scheduler.step()
        model.train()
        loss_record1, loss_record2 = AvgMeter(), AvgMeter()

        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- get data ----
                images, flows, gts = pack
                images = Variable(images).cuda()
                flows = Variable(flows).cuda()
                gts = Variable(gts).cuda()
                # ---- multi-scale training ----
                trainsize = int(round(opt.trainsize*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    flows = F.upsample(flows, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                preds = model(images, flows)
                # ---- cal loss ----
                loss1 = BCE(preds[0], gts)
                loss2 = BCE(preds[1], gts)
                loss = loss1 + loss2
                # ---- backward ----
                loss.backward()
                optimizer.step()
                # ---- show loss ----
                if rate == 1:
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
            if i % 25 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(), loss_record2.show()))

        os.makedirs(save_path, exist_ok=True)
        if epoch > 10:
            if (epoch+1) % 1 == 0:
                torch.save(model.state_dict(), save_path + opt.trainset + '-{}epoch.pth'.format(epoch))
                print('[Model Saved] Path: {}'.format(save_path + opt.trainset + '-{}epoch.pth'.format(epoch)))


if __name__ == '__main__':
    main()