import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
from utils.dataloader import get_test_loader
from lib.fsnet import FSNet
import argparse
import time
from scipy import misc  # NOTES: scipy=1.2.2
from PIL import Image
import collections


def demo(opt):
    model = FSNet().cuda()
    pretrain = torch.load(opt.model_path)
    if len(opt.gpu_id) > 1:
        # for the multiple gpus
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_id)
        model.load_state_dict(pretrain)
    else:
        # for a single gpu
        new_dict = collections.OrderedDict()
        for k, v in pretrain.items():
            new_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_dict)

    model.eval()

    for dataset in opt.test_dataset:
        save_path = opt.test_save + dataset + '/'
        os.makedirs(save_path, exist_ok=True)

        test_loader, dataset_size = get_test_loader(
            test_root=opt.dataset_path + dataset, batchsize=opt.batchsize,
            trainsize=opt.testsize, shuffle=False, num_workers=3, pin_memory=True)
        with torch.no_grad():
            img_count = 1
            time_total = 0
            for step, data_pack in enumerate(test_loader):
                images, flows, img_paths = data_pack
                bs, _, _, _ = images.size()

                images = images.cuda()
                flows = flows.cuda()

                time_start = time.clock()
                sals, edge = model(images, flows)
                cur_time = (time.clock() - time_start)

                time_total += cur_time

                for index in range(bs):
                    sal = sals[index, :, :, :].unsqueeze(0)
                    tmp = img_paths[index].split('/')
                    os.makedirs(os.path.join(save_path, tmp[-3]), exist_ok=True)
                    sal_name = tmp[-3] + '/' + tmp[-1].replace('.jpg', '.png')

                    gt = Image.open(img_paths[index])
                    gt = np.asarray(gt, np.float32)
                    gt /= (gt.max() + 1e-8)

                    sal = F.upsample(sal, size=(gt.shape[0], gt.shape[1]), mode='bilinear', align_corners=True)
                    sal = sal.sigmoid().data.cpu().numpy().squeeze()
                    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                    misc.imsave(save_path + sal_name, sal)

                    print('[INFO-Test] Dataset: {}, Image: {} ({}/{}), '
                          'TimeCom: {}'.format(dataset, sal_name, img_count, dataset_size, cur_time / bs))
                    img_count += 1
            print("\n[INFO-Test-Done] FPS: {}".format(dataset_size / time_total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=list,
                        default=[0], help='choose the specific device')
    parser.add_argument('--testsize', type=int,
                        default=352, help='the model input')
    parser.add_argument('--test_dataset', type=list,  help='your test dataset name assigned in the img/gt dictionary',
                        default=['DAVIS', 'FBMS', 'SegTrack-V2', 'MCL', 'DAVSOD', 'DAVSOD-Difficult-20', 'DAVSOD-Normal-25'])
    parser.add_argument('--model_path', type=str,
                        default='./snapshot/FSNet/2021-ICCV-FSNet-20epoch-new.pth')
    parser.add_argument('--test_save', type=str,
                        default='./result/FSNet-New/')
    parser.add_argument('--batchsize', type=int,
                        default=32)    # we only set BS=24 for efficient inference
    parser.add_argument('-dataset_path', type=str,
                        default='./dataset/TestSet/')

    option = parser.parse_args()

    demo(opt=option)
