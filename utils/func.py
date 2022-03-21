import torch
from torch.autograd import Variable
import numpy as np

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def update_predict(model):
    # load pretrained model (rgb+interaction modules)
    # ---- copy model-a to model-b ----
    model_dict = model.state_dict()  # copy base models to the object models
    state_dict = torch.load('../snapshot/FSNet/2021-ICCV-FSNet_rgb-20epoch-new.pth') #
    # ---- for checking state_dict ----
    # for k, v in state_dict.items():
    #     print(k, ':', v.min(), v.max())
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model_dict}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    # load pretrained model (flow)
    # ---- copy model-a to model-b ----
    model_dict = model.state_dict()  # copy base models to the object models
    state_dict = torch.load('../snapshot/FSNet/2021-ICCV-FSNet_flow-20epoch-new.pth') #
    # new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    load_keyword_list = ['resnet.conv1_flow', 'resnet.bn1_flow', 'resnet.layer1_flow', 'resnet.layer2_flow',
                         'resnet.layer3_flow', 'resnet.layer4_flow']
    for keyword in load_keyword_list:
        for k, v in state_dict.items():
            if 'running_mean' not in k and 'running_var' not in k:
                if keyword in k:
                    model_dict[k] = v
                    print('[Checking] load flow branch -> key-name: {}'.format(k))
    # model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)