import os, glob, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class VideoObjDataset(data.Dataset):
    def __init__(self, image_root, flow_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.flows = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.flows = sorted(self.flows)
        self.gts = sorted(self.gts)

        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        flow = self.rgb_loader(self.flows[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        flow = self.img_transform(flow)
        gt = self.gt_transform(gt)

        return image, flow, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_train_loader(image_root, flow_root, gt_root, batchsize, trainsize,
                     shuffle=True, num_workers=12, pin_memory=True):

    dataset = VideoObjDataset(image_root, flow_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_loader(data.Dataset):
    def __init__(self, test_root, testsize):
        self.testsize = testsize
        self.images, self.flows = [], []

        for seqName in os.listdir(test_root):
            seqImage = os.path.join(test_root, seqName, 'Frame')
            seqFlow = os.path.join(test_root, seqName, 'OF_FlowNet2')
            self.images += sorted([seqImage + '/' + f for f in os.listdir(seqImage) if f.endswith('.jpg')])[:-1]
            self.flows += sorted([seqFlow + "/" + f for f in os.listdir(seqFlow) if f.endswith('.jpg')])
            assert len(self.images) == len(self.flows)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)

    def __getitem__(self, index):
        assert self.images[index].split('/')[-1] == self.flows[index].split('/')[-1]
        images = self.rgb_loader(self.images[index])
        flows = self.rgb_loader(self.flows[index])

        images = self.transform(images)
        flows = self.transform(flows)

        img_path_list = self.images[index]

        return images, flows, img_path_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_test_loader(test_root, batchsize, trainsize, shuffle=False, num_workers=12, pin_memory=True):

    dataset = test_loader(test_root, trainsize)
    dataset_size = dataset.size

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, dataset_size