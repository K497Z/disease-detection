import json
import os
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

from misc.caption_dataset import ps_train_dataset, ps_eval_dataset
from misc.utils import is_using_distributed


def get_self_supervised_augmentation(img_size):#用于生成自监督学习（如 SimCLR）中使用的数据增强管道。这些增强操作旨在通过随机变换生成正样本对，从而帮助模型学习更鲁棒的特征表示
    class GaussianBlur(object):#用于实现高斯模糊增强，但是这个代码运行直接跳过了
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]): #接受一个 sigma 参数，表示高斯模糊的标准差范围（例如 [0.1, 2.0]
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug = transforms.Compose([#将多个增强操作组合成一个管道
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.), antialias=True),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),#p = 0.8 意味着有 80% 的概率会对输入图像应用 ColorJitter 操作
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return aug


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class cuhkpedes_eval(torch.utils.data.Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.pid2txt, self.pid2img = {}, {}
        self.txt_ids, self.img_ids = [], []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            if ann['image_id'] not in self.pid2txt.keys():
                self.pid2txt[ann['image_id']] = []
                self.pid2img[ann['image_id']] = []
            self.pid2img[ann['image_id']].append(img_id)
            self.img_ids.append(ann['image_id'])
            for i, caption in enumerate(ann['caption']):
                self.text.append(caption)
                self.pid2txt[ann['image_id']].append(txt_id)
                self.txt_ids.append(ann['image_id'])
                txt_id += 1

        for tid in range(len(self.text)):
            self.txt2img[tid] = self.pid2img[self.txt_ids[tid]]
        for iid in range(len(self.image)):
            self.img2txt[iid] = self.pid2txt[self.img_ids[iid]]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path)
        image = self.transform(image)

        return image, index


def build_pedes_data(config):#用于构建训练和测试数据加载器（DataLoader）
    size = config.experiment.input_resolution
    if isinstance(size, int):#从配置中获取图像分辨率，并将其转换为 (height, width) 格式，如果 size 是整数，则将其转换为 (size, size)，但这里不是所以直接跳过
        size = (size, size)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#进行归一化操作时所使用的均值（mean）和标准差（std）
    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),#Resize：将图像调整为指定分辨率
        transforms.ToTensor(),#ToTensor：将图像转换为张量
        normalize #Normalize：对图像进行归一化
    ])

    # rand_from = [  # rand_from：包含多种数据增强操作（如颜色抖动、随机旋转、随机裁剪等
    #     # transforms.ColorJitter(.1, .1, .1, 0),
    #     # transforms.RandomRotation(15),
    #     # transforms.RandomResizedCrop(size, (0.9, 1.0), antialias=True),
    #     # transforms.RandomGrayscale(),
    #     # transforms.RandomHorizontalFlip(),
    #     # transforms.RandomErasing(scale=(0.10, 0.20)),
    # ]
    rand_from = [#rand_from：包含多种数据增强操作（如颜色抖动、随机旋转、随机裁剪等
        transforms.ColorJitter(0.3, 0.4, 0.3, 0.1),
        # transforms.ColorJitter(contrast=(1.0,1.2)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size, (0.65, 1.0), antialias=True),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.08, 0.15)),
        transforms.RandomVerticalFlip()
    ]
    aug = Choose(rand_from, size) #从 rand_from 中选择两种数据增强操作
    aug_ss = get_self_supervised_augmentation(size) #aug_ss可以直接应用到图像上


    train_dataset = ps_train_dataset(config.anno_dir, config.image_dir, aug, aug_ss, split='train', max_words=config.experiment.text_length)#这里原本是77，我给改成160了
    test_dataset = ps_eval_dataset(config.anno_dir, config.image_dir, val_transform, split='test', max_words=config.experiment.text_length)

    if is_using_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    test_sampler = None

    config_data = config.data
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_data.batch_size,
        shuffle=train_sampler is None,#数据打乱
        num_workers=config_data.num_workers,#如果 num_workers > 0，DataLoader 会使用多进程加载数据，可能导致子进程崩溃
        pin_memory=True,
        sampler=train_sampler,#采样策略
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )

    return {
        'train_loader': train_loader,
        'train_sampler': train_sampler,
        'test_loader': test_loader,
    }


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Choose:
    def __init__(self, rand_from, size):
        self.choose_from = rand_from
        self.size = size

    def __call__(self, image):
        aug_choice = np.random.choice(self.choose_from, 3)#使用 np.random.choice 从 self.choose_from 中随机选择两个增强操作
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            *aug_choice,#动态插入随机选择的两个增强操作
            normalize
        ])(image)
