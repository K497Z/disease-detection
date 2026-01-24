import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset(Dataset):#若要使用 DataLoader 加载数据，往往需要自定义一个继承自 torch.utils.data.Dataset 的类
    def __init__(self, ann_root, image_root, transform, aug_ss, split, max_words=30):
        ann_file = os.path.join(ann_root, split, 'converted_reformatted.json')
        # ann_file = os.path.join(ann_root, split, 'converted_augment.json')

        anns = json.load(open(ann_file, encoding='utf-8'))
        self.transform = transform

        self.person2text = defaultdict(list) #使用 defaultdict(list) 初始化一个字典，用于存储每个人物索引对应的文本列表。当访问不存在的键时，defaultdict 会自动创建一个空列表作为默认值
        person_id2idx = {} # ID 到索引的映射关系
        n = 0 #一个计数器，用于为每个人物分配唯一的索引
        self.pairs = [] #self.pairs：一个列表，用于存储图像路径、文本描述、回译文本描述和人物索引的元组

        for ann in anns: #ann是列表里面每个字典元素
            image_path = os.path.join(image_root, split ,ann['file_path'])#获取图片路径
            # image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            if person_id not in person_id2idx.keys(): #给每个person_id配一个索引
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])#列表与整数相乘的操作会将列表中的元素重复指定的次数。这里将只包含一个空字符串的列表 [''] 乘以 ann['captions'] 的长度，就会得到一个长度与 ann['captions'] 相同、元素都为空字符串的列表
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):#使用 zip() 函数同时遍历原始文本描述 captions 和回译文本描述 captions_bt
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)


            # captions = ann['captions']
            # captions_bt = ann['captions_bt']
            # # 保证每张图片有至少两个描述，缺失就复制一份
            # if len(captions) >= 2:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = pre_caption(captions[1], max_words)
            # else:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = caption_1  # 如果只有一个描述，就复制
            #
            # # 处理回译 captions_bt
            # if len(captions_bt) >= 2:
            #     bt_1 = pre_caption(captions_bt[0], max_words)
            #     bt_2 = pre_caption(captions_bt[1], max_words)
            # else:
            #     bt_1 = pre_caption(captions_bt[0], max_words)
            #     bt_2 = bt_1  # 如果只有一个回译描述，就复制
            #
            # self.pairs.append((image_path, caption_1, caption_2, bt_1, bt_2, person_idx))

        self.augmentation_ss = aug_ss
#aug_ss 通常是专门为自监督学习（Self-Supervised Learning）设计的数据增强操作。自监督学习是一种无需人工标注数据的学习方式，通过构建一些代理任务（如预测图像的旋转角度、判断图像的裁剪位置等）让模型自动学习到数据的内在结构和特征。其目的是生成正样本对（即从同一图像通过不同的增强操作得到的两个不同版本的图像），让模型学习到不同视角下图像的不变特征
#transform 通常是一个常规的数据预处理和增强的组合操作，其目的是将原始图像数据转换为适合模型输入的格式，同时通过一些常见的增强手段（如调整大小、裁剪、翻转等）增加数据的多样性，帮助模型学习到更具泛化能力的特征。它主要侧重于为模型训练提供多样化但相对稳定的输入数据。其目的是将原始图像数据转换为适合模型输入的格式，同时通过一些常见的增强手段（如调整大小、裁剪、翻转等）增加数据的多样性，帮助模型学习到更具泛化能力的特征。它主要侧重于为模型训练提供多样化但相对稳定的输入数据
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):#该方法的主要功能是根据给定的索引 index，从数据集的样本对列表 self.pairs 中获取对应的图像路径、文本描述等信息，然后对图像进行一系列的处理和数据增强操作，最后将处理后的图像、文本描述以及其他相关信息封装成一个字典并返回
        # image_path, caption_1,caption_2,bt_1,bt_2, person = self.pairs[index]
        image_path, caption,caption_bt, person = self.pairs[index]

        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        aug1 = self.transform(image_pil.convert('RGB'))
        # aug_ss_1 = self.augmentation_ss(image_pil)
        # aug_ss_2 = self.augmentation_ss(image_pil) #这里我都手动给改成RGB形式了
        aug_ss_1 = self.augmentation_ss(image_pil.convert('RGB'))
        aug_ss_2 = self.augmentation_ss(image_pil.convert('RGB'))
        return {
            'image': image,
            # 'caption_1': caption_1,
            # 'caption_2': caption_2,
            # 'caption_bt_1': bt_1,
            # 'caption_bt_2': bt_2,
            "caption": caption,
            "caption_bt": caption_bt,
            'id': person,
            'aug1': aug1,
            # 'aug_ss_1': aug_ss_1,
            # 'aug_ss_2': aug_ss_2
        }


class ps_eval_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        # ann_file = os.path.join(ann_root, split , 'converted.json')
        ann_file = os.path.join(ann_root, split , 'converted_reformatted.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform

        # self.text = []
        self.image = []
        # self.txt2person = []
        # self.img2person = []
        self.person2text = defaultdict(list)
        person_id2idx = {}  # ID 到索引的映射关系
        n = 0  # 一个计数器，用于为每个人物分配唯一的索引
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, split, ann['file_path'])
            # image_path = os.path.join(image_root, ann['file_path'])

            person_id = ann['id']
            if person_id not in person_id2idx.keys(): #给每个person_id配一个索引
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]

            # captions = ann['captions']
            # # 保证每张图片有至少两个描述，缺失就复制一份
            # if len(captions) >= 2:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = pre_caption(captions[1], max_words)
            # else:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = caption_1  # 如果只有一个描述，就复制
            #
            # self.pairs.append((image_path, caption_1, caption_2, person_idx))

            for caption in ann['captions']:#ann['captions'] 存储了当前数据条目中所有的文本描述
                caption = pre_caption(caption, max_words)
                # self.txt2person.append(person_id)#这两个是把单词与id对应起来
                self.pairs.append((image_path, caption, person_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # image_path, caption_1,caption_2,person = self.pairs[index]
        image_path, caption, person = self.pairs[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # text = self.text[index]
        return {
            'image': image,
            # 'caption_1': caption_1,
            # 'caption_2': caption_2,
            "caption": caption,
            'id': person,
        }




def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~,])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])#caption_words[:max_words]：这是一个切片操作，从 caption_words 列表中截取前 max_words 个元素。
        #' '.join(...)：这是字符串的 join() 方法，用于将一个可迭代对象（这里是列表）中的元素连接成一个字符串。' ' 表示连接时使用空格作为分隔符
    return caption