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


class ps_train_dataset(Dataset):# To use DataLoader for data loading, a custom class inheriting from torch.utils.data.Dataset is often required
    def __init__(self, ann_root, image_root, transform, aug_ss, split, max_words=30):
        ann_file = os.path.join(ann_root, split, 'converted_reformatted.json')
        # ann_file = os.path.join(ann_root, split, 'converted_augment.json')

        anns = json.load(open(ann_file, encoding='utf-8'))
        self.transform = transform

        self.person2text = defaultdict(list) # Initialize a dictionary using defaultdict(list) to store the list of texts corresponding to each person index. When accessing a non-existent key, defaultdict automatically creates an empty list as the default value
        person_id2idx = {} # Mapping relationship from ID to index
        n = 0 # A counter used to assign a unique index to each person
        self.pairs = [] # self.pairs: A list used to store tuples of image path, text description, back-translated text description, and person index

        for ann in anns: # ann is each dictionary element in the list
            image_path = os.path.join(image_root, split ,ann['file_path'])# Get image path
            # image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            if person_id not in person_id2idx.keys(): # Assign an index to each person_id
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])# The operation of multiplying a list by an integer repeats the elements in the list a specified number of times. Here, multiplying the list [''] containing only one empty string by the length of ann['captions'] results in a list of empty strings with the same length as ann['captions']
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):# Use the zip() function to simultaneously iterate through the original text descriptions captions and the back-translated text descriptions captions_bt
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)


            # captions = ann['captions']
            # captions_bt = ann['captions_bt']
            # # Ensure each image has at least two descriptions; duplicate if missing
            # if len(captions) >= 2:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = pre_caption(captions[1], max_words)
            # else:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = caption_1  # If there is only one description, duplicate it
            #
            # # Process back-translated captions_bt
            # if len(captions_bt) >= 2:
            #     bt_1 = pre_caption(captions_bt[0], max_words)
            #     bt_2 = pre_caption(captions_bt[1], max_words)
            # else:
            #     bt_1 = pre_caption(captions_bt[0], max_words)
            #     bt_2 = bt_1  # If there is only one back-translated description, duplicate it
            #
            # self.pairs.append((image_path, caption_1, caption_2, bt_1, bt_2, person_idx))

        self.augmentation_ss = aug_ss
# aug_ss is typically a data augmentation operation specifically designed for Self-Supervised Learning. Self-supervised learning is a learning method without manual data annotation, which allows the model to automatically learn the internal structure and features of data by constructing proxy tasks (such as predicting image rotation angle, determining image crop position, etc.). Its purpose is to generate positive sample pairs (i.e., two different versions of images obtained from the same image through different augmentation operations), allowing the model to learn invariant features of images under different perspectives.
# transform is usually a combined operation of conventional data preprocessing and augmentation. Its purpose is to convert raw image data into a format suitable for model input, while increasing data diversity through common augmentation means (such as resizing, cropping, flipping, etc.) to help the model learn more generalizable features. It mainly focuses on providing diverse but relatively stable input data for model training.
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):# The main function of this method is to obtain the corresponding image path, text description, and other information from the dataset's sample pair list self.pairs based on the given index, then perform a series of processing and data augmentation operations on the image, and finally encapsulate the processed image, text description, and other related information into a dictionary and return it.
        # image_path, caption_1,caption_2,bt_1,bt_2, person = self.pairs[index]
        image_path, caption,caption_bt, person = self.pairs[index]

        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        aug1 = self.transform(image_pil.convert('RGB'))
        # aug_ss_1 = self.augmentation_ss(image_pil)
        # aug_ss_2 = self.augmentation_ss(image_pil) # I have manually changed it to RGB format here
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
        person_id2idx = {}  # Mapping relationship from ID to index
        n = 0  # A counter used to assign a unique index to each person
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, split, ann['file_path'])
            # image_path = os.path.join(image_root, ann['file_path'])

            person_id = ann['id']
            if person_id not in person_id2idx.keys(): # Assign an index to each person_id
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]

            # captions = ann['captions']
            # # Ensure each image has at least two descriptions; duplicate if missing
            # if len(captions) >= 2:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = pre_caption(captions[1], max_words)
            # else:
            #     caption_1 = pre_caption(captions[0], max_words)
            #     caption_2 = caption_1  # If there is only one description, duplicate it
            #
            # self.pairs.append((image_path, caption_1, caption_2, person_idx))

            for caption in ann['captions']:# ann['captions'] stores all text descriptions in the current data entry
                caption = pre_caption(caption, max_words)
                # self.txt2person.append(person_id)# These two correspond words to IDs
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
        caption = ' '.join(caption_words[:max_words])# caption_words[:max_words]: This is a slicing operation that extracts the first max_words elements from the caption_words list.
        # ' '.join(...): This is the string join() method, used to concatenate elements in an iterable object (here a list) into a string. ' ' indicates using a space as the separator during concatenation.
    return caption
