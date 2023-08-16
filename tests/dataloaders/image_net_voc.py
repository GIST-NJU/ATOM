import random
import torch.utils.data as data
import json
import os
from PIL import Image
import numpy as np
import torch
import pickle
import shutil

from .followup import img_followup


def gen_data(root, phase, image_num, followup):
    if followup:
        image_num *= 7
    cat2idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
               'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
               'cow': 9, 'dining table': 10, 'dog': 11, 'horse': 12,
               'motorbike': 13, 'person': 14, 'potted plant': 15,
               'sheep': 16, 'sofa': 17, 'train': 18, 'tv': 19}

    anno_path = os.path.join(root, '{}_anno.json'.format(phase))
    images_path = os.path.join(root, phase)

    if os.path.exists(images_path) and len(os.listdir(images_path)) == image_num:
        imgs_name = os.listdir(images_path)
        print('data {} already exists'.format(phase))
        anno_list = [{"file_name": img_name, "labels": []} for img_name in imgs_name]
    else:
        print("data {} not exists, generating...".format(phase))
        if os.path.exists(images_path):  # remove uncompleted data
            shutil.rmtree(images_path)
        os.makedirs(images_path)
        if not followup:
            imgs_no = random.sample(range(100000), image_num)
            imgs_no.sort()
            imgs_name = ["ILSVRC2012_test_{:0>8d}.JPEG".format(i+1) for i in imgs_no]
            source_path = os.path.join(root, "test")
            for img_name in imgs_name:
                shutil.copy2(os.path.join(source_path, img_name), os.path.join(images_path, img_name))
            anno_list = [{"file_name": img_name, "labels": []} for img_name in imgs_name]
        else:
            source_path = os.path.join(root, phase.replace("_followup", ""))
            imgs_name = img_followup(source_path, images_path)
            anno_list = [{"file_name": img_name, "labels": []} for img_name in imgs_name]
        print("data {} generated".format(phase))
    json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
    json.dump(anno_list, open(anno_path, 'w'))


class ImageNetVoc(data.Dataset):
    def __init__(self, root, phase, image_num, followup=False, transform=None, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform

        gen_data(root, phase, image_num, followup)
        self.get_anno()

        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, self.phase, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, torch.tensor(self.inp)), target

    def get_number_classes(self):
        return self.num_classes

    def get_cat2id(self):
        return self.cat2idx
