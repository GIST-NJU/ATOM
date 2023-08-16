import torch.utils.data as data
import json
import os
from PIL import Image
import numpy as np
import torch
import pickle
from .followup import img_followup_list


def getmydata(root, dst_dir, phase, followup=False):
    anno = os.path.join(dst_dir, '{}_anno.json'.format(phase))
    cat2idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
               'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
               'cow': 9, 'dining table': 10, 'dog': 11, 'horse': 12,
               'motorbike': 13, 'person': 14, 'potted plant': 15,
               'sheep': 16, 'sofa': 17, 'train': 18, 'tv': 19}

    img_dirs = os.listdir(root)
    images = []
    labels_list = []
    for img_dir in img_dirs:
        img_dir_path = os.path.join(root, img_dir)
        if not os.path.isdir(img_dir_path):
            continue
        imgs_name = os.listdir(img_dir_path)
        for i in range(len(imgs_name)):
            if imgs_name[i].endswith('.txt'):
                continue
            labelset = imgs_name[i].split("_")[0].split("-")
            labels = [cat2idx[label] for label in labelset]
            images.append(os.path.join(img_dir, imgs_name[i]))
            labels_list.append(labels)
    if followup:
        images = img_followup_list(images, root, dst_dir)
        labels_list = [val for val in labels_list for i in range(7)]
    anno_list = [{"file_name": f, "labels": l} for f, l in zip(images, labels_list)]
    json.dump(cat2idx, open(os.path.join(dst_dir, 'category.json'), 'w'))
    json.dump(anno_list, open(anno, 'w'))


class GoogleVoc(data.Dataset):
    def __init__(self, root, followup=False, transform=None, phase='train', inp_name=None):
        self.src_dir = root
        if followup:
            self.root = root + "_followup"
        else:
            self.root = root
        self.follow = followup
        self.phase = phase
        self.img_list = []
        self.transform = transform
        getmydata(self.src_dir, self.root, phase, followup)
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
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, torch.tensor(self.inp)), target

    def get_cat2id(self):
        return self.cat2idx
