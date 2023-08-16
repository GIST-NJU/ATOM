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
    cat2idx = {
        "airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball bat": 4,
        "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9,
        "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14,
        "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19,
        "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24,
        "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29,
        "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34,
        "hair dryer": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39,
        "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44,
        "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49,
        "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54,
        "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59,
        "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64,
        "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69,
        "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74,
        "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79
    }

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


class GoogleCoco(data.Dataset):
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

