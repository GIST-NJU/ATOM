import json
import os
from PIL import Image
import numpy as np
import pickle
from torchvision.datasets.vision import VisionDataset
from .followup import img_followup_list


def getmydata(root, dst_dir, phase, part: (int, int), followup=False):
    anno = os.path.join(dst_dir, '{}_p{}_anno.json'.format(phase, part[0]))
    cat2idx = {
        'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
        'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
        'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
        'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
        'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
        'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40,
        'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
        'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56,
        'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
        'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70,
        'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77,
        'hair dryer': 78, 'toothbrush': 79
    }

    img_dirs = os.listdir(root)
    images = []
    labels_list = []
    for img_dir in list(np.array_split(img_dirs, part[1])[part[0]]):
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


class GoogleCoco2Partition(VisionDataset):
    def __init__(self, root, part: (int, int), followup=False, transform=None, phase='val', inp_name=None):
        self.src_dir = root
        if followup:
            self.root = root + "_followup"
        else:
            self.root = root
        self.follow = followup
        self.phase = phase
        self.part = part
        self.img_list = []
        self.transform = transform
        getmydata(self.src_dir, self.root, phase, part, followup)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root,  '{}_p{}_anno.json'.format(self.phase, self.part[0]))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, filename)).convert("RGBA").convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return filename, img, target

    def get_cat2id(self):
        return self.cat2idx
