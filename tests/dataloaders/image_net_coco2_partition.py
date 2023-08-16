import random
import json
import os
from PIL import Image
import numpy as np
import pickle
import shutil
import functools
from torchvision.datasets.vision import VisionDataset

from .followup import img_followup


def gen_data(root, phase, image_num, followup, part: (int, int)):
    if followup:
        image_num *= 7
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

    anno_path = os.path.join(root, '{}_p{}_anno.json'.format(phase, part[0]))
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

    def my_compare(x, y):
        if x["file_name"] < y["file_name"]:
            return -1
        elif x["file_name"] > y["file_name"]:
            return 1
        else:
            return 0
    anno_list.sort(key=functools.cmp_to_key(my_compare))

    anno_list = list(np.array_split(anno_list, part[1])[part[0]])
    json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
    json.dump(anno_list, open(anno_path, 'w'))


class ImageNetCoco2Partition(VisionDataset):
    def __init__(self, root, part: (int, int), phase, image_num, followup=False, transform=None, inp_name=None):
        self.root = root
        self.phase = phase
        self.part = part
        self.img_list = []
        self.transform = transform

        gen_data(root, phase, image_num, followup, part)
        self.get_anno()

        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root,  '{}_p{}_anno.json'.format(self.phase, self.part[0]))
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
        img = Image.open(os.path.join(self.root, self.phase, filename)).convert("RGBA").convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return filename, img, target

    def get_number_classes(self):
        return self.num_classes

    def get_cat2id(self):
        return self.cat2idx
