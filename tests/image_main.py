import traceback
import warnings
import time
import argparse
import os
import sys

from image_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')


# data info
root = "../data"
data_info = [
    {
        "data_name": "imagenet_voc",
        "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
        "way_num": 1,
        "image_num": 100,
        "num_classes": 20,
        "res_path": os.path.join(root, "results"),
        "inp_name": "file/voc/voc_glove_word2vec.pkl",
        "graph_file": "file/voc/voc_adj.pkl",
    },
    {
        "data_name": "imagenet_voc",
        "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
        "way_num": 2,
        "image_num": 898,
        "num_classes": 20,
        "res_path": os.path.join(root, "results"),
        "inp_name": "file/voc/voc_glove_word2vec.pkl",
        "graph_file": "file/voc/voc_adj.pkl",
    },
    {
        "data_name": "imagenet_coco",
        "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
        "way_num": 1,
        "image_num": 400,
        "num_classes": 80,
        "res_path": os.path.join(root, "results"),
        "inp_name": "file/coco/coco_glove_word2vec.pkl",
        "graph_file": "file/coco/coco_adj.pkl",
    },
    {
        "data_name": "imagenet_coco",
        "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
        "way_num": 2,
        "image_num": 13297,
        "num_classes": 80,
        "res_path": os.path.join(root, "results"),
        "inp_name": "file/coco/coco_glove_word2vec.pkl",
        "graph_file": "file/coco/coco_adj.pkl",
    },
    # {
    #     "data_name": "imagenet_photo",
    #     "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
    #     "way_num": 1,
    #     "image_num": 160,
    #     "num_classes": 20,
    #     "res_path": os.path.join(root, "results"),
    #     "inp_name": "data/voc/voc_glove_word2vec.pkl",
    #     "graph_file": "data/voc/voc_adj.pkl",
    # },
    # {
    #     "data_name": "imagenet_photo",
    #     "data": os.path.join(root, "imgs", "ILSVRC2012_img_test_v10102019"),
    #     "way_num": 2,
    #     "image_num": 2305,
    #     "num_classes": 20,
    #     "res_path": os.path.join(root, "results"),
    #     "inp_name": "data/voc/voc_glove_word2vec.pkl",
    #     "graph_file": "data/voc/voc_adj.pkl",
    # },
]

# model info
model_info = [
    # 0 msrn
    {
        "model_name": "msrn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/msrn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "pretrained": 1,
        "pretrain_model": "pretrained/resnet101_for_msrn.pth.tar",
        "pool_ratio": 0.2,
        "backbone": "resnet101",
        "save_model_path": "checkpoints/msrn",
    },
    # 1 ml gcn
    {
        "model_name": "mlgcn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/ml_gcn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "save_model_path": "checkpoints/mlgcn",
    },
    # 2 dsdl
    {
        "model_name": "dsdl",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 200,
        "epoch_step": [40],
        "start_epoch": 0,
        "lr": 0.01,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/dsdl/voc_checkpoint.pth.tar",
        "evaluate": True,
        "lambd": 10.0,
        "beta": 0.0001,
        "device_ids": [0],
        "save_model_path": "checkpoints/dsdl",
    },
    # 3 asl
    {
        "model_name": "asl",
        "model_type": "tresnet_xl",
        "model_path": "checkpoints/asl/PASCAL_VOC_TResNet_xl_448_96.0.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 32,
        "print_freq": 64,
    },
    # 4 mcar
    {
        "model_name": "mcar",
        "image_size": 448,
        "batch_size": 1,
        "threshold": 0.6,
        "bm": "resnet101",
        "ps": "avg",
        "topN": 4,
        "workers": 4,
        "epochs": 60,
        "epoch_step": [30, 50],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/mcar/model_best_94.7850.pth.tar",
        "evaluate": True,
        "save_model_path": "checkpoints/mcar",
    },
    # 5 ml decoder
    {
        "model_name": "mldecoder",
        "model_type": "tresnet_l",
        "model_path": "checkpoints/ml_decoder/tresnet_l_COCO__448_90_0.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.5,
        "batch_size": 8,
        "print_freq": 64,
        "use_ml_decoder": 1,
        "num_of_groups": -1,
        "decoder_embedding": 768,
        "zsl": 0
    },
]

TASKS = [
    # voc msrn
    {
        "task_name": "imagenet_voc1followup_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader":  ImageNetVoc,
    },
    {
        "task_name": "imagenet_voc2followup_msrn",
        "args": {**data_info[1], **model_info[0]},
        "dataloader":  ImageNetVoc,
    },
    # voc mlgcn
    {
        "task_name": "imagenet_voc1followup_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader":  ImageNetVoc,
    },
    {
        "task_name": "imagenet_voc2followup_mlgcn",
        "args": {**data_info[1], **model_info[1]},
        "dataloader":  ImageNetVoc,
    },
    # voc dsdl
    {
        "task_name": "imagenet_voc1followup_dsdl",
        "args": {**data_info[0], **model_info[2]},
        "dataloader":  ImageNetVoc,
    },
    {
        "task_name": "imagenet_voc2followup_dsdl",
        "args": {**data_info[1], **model_info[2]},
        "dataloader":  ImageNetVoc,
    },
    # voc asl
    {
        "task_name": "imagenet_voc1followup_asl",
        "args": {**data_info[0], **model_info[3]},
        "dataloader":  ImageNetVoc2,
    },
    {
        "task_name": "imagenet_voc2followup_asl",
        "args": {**data_info[1], **model_info[3]},
        "dataloader":  ImageNetVoc2,
    },
    # voc mcar
    {
        "task_name": "imagenet_voc1followup_mcar",
        "args": {**data_info[0], **model_info[4]},
        "dataloader":  ImageNetVoc,
    },
    {
        "task_name": "imagenet_voc2followup_mcar",
        "args": {**data_info[1], **model_info[4]},
        "dataloader":  ImageNetVoc,
    },

    # coco msrn
    {
        "task_name": "imagenet_coco1followup_msrn",
        "args": {
            **data_info[2], **model_info[0],
            "pool_ratio": 0.05,
            "resume": "checkpoints/msrn/coco_checkpoint.pth.tar",
        },
        "dataloader":  ImageNetCoco,
    },
    {
        "task_name": "imagenet_coco2followup_msrn",
        "args": {
            **data_info[3], **model_info[0],
            "pool_ratio": 0.05,
            "resume": "checkpoints/msrn/coco_checkpoint.pth.tar",
        },
        "dataloader":  ImageNetCoco,
    },
    # coco mlgcn
    {
        "task_name": "imagenet_coco1followup_mlgcn",
        "args": {
            **data_info[2], **model_info[1],
            "resume": "checkpoints/ml_gcn/coco_checkpoint.pth.tar",
        },
        "dataloader":  ImageNetCoco,
    },
    {
        "task_name": "imagenet_coco2followup_mlgcn",
        "args": {
            **data_info[3], **model_info[1],
            "resume": "checkpoints/ml_gcn/coco_checkpoint.pth.tar",
        },
        "dataloader":  ImageNetCoco,
    },
    # coco dsdl
    {
        "task_name": "imagenet_coco1followup_dsdl",
        "args": {
            **data_info[2], **model_info[2],
            "resume": "checkpoints/dsdl/coco_checkpoint.pth.tar",
            "lambd": 0.01,
            "beta": 0.01,
        },
        "dataloader":  ImageNetCoco,
    },
    {
        "task_name": "imagenet_coco2followup_dsdl",
        "args": {
            **data_info[3], **model_info[2],
            "resume": "checkpoints/dsdl/coco_checkpoint.pth.tar",
            "lambd": 0.01,
            "beta": 0.01,
        },
        "dataloader":  ImageNetCoco,
    },
    # coco asl
    {
        "task_name": "imagenet_coco1followup_asl",
        "args": {
            **data_info[2], **model_info[3],
            "batch_size": 1,
            "model_type": "tresnet_l",
            "model_path": "checkpoints/asl/MS_COCO_TRresNet_L_448_86.6.pth",
        },
        "dataloader":  ImageNetCoco2,
    },
    {
         "task_name": "imagenet_coco2followup_asl",
        "args": {
            **data_info[3], **model_info[3],
            "batch_size": 1,
            "model_type": "tresnet_l",
            "model_path": "checkpoints/asl/MS_COCO_TRresNet_L_448_86.6.pth",
            "part": 10,
        },
        "dataloader":  ImageNetCoco2Partition,
    },
    # coco ml_decoder
    {
        "task_name": "imagenet_coco1followup_mldecoder",
        "args": {**data_info[2], **model_info[5]},
        "dataloader":  ImageNetCoco2,
    },
    {
        "task_name": "imagenet_coco2followup_mldecoder",
        "args": {
            **data_info[3], **model_info[5],
            "part": 15,
        },
        "dataloader":  ImageNetCoco2Partition,
    },
]

if __name__ == "__main__":
    with open("errors.txt", 'w') as f:
        f.write("")
    for task in TASKS:
        print("task: {} started".format(task["task_name"]))
        start = time.time()
        args = argparse.Namespace(**task["args"])
        args.dataloader = task["dataloader"]
        args.followup = True if "--followup" in sys.argv else False
        args.repeat = 5
        args.start_no = 1
        if args.followup:
            args.data_name += "_followup"
        try:
            runner(args)
        except Exception as e:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
