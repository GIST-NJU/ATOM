import traceback
import warnings
import time
import argparse
import os
import sys
import math

from default_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')


# data info
root = "../data"
data_info = [
    {
        "data_name": "voc",
        "data": os.path.join(root, "imgs", "default", "voc"),
        "phase": "test",
        "num_classes": 20,
        "res_path": os.path.join(root, "default", "voc", "results"),
        "inp_name": "files/voc/voc_glove_word2vec.pkl",
        "graph_file": "files/voc/voc_adj.pkl",
    },
    {
        "data_name": "coco",
        "data": os.path.join(root, "imgs", "default", "coco", "coco"),
        "phase": "val",
        "num_classes": 80,
        "res_path": os.path.join(root, "default", "coco", "results"),
        "inp_name": "files/coco/coco_glove_word2vec.pkl",
        "graph_file": "files/coco/coco_adj.pkl",
    },
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
        "batch_size": 8,
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
        "batch_size": 1,
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
        "task_name": "voc_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader": Voc2007Classification,
    },
    # voc mlgcn
    {
        "task_name": "voc_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader": Voc2007Classification,
    },
    # voc dsdl
    {
        "task_name": "voc_dsdl",
        "args": {**data_info[0], **model_info[2]},
        "dataloader": Voc2007Classification,
    },
    # voc asl
    {
        "task_name": "voc_asl",
        "args": {**data_info[0], **model_info[3]},
        "dataloader": Voc2007Classification2,
    },
    # voc mcar
    {
        "task_name": "voc_mcar",
        "args": {**data_info[0], **model_info[4]},
        "dataloader": Voc2007Classification,
    },

    # coco msrn
    {
        "task_name": "coco_msrn",
        "args": {
            **data_info[1], **model_info[0],
            "pool_ratio": 0.05,
            "resume": "checkpoints/msrn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco mlgcn
    {
        "task_name": "coco_mlgcn",
        "args": {
            **data_info[1], **model_info[1],
            "resume": "checkpoints/ml_gcn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco dsdl
    {
        "task_name": "coco_dsdl",
        "args": {
            **data_info[1], **model_info[2],
            "resume": "checkpoints/dsdl/coco_checkpoint.pth.tar",
            "lambd": 0.01,
            "beta": 0.01,
        },
        "dataloader": COCO2014Classification,
    },
    # coco asl
    {
        "task_name": "coco1_asl",
        "args": {
            **data_info[1], **model_info[3],
            "batch_size": 1,
            "model_type": "tresnet_l",
            "model_path": "checkpoints/asl/MS_COCO_TRresNet_L_448_86.6.pth",
            "part": 10,
            "print_freq": 640,
        },
        "dataloader": COCO2014Classification2Partition,
    },
    # coco ml_decoder
    {
        "task_name": "coco_mldecoder",
        "args": {
            **data_info[1], **model_info[5],
            "part": 10,
            "print_freq": 640,
        },
        "dataloader": COCO2014Classification2Partition,
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
        if args.followup:
            args.data_name += "_followup"
            if args.__contains__("part"):
                args.part *= 7
            args.batch_size = math.ceil(args.batch_size / 2)
        try:
            runner(args)
        except Exception as e:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
