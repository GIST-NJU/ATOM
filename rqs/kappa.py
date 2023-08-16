from itertools import combinations
import pandas as pd
import os
import pycm

import warnings
warnings.filterwarnings('ignore')

root = "../data/info/annotation"
labels = {
    "voc": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant",
            "sheep", "sofa", "train", "tv"],
    "coco": ["airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove",
             "bear", "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl",
             "broccoli", "bus", "cake", "car", "carrot", "cat", "cell phone", "chair", "clock",
             "couch", "cow", "cup", "dining table", "dog", "donut", "elephant",
             "fire hydrant", "fork", "frisbee", "giraffe", "hair dryer", "handbag", "horse",
             "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle",
             "mouse", "orange", "oven", "parking meter", "person", "pizza", "potted plant",
             "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink", "skateboard",
             "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
             "surfboard", "teddy bear", "tennis racket", "tie", "toaster", "toilet",
             "toothbrush", "traffic light", "train", "truck", "tv", "umbrella", "vase",
             "wine glass", "zebra"],
    "photo": [
        # Confidential information, labels of PHOTO dataset
    ],
}


def cal_kappa():
    all_vote_df = pd.DataFrame()
    for data in labels.keys():
        for way in [1, 2]:
            print(data, way)
            vote_df = pd.read_excel(os.path.join(
                root, f"{data}{way}annotation.xlsx"
            )).iloc[:, :4]
            vote_df.columns = ["Image", "A", "B", "C"]
            # print(vote_df)
            lcs = list(combinations(labels[data], way))
            vote_df = vote_df[vote_df["Image"].map(
                lambda x: tuple(sorted(x.split("_")[0].split("-"))) in lcs
            )].reset_index(drop=True)
            all_vote_df = pd.concat([all_vote_df, vote_df], axis=0).reset_index(drop=True)
    print(all_vote_df)
    kappas = [
        pycm.ConfusionMatrix(all_vote_df["A"].to_numpy(), all_vote_df["B"].to_numpy(), classes=[0, 1, 2]).overall_stat["Kappa"],
        pycm.ConfusionMatrix(all_vote_df["A"].to_numpy(), all_vote_df["C"].to_numpy(), classes=[0, 1, 2]).overall_stat["Kappa"],
        pycm.ConfusionMatrix(all_vote_df["B"].to_numpy(), all_vote_df["C"].to_numpy(), classes=[0, 1, 2]).overall_stat["Kappa"],
    ]
    kappa = sum(kappas) / 3
    print(kappas)
    print(kappa)
    return


if __name__ == "__main__":
    cal_kappa()
