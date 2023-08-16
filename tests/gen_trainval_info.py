import os
import pandas as pd
import torch
from dataloaders.default_voc2 import Voc2007Classification2
from dataloaders.default_coco2 import COCO2014Classification2
import joblib
from itertools import combinations


root = "../data"


if __name__ == "__main__":
    voc_inp = os.path.join("files", "voc", "voc_glove_word2vec.pkl")
    coco_inp = os.path.join("files", "coco", "coco_glove_word2vec.pkl")
    datasets = []
    train_dataset_voc = Voc2007Classification2(os.path.join(root, "imgs", "default", "voc"), "trainval", inp_name=voc_inp)
    val_dataset_voc = Voc2007Classification2(os.path.join(root, "imgs", "default", "voc"), phase="test", inp_name=voc_inp)
    train_dataset_coco = COCO2014Classification2(os.path.join(root, "imgs", "default", "coco", "coco"), "train", inp_name=coco_inp)
    val_dataset_coco = COCO2014Classification2(os.path.join(root, "imgs", "default", "coco", "coco"), phase="val", inp_name=coco_inp)
    datasets.append(("train", "voc", train_dataset_voc))
    datasets.append(("val", "voc", val_dataset_voc))
    datasets.append(("train", "coco", train_dataset_coco))
    datasets.append(("val", "coco", val_dataset_coco))
    for phase, data, dataloader in datasets:
        print(f"processing {data} {phase} data")
        cat2id = dataloader.get_cat2id()
        id2cat = list(cat2id.keys())
        for w in [1, 2]:
            lcs = set()
            for _, _, target in dataloader:
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                lcs = lcs.union(set(combinations(sorted([id2cat[i] for i in torch.nonzero(target).view(-1)]), w)))
            print(sorted(lcs))
            joblib.dump(lcs, os.path.join(root, "info", "trainval_info", f"{data}_{phase}_lcs_{w}.pkl"))

        lc_counts_df = pd.DataFrame(
            columns=["count"],
            index=["|".join(lc) for lc in combinations(sorted(id2cat), 2)]
        ).fillna(0)
        for filename, _, target in dataloader:
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            labels = [id2cat[i] for i in torch.nonzero(target).view(-1)]
            weight = 1
            for lc in combinations(sorted(labels), 2):
                lc_counts_df.loc["|".join(lc), "count"] += weight
        lc_counts_df.sort_values(by="count", ascending=False, inplace=True)
        print(lc_counts_df)
        lc_counts_df.to_excel(os.path.join(root, "info", "trainval_info", f"lc_count_{data}_{phase}.xlsx"))
