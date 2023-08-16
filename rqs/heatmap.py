import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import combinations
import joblib
from utils import myround

DATA_ROOT = "../data"
HEAT_VALUE = {
    "con": 1,
    "vul": 0.3,
    "inc": 0.1,
    "com": 0,
    "max": 1,
    "min": 0,
    "invalid": -0.1,
    "no_need": -0.1,
    "replace_str": {
        -0.1: "△",
        -0.09: "▢",
    }
}

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


def get_heat_map(data, title, x, y, data_name):
    fig, ax = plt.subplots(figsize=(len(x) / 2, len(x) / 2))

    if y:
        data[np.triu_indices_from(data)] = np.nan

    im = ax.imshow(
        data,
        cmap=plt.cm.hot_r,
        vmin=HEAT_VALUE["invalid"], vmax=HEAT_VALUE["max"]
    )

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))

    ax.set_xticklabels(x, fontsize=16)
    ax.set_yticklabels(y, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if y:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for i in range(len(y)) or range(1):
        for j in range(len(x)):
            if data[i, j] < 0:
                ax.text(j, i, HEAT_VALUE["replace_str"][data[i, j]], ha="center", va="center", color="r")
            elif data[i, j] < (HEAT_VALUE["max"] - HEAT_VALUE["invalid"]) / 8 * 2:
                ax.text(j, i, "{:.2f}".format(myround(data[i, j], 2)).rstrip("0").rstrip("."), ha="center", va="center",
                        color="k")
            else:
                ax.text(j, i, "{:.2f}".format(myround(data[i, j], 2)).rstrip("0").rstrip("."), ha="center", va="center",
                        color="w")
    # ax.set_title(title)
    fig.tight_layout()

    # cmap = mpl.cm.hot_r
    # newcolors = cmap(np.linspace(0, 1, 256))
    # newcmap = ListedColormap(
    #     newcolors[int(
    #         256 * abs(HEAT_VALUE["invalid"] / (HEAT_VALUE["max"] - HEAT_VALUE["invalid"]) - HEAT_VALUE["min"])
    #     ):]
    # )
    # norm = mpl.colors.Normalize(vmin=HEAT_VALUE["min"], vmax=HEAT_VALUE["max"])
    # if data_name == "voc":
    #     if y:
    #         ax_fcb = fig.add_axes([1, 0.143, 0.03, 0.8305])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb)
    #     else:
    #         ax_fcb = fig.add_axes([0.067, 0.3, 0.918, 0.03])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb, orientation='horizontal')
    # elif data_name == "coco":
    #     if y:
    #         ax_fcb = fig.add_axes([1.01, 0.0397, 0.01, 0.9536])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb)
    #     else:
    #         ax_fcb = fig.add_axes([0.015, 0.45, 0.981, 0.01])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb, orientation='horizontal')
    # else:
    #     if y:
    #         ax_fcb = fig.add_axes([1, 0.0771, 0.03, 0.909])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb)
    #     else:
    #         ax_fcb = fig.add_axes([0.0124, 0.39, 0.9781, 0.03])
    #         fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap), cax=ax_fcb, orientation='horizontal')
    # fcb.ax.tick_params(labelsize=12, width=0.5, length=0.5)
    # fcb.outline.set_linewidth(0.5)

    # plt.show()
    output_path = os.path.join(DATA_ROOT, "heat_map", f"{title}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def reshape1(data_df, x):
    data_df.loc[x["img"], "heat"] = x["heat"]
    return


def reshape2(data_df, x):
    data_df.loc[x["img1"], x["img2"]] = x["heat"]
    data_df.loc[x["img2"], x["img1"]] = x["heat"]
    return


def init_df1(data_df, x):
    data_df.loc[x["title"][0], "heat"] = HEAT_VALUE["no_need"]
    return


def init_df2(data_df, x):
    data_df.loc[x["title"][0], x["title"][1]] = HEAT_VALUE["no_need"]
    data_df.loc[x["title"][1], x["title"][0]] = HEAT_VALUE["no_need"]
    return


def get_heat(x):
    f = x.mean()
    return f


def rq3(result_dfs, heap_map=True):
    mr_name = ['scale', 'rotation', 'contrast', 'saturation', 'brightness', 'sharp', 'gaussian']
    rq3_df1 = pd.DataFrame(columns=["k", "Dataset", "MICS", "0_rate"])
    rq3_df2 = {
        "voc": pd.DataFrame(columns=["bucket"]),
        "coco": pd.DataFrame(columns=["bucket"])
    }
    # rq3_df3 = {
    #     "voc": pd.DataFrame(columns=["bucket"]),
    #     "coco": pd.DataFrame(columns=["bucket"])
    # }
    train_datasets_count = {}
    for data in ["voc", "coco"]:
        train_datasets_count[data] = pd.read_excel(os.path.join(
            DATA_ROOT, "info", "trainval_info", f"lc_count_{data}_train.xlsx"
        ), index_col=0)
        # rq3_df3[data] = train_datasets_count[data][["count"]].copy()
        # rq3_df3[data].reset_index(inplace=True)
        # rq3_df3[data].rename(columns={"count": "bucket", "index": "lc_count"}, inplace=True)
        # rq3_df3[data] = rq3_df3[data][["bucket", "lc_count"]]
    for model, data, way, result_df, _ in result_dfs:
        print(model, data, way)
        lc_df = result_df[["title"]].copy()
        lc_df["title"] = lc_df["title"].apply(lambda x: list(x))
        result_df = result_df[
            ((result_df["mt"] > 0) & (result_df["union_match"] == True) & (result_df["inter_match"] == False)) |
            (result_df["mt"] == 0)
            ]
        result_df = result_df[["img", "pred", "title"] + ["mr_pred_" + mr for mr in mr_name]]

        result_df["img"] = result_df["img"].apply(
            lambda x: x.split("_")[0]
        )
        result_df["heat"] = result_df[["title", "pred"] + ["mr_pred_" + mr for mr in mr_name]].apply(
            lambda x: sum([1 if x["title"].issubset(r) else 0 for r in x[1:]]) / 8,
            axis=1
        )
        result_df = result_df[["img", "heat"]].groupby("img").agg(get_heat).reset_index()
        lc_count_tmp = result_df.copy().rename(columns={"img": "lc"})
        lc_count_tmp["lc"] = lc_count_tmp["lc"].apply(
            lambda x: "|".join(sorted(x.split("-")))
        )
        result_df.sort_values("img", inplace=True)

        if way == 2 and data != "photo":
            rq3_df1.loc[len(rq3_df1)] = [
                way, data.upper(), model.upper(),
                len(result_df[result_df["heat"] == 0]) / len(result_df)
            ]
            result_df_tmp = result_df.copy()
            result_df_tmp["img"] = result_df_tmp["img"].apply(
                lambda x: "|".join(sorted(x.split("-")))
            )
            result_df_tmp.rename(columns={"heat": "bucket"}, inplace=True)
            result_df_tmp = result_df_tmp[["bucket", "img"]]
            result_df_tmp["bucket"] = result_df_tmp["bucket"].apply(
                lambda x: myround(x, 2)
            )
            result_df_tmp[model.upper()] = result_df_tmp["img"].apply(
                lambda x: train_datasets_count[data].loc[x, "count"]
            )
            result_df_tmp = result_df_tmp.groupby("bucket").agg({
                "img": "count", model.upper(): "mean"
            }).reset_index()
            result_df_tmp.rename(columns={"img": f"{model.upper()}_count"}, inplace=True)
            rq3_df2[data] = pd.merge(rq3_df2[data], result_df_tmp, on="bucket", how="outer")

            # result_df_tmp = result_df.copy()
            # result_df_tmp["img"] = result_df_tmp["img"].apply(
            #     lambda x: "|".join(sorted(x.split("-")))
            # )
            # result_df_tmp.rename(columns={"heat": model.upper(), "img": "lc_count"}, inplace=True)
            # result_df_tmp = result_df_tmp[["lc_count", model.upper()]]
            # rq3_df3[data] = pd.merge(rq3_df3[data], result_df_tmp, on="lc_count", how="outer")

        if heap_map:
            if way == 1:
                data_df = pd.DataFrame(columns=["heat"], index=labels[data]).fillna(HEAT_VALUE["invalid"])
                lc_df.apply(lambda x: init_df1(data_df, x), axis=1)
                result_df.apply(lambda x: reshape1(data_df, x), axis=1)
                data_df.sort_values(by="heat", inplace=True, ascending=False)
                data_array = np.array([data_df["heat"]])
            else:
                data_df = pd.DataFrame(columns=labels[data], index=labels[data]).fillna(HEAT_VALUE["invalid"])
                lc_df.apply(lambda x: init_df2(data_df, x), axis=1)
                result_df["img1"] = result_df["img"].apply(
                    lambda x: x.split("-")[0]
                )
                result_df["img2"] = result_df["img"].apply(
                    lambda x: x.split("-")[1]
                )
                result_df.apply(lambda x: reshape2(data_df, x), axis=1)
                data_df["avg"] = data_df.apply(
                    lambda x: x[x >= 0].sum() / len(x[x >= 0]), axis=1
                )
                data_df.sort_values(by="avg", inplace=True, ascending=False)
                data_df.drop("avg", axis=1, inplace=True)
                data_df = data_df[data_df.index.tolist()]
                data_array = np.array(data_df)
            get_heat_map(
                data_array, f"{'google_' if data != 'photo' else ''}{data}_{way}way_{model}",
                data_df.index.tolist() if way == 1 else data_df.columns.tolist(),
                [] if way == 1 else data_df.index.tolist(),
                data
            )

    rq3_df1.loc[len(rq3_df1)] = [
        "-", "-", "-", rq3_df1["0_rate"].mean()
    ]

    print(rq3_df1)
    for data in ["voc", "coco"]:
        rq3_df2[data].sort_values(by="bucket", inplace=True)
        # rq3_df3[data].sort_values(by="bucket", inplace=True, ascending=False, ignore_index=True)
        # rq3_df3[data] = rq3_df3[data].groupby("bucket").agg({
        #     "lc_count": "count", **{col: "mean" for col in rq3_df3[data].columns if col not in ["lc_count", "bucket"]}
        # }).reset_index()
        print(rq3_df2[data])
        # print(rq3_df3[data])
    return
