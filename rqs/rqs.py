import os
import sys
from scipy.special import comb
import pandas as pd
import joblib
from itertools import combinations
from utils import myround, merge_df_cells
from heatmap import rq3


root = "../data"
models = [
    ("msrn", "voc"),
    ("mlgcn", "voc"),
    ("dsdl", "voc"),
    ("asl", "voc"),
    ("mcar", "voc"),
    ("msrn", "coco"),
    ("mlgcn", "coco"),
    ("dsdl", "coco"),
    ("asl", "coco"),
    ("mldecoder", "coco"),
    ("photo", "photo"),
]
labels_num = {
    "voc": 20,
    "coco": 80,
    "photo": 32,
}


def rq1(result_dfs):
    rq1_df = pd.DataFrame(
        columns=[
            "k", "Dataset", "|Lk|", "|L|",
            "all", "gt=0", "gt=1", "gt=2",
            "IMR", "LCMR",
        ]
    )
    compare_df = pd.DataFrame(
        columns=[
            "k", "Dataset",
            "TrainSet", "T_Inter", "T_Add", "T_Def",
            "ValSet", "V_Inter", "V_Add", "V_Def",
        ]
    )
    train_val_datasets = {}
    for data in ["voc", "coco"]:
        for w in [1, 2]:
            for phase in ["train", "val"]:
                train_val_datasets[(data, w, phase)] = joblib.load(
                    os.path.join(root, "info", "trainval_info", f"{data}_{phase}_lcs_{w}.pkl"))
    for model, data, way, result_df, _ in result_dfs:
        if model != "msrn" and model != "photo":
            continue
        line = [
            way,  # k
            data.upper(),  # data
            "{:,}".format(int(comb(labels_num[data], way))),  # |L_k|
        ]

        result_df["lcs"] = result_df["title"].apply(lambda x: set(combinations(sorted(x), way)))
        result_df["tmp"] = 0
        lcs_df = result_df[["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
        lcs_df = lcs_df.unstack(level=-1).reset_index()
        lcs = lcs_df.loc[lcs_df["tmp"] == 0, 0][0]
        line.append("{:,}".format(len(lcs)))  # |L|

        line.append("{:,}".format(len(result_df)))  # all
        line.append("{:,}".format(
            len(result_df[result_df["gt"] == 0])
        ))  # gt=0
        line.append("{:,}".format(
            len(result_df[result_df["gt"] == 1])
        ))  # gt=1
        line.append("{:,}".format(
            len(result_df[result_df["gt"] == 2])
        ))  # gt=2

        line.append("{:.1f}".format(myround(
            len(result_df[result_df["gt"] != 2]) / len(result_df) * 100, 1
        )).rstrip("0").rstrip(".") + "%"),  # IMR

        lcs_df = result_df[result_df["gt"] != 2][["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
        lcs_df = lcs_df.unstack(level=-1).reset_index()
        lcs_r = lcs_df.loc[lcs_df["tmp"] == 0, 0][0]
        # line.append(len(lcs_r))  # LCM
        line.append("{:.1f}".format(myround(len(lcs_r) / len(lcs) * 100, 1)).rstrip("0").rstrip(".") + "%")  # LCMR

        if model == "msrn":
            train_lcs = train_val_datasets[(data, way, "train")]
            val_lcs = train_val_datasets[(data, way, "val")]
            compare_line = [
                way,
                data.upper(),
                len(train_lcs),  # TrainSet
                len(train_lcs.intersection(lcs)),  # Intersection
                len(lcs.difference(train_lcs)),  # Additional
                len(train_lcs.difference(lcs)),  # Deficiency
                len(val_lcs),  # ValSet
                len(val_lcs.intersection(lcs)),  # Intersection
                len(lcs.difference(val_lcs)),  # Additional
                len(val_lcs.difference(lcs)),  # Deficiency
            ]
            compare_df.loc[len(compare_df)] = compare_line

        rq1_df.loc[len(rq1_df)] = line
    print(rq1_df)
    print(compare_df)
    return rq1_df


def rq2(result_dfs):
    rq2_df1 = pd.DataFrame(
        columns=[
            "k", "Dataset", "MICS",
            "# of E", "# of E_error(FP)", "# of E_error(FN)", "# of G",
            "LCC", "LCC_error", "E_labels", "E_labels_error",
        ]
    )
    rq2_df2 = pd.DataFrame(
        columns=[
            "k", "Dataset", "MICS",
            "ATOM", "ATOM_ratio",
            "ATOM_error", "ATOM_error_ratio",
            "Rand", "Rand_ratio",
            "Rand_error", "Rand_error_ratio",
        ]
    )
    for model, data, way, result_df, random_result_dfs in result_dfs:
        print(model, data, way)

        # df1
        line = [way, data.upper(), model.upper().replace("MLDECODER", "MLD").replace("PHOTO", "-")]

        e_df = result_df[
            (result_df["mt"] > 0) & (result_df["union_match"] == True) & (result_df["inter_match"] == False)]
        e_error_df = e_df[e_df["gt"] == 2]
        g_df = result_df[
            (result_df["mt"] > 0) & ((result_df["union_match"] != True) | (result_df["inter_match"] == True))]
        line.append("{:,}".format(len(e_df)))  # # of E
        line.append("{:,}".format(len(e_error_df)))  # # of E_error(false positive)
        line.append("{:,}".format(len(g_df[g_df["gt"] != 2])))  # # of E_error(false negative)
        line.append("{:,}".format(len(g_df)))  # # of G

        lcc_df = result_df[["title", "mt", "lc", "gt"]].copy()
        lcc_df["con"] = lcc_df.apply(lambda x: x["mt"] + x["lc"], axis=1)
        lcc_df["title"] = lcc_df["title"].apply(lambda x: "|".join(sorted(x)))
        lcc_df1 = lcc_df.groupby("title").prod()
        lcc_df2 = lcc_df[lcc_df["gt"] != 2].groupby("title").prod()
        line.append("{:.1f}".format(myround(
            len(lcc_df1[lcc_df1["con"] == 0]) / len(lcc_df1) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # LCC
        line.append("{:.1f}".format(myround(
            (len(lcc_df1[lcc_df1["con"] == 0]) - len(lcc_df2[lcc_df2["con"] == 0])) / len(lcc_df1) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # LCC_error

        line.append("{:.1f}".format(myround(
            len(e_df["title"].drop_duplicates()) / len(result_df["title"].drop_duplicates()) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # E_labels
        line.append("{:.1f}".format(myround(
            (len(e_df["title"].drop_duplicates()) - len(e_df[e_df["gt"] != 2]["title"].drop_duplicates()))
            / len(result_df["title"].drop_duplicates()) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # E_labels_error

        rq2_df1.loc[len(rq2_df1)] = line

        # df2
        line = [way, data.upper(), model.upper().replace("MLDECODER", "MLD").replace("PHOTO", "-")]

        lcs_df = pd.DataFrame()
        lcs_df["lcs"] = result_df[['pred'] + [x for x in result_df.columns if x.startswith('mr_pred')]].apply(
            lambda x: set.union(*[set(combinations(sorted(xi), way)) for xi in x.tolist()]),
            axis=1
        )
        lcs_df["tmp"] = 0
        lcs_df = lcs_df[["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
        lcs_df = lcs_df.unstack(level=-1).reset_index()
        lcs = lcs_df.loc[lcs_df["tmp"] == 0, 0][0]
        line.append("{:,}".format(len(lcs)))  # ATOM cover labels
        line.append("{:.1f}".format(myround(len(lcs) / comb(labels_num[data], way) * 100, 1)).rstrip("0").rstrip(
            ".") + "%")  # ATOM cover labels ratio

        eg_df = result_df[result_df["mt"] > 0]
        lcs_df = pd.DataFrame()
        lcs_df["lcs"] = eg_df[['pred'] + [x for x in eg_df.columns if x.startswith('mr_pred')]].apply(
            lambda x: set.union(*[set(combinations(sorted(xi), way)) for xi in x.tolist()]),
            axis=1
        )
        lcs_df["tmp"] = 0
        lcs_df = lcs_df[["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
        lcs_df = lcs_df.unstack(level=-1).reset_index()
        lcs = lcs_df.loc[lcs_df["tmp"] == 0, 0][0]
        line.append("{:,}".format(len(lcs)))  # ATOM labels
        line.append("{:.1f}".format(myround(len(lcs) / comb(labels_num[data], way) * 100, 1)).rstrip("0").rstrip(
            ".") + "%")  # ATOM labels ratio

        # random
        random_eg_dfs = [rdf[rdf["mt"] > 0] for rdf in random_result_dfs]
        random_cover_lcs = []
        random_eg_lcs = []
        for random_result_df, random_eg_df in zip(random_result_dfs, random_eg_dfs):
            random_cover_lcs_df = pd.DataFrame()
            random_cover_lcs_df["lcs"] = random_result_df[
                ['pred'] + [x for x in random_result_df.columns if x.startswith('mr_pred')]].apply(
                lambda x: set.union(*[set(combinations(sorted(xi), way)) for xi in x.tolist()]),
                axis=1
            )
            random_cover_lcs_df["tmp"] = 0
            random_cover_lcs_df = random_cover_lcs_df[["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
            random_cover_lcs_df = random_cover_lcs_df.unstack(level=-1).reset_index()
            lcs = random_cover_lcs_df.loc[random_cover_lcs_df["tmp"] == 0, 0][0]
            random_cover_lcs.append(len(lcs))

            random_eg_lcs_df = pd.DataFrame()
            random_eg_lcs_df["lcs"] = random_eg_df[
                ['pred'] + [x for x in random_eg_df.columns if x.startswith('mr_pred')]].apply(
                lambda x: set.union(*[set(combinations(sorted(xi), way)) for xi in x.tolist()]),
                axis=1
            )
            random_eg_lcs_df["tmp"] = 0
            random_eg_lcs_df = random_eg_lcs_df[["lcs", "tmp"]].groupby(by=["tmp"]).agg(lambda x: set.union(*x))
            random_eg_lcs_df = random_eg_lcs_df.unstack(level=-1).reset_index()
            lcs = random_eg_lcs_df.loc[random_eg_lcs_df["tmp"] == 0, 0][0]
            random_eg_lcs.append(len(lcs))
        line.append("{:,.1f}".format(myround(
            sum(random_cover_lcs) / 5, 1
        )).rstrip("0").rstrip("."))  # random cover labels
        line.append("{:.1f}".format(myround(
            sum(random_cover_lcs) / 5 / comb(labels_num[data], way) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # random cover labels ratio
        line.append("{:,.1f}".format(myround(
            sum(random_eg_lcs) / 5, 1
        )).rstrip("0").rstrip("."))  # random labels
        line.append("{:.1f}".format(myround(
            sum(random_eg_lcs) / 5 / comb(labels_num[data], way) * 100, 1
        )).rstrip("0").rstrip(".") + "%")  # random labels ratio

        rq2_df2.loc[len(rq2_df2)] = line

    rq2_df1["# of E(FP)"] = rq2_df1[["# of E", "# of E_error(FP)"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df1["LCC(error)"] = rq2_df1[["LCC", "LCC_error"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df1["E_labels(error)"] = rq2_df1[["E_labels", "E_labels_error"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df1 = rq2_df1[[
        "k", "Dataset", "MICS",
        "# of E(FP)", "# of G", "LCC(error)", "E_labels(error)",
    ]]

    rq2_df2_tmp = rq2_df2.copy()
    rq2_df2_tmp = rq2_df2_tmp[[
        "k", "Dataset", "MICS",
        "ATOM_ratio", "Rand_ratio", "ATOM_error_ratio", "Rand_error_ratio",
    ]]

    rq2_df2["ATOM"] = rq2_df2[["ATOM", "ATOM_ratio"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df2["Rand"] = rq2_df2[["Rand", "Rand_ratio"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df2["ATOM_error"] = rq2_df2[["ATOM_error", "ATOM_error_ratio"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df2["Rand_error"] = rq2_df2[["Rand_error", "Rand_error_ratio"]].apply(
        lambda x: f"{x[0]} ({x[1]})", axis=1
    )
    rq2_df2.drop([x for x in rq2_df2.columns if x.endswith('ratio')], axis=1, inplace=True)

    print(rq2_df1)
    print(rq2_df2)
    return rq2_df1, rq2_df2_tmp


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 1000)
    pd.set_option('display.width', 1000)
    args = sys.argv
    rqs = []
    for rq in range(1, 4):
        if "rq" + str(rq) in args:
            rqs.append(rq)
    if not rqs:
        rqs = ["rq1", "rq2", "rq3"]
    result_dfs = []
    for w in [1, 2]:
        for model, data in models:
            result_df = joblib.load(os.path.join(
                root,
                "info",
                "all_info",
                f"google_{data}{w}_{model}.pkl"
            ))
            if "rq2" in rqs:
                random_dfs = joblib.load(os.path.join(
                    root,
                    "info",
                    "all_info",
                    f"imagenet_{data}{w}_{model}.pkl"
                ))
            else:
                random_dfs = None
            result_dfs.append((model, data, w, result_df, random_dfs))

    if "rq1" in rqs:
        rq1_df = rq1(result_dfs)
        rq1_df.to_excel(os.path.join(root, "rq1.xlsx"), index=False)
        merge_df_cells(os.path.join(root, "rq1.xlsx"), ["A2:A4", "A5:A7"])

    if "rq2" in rqs:
        rq21_df, rq22_df = rq2(result_dfs)
        rq21_df.to_excel(os.path.join(root, "rq2_1.xlsx"), index=False)
        rq22_df.to_excel(os.path.join(root, "rq2_2.xlsx"), index=False)
        merge_df_cells(os.path.join(root, "rq2_1.xlsx"), ["A2:A12", "A13:A23", "B2:B6", "B7:B11", "B13:B17", "B18:B22"])
        merge_df_cells(os.path.join(root, "rq2_2.xlsx"), ["A2:A12", "A13:A23", "B2:B6", "B7:B11", "B13:B17", "B18:B22"])

    if "rq3" in rqs:
        rq3(result_dfs)
