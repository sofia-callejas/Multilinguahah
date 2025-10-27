import argparse
from collections import defaultdict
import os
import os.path as osp

import numpy as np
import pandas as pd

from laughter_detection.core.utils import load_labels, load_preds
from laughter_detection.core.evaluation import (
    get_detection_scores,
    get_temporal_scores,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_dir", type=str, help="Path to the prediction directory (.csv)"
    )
    parser.add_argument(
        "model_dir", type=str, help="Path to the other models directory (.csv)"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to the output .csv directory"
    )
    args = parser.parse_args()

    return args

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def df_en_tuples(df):
    return [tuple(round(val, 2) for val in ligne) for ligne in df[['t0', 't1']].values]

def overlaps_with_base(t0, t1, base_intervals):
    for bt0, bt1 in base_intervals:
        if t0 < bt1 and t1 > bt0:  # cualquier solapamiento
            return True
    return False


if __name__ == "__main__":
    args = parse_arguments()
    pred_dir = args.pred_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    pred_filenames = sorted(os.listdir(pred_dir))
    model_filenames = sorted(os.listdir(model_dir))

    pred_dict = {osp.splitext(f)[0]: f for f in pred_filenames}
    model_dict = {osp.splitext(f)[0]: f for f in model_filenames}




    common_keys = set(pred_dict.keys()) & set(model_dict.keys())



    temporal_scores = {}
    detect_scores = defaultdict(list)

    temporal_scores, detect_scores = {}, defaultdict(list)
    
    for key in sorted(common_keys):
        pred_name = pred_dict[key]
        model_name = model_dict[key]

        pred_timecodes = load_preds(osp.join(pred_dir, pred_name))
        pred_timecodes = convertir_en_tuples(pred_timecodes)

        model_timecodes = pd.read_csv(osp.join(model_dir, model_name))
        model_baseline = model_timecodes.loc[model_timecodes["source"] == "Initial", ["t0", "t1"]]
        model_paper = model_timecodes.loc[model_timecodes["label"] == "risa", ["t0", "t1"]]

        df_pred = pd.DataFrame(pred_timecodes, columns=["t0", "t1"])
        pred_timecodes = df_pred.copy()

        base_paper_intervals = list(zip(model_paper["t0"].tolist(), model_paper["t1"].tolist()))
        df_model_filtered_base_paper = df_pred[~df_pred.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_paper_intervals), axis=1)]

        model_paper["source"] = "Initial"
        df_model_filtered_base_paper["source"] = "Added"
        
        model_union_base_pred_paper = pd.concat([model_paper, df_model_filtered_base_paper], ignore_index=True)
        model_union_base_pred_paper["label"] = "risa"
        clean_name = pred_name.replace(".pk", "")
        output_file = os.path.join(output_dir, f"{clean_name}.csv")
        model_union_base_pred_paper.to_csv(output_file, index=False)






