import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import difflib
from laughter_detection.core.utils import load_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("model_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--base_paper", action="store_true")
    return parser.parse_args()

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def overlaps_with_base(t0, t1, base_intervals):
    for bt0, bt1 in base_intervals:
        if t0 < bt1 and t1 > bt0:
            return True
    return False

def best_match(name: str, candidates: set) -> str:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.5)
    return matches[0] if matches else name

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_filenames = sorted(os.listdir(args.pred_dir))
    model_filenames = sorted(os.listdir(args.model_dir))

    pred_dict = {osp.splitext(f)[0]: f for f in pred_filenames}
    model_dict = {osp.splitext(f)[0]: f for f in model_filenames}

    for clean_name in sorted(pred_dict.keys()):
        original_name = best_match(clean_name, model_dict.keys())
        if original_name is None:
            original_name = clean_name  

        model_name = model_dict.get(clean_name)
        pred_name = pred_dict.get(original_name)

        if model_name is None or pred_name is None:
            continue

        pred_timecodes = load_preds(osp.join(args.pred_dir, pred_name))
        pred_timecodes = convertir_en_tuples(pred_timecodes)
        df_pred = pd.DataFrame(pred_timecodes, columns=["t0", "t1"])

        model_timecodes = pd.read_csv(osp.join(args.model_dir, model_name))
        model_paper = model_timecodes.loc[model_timecodes["label"] == "risa", ["t0", "t1"]]

        if args.base_paper:
            base_paper_intervals = list(zip(model_paper["t0"].tolist(), model_paper["t1"].tolist()))
            df_filtered = df_pred[~df_pred.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_paper_intervals), axis=1)].copy()
            model_paper_c = model_paper.copy()
            model_paper_c["source"] = "Initial"
            df_filtered["source"] = "Added"
            df_union = pd.concat([model_paper_c, df_filtered], ignore_index=True)
        else:
            base_pred_intervals = list(zip(df_pred["t0"].tolist(), df_pred["t1"].tolist()))
            df_filtered = model_paper[~model_paper.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_pred_intervals), axis=1)].copy()
            df_pred_c = df_pred.copy()
            df_pred_c["source"] = "Initial"
            df_filtered["source"] = "Added"
            df_union = pd.concat([df_pred_c, df_filtered], ignore_index=True)

        df_union["label"] = "risa"
        clean_name_out = pred_name.replace(".pk", "")
        output_file = os.path.join(args.output_dir, f"{clean_name_out}.csv")
        df_union.to_csv(output_file, index=False)