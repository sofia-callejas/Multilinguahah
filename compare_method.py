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
        "label_dir", type=str, help="Path to the label directory (.pickle)"
    )
    parser.add_argument("language", type=str, help="language")
    parser.add_argument(
        "output_dir", type=str, help="Path to the output score directory"
    )
    parser.add_argument(
        "iou_threshold", type=float, help="iou_threshold"
    )
    args = parser.parse_args()

    return args

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def df_en_tuples(df):
    return [tuple(round(val, 2) for val in ligne) for ligne in df[['t0', 't1']].values]

def calculate_iou(interval1, interval2):
    """
    Calculate Intersection over Union (IoU) for two intervals.
    Each interval is a tuple (start, end).
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    area1 = end1 - start1
    area2 = end2 - start2
    union = area1 + area2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union

def evaluate_predictions_with_iou(predictions, labels, iou_threshold=0.5):
    """
    Evaluate predictions against labels using IoU.
    Returns counts of True Positives (TP), False Positives (FP), and False Negatives (FN).
    """
    TP = 0
    FP = 0
    FN = 0

    # Track which labels have been matched
    matched_labels = set()
    not_matched_labels = set()

    # Check each prediction
    for pred in predictions:
        best_iou = 0.0
        best_label_idx = -1

        for i, label in enumerate(labels):
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_labels.add(best_label_idx)  # Mark this label as matched
        else:
            FP += 1
            # not_matched_labels.add(label)

    # Check for FN
    for i, label in enumerate(labels):
        best_iou = 0.0
        for pred in predictions:
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou < iou_threshold:
            not_matched_labels.add(label)

    # Count False Negatives (labels not matched by any prediction)
    # FN = len(labels) - len(matched_labels)
    FN = len(not_matched_labels)

    return TP, FP, FN, not_matched_labels

def calculate_metrics(TP, FP, FN):
    """
    Calculate precision, recall, and accuracy given TP, FP, and FN.
    """
    if 2*TP + FP + FN == 0:
        f1 = 0.0
    else:
        f1 = 2*TP / (2*TP + FP + FN)
    
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    total = TP + FP + FN
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = TP / total

    return precision, recall, accuracy, f1

def overlaps_with_base(t0, t1, base_intervals):
    for bt0, bt1 in base_intervals:
        if t0 < bt1 and t1 > bt0:  # cualquier solapamiento
            return True
    return False


if __name__ == "__main__":
    args = parse_arguments()
    pred_dir = args.pred_dir
    label_dir = args.label_dir
    model_dir = args.model_dir
    language = args.language
    output_dir = args.output_dir
    iou_threshold = args.iou_threshold

    os.makedirs(output_dir, exist_ok=True)

    pred_filenames = sorted(os.listdir(pred_dir))
    label_filenames = sorted(os.listdir(label_dir))
    model_filenames = sorted(os.listdir(model_dir))

    pred_dict = {osp.splitext(f)[0]: f for f in pred_filenames}
    label_dict = {osp.splitext(f)[0]: f for f in label_filenames}
    model_dict = {osp.splitext(f)[0]: f for f in model_filenames}

    common_keys = set(pred_dict.keys()) & set(label_dict.keys())

    temporal_scores = {}
    detect_scores = defaultdict(list)

    temporal_scores, detect_scores = {}, defaultdict(list)

    TP_all_paper = 0
    FP_all_paper = 0
    FN_all_paper = 0

    TP_all_baseline = 0
    FP_all_baseline = 0
    FN_all_baseline = 0
    
    TP_all = 0
    FP_all = 0
    FN_all = 0

    TP_all_union_paper = 0
    FP_all_union_paper = 0
    FN_all_union_paper = 0

    TP_all_union_pred = 0
    FP_all_union_pred = 0
    FN_all_union_pred = 0
    
    for key in sorted(common_keys):
        pred_name = pred_dict[key]
        label_name = label_dict[key]
        model_name = model_dict[key]

        pred_timecodes = load_preds(osp.join(pred_dir, pred_name))
        pred_timecodes = convertir_en_tuples(pred_timecodes)

        true_timecodes = pd.read_csv(osp.join(label_dir, label_name),delimiter=";")

        model_timecodes = pd.read_csv(osp.join(model_dir, model_name))
        model_baseline = model_timecodes.loc[model_timecodes["source"] == "Initial", ["t0", "t1"]]
        model_paper = model_timecodes.loc[:, ["t0", "t1"]]

        df_pred = pd.DataFrame(pred_timecodes, columns=["t0", "t1"])
        pred_timecodes = df_pred.copy()

        base_pred_intervals = list(zip(df_pred["t0"].tolist(), df_pred["t1"].tolist()))
        df_pred_filtered_base_model = model_paper[~model_paper.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_pred_intervals), axis=1)]

        base_paper_intervals = list(zip(model_paper["t0"].tolist(), model_paper["t1"].tolist()))
        df_model_filtered_base_paper = df_pred[~df_pred.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_paper_intervals), axis=1)]

        model_union_base_paper = pd.concat([model_paper, df_model_filtered_base_paper], ignore_index=True)
        model_union_base_pred = pd.concat([df_pred, df_pred_filtered_base_model], ignore_index=True)

        model_baseline = df_en_tuples(model_baseline)
        model_paper = df_en_tuples(model_paper)
        true_timecodes = df_en_tuples(true_timecodes)
        pred_timecodes = df_en_tuples(pred_timecodes)
        model_union_base_paper = df_en_tuples(model_union_base_paper)
        model_union_base_pred = df_en_tuples(model_union_base_pred)

        TP_union_paper, FP_union_paper, FN_union_paper, not_matched_union = evaluate_predictions_with_iou(model_union_base_paper, true_timecodes, iou_threshold)
        TP_all_union_paper += TP_union_paper
        FP_all_union_paper += FP_union_paper
        FN_all_union_paper += FN_union_paper
        
        TP_union_pred, FP_union_pred, FN_union_pred, not_matched_union = evaluate_predictions_with_iou(model_union_base_pred, true_timecodes, iou_threshold)
        TP_all_union_pred += TP_union_pred
        FP_all_union_pred += FP_union_pred
        FN_all_union_pred += FN_union_pred

        TP_paper, FP_paper, FN_paper, not_matched_paper = evaluate_predictions_with_iou(model_paper, true_timecodes, iou_threshold)
        TP_all_paper += TP_paper
        FP_all_paper += FP_paper
        FN_all_paper += FN_paper

        TP_baseline, FP_baseline, FN_baseline, not_matched_baseline = evaluate_predictions_with_iou(model_baseline, true_timecodes, iou_threshold)
        TP_all_baseline += TP_baseline
        FP_all_baseline += FP_baseline
        FN_all_baseline += FN_baseline

        TP, FP, FN, not_matched = evaluate_predictions_with_iou(pred_timecodes, true_timecodes, iou_threshold)
        TP_all += TP
        FP_all += FP
        FN_all += FN

    precision_paper, recall_paper, accuracy_paper, f1_paper= calculate_metrics(TP_all_paper, FP_all_paper, FN_all_paper)   
    precision_baseline, recall_baseline, accuracy_baseline, f1_baseline= calculate_metrics(TP_all_baseline, FP_all_baseline, FN_all_baseline)
    precision_union_paper, recall_union_paper, accuracy_union_paper, f1_union_paper= calculate_metrics(TP_all_union_paper, FP_all_union_paper, FN_all_union_paper)
    precision_union_pred, recall_union_pred, accuracy_union_pred, f1_union_pred= calculate_metrics(TP_all_union_pred, FP_all_union_pred, FN_all_union_pred)

    precision, recall, accuracy, f1= calculate_metrics(TP_all, FP_all, FN_all)   
   
    
import pandas as pd

results = {
    "paper": {
        "precision": precision_paper,
        "recall": recall_paper,
        "accuracy": accuracy_paper,
        "F1": f1_paper,
        "TP": TP_all_paper,
        "FP": FP_all_paper,
        "FN": FN_all_paper
    },
    "baseline": {
        "precision": precision_baseline,
        "recall": recall_baseline,
        "accuracy": accuracy_baseline,
        "F1": f1_baseline,
        "TP": TP_all_baseline,
        "FP": FP_all_baseline,
        "FN": FN_all_baseline
    },
    "isolation": {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "F1": f1,
        "TP": TP_all,
        "FP": FP_all,
        "FN": FN_all
    },
    "union_paper": {
        "precision": precision_union_paper,
        "recall": recall_union_paper,
        "accuracy": accuracy_union_paper,
        "F1": f1_union_paper,
        "TP": TP_all_union_paper,
        "FP": FP_all_union_paper,
        "FN": FN_all_union_paper
    },
    "union_pred": {
        "precision": precision_union_pred,
        "recall": recall_union_pred,
        "accuracy": accuracy_union_pred,
        "F1": f1_union_pred,
        "TP": TP_all_union_pred,
        "FP": FP_all_union_pred,
        "FN": FN_all_union_pred
    }
}

df_results = pd.DataFrame(results).T

output_file = os.path.join(output_dir, "metrics.csv")

df_new = df_results.copy()
df_new["language"] = language
df_new["oui_threshold"] = iou_threshold
df_new["method"] = df_new.index  # keep paper / baseline / ours explicitly
df_new = df_new.reset_index(drop=True)

if os.path.exists(output_file):
    df_db = pd.read_csv(output_file)
    
    df_db = pd.concat([df_db, df_new], ignore_index=True)
    df_db = df_db.drop_duplicates(
        subset=["method", "language", "oui_threshold"],
        keep="last"
    )
else:
    df_db = df_new

print(df_db)
df_db.to_csv(output_file, index=False)
    

