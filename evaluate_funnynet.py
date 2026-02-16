import argparse
from collections import defaultdict
import os
import os.path as osp

import numpy as np
import pandas as pd
import json
import re

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
    parser.add_argument(
        "output_dir", type=str, help="Path to the output score directory"
    )
    parser.add_argument(
        "iou_threshold", type=float, help="iou_threshold"
    )
    parser.add_argument(
        "model_type", type=str, help="isolation"
    )
    args = parser.parse_args()

    return args

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def df_en_tuples(df):
    return [tuple(round(val, 2) for val in ligne) for ligne in df[['t0', 't1']].values]

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan


def json_to_segments_df(json_data):
    """
    Converts:
    {'0': {'start_sec': x, 'end_sec': y}, ...}
    → DataFrame with columns ['t0', 't1']
    """
    segments = [
        (v["start_sec"], v["end_sec"])
        for v in json_data.values()
    ]
    return pd.DataFrame(segments, columns=["t0", "t1"])

def match_predictions_to_gt_by_start(preds, gts):
    """
    preds: list of (start, end)
    gts:   list of (start, end)

    Greedy 1-to-1 matching using closest start time,
    filtered by IoU threshold.
    Returns list of (gt_interval, pred_interval)
    """

    matches = []
    used_preds = set()

    for gt in gts:
        gt_start, gt_end = gt

        best_dist = float("inf")
        best_pred = None
        best_idx = -1

        for i, pred in enumerate(preds):
            if i in used_preds:
                continue

            pred_start, pred_end = pred
            dist = abs(pred_start - gt_start)

            if dist < best_dist:
                best_dist = dist
                best_pred = pred
                best_idx = i

        # Only accept if IoU passes threshold
        if best_pred is not None:
            if calculate_iou(gt, best_pred) >= 0.7:
                matches.append((gt, best_pred))
                used_preds.add(best_idx)

    return matches


def temporal_mae(matches):
    """
    Compute MAE for start and end times from matched segments.
    """
    if len(matches) == 0:
        return np.nan, np.nan

    start_errors = []
    end_errors = []

    for (gt_s, gt_e), (pr_s, pr_e) in matches:
        start_errors.append(abs(gt_s - pr_s))
        end_errors.append(abs(gt_e - pr_e))

    return np.mean(start_errors), np.mean(end_errors)


def bootstrap_ci(metric_fn, gt_segments, pred_segments, n_bootstrap=1000, alpha=0.05):
    values = []

    n = len(gt_segments)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        gt_sample = [gt_segments[i] for i in idx]
        pr_sample = [pred_segments[i] for i in idx]

        values.append(metric_fn(gt_sample, pr_sample))

    values = np.array(values)
    mean = np.mean(values)
    lower = np.percentile(values, 100 * (alpha / 2))
    upper = np.percentile(values, 100 * (1 - alpha / 2))

    return mean, (upper - mean)

def calculate_iou(a, b):
    s1, e1 = a
    s2, e2 = b

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    if union <= 0:
        return 0.0

    return inter / union

import math
def adaptive_iou_threshold(label, tau_min=0.3, tau_max=0.7, max_duration=5.0):
    """
    Calcule le seuil IoU adaptatif en fonction de la durée du GT
    """
    start1, end1 = label
    gt_duration = end1 - start1
    d = max(gt_duration, 1e-6)  
    tau = tau_min + (tau_max - tau_min) * math.log(1 + d) / math.log(1 + max_duration)
    return tau

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
            if iou_threshold == 0:
                iou_threshold = adaptive_iou_threshold(label,tau_min=0.1,tau_max=0.8)
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
        if t0 < bt1 and t1 > bt0: 
            return True
    return False

def remove_time_suffix(key):
    return re.sub(r'_\d+_\d+$', '', key)


if __name__ == "__main__":
    args = parse_arguments()
    pred_dir = args.pred_dir
    label_dir = args.label_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    model_type = args.model_type
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
    mae_start_all_paper = []
    mae_end_all_paper = []

    TP_all_baseline = 0
    FP_all_baseline = 0
    FN_all_baseline = 0
    mae_start_all_baseline = []
    mae_end_all_baseline = []
    
    TP_all = 0
    FP_all = 0
    FN_all = 0
    mae_start_all = []
    mae_end_all = []

    TP_all_union_paper = 0
    FP_all_union_paper = 0
    FN_all_union_paper = 0
    mae_start_all_union_paper = []
    mae_end_all_union_paper = []

    TP_all_union_pred_base_paper = 0
    FP_all_union_pred_base_paper = 0
    FN_all_union_pred_base_paper = 0
    mae_start_all_union_paper_base_paper = []
    mae_end_all_union_paper_base_paper = []

    TP_all_union_pred_base_baseline = 0
    FP_all_union_pred_base_baseline = 0
    FN_all_union_pred_base_baseline = 0
    mae_start_all_union_pred_base_baseline = []
    mae_end_all_union_pred_base_baseline  = []

    TP_all_union_baseline = 0
    FP_all_union_baseline = 0
    FN_all_union_baseline = 0
    mae_start_all_union_baseline = []
    mae_end_all_union_baseline  = []

    print(common_keys)
    
    for key in sorted(common_keys):

        pred_name = pred_dict[key]
        label_name = label_dict[key]
        model_name = model_dict[key]

        pred_timecodes = load_preds(osp.join(pred_dir, pred_name))
        pred_timecodes = convertir_en_tuples(pred_timecodes)

        
        with open(osp.join(label_dir, label_name), "r", encoding="utf-8") as f:
            import csv
            sample = f.read(2048)  # small sample
            delimiter = csv.Sniffer().sniff(sample).delimiter

        true_timecodes = pd.read_csv(osp.join(label_dir, label_name),delimiter=delimiter)
        with open(osp.join(model_dir, model_name), "r") as f:
            model_json = json.load(f)

        model_timecodes = json_to_segments_df(model_json)

        df_pred = pd.DataFrame(pred_timecodes, columns=["t0", "t1"])

        pred_timecodes = df_pred.copy()

        base_pred_intervals_paper = list(zip(df_pred["t0"].tolist(), df_pred["t1"].tolist()))
        df_pred_filtered_base_model = model_timecodes[~model_timecodes.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_pred_intervals_paper), axis=1)]

        base_pred_intervals_baseline = list(zip(df_pred["t0"].tolist(), df_pred["t1"].tolist()))
        df_pred_filtered_base_baseline = model_timecodes[~model_timecodes.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_pred_intervals_baseline), axis=1)]

        base_paper_intervals = list(zip(model_timecodes["t0"].tolist(), model_timecodes["t1"].tolist()))
        df_model_filtered_base_paper = df_pred[~df_pred.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_paper_intervals), axis=1)]

        base_baseline_intervals = list(zip(model_timecodes["t0"].tolist(), model_timecodes["t1"].tolist()))
        df_model_filtered_base_baseline = df_pred[~df_pred.apply(lambda row: overlaps_with_base(row.t0, row.t1, base_baseline_intervals), axis=1)] 


        model_union_base_paper = pd.concat([model_timecodes, df_model_filtered_base_paper], ignore_index=True)
        model_union_base_pred_paper = pd.concat([df_pred, df_pred_filtered_base_model], ignore_index=True)
        model_union_base_pred_baseline = pd.concat([df_pred, df_pred_filtered_base_baseline], ignore_index=True)
        model_union_base_baseline = pd.concat([model_timecodes, df_model_filtered_base_baseline], ignore_index=True)

        model_baseline = df_en_tuples(model_timecodes)
        model_paper = df_en_tuples(model_timecodes)
        true_timecodes = df_en_tuples(true_timecodes)
        pred_timecodes = df_en_tuples(pred_timecodes)
        model_union_base_paper = df_en_tuples(model_union_base_paper)
        model_union_base_pred_paper = df_en_tuples(model_union_base_pred_paper)
        model_union_base_pred_baseline = df_en_tuples(model_union_base_pred_baseline)
        model_union_base_baseline = df_en_tuples(model_union_base_baseline)

        TP_union_paper, FP_union_paper, FN_union_paper, not_matched_union = evaluate_predictions_with_iou(model_union_base_paper, true_timecodes, iou_threshold)
        TP_all_union_paper += TP_union_paper
        FP_all_union_paper += FP_union_paper
        FN_all_union_paper += FN_union_paper

        matches = match_predictions_to_gt_by_start(model_union_base_paper, true_timecodes)
        mae_s_union_paper, mae_e_union_paper = temporal_mae(matches)
        if not np.isnan(mae_s_union_paper):
            mae_start_all_union_paper.append(mae_s_union_paper)
            mae_end_all_union_paper.append(mae_e_union_paper)

        TP_union_baseline, FP_union_baseline, FN_union_baseline, not_matched_union = evaluate_predictions_with_iou(model_union_base_baseline, true_timecodes, iou_threshold)
        TP_all_union_baseline += TP_union_baseline
        FP_all_union_baseline += FP_union_baseline
        FN_all_union_baseline += FN_union_baseline

        matches = match_predictions_to_gt_by_start(model_union_base_baseline, true_timecodes)
        mae_s_union_baseline, mae_e_union_baseline = temporal_mae(matches)
        if not np.isnan(mae_s_union_baseline):
            mae_start_all_union_baseline.append(mae_s_union_baseline)
            mae_end_all_union_baseline.append(mae_e_union_baseline)
        
        TP_union_pred_base_paper, FP_union_pred_base_paper, FN_union_pred_base_paper, not_matched_union = evaluate_predictions_with_iou(model_union_base_pred_paper, true_timecodes, iou_threshold)
        TP_all_union_pred_base_paper += TP_union_pred_base_paper
        FP_all_union_pred_base_paper += FP_union_pred_base_paper
        FN_all_union_pred_base_paper += FN_union_pred_base_paper

        matches = match_predictions_to_gt_by_start(model_union_base_pred_paper, true_timecodes)
        mae_s_union_paper_base_paper, mae_e_union_paper_base_paper = temporal_mae(matches)
        if not np.isnan(mae_s_union_paper_base_paper):
            mae_start_all_union_paper_base_paper.append(mae_s_union_paper_base_paper)
            mae_end_all_union_paper_base_paper.append(mae_e_union_paper_base_paper)

        TP_union_pred_base_baseline, FP_union_pred_base_baseline, FN_union_pred_base_baseline, not_matched_union = evaluate_predictions_with_iou(model_union_base_pred_baseline, true_timecodes, iou_threshold)
        TP_all_union_pred_base_baseline += TP_union_pred_base_baseline
        FP_all_union_pred_base_baseline += FP_union_pred_base_baseline
        FN_all_union_pred_base_baseline += FN_union_pred_base_baseline

        matches = match_predictions_to_gt_by_start(model_union_base_pred_baseline, true_timecodes)
        mae_s_union_base_pred_baseline, mae_e_union_base_pred_baseline = temporal_mae(matches)
        if not np.isnan(mae_s_union_base_pred_baseline):
            mae_start_all_union_pred_base_baseline.append(mae_s_union_base_pred_baseline)
            mae_end_all_union_pred_base_baseline.append(mae_e_union_base_pred_baseline)

        TP_paper, FP_paper, FN_paper, not_matched_paper = evaluate_predictions_with_iou(model_paper, true_timecodes, iou_threshold)
        TP_all_paper += TP_paper
        FP_all_paper += FP_paper
        FN_all_paper += FN_paper

        matches = match_predictions_to_gt_by_start(model_paper, true_timecodes)
        mae_s_paper, mae_e_paper = temporal_mae(matches)
        if not np.isnan(mae_s_paper):
            mae_start_all_paper.append(mae_s_paper)
            mae_end_all_paper.append(mae_e_paper)

        TP_baseline, FP_baseline, FN_baseline, not_matched_baseline = evaluate_predictions_with_iou(model_baseline, true_timecodes, iou_threshold)
        TP_all_baseline += TP_baseline
        FP_all_baseline += FP_baseline
        FN_all_baseline += FN_baseline

        matches = match_predictions_to_gt_by_start(model_baseline, true_timecodes)
        mae_s_baseline, mae_e_baseline = temporal_mae(matches)
        if not np.isnan(mae_s_baseline):
            mae_start_all_baseline.append(mae_s_baseline)
            mae_end_all_baseline.append(mae_e_baseline)

        TP, FP, FN, not_matched = evaluate_predictions_with_iou(pred_timecodes, true_timecodes, iou_threshold)
        TP_all += TP
        FP_all += FP
        FN_all += FN

        matches = match_predictions_to_gt_by_start(pred_timecodes, true_timecodes)
        mae_s, mae_e = temporal_mae(matches)
        if not np.isnan(mae_s):
            mae_start_all.append(mae_s)
            mae_end_all.append(mae_e)

    precision_paper, recall_paper, accuracy_paper, f1_paper= calculate_metrics(TP_all_paper, FP_all_paper, FN_all_paper)   
    precision_baseline, recall_baseline, accuracy_baseline, f1_baseline= calculate_metrics(TP_all_baseline, FP_all_baseline, FN_all_baseline)
    precision_union_paper, recall_union_paper, accuracy_union_paper, f1_union_paper= calculate_metrics(TP_all_union_paper, FP_all_union_paper, FN_all_union_paper)

    precision_union_pred_base_paper, recall_union_pred_base_paper, accuracy_union_pred_base_paper, f1_union_pred_base_paper= calculate_metrics(TP_all_union_pred_base_paper, FP_all_union_pred_base_paper, FN_all_union_pred_base_paper)
    precision_union_pred_base_baseline, recall_union_pred_base_baseline, accuracy_union_pred_base_baseline, f1_union_pred_base_baseline= calculate_metrics(TP_all_union_pred_base_baseline, FP_all_union_pred_base_baseline, FN_all_union_pred_base_baseline)

    precision_union_baseline, recall_union_baseline, accuracy_union_baseline, f1_union_baseline= calculate_metrics(TP_all_union_baseline, FP_all_union_baseline, FN_all_union_baseline)


    precision, recall, accuracy, f1= calculate_metrics(TP_all, FP_all, FN_all)   

    mae_start = safe_mean(mae_start_all)
    mae_end = safe_mean(mae_end_all)

    mae_start_paper = safe_mean(mae_start_all_paper)
    mae_end_paper = safe_mean(mae_end_all_paper)

    mae_start_union_paper = safe_mean(mae_start_all_union_paper)
    mae_end_union_paper = safe_mean(mae_end_all_union_paper)

    mae_start_union_baseline = safe_mean(mae_start_all_union_baseline)
    mae_end_union_baseline = safe_mean(mae_end_all_union_baseline)

    mae_start_union_paper_base_paper = safe_mean(mae_start_all_union_paper_base_paper)
    mae_end_union_paper_base_paper = safe_mean(mae_end_all_union_paper_base_paper)

    mae_start_union_pred_base_baseline = safe_mean(mae_start_all_union_pred_base_baseline)
    mae_end_union_pred_base_baseline = safe_mean(mae_end_all_union_pred_base_baseline)

    mae_start_baseline = safe_mean(mae_start_all_baseline)
    mae_end_baseline = safe_mean(mae_end_all_baseline)
   
    
import pandas as pd

results = {
    "paper": {
        "precision": precision_paper,
        "recall": recall_paper,
        "accuracy": accuracy_paper,
        "F1": f1_paper,
        "TP": TP_all_paper,
        "FP": FP_all_paper,
        "FN": FN_all_paper,
        "mae_start" : mae_start_paper,
        "mae end": mae_end_paper 
    },
    "baseline": {
        "precision": precision_baseline,
        "recall": recall_baseline,
        "accuracy": accuracy_baseline,
        "F1": f1_baseline,
        "TP": TP_all_baseline,
        "FP": FP_all_baseline,
        "FN": FN_all_baseline,
        "mae_start" : mae_start_baseline,
        "mae end": mae_end_baseline 
    },
    str(model_type): {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "F1": f1,
        "TP": TP_all,
        "FP": FP_all,
        "FN": FN_all,
        "mae_start" : mae_start,
        "mae end": mae_end 
    },
    "union_paper": {
        "precision": precision_union_paper,
        "recall": recall_union_paper,
        "accuracy": accuracy_union_paper,
        "F1": f1_union_paper,
        "TP": TP_all_union_paper,
        "FP": FP_all_union_paper,
        "FN": FN_all_union_paper,
        "mae_start" : mae_start_union_paper,
        "mae end": mae_end_union_paper 
    },
    "union_pred_base_paper": {
        "precision": precision_union_pred_base_paper,
        "recall": recall_union_pred_base_paper,
        "accuracy": accuracy_union_pred_base_paper,
        "F1": f1_union_pred_base_paper,
        "TP": TP_all_union_pred_base_paper,
        "FP": FP_all_union_pred_base_paper,
        "FN": FN_all_union_pred_base_paper,
        "mae_start" : mae_start_union_paper_base_paper,
        "mae end": mae_end_union_paper_base_paper 
    },
        "union_pred_base_baseline": {
        "precision": precision_union_pred_base_baseline,
        "recall": recall_union_pred_base_baseline,
        "accuracy": accuracy_union_pred_base_baseline,
        "F1": f1_union_pred_base_baseline,
        "TP": TP_all_union_pred_base_baseline,
        "FP": FP_all_union_pred_base_baseline,
        "FN": FN_all_union_pred_base_baseline,
        "mae_start" : mae_start_union_pred_base_baseline,
        "mae end": mae_end_union_pred_base_baseline 
    },
        "union_baseline": {
        "precision": precision_union_baseline,
        "recall": recall_union_baseline,
        "accuracy": accuracy_union_baseline,
        "F1": f1_union_baseline,
        "TP": TP_all_union_baseline,
        "FP": FP_all_union_baseline,
        "FN": FN_all_union_baseline,
        "mae_start" : mae_start_union_baseline,
        "mae end": mae_end_union_baseline 
    }
}

df_results = pd.DataFrame(results).T
print(df_results)
exit()

output_file = os.path.join(output_dir, "metrics.csv")

df_new = df_results.copy()
df_new["oui_threshold"] = iou_threshold
df_new["method"] = df_new.index 
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

#print(df_db[df_db["language"]=="en_uk"])

print(df_db)
df_db.to_csv(output_file, index=False)
    

