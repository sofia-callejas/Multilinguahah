import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import math
import csv
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("label_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("iou_threshold", type=float)
    parser.add_argument("--method", type=str, required=True, choices=["funnynet", "gillick", "omine_paper", "omine_baseline", "multilinguahah"])
    return parser.parse_args()

def load_preds(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def df_en_tuples(df):
    if df.empty:
        return []
    return [tuple(round(val, 2) for val in ligne) for ligne in df[['t0', 't1']].values]

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan

def match_predictions_to_gt_by_start(preds, gts):
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
        if best_pred is not None:
            if calculate_iou(gt, best_pred) >= 0.7:
                matches.append((gt, best_pred))
                used_preds.add(best_idx)
    return matches

def temporal_mae(matches):
    if len(matches) == 0:
        return np.nan, np.nan
    start_errors = []
    end_errors = []
    for (gt_s, gt_e), (pr_s, pr_e) in matches:
        start_errors.append(abs(gt_s - pr_s))
        end_errors.append(abs(gt_e - pr_e))
    return np.mean(start_errors), np.mean(end_errors)

def calculate_iou(a, b):
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    if union <= 0:
        return 0.0
    return inter / union

def adaptive_iou_threshold(label, tau_min=0.3, tau_max=0.7, max_duration=5.0):
    start1, end1 = label
    gt_duration = end1 - start1
    d = max(gt_duration, 1e-6)  
    tau = tau_min + (tau_max - tau_min) * math.log(1 + d) / math.log(1 + max_duration)
    return tau

def evaluate_predictions_with_iou(predictions, labels, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0
    matched_labels = set()
    not_matched_labels = set()
    
    for pred in predictions:
        best_iou = 0.0
        best_label_idx = -1
        for i, label in enumerate(labels):
            current_iou_threshold = adaptive_iou_threshold(label,tau_min=0.1,tau_max=0.8) if iou_threshold == 0 else iou_threshold
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i
        if best_iou >= (adaptive_iou_threshold(labels[best_label_idx],tau_min=0.1,tau_max=0.8) if iou_threshold == 0 else iou_threshold):
            TP += 1
            matched_labels.add(best_label_idx) 
        else:
            FP += 1

    for i, label in enumerate(labels):
        best_iou = 0.0
        current_iou_threshold = adaptive_iou_threshold(label,tau_min=0.1,tau_max=0.8) if iou_threshold == 0 else iou_threshold
        for pred in predictions:
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
        if best_iou < current_iou_threshold:
            not_matched_labels.add(label)

    FN = len(not_matched_labels)
    return TP, FP, FN, not_matched_labels

def calculate_metrics(TP, FP, FN):
    f1 = 0.0 if 2*TP + FP + FN == 0 else 2*TP / (2*TP + FP + FN)
    precision = 0.0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0.0 if TP + FN == 0 else TP / (TP + FN)
    total = TP + FP + FN
    accuracy = 0.0 if total == 0 else TP / total
    return precision, recall, accuracy, f1

def bootstrap_ci_f1(gt_segments_list, pred_segments_list, threshold, n_bootstrap=1000, alpha=0.05):
    n = len(gt_segments_list)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        gt_sample = [gt_segments_list[i] for i in idx]
        pr_sample = [pred_segments_list[i] for i in idx]
        
        TP_total = FP_total = FN_total = 0
        for gt, pr in zip(gt_sample, pr_sample):
            TP, FP, FN, _ = evaluate_predictions_with_iou(pr, gt, threshold)
            TP_total += TP
            FP_total += FP
            FN_total += FN
        
        _, _, _, f1 = calculate_metrics(TP_total, FP_total, FN_total)
        values.append(f1)
        
    values = np.array(values)
    mean = np.mean(values)
    lower = np.percentile(values, 100 * (alpha / 2))
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return mean, lower, upper

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_filenames = sorted(os.listdir(args.pred_dir))
    label_filenames = sorted(os.listdir(args.label_dir))

    pred_dict = {osp.splitext(f)[0]: f for f in pred_filenames}
    label_dict = {osp.splitext(f)[0]: f for f in label_filenames}
    common_keys = set(pred_dict.keys()) & set(label_dict.keys())

    TP_all = FP_all = FN_all = 0
    mae_start_all = []
    mae_end_all = []
    
    list_labels = []
    list_preds = []

    for key in sorted(common_keys):
        pred_name = pred_dict[key]
        label_name = label_dict[key]

        with open(osp.join(args.label_dir, label_name), "r", encoding="utf-8") as f:
            sample = f.read(2048)
            delimiter = csv.Sniffer().sniff(sample).delimiter
        true_df = pd.read_csv(osp.join(args.label_dir, label_name), delimiter=delimiter)
        true_timecodes = df_en_tuples(true_df)

        pred_path = osp.join(args.pred_dir, pred_name)
        
        if args.method == "omine_paper":
            model_timecodes = pd.read_csv(pred_path)
            model_filtered = model_timecodes.loc[model_timecodes["label"] == "risa", ["t0", "t1"]]
            pred_timecodes = df_en_tuples(model_filtered)
        elif args.method == "omine_baseline":
            model_timecodes = pd.read_csv(pred_path)
            model_filtered = model_timecodes.loc[model_timecodes["source"] == "Initial", ["t0", "t1"]]
            pred_timecodes = df_en_tuples(model_filtered)
        elif args.method == "gillick" and pred_path.endswith('.csv'):
            g_df = pd.read_csv(pred_path)
            pred_timecodes = df_en_tuples(g_df)
        else:
            raw_preds = load_preds(pred_path)
            if isinstance(raw_preds, pd.DataFrame):
                pred_timecodes = df_en_tuples(raw_preds)
            else:
                pred_timecodes = convertir_en_tuples(raw_preds)

        list_labels.append(true_timecodes)
        list_preds.append(pred_timecodes)

        TP, FP, FN, _ = evaluate_predictions_with_iou(pred_timecodes, true_timecodes, args.iou_threshold)
        TP_all += TP
        FP_all += FP
        FN_all += FN

        matches = match_predictions_to_gt_by_start(pred_timecodes, true_timecodes)
        mae_s, mae_e = temporal_mae(matches)
        if not np.isnan(mae_s):
            mae_start_all.append(mae_s)
            mae_end_all.append(mae_e)

    precision, recall, accuracy, f1 = calculate_metrics(TP_all, FP_all, FN_all)   
    mae_start = safe_mean(mae_start_all)
    mae_end = safe_mean(mae_end_all)

    mean_f1, ci_low, ci_high = bootstrap_ci_f1(list_labels, list_preds, args.iou_threshold, n_bootstrap=1000, alpha=0.05)

    results = {
        args.method: {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "F1": f1,
            "F1_bootstrap_mean": mean_f1,
            "F1_ci_low": ci_low,
            "F1_ci_high": ci_high,
            "TP": TP_all,
            "FP": FP_all,
            "FN": FN_all,
            "mae_start": mae_start,
            "mae end": mae_end 
        }
    }

    df_results = pd.DataFrame(results).T
    df_results["method"] = df_results.index
    df_results["iou_threshold"] = args.iou_threshold
    df_results = df_results.reset_index(drop=True)

    output_file = os.path.join(args.output_dir, "metrics.csv")
    
    if os.path.exists(output_file):
        df_db = pd.read_csv(output_file)
        df_db = pd.concat([df_db, df_results], ignore_index=True)
        df_db = df_db.drop_duplicates(subset=["method", "iou_threshold"], keep="last")
    else:
        df_db = df_results

    df_db.to_csv(output_file, index=False)