import os
import os.path as osp
import numpy as np
import pandas as pd
import csv
import argparse
from functools import partial

from confidence_intervals import evaluate_with_conf_int
from laughter_detection.core.utils import load_preds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["en_us","en_uk","es_latam","es","fr","it","pt","fr_ca","hu","cs"])
    parser.add_argument("--base-label", type=str, default="/home/scallejas/data/test")
    parser.add_argument("--base-pred", type=str, default="/home/scallejas/data/train/laughter_all")
    parser.add_argument("--base-funnynet", type=str, default="/home/scallejas/data/train/laughter_all")
    parser.add_argument("--base-gillick", type=str, default="/home/scallejas/data/train")
    parser.add_argument("--base-model", type=str, default="/home/scallejas/data/text-model/laughter_detection/validaciones_nahuel")
    parser.add_argument("--output-dir", type=str, default="/home/scallejas/results_all_languages")
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--bootstraps", type=int, default=10000)
    return parser.parse_args()

def df_en_tuples(df):
    if df.empty:
        return []
    return [tuple(round(val, 2) for val in row) for row in df[['t0', 't1']].values]

def calculate_iou(interval1, interval2):
    s1, e1 = interval1
    s2, e2 = interval2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0

def evaluate_predictions_with_iou(preds, labels, threshold):
    if not labels: return 0, len(preds), 0
    if not preds:  return 0, 0, len(labels)

    TP = 0
    matched_preds = set()
    matched_labels = set()

    candidates = []
    for p_idx, p in enumerate(preds):
        for l_idx, l in enumerate(labels):
            iou = calculate_iou(p, l)
            if iou >= threshold:
                candidates.append((iou, p_idx, l_idx))
    
    candidates.sort(key=lambda x: x[0], reverse=True)

    for iou, p_idx, l_idx in candidates:
        if p_idx not in matched_preds and l_idx not in matched_labels:
            TP += 1
            matched_preds.add(p_idx)
            matched_labels.add(l_idx)

    FP = len(preds) - len(matched_preds)
    FN = len(labels) - len(matched_labels)
    return TP, FP, FN

def calculate_f1(TP, FP, FN):
    denom = (2 * TP + FP + FN)
    return 2 * TP / denom if denom > 0 else 0.0

def compute_f1_wrapper(y_true, y_pred, threshold):
    TP_total = FP_total = FN_total = 0
    for gt, pr in zip(y_true, y_pred):
        TP, FP, FN = evaluate_predictions_with_iou(pr, gt, threshold)
        TP_total += TP
        FP_total += FP
        FN_total += FN
    return calculate_f1(TP_total, FP_total, FN_total)

def merge_prioritized_intervals(primary, secondary):
    merged = list(primary)
    for s2, e2 in secondary:
        has_intersection = False
        for s1, e1 in primary:
            if max(s1, s2) < min(e1, e2):
                has_intersection = True
                break
        
        if not has_intersection:
            merged.append((s2, e2))
            
    return sorted(merged, key=lambda x: x[0])

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    f1_evaluator = partial(compute_f1_wrapper, threshold=args.iou_threshold)

    results_data = []
    all_labels_global = []
    all_preds_custom_global = []
    all_preds_funnynet_global = []
    all_preds_gillick_global = []
    all_preds_paper_global = []
    all_preds_baseline_global = []
    all_preds_custom_base_global = []
    all_preds_base_custom_global = []

    for lang in args.languages:
        pred_dir = f"{args.base_pred}/{lang}/wav2clip/isolation"
        funny_dir = f"{args.base_funnynet}/{lang}/byola/funnynet"
        gillick_dir = f"{args.base_gillick}/{lang}/raw/gillick"
        label_dir = f"{args.base_label}/{lang}/audio/labels"
        model_dir = f"{args.base_model}/{lang}"

        if not (osp.exists(pred_dir) and osp.exists(funny_dir) and osp.exists(gillick_dir) and osp.exists(label_dir) and osp.exists(model_dir)):
            continue

        pred_map = {osp.splitext(f)[0]: f for f in os.listdir(pred_dir)}
        funny_map = {osp.splitext(f)[0]: f for f in os.listdir(funny_dir)}
        gillick_map = {osp.splitext(f)[0]: f for f in os.listdir(gillick_dir)}
        label_map = {osp.splitext(f)[0]: f for f in os.listdir(label_dir)}
        model_map = {osp.splitext(f)[0]: f for f in os.listdir(model_dir)}
        
        common_keys = sorted(list(set(pred_map) & set(funny_map) & set(gillick_map) & set(label_map) & set(model_map)))

        lang_labels, lang_custom, lang_funnynet, lang_gillick, lang_paper, lang_baseline = [], [], [], [], [], []
        lang_custom_base, lang_base_custom = [], []

        for key in common_keys:
            try:
                p_raw = load_preds(osp.join(pred_dir, pred_map[key]))
                preds_custom = df_en_tuples(pd.DataFrame(p_raw, columns=["t0", "t1"]))

                f_raw = load_preds(osp.join(funny_dir, funny_map[key]))
                preds_funnynet = df_en_tuples(pd.DataFrame(f_raw, columns=["t0", "t1"]))

                g_path = osp.join(gillick_dir, gillick_map[key])
                if g_path.endswith('.csv'):
                    g_df = pd.read_csv(g_path)
                    preds_gillick = df_en_tuples(g_df)
                else:
                    g_raw = load_preds(g_path)
                    preds_gillick = df_en_tuples(pd.DataFrame(g_raw, columns=["t0", "t1"]))

                l_path = osp.join(label_dir, label_map[key])
                with open(l_path, 'r') as f:
                    sep = ';' if ';' in f.readline() else ','
                l_df = pd.read_csv(l_path, sep=sep)
                preds_labels = df_en_tuples(l_df)

                m_df = pd.read_csv(osp.join(model_dir, model_map[key]))
                preds_paper = df_en_tuples(m_df[m_df["label"] == "risa"])
                preds_baseline = df_en_tuples(m_df[m_df["source"] == "Initial"])
                
                preds_custom_base = merge_prioritized_intervals(preds_custom, preds_baseline)
                preds_base_custom = merge_prioritized_intervals(preds_baseline, preds_custom)

            except Exception as e:
                continue

            lang_custom.append(preds_custom)
            lang_funnynet.append(preds_funnynet)
            lang_gillick.append(preds_gillick)
            lang_labels.append(preds_labels)
            lang_paper.append(preds_paper)
            lang_baseline.append(preds_baseline)
            lang_custom_base.append(preds_custom_base)
            lang_base_custom.append(preds_base_custom)

        all_labels_global.extend(lang_labels)
        all_preds_custom_global.extend(lang_custom)
        all_preds_funnynet_global.extend(lang_funnynet)
        all_preds_gillick_global.extend(lang_gillick)
        all_preds_paper_global.extend(lang_paper)
        all_preds_baseline_global.extend(lang_baseline)
        all_preds_custom_base_global.extend(lang_custom_base)
        all_preds_base_custom_global.extend(lang_base_custom)

        models_to_evaluate = [
            ("custom", lang_custom), 
            ("funnynet", lang_funnynet), 
            ("gillick", lang_gillick),
            ("paper", lang_paper), 
            ("baseline", lang_baseline),
            ("custom+base", lang_custom_base),
            ("base+custom", lang_base_custom)
        ]

        for name, pred_list in models_to_evaluate:
            mean, ci = evaluate_with_conf_int(
                np.array(pred_list, dtype=object), 
                f1_evaluator, 
                np.array(lang_labels, dtype=object),
                num_bootstraps=args.bootstraps, alpha=5
            )
            
            margin = (ci[1] - ci[0]) / 2
            formatted_val = f"{mean:.3f} (±{margin:.3f})"
            
            results_data.append({
                "language": lang, "model": name, "mean": mean, 
                "low": ci[0], "high": ci[1], "display": formatted_val
            })

    global_models_to_evaluate = [
        ("custom", all_preds_custom_global), 
        ("funnynet", all_preds_funnynet_global),
        ("gillick", all_preds_gillick_global),  
        ("paper", all_preds_paper_global), 
        ("baseline", all_preds_baseline_global),
        ("custom+base", all_preds_custom_base_global),
        ("base+custom", all_preds_base_custom_global)
    ]

    for name, pred_list in global_models_to_evaluate:
        mean, ci = evaluate_with_conf_int(
            np.array(pred_list, dtype=object), 
            f1_evaluator, 
            np.array(all_labels_global, dtype=object),
            num_bootstraps=args.bootstraps, alpha=5
        )
        
        margin = (ci[1] - ci[0]) / 2
        formatted_val = f"{mean:.3f} (±{margin:.3f})"
        
        results_data.append({
            "language": "ALL", "model": name, "mean": mean, 
            "low": ci[0], "high": ci[1], "display": formatted_val
        })

    df_final = pd.DataFrame(results_data)
    save_path = osp.join(args.output_dir, "final_results_formatted.csv")
    df_final.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()