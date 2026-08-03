import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_root", type=str)
    parser.add_argument("model_root", type=str)
    parser.add_argument("gillick_root", type=str)
    parser.add_argument("funnynet_root", type=str)
    parser.add_argument("label_root", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("iou_threshold", type=float)
    parser.add_argument("model_type", type=str)
    return parser.parse_args()

def load_preds(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def convertir_en_tuples(liste_de_listes):
    return [tuple(pair) for pair in liste_de_listes]

def read_csv_flexible(path):
    with open(path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = csv.Sniffer().sniff(sample).delimiter
        df = pd.read_csv(f, sep=delimiter)
    df.columns = df.columns.str.strip()
    return df

def df_en_tuples(df):
    start_col = end_col = None
    for name in df.columns:
        if name.lower() in ["t0", "start", "start_sec", "time_start"]:
            start_col = name
        if name.lower() in ["t1", "end", "end_sec", "time_end"]:
            end_col = name
    if start_col is None or end_col is None:
        raise ValueError(f"Cannot find start/end columns in {df.columns}")
    return [tuple(round(val,2) for val in row) for row in df[[start_col,end_col]].values]

def calculate_iou(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    intersection = max(0, min(end1,end2) - max(start1,start2))
    union = (end1-start1) + (end2-start2) - intersection
    return 0.0 if union==0 else intersection/union

def calculate_metrics(TP, FP, FN):
    f1 = 0.0 if 2*TP+FP+FN==0 else 2*TP/(2*TP+FP+FN)
    precision = 0.0 if TP+FP==0 else TP/(TP+FP)
    recall = 0.0 if TP+FN==0 else TP/(TP+FN)
    total = TP+FP+FN
    accuracy = 0.0 if total==0 else TP/total
    return precision, recall, accuracy, f1

def get_interval_accuracies(gt_segments, pred_segments, iou_threshold=0.5):
    records = []
    used_preds = set()
    for gt in gt_segments:
        best_iou = 0
        best_idx = -1
        for i,pred in enumerate(pred_segments):
            iou = calculate_iou(gt,pred)
            if iou>best_iou and i not in used_preds:
                best_iou = iou
                best_idx = i
        matched = best_iou>=iou_threshold
        if matched: used_preds.add(best_idx)
        gt_start, gt_end = gt
        records.append({"length":gt_end-gt_start,"correct":int(matched)})
    return pd.DataFrame(records)

if __name__=="__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    languages = [d for d in os.listdir(args.label_root) if osp.isdir(osp.join(args.label_root,d))]
    all_true, all_pred, all_baseline, all_funnynet, all_gillick = [], [], [], [], []

    for lang in languages:
        if lang == "es_ch":
            continue
        pred_dir = osp.join(args.pred_root, lang,"byola-f","isolation")
        model_dir = osp.join(args.model_root, lang)
        label_dir = osp.join(args.label_root, lang, "audio", "labels")
        gillick_dir = osp.join(args.gillick_root, lang,"raw","gillick")
        funnynet_dir = osp.join(args.funnynet_root, lang, "byola", "funnynet")

        if not osp.exists(pred_dir) or not osp.exists(model_dir) or not osp.exists(label_dir):
            continue

        pred_files = sorted(os.listdir(pred_dir))
        label_files = sorted(os.listdir(label_dir))
        model_files = sorted(os.listdir(model_dir))
        gillick_files = sorted(os.listdir(gillick_dir))
        funnynet_files = sorted(os.listdir(funnynet_dir))

        pred_dict = {osp.splitext(f)[0]:f for f in pred_files}
        label_dict = {osp.splitext(f)[0]:f for f in label_files}
        model_dict = {osp.splitext(f)[0]:f for f in model_files}
        gillick_dict = {osp.splitext(f)[0]:f for f in gillick_files}
        funnynet_dict =  {osp.splitext(f)[0]:f for f in funnynet_files}

        common_keys = set(pred_dict.keys()) & set(label_dict.keys()) & set(model_dict.keys())
        if len(common_keys)==0:
            continue

        for key in sorted(common_keys):
            pred_timecodes = convertir_en_tuples(load_preds(osp.join(pred_dir,pred_dict[key])))
            funnynet_timecodes = convertir_en_tuples(load_preds(osp.join(funnynet_dir,funnynet_dict[key])))
            true_timecodes = df_en_tuples(read_csv_flexible(osp.join(label_dir,label_dict[key])))
            gillick_timecodes = df_en_tuples(read_csv_flexible(osp.join(gillick_dir,gillick_dict[key])))

            model_df = read_csv_flexible(osp.join(model_dir,model_dict[key]))
            if "source" in model_df.columns:
                baseline_df = model_df.loc[model_df["source"]=="Initial", ["t0","t1"]]
            else:
                baseline_df = model_df[["t0","t1"]]

            all_true.extend(true_timecodes)
            all_pred.extend(pred_timecodes)
            all_baseline.extend(df_en_tuples(baseline_df))
            all_funnynet.extend(funnynet_timecodes)
            all_gillick.extend(gillick_timecodes)

    methods = {
        "Omine": all_baseline,
        "MultiLinguahah": all_pred,
        "FunnyNet": all_funnynet,
        "Gillick": all_gillick,
    }

    colors = {
        "MultiLinguahah":"#b41f4e", 
        "Omine": "#8e8e8e",
        "FunnyNet": "#241fb4" ,
        "Gillick": "#601fb4" ,          
    }

    plt.figure(figsize=(10,6))
    for name, segments in methods.items():
        df = get_interval_accuracies(all_true, segments, iou_threshold=args.iou_threshold)
        lengths = df["length"].values
        corrects = df["correct"].values
        sorted_idx = np.argsort(lengths)
        plt.plot(lengths[sorted_idx], corrects[sorted_idx], label=name)
    plt.xlabel("Interval Length (s)")
    plt.ylabel("Matched (1) or Not (0)")
    plt.title("Detection matches vs interval length (continuous)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(args.output_dir,"continuous_intervals.png"), dpi=300)
    plt.close()

    interval_bins = [(0,1),(1,10)]
    plt.figure(figsize=(8,5))
    for name, segments in methods.items():
        df = get_interval_accuracies(all_true, segments, iou_threshold=args.iou_threshold)
        f1_scores = []
        for start,end in interval_bins:
            bin_df = df[(df["length"]>=start)&(df["length"]<end)]
            if len(bin_df)==0:
                f1_scores.append(np.nan)
            else:
                TP = bin_df["correct"].sum()
                FP = len(bin_df)-TP
                FN = len(bin_df)-TP
                _, _, _, f1 = calculate_metrics(TP, FP, FN)
                f1_scores.append(f1)
        plt.bar([f"{name}\n0-1s", f"{name}\n1-10s"], f1_scores, alpha=0.7, label=name)
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Interval Length (0-1s, 1-10s) Standup4AI IoU = 0.7")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(osp.join(args.output_dir,"f1_0-1_1-10s.png"), dpi=300)
    plt.close()

    fixed_bins = [(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,3.5),(3.5,4)]
    bin_centers = np.array([(start + end) / 2 for start, end in fixed_bins])
    total_width = 0.4  
    num_methods = len(methods)
    bar_width = total_width / num_methods
    plt.figure(figsize=(10,6))
    for i, (name, segments) in enumerate(methods.items()):
        df = get_interval_accuracies(all_true, segments, iou_threshold=args.iou_threshold)
        f1_scores = []
        for start,end in fixed_bins:
            bin_df = df[(df["length"]>=start)&(df["length"]<end)]
            if len(bin_df) == 0:
                f1_scores.append(0) 
            else:
                TP = bin_df["correct"].sum()
                FP = len(bin_df) - TP
                FN = len(bin_df) - TP
                _, _, _, f1 = calculate_metrics(TP, FP, FN)
                f1_scores.append(f1)
        current_color = colors.get(name, "gray")
        offset = (i - (num_methods - 1) / 2) * bar_width
        plt.bar(bin_centers + offset, f1_scores, width=bar_width, 
            label=name, color=current_color, alpha=0.9)

    plt.xlabel("Interval Length (s)", fontsize=15)
    plt.ylabel("F1 Score", fontsize=15)
    plt.title("F1 Score vs Interval Length\nStandup4AI IoU = 0.7", fontsize=20)
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.savefig(osp.join(args.output_dir,"f1_fixed_bins.png"), dpi=300)
    plt.close()