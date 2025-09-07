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
        "pred_dir", type=str, help="Path to the prediction directory (.pickle)"
    )
    parser.add_argument(
        "label_dir", type=str, help="Path to the label directory (.pickle)"
    )
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory (.wav)")
    parser.add_argument(
        "output_dir", type=str, help="Path to the output score directory"
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


if __name__ == "__main__":
    args = parse_arguments()
    pred_dir = args.pred_dir
    label_dir = args.label_dir
    audio_dir = args.audio_dir
    output_dir = args.output_dir

    pred_filenames = sorted(os.listdir(pred_dir))
    label_filenames = sorted(os.listdir(label_dir))

    temporal_scores, detect_scores = {}, defaultdict(list)
    
    TP_all = 0
    FP_all = 0
    FN_all = 0
    print(pred_filenames, label_filenames)
    
    for pred_name, label_name in zip(pred_filenames, label_filenames):
        audio_path = osp.join(audio_dir, pred_name[:-3] + ".wav")

        # Load predicted and true laughter timecodes
        pred_timecodes = load_preds(osp.join(pred_dir, pred_name))
        pred_timecodes = convertir_en_tuples(pred_timecodes)
        true_timecodes = pd.read_csv(osp.join(label_dir, label_name),delimiter=";")
        true_timecodes = df_en_tuples(true_timecodes)

        iou_threshold = 0.2  # Set the IoU threshold
        
        TP, FP, FN, not_matched = evaluate_predictions_with_iou(pred_timecodes, true_timecodes, iou_threshold)
        TP_all += TP
        FP_all += FP
        FN_all += FN

    precision, recall, accuracy, f1= calculate_metrics(TP_all, FP_all, FN_all)   
    
    
    print(pred_dir[5:])
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1: {f1:.2f}")
    print('\n')
    
    results = {
    'precision': precision,
    'recall': recall,
    'accuracy': accuracy,
    'F1': f1,
    'TP_all': TP_all,
    'FP_all': FP_all,
    'FN_all': FN_all           
        }

    df = pd.DataFrame([results])
    
    df.to_csv(output_dir, index=False)
    
    

