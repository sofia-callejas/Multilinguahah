from collections import Counter, defaultdict
import argparse
import os
import os.path as osp
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Tuple
from laughter_detection.core.embedding import Embedding
from laughter_detection.core.voice_remover import VoiceRemover


def merge_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or consecutive time segments."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    index, length = 0, len(segments)

    while (length > 1) and (length - index > 1):
        if max(segments[index]) >= min(segments[index + 1]):
            new_segment = [min(segments[index]), max(segments[index + 1])]
            segments.pop(index)
            segments.insert(index, new_segment)
            segments.pop(index + 1)
        else:
            index += 1
        length = len(segments)
    return segments


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        "-data",
        type=str,
        help="path of the data",
        default="~/data/train",
    )
    parser.add_argument(
        "--labels_dir",
        "-labels",
        type=str,
        help="path of the labels",
        default="test",
    )
    parser.add_argument(
        "--embedding-name",
        "-e",
        type=str,
        help="embedding model to use.",
        default="byola",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="isolation method",
        default="isolation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    cluster_method = args.method
    root_dir = os.path.expanduser(args.root_dir)
    labels_dir = args.labels_dir

    model_path = os.path.join(root_dir, "models", embedding_name, cluster_method)
    os.makedirs(model_path, exist_ok=True)

    train_embeddings = []
    train_nonsilent_timecodes, train_episode_filenames = [], []
    path_laughter_dir = []
    filename_to_laughter_dir = {}

    # Walk through dataset
    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith("raw"):  # only process raw/ folders
            lang_dir = os.path.dirname(subdir)         # e.g. data/train/cs
            lang_code = os.path.basename(lang_dir)

            laughter_dir = osp.join(root_dir, "laughter_all",lang_code, embedding_name, cluster_method)

            os.makedirs(laughter_dir, exist_ok=True)

            diff_dir = os.path.join(lang_dir, "diff")  
            embedding_dir = os.path.join(lang_dir, "embedding",embedding_name)
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)
            test_labels_dir = os.path.join(labels_dir, lang_code, "audio","labels")
            #test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv"))

            raw_files = [f for f in files if f.endswith(".wav")]

            remove_voice = VoiceRemover(subdir)
            get_embeddings = Embedding(embedding_name, subdir)

            for filename in raw_files:
                input_path = os.path.join(subdir, filename)
                diff_path = os.path.join(diff_dir, filename)
                embedding_path = os.path.join(embedding_dir, filename.replace(".wav", ".pt"))


                if not os.path.exists(diff_path):
                    print(f"Processing {input_path} → {diff_path}")
                    remove_voice._get_diff(audio_filename=filename)
                
                if not os.path.exists(embedding_path):
                    print(f"Creating embedding for {embedding_path}")
                    get_embeddings._get_embeddings(audio_filename=filename)

                embedding = torch.load(embedding_path)
                current_nonsilent = get_embeddings._get_nonsilent(filename)
                current_filenames = [filename for _ in current_nonsilent]

                if len(embedding.shape) < 2:
                    embedding = embedding.unsqueeze(0)

                if embedding_name.startswith("b+w"):
                    target_dim = 2560
                elif embedding_name.startswith("byola"):
                    target_dim = 2480
                elif embedding_name.startswith("audioset"):
                    target_dim = 2480
                elif embedding_name.startswith("wav2clip"):
                    target_dim = 512
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_name}")

                if embedding.shape[1] < target_dim:
                    pad_size = target_dim - embedding.shape[1]
                    padding = torch.zeros((embedding.shape[0], pad_size),
                                  device=embedding.device,
                                  dtype=embedding.dtype)
                    embedding = torch.cat([embedding, padding], dim=1)
                elif embedding.shape[1] > target_dim:
                    embedding = embedding[:, :target_dim]
                
                path_laughter_dir.append(laughter_dir)
                filename_to_laughter_dir[filename] = laughter_dir
                train_embeddings.append(embedding)
                train_nonsilent_timecodes.extend(current_nonsilent)
                train_episode_filenames.extend(current_filenames)

    train_audio_embeddings = torch.vstack(train_embeddings)

    if cluster_method == "isolation":
        isolation_model = joblib.load(os.path.join(model_path, "isolation.joblib"))
        test_np = train_audio_embeddings.cpu().numpy()
        preds = isolation_model.predict(test_np)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        music_cluster_set = {1}
        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])

    elif cluster_method == "funnynet":
        clusterer = joblib.load(os.path.join(model_path, "kmeans.joblib"))
        test_np = train_audio_embeddings.cpu().numpy().astype(np.float32)
        cluster_labels = clusterer.predict(test_np)
        cluster_counts = Counter(cluster_labels)
        largest_cluster = max(cluster_counts, key=cluster_counts.get)
        cluster_results = np.array([
            1 if lbl == largest_cluster else 0
            for lbl in cluster_labels
        ])
        music_cluster_set = {1}

        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])

    music_indices = np.where(np.isin(remapped_results, list(music_cluster_set)))[0]

    laughter_timecodes = defaultdict(list)

    n_detections = len(train_nonsilent_timecodes)
    for detection_index in range(n_detections):
        if detection_index in music_indices:
            continue
        timecode = train_nonsilent_timecodes[detection_index]
        filename = train_episode_filenames[detection_index]
        laughter_timecodes[filename].append(timecode)

    for filename, timecodes in laughter_timecodes.items():
        merged_timecodes = merge_segments(timecodes)
        laughter_timecodes[filename] = merged_timecodes

    pred_timecodes = dict(laughter_timecodes)

    for i, (current_filename, current_timecodes) in enumerate(pred_timecodes.items()):
        laughter_filename = f"{current_filename[:-4]}.pk"
        from pathlib import Path
        laughter_dir = filename_to_laughter_dir.get(current_filename)

        if laughter_dir is None:
            print(f"Warning: No laughter directory found for file {current_filename}. Skipping.")
            continue

        laughter_path = osp.join(laughter_dir, laughter_filename)

        parts = Path(laughter_path).parts
        lang_index = parts.index("laughter_all") + 1
        language = parts[lang_index]
        filename = Path(laughter_path).with_suffix(".wav").name

        new_path = Path(root_dir) / language / "raw" / filename
        
        if not new_path.exists():
            print(new_path) 

        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)

    all_filenames = set(filename_to_laughter_dir.keys())
    predicted_filenames = set(pred_timecodes.keys())
    missing_filenames = all_filenames - predicted_filenames
    print(missing_filenames)

    for filename in missing_filenames:
        laughter_dir = filename_to_laughter_dir.get(filename)
        if laughter_dir is None:
            print(f"Warning: No laughter directory found for file {filename}. Skipping.")
            continue

        laughter_filename = f"{filename[:-4]}.pk"
        laughter_path = osp.join(laughter_dir, laughter_filename)


        parts = Path(laughter_path).parts
        lang_index = parts.index("laughter_all") + 1
        language = parts[lang_index]
        original_audio_path = Path(root_dir) / language / "raw" / filename
        if not original_audio_path.exists():
            print(f"Missing audio file: {original_audio_path}")


        with open(laughter_path, "wb") as f:
            pickle.dump([], f)

