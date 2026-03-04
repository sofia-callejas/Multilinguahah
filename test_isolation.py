"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""

from collections import Counter, defaultdict
import argparse
import os
import os.path as osp
import pickle
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from umap import UMAP
import torch
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Tuple
from laughter_detection.core.embedding import Embedding
from laughter_detection.core.voice_remover import VoiceRemover

def merge_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    index, lenght = 0, len(segments)
    while (lenght > 1) and (lenght - index > 1):
    # Check if the two consecutive segment share a part
        if max(segments[index]) >= min(segments[index + 1]):
            new_segment = [min(segments[index]), max(segments[index + 1])]
            # Add the merged segment and revove the originals
            segments.pop(index)
            segments.insert(index, new_segment)
            segments.pop(index + 1)

        else:
            index += 1
        lenght = len(segments)

    return segments


def plot_projection(embeddings_2d, projection_name):
    plt.figure(figsize=(10, 7))
    n_clusters = len(set(remapped_results))

    for cluster_id in range(n_clusters):
        points = embeddings_2d[remapped_results == cluster_id]
        plt.scatter(points[:, 0], points[:, 1],
                    label=f"Cluster {cluster_id}", alpha=0.7)

    if cluster_method == "kmeans" and projection_name == "pca" and centroids is not None:
        centroids_2d = pca.transform(centroids)
        for cluster_id in range(len(centroids_2d)):
            x, y = centroids_2d[cluster_id]
            plt.scatter(x, y, color='black', marker='X', s=200, edgecolor='white', zorder=5)
            plt.text(x + 0.2, y, f"({x:.2f}, {y:.2f})", fontsize=9, color='black')

    plt.title(f"{cluster_method.upper()} Clusters – {projection_name.upper()}")
    plt.xlabel(f"{projection_name.upper()} 1")
    plt.ylabel(f"{projection_name.upper()} 2")
    plt.legend()
    
    plot_dir = osp.join(laughter_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = osp.join(plot_dir, f"laughter_clusters_{cluster_method}_{projection_name}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_projection_test(
    embeddings_2d,                   # 2D projected embeddings (e.g., PCA or UMAP)
    projection_name,                 # "pca" or "umap"
    remapped_results,                # Cluster labels
    cluster_method,                 # e.g., "kmeans" or "manual"
    embedding_name,                 # Name of the embedding model
    laughter_dir,                   # Root directory to save plot
    centroids=None,                 # Optional: original centroids in high-dim
    projection_model=None,          # PCA or UMAP model (with .transform method)
    music_cluster_set=None          # Optional: clusters to gray out
):
    
    plt.figure(figsize=(10, 7))

    #n_clusters = len(set(remapped_results))
    #print(n_clusters)

    for cluster_id in sorted(set(remapped_results)):
        mask = np.array(remapped_results) == cluster_id
        points = embeddings_2d[mask]

        is_music = music_cluster_set is not None and cluster_id in music_cluster_set
        label = f"Cluster {cluster_id} ({'other' if is_music else 'laugh'})"
        color = 'gray' if is_music else None
        alpha = 0.3 if is_music else 0.7

        plt.scatter(points[:, 0], points[:, 1], label=label, color=color, alpha=alpha)

    if centroids is not None and projection_model is not None:
        centroids_2d = projection_model.transform(centroids)
        for cluster_id, (x, y) in enumerate(centroids_2d):
            plt.scatter(x, y, color='black', marker='X', s=200, edgecolor='white', zorder=5)
            plt.text(x + 0.2, y, f"({x:.2f}, {y:.2f})", fontsize=9, color='black')

    plt.title(f"{cluster_method.upper()} Clusters – {projection_name.upper()} ({embedding_name})")
    plt.xlabel(f"{projection_name.upper()} 1")
    plt.ylabel(f"{projection_name.upper()} 2")
    plt.legend()

    plot_dir = osp.join(laughter_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = osp.join(plot_dir, f"laughter_clusters_{projection_name}.png")
    plt.savefig(plot_path)
    plt.close()

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
        default="/home/vbarrier/data/standup/laughter_detection/test_laughters_manual_annotation",
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    cluster_method = args.method
    root_dir = args.root_dir
    labels_dir = args.labels_dir
    model_path = os.path.join(root_dir, "models" ,embedding_name, cluster_method)
    os.makedirs(model_path, exist_ok=True)

    train_embeddings = []
    test_embeddings = []
    test_nonsilent_timecodes, test_episode_filenames = [], []
    path_laughter_dir = []
    filename_to_laughter_dir = {}


    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith("raw"):  # only process raw/ folders
            lang_dir = os.path.dirname(subdir)         # e.g. data/train/cs
            lang_code = os.path.basename(lang_dir)

            laughter_dir = osp.join(root_dir, "laughter",lang_code, embedding_name, cluster_method)

            os.makedirs(laughter_dir, exist_ok=True)

            diff_dir = os.path.join(lang_dir, "diff")  
            embedding_dir = os.path.join(lang_dir, "embedding",embedding_name)
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)
            test_labels_dir = os.path.join(labels_dir, lang_code)
            test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv"))

            raw_files = [f for f in files if f.endswith(".wav")]

            remove_voice = VoiceRemover(subdir)
            get_embeddings = Embedding(embedding_name, subdir)

            for filename in raw_files:
                if os.path.splitext(filename)[0] in test_files:
                    input_path = os.path.join(subdir, filename)
                    diff_path = os.path.join(diff_dir, filename)
                    embedding_path = os.path.join(embedding_dir, filename.replace(".wav", ".pt"))

                    if os.path.exists(embedding_path):
                        os.remove(embedding_path)
                        print(f"Deleted: {embedding_path}")
                    else:
                        print(f"File not found: {embedding_path}")


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
                    test_embeddings.append(embedding)
                    test_nonsilent_timecodes.extend(current_nonsilent)      
                    test_episode_filenames.extend(current_filenames)

    test_audio_embedding =  torch.vstack(test_embeddings)

    #test
    if cluster_method == "isolation":
        centroids = None
        isolation_model = joblib.load(os.path.join(model_path, "isolation.joblib"))
        test_np = test_audio_embedding.cpu().numpy()
        preds = isolation_model.predict(test_np)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        music_cluster_set = {1}

        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])

    music_indices = np.where(np.isin(remapped_results, list(music_cluster_set)))[0]

    laughter_timecodes = defaultdict(list)
    n_detections = len(test_nonsilent_timecodes)
    for detection_index in range(n_detections):
        if detection_index in music_indices:
            continue
        timecode = test_nonsilent_timecodes[detection_index]
        filename = test_episode_filenames[detection_index]
        laughter_timecodes[filename].append(timecode)

    for filename, timecodes in laughter_timecodes.items():
        merged_timecodes = merge_segments(timecodes)
        laughter_timecodes[filename] = merged_timecodes

    print(dict(laughter_timecodes))


    pred_timecodes = dict(laughter_timecodes)


    for i, (current_filename, current_timecodes) in enumerate(pred_timecodes.items()):
        laughter_filename = f"{current_filename[:-4]}.pk"
        from pathlib import Path
        laughter_dir = filename_to_laughter_dir.get(current_filename)
        if laughter_dir is None:
            print(f"Warning: No laughter directory found for file {current_filename}. Skipping.")
            continue

        path = path_laughter_dir[i]
        path_ = Path(path_laughter_dir[i])
        laughter_path = osp.join(laughter_dir, laughter_filename)

        parts = Path(laughter_path).parts
        lang_index = parts.index("laughter") + 1
        language = parts[lang_index]
        filename = Path(laughter_path).with_suffix(".csv").name

        new_path = Path("test") / language / "audio" / "labels" / filename
        if not new_path.exists():
            print(new_path) 

        # Save laughter timecodes
        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)



            