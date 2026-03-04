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
import hdbscan
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
        centroids_2d = projection_model.transform(centroids)[:, :2] 
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

            diff_dir = os.path.join(lang_dir, "diff")  # e.g. data/train/cs/diff
            embedding_dir = os.path.join(lang_dir, "embedding",embedding_name)
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)

            raw_files = [f for f in files if f.endswith(".wav")]
            # instantiate per language
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
                
                #test_labels_dir = os.path.join(labels_dir, "diff")

                #test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv"))

                #if os.path.splitext(filename)[0] in test_files:
                #    path_laughter_dir.append(laughter_dir)
                #    filename_to_laughter_dir[filename] = laughter_dir
                #    test_embeddings.append(embedding)
                #    test_nonsilent_timecodes.extend(current_nonsilent)      
                #    test_episode_filenames.extend(current_filenames)
                #else:
                train_embeddings.append(embedding)

    for subdir, _, files in os.walk(labels_dir):
        if subdir.endswith("raw"):  # only process raw/ folders
            lang_dir_labels = os.path.dirname(subdir)         # e.g. data/train/cs
            lang_code_labels = os.path.basename(lang_dir_labels)

            laughter_dir_labels = osp.join(labels_dir, "laughter", embedding_name, cluster_method)

            os.makedirs(laughter_dir, exist_ok=True)

            diff_dir = os.path.join(lang_dir_labels, "diff")  # e.g. data/train/cs/diff
            embedding_dir = os.path.join(lang_dir_labels, "embedding",embedding_name)
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)

            raw_files = [f for f in files if f.endswith(".wav")]
            # instantiate per language
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
                
                #test_labels_dir = os.path.join(labels_dir, "diff")

                #test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv"))

                #if os.path.splitext(filename)[0] in test_files:
                path_laughter_dir.append(laughter_dir)
                filename_to_laughter_dir[filename] = laughter_dir
                test_embeddings.append(embedding)
                test_nonsilent_timecodes.extend(current_nonsilent)      
                test_episode_filenames.extend(current_filenames)
                #else:

    train_audio_embeddings = torch.vstack(train_embeddings)
    test_audio_embedding =  torch.vstack(test_embeddings)



    if cluster_method == "isolation":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(train_audio_embeddings.cpu().numpy())
        isolation = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        isolation.fit(embeddings_norm)
        joblib.dump(isolation, os.path.join(model_path, "isolation.joblib"))
        preds = isolation.predict(embeddings_norm)
        cluster_results = np.array([1 if p == 1 else 0 for p in preds])
        centroids = None

    
    elif cluster_method == "funnynet":
        clusterer = KMeans(n_clusters=4, random_state=42)
        train_np = train_audio_embeddings.cpu().numpy().astype(np.float32)
        cluster_labels = clusterer.fit_predict(train_np)
        centroids = np.array([train_np[cluster_labels == i].mean(axis=0) for i in range(4)])

        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        smallest_cluster = min(cluster_sizes, key=cluster_sizes.get)

        cluster_results = np.array([
            1 if lbl == smallest_cluster else 0
            for lbl in cluster_labels
        ])

        os.makedirs(model_path, exist_ok=True)
        joblib.dump(clusterer, os.path.join(model_path, "kmeans.joblib"))

    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}")

    cluster_counts = Counter(cluster_results)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    cluster_id_remap = {old: new for new, (old, _) in enumerate(sorted_clusters)}
    remapped_results = np.array([cluster_id_remap[c] for c in cluster_results])

    embedding_np = train_audio_embeddings.cpu().numpy()

    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(train_audio_embeddings.cpu().numpy())
    pca_dir = osp.join(model_path,  "pca_model.joblib")
    joblib.dump(pca, pca_dir)
    
    if cluster_method == "isolation-kmeans" or cluster_method == "funnynet":
        centroids_2d = pca.transform(centroids)

    umap_model = UMAP(n_neighbors=15, n_components=5, random_state=42)
    embeddings_2d_umap = umap_model.fit_transform(embedding_np)
    joblib.dump(umap_model, osp.join(model_path, "umap_model.joblib"))

    cluster_infos = []

    for cluster_id in np.unique(remapped_results):
        indices = np.where(remapped_results == cluster_id)[0]
        cluster_points_pca = embeddings_2d_pca[indices]
        cluster_points_umap = embeddings_2d_umap[indices]

        if cluster_method == "kmeans" or cluster_method == "funnynet":
            centroid_pca = centroids_2d[cluster_id]
        else:
            centroid_pca = None  

        cluster_infos.append({
            "cluster_id": int(cluster_id),
            "indices": indices.tolist(),
            "points_pca": cluster_points_pca.tolist(),
            "centroid_pca": centroid_pca.tolist() if centroid_pca is not None else None,
            "points_umap": cluster_points_umap.tolist(),
        })
    
    cluster_info_path = osp.join(model_path, f"clusters_{cluster_method}.json")
    with open(cluster_info_path, "w") as f:
        json.dump(cluster_infos, f, indent=2)

    plot_projection(embeddings_2d_pca, "pca")
    plot_projection(embeddings_2d_umap, "umap")

    #test
    if cluster_method == "isolation":
        centroids = None
        isolation_model = joblib.load(os.path.join(model_path, "isolation.joblib"))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        test_np = test_audio_embedding.cpu().numpy()
        test_np = scaler.fit_transform(test_np)
        
        preds = isolation_model.predict(test_np)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        music_cluster_set = {1}

        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])
        

    elif cluster_method == "funnynet":
        clusterer = joblib.load(os.path.join(model_path, "kmeans.joblib"))
        test_np = test_audio_embedding.cpu().numpy().astype(np.float32)

        cluster_labels = clusterer.predict(test_np)

        cluster_counts = Counter(cluster_labels)

        smallest_cluster = min(cluster_counts, key=cluster_counts.get)

        cluster_counts = np.array([
            1 if lbl == smallest_cluster else 0
            for lbl in cluster_labels
        ])
        music_cluster_set = {1}

        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])




    #plot_projection_test(
    #    embeddings_2d=pca.transform(test_audio_embedding.cpu().numpy()),
    #    projection_name="pca",
    #    remapped_results=remapped_results,
    #    cluster_method=cluster_method,
    #    embedding_name=embedding_name,
    #    laughter_dir=model_path,
    #    centroids=centroids,
    #    projection_model=pca,
    #    music_cluster_set=music_cluster_set
    #)

    #plot_projection_test(
    #    embeddings_2d=umap_model.transform(test_audio_embedding.cpu().numpy()),
    #    projection_name="umap",
    #    remapped_results=remapped_results,
    #    cluster_method=cluster_method,
    #    embedding_name=embedding_name,
    #    laughter_dir=model_path,
    #    centroids=centroids,
    #    projection_model=umap_model,
    #    music_cluster_set=music_cluster_set
    #)

    music_indices = np.where(np.isin(remapped_results, list(music_cluster_set)))[0]
    print(len(remapped_results))
    print(len(music_indices))

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
        #if not new_path.exists():
        #    print(new_path) 

        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)


            