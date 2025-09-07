"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""

from collections import Counter, defaultdict
import argparse
import os
import joblib
import os.path as osp
import pickle
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

from laughter_detection.core.embedding import Embedding

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-name",
        "-e",
        type=str,
        help="embedding model to use.",
        default="byola",
    )
    parser.add_argument(
        "--cluster-numbers",
        "-c",
        type=str,
        help="numbers of clusters",
        default="10",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="cluster method",
        default="kmeans",
    )
    parser.add_argument(
        "--langue",
        "-l",
        type=str,
        help="cluster method",
        default="kmeans",
    )
    args = parser.parse_args()
    return args

def plot_projection(
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
    print(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = osp.join(plot_dir, f"laughter_clusters_{projection_name}.png")
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    cluster_numbers = args.cluster_numbers
    cluster_method = args.method
    langue = args.langue
    root_dir = os.path.expanduser("~/data/all")
    root_data = os.path.expanduser("~/data")

    raw_data_langue = os.path.join(root_data,langue)
    diff = os.path.join(root_data,langue,"diff")
    embedding_raw = os.path.join(root_data,langue,"embedding")

    pca_dir= osp.join(root_dir, "laughter", embedding_name, cluster_method,"pca_model.joblib")
    umap_dir= osp.join(root_dir, "laughter", embedding_name, cluster_method, "umap_model.joblib")
    cluster_info_path = osp.join(root_dir, "laughter", embedding_name, cluster_method,langue, f"clusters_{cluster_method}.json")
    
    model_path = os.path.join(root_dir, "models" ,embedding_name, cluster_method , "isolation.joblib")
    laughter_dir = osp.join(root_dir, "laughter", embedding_name, cluster_method ,langue, "test")
    centroid_dir = osp.join(root_dir, "laughter", embedding_name, cluster_method,"centroids.npy")

    os.makedirs(laughter_dir, exist_ok=True)

    laughter_detector = Embedding(
        embedding_name[:-3], raw_data_langue
    )
    
    test_nonsilent_timecodes, test_episode_filenames = [], []
    embedding_list_train = []
    embedding_list_test = []
    for filename in os.listdir(raw_data_langue):
        raw_path = os.path.join(raw_data_langue, filename)
        diff_path = os.path.join(diff, filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            #nonsilent_timecodes.extend(current_nonsilent)
            #episode_filenames.extend(current_filenames)
            embedding_path = os.path.join(embedding_raw,embedding_name,filename[:-4] + ".pt")
            if os.path.isfile(embedding_path):
                embedding = torch.load(embedding_path)

                if embedding_name.startswith("b+w"):
                    target_dim = 2560
                elif embedding_name.startswith("byola"):
                    target_dim = 2480
                elif embedding_name.startswith("wav2clip"):
                    target_dim = 512
                if len(embedding.shape) < 2:
                    embedding = embedding.unsqueeze(0) 

                if embedding.shape[1] < target_dim:
                    pad_size = target_dim - embedding.shape[1]
                    padding = torch.zeros((embedding.shape[0], pad_size), device=embedding.device, dtype=embedding.dtype)
                    embedding = torch.cat([embedding, padding], dim=1)
    
                elif embedding.shape[1] > target_dim:
                    embedding = embedding[:, :target_dim]

            parts = os.path.normpath(raw_data_langue).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test.append(embedding)
                test_nonsilent_timecodes.extend(current_nonsilent)      
                test_episode_filenames.extend(current_filenames)
            else:
                embedding_list_train.append(embedding)
    
    train_audio_embeddings = torch.vstack(embedding_list_train)
    test_audio_embeddings = torch.vstack(embedding_list_test)


    if cluster_method == "kmeans":
        centroids = np.load(centroid_dir)
        cluster_results = pairwise_distances_argmin(test_audio_embeddings, centroids)
        with open(cluster_info_path, "r") as f:
            cluster_infos = json.load(f)
        train_cluster_order = [info["cluster_id"] for info in cluster_infos]
        cluster_id_remap = {old_id: new_id for new_id, old_id in enumerate(train_cluster_order)}
        remapped_results = np.array([cluster_id_remap.get(old_id, -1) for old_id in cluster_results])
        music_cluster_set = {0,8,10,19}

    elif cluster_method == "spectral":
        spectral = SpectralClustering(
                n_clusters=int(cluster_numbers),
                affinity='nearest_neighbors',
              n_neighbors=10,
                assign_labels='kmeans',
                random_state=0
                )
        cluster_results = spectral.fit_predict(test_audio_embeddings)
    
    elif cluster_method == "isolation":
        centroids = None
        isolation_model = joblib.load(model_path)
        test_np = test_audio_embeddings.cpu().numpy()
        preds = isolation_model.predict(test_np)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        music_cluster_set = {1}

        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])

    pca = joblib.load(pca_dir)
    umap_model = joblib.load(umap_dir)

    plot_projection(
        embeddings_2d=pca.transform(test_audio_embeddings.cpu().numpy()),
        projection_name="pca",
        remapped_results=remapped_results,
        cluster_method=cluster_method,
        embedding_name=embedding_name,
        laughter_dir=laughter_dir,
        centroids=centroids,
        projection_model=pca,
        music_cluster_set=music_cluster_set
    )

    plot_projection(
        embeddings_2d=umap_model.transform(test_audio_embeddings.cpu().numpy()),
        projection_name="umap",
        remapped_results=remapped_results,
        cluster_method=cluster_method,
        embedding_name=embedding_name,
        laughter_dir=laughter_dir,
        centroids=centroids,
        projection_model=umap_model,
        music_cluster_set=music_cluster_set
    )

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
        merged_timecodes = laughter_detector._merge_segments(timecodes)
        laughter_timecodes[filename] = merged_timecodes

    print(dict(laughter_timecodes))


    pred_timecodes = dict(laughter_timecodes)

    for current_filename, current_timecodes in pred_timecodes.items():
        laughter_filename = f"{current_filename[:-4]}.pk"
        laughter_path = osp.join(laughter_dir, laughter_filename)

        # Save laughter timecodes
        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)
            