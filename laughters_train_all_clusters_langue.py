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
from laughter_detection.core.embedding import Embedding

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
        help="langue",
        default="cs",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    cluster_numbers = args.cluster_numbers
    cluster_method = args.method
    langue = args.langue
    root_dir = os.path.expanduser("~/data/all")
    root_data = os.path.expanduser("~/data")
    cluster_path = os.path.join(root_dir,"elbow_analysis",cluster_method)
    model_path = os.path.join(root_dir, "models" ,embedding_name, cluster_method, langue)
    laughter_dir = osp.join(root_dir, "laughter", embedding_name, cluster_method, langue)

    os.makedirs(laughter_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    laughter_detector_cs = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/cs")
        )
    laughter_detector_en_uk = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/en_uk")
        )
    laughter_detector_es = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/es")
        )
    laughter_detector_fr = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/fr")
        )
    laughter_detector_hu = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/hu")
        )
    laughter_detector_it = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/it")
        )
    laughter_detector_pt = Embedding(
            embedding_name, os.path.expanduser("~/data/work/16k/pt")
        )

#cs

    train_nonsilent_timecodes_cs, train_episode_filenames_cs = [], []
    embedding_list_train_cs = []
    embedding_list_test_cs = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/cs")):
        raw_path = os.path.join(os.path.expanduser("~~/data/work/16k/cs"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_cs._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/cs"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/cs")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_cs.append(embedding)
                train_nonsilent_timecodes_cs.extend(current_nonsilent)      
                train_episode_filenames_cs.extend(current_filenames)
            else:
                embedding_list_test_cs.append(embedding)

#en_uk

    train_nonsilent_timecodes_en_uk, train_episode_filenames_en_uk = [], []
    embedding_list_train_en_uk = []
    embedding_list_test_en_uk = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/en_uk")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/en_uk"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_en_uk._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/en_uk"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/en_uk")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_en_uk.append(embedding)
                train_nonsilent_timecodes_en_uk.extend(current_nonsilent)      
                train_episode_filenames_en_uk.extend(current_filenames)
            else:
                embedding_list_test_en_uk.append(embedding)

#es

    train_nonsilent_timecodes_es, train_episode_filenames_es = [], []
    embedding_list_train_es = []
    embedding_list_test_es = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/es")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/es"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_es._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/es"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/es")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_es.append(embedding)
                train_nonsilent_timecodes_es.extend(current_nonsilent)      
                train_episode_filenames_es.extend(current_filenames)
            else:
                embedding_list_test_es.append(embedding)

#fr

    train_nonsilent_timecodes_fr, train_episode_filenames_fr = [], []
    embedding_list_train_fr = []
    embedding_list_test_fr = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/fr")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/fr"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_fr._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/fr"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/fr")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_fr.append(embedding)
                train_nonsilent_timecodes_fr.extend(current_nonsilent)      
                train_episode_filenames_fr.extend(current_filenames)
            else:
                embedding_list_test_fr.append(embedding)

#hu

    train_nonsilent_timecodes_hu, train_episode_filenames_hu = [], []
    embedding_list_train_hu = []
    embedding_list_test_hu = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/hu")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/hu"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_hu._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/hu"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/hu")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_hu.append(embedding)
                train_nonsilent_timecodes_hu.extend(current_nonsilent)      
                train_episode_filenames_hu.extend(current_filenames)
            else:
                embedding_list_test_hu.append(embedding)

#it

    train_nonsilent_timecodes_it, train_episode_filenames_it = [], []
    embedding_list_train_it = []
    embedding_list_test_it = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/it")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/it"), filename)
        diff_path = os.path.join(os.path.expanduser("~/data/it/diff"), filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector_it._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/it"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/it")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_it.append(embedding)
                train_nonsilent_timecodes_it.extend(current_nonsilent)      
                train_episode_filenames_it.extend(current_filenames)
            else:
                embedding_list_test_it.append(embedding)

#pt

    train_nonsilent_timecodes_pt, train_episode_filenames_pt = [], []
    embedding_list_train_pt = []
    embedding_list_test_pt = []
    for filename in os.listdir(os.path.expanduser("~/data/work/16k/pt")):
        raw_path = os.path.join(os.path.expanduser("~/data/work/16k/pt"), filename)
        diff_path = os.path.join(os.path.expanduser("~/data/pt/diff"), filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector_pt._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/work/16k/pt"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            parts = os.path.normpath(os.path.expanduser("~/data/pt")).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] not in test_labels:
                embedding_list_train_pt.append(embedding)
                train_nonsilent_timecodes_pt.extend(current_nonsilent)      
                train_episode_filenames_pt.extend(current_filenames)
            else:
                embedding_list_test_pt.append(embedding)

    embedding_list_train = [
        *embedding_list_train_cs,
        *embedding_list_train_es,
        *embedding_list_train_en_uk,
        *embedding_list_train_fr,
        *embedding_list_train_hu,
        *embedding_list_train_pt,
        *embedding_list_train_it,
    ]

    embedding_list_test = [
        *embedding_list_test_cs,
        *embedding_list_test_es,
        *embedding_list_test_en_uk,
        *embedding_list_test_fr,
        *embedding_list_test_hu,
        *embedding_list_test_pt,
        *embedding_list_test_it,
    ]

    train_audio_embeddings = torch.vstack(embedding_list_train)
    test_audio_embeddings = torch.vstack(embedding_list_test)

    if cluster_method == "kmeans":
        k_means = KMeans(n_clusters=int(cluster_numbers),random_state=42,n_init=1)
        cluster_results = k_means.fit_predict(train_audio_embeddings)
        centroids_dir = osp.join(laughter_dir, "centroids.npy")
        centroids = k_means.cluster_centers_
        np.save(centroids_dir, k_means.cluster_centers_)

    elif cluster_method == "spectral":
        spectral = SpectralClustering(
                n_clusters=int(cluster_numbers),
                affinity='nearest_neighbors',
                n_neighbors=10,
                assign_labels='kmeans',
                random_state=42
                )
        cluster_results = spectral.fit_predict(test_audio_embeddings)
    
    elif cluster_method == "isolation":
        isolation = IsolationForest(contamination='auto', random_state=42)
        isolation.fit(train_audio_embeddings)
        model_path = os.path.join(model_path, "isolation.joblib")
        joblib.dump(isolation, model_path)
        preds = isolation.predict(train_audio_embeddings)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        centroids = None
    
    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}")

    cluster_counts = Counter(cluster_results)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    cluster_id_remap = {old: new for new, (old, _) in enumerate(sorted_clusters)}
    remapped_results = np.array([cluster_id_remap[c] for c in cluster_results])

    embedding_np = train_audio_embeddings.cpu().numpy()

    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(train_audio_embeddings.cpu().numpy())
    pca_dir = osp.join(laughter_dir,  "pca_model.joblib")
    joblib.dump(pca, pca_dir)
    
    if cluster_method == "kmeans":
        centroids_2d = pca.transform(k_means.cluster_centers_)

    umap_model = UMAP(n_components=2, random_state=42)
    embeddings_2d_umap = umap_model.fit_transform(embedding_np)
    joblib.dump(umap_model, osp.join(laughter_dir, "umap_model.joblib"))

    cluster_infos = []

    for cluster_id in np.unique(remapped_results):
        indices = np.where(remapped_results == cluster_id)[0]
        cluster_points_pca = embeddings_2d_pca[indices]
        cluster_points_umap = embeddings_2d_umap[indices]

        if cluster_method == "kmeans":
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
    
    cluster_info_path = osp.join(laughter_dir, f"clusters_{cluster_method}.json")
    with open(cluster_info_path, "w") as f:
        json.dump(cluster_infos, f, indent=2)


    plot_projection(embeddings_2d_pca, "pca")
    plot_projection(embeddings_2d_umap, "umap")

    timecodes_map = {
        "cs": train_nonsilent_timecodes_cs,
        "en_uk": train_nonsilent_timecodes_en_uk,
        "es": train_nonsilent_timecodes_es,
        "fr": train_nonsilent_timecodes_fr,
        "hu": train_nonsilent_timecodes_hu,
        "it": train_nonsilent_timecodes_it,
        "pt": train_nonsilent_timecodes_pt,
    }

    filenames_map = {
        "cs": train_episode_filenames_cs,
        "en_uk": train_episode_filenames_en_uk,
        "es": train_episode_filenames_es,
        "fr": train_episode_filenames_fr,
        "hu": train_episode_filenames_hu,
        "it": train_episode_filenames_it,
        "pt": train_episode_filenames_pt,
    }

    laughter_detectors_map = {
        "cs": laughter_detector_cs,
        "en_uk": laughter_detector_en_uk,
        "es": laughter_detector_es,
        "fr": laughter_detector_fr,
        "hu": laughter_detector_hu,
        "it": laughter_detector_it,
        "pt": laughter_detector_pt,
    }

    current_timecodes = timecodes_map.get(langue)
    current_filenames = filenames_map.get(langue)
    current_detector = laughter_detectors_map.get(langue)

    cluster_to_timecodes = defaultdict(lambda: defaultdict(list)) 
    n_detections = len(current_timecodes)

    for detection_index in range(n_detections):
        cluster = remapped_results[detection_index]
        timecode = current_timecodes[detection_index]
        filename = current_filenames[detection_index]

        cluster_to_timecodes[cluster][filename].append(timecode)

    for cluster_id, file_timecodes_dict in cluster_to_timecodes.items():
        for filename, timecodes in file_timecodes_dict.items():
            merged_timecodes = current_detector._merge_segments(timecodes)
            file_timecodes_dict[filename] = merged_timecodes

    print(dict(cluster_to_timecodes))

    for cluster_id, file_timecodes_dict in cluster_to_timecodes.items():
        cluster_dir = os.path.join(laughter_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        for filename, timecodes in file_timecodes_dict.items():
            laughter_filename = f"{filename[:-4]}.pk"
            laughter_path = osp.join(cluster_dir, laughter_filename)
            with open(laughter_path, "wb") as f:
                pickle.dump(timecodes, f)
            