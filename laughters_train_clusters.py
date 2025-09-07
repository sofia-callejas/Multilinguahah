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
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import joblib
import matplotlib.pyplot as plt

from laughter_detection.core.embedding import Embedding

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="Path to the root of FunnyNet dataset"
    )
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    cluster_numbers = args.cluster_numbers
    cluster_method = args.method
    root_dir = args.root_dir
    diff = os.path.join(root_dir,"diff")
    cluster_path = os.path.join(root_dir,"elbow_analysis",cluster_method)
    
    laughter_dir = osp.join(root_dir, "laughter", embedding_name, cluster_method)
    os.makedirs(laughter_dir, exist_ok=True)
    laughter_detector = Embedding(
        embedding_name, root_dir
    )
    
    train_nonsilent_timecodes, train_episode_filenames = [], []
    embedding_list_train = []
    embedding_list_test = []
    for filename in os.listdir(root_dir):
        raw_path = os.path.join(root_dir, filename)
        diff_path = os.path.join(diff, filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            #nonsilent_timecodes.extend(current_nonsilent)
            #episode_filenames.extend(current_filenames)
            embedding_path = os.path.join(root_dir, "embedding",embedding_name,filename[:-4] + ".pt")
            if os.path.isfile(embedding_path):
                embedding = torch.load(embedding_path)

                if embedding_name == "b+w":
                    target_dim = 2560
                elif embedding_name == "byola":
                    target_dim = 2480
                elif embedding_name == "wav2clip":
                    target_dim = 512
                if len(embedding.shape) < 2:
                    embedding = embedding.unsqueeze(0) 

                if embedding.shape[1] < target_dim:
                    pad_size = target_dim - embedding.shape[1]
                    padding = torch.zeros((embedding.shape[0], pad_size), device=embedding.device, dtype=embedding.dtype)
                    embedding = torch.cat([embedding, padding], dim=1)
    
                elif embedding.shape[1] > target_dim:
                    embedding = embedding[:, :target_dim]

            parts = os.path.normpath(root_dir).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test.append(embedding)
            else:
                embedding_list_train.append(embedding)
                train_nonsilent_timecodes.extend(current_nonsilent)      
                train_episode_filenames.extend(current_filenames)
    
    train_audio_embeddings = torch.vstack(embedding_list_train)
    test_audio_embeddings = torch.vstack(embedding_list_test)

    if cluster_method == "kmeans":
        k_means = KMeans(n_clusters=int(cluster_numbers),random_state=42,n_init=1)
        cluster_results = k_means.fit_predict(train_audio_embeddings)
        plot_dir = osp.join(laughter_dir, "centroids.npy")
        np.save(plot_dir, k_means.cluster_centers_)

        #cluster_results = k_means.predict(test_audio_embeddings)

    elif cluster_method == "spectral":
        spectral = SpectralClustering(
                n_clusters=int(cluster_numbers),
                affinity='nearest_neighbors',
                n_neighbors=10,
                assign_labels='kmeans',
                random_state=0
                )
        cluster_results = spectral.fit_predict(test_audio_embeddings)

    cluster_counts = Counter(cluster_results)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
    remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(train_audio_embeddings.cpu().numpy())
    pca_dir = osp.join(laughter_dir,  "pca_model.joblib")
    joblib.dump(pca, pca_dir)
    centroids_2d = pca.transform(k_means.cluster_centers_)


    plt.figure(figsize=(10, 7))
    for cluster_id in range(int(cluster_numbers)):
        cluster_points = embeddings_2d[remapped_results == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"Cluster {cluster_id}", alpha=0.7)
        
        x, y = centroids_2d[cluster_id]
        plt.scatter(x, y, color='black', marker='X', s=200, edgecolor='white', zorder=5)
        
        plt.text(x + 0.2, y, f"({x:.2f}, {y:.2f})", fontsize=9, color='black')
    plt.title(f"Clusters with {cluster_method} ({embedding_name})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()

    plot_dir = osp.join(laughter_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = osp.join(plot_dir, "laughter_clusters.png")
    plt.savefig(plot_path)
    plt.close()

    cluster_to_timecodes = defaultdict(lambda: defaultdict(list))

    for detection_index in range(len(train_nonsilent_timecodes)):
        cluster = remapped_results[detection_index]
        timecode = train_nonsilent_timecodes[detection_index]
        filename = train_episode_filenames[detection_index]
        cluster_to_timecodes[cluster][filename].append(timecode)

    #for cluster_id, file_timecodes_dict in cluster_to_timecodes.items():
    #    for filename, timecodes in file_timecodes_dict.items():
    #        merged_timecodes = laughter_detector._merge_segments(timecodes)
    #        file_timecodes_dict[filename] = merged_timecodes

    #pred_timecodes = dict(cluster_to_timecodes)


    for cluster_id, file_timecodes_dict in cluster_to_timecodes.items():
        cluster_dir = os.path.join(laughter_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        for filename, timecodes in file_timecodes_dict.items():
            laughter_filename = f"{filename[:-4]}.pk"
            laughter_path = osp.join(cluster_dir, laughter_filename)
            with open(laughter_path, "wb") as f:
                pickle.dump(timecodes, f)
            