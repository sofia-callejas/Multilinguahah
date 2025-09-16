"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""


import argparse
import os
import os.path as osp
import glob
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

from laughter_detection.core.embedding import Embedding
from laughter_detection.core.voice_remover import VoiceRemover


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
        "--cluster-range",
        "-c",
        type=str,
        help="cluster range",
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
    cluster_range = args.cluster_range
    cluster_method = args.method
    root_dir = args.root_dir
    labels_dir = args.labels_dir

    cluster_path = os.path.join(root_dir,"elbow_analysis",embedding_name,cluster_method)
    os.makedirs(cluster_path, exist_ok=True)

    train_embeddings = []

    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith("raw"):  # only process raw/ folders
            lang_dir = os.path.dirname(subdir)         # e.g. data/train/cs
            lang_code = os.path.basename(lang_dir)
            diff_dir = os.path.join(lang_dir, "diff")  # e.g. data/train/cs/diff
            embedding_dir = os.path.join(lang_dir, "embedding",embedding_name)
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)

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
                    print(f"Creating embedding for {filename}")
                    get_embeddings._get_embeddings(audio_filename=filename)

                embedding = torch.load(embedding_path)

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
                
                test_labels_dir = os.path.join(labels_dir, lang_code, "audio","labels")
                test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv"))

                if os.path.splitext(filename)[0] not in test_files:
                    train_embeddings.append(embedding)

    train_audio_embeddings = torch.vstack(train_embeddings)

    inertias = []
    k_range = range(1, int(cluster_range))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(train_audio_embeddings)

    for k in k_range:
        if cluster_method == "kmeans":
            print(f"Fitting KMeans with k={k}")
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(train_audio_embeddings)
            inertias.append(kmeans.inertia_)

            plt.figure(figsize=(20, 20))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='nipy_spectral')
            plt.title(f"KMeans Clustering (k={k})")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.colorbar(scatter, label="Cluster ID")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_path, f"cluster_k_{k}.png"))
            plt.close()

        elif cluster_method == "spectral":
            print(f"Fitting Spectral Clustering with k={k}")
            spectral = SpectralClustering(
                n_clusters=k,
                affinity='nearest_neighbors',
                n_neighbors=10,
                assign_labels='kmeans',
                random_state=0
                )
            labels = spectral.fit_predict(train_audio_embeddings)

            if k == 1:
                inertias.append(0)  
            else:
                score = silhouette_score(train_audio_embeddings, labels)
                inertias.append(score)

            plt.figure(figsize=(20, 20))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='nipy_spectral')
            plt.title(f"KMeans Clustering (k={k})")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.colorbar(scatter, label="Cluster ID")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_path, f"cluster_k_{k}.png"))
            plt.close()
    
    if cluster_method == "kmeans":
        plt.figure(figsize=(6, 4))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method')
        plt.xticks(list(k_range)) 
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_path, "elbow_plot.png"))
        plt.close()
    elif cluster_method == "spectral":
        plt.figure(figsize=(6, 4))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.xticks(list(k_range)) 
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_path, "silhouette_plot.png"))
        plt.close()


