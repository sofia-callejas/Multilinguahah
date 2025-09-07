"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""


import argparse
import os
import os.path as osp
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
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
    cluster_path = os.path.join(root_dir,"elbow_analysis",embedding_name,cluster_method)
    os.makedirs(cluster_path, exist_ok=True)

    
    embedding_list_train = []
    embedding_list_test = []
    for filename in os.listdir(root_dir):
        embedding_path = os.path.join(root_dir, "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path):
            embedding = torch.load(embedding_path)
            if embedding_name == "b+w":
                target_dim = 2560
            elif embedding_name == "byola":
                target_dim = 2480
            elif embedding_name == "byola-v2":
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
    
    train_audio_embeddings = torch.vstack(embedding_list_train)
    test_audio_embeddings = torch.vstack(embedding_list_test)

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

            plt.figure(figsize=(12, 12))
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

            plt.figure(figsize=(12, 12))
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
            