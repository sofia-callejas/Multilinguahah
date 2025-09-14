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
    #create the cluster path
    cluster_path = os.path.join(root_dir,"elbow_analysis",embedding_name,cluster_method)
    os.makedirs(cluster_path, exist_ok=True)

    #remove the voice of the languages 

    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith("raw"):  # only process raw/ folders
            lang_dir = os.path.dirname(subdir)         # e.g. data/train/cs
            diff_dir = os.path.join(lang_dir, "diff")  # e.g. data/train/cs/diff
            embedding_dir = os.path.join(lang_dir, "embedding")
            os.makedirs(diff_dir, exist_ok=True)
            os.makedirs(embedding_dir, exist_ok=True)

            raw_files = [f for f in files if f.endswith(".wav")]
            diff_files = [f for f in os.listdir(diff_dir) if f.endswith(".wav")]
            embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]

            # quick check: skip if everything already processed
            if len(raw_files) == len(diff_files):
                print(f"Skipping {lang_dir}, all {len(raw_files)} files already processed")
                continue
            if len(raw_files) == len(embedding_files):
                print(f"Skipping {lang_dir}, all {len(raw_files)} files already processed")
                continue

            # instantiate per language
            remove_voice = VoiceRemover(subdir)
            get_embeddings = Embedding(embedding_name, subdir)

            for filename in raw_files:
                input_path = os.path.join(subdir, filename)
                diff_path = os.path.join(diff_dir, filename)

                if not os.path.exists(diff_path):
                    print(f"Processing {input_path} → {diff_path}")
                    remove_voice._get_diff(audio_filename=filename)
                    get_embeddings._get_embeddings(audio_filename=filename)


    #create the embedding

    exit()
    #cs
    embedding_list_train_cs = []
    embedding_list_test_cs = []
    for filename in os.listdir(os.path.expanduser("~/data/cs")):
        embedding_path_cs = os.path.join(os.path.expanduser("~/data/cs"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_cs):
            embedding = torch.load(embedding_path_cs)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_cs.append(embedding)
            else:
                embedding_list_train_cs.append(embedding)
    
    #es 
    embedding_list_train_es = []
    embedding_list_test_es = []
    for filename in os.listdir(os.path.expanduser("~/data/es")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/es"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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
            
            parts = os.path.normpath(os.path.expanduser(os.path.expanduser("~/data/es"))).split(os.sep)
            test_path = os.path.join(*parts[-2:],"audio","labels")
            test_filenames = sorted(os.listdir(test_path))
            test_labels = [osp.splitext(f)[0] for f in test_filenames]

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_es.append(embedding)
            else:
                embedding_list_train_es.append(embedding)

#en_uk
    embedding_list_train_en_uk = []
    embedding_list_test_en_uk = []
    for filename in os.listdir(os.path.expanduser("~/data/en_uk")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/en_uk"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_en_uk.append(embedding)
            else:
                embedding_list_train_en_uk.append(embedding)     
    
#fr

    embedding_list_train_fr = []
    embedding_list_test_fr = []
    for filename in os.listdir(os.path.expanduser("~/data/fr")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/fr"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_fr.append(embedding)
            else:
                embedding_list_train_fr.append(embedding) 

#hu

    embedding_list_train_hu = []
    embedding_list_test_hu = []
    for filename in os.listdir(os.path.expanduser("~/data/hu")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/hu"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_hu.append(embedding)
            else:
                embedding_list_train_hu.append(embedding)

#pt

    embedding_list_train_pt = []
    embedding_list_test_pt = []
    for filename in os.listdir(os.path.expanduser("~/data/pt")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/pt"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_pt.append(embedding)
            else:
                embedding_list_train_pt.append(embedding)

#it    
    
    embedding_list_train_it = []
    embedding_list_test_it = []
    for filename in os.listdir(os.path.expanduser("~/data/it")):
        embedding_path_es = os.path.join(os.path.expanduser("~/data/it"), "embedding",embedding_name,filename[:-4] + ".pt")
        if os.path.isfile(embedding_path_es):
            embedding = torch.load(embedding_path_es)
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_it.append(embedding)
            else:
                embedding_list_train_it.append(embedding)    
    
    
    
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
            