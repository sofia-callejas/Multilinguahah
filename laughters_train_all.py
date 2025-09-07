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
    
    laughter_dir = osp.join(root_dir, "laughter", embedding_name, cluster_method,langue)

    os.makedirs(laughter_dir, exist_ok=True)
    laughter_detector_cs = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/cs")
        )
    laughter_detector_en_uk = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/en_uk")
        )
    laughter_detector_es = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/es")
        )
    laughter_detector_fr = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/fr")
        )
    laughter_detector_hu = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/hu")
        )
    laughter_detector_it = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/it")
        )
    laughter_detector_pt = Embedding(
            embedding_name[:-3], os.path.expanduser("~/data/pt")
        )

#cs

    test_nonsilent_timecodes_cs, test_episode_filenames_cs = [], []
    embedding_list_train_cs = []
    embedding_list_test_cs = []
    for filename in os.listdir(os.path.expanduser("~/data/cs")):
        raw_path = os.path.join(os.path.expanduser("~/data/cs"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_cs._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/cs"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_cs.append(embedding)
                test_nonsilent_timecodes_cs.extend(current_nonsilent)      
                test_episode_filenames_cs.extend(current_filenames)
            else:
                embedding_list_train_cs.append(embedding)

#en_uk

    test_nonsilent_timecodes_en_uk, test_episode_filenames_en_uk = [], []
    embedding_list_train_en_uk = []
    embedding_list_test_en_uk = []
    for filename in os.listdir(os.path.expanduser("~/data/en_uk")):
        raw_path = os.path.join(os.path.expanduser("~/data/en_uk"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_en_uk._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/en_uk"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_en_uk.append(embedding)
                test_nonsilent_timecodes_en_uk.extend(current_nonsilent)      
                test_episode_filenames_en_uk.extend(current_filenames)
            else:
                embedding_list_train_en_uk.append(embedding)

#es

    test_nonsilent_timecodes_es, test_episode_filenames_es = [], []
    embedding_list_train_es = []
    embedding_list_test_es = []
    for filename in os.listdir(os.path.expanduser("~/data/es")):
        raw_path = os.path.join(os.path.expanduser("~/data/es"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_es._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/es"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_es.append(embedding)
                test_nonsilent_timecodes_es.extend(current_nonsilent)      
                test_episode_filenames_es.extend(current_filenames)
            else:
                embedding_list_train_es.append(embedding)

#fr

    test_nonsilent_timecodes_fr, test_episode_filenames_fr = [], []
    embedding_list_train_fr = []
    embedding_list_test_fr = []
    for filename in os.listdir(os.path.expanduser("~/data/fr")):
        raw_path = os.path.join(os.path.expanduser("~/data/fr"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_fr._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/fr"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_fr.append(embedding)
                test_nonsilent_timecodes_fr.extend(current_nonsilent)      
                test_episode_filenames_fr.extend(current_filenames)
            else:
                embedding_list_train_fr.append(embedding)

#hu

    test_nonsilent_timecodes_hu, test_episode_filenames_hu = [], []
    embedding_list_train_hu = []
    embedding_list_test_hu = []
    for filename in os.listdir(os.path.expanduser("~/data/hu")):
        raw_path = os.path.join(os.path.expanduser("~/data/hu"), filename)
        if os.path.isfile(raw_path):
            current_nonsilent = laughter_detector_hu._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/hu"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_hu.append(embedding)
                test_nonsilent_timecodes_hu.extend(current_nonsilent)      
                test_episode_filenames_hu.extend(current_filenames)
            else:
                embedding_list_train_hu.append(embedding)

#it

    test_nonsilent_timecodes_it, test_episode_filenames_it = [], []
    embedding_list_train_it = []
    embedding_list_test_it = []
    for filename in os.listdir(os.path.expanduser("~/data/it")):
        raw_path = os.path.join(os.path.expanduser("~/data/it"), filename)
        diff_path = os.path.join(os.path.expanduser("~/data/it/diff"), filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector_it._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/it"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_it.append(embedding)
                test_nonsilent_timecodes_it.extend(current_nonsilent)      
                test_episode_filenames_it.extend(current_filenames)
            else:
                embedding_list_train_it.append(embedding)

#pt

    test_nonsilent_timecodes_pt, test_episode_filenames_pt = [], []
    embedding_list_train_pt = []
    embedding_list_test_pt = []
    for filename in os.listdir(os.path.expanduser("~/data/pt")):
        raw_path = os.path.join(os.path.expanduser("~/data/pt"), filename)
        diff_path = os.path.join(os.path.expanduser("~/data/pt/diff"), filename)
        if os.path.isfile(raw_path) and os.path.isfile(diff_path):
            current_nonsilent = laughter_detector_pt._get_nonsilent(filename)
            current_filenames = [filename for _ in current_nonsilent]
            embedding_path = os.path.join(os.path.expanduser("~/data/pt"), "embedding",embedding_name,filename[:-4] + ".pt")
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

            if os.path.splitext(filename)[0] in test_labels:
                embedding_list_test_pt.append(embedding)
                test_nonsilent_timecodes_pt.extend(current_nonsilent)      
                test_episode_filenames_pt.extend(current_filenames)
            else:
                embedding_list_train_pt.append(embedding)

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
        k_means = KMeans(n_clusters=int(cluster_numbers))
        k_means.fit(train_audio_embeddings)
        if langue == "cs":
            test_audio_embeddings_cs = torch.vstack(embedding_list_test_cs)
            cluster_results = k_means.predict(test_audio_embeddings_cs)
        elif langue == "en_uk":
            test_audio_embeddings_en_uk = torch.vstack(embedding_list_test_en_uk)
            cluster_results = k_means.predict(test_audio_embeddings_en_uk)
        elif langue == "es":
            test_audio_embeddings_es = torch.vstack(embedding_list_test_es)
            cluster_results = k_means.predict(test_audio_embeddings_es)
        elif langue == "fr":
            test_audio_embeddings_fr = torch.vstack(embedding_list_test_fr)
            cluster_results = k_means.predict(test_audio_embeddings_fr)
        elif langue == "hu":
            test_audio_embeddings_hu = torch.vstack(embedding_list_test_hu)
            cluster_results = k_means.predict(test_audio_embeddings_hu)
        elif langue == "it":
            test_audio_embeddings_it = torch.vstack(embedding_list_test_it)
            cluster_results = k_means.predict(test_audio_embeddings_it)
        elif langue == "pt":
            test_audio_embeddings_pt = torch.vstack(embedding_list_test_pt)
            cluster_results = k_means.predict(test_audio_embeddings_pt)
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
    music_cluster = (
        min(cluster_counts, key=cluster_counts.get) if int(cluster_numbers) != 1 else -1
    )

    (music_indices,) = np.where(cluster_results == music_cluster)
    laughter_timecodes = defaultdict(list)
    if langue == "cs":
        n_detections = len(test_nonsilent_timecodes_cs)
    elif langue == "en_uk":
        n_detections = len(test_nonsilent_timecodes_en_uk)
    elif langue == "es":
        n_detections = len(test_nonsilent_timecodes_es)
    elif langue == "fr":
        n_detections = len(test_nonsilent_timecodes_fr)
    elif langue == "hu":
        n_detections = len(test_nonsilent_timecodes_hu)
    elif langue == "it":
        n_detections = len(test_nonsilent_timecodes_it)
    elif langue == "pt":
        n_detections = len(test_nonsilent_timecodes_pt)
    for detection_index in range(n_detections):
        if detection_index in music_indices:
            continue
        if langue == "cs":
            timecode = test_nonsilent_timecodes_cs[detection_index]
            filename = test_episode_filenames_cs[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "en_uk":
            timecode = test_nonsilent_timecodes_en_uk[detection_index]
            filename = test_episode_filenames_en_uk[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "es":
            timecode = test_nonsilent_timecodes_es[detection_index]
            filename = test_episode_filenames_es[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "fr":
            timecode = test_nonsilent_timecodes_fr[detection_index]
            filename = test_episode_filenames_fr[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "hu":
            timecode = test_nonsilent_timecodes_hu[detection_index]
            filename = test_episode_filenames_hu[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "it":
            timecode = test_nonsilent_timecodes_it[detection_index]
            filename = test_episode_filenames_it[detection_index]
            laughter_timecodes[filename].append(timecode)
        elif langue == "pt":
            timecode = test_nonsilent_timecodes_pt[detection_index]
            filename = test_episode_filenames_pt[detection_index]
            laughter_timecodes[filename].append(timecode)

    for filename, timecodes in laughter_timecodes.items():
        if langue == "cs":    
            merged_timecodes = laughter_detector_cs._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "en_uk":
            merged_timecodes = laughter_detector_en_uk._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "es":
            merged_timecodes = laughter_detector_es._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "fr":
            merged_timecodes = laughter_detector_fr._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "hu":
            merged_timecodes = laughter_detector_hu._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "it":
            merged_timecodes = laughter_detector_it._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes
        elif langue == "pt":
            merged_timecodes = laughter_detector_pt._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes

    print(dict(laughter_timecodes))

    pred_timecodes = dict(laughter_timecodes)

    for current_filename, current_timecodes in pred_timecodes.items():
        laughter_filename = f"{current_filename[:-4]}.pk"
        laughter_path = osp.join(laughter_dir, laughter_filename)

        # Save laughter timecodes
        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)
            