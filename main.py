import argparse
import os
import os.path as osp
import pickle
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from umap import UMAP

from laughter_detection.core.embedding import Embedding
from laughter_detection.core.voice_remover import VoiceRemover

def merge_segments(segments):
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

def plot_projection(embeddings_2d, projection_name, remapped_results, cluster_method, laughter_dir, centroids=None, projection_model=None):
    plt.figure(figsize=(10, 7))
    n_clusters = len(set(remapped_results))
    for cluster_id in range(n_clusters):
        points = embeddings_2d[remapped_results == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.7)
    if cluster_method in ["kmeans", "funnynet"] and projection_name == "pca" and centroids is not None and projection_model is not None:
        centroids_2d = projection_model.transform(centroids)[:, :2]
        for cluster_id, (x, y) in enumerate(centroids_2d):
            plt.scatter(x, y, color='black', marker='X', s=200, edgecolor='white', zorder=5)
            plt.text(x + 0.2, y, f"({x:.2f}, {y:.2f})", fontsize=9, color='black')
    plt.title(f"{cluster_method.upper()} Clusters - {projection_name.upper()}")
    plt.xlabel(f"{projection_name.upper()} 1")
    plt.ylabel(f"{projection_name.upper()} 2")
    plt.legend()
    plot_dir = osp.join(laughter_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = osp.join(plot_dir, f"laughter_clusters_{cluster_method}_{projection_name}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_projection_test(embeddings_2d, projection_name, remapped_results, cluster_method, embedding_name, laughter_dir, centroids=None, projection_model=None, music_cluster_set=None):
    plt.figure(figsize=(10, 7))
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
    plt.title(f"{cluster_method.upper()} Clusters - {projection_name.upper()} ({embedding_name})")
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
    parser.add_argument("--root_dir", "-data", type=str, default="~/data/train")
    parser.add_argument("--labels_dir", "-labels", type=str, default="test")
    parser.add_argument("--embedding_name", "-e", type=str, default="byola")
    parser.add_argument("--method", "-m", type=str, default="isolation")
    parser.add_argument("--cluster_numbers", "-c", type=str, default="3")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["test_isolation", "train_clusters", "train_isolation_all", "train_isolation_audioset", "train_isolation"])
    return parser.parse_args()

def process_file(filename, subdir, diff_dir, embedding_dir, args, remove_voice, get_embeddings, task):
    input_path = os.path.join(subdir, filename)
    diff_path = os.path.join(diff_dir, filename)
    embedding_path = os.path.join(embedding_dir, filename.replace(".wav", ".pt"))
    if task == "test_isolation" and os.path.exists(embedding_path):
        os.remove(embedding_path)
    if not os.path.exists(diff_path):
        remove_voice._get_diff(audio_filename=filename)
    if not os.path.exists(embedding_path):
        get_embeddings._get_embeddings(audio_filename=filename)
    embedding = torch.load(embedding_path)
    if len(embedding.shape) < 2:
        embedding = embedding.unsqueeze(0)
    target_dim = 2560 if args.embedding_name.startswith("b+w") else (512 if args.embedding_name.startswith("wav2clip") else 2048)
    if embedding.shape[1] < target_dim:
        pad_size = target_dim - embedding.shape[1]
        padding = torch.zeros((embedding.shape[0], pad_size), device=embedding.device, dtype=embedding.dtype)
        embedding = torch.cat([embedding, padding], dim=1)
    elif embedding.shape[1] > target_dim:
        embedding = embedding[:, :target_dim]
    return embedding, get_embeddings._get_nonsilent(filename)

def process_directories(args):
    data = {"train_embeddings": [], "test_embeddings": [], "test_nonsilent_timecodes": [], "test_episode_filenames": [], "path_laughter_dir": [], "filename_to_laughter_dir": {}}
    if args.task == "train_isolation_audioset":
        for subdir, _, files in os.walk(args.root_dir):
            if subdir.endswith("raw"):
                lang_dir = os.path.dirname(subdir)
                lang_code = os.path.basename(lang_dir)
                laughter_dir = osp.join(args.root_dir, "laughter", lang_code, args.embedding_name, args.method)
                os.makedirs(laughter_dir, exist_ok=True)
                diff_dir = os.path.join(lang_dir, "diff")
                embedding_dir = os.path.join(lang_dir, "embedding", args.embedding_name)
                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(embedding_dir, exist_ok=True)
                raw_files = [f for f in files if f.endswith(".wav")]
                remove_voice = VoiceRemover(subdir)
                get_embeddings = Embedding(args.embedding_name, subdir)
                for filename in raw_files:
                    embedding, _ = process_file(filename, subdir, diff_dir, embedding_dir, args, remove_voice, get_embeddings, args.task)
                    data["train_embeddings"].append(embedding)
        for subdir, _, files in os.walk(args.labels_dir):
            if subdir.endswith("raw"):
                lang_dir = os.path.dirname(subdir)
                lang_code = os.path.basename(lang_dir)
                laughter_dir = osp.join(args.labels_dir, "laughter", args.embedding_name, args.method)
                os.makedirs(laughter_dir, exist_ok=True)
                diff_dir = os.path.join(lang_dir, "diff")
                embedding_dir = os.path.join(lang_dir, "embedding", args.embedding_name)
                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(embedding_dir, exist_ok=True)
                raw_files = [f for f in files if f.endswith(".wav")]
                remove_voice = VoiceRemover(subdir)
                get_embeddings = Embedding(args.embedding_name, subdir)
                for filename in raw_files:
                    embedding, current_nonsilent = process_file(filename, subdir, diff_dir, embedding_dir, args, remove_voice, get_embeddings, args.task)
                    data["path_laughter_dir"].append(laughter_dir)
                    data["filename_to_laughter_dir"][filename] = laughter_dir
                    data["test_embeddings"].append(embedding)
                    data["test_nonsilent_timecodes"].extend(current_nonsilent)
                    data["test_episode_filenames"].extend([filename for _ in current_nonsilent])
    else:
        for subdir, _, files in os.walk(args.root_dir):
            if subdir.endswith("raw"):
                lang_dir = os.path.dirname(subdir)
                lang_code = os.path.basename(lang_dir)
                dir_suffix = "laughter_all" if args.task == "train_isolation_all" else "laughter"
                laughter_dir = osp.join(args.root_dir, dir_suffix, lang_code, args.embedding_name, args.method)
                os.makedirs(laughter_dir, exist_ok=True)
                diff_dir = os.path.join(lang_dir, "diff")
                embedding_dir = os.path.join(lang_dir, "embedding", args.embedding_name)
                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(embedding_dir, exist_ok=True)
                test_labels_dir = os.path.join(args.labels_dir, lang_code, "audio", "labels") if args.task == "train_clusters" else os.path.join(args.labels_dir, lang_code)
                if args.task == "train_isolation_all":
                    test_labels_dir = os.path.join(args.labels_dir, lang_code, "audio", "labels")
                test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_labels_dir) if f.endswith(".csv")) if os.path.exists(test_labels_dir) else set()
                raw_files = [f for f in files if f.endswith(".wav")]
                remove_voice = VoiceRemover(subdir)
                get_embeddings = Embedding(args.embedding_name, subdir)
                for filename in raw_files:
                    is_test = os.path.splitext(filename)[0] in test_files
                    if args.task == "test_isolation" and not is_test:
                        continue
                    embedding, current_nonsilent = process_file(filename, subdir, diff_dir, embedding_dir, args, remove_voice, get_embeddings, args.task)
                    data["filename_to_laughter_dir"][filename] = laughter_dir
                    if args.task == "train_isolation_all" or is_test:
                        data["path_laughter_dir"].append(laughter_dir)
                        data["test_embeddings"].append(embedding)
                        data["test_nonsilent_timecodes"].extend(current_nonsilent)
                        data["test_episode_filenames"].extend([filename for _ in current_nonsilent])
                    if not is_test or args.task in ["train_isolation", "train_isolation_all"]:
                        data["train_embeddings"].append(embedding)
    if data["train_embeddings"]:
        data["train_embeddings"] = torch.vstack(data["train_embeddings"])
    if data["test_embeddings"]:
        data["test_embeddings"] = torch.vstack(data["test_embeddings"])
    return data

def main():
    args = parse_arguments()
    model_path = os.path.join(args.root_dir, "models", args.embedding_name, args.method)
    os.makedirs(model_path, exist_ok=True)
    data = process_directories(args)
    
    if args.task == "test_isolation":
        isolation_model = joblib.load(os.path.join(model_path, "isolation.joblib"))
        preds = isolation_model.predict(data["test_embeddings"])
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
        music_cluster_set = {1}
        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[old] for old in cluster_results])
        music_indices = np.where(np.isin(remapped_results, list(music_cluster_set)))[0]
        laughter_timecodes = defaultdict(list)
        for i in range(len(data["test_nonsilent_timecodes"])):
            if i not in music_indices:
                laughter_timecodes[data["test_episode_filenames"][i]].append(data["test_nonsilent_timecodes"][i])
        for filename, timecodes in laughter_timecodes.items():
            laughter_timecodes[filename] = merge_segments(timecodes)
        for i, (current_filename, current_timecodes) in enumerate(laughter_timecodes.items()):
            laughter_dir = data["filename_to_laughter_dir"].get(current_filename)
            if not laughter_dir: continue
            laughter_path = osp.join(laughter_dir, f"{current_filename[:-4]}.pk")
            os.makedirs(Path(laughter_path).parent, exist_ok=True)
            with open(laughter_path, "wb") as f:
                pickle.dump(current_timecodes, f)
        return

    if args.task == "train_clusters":
        k_means = KMeans(n_clusters=int(args.cluster_numbers), random_state=42, n_init=1)
        cluster_results = k_means.fit_predict(data["train_embeddings"])
        np.save(osp.join(model_path, "centroids.npy"), k_means.cluster_centers_)
        cluster_counts = Counter(cluster_results)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_id_remap = {old: new for new, (old, _) in enumerate(sorted_clusters)}
        remapped_results = np.array([cluster_id_remap[c] for c in cluster_results])
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(data["train_embeddings"])
        joblib.dump(pca, osp.join(model_path, "pca_model.joblib"))
        umap_model = UMAP(n_components=2, random_state=42)
        embeddings_2d_umap = umap_model.fit_transform(data["train_embeddings"])
        joblib.dump(umap_model, osp.join(model_path, "umap_model.joblib"))
        cluster_infos = []
        for cluster_id in np.unique(remapped_results):
            indices = np.where(remapped_results == cluster_id)[0]
            cluster_infos.append({"cluster_id": int(cluster_id), "indices": indices.tolist(), "points_pca": embeddings_2d_pca[indices].tolist(), "centroid_pca": pca.transform(k_means.cluster_centers_)[cluster_id].tolist(), "points_umap": embeddings_2d_umap[indices].tolist()})
        with open(osp.join(model_path, f"clusters_{args.method}.json"), "w") as f:
            json.dump(cluster_infos, f, indent=2)
        plot_projection(embeddings_2d_pca, "pca", remapped_results, args.method, model_path, k_means.cluster_centers_, pca)
        plot_projection(embeddings_2d_umap, "umap", remapped_results, args.method, model_path, None, None)
        centroids = np.load(osp.join(model_path, "centroids.npy"))
        test_cluster_results = pairwise_distances_argmin(data["test_embeddings"], centroids)
        test_remapped = np.array([{old_id: new_id for new_id, old_id in enumerate([info["cluster_id"] for info in cluster_infos])}.get(old, -1) for old in test_cluster_results])
        music_cluster_set = {min(test_remapped, key=Counter(test_cluster_results).get)} if int(args.cluster_numbers) != 1 else set()
        plot_projection_test(pca.transform(data["test_embeddings"].cpu().numpy()), "pca", test_remapped, args.method, args.embedding_name, model_path, centroids, pca, music_cluster_set)
        plot_projection_test(umap_model.transform(data["test_embeddings"].cpu().numpy()), "umap", test_remapped, args.method, args.embedding_name, model_path, centroids, umap_model, music_cluster_set)
        music_indices = np.where(np.isin(test_remapped, list(music_cluster_set)))[0]
        laughter_timecodes = defaultdict(list)
        for i in range(len(data["test_nonsilent_timecodes"])):
            if i not in music_indices:
                laughter_timecodes[data["test_episode_filenames"][i]].append(data["test_nonsilent_timecodes"][i])
        for filename, timecodes in laughter_timecodes.items():
            with open(osp.join(data["path_laughter_dir"][list(data["filename_to_laughter_dir"].keys()).index(filename)], f"{filename[:-4]}.pk"), "wb") as f:
                pickle.dump(merge_segments(timecodes), f)
        return

    centroids = None
    if args.method == "isolation":
        if args.task == "train_isolation_audioset":
            scaler = StandardScaler()
            embeddings_norm = scaler.fit_transform(data["train_embeddings"])
            isolation = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
            isolation.fit(embeddings_norm)
            joblib.dump(isolation, os.path.join(model_path, "isolation.joblib"))
            preds = isolation.predict(embeddings_norm)
            cluster_results = np.array([1 if p == 1 else 0 for p in preds])
        else:
            isolation = IsolationForest(contamination='auto', random_state=42)
            isolation.fit(data["train_embeddings"].cpu().numpy())
            joblib.dump(isolation, os.path.join(model_path, "isolation.joblib"))
            preds = isolation.predict(data["train_embeddings"].cpu().numpy())
            cluster_results = np.array([0 if p == 1 else 1 for p in preds])
    elif args.method == "funnynet":
        clusterer = KMeans(n_clusters=4, random_state=42)
        cluster_labels = clusterer.fit_predict(data["train_embeddings"].astype(np.float32))
        centroids = np.array([data["train_embeddings"].cpu().numpy()[cluster_labels == i].mean(axis=0) for i in range(4)])
        cluster_sizes = dict(zip(*np.unique(cluster_labels, return_counts=True)))
        smallest = min(cluster_sizes, key=cluster_sizes.get)
        cluster_results = np.array([1 if lbl == smallest else 0 for lbl in cluster_labels])
        joblib.dump(clusterer, os.path.join(model_path, "kmeans.joblib"))
        
    if args.task in ["train_isolation_audioset", "train_isolation"]:
        cluster_id_remap = {old: new for new, (old, _) in enumerate(sorted(Counter(cluster_results).items(), key=lambda x: x[1], reverse=True))}
        remapped_results = np.array([cluster_id_remap[c] for c in cluster_results])
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(data["train_embeddings"])
        joblib.dump(pca, osp.join(model_path, "pca_model.joblib"))
        umap_model = UMAP(n_neighbors=15, n_components=5, random_state=42) if args.task == "train_isolation_audioset" else UMAP(n_components=2, random_state=42)
        embeddings_2d_umap = umap_model.fit_transform(data["train_embeddings"])
        joblib.dump(umap_model, osp.join(model_path, "umap_model.joblib"))
        cluster_infos = []
        for cluster_id in np.unique(remapped_results):
            indices = np.where(remapped_results == cluster_id)[0]
            cluster_infos.append({"cluster_id": int(cluster_id), "indices": indices.tolist(), "points_pca": embeddings_2d_pca[indices].tolist(), "centroid_pca": pca.transform(centroids)[cluster_id].tolist() if centroids is not None else None, "points_umap": embeddings_2d_umap[indices].tolist()})
        with open(osp.join(model_path, f"clusters_{args.method}.json"), "w") as f:
            json.dump(cluster_infos, f, indent=2)
        plot_projection(embeddings_2d_pca, "pca", remapped_results, args.method, model_path, centroids, pca)
        plot_projection(embeddings_2d_umap, "umap", remapped_results, args.method, model_path, None, None)

    test_np = data["test_embeddings"]
    if args.method == "isolation":
        isolation_model = joblib.load(os.path.join(model_path, "isolation.joblib"))
        if args.task == "train_isolation_audioset":
            test_np = StandardScaler().fit_transform(test_np)
        preds = isolation_model.predict(test_np)
        cluster_results = np.array([0 if p == 1 else 1 for p in preds])
    elif args.method == "funnynet":
        clusterer = joblib.load(os.path.join(model_path, "kmeans.joblib"))
        cluster_labels = clusterer.predict(test_np.astype(np.float32))
        target_cluster = max(Counter(cluster_labels), key=Counter(cluster_labels).get) if args.task == "train_isolation_all" else min(Counter(cluster_labels), key=Counter(cluster_labels).get)
        cluster_results = np.array([1 if lbl == target_cluster else 0 for lbl in cluster_labels])

    cluster_id_remap = {old: new for new, (old, _) in enumerate(sorted(Counter(cluster_results).items(), key=lambda x: x[1], reverse=True))}
    remapped_results = np.array([cluster_id_remap[c] for c in cluster_results])
    music_cluster_set = {1}
    
    if args.task in ["train_isolation_audioset", "train_isolation"]:
        plot_projection_test(pca.transform(test_np), "pca", remapped_results, args.method, args.embedding_name, model_path, centroids, pca, music_cluster_set)
        plot_projection_test(umap_model.transform(test_np), "umap", remapped_results, args.method, args.embedding_name, model_path, centroids, umap_model, music_cluster_set)

    music_indices = np.where(np.isin(remapped_results, list(music_cluster_set)))[0]
    laughter_timecodes = defaultdict(list)
    for i in range(len(data["test_nonsilent_timecodes"])):
        if i not in music_indices:
            laughter_timecodes[data["test_episode_filenames"][i]].append(data["test_nonsilent_timecodes"][i])
    for filename, timecodes in laughter_timecodes.items():
        laughter_timecodes[filename] = merge_segments(timecodes)

    for i, (current_filename, current_timecodes) in enumerate(laughter_timecodes.items()):
        laughter_dir = data["filename_to_laughter_dir"].get(current_filename)
        if not laughter_dir: continue
        os.makedirs(laughter_dir, exist_ok=True)
        with open(osp.join(laughter_dir, f"{current_filename[:-4]}.pk"), "wb") as f:
            pickle.dump(current_timecodes, f)

    if args.task == "train_isolation_all":
        for filename in set(data["filename_to_laughter_dir"].keys()) - set(laughter_timecodes.keys()):
            laughter_dir = data["filename_to_laughter_dir"].get(filename)
            if not laughter_dir: continue
            os.makedirs(laughter_dir, exist_ok=True)
            with open(osp.join(laughter_dir, f"{filename[:-4]}.pk"), "wb") as f:
                pickle.dump([], f)

if __name__ == "__main__":
    main()