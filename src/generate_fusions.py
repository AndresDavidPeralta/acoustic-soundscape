# Imports 

import os
import shutil
import numpy as np
import pandas as pd
import yaml


def load_energy_csv(csv_path):
    df = pd.read_csv(csv_path)
    df_dict = dict(zip(df['id_segment'], df['energia']))
    return df_dict

def get_segments(folder, recording_id):
    segments = []
    for fname in os.listdir(folder):
        if fname.startswith(recording_id + "_") and fname.endswith(".npy"):
            segments.append(os.path.join(folder, fname))
    return segments

def fuse_embeddings(embeddings, method='average', energies=None, segment_ids=None):
    arr = np.stack(embeddings)
    if method == 'average':
        return np.mean(arr, axis=0)
    elif method == 'max':
        return np.max(arr, axis=0)
    elif method == 'sum':
        return np.sum(arr, axis=0)
    elif method == 'weighted':
        if energies is None or segment_ids is None:
            raise ValueError("Energies and segment IDs required for weighted average.")
        # Remove "_embedding" suffix before searching the dict
        segment_ids_clean = [sid.replace("_embedding", "").replace("_mfcc", "") for sid in segment_ids]
        valid_pairs = [(emb, sid) for emb, sid in zip(embeddings, segment_ids_clean) if sid in energies]
        if not valid_pairs:
            return None
        embeddings_filtered, segment_ids_filtered = zip(*valid_pairs)
        arr_filtered = np.stack(embeddings_filtered)
        weights = np.array([energies[sid] for sid in segment_ids_filtered])
        weights = weights / np.sum(weights)
        return np.average(arr_filtered, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

def process_folder(input_folder, output_folder, energy_dict, method):
    os.makedirs(output_folder, exist_ok=True)
    logs = []

    if not os.path.exists(input_folder):
        logs.append(f"Folder not found: {input_folder}")
        return logs

    recording_ids = set('_'.join(fname.split('_')[:2]) for fname in os.listdir(input_folder) if fname.endswith(".npy"))

    for rid in recording_ids:
        segment_paths = get_segments(input_folder, rid)
        embeddings = []
        segment_ids = []

        for seg_path in segment_paths:
            emb = np.load(seg_path)
            embeddings.append(emb)
            seg_id = os.path.basename(seg_path).replace(".npy", "")
            segment_ids.append(seg_id)

        if not embeddings:
            continue

        fused = fuse_embeddings(embeddings, method=method, energies=energy_dict, segment_ids=segment_ids)

        if fused is None:
            logs.append(f"{rid}: ignored (no valid segments for weighted)")
            continue

        save_path = os.path.join(output_folder, f"{rid}.npy")
        np.save(save_path, fused)
        logs.append(f"{rid}: {len(segment_paths)} segments merged using {method}")

    return logs

def copy_original_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.endswith(".npy"):
            src_path = os.path.join(input_folder, fname)
            dst_path = os.path.join(output_folder, fname)
            shutil.copy2(src_path, dst_path)

def generate_all_fusions(config):
    energy_dict = load_energy_csv(config["energy_csv_path"])
    base_folder = config["experiment_folder_base"]
    output_base = config["fusion_output_base"]
    methods = config["fusion_methods"]
    fusion_targets = config["fusion_targets"]

    logs_all = []

    for exp_name, targets in fusion_targets.items():
        exp_folder = os.path.join(base_folder, exp_name)
        train_folder = os.path.join(exp_folder, "train")
        test_folder = os.path.join(exp_folder, "test")

        for method in methods:
            # ===== Train =====
            if "train" in targets:
                out_train = os.path.join(output_base, exp_name, f"train_{method}")
                logs = process_folder(train_folder, out_train, energy_dict, method)
                logs_all.extend([f"[{exp_name}-TRAIN-{method}] " + l for l in logs])
            else:
                out_train = os.path.join(output_base, exp_name, f"train_{method}")
                copy_original_folder(train_folder, out_train)
                logs_all.append(f"[{exp_name}-TRAIN-{method}] Copying without merging")

            # ===== Test =====
            if "test" in targets:
                out_test = os.path.join(output_base, exp_name, f"test_{method}")
                logs = process_folder(test_folder, out_test, energy_dict, method)
                logs_all.extend([f"[{exp_name}-TEST-{method}] " + l for l in logs])
            else:
                out_test = os.path.join(output_base, exp_name, f"test_{method}")
                copy_original_folder(test_folder, out_test)
                logs_all.append(f"[{exp_name}-TEST-{method}] Copying without merging")

    log_path = os.path.join(output_base, "log_fusions.txt")
    os.makedirs(output_base, exist_ok=True)
    with open(log_path, "w") as f:
        for line in logs_all:
            f.write(line + "\n")

    print(f"==== [INFO] Merger completed. Logs saved in: {log_path} ====")

if __name__ == "__main__":
    with open(".../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_all_fusions(config)
