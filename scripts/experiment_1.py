# Imports

import os
import shutil
import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Configuration
source_dir = r"...Raw_data_path/birdclef-2025/train_audio"
experiment_dir = r"...\Segmented_data\birdclef-2025/experimento_1"
train_dir = os.path.join(experiment_dir, "train")
test_dir = os.path.join(experiment_dir, "test")
log_path = os.path.join(experiment_dir, "log_movements.txt")

segment_duration = 5.0  # Seconds

# Prepare folders
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Open log
with open(log_path, "w", encoding="utf-8") as log_file:

    # Browse subfolders
    for subdir_name in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        # Filter valid files
        files = [
            f for f in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, f)) and Path(f).suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]
        ]

        if len(files) == 0:
            continue

        # Shuffle randomly
        random.shuffle(files)

        # Calculate split
        num_files = len(files)
        if num_files == 1:
            test_files = []
            train_files = files
        else:
            num_test = max(int(num_files * 0.3), 1)
            test_files = files[:num_test]
            train_files = files[num_test:]

        # Process train files
        for f in train_files:
            file_path = os.path.join(subdir_path, f)
            ext = Path(f).suffix.lower()
            base_stem = Path(f).stem

            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
            except Exception as e:
                print(f"==== Error loading {f}: {e} ====")
                continue

            samples_per_segment = int(segment_duration * sr)
            num_segments = int(len(y) / samples_per_segment)

            if num_segments == 0:
                print(f"==== File too short (discarded): {f} ====")
                continue

            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]

                if len(segment) < samples_per_segment:
                    continue

                new_name = f"{subdir_name}_{base_stem}_{i}{ext}"
                dst_path = os.path.join(train_dir, new_name)
                sf.write(dst_path, segment, sr)
                log_file.write(f"{new_name} train\n")

        # Process test files
        for f in test_files:
            file_path = os.path.join(subdir_path, f)
            ext = Path(f).suffix.lower()
            base_stem = Path(f).stem

            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
            except Exception as e:
                print(f"==== Error loading {f}: {e} ====")
                continue

            samples_per_segment = int(segment_duration * sr)
            num_segments = int(len(y) / samples_per_segment)

            if num_segments == 0:
                print(f"==== File too short (discarded): {f} ====")
                continue

            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]

                if len(segment) < samples_per_segment:
                    continue

                new_name = f"{subdir_name}_{base_stem}_{i}{ext}"
                dst_path = os.path.join(test_dir, new_name)
                sf.write(dst_path, segment, sr)
                log_file.write(f"{new_name} → test\n")

print("==== [INFO] Processing completed. Segments saved and log generated ====")
