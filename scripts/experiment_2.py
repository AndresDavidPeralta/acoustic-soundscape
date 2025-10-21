# Imports 

import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Configuration

source_dir = r".../Raw_data/birdclef-2025/train_audio"
experiment_dir = r"../segmented_data/birdclef-2025/experimento_2"
train_dir = os.path.join(experiment_dir, "train")
test_dir = os.path.join(experiment_dir, "test")
log_path = os.path.join(experiment_dir, "log_movements.txt")

segment_duration = 5.0  # seconds

# Prepare folders
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Open log
with open(log_path, "w", encoding="utf-8") as log_file:

    
    # Process each subfolder
    for subdir_name in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)

            # Filter by valid extensions
            
            ext = Path(file_name).suffix.lower()
            if ext not in [".wav", ".mp3", ".flac", ".ogg"]:
                continue

            # Upload audio
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
            except Exception as e:
                print(f" ===== Error loading ===== {file_name}: {e}")
                continue

            samples_per_segment = int(segment_duration * sr)
            num_segments = int(len(y) / samples_per_segment)

            if num_segments == 0:
                print(f"====== File too short (discarded) =====: {file_name}")
                continue

            # Calculate RMS by segment
            rms_energies = []
            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]
                rms = np.sqrt(np.mean(segment ** 2))
                rms_energies.append(rms)

            max_energy_idx = int(np.argmax(rms_energies))

            # Save segment test
            start_test = max_energy_idx * samples_per_segment
            end_test = start_test + samples_per_segment
            segment_test = y[start_test:end_test]

            base_stem = Path(file_name).stem
            test_name = f"{subdir_name}_{base_stem}_{max_energy_idx}{ext}"
            test_save_path = os.path.join(test_dir, test_name)

            sf.write(test_save_path, segment_test, sr)
            log_file.write(f"{test_name}  test\n")

            # Save segments train
            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]

                if len(segment) < samples_per_segment:
                    continue  # discard incomplete segments

                train_name = f"{subdir_name}_{base_stem}_{i}{ext}"
                train_save_path = os.path.join(train_dir, train_name)
                sf.write(train_save_path, segment, sr)
                log_file.write(f"{train_name}  train\n")

print("===== [INFO] Processing completed. Segments saved and log generated =====")
