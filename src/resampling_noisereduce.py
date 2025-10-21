# Imports and libraries 

import os
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path

# Paths configuration 
base_dirs = [
    r".../Segmented_data/birdclef-2025/experimento_#/train",
    r"...Segmented_data/birdclef-2025/experimento_#/test",
]

log_path = r".../log_resample_noise_#.txt"

target_sr = 32000  # Sampling rate 32 kHz

# Processing
with open(log_path, "w", encoding="utf-8") as log_file:
    for base_dir in base_dirs:
        for file_name in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file_name)
            ext = Path(file_name).suffix.lower()

            if ext not in [".wav", ".mp3", ".flac", ".ogg"]:
                continue

            try:
                # Upload audio
                y, sr = librosa.load(file_path, sr=None, mono=True)
                # Resamplear
                if sr != target_sr:
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                else:
                    y_resampled = y

                # Apply noise reduce
                y_denoised = nr.reduce_noise(y=y_resampled, sr=target_sr)

                # Sobrecribir archivo
                sf.write(file_path, y_denoised, target_sr)

                # Log
                log_file.write(f"==== {file_name} → {target_sr} Hz, noise reduce applied\n ====")
                print(f"==== Processed: {file_name} ====")

            except Exception as e:
                print(f"==== Error processing {file_name}: {e} ====")
                log_file.write(f"{file_name} ERROR: {e}\n")

print("===== [INFO] Processing completed and generated log =====")
