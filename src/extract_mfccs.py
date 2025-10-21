# Imports 

import os
import librosa as lb
import numpy as np
import soundfile as sf
from scipy.stats import skew, kurtosis

input_dirs = {
    "train": r".../Segmented_data/birdclef-2025/experimento_#/train",
    "test": r".../Segmented_data/birdclef-2025/experimento_#/test",
}

output_base = r".../Exit route/experimento_#"
checkpoints = {
    "train": r".../Exit route/experimento_#/checkpoint_train.txt",
    "test": r".../Exit route/experimento_#/checkpoint_test.txt",
}

segment_batch_size = 100

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            processed = set(line.strip() for line in f.readlines())
    else:
        processed = set()
    return processed

for split in ["train", "test"]:
    input_dir = input_dirs[split]
    output_dir = os.path.join(output_base, split)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = checkpoints[split]
    processed_files = load_checkpoint(checkpoint_path)

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    remaining_files = [f for f in all_files if f not in processed_files]

    print(f"==== Processing {len(remaining_files)} files in {split}... =====")

    for batch_start in range(0, len(remaining_files), segment_batch_size):
        batch_files = remaining_files[batch_start:batch_start + segment_batch_size]

        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_file:
            for file_name in batch_files:
                input_path = os.path.join(input_dir, file_name)
                base_stem = os.path.splitext(file_name)[0]
                mfcc_name = f"{base_stem}_mfcc.npy"
                output_path = os.path.join(output_dir, mfcc_name)

                try:
                    # Read audio
                    signal, sr = sf.read(input_path, dtype='float32')
                    
                    # If you are already at 32 kHz and filtered, do not resample
                    if sr != 32000:
                        signal = lb.resample(signal, orig_sr=sr, target_sr=32000)

                    # Extract MFCC
                    mfcc = lb.feature.mfcc(y=signal, sr=32000, n_mfcc=40, n_fft=1024, hop_length=512)
                    print(f"==== {file_name}  MFCC shape before pooling: {mfcc.shape} ====")

                    # Calculate statistics
                    mean_vec = np.mean(mfcc, axis=1)
                    std_vec = np.std(mfcc, axis=1)
                    skew_vec = skew(mfcc, axis=1)
                    kurt_vec = kurtosis(mfcc, axis=1)

                    # Select only the first 8 kurtosis dimensions
                    kurt_vec_short = kurt_vec[:8]

                    # Concatenate
                    final_vector = np.concatenate([mean_vec, std_vec, skew_vec, kurt_vec_short]).astype(np.float32)
                    assert final_vector.shape == (128,), f"Unexpected Shape: {final_vector.shape}"

                    print(f"==== {file_name} Final vector shape: {final_vector.shape} ====")

                    # Save
                    np.save(output_path, final_vector)

                    # Checkpoint
                    ckpt_file.write(file_name + "\n")

                    print(f"==== {file_name} processed, MFCC saved: {mfcc_name} ====")

                except Exception as e:
                    print(f"==== Error in {file_name}: {e} ====")
                    continue

print("==== [INFO] Extraction completed. MFCC vectors saved and checkpoints updated =====")
