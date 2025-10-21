# extract_embeddings.py
# Purpose: Extract Perch embeddings from audio (train/test) and save as .npy vectors.
# Usage examples:
#   python extract_embeddings.py
#   python extract_embeddings.py --train-dir "C:\path\to\train" --test-dir "C:\path\to\test" --output-base "C:\out\embeddings"
#
# Notes:
# - Defaults preserve the author's original Windows paths so it works out-of-the-box.
# - You can override input/output via CLI flags (see argparse below).
# - Embedding shape is expected to be (1280,) for Perch.

# Imports 

import os
import argparse
import librosa as lb
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

# paths

DEFAULT_TRAIN = r".../Segmented_data/experimento_#/train"
DEFAULT_TEST  = r".../Segmented_data/experimento_#/test"
DEFAULT_OUTPUT= r".../Embeddings/experiment_#"
DEFAULT_CKPT_T= r".../experimento_#/Embeddings/checkpoint_train.txt"
DEFAULT_CKPT_E= r".../experimento_#/Embeddings/checkpoint_test.txt"

def parse_args():
    ap = argparse.ArgumentParser(description="Extract Perch embeddings for train/test splits")
    ap.add_argument("--train-dir", default=DEFAULT_TRAIN, help="Path to train audio directory")
    ap.add_argument("--test-dir",  default=DEFAULT_TEST,  help="Path to test audio directory")
    ap.add_argument("--output-base", default=DEFAULT_OUTPUT, help="Base output folder for embeddings/{train,test}")
    ap.add_argument("--checkpoint-train", default=DEFAULT_CKPT_T, help="Path to train checkpoint file")
    ap.add_argument("--checkpoint-test",  default=DEFAULT_CKPT_E, help="Path to test checkpoint file")
    ap.add_argument("--batch-size", type=int, default=100, help="Files per batch")
    ap.add_argument("--sample-rate", type=int, default=32000, help="Target sampling rate")
    return ap.parse_args()

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            processed = set(line.strip() for line in f if line.strip())
    else:
        processed = set()
    return processed

def main():
    args = parse_args()

    input_dirs = {"train": args.train_dir, "test": args.test_dir}
    checkpoints = {"train": args.checkpoint_train, "test": args.checkpoint_test}
    output_base = args.output_base
    batch_size  = args.batch_size
    target_sr   = args.sample_rate

    os.makedirs(output_base, exist_ok=True)

    print("==== Loading Perch model from TensorFlow Hub... ====")
    model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8')
    print("==== [INFO] Model loaded ====")

    for split in ["train", "test"]:
        input_dir = input_dirs[split]
        output_dir = os.path.join(output_base, split)
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = checkpoints[split]
        processed_files = load_checkpoint(checkpoint_path)

        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        remaining_files = [f for f in all_files if f not in processed_files]

        print(f"==== [INFO] Processing {len(remaining_files)} files in {split}... ====")

        for batch_start in range(0, len(remaining_files), batch_size):
            batch_files = remaining_files[batch_start:batch_start + batch_size]

            with open(checkpoint_path, "a", encoding="utf-8") as ckpt_file:
                for file_name in batch_files:
                    input_path = os.path.join(input_dir, file_name)
                    base_stem = os.path.splitext(file_name)[0]
                    embedding_name = f"{base_stem}_embedding.npy"
                    output_path = os.path.join(output_dir, embedding_name)

                    try:
                        # Read audio
                        signal, sr = sf.read(input_path, dtype='float32')

                        # Resample if needed
                        if sr != target_sr:
                            signal = lb.resample(signal, orig_sr=sr, target_sr=target_sr)

                        # Prepare tensor
                        y_tensor = np.expand_dims(signal, axis=0)

                        # Inference
                        outputs = model.infer_tf(y_tensor)
                        embeddings = outputs["embedding"]
                        embedding_vector = embeddings.numpy()[0].astype(np.float32)  # expected shape (1280,)

                        assert embedding_vector.shape == (1280,), f"Unexpected shape: {embedding_vector.shape}"

                        # Save
                        np.save(output_path, embedding_vector)

                        # Checkpoint
                        ckpt_file.write(file_name + "\n")

                        print(f"==== [INFO] {file_name} processed, embedding shape: {embedding_vector.shape}, saved ====")

                    except Exception as e:
                        print(f"==== Error in {file_name}: {e} ====")
                        continue

    print("==== [INFO] Extraction completed. Embeddings saved and checkpoints updated. ====")

if __name__ == "__main__":
    main()
