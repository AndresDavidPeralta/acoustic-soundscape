# Imports 

import os
import librosa
import pandas as pd

# Paths 

train_dir = r"...Audio_path/trian"
test_dir = r"...Audio_path/test"
output_csv = r"/home/andresperalta/proyectos/recuperacion_sonora/Paper_Webmedia/Results/segment_energies.csv"

# Browse folders

all_files = []
for folder in [train_dir, test_dir]:
    for fname in os.listdir(folder):
        if fname.endswith(".ogg") or fname.endswith(".wav") or fname.endswith(".mp3"):
            fpath = os.path.join(folder, fname)
            all_files.append((fname.split('.')[0], fpath))  # base name without extension

# Calculate the energy

data = []
for seg_id, audio_path in all_files:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = (sum(y**2) / len(y)) ** 0.5
        data.append({"id_segment": seg_id, "energia": rms})
    except Exception as e:
        print(f"==== [ERROR] Could not be processed {audio_path}: {e} ====")

# Save CSV

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"==== [INFO] CSV saved en: {output_csv} =====")
