# Imports 

import os
import numpy as np
import pandas as pd
import time
import pickle
import logging
import matplotlib.pyplot as plt

# Libraries 

from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from vectordb import HNSWVectorDB, InMemoryExactNNVectorDB

class EmbeddingDoc128(BaseDoc):
    id: str
    vector: NdArray[128]
    metadata: dict

def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger()

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return None

def save_checkpoint(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def build_and_index(fusion_folder, method, workspace, metadata_csv):
    df_meta = pd.read_csv(metadata_csv)
    docs = []

    file_list = [f for f in os.listdir(fusion_folder) if f.endswith(".npy")]
    file_list.sort()

    for fname in file_list:
        fpath = os.path.join(fusion_folder, fname)
        emb = np.load(fpath).astype(np.float32)

        primary_key = fname.split("_")[0]
        # We clean the "_mfcc" suffix
        id_clean = fname.replace("_mfcc.npy", "").replace(".npy", "")

        row = df_meta[df_meta["primary_label"].astype(str) == primary_key]
        metadata = row.iloc[0].to_dict() if not row.empty else {}

        doc = EmbeddingDoc128(
            id=id_clean,
            vector=emb,
            metadata=metadata
        )
        docs.append(doc)

    os.makedirs(workspace, exist_ok=True)

    if method == 'hnsw':
        db = HNSWVectorDB[EmbeddingDoc128](workspace=workspace)
    else:
        db = InMemoryExactNNVectorDB[EmbeddingDoc128](workspace=workspace)

    if len(docs) == 0:
        print(f"===== No valid embeddings were found in: {fusion_folder}, This experiment is omitted ====")
        return None, []

    db.index(inputs=DocList[EmbeddingDoc128](docs))
    return db, docs

def evaluate_experiment(db, doc_list, query_folder, batch_size=1000, checkpoint=None, log_txt_path=None):
    query_files = [f for f in os.listdir(query_folder) if f.endswith(".npy")]
    query_files.sort()

    hits_1 = 0
    hits_5 = 0
    times = []

    logs = []

    for qf in query_files:
        fpath = os.path.join(query_folder, qf)
        query_emb = np.load(fpath).astype(np.float32)
        id_clean = qf.replace("_mfcc.npy", "").replace(".npy", "")
        query_doc = EmbeddingDoc128(id=id_clean, vector=query_emb, metadata={})

        start = time.time()
        results = db.search(DocList([query_doc]), search_field="vector", limit=5)
        end = time.time()

        top_match = results[0].matches[0]
        top_species = top_match.metadata.get("primary_label", "")
        query_species = id_clean.split("_")[0]

        if top_species == query_species:
            hits_1 += 1

        top_k_species = [m.metadata.get("primary_label", "") for m in results[0].matches]
        if query_species in top_k_species:
            hits_5 += 1

        times.append(end - start)

        # Detailed metadata de top-5
        top5_info = ""
        for idx, m in enumerate(results[0].matches):
            top5_info += f"Top-{idx+1}: {m.id} (Species: {m.metadata.get('primary_label', '')})\n"
            top5_info += f"Metadata: {m.metadata}\n\n"

        log_text = f"Query: {id_clean}\n"
        log_text += f"Query metadata: {query_doc.metadata}\n\n"
        log_text += f"Top-1: {top_match.id} (Species: {top_species})\n"
        log_text += f"Top-1 metadata: {top_match.metadata}\n\n"
        log_text += f"Hit@1: {top_species == query_species}\n"
        log_text += f"Top-5 species: {top_k_species}\n\n"
        log_text += f"Detailed Top-5:\n{top5_info}"
        log_text += f"Time: {end - start:.4f}s\n"
        log_text += "-" * 60 + "\n"
        logs.append(log_text)

    # Save to file
    if log_txt_path:
        with open(log_txt_path, "w") as f:
            for entry in logs:
                f.write(entry)

    h1_score = hits_1 / len(query_files) if len(query_files) > 0 else 0
    h5_score = hits_5 / len(query_files) if len(query_files) > 0 else 0
    avg_time = np.mean(times) if times else 0
    std_time = np.std(times) if times else 0

    metrics = {
        "H@1": h1_score,
        "H@5": h5_score,
        "AvgTime": avg_time,
        "StdTime": std_time,
    }

    checkpoint_data = {
        "completed_queries": len(query_files),
        "metrics": metrics,
    }

    return metrics, checkpoint_data

def save_results_excel(results, excel_path):
    df_all = pd.DataFrame(results)
    df_all.to_excel(excel_path, index=False)
    print(f"==== [INFO] Results saved at {excel_path} ====")


