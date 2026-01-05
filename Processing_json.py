import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def normalize_embeddings(data):                # Normalize API response
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]

    if isinstance(data, list):
        embeddings = []
        for item in data:
            if "embeddings" in item:
                embeddings.extend(item["embeddings"])
        return embeddings

    return None


def get_embedding_dim():            # Get embedding dimension once
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": ["test"]}
    )
    data = r.json()
    return len(data["embeddings"][0])


EMBED_DIM = get_embedding_dim()


def embed_single_text(text):      # Embed a single text safely
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [text]}
    )
    data = r.json()
    embeddings = normalize_embeddings(data)

    if embeddings:
        return embeddings[0]

    return None


def create_embedding(text_list, batch_size=32):    # Create embeddings with batching
    all_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": batch}
        )
        data = r.json()
        embeddings = normalize_embeddings(data)

        # If batch fails or NaN happens
        if embeddings is None or len(embeddings) != len(batch):
            print(" Batch issue, retrying smaller batches")

            small_size = 8  # safe small batch

            for j in range(0, len(batch), small_size):
                small_batch = batch[j:j + small_size]

                r2 = requests.post(
                    "http://localhost:11434/api/embed",
                    json={"model": "bge-m3", "input": small_batch}
                )
                data2 = r2.json()
                small_embeddings = normalize_embeddings(data2)

                if small_embeddings and len(small_embeddings) == len(small_batch):
                    all_embeddings.extend(small_embeddings)
                else:
                    # last fallback: single text
                    for text in small_batch:
                        vec = embed_single_text(text)
                        all_embeddings.append(vec if vec else [0.0] * EMBED_DIM)

        else:
            all_embeddings.extend(embeddings)

    return all_embeddings


jsons = os.listdir("newjsons")               # Main logic 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")

    texts = [c["text"] for c in content["chunks"]]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content["chunks"]):  # enumerate() provides the indexing & iteration.
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)   


df = pd.DataFrame.from_records(my_dicts)
# Save this DataFrame
joblib.dump(df, "embeddings.joblib")


