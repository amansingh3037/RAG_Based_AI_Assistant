import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
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


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2", 
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question:- ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# find similarities of questions embeddings with other embeddings!!
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
# print(similarities)
top_result = 5
max_indx = similarities.argsort()[::-1][0:top_result]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title","number","text"]])


prompt = f''' This is a web development course. Here are the video subtitles chunks containing video title, video number, start time in seconds, end time in seconds,and the text at that time :-

{new_df[["title","number","start","end","text"]].to_json(orient="records")}


"{incoming_query}"
The user asked this query related to the video,Now you have to answer just like a human do (don't mention the above format, its just for you) where and how much content is taught in which video (in which video and at which timestamp) and also guide the user to go to that particular video.If user asks unrelated queries, then tell the user to ask only the course related queries '''


with open("prompt.txt","w") as f:
    f.write(prompt)


response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index,"-->","Video No:",item["number"] ," Title:-",item["title"] ," Info:-",item["text"]," Starting:",item["start"]," Ending:",item["end"])

