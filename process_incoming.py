
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import joblib

app = Flask(__name__)

# -------------------------------
# EMBEDDING UTILITIES
# -------------------------------

def normalize_embeddings(data):
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]

    if isinstance(data, list):
        embeddings = []
        for item in data:
            if "embeddings" in item:
                embeddings.extend(item["embeddings"])
        return embeddings

    return None


def get_embedding_dim():
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": ["test"]}
    )
    data = r.json()
    return len(data["embeddings"][0])


EMBED_DIM = get_embedding_dim()


def embed_single_text(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [text]}
    )
    data = r.json()
    embeddings = normalize_embeddings(data)

    if embeddings:
        return embeddings[0]

    return None


def create_embedding(text_list, batch_size=32):
    all_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": batch}
        )
        data = r.json()
        embeddings = normalize_embeddings(data)

        if embeddings is None or len(embeddings) != len(batch):
            small_size = 8

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
                    for text in small_batch:
                        vec = embed_single_text(text)
                        all_embeddings.append(vec if vec else [0.0] * EMBED_DIM)
        else:
            all_embeddings.extend(embeddings)

    return all_embeddings


def inference(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    response = r.json()
    return response["response"]


# -------------------------------
# LOAD PRECOMPUTED EMBEDDINGS
# -------------------------------

df = joblib.load("embeddings.joblib")


# -------------------------------
# RAG PIPELINE FUNCTION
# -------------------------------

def rag_pipeline(incoming_query):

    question_embedding = create_embedding([incoming_query])[0]

    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

    top_result = 5
    max_indx = similarities.argsort()[::-1][0:top_result]

    new_df = df.loc[max_indx]

    prompt = f"""
    You are an AI teaching assistant for a Web Development course.

    Course Content:
    {new_df[["title","number","start","end","text"]].to_json(orient="records")}

    User Question:
    "{incoming_query}"

    Instructions:
    1. Answer like a professional instructor.
    2. Do not mention subtitles.
    3. Clearly explain video number and timestamps.
    4. Use point-wise format.
    5. If unrelated, politely decline.
    6. Explain the query in 1-2 lines for better understanding.
    """

    response = inference(prompt)

    return response


# -------------------------------
# FLASK ROUTES
# -------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"response": "Please enter a valid question."})

    try:
        response = rag_pipeline(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


# -------------------------------
# RUN APP
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)