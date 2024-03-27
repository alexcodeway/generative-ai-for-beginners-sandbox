import ollama
import pandas as pd
import numpy as np

model = 'nomic-embed-text'

SIMILARITIES_RESULTS_THRESHOLD = 0.5 # had to lower threshold for nomic; not as accurate?
DATASET_NAME = '08-building-search-applications\\embedding_index_nomic_3m.json'

def load_dataset(source: str) -> pd.core.frame.DataFrame:
    # Load the video session index
    pd_vectors = pd.read_json(source)
    return pd_vectors.drop(columns=["text"], errors="ignore").fillna("")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_videos(
    query: str, dataset: pd.core.frame.DataFrame, rows: int
) -> pd.core.frame.DataFrame:
    # create a copy of the dataset
    video_vectors = dataset.copy()

    # get the embeddings for the query    
    query_embeddings = ollama.embeddings(model=model, prompt=query)['embedding']

    # create a new column with the calculated similarity for each row
    video_vectors["similarity"] = video_vectors["nomic"].apply(
        lambda x: cosine_similarity(np.array(query_embeddings), np.array(x))
    )

    # filter the videos by similarity
    mask = video_vectors["similarity"] >= SIMILARITIES_RESULTS_THRESHOLD
    video_vectors = video_vectors[mask].copy()

    # sort the videos by similarity
    video_vectors = video_vectors.sort_values(by="similarity", ascending=False).head(
        rows
    )

    # return the top rows
    return video_vectors.head(rows)

def display_results(videos: pd.core.frame.DataFrame, query: str):
    def _gen_yt_url(video_id: str, seconds: int) -> str:
        """convert time in format 00:00:00 to seconds"""
        return f"https://youtu.be/{video_id}?t={seconds}"

    print(f"\nVideos similar to '{query}':")
    for _, row in videos.iterrows():
        youtube_url = _gen_yt_url(row["videoId"], row["seconds"])
        print(f" - {row['title']}")
        print(f"   Summary: {' '.join(row['summary'].split()[:15])}...")
        print(f"   YouTube: {youtube_url}")
        print(f"   Similarity: {row['similarity']}")
        print(f"   Speakers: {row['speaker']}")

pd_vectors = load_dataset(DATASET_NAME)

# get user query from imput
while True:
    query = input("Enter a query: ")
    if query == "exit":
        break
    videos = get_videos(query, pd_vectors, 5)
    display_results(videos, query)