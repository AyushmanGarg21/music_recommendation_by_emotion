import numpy as np
import pandas as pd


def calculate_cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1, dtype=float)
        vec2 = np.array(vec2, dtype=float)
    except ValueError:
        # Handle the case where conversion to float fails (e.g., string data)
        print("Error: Input vectors cannot be converted to numeric arrays.")
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim



def setSongs():
    per = np.load("emotion_all.npy")
    songs = pd.read_csv('songs.csv')
    avg = np.average(per, axis=0)
    avg = np.float64(avg)
    songs['values'] = np.array([songs['angry'],songs['Fear'],songs['happy'],songs['neutral'],songs['sad'],songs['surprise']]).T.tolist()
    songs['score'] = songs['values'].apply(lambda x: calculate_cosine_similarity(x, avg))

    songs = songs.sort_values("score", ascending=False)
    top_10_titles = songs["title"].head(10).to_list()
    np.save("song_rec.npy", top_10_titles)