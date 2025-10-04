import joblib, pandas as pd
import numpy as np
import gradio as gr

# Load the saved model
model_data = joblib.load('recommender_model.pkl')
user_factors = model_data['user_factors']
item_factors = model_data['item_factors']
user_item_matrix = model_data['user_item_matrix']
movies = model_data['movies']
ratings = model_data['ratings']

def recommend_movies(user_id, N=10):
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_vector = user_factors[user_idx].reshape(1, -1)
        predicted_ratings = np.dot(user_vector, item_factors).flatten()
        
        rated_movies = set(ratings[ratings.userId == user_id]['movieId'].tolist())
        all_movie_ids = user_item_matrix.columns.tolist()
        unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
        
        if not unrated_movies:
            return []
        
        movie_id_to_idx = {mid: idx for idx, mid in enumerate(all_movie_ids)}
        predictions = [(movie_id, predicted_ratings[movie_id_to_idx[movie_id]]) for movie_id in unrated_movies]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        movie_titles = movies.set_index('movieId')['title'].to_dict()
        return [(mid, movie_titles.get(mid, f"Movie {mid}")) for mid, _ in predictions[:N]]
    except:
        return []

def ui_recommend(uid, n):
    recs = recommend_movies(int(uid), int(n))
    return "\n".join([f"{mid}: {title}" for mid,title in recs])

gr.Interface(ui_recommend, ["text","slider"], "text",
             title="Movie Recommender").launch()
