import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv(r'D:\movies.csv')
ratings = pd.read_csv(r'D:\ratings.csv')

movies.drop(columns=['genres'], inplace=True)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["title"])

def search(title):
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

def find_recs(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    return similar_user_recs, all_user_recs

def movie_scores(movie_id):
    similar_user_recs, all_user_recs = find_recs(movie_id)
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title"]]

def get_movie_recommendations():
    movie_title = input("Enter a movie title to search for similar movies: ")
    if movie_title:
        results = search(movie_title)
        if not results.empty:
            print(f"Top 5 similar movies to '{movie_title}':")
            print(results[["title"]])
            movie_id = results.iloc[0]["movieId"]
            print("\nRecommended movies based on your movie:")
            recs = movie_scores(movie_id)
            if not recs.empty:
                print(recs[["score", "title"]])
            else:
                print("No recommendations found.")
        else:
            print("No similar movies found.")
    else:
        print("Please enter a movie title.")

if __name__ == "__main__":
    get_movie_recommendations()