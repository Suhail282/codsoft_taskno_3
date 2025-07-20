import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data (MovieLens dataset or your own CSV)
df = pd.read_csv('movies.csv')  # Must have 'title' and 'description' columns

# TF-IDF Vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
df['description'] = df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create reverse index for title to index lookup
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, top_n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example usage
if __name__ == "__main__":
    movie = input("Enter a movie title: ")
    print("\nRecommended movies:")
    try:
        recommendations = recommend(movie)
        for rec in recommendations:
            print("•", rec)
    except KeyError:
        print("Movie not found.")
